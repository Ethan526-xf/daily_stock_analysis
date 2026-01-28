# -*- coding: utf-8 -*-
import os
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from feishu_doc import FeishuDocManager
from notion_client import Client
from config import get_config, Config
from storage import get_db, DatabaseManager
from data_provider import DataFetcherManager
from data_provider.akshare_fetcher import AkshareFetcher
from analyzer import GeminiAnalyzer, AnalysisResult, STOCK_NAME_MAP
from notification import NotificationService
from search_service import SearchService
from market_analyzer import MarketAnalyzer

# 日志配置
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'

def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    level = logging.DEBUG if debug else logging.INFO
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime('%Y%m%d')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

class StockAnalysisPipeline:
    def __init__(self, config: Optional[Config] = None, max_workers: Optional[int] = None):
        self.config = config or get_config()
        self.max_workers = max_workers or self.config.max_workers
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.akshare_fetcher = AkshareFetcher()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()
        
        # 修复搜索服务：将环境变量字符串包装成列表
        tavily_env = os.environ.get("TAVILY_API_KEYS")
        t_keys = self.config.tavily_api_keys or ([tavily_env] if tavily_env else [])
        self.search_service = SearchService(
            bocha_keys=self.config.bocha_api_keys,
            tavily_keys=t_keys,
        )

    def analyze_stock(self, code: str) -> Optional[AnalysisResult]:
        try:
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            realtime_quote = self.akshare_fetcher.get_realtime_quote(code)
            if realtime_quote and realtime_quote.name: stock_name = realtime_quote.name
            
            news_context = None
            if self.search_service.is_available:
                intel_results = self.search_service.search_comprehensive_intel(stock_code=code, stock_name=stock_name)
                news_context = self.search_service.format_intel_report(intel_results, stock_name)
            
            context = self.db.get_analysis_context(code)
            if not context: return None
            
            # 增强上下文，确保存入涨跌幅
            enhanced_context = context.copy()
            enhanced_context['stock_name'] = stock_name
            current_change = 0.0
            if realtime_quote:
                current_change = getattr(realtime_quote, 'pct_chg', 0.0)
            elif context.get('raw_data'):
                # 如果实时行情失败，从数据库最后一行抓取
                current_change = context['raw_data'][-1].get('涨跌幅', 0.0)
            
            result = self.analyzer.analyze(enhanced_context, news_context=news_context)
            if result:
                # 关键：将市场涨跌幅“强行注入”结果对象，供 Notion 使用
                result.market_pct_change = current_change
            return result
        except Exception as e:
            logger.error(f"分析失败: {e}")
            return None

    def run(self, stock_codes=None, dry_run=False):
        if stock_codes is None: stock_codes = self.config.stock_list
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {executor.submit(self.analyze_stock, code): code for code in stock_codes}
            for future in as_completed(future_to_code):
                res = future.result()
                if res: results.append(res)
        return results

def run_full_analysis(config, args, stock_codes):
    pipeline = StockAnalysisPipeline(config=config, max_workers=args.workers)
    results = pipeline.run(stock_codes=stock_codes, dry_run=args.dry_run)
    
    # --- Notion 同步逻辑修复版 ---
    notion_token = os.environ.get("NOTION_TOKEN")
    database_id = "bf217c149f1e4ab2918f58fc2a813213" 

    if notion_token and results:
        logger.info("正在同步分析结果到 Notion...")
        notion = Client(auth=notion_token)
        for r in results:
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                report_text = pipeline.notifier.generate_single_stock_report(r)
                # 抓取我们注入的真实市场涨跌幅
                change_val = getattr(r, 'market_pct_change', 0.0)

                properties = {
                    "Stock name": {"title": [{"text": {"content": f"{getattr(r, 'name', '未知')}({getattr(r, 'code', '')})"}}]},
                    "分析日期": {"date": {"start": today}},
                    # Notion 格式：1.96% 需要存为 0.0196
                    "涨跌幅%": {"number": float(change_val) / 100}, 
                    "完整分析": {"rich_text": [{"type": "text", "text": {"content": str(report_text)[:1900]}}]}
                }
                notion.pages.create(parent={"database_id": database_id}, properties=properties)
                logger.info(f"Notion 同步成功: {getattr(r, 'name', '未知')}")
            except Exception as e:
                logger.error(f"Notion 同步失败: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stocks', type=str)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    
    config = get_config()
    setup_logging(debug=args.debug, log_dir=config.log_dir)
    stock_codes = [c.strip() for c in args.stocks.split(',')] if args.stocks else None
    run_full_analysis(config, args, stock_codes)

if __name__ == "__main__":
    main()
