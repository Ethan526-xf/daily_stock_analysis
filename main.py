# -*- coding: utf-8 -*-
import os
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
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

# 标准日志格式
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'

def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    level = logging.DEBUG if debug else logging.INFO
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
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
        
        # 兼容 Tavily 搜索 key
        t_key = os.environ.get("TAVILY_API_KEYS") or os.environ.get("TAVILY_API_KEY")
        self.search_service = SearchService(
            bocha_keys=self.config.bocha_api_keys,
            tavily_keys=[t_key] if t_key else [],
        )

    def analyze_stock(self, code: str) -> Optional[AnalysisResult]:
        try:
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            realtime_quote = self.akshare_fetcher.get_realtime_quote(code)
            current_pct = 0.0
            if realtime_quote:
                stock_name = realtime_quote.name or stock_name
                current_pct = getattr(realtime_quote, 'pct_chg', getattr(realtime_quote, '涨跌幅', 0.0))
            
            context = self.db.get_analysis_context(code)
            if not context: return None

            # 保底逻辑：如果实时行情挂了，从历史记录抓取最后一天的涨跌幅
            if current_pct == 0.0 and context.get('raw_data'):
                last_row = context['raw_data'][-1]
                current_pct = last_row.get('涨跌幅', last_row.get('pct_chg', 0.0))
            
            news_context = None
            if self.search_service.is_available:
                intel = self.search_service.search_comprehensive_intel(stock_code=code, stock_name=stock_name)
                news_context = self.search_service.format_intel_report(intel, stock_name)
            
            result = self.analyzer.analyze(context, news_context=news_context)
            if result:
                # --- 数据硬注入：将涨跌幅和名字传给结果对象 ---
                result.name = stock_name
                result.code = code
                result.realtime_change = current_pct
            return result
        except Exception as e:
            logger.error(f"[{code}] 分析出错: {e}")
            return None

    def run(self, stock_codes=None):
        if not stock_codes: stock_codes = self.config.stock_list
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.analyze_stock, code): code for code in stock_codes}
            for f in as_completed(futures):
                res = f.result()
                if res: results.append(res)
        return results

def run_full_analysis(config, args, stock_codes):
    pipeline = StockAnalysisPipeline(config=config, max_workers=args.workers)
    results = pipeline.run(stock_codes=stock_codes)
    
    notion_token = os.environ.get("NOTION_TOKEN")
    database_id = "bf217c149f1e4ab2918f58fc2a813213" 

    if notion_token and results:
        logger.info("正在推送高质量分析到 Notion...")
        notion = Client(auth=notion_token)
        for r in results:
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                report_body = pipeline.notifier.generate_single_stock_report(r)
                # 获取注入的涨跌幅数据
                change_val = getattr(r, 'realtime_change', 0.0)

                properties = {
                    "Stock name": {"title": [{"text": {"content": f"{r.name}({r.code})"}}]},
                    "分析日期": {"date": {"start": today}},
                    "涨跌幅%": {"number": float(change_val) / 100}, 
                    # 将正文填入表格的“完整分析”列
                    "完整分析": {"rich_text": [{"type": "text", "text": {"content": str(report_body)[:1900]}}]}
                }
                notion.pages.create(parent={"database_id": database_id}, properties=properties)
                logger.info(f"✅ {r.name} 同步成功")
            except Exception as e:
                # 严格缩进，修复 IndentationError
                logger.error(f"❌ 同步失败: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stocks', type=str)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    config = get_config()
    setup_logging(log_dir=config.log_dir)
    stock_codes = [c.strip() for c in args.stocks.split(',')] if args.stocks else None
    run_full_analysis(config, args, stock_codes)

if __name__ == "__main__":
    main()
