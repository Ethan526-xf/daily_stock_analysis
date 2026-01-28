# -*- coding: utf-8 -*-
"""
A股自选股智能分析系统 - 完整修复版
"""
import os
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from feishu_doc import FeishuDocManager
from notion_client import Client  # 新增：Notion 连接器
from config import get_config, Config
from storage import get_db, DatabaseManager
from data_provider import DataFetcherManager
from data_provider.akshare_fetcher import AkshareFetcher, RealtimeQuote, ChipDistribution
from analyzer import GeminiAnalyzer, AnalysisResult, STOCK_NAME_MAP
from notification import NotificationService, NotificationChannel
from search_service import SearchService, SearchResponse
from enums import ReportType
from stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult
from market_analyzer import MarketAnalyzer

# 配置日志格式
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    level = logging.DEBUG if debug else logging.INFO
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f"stock_analysis_{today_str}.log"
    debug_log_file = log_path / f"stock_analysis_debug_{today_str}.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)
    debug_handler = RotatingFileHandler(debug_log_file, maxBytes=50 * 1024 * 1024, backupCount=3, encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(debug_handler)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class StockAnalysisPipeline:
    def __init__(self, config: Optional[Config] = None, max_workers: Optional[int] = None):
        self.config = config or get_config()
        self.max_workers = max_workers or self.config.max_workers
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.akshare_fetcher = AkshareFetcher()
        self.trend_analyzer = StockTrendAnalyzer()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()
        
        # 兼容 Tavily 搜索 key
        t_keys = self.config.tavily_api_keys or os.environ.get("TAVILY_API_KEYS") or os.environ.get("TAVILY_API_KEY")
        self.search_service = SearchService(
            bocha_keys=self.config.bocha_api_keys,
            tavily_keys=[t_keys] if isinstance(t_keys, str) else t_keys,
        )

    def fetch_and_save_stock_data(self, code: str, force_refresh: bool = False) -> Tuple[bool, Optional[str]]:
        try:
            today = date.today()
            if not force_refresh and self.db.has_today_data(code, today):
                return True, None
            df, source_name = self.fetcher_manager.get_daily_data(code, days=30)
            if df is None or df.empty: return False, "数据为空"
            self.db.save_daily_data(df, code, source_name)
            return True, None
        except Exception as e:
            return False, str(e)

    def analyze_stock(self, code: str) -> Optional[AnalysisResult]:
        try:
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            realtime_quote = self.akshare_fetcher.get_realtime_quote(code)
            current_pct_change = 0.0
            if realtime_quote:
                if realtime_quote.name: stock_name = realtime_quote.name
                # 记录实时涨跌幅数值
                current_pct_change = getattr(realtime_quote, 'pct_chg', 0.0)
            
            chip_data = self.akshare_fetcher.get_chip_distribution(code)
            news_context = None
            if self.search_service.is_available:
                intel_results = self.search_service.search_comprehensive_intel(stock_code=code, stock_name=stock_name)
                news_context = self.search_service.format_intel_report(intel_results, stock_name)
            
            context = self.db.get_analysis_context(code)
            if not context: return None
            
            enhanced_context = context.copy()
            enhanced_context['stock_name'] = stock_name
            result = self.analyzer.analyze(enhanced_context, news_context=news_context)
            if result:
                # 将涨跌幅数据注入到结果对象中
                result.realtime_pct_change = current_pct_change
                result.name = stock_name
                result.code = code
            return result
        except Exception as e:
            logger.error(f"[{code}] 分析失败: {e}")
            return None

    def process_single_stock(self, code: str, skip_analysis=False, **kwargs) -> Optional[AnalysisResult]:
        self.fetch_and_save_stock_data(code)
        if skip_analysis: return None
        return self.analyze_stock(code)

    def run(self, stock_codes: Optional[List[str]] = None, dry_run=False, send_notification=True) -> List[AnalysisResult]:
        if stock_codes is None:
            self.config.refresh_stock_list()
            stock_codes = self.config.stock_list
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {executor.submit(self.process_single_stock, code, skip_analysis=dry_run): code for code in stock_codes}
            for future in as_completed(future_to_code):
                try:
                    res = future.result()
                    if res: results.append(res)
                except Exception as e: logger.error(f"任务失败: {e}")
        if results and send_notification and not dry_run:
            report = self.notifier.generate_dashboard_report(results)
            self.notifier.save_report_to_file(report)
            if self.notifier.is_available(): self.notifier.send(report)
        return results

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stocks', type=str)
    parser.add_argument('--no-notify', action='store_true')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--no-market-review', action='store_true')
    return parser.parse_args()

def run_market_review(notifier, analyzer, search):
    try:
        market_analyzer = MarketAnalyzer(search_service=search, analyzer=analyzer)
        report = market_analyzer.run_daily_review()
        if report:
            notifier.save_report_to_file(f"# 大盘复盘\n\n{report}", f"market_review_{datetime.now().strftime('%Y%m%d')}.md")
            if notifier.is_available(): notifier.send(f"🎯 大盘复盘\n\n{report}")
            return report
    except Exception as e: logger.error(f"复盘失败: {e}")
    return None

def run_full_analysis(config, args, stock_codes):
    pipeline = StockAnalysisPipeline(config=config, max_workers=args.workers)
    results = pipeline.run(stock_codes=stock_codes, dry_run=args.dry_run, send_notification=not args.no_notify)
    
    market_report = ""
    if not args.no_market_review:
        market_report = run_market_review(pipeline.notifier, pipeline.analyzer, pipeline.search_service) or ""

    # === Notion 同步逻辑修复版 ===
    notion_token = os.environ.get("NOTION_TOKEN")
    database_id = "bf217c149f1e4ab2918f58fc2a813213" 

    if notion_token and results and not args.dry_run:
        logger.info("正在执行 Notion 数据库同步...")
        notion = Client(auth=notion_token)
        for r in results:
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                report_text = pipeline.notifier.generate_single_stock_report(r)
                # 抓取我们注入的真实涨跌幅
                change_val = getattr(r, 'realtime_pct_change', 0.0)

                properties = {
                    "Stock name": {"title": [{"text": {"content": f"{r.name}({r.code})"}}]},
                    "分析日期": {"date": {"start": today}},
                    # Notion 格式：1% 对应 0.01
                    "涨跌幅%": {"number": float(change_val) / 100}, 
                    # 确保名字和 Notion 表头完全一致
                    "完整分析": {"rich_text": [{"type": "text", "text": {"content": str(report_text)[:1900]}}]}
                }
                notion.pages.create(parent={"database_id": database_id}, properties=properties)
                logger.info(f"Notion 同步成功: {r.name}")
            except Exception as e:
                logger.error(f"Notion 同步失败 ({getattr(r, 'name', '未知')}): {e}")

    try:
        feishu_doc = FeishuDocManager()
        if feishu_doc.is_configured() and (results or market_report):
            now = datetime.now(timezone(timedelta(hours=8)))
            doc_title = f"{now.strftime('%Y-%m-%d %H:%M')} 大盘复盘"
            full_content = ""
            if market_report: full_content += f"# 📈 大盘复盘\n\n{market_report}\n\n---\n\n"
            if results: full_content += f"# 🚀 个股决策仪表盘\n\n{pipeline.notifier.generate_dashboard_report(results)}"
            feishu_doc.create_daily_doc(doc_title, full_content)
    except Exception as e: logger.error(f"飞书同步失败: {e}")

def main():
    args = parse_arguments()
    config = get_config()
    setup_logging(debug=args.debug, log_dir=config.log_dir)
    stock_codes = [c.strip() for c in args.stocks.split(',')] if args.stocks else None
    run_full_analysis(config, args, stock_codes)
    return 0

if __name__ == "__main__":
    sys.exit(main())
