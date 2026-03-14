"""
A股数据采集器 - 使用akshare免费接口
包括：指数、板块、个股、资金流向等数据
"""
import akshare as ak
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import time

from .base_collector import BaseDataCollector


class AStockCollector(BaseDataCollector):
    """A股数据采集器 - 完全免费"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.index_symbols = config.get('index_symbols', [
            "000001",  # 上证指数
            "399001",  # 深证成指
            "399006",  # 创业板指
        ])
        
    async def collect(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        采集A股指数数据
        
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
        """
        all_data = []
        
        for symbol in self.index_symbols:
            try:
                logger.info(f"采集指数 {symbol} 数据...")
                df = self._get_index_data(symbol, start_date, end_date)
                if not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
                time.sleep(0.5)  # 避免请求过快
            except Exception as e:
                logger.error(f"采集指数 {symbol} 失败: {e}")
                
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def _get_index_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取单个指数数据 - 使用akshare免费接口
        
        akshare提供多个免费接口：
        - stock_zh_index_daily: 历史日线数据
        - stock_zh_index_spot: 实时行情
        """
        try:
            # 使用akshare获取指数历史数据（免费）
            df = ak.stock_zh_index_daily(symbol=f"sh{symbol}" if symbol.startswith('0') else f"sz{symbol}")
            
            if df.empty:
                # 尝试另一种格式
                df = ak.stock_zh_index_daily(symbol=symbol)
            
            if not df.empty:
                df = df.rename(columns={
                    'date': 'trade_date',
                    'open': 'open',
                    'high': 'high', 
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
                
                if start_date:
                    df = df[df['trade_date'] >= start_date]
                if end_date:
                    df = df[df['trade_date'] <= end_date]
                    
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                
            return df
            
        except Exception as e:
            logger.error(f"获取指数 {symbol} 数据失败: {e}")
            return pd.DataFrame()
    
    async def get_latest(self) -> pd.DataFrame:
        """获取最新行情"""
        try:
            # 获取实时行情（免费）
            df = ak.stock_zh_index_spot()
            return df
        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            return pd.DataFrame()
    
    def get_sector_data(self, sector_symbol: str, start_date: str = None) -> pd.DataFrame:
        """
        获取板块指数数据
        
        Args:
            sector_symbol: 板块代码
            start_date: 开始日期
        """
        try:
            df = ak.stock_board_industry_index_em(symbol=sector_symbol)
            return df
        except Exception as e:
            logger.error(f"获取板块数据失败: {e}")
            return pd.DataFrame()
    
    def get_all_sectors_realtime(self) -> pd.DataFrame:
        """获取所有板块实时行情（免费）"""
        try:
            # 东方财富板块行情（免费爬取）
            df = ak.stock_board_industry_name_em()
            return df
        except Exception as e:
            logger.error(f"获取板块行情失败: {e}")
            return pd.DataFrame()
    
    def get_sector_stocks(self, sector_name: str) -> pd.DataFrame:
        """
        获取板块成分股
        
        Args:
            sector_name: 板块名称，如"石油行业"
        """
        try:
            df = ak.stock_board_industry_cons_em(symbol=sector_name)
            return df
        except Exception as e:
            logger.error(f"获取板块成分股失败: {e}")
            return pd.DataFrame()
    
    def get_north_flow(self) -> pd.DataFrame:
        """
        获取北向资金流向（免费）
        用于判断外资情绪
        """
        try:
            # 北向资金数据（免费）
            df = ak.stock_hsgt_north_net_flow_in_em()
            df['trade_date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"获取北向资金失败: {e}")
            return pd.DataFrame()
    
    def get_margin_data(self) -> pd.DataFrame:
        """
        获取融资融券数据（免费）
        用于判断市场情绪
        """
        try:
            # 融资融券汇总（免费）
            df = ak.stock_margin_underlying_info_sz_sh(date=datetime.now().strftime("%Y%m%d"))
            return df
        except Exception as e:
            logger.error(f"获取融资融券数据失败: {e}")
            return pd.DataFrame()
    
    def get_stock_news(self, stock_code: str = None) -> pd.DataFrame:
        """
        获取股票新闻（免费爬取）
        
        Args:
            stock_code: 股票代码，为None时获取市场新闻
        """
        try:
            if stock_code:
                df = ak.stock_news_em(symbol=stock_code)
            else:
                # 获取市场整体新闻
                df = ak.stock_news_em(symbol="财经新闻")
            return df
        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return pd.DataFrame()


class SectorAnalysisCollector(BaseDataCollector):
    """板块分析数据采集器 - 用于识别板块分化"""
    
    async def collect(self, trade_date: str = None) -> pd.DataFrame:
        """
        采集板块涨跌数据
        
        Args:
            trade_date: 交易日期 YYYYMMDD
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")
            
        try:
            # 获取行业板块涨跌幅（免费）
            df = ak.stock_board_industry_name_em()
            return df
        except Exception as e:
            logger.error(f"采集板块数据失败: {e}")
            return pd.DataFrame()
    
    async def get_latest(self) -> pd.DataFrame:
        """获取最新板块数据"""
        return await self.collect()
    
    def get_sector_kline(self, sector_name: str, period: str = "daily") -> pd.DataFrame:
        """
        获取板块K线数据
        
        Args:
            sector_name: 板块名称
            period: 周期 daily/weekly
        """
        try:
            df = ak.stock_board_industry_hist_em(
                symbol=sector_name,
                period=period,
                adjust=""
            )
            return df
        except Exception as e:
            logger.error(f"获取板块K线失败: {e}")
            return pd.DataFrame()
    
    def get_concept_sectors(self) -> pd.DataFrame:
        """获取概念板块数据（免费）"""
        try:
            df = ak.stock_board_concept_name_em()
            return df
        except Exception as e:
            logger.error(f"获取概念板块失败: {e}")
            return pd.DataFrame()


# 测试代码
if __name__ == "__main__":
    import asyncio
    
    async def test():
        config = {
            'index_symbols': ['000001', '399001', '399006']
        }
        
        collector = AStockCollector(config)
        
        # 测试获取板块实时数据
        print("获取板块实时数据...")
        sectors = collector.get_all_sectors_realtime()
        print(sectors.head())
        
        # 测试北向资金
        print("\n获取北向资金...")
        north_flow = collector.get_north_flow()
        print(north_flow.tail())
    
    asyncio.run(test())
