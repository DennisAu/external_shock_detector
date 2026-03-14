"""
免费数据源汇总
整理所有可用的免费数据接口和爬虫方法
"""
import akshare as ak
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
import time


class FreeDataSources:
    """
    免费数据源集合
    所有数据源都是免费的，无需token或付费
    """
    
    # ===============================
    # A股市场数据（全部免费）
    # ===============================
    
    @staticmethod
    def get_index_realtime() -> pd.DataFrame:
        """
        获取A股主要指数实时行情
        免费接口: ak.stock_zh_index_spot_em()
        """
        try:
            df = ak.stock_zh_index_spot_em()
            # 筛选主要指数
            main_indices = ['上证指数', '深证成指', '创业板指', '沪深300', '上证50', '中证500', '科创50']
            df = df[df['名称'].isin(main_indices)]
            return df
        except Exception as e:
            logger.error(f"获取指数实时行情失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_index_history(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取指数历史数据
        免费接口: ak.stock_zh_index_daily()
        
        Args:
            symbol: 指数代码，如 sh000001
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
        """
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
            return df
        except Exception as e:
            logger.error(f"获取指数历史数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_sector_realtime() -> pd.DataFrame:
        """
        获取行业板块实时行情
        免费接口: ak.stock_board_industry_name_em()
        用于分析板块分化
        """
        try:
            df = ak.stock_board_industry_name_em()
            return df
        except Exception as e:
            logger.error(f"获取板块行情失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_sector_history(sector_name: str, period: str = "daily") -> pd.DataFrame:
        """
        获取板块历史K线
        免费接口: ak.stock_board_industry_hist_em()
        
        Args:
            sector_name: 板块名称，如 "石油行业"
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
            logger.error(f"获取板块历史数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_concept_sectors() -> pd.DataFrame:
        """
        获取概念板块列表
        免费接口: ak.stock_board_concept_name_em()
        """
        try:
            df = ak.stock_board_concept_name_em()
            return df
        except Exception as e:
            logger.error(f"获取概念板块失败: {e}")
            return pd.DataFrame()
    
    # ===============================
    # 资金流向数据（全部免费）
    # ===============================
    
    @staticmethod
    def get_north_flow() -> pd.DataFrame:
        """
        获取北向资金流向
        免费接口: 多种备选方案
        用于验证外资情绪传导
        """
        try:
            # 尝试多种akshare接口
            try:
                df = ak.stock_hsgt_north_net_flow_in_em()
            except:
                try:
                    df = ak.stock_hsgt_hist_em(symbol="北向资金")
                except:
                    try:
                        df = ak.stock_em_hsgt_north_net_flow_in(indicator="北向资金")
                    except:
                        # 返回空DataFrame
                        return pd.DataFrame()
            
            if not df.empty:
                if 'date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"获取北向资金失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_sector_money_flow(sector_name: str = None) -> pd.DataFrame:
        """
        获取板块资金流向
        免费接口: ak.stock_board_industry_fund_flow_em()
        """
        try:
            if sector_name:
                df = ak.stock_individual_fund_flow(stock=sector_name, market="sh")
            else:
                df = ak.stock_board_industry_fund_flow_em()
            return df
        except Exception as e:
            logger.error(f"获取板块资金流失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_margin_data(date: str = None) -> pd.DataFrame:
        """
        获取融资融券数据
        免费接口: ak.stock_margin_detail_sz_date() / ak.stock_margin_underlying_info_sz_sh()
        用于计算内生风险偏好因子
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y%m%d")
            # 深市融资融券
            df_sz = ak.stock_margin_detail_sz_date(date=date)
            # 沪市融资融券
            df_sh = ak.stock_margin_underlying_info_sz_sh(date=date)
            
            # 合并
            if not df_sz.empty and not df_sh.empty:
                df = pd.concat([df_sz, df_sh], ignore_index=True)
            else:
                df = df_sz if not df_sz.empty else df_sh
            
            return df
        except Exception as e:
            logger.error(f"获取融资融券数据失败: {e}")
            return pd.DataFrame()
    
    # ===============================
    # 全球市场数据（yfinance免费）
    # ===============================
    
    @staticmethod
    def get_global_index(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取全球指数数据
        免费接口: yfinance
        
        Args:
            symbol: 指数代码
                - 美股: ^GSPC(标普500), ^IXIC(纳指), ^DJI(道指), ^VIX(恐慌指数)
                - 欧洲: ^FTSE(英国), ^GDAXI(德国), ^FCHI(法国)
                - 亚洲: ^N225(日经), ^KS11(韩国), ^HSI(恒生)
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            if start_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period="1y")
            
            df = df.reset_index()
            df['trade_date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            return df
        except Exception as e:
            logger.error(f"获取全球指数失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_crude_oil(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取原油期货数据
        免费接口: yfinance
        代码: CL=F (WTI原油), BZ=F (布伦特原油)
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker("CL=F")  # WTI原油
            
            if start_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period="1y")
            
            df = df.reset_index()
            df['trade_date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            return df
        except Exception as e:
            logger.error(f"获取原油数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_bdti_index() -> pd.DataFrame:
        """
        获取BDTI原油运输指数
        免费接口: yfinance
        代码: ^BDTI
        用于验证运输端实锤
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker("^BDTI")
            df = ticker.history(period="1y")
            
            df = df.reset_index()
            df['trade_date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close'
            })
            return df
        except Exception as e:
            logger.error(f"获取BDTI指数失败: {e}")
            return pd.DataFrame()
    
    # ===============================
    # 国内经济指标（免费爬取）
    # ===============================
    
    @staticmethod
    def get_pmi_data() -> pd.DataFrame:
        """
        获取PMI数据
        免费接口: ak.macro_china_pmi_yearly()
        用于内生性验证
        """
        try:
            df = ak.macro_china_pmi_yearly()
            return df
        except Exception as e:
            logger.error(f"获取PMI数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_social_financing() -> pd.DataFrame:
        """
        获取社融数据
        免费接口: ak.macro_china_shrzgm()
        用于内生性验证
        """
        try:
            df = ak.macro_china_shrzgm()
            return df
        except Exception as e:
            logger.error(f"获取社融数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_interest_rate() -> pd.DataFrame:
        """
        获取银行间利率数据
        免费接口: ak.rate_interbank(market=" Shibor")
        用于内生因子：DR007
        """
        try:
            # Shibor利率
            df = ak.rate_interbank(market="Shibor")
            return df
        except Exception as e:
            logger.error(f"获取利率数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_bond_yield() -> pd.DataFrame:
        """
        获取国债收益率
        免费接口: ak.bond_china_yield(start_date="20000206")
        用于内生因子：10年期国债收益率
        """
        try:
            df = ak.bond_china_yield(start_date="20200101")
            return df
        except Exception as e:
            logger.error(f"获取国债收益率失败: {e}")
            return pd.DataFrame()
    
    # ===============================
    # 新闻与舆情（免费爬取）
    # ===============================
    
    @staticmethod
    def get_stock_news(symbol: str = None) -> pd.DataFrame:
        """
        获取股票新闻
        免费接口: ak.stock_news_em()
        用于事件关键词匹配
        """
        try:
            if symbol:
                df = ak.stock_news_em(symbol=symbol)
            else:
                # 获取财经头条
                df = ak.stock_news_em(symbol="财经")
            return df
        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_global_news() -> pd.DataFrame:
        """
        获取全球财经新闻
        免费接口: ak.stock_news_global_em()
        用于监控地缘政治事件
        """
        try:
            df = ak.stock_news_global_em()
            return df
        except Exception as e:
            logger.error(f"获取全球新闻失败: {e}")
            return pd.DataFrame()
    
    # ===============================
    # 涨跌停与市场情绪（免费）
    # ===============================
    
    @staticmethod
    def get_limit_up_pool(date: str = None) -> pd.DataFrame:
        """
        获取涨停股票池
        免费接口: ak.stock_zt_pool_em()
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y%m%d")
            df = ak.stock_zt_pool_em(date=date)
            return df
        except Exception as e:
            logger.error(f"获取涨停池失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_limit_down_pool(date: str = None) -> pd.DataFrame:
        """
        获取跌停股票池
        免费接口: ak.stock_dt_pool_em()
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y%m%d")
            df = ak.stock_dt_pool_em(date=date)
            return df
        except Exception as e:
            logger.error(f"获取跌停池失败: {e}")
            return pd.DataFrame()
    
    # ===============================
    # 特殊板块：原油相关
    # ===============================
    
    @staticmethod
    def get_oil_stocks() -> List[str]:
        """
        获取原油产业链相关股票
        用于行业横截面验证
        """
        oil_stocks = {
            'upstream': {  # 上游开采
                '中国石油': '601857',
                '中国石化': '600028',
                '中海油服': '601808',
                '海油工程': '600583',
                '杰瑞股份': '002353',
            },
            'midstream': {  # 中游运输
                '招商轮船': '601872',
                '中远海能': '600026',
                '招商南油': '601975',
            },
            'downstream': {  # 下游炼化
                '上海石化': '600688',
                '华锦股份': '000059',
                '恒力石化': '600346',
                '荣盛石化': '002493',
            },
            'oil_services': {  # 油服
                '石化油服': '600871',
                '贝肯能源': '002828',
                '通源石油': '300164',
            }
        }
        return oil_stocks
    
    @staticmethod
    def get_cost_sensitive_stocks() -> Dict[str, List[str]]:
        """
        获取原油成本敏感型股票
        用于行业横截面验证
        """
        sensitive_stocks = {
            'aviation': {  # 航空（成本敏感）
                '中国国航': '601111',
                '南方航空': '600029',
                '东方航空': '600115',
                '春秋航空': '601021',
            },
            'shipping': {  # 航运（成本敏感）
                '中远海控': '601919',
                '中集集团': '000039',
            },
            'logistics': {  # 物流
                '顺丰控股': '002352',
                '圆通速递': '600233',
            },
            'chemical': {  # 化工
                '万华化学': '600309',
                '华鲁恒升': '600426',
                '恒逸石化': '000703',
            }
        }
        return sensitive_stocks
    
    # ===============================
    # 加密货币数据（yfinance免费）
    # ===============================
    
    @staticmethod
    def get_crypto_price(
        symbol: str = "BTC-USD",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取加密货币价格数据
        免费接口: yfinance
        
        Args:
            symbol: 加密货币代码
                - BTC-USD: 比特币
                - ETH-USD: 以太坊
                - BTC=F: 比特币期货
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            if start_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period="1y")
            
            df = df.reset_index()
            df['trade_date'] = pd.to_datetime(df['Date'])
            df['symbol'] = symbol
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            return df
        except Exception as e:
            logger.error(f"获取加密货币数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_btc_price(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取比特币价格"""
        return FreeDataSources.get_crypto_price("BTC-USD", start_date, end_date)
    
    @staticmethod
    def get_eth_price(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取以太坊价格"""
        return FreeDataSources.get_crypto_price("ETH-USD", start_date, end_date)
    
    # ===============================
    # 汇率数据（yfinance免费）
    # ===============================
    
    @staticmethod
    def get_forex_rate(
        symbol: str = "DX-Y.NYB",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取汇率数据
        免费接口: yfinance
        
        Args:
            symbol: 汇率代码
                - DX-Y.NYB: 美元指数
                - CNY=X: 美元人民币
                - JPY=X: 美元日元
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            if start_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period="1y")
            
            df = df.reset_index()
            df['trade_date'] = pd.to_datetime(df['Date'])
            df['symbol'] = symbol
            return df
        except Exception as e:
            logger.error(f"获取汇率数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_dxy(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取美元指数"""
        return FreeDataSources.get_forex_rate("DX-Y.NYB", start_date, end_date)
    
    @staticmethod
    def get_usdcny(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取美元人民币汇率"""
        return FreeDataSources.get_forex_rate("CNY=X", start_date, end_date)
    
    # ===============================
    # 美债收益率数据（yfinance免费）
    # ===============================
    
    @staticmethod
    def get_treasury_yield(
        symbol: str = "^TNX",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取美国国债收益率数据
        免费接口: yfinance
        
        Args:
            symbol: 收益率代码
                - ^TNX: 10年期美债收益率
                - ^FVX: 2年期美债收益率
                - ^TYX: 30年期美债收益率
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            if start_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period="1y")
            
            df = df.reset_index()
            df['trade_date'] = pd.to_datetime(df['Date'])
            df['symbol'] = symbol
            # 收益率通常用Close列
            df['yield'] = df['Close'] / 100  # 转换为小数
            return df
        except Exception as e:
            logger.error(f"获取美债收益率失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_us_10y_yield(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取美国10年期国债收益率"""
        return FreeDataSources.get_treasury_yield("^TNX", start_date, end_date)
    
    @staticmethod
    def get_us_2y_yield(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取美国2年期国债收益率"""
        return FreeDataSources.get_treasury_yield("^FVX", start_date, end_date)
    
    # ===============================
    # 贵金属数据（yfinance免费）
    # ===============================
    
    @staticmethod
    def get_precious_metals(
        symbol: str = "GC=F",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取贵金属期货数据
        免费接口: yfinance
        
        Args:
            symbol: 贵金属代码
                - GC=F: 黄金期货
                - SI=F: 白银期货
                - PL=F: 铂金期货
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            if start_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period="1y")
            
            df = df.reset_index()
            df['trade_date'] = pd.to_datetime(df['Date'])
            df['symbol'] = symbol
            return df
        except Exception as e:
            logger.error(f"获取贵金属数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_gold_price(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取黄金价格"""
        return FreeDataSources.get_precious_metals("GC=F", start_date, end_date)
    
    @staticmethod
    def get_silver_price(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取白银价格"""
        return FreeDataSources.get_precious_metals("SI=F", start_date, end_date)
    
    # ===============================
    # 基本金属数据（yfinance免费）
    # ===============================
    
    @staticmethod
    def get_base_metals(
        symbol: str = "HG=F",
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取基本金属期货数据
        免费接口: yfinance
        
        Args:
            symbol: 金属代码
                - HG=F: 铜期货
                - ZN=F: 锌期货
                - NI=F: 镍期货
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            if start_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period="1y")
            
            df = df.reset_index()
            df['trade_date'] = pd.to_datetime(df['Date'])
            df['symbol'] = symbol
            return df
        except Exception as e:
            logger.error(f"获取基本金属数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_copper_price(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取铜价"""
        return FreeDataSources.get_base_metals("HG=F", start_date, end_date)
    
    # ===============================
    # 加密货币相关A股板块
    # ===============================
    
    @staticmethod
    def get_crypto_related_stocks() -> Dict[str, Dict[str, str]]:
        """
        获取加密货币相关A股股票
        用于行业横截面验证
        """
        crypto_stocks = {
            'blockchain': {  # 区块链概念
                '飞天诚信': '300386',
                '四方精创': '300468',
                '广电运通': '002152',
                '恒生电子': '600570',
                '高伟达': '300465',
            },
            'mining': {  # 矿机/算力
                '中嘉博创': '000977',
                '网宿科技': '300017',
                '浪潮信息': '000977',
            },
            'digital_currency': {  # 数字货币
                '数字认证': '300579',
                '格尔软件': '603232',
                '卫士通': '002268',
            }
        }
        return crypto_stocks


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试免费数据源")
    print("=" * 60)
    
    # 测试A股指数
    print("\n1. 测试A股指数实时行情...")
    df_index = FreeDataSources.get_index_realtime()
    print(df_index.head() if not df_index.empty else "获取失败")
    
    # 测试板块数据
    print("\n2. 测试板块实时行情...")
    df_sector = FreeDataSources.get_sector_realtime()
    print(df_sector.head() if not df_sector.empty else "获取失败")
    
    # 测试北向资金
    print("\n3. 测试北向资金...")
    df_north = FreeDataSources.get_north_flow()
    print(df_north.tail() if not df_north.empty else "获取失败")
    
    # 测试原油数据
    print("\n4. 测试原油期货...")
    df_oil = FreeDataSources.get_crude_oil()
    print(df_oil.tail() if not df_oil.empty else "获取失败")
    
    print("\n" + "=" * 60)
    print("测试完成")
