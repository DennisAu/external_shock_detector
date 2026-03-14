"""
工具函数模块
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import logging

# 配置日志
def setup_logger(name: str = "shock_detector", level: str = "INFO"):
    """配置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger


def get_trading_days(start_date: str, end_date: str) -> List[datetime]:
    """
    获取交易日列表
    
    Args:
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
    
    Returns:
        交易日列表
    """
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        trading_days = pd.to_datetime(df['trade_date'])
        mask = (trading_days >= start) & (trading_days <= end)
        
        return trading_days[mask].tolist()
        
    except Exception as e:
        # 如果获取失败，返回工作日
        logging.warning(f"获取交易日失败: {e}，使用工作日代替")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return dates.tolist()


def calculate_technical_indicators(
    df: pd.DataFrame,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    计算常用技术指标
    
    Args:
        df: 包含价格数据的DataFrame
        price_col: 价格列名
    """
    result = df.copy()
    
    # 移动平均
    result['ma5'] = result[price_col].rolling(5).mean()
    result['ma10'] = result[price_col].rolling(10).mean()
    result['ma20'] = result[price_col].rolling(20).mean()
    result['ma60'] = result[price_col].rolling(60).mean()
    
    # 收益率
    result['return_1d'] = result[price_col].pct_change(1)
    result['return_5d'] = result[price_col].pct_change(5)
    result['return_10d'] = result[price_col].pct_change(10)
    result['return_20d'] = result[price_col].pct_change(20)
    
    # 波动率
    result['volatility_5d'] = result['return_1d'].rolling(5).std()
    result['volatility_20d'] = result['return_1d'].rolling(20).std()
    result['volatility_60d'] = result['return_1d'].rolling(60).std()
    
    # 振幅
    result['amplitude'] = (result['high'] - result['low']) / result[price_col]
    
    # 换手率（如果有volume）
    if 'volume' in result.columns:
        result['turnover'] = result['volume'] / result['volume'].rolling(20).mean()
    
    return result


def align_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    date_col: str = 'trade_date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对齐两个DataFrame的日期
    
    Args:
        df1, df2: 需要对齐的DataFrame
        date_col: 日期列名
    
    Returns:
        对齐后的两个DataFrame
    """
    # 确保日期格式一致
    df1 = df1.copy()
    df2 = df2.copy()
    
    df1[date_col] = pd.to_datetime(df1[date_col])
    df2[date_col] = pd.to_datetime(df2[date_col])
    
    # 找到共同日期
    common_dates = set(df1[date_col]) & set(df2[date_col])
    
    # 筛选
    df1_aligned = df1[df1[date_col].isin(common_dates)].sort_values(date_col)
    df2_aligned = df2[df2[date_col].isin(common_dates)].sort_values(date_col)
    
    return df1_aligned, df2_aligned


def normalize_series(series: pd.Series, method: str = 'zscore') -> pd.Series:
    """
    标准化序列
    
    Args:
        series: 待标准化的序列
        method: 标准化方法 (zscore, minmax, robust)
    """
    if method == 'zscore':
        return (series - series.mean()) / series.std()
    elif method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'robust':
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        return (series - median) / iqr
    else:
        return series


def detect_outliers(
    series: pd.Series,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.Series:
    """
    检测异常值
    
    Args:
        series: 待检测的序列
        method: 检测方法 (iqr, zscore)
        threshold: 阈值
    
    Returns:
        布尔序列，True表示异常值
    """
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (series < lower) | (series > upper)
    elif method == 'zscore':
        z_score = (series - series.mean()) / series.std()
        return abs(z_score) > threshold
    else:
        return pd.Series([False] * len(series))


def calculate_correlation_matrix(
    df: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    计算相关性矩阵
    
    Args:
        df: 数据DataFrame
        method: 相关性计算方法 (pearson, spearman, kendall)
    """
    return df.corr(method=method)


def resample_data(
    df: pd.DataFrame,
    freq: str = 'W',
    date_col: str = 'trade_date',
    agg_dict: dict = None
) -> pd.DataFrame:
    """
    重采样数据
    
    Args:
        df: 数据DataFrame
        freq: 重采样频率 (W=周, M=月, Q=季)
        date_col: 日期列名
        agg_dict: 聚合方式字典
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    # 只聚合存在的列
    valid_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    return df.resample(freq).agg(valid_agg).dropna()


class CacheManager:
    """
    简单的缓存管理器
    """
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str):
        """获取缓存"""
        return self.cache.get(key)
    
    def set(self, key: str, value):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # 删除最早的缓存
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()


# 全局缓存实例
cache = CacheManager()
