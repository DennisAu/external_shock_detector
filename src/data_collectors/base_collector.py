"""
数据采集器基类
所有数据采集都使用免费接口或爬虫
"""
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import time
from functools import wraps


def retry(max_retries=3, delay=1, backoff=2):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"重试{max_retries}次后仍失败: {e}")
                        raise
                    logger.warning(f"第{retries}次重试，等待{current_delay}秒...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


class BaseDataCollector(ABC):
    """数据采集器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    @abstractmethod
    async def collect(self, *args, **kwargs) -> pd.DataFrame:
        """采集数据"""
        pass
    
    @abstractmethod
    async def get_latest(self) -> pd.DataFrame:
        """获取最新数据"""
        pass
    
    def safe_request(self, url: str, params: Dict = None) -> Dict:
        """安全请求（同步版本）"""
        import requests
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"请求失败 {url}: {e}")
            return {}
