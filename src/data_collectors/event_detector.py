"""
外部冲击事件检测器
基于新闻关键词和量化指标的双重验证
"""
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass


@dataclass
class ShockEvent:
    """外部冲击事件数据类"""
    event_id: str
    event_type: str  # geopolitical, commodity_shock, financial_crisis, etc.
    event_date: datetime  # T0 事件爆发日
    trigger_source: str  # 触发源描述
    confidence_score: float  # 置信度评分 0-1
    
    # 量化验证指标
    bdti_change: float  # BDTI原油运输指数涨幅
    oil_price_change: float  # 布伦特原油涨幅
    vix_change: float  # VIX变化
    
    # 事件窗口
    analysis_window_start: datetime  # T-2
    analysis_window_end: datetime  # T+10
    clean_period_start: datetime  # 洁净期开始（T-60）
    clean_period_end: datetime  # 洁净期结束（T-3）
    
    # 验证状态
    is_validated: bool = False
    validation_details: Dict = None


class EventKeywordMatcher:
    """事件关键词匹配器"""
    
    # 地缘政治事件关键词
    GEOPOLITICAL_KEYWORDS = {
        'war_conflict': ['战争', '军事冲突', '空袭', '导弹袭击', '入侵', '武装冲突'],
        'strait_blockade': ['霍尔木兹', '苏伊士运河', '海峡封锁', '航道中断', '航道受阻'],
        'oil_facility': ['油田遇袭', '炼油厂爆炸', '油管中断', '储油设施'],
        'sanction': ['制裁', '出口禁令', '贸易限制', '石油禁运'],
        'terror_attack': ['恐怖袭击', '油轮袭击', '港口爆炸'],
    }
    
    # 原油供应链关键词
    OIL_SUPPLY_KEYWORDS = {
        'production_cut': ['减产', 'OPEC', '产量下降', '供应中断'],
        'transport_disruption': ['油轮', '运输中断', '航运受阻', '港口关闭'],
        'price_shock': ['原油暴涨', '油价飙升', '能源危机'],
    }
    
    # 加密货币冲击关键词 (新增)
    CRYPTO_KEYWORDS = {
        'exchange_collapse': ['交易所暴雷', '交易所倒闭', 'FTX', '币安', '提现困难'],
        'price_crash': ['BTC暴跌', '比特币崩盘', '加密货币崩盘', '币圈暴跌'],
        'regulation': ['加密货币监管', '挖矿禁令', '虚拟货币禁令', '稳定币监管'],
        'stablecoin': ['稳定币脱锚', 'UST崩盘', 'Luna崩盘', '稳定币危机'],
        'major_event': ['比特币ETF', '以太坊升级', '减半', '比特币减半'],
    }
    
    # 贵金属异动关键词 (新增)
    PRECIOUS_METALS_KEYWORDS = {
        'gold_surge': ['黄金暴涨', '金价飙升', '避险买盘', '央行购金'],
        'silver_squeeze': ['白银逼空', '银价暴涨'],
        'safe_haven': ['避险需求激增', '避险资产'],
    }
    
    # 汇率冲击关键词 (新增)
    FOREX_KEYWORDS = {
        'dxy_surge': ['美元暴涨', '美元指数飙升', '美元走强'],
        'cny_depreciation': ['人民币贬值', '人民币破7', '汇率贬值'],
        'intervention': ['汇率干预', '央行干预汇率', '日元干预'],
        'currency_crisis': ['货币崩盘', '汇率危机', '新兴市场货币'],
    }
    
    # 利率冲击关键词 (新增)
    RATE_KEYWORDS = {
        'yield_surge': ['美债收益率飙升', '收益率暴涨', '国债收益率上升'],
        'yield_inversion': ['收益率倒挂', '期限倒挂', '衰退信号'],
        'fed_policy': ['美联储加息', '美联储降息', '缩表', '量化紧缩'],
        'liquidity': ['流动性紧缩', '流动性危机', '信用紧缩'],
    }
    
    # 排除关键词（避免误识别）
    EXCLUDE_KEYWORDS = [
        '演习', '训练', '模拟', '预测', '分析', '评论',
        '历史', '回顾', '假设', '如果', '可能', '或将'
    ]
    
    @classmethod
    def match_event(cls, text: str) -> Tuple[bool, str, float]:
        """
        匹配事件关键词
        
        Returns:
            (是否匹配, 事件类型, 置信度)
        """
        text = text.lower()
        
        # 排除词检查
        for exclude_word in cls.EXCLUDE_KEYWORDS:
            if exclude_word in text:
                return False, '', 0.0
        
        # 各类事件匹配得分
        scores = {}
        
        # 地缘政治匹配
        geo_score = 0
        geo_type = ''
        for category, keywords in cls.GEOPOLITICAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    geo_score += 1
                    geo_type = category
        if geo_score > 0:
            scores[f'geopolitical_{geo_type}'] = geo_score
        
        # 原油供应链匹配
        oil_score = 0
        oil_type = ''
        for category, keywords in cls.OIL_SUPPLY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    oil_score += 1
                    oil_type = category
        if oil_score > 0:
            scores[f'oil_supply_{oil_type}'] = oil_score
        
        # 加密货币匹配 (新增)
        crypto_score = 0
        crypto_type = ''
        for category, keywords in cls.CRYPTO_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    crypto_score += 1
                    crypto_type = category
        if crypto_score > 0:
            scores[f'crypto_{crypto_type}'] = crypto_score
        
        # 贵金属匹配 (新增)
        metals_score = 0
        metals_type = ''
        for category, keywords in cls.PRECIOUS_METALS_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    metals_score += 1
                    metals_type = category
        if metals_score > 0:
            scores[f'precious_metals_{metals_type}'] = metals_score
        
        # 汇率匹配 (新增)
        forex_score = 0
        forex_type = ''
        for category, keywords in cls.FOREX_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    forex_score += 1
                    forex_type = category
        if forex_score > 0:
            scores[f'forex_{forex_type}'] = forex_score
        
        # 利率匹配 (新增)
        rate_score = 0
        rate_type = ''
        for category, keywords in cls.RATE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    rate_score += 1
                    rate_type = category
        if rate_score > 0:
            scores[f'rate_{rate_type}'] = rate_score
        
        # 如果没有匹配到任何事件
        if not scores:
            return False, '', 0.0
        
        # 选择得分最高的事件类型
        best_event_type = max(scores, key=scores.get)
        best_score = scores[best_event_type]
        
        # 计算置信度
        confidence = min(best_score / 5.0, 1.0)  # 最高5个关键词
        
        return True, best_event_type, confidence


class EventValidator:
    """
    事件有效性验证器
    实现四重验证机制
    """
    
    # 验证阈值
    THRESHOLDS = {
        # 事件触发阈值
        'bdti_single_day': 0.05,  # BDTI单日涨幅≥5%
        'bdti_3day': 0.10,  # BDTI 3日累计涨幅≥10%
        
        # 价格冲击阈值
        'oil_single_day': 0.03,  # 原油单日涨幅≥3%
        'oil_3day': 0.06,  # 原油3日累计涨幅≥6%
        
        # 统计显著性
        'oil_std_multiplier': 2.0,  # 涨幅突破2倍标准差
        
        # 前置期检查
        'pre_event_days': 10,  # 事件前10个交易日
        'clean_period_days': 60,  # 洁净期60个交易日
    }
    
    def __init__(self):
        self.keyword_matcher = EventKeywordMatcher()
    
    def validate_event(
        self,
        news_text: str,
        event_date: datetime,
        bdti_data: pd.DataFrame,
        oil_data: pd.DataFrame,
        domestic_indicators: Dict = None
    ) -> Tuple[bool, ShockEvent]:
        """
        验证事件有效性
        
        Args:
            news_text: 新闻文本
            event_date: 事件日期
            bdti_data: BDTI运输指数数据
            oil_data: 原油价格数据
            domestic_indicators: 国内经济指标（PMI、社融等）
        
        Returns:
            (是否有效事件, 事件对象)
        """
        # Step 1: 关键词匹配
        matched, event_type, keyword_confidence = self.keyword_matcher.match_event(news_text)
        if not matched:
            return False, None
        
        # Step 2: 运输端验证
        bdti_valid, bdti_change = self._validate_bdti(event_date, bdti_data)
        if not bdti_valid:
            logger.info(f"BDTI运输指数验证未通过")
            return False, None
        
        # Step 3: 价格冲击验证
        oil_valid, oil_change = self._validate_oil_price(event_date, oil_data)
        if not oil_valid:
            logger.info(f"原油价格冲击验证未通过")
            return False, None
        
        # Step 4: 非内生性验证
        if domestic_indicators:
            endogenous = self._check_endogenous(event_date, domestic_indicators)
            if endogenous:
                logger.info(f"内生性检查未通过：存在国内驱动因素")
                return False, None
        
        # 构建事件对象
        event = ShockEvent(
            event_id=f"EVT_{event_date.strftime('%Y%m%d')}_{event_type}",
            event_type=event_type,
            event_date=event_date,
            trigger_source=news_text[:100],
            confidence_score=keyword_confidence,
            bdti_change=bdti_change,
            oil_price_change=oil_change,
            vix_change=0.0,  # 后续填充
            analysis_window_start=event_date - timedelta(days=2),
            analysis_window_end=event_date + timedelta(days=10),
            clean_period_start=event_date - timedelta(days=60),
            clean_period_end=event_date - timedelta(days=3),
            is_validated=True,
            validation_details={
                'keyword_matched': True,
                'bdti_validated': bdti_valid,
                'oil_validated': oil_valid,
                'non_endogenous': True
            }
        )
        
        return True, event
    
    def _validate_bdti(self, event_date: datetime, bdti_data: pd.DataFrame) -> Tuple[bool, float]:
        """
        验证BDTI原油运输指数
        
        条件：单日涨幅≥5% 或 3日累计涨幅≥10%
        """
        if bdti_data.empty:
            return False, 0.0
        
        # 获取事件日期前后的数据
        event_idx = bdti_data[bdti_data['trade_date'] == event_date].index
        if len(event_idx) == 0:
            # 找最近的交易日
            bdti_data['date_diff'] = (bdti_data['trade_date'] - event_date).abs()
            event_idx = bdti_data['date_diff'].idxmin()
        
        # 计算涨跌幅
        try:
            current = bdti_data.loc[event_idx, 'close'].values[0]
            prev_1d = bdti_data.loc[event_idx - 1, 'close'].values[0]
            prev_3d = bdti_data.loc[event_idx - 3, 'close'].values[0]
            
            change_1d = (current - prev_1d) / prev_1d
            change_3d = (current - prev_3d) / prev_3d
            
            if change_1d >= self.THRESHOLDS['bdti_single_day']:
                return True, change_1d
            if change_3d >= self.THRESHOLDS['bdti_3day']:
                return True, change_3d
            
        except Exception as e:
            logger.error(f"BDTI计算失败: {e}")
        
        return False, 0.0
    
    def _validate_oil_price(self, event_date: datetime, oil_data: pd.DataFrame) -> Tuple[bool, float]:
        """
        验证原油价格冲击
        
        条件：单日涨幅≥3% 或 3日累计涨幅≥6%，且突破2倍标准差
        """
        if oil_data.empty:
            return False, 0.0
        
        try:
            # 找事件日期对应的数据
            oil_data['date_diff'] = (oil_data['trade_date'] - event_date).abs()
            event_idx = oil_data['date_diff'].idxmin()
            
            current = oil_data.loc[event_idx, 'close']
            prev_1d = oil_data.loc[event_idx - 1, 'close']
            prev_3d = oil_data.loc[event_idx - 3, 'close']
            
            change_1d = (current - prev_1d) / prev_1d
            change_3d = (current - prev_3d) / prev_3d
            
            # 计算过去1年的标准差
            oil_data['returns'] = oil_data['close'].pct_change()
            annual_std = oil_data.loc[:event_idx, 'returns'].tail(252).std()
            
            # 验证条件
            threshold_std = annual_std * self.THRESHOLDS['oil_std_multiplier']
            
            if change_1d >= self.THRESHOLDS['oil_single_day'] and change_1d >= threshold_std:
                return True, change_1d
            if change_3d >= self.THRESHOLDS['oil_3day'] and change_3d / 3 >= threshold_std:
                return True, change_3d
            
        except Exception as e:
            logger.error(f"原油价格验证失败: {e}")
        
        return False, 0.0
    
    def _check_endogenous(self, event_date: datetime, indicators: Dict) -> bool:
        """
        检查是否存在内生性驱动因素
        
        如果事件前10个交易日存在以下情况，判定为内生驱动：
        1. PMI显著变化
        2. 社融显著变化
        3. 国内原油产量/进口量显著变化
        """
        # 这里简化处理，实际需要获取具体数据
        return False


class HistoricalEventDatabase:
    """
    历史事件数据库
    存储和管理历史外部冲击事件
    """
    
    # 预定义的历史重大事件 (扩展版)
    HISTORICAL_EVENTS = [
        # ============ 能源冲击 ============
        {
            'event_id': 'EVT_20190914_saudi_attack',
            'event_type': 'geopolitical_oil_facility',
            'event_date': '2019-09-14',
            'description': '沙特阿美油田遇袭，原油产能减半',
            'oil_price_change': 0.15,  # 布伦特原油单日涨幅15%
            'impact_days': 5,
        },
        {
            'event_id': 'EVT_20220224_ukraine_war',
            'event_type': 'geopolitical_war_conflict',
            'event_date': '2022-02-24',
            'description': '俄乌冲突爆发',
            'oil_price_change': 0.08,
            'impact_days': 10,
        },
        {
            'event_id': 'EVT_20231007_gaza_conflict',
            'event_type': 'geopolitical_war_conflict',
            'event_date': '2023-10-07',
            'description': '巴以冲突爆发',
            'oil_price_change': 0.05,
            'impact_days': 7,
        },
        {
            'event_id': 'EVT_20231201_red_sea_crisis',
            'event_type': 'geopolitical_strait_blockade',
            'event_date': '2023-12-01',
            'description': '红海危机，胡塞武装袭击商船',
            'oil_price_change': 0.03,
            'impact_days': 15,
        },
        
        # ============ 加密货币冲击 (新增) ============
        {
            'event_id': 'EVT_20221111_ftx_collapse',
            'event_type': 'crypto_exchange_collapse',
            'event_date': '2022-11-11',
            'description': 'FTX交易所申请破产，加密货币市场崩盘',
            'btc_price_change': -0.20,  # BTC下跌20%
            'impact_days': 10,
        },
        {
            'event_id': 'EVT_20220509_terra_collapse',
            'event_type': 'crypto_stablecoin',
            'event_date': '2022-05-09',
            'description': 'Terra/Luna稳定币脱锚崩盘',
            'btc_price_change': -0.15,
            'impact_days': 7,
        },
        {
            'event_id': 'EVT_20240111_btc_etf',
            'event_type': 'crypto_major_event',
            'event_date': '2024-01-11',
            'description': '比特币现货ETF获批上市',
            'btc_price_change': 0.05,
            'impact_days': 5,
        },
        {
            'event_id': 'EVT_20200312_covid_crypto_crash',
            'event_type': 'crypto_price_crash',
            'event_date': '2020-03-12',
            'description': '疫情恐慌导致BTC单日暴跌50%',
            'btc_price_change': -0.50,
            'impact_days': 5,
        },
        
        # ============ 汇率冲击 (新增) ============
        {
            'event_id': 'EVT_20220922_jpy_intervention',
            'event_type': 'forex_intervention',
            'event_date': '2022-09-22',
            'description': '日本央行干预日元汇率',
            'usdjpy_change': -0.02,
            'impact_days': 3,
        },
        {
            'event_id': 'EVT_20220815_cny_depreciation',
            'event_type': 'forex_cny_depreciation',
            'event_date': '2022-08-15',
            'description': '人民币快速贬值破7',
            'usdcny_change': 0.02,
            'impact_days': 5,
        },
        
        # ============ 利率冲击 (新增) ============
        {
            'event_id': 'EVT_20230310_sv_b_collapse',
            'event_type': 'financial_crisis',
            'event_date': '2023-03-10',
            'description': '硅谷银行倒闭，美债收益率暴跌',
            'yield_change': -0.05,  # 收益率下降50BP
            'impact_days': 10,
        },
        {
            'event_id': 'EVT_20221021_uk_gilt_crisis',
            'event_type': 'financial_crisis',
            'event_date': '2022-10-21',
            'description': '英国养老金危机，国债收益率剧烈波动',
            'yield_change': 0.03,
            'impact_days': 7,
        },
        
        # ============ 贵金属异动 (新增) ============
        {
            'event_id': 'EVT_20200806_gold_record',
            'event_type': 'precious_metals_gold_surge',
            'event_date': '2020-08-06',
            'description': '黄金创历史新高突破2000美元',
            'gold_price_change': 0.03,
            'impact_days': 3,
        },
    ]
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.events = []
        self._load_historical_events()
    
    def _load_historical_events(self):
        """加载历史事件"""
        for event_data in self.HISTORICAL_EVENTS:
            event = ShockEvent(
                event_id=event_data['event_id'],
                event_type=event_data['event_type'],
                event_date=pd.to_datetime(event_data['event_date']),
                trigger_source=event_data['description'],
                confidence_score=1.0,  # 历史事件置信度100%
                bdti_change=0.0,  # 需要补充
                oil_price_change=event_data.get('oil_price_change', 0.0),  # 使用get避免KeyError
                vix_change=0.0,
                analysis_window_start=pd.to_datetime(event_data['event_date']) - timedelta(days=2),
                analysis_window_end=pd.to_datetime(event_data['event_date']) + timedelta(days=event_data['impact_days']),
                clean_period_start=pd.to_datetime(event_data['event_date']) - timedelta(days=60),
                clean_period_end=pd.to_datetime(event_data['event_date']) - timedelta(days=3),
                is_validated=True
            )
            self.events.append(event)
    
    def get_events_in_range(self, start_date: datetime, end_date: datetime) -> List[ShockEvent]:
        """获取时间范围内的事件"""
        return [
            e for e in self.events
            if start_date <= e.event_date <= end_date
        ]
    
    def add_event(self, event: ShockEvent):
        """添加新事件"""
        self.events.append(event)
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        records = []
        for e in self.events:
            records.append({
                'event_id': e.event_id,
                'event_type': e.event_type,
                'event_date': e.event_date,
                'trigger_source': e.trigger_source,
                'confidence_score': e.confidence_score,
                'oil_price_change': e.oil_price_change,
                'is_validated': e.is_validated
            })
        return pd.DataFrame(records)
