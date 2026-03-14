"""
行业横截面排他性验证模块
通过分析行业分化特征，区分外生原油冲击 vs 内生下跌
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass
from scipy import stats


@dataclass
class SectorCrossSectionResult:
    """行业横截面验证结果"""
    is_oil_shock_pattern: bool  # 是否符合原油冲击特征
    confidence: float  # 置信度
    leading_sectors: List[str]  # 领涨板块
    lagging_sectors: List[str]  # 领跌板块
    sector_divergence: float  # 板块分化度
    oil_correlation: float  # 与原油成本敏感度的相关性
    summary: str  # 摘要


class SectorCrossSectionVerifier:
    """
    行业横截面排他性验证器
    
    外生原油冲击的必备行业特征：
    1. 领涨板块明确：原油开采、油服、油气运输、煤炭（替代能源）获得正超额收益
    2. 领跌板块逻辑清晰：航空机场、航运、炼油化工等成本敏感行业显著跑输
    3. 相关性验证：行业涨跌幅与原油成本敏感度的相关系数绝对值≥0.7
    
    内生下跌的反向排除特征：
    - 普跌为主，无原油成本驱动的板块分化
    - 领跌板块为高估值成长股，与原油产业链无关
    """
    
    # 原油相关行业分类
    OIL_BENEFICIARY_SECTORS = [
        '石油开采', '油服工程', '油气运输', '煤炭开采',  # 直接受益
        '油服', '石油加工', '天然气',  # 相关产业
    ]
    
    OIL_COST_SENSITIVE_SECTORS = [
        '航空机场', '航运港口', '物流',  # 运输成本敏感
        '炼油化工', '化学纤维', '塑料',  # 原料成本敏感
        '汽车整车', '交运设备',  # 间接受影响
    ]
    
    # 原油成本敏感度系数（根据历史数据估算）
    OIL_SENSITIVITY_SCORES = {
        '石油开采': 0.9,  # 高度受益
        '油服工程': 0.85,
        '油气运输': 0.7,
        '煤炭开采': 0.6,  # 替代效应
        '石油加工': 0.5,
        '航空机场': -0.8,  # 高度受损
        '航运港口': -0.7,
        '物流': -0.5,
        '炼油化工': -0.6,
        '化学纤维': -0.5,
        '汽车整车': -0.3,
        '电力': -0.2,
        '钢铁': -0.2,
        '水泥': -0.1,
        '银行': 0.0,  # 不敏感
        '房地产': 0.0,
        '计算机': 0.0,
        '医药生物': 0.0,
        '食品饮料': 0.0,
    }
    
    # 验证阈值
    THRESHOLDS = {
        'min_divergence': 0.03,  # 最小板块分化度3%
        'excess_return_std': 2.0,  # 超额收益突破2倍标准差
        'min_correlation': 0.7,  # 最小相关系数
    }
    
    def __init__(self):
        pass
    
    def verify(
        self,
        sector_returns: Dict[str, float],
        benchmark_return: float,
        historical_sector_vol: Dict[str, float] = None
    ) -> SectorCrossSectionResult:
        """
        验证行业横截面特征
        
        Args:
            sector_returns: 各行业收益率 {行业名: 收益率}
            benchmark_return: 基准收益率（如沪深300）
            historical_sector_vol: 各行业历史波动率
        """
        # 1. 计算超额收益
        excess_returns = {
            sector: ret - benchmark_return
            for sector, ret in sector_returns.items()
        }
        
        # 2. 找出领涨和领跌板块
        sorted_sectors = sorted(excess_returns.items(), key=lambda x: x[1], reverse=True)
        leading_sectors = [s[0] for s in sorted_sectors[:5]]  # 领涨前5
        lagging_sectors = [s[0] for s in sorted_sectors[-5:]]  # 领跌后5
        
        # 3. 计算板块分化度
        divergence = self._calculate_divergence(excess_returns)
        
        # 4. 计算与原油敏感度的相关性
        correlation = self._calculate_oil_correlation(excess_returns)
        
        # 5. 验证领涨板块是否为原油受益行业
        beneficiary_match = self._check_beneficiary_match(leading_sectors)
        
        # 6. 验证领跌板块是否为成本敏感行业
        sensitive_match = self._check_sensitive_match(lagging_sectors)
        
        # 7. 综合判断
        is_oil_shock_pattern = self._determine_pattern(
            divergence, correlation, beneficiary_match, sensitive_match
        )
        
        # 计算置信度
        confidence = self._calculate_confidence(
            divergence, correlation, beneficiary_match, sensitive_match
        )
        
        # 生成摘要
        summary = self._generate_summary(
            is_oil_shock_pattern, divergence, correlation,
            leading_sectors[:3], lagging_sectors[:3]
        )
        
        return SectorCrossSectionResult(
            is_oil_shock_pattern=is_oil_shock_pattern,
            confidence=confidence,
            leading_sectors=leading_sectors,
            lagging_sectors=lagging_sectors,
            sector_divergence=divergence,
            oil_correlation=correlation,
            summary=summary
        )
    
    def _calculate_divergence(self, excess_returns: Dict[str, float]) -> float:
        """
        计算板块分化度
        定义：领涨板块平均收益 - 领跌板块平均收益
        """
        if len(excess_returns) < 10:
            return 0.0
        
        sorted_returns = sorted(excess_returns.values(), reverse=True)
        top_avg = np.mean(sorted_returns[:5])  # 前5名平均
        bottom_avg = np.mean(sorted_returns[-5:])  # 后5名平均
        
        divergence = top_avg - bottom_avg
        return divergence
    
    def _calculate_oil_correlation(self, excess_returns: Dict[str, float]) -> float:
        """
        计算行业涨跌幅与原油成本敏感度的相关性
        """
        # 提取有敏感度分数的行业
        common_sectors = set(excess_returns.keys()) & set(self.OIL_SENSITIVITY_SCORES.keys())
        
        if len(common_sectors) < 10:
            return 0.0
        
        returns = []
        sensitivities = []
        
        for sector in common_sectors:
            returns.append(excess_returns[sector])
            sensitivities.append(self.OIL_SENSITIVITY_SCORES[sector])
        
        # 计算相关系数
        correlation, p_value = stats.pearsonr(returns, sensitivities)
        
        return correlation
    
    def _check_beneficiary_match(self, leading_sectors: List[str]) -> float:
        """
        检查领涨板块是否为原油受益行业
        返回匹配比例
        """
        if not leading_sectors:
            return 0.0
        
        match_count = 0
        for sector in leading_sectors:
            for pattern in self.OIL_BENEFICIARY_SECTORS:
                if pattern in sector:
                    match_count += 1
                    break
        
        return match_count / len(leading_sectors)
    
    def _check_sensitive_match(self, lagging_sectors: List[str]) -> float:
        """
        检查领跌板块是否为成本敏感行业
        返回匹配比例
        """
        if not lagging_sectors:
            return 0.0
        
        match_count = 0
        for sector in lagging_sectors:
            for pattern in self.OIL_COST_SENSITIVE_SECTORS:
                if pattern in sector:
                    match_count += 1
                    break
        
        return match_count / len(lagging_sectors)
    
    def _determine_pattern(
        self,
        divergence: float,
        correlation: float,
        beneficiary_match: float,
        sensitive_match: float
    ) -> bool:
        """
        综合判断是否为原油冲击模式
        """
        # 条件1：板块分化度足够大
        condition1 = divergence >= self.THRESHOLDS['min_divergence']
        
        # 条件2：与原油敏感度相关性高
        condition2 = abs(correlation) >= self.THRESHOLDS['min_correlation']
        
        # 条件3：领涨板块匹配原油受益行业
        condition3 = beneficiary_match >= 0.4  # 至少40%匹配
        
        # 条件4：领跌板块匹配成本敏感行业
        condition4 = sensitive_match >= 0.4
        
        # 至少满足3个条件
        return sum([condition1, condition2, condition3, condition4]) >= 3
    
    def _calculate_confidence(
        self,
        divergence: float,
        correlation: float,
        beneficiary_match: float,
        sensitive_match: float
    ) -> float:
        """
        计算置信度
        """
        # 各维度得分
        divergence_score = min(divergence / 0.05, 1.0)  # 分化度得分
        correlation_score = min(abs(correlation), 1.0)  # 相关性得分
        pattern_score = (beneficiary_match + sensitive_match) / 2  # 模式匹配得分
        
        # 加权平均
        confidence = (
            divergence_score * 0.3 +
            correlation_score * 0.4 +
            pattern_score * 0.3
        )
        
        return confidence
    
    def _generate_summary(
        self,
        is_oil_shock: bool,
        divergence: float,
        correlation: float,
        leading: List[str],
        lagging: List[str]
    ) -> str:
        """生成摘要"""
        if is_oil_shock:
            summary = f"✓ 行业横截面验证通过，符合原油冲击特征\n"
        else:
            summary = f"✗ 行业横截面验证未通过，不符合原油冲击特征\n"
        
        summary += f"  - 板块分化度: {divergence*100:.2f}%\n"
        summary += f"  - 原油敏感度相关性: {correlation:.3f}\n"
        summary += f"  - 领涨板块: {', '.join(leading)}\n"
        summary += f"  - 领跌板块: {', '.join(lagging)}"
        
        return summary


class InternalDeclineDetector:
    """
    内生下跌检测器
    用于排除内生悲观预期导致的下跌
    """
    
    # 内生下跌的行业特征
    INTERNAL_DECLINE_PATTERNS = {
        'growth_decline': {
            'leading_decline': ['计算机', '电子', '通信', '传媒', '新能源'],
            'pattern': '成长股领跌，价值股相对抗跌'
        },
        'risk_off': {
            'leading_decline': ['非银金融', '有色金属', '军工', '机械'],
            'pattern': '高风险偏好板块领跌'
        },
        'cyclical_decline': {
            'leading_decline': ['钢铁', '煤炭', '有色金属', '化工'],
            'pattern': '周期股领跌，反映经济下行预期'
        }
    }
    
    def __init__(self):
        pass
    
    def detect(
        self,
        sector_returns: Dict[str, float],
        market_sentiment: Dict[str, float] = None
    ) -> Dict:
        """
        检测是否为内生下跌
        
        Returns:
            检测结果字典
        """
        result = {
            'is_internal_decline': False,
            'decline_type': None,
            'confidence': 0.0,
            'details': {}
        }
        
        # 找出领跌板块
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1])
        leading_decline = [s[0] for s in sorted_sectors[:5]]
        
        # 检查是否匹配内生下跌模式
        best_match = None
        best_match_score = 0
        
        for decline_type, pattern_info in self.INTERNAL_DECLINE_PATTERNS.items():
            expected_decline = pattern_info['leading_decline']
            
            # 计算匹配分数
            match_count = sum(1 for s in leading_decline if any(e in s for e in expected_decline))
            match_score = match_count / len(expected_decline)
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_match = decline_type
        
        # 判断
        if best_match_score >= 0.5:
            result['is_internal_decline'] = True
            result['decline_type'] = best_match
            result['confidence'] = best_match_score
            result['details'] = {
                'leading_decline': leading_decline,
                'pattern_description': self.INTERNAL_DECLINE_PATTERNS[best_match]['pattern']
            }
        
        return result


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试行业横截面验证模块")
    print("=" * 60)
    
    # 模拟原油冲击场景的行业收益
    oil_shock_returns = {
        '石油开采': 0.05,
        '油服工程': 0.04,
        '油气运输': 0.03,
        '煤炭开采': 0.02,
        '航空机场': -0.06,
        '航运港口': -0.05,
        '物流': -0.04,
        '炼油化工': -0.03,
        '银行': -0.01,
        '房地产': -0.015,
        '计算机': -0.01,
        '医药生物': -0.005,
        '食品饮料': 0.005,
        '电力': -0.02,
        '钢铁': -0.01,
    }
    
    verifier = SectorCrossSectionVerifier()
    result = verifier.verify(oil_shock_returns, benchmark_return=-0.02)
    
    print(f"\n{result.summary}")
    print(f"\n置信度: {result.confidence:.2f}")
    print(f"是否为原油冲击模式: {result.is_oil_shock_pattern}")
    
    print("\n" + "=" * 60)
    
    # 测试内生下跌检测
    internal_returns = {
        '计算机': -0.05,
        '电子': -0.04,
        '通信': -0.04,
        '传媒': -0.03,
        '新能源': -0.03,
        '银行': -0.01,
        '石油开采': -0.01,
        '煤炭': -0.015,
        '钢铁': -0.02,
        '食品饮料': 0.005,
    }
    
    detector = InternalDeclineDetector()
    internal_result = detector.detect(internal_returns)
    
    print(f"\n内生下跌检测:")
    print(f"  是否内生下跌: {internal_result['is_internal_decline']}")
    print(f"  下跌类型: {internal_result['decline_type']}")
    print(f"  置信度: {internal_result['confidence']:.2f}")
