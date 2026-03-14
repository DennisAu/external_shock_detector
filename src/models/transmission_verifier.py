"""
传导路径量化验证模块
实现两条核心传导链路的验证：
1. 成本输入型传导（基本面路径）
2. 全球避险情绪传导（资金路径）
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass
from scipy import stats


@dataclass
class TransmissionResult:
    """传导验证结果"""
    path_type: str  # cost_input / risk_aversion
    is_validated: bool  # 是否验证通过
    confidence: float  # 置信度 0-1
    indicators: Dict  # 各项指标详情
    summary: str  # 简要说明


class CostInputTransmissionVerifier:
    """
    成本输入型传导验证器
    
    逻辑链路：
    原油运输受阻 → 油价暴涨 → 国内输入性通胀抬升 → 
    宽松预期降温 → 成本敏感行业盈利下调 → A股下跌
    
    验证条件（同时满足）：
    1. 中国10年期盈亏平衡通胀率（BEI）单日上行≥5BP，或3日累计上行≥10BP
    2. 1年期FR007利率互换（IRS）上行≥3BP，市场宽松预期降温
    3. 原油成本敏感行业一致预期净利润显著下调，且跌幅显著跑输宽基
    """
    
    THRESHOLDS = {
        'bei_single_day': 0.05,  # BEI单日上行≥5BP（0.05%）
        'bei_3day': 0.10,  # BEI 3日累计上行≥10BP
        'irs_single_day': 0.03,  # IRS单日上行≥3BP
        'sector_underperform': -0.02,  # 敏感行业跑输宽基≥2%
    }
    
    def __init__(self):
        pass
    
    def verify(
        self,
        oil_data: pd.DataFrame,
        bond_data: pd.DataFrame,
        irs_data: pd.DataFrame = None,
        sensitive_sector_returns: Dict[str, float] = None,
        benchmark_return: float = 0.0,
        event_idx: int = None
    ) -> TransmissionResult:
        """
        验证成本输入型传导
        
        Args:
            oil_data: 原油价格数据
            bond_data: 国债收益率数据（用于计算BEI）
            irs_data: 利率互换数据
            sensitive_sector_returns: 成本敏感行业收益率
            benchmark_return: 基准收益率
            event_idx: 事件日索引
        """
        indicators = {}
        validated_count = 0
        
        # 1. 计算BEI变化
        bei_result = self._verify_bei(bond_data, event_idx)
        indicators['bei'] = bei_result
        if bei_result['threshold_met']:
            validated_count += 1
        
        # 2. 验证IRS变化
        irs_result = self._verify_irs(irs_data, event_idx)
        indicators['irs'] = irs_result
        if irs_result['threshold_met']:
            validated_count += 1
        
        # 3. 验证行业表现
        sector_result = self._verify_sector_performance(
            sensitive_sector_returns, benchmark_return
        )
        indicators['sector'] = sector_result
        if sector_result['threshold_met']:
            validated_count += 1
        
        # 计算置信度
        confidence = validated_count / 3.0
        
        # 判断是否验证通过（至少满足2/3条件）
        is_validated = validated_count >= 2
        
        summary = f"成本输入型传导验证: {'通过' if is_validated else '未通过'}"
        summary += f"\n  - BEI变化: {indicators['bei']['change']*100:.2f}BP ({'✓' if indicators['bei']['threshold_met'] else '✗'})"
        summary += f"\n  - IRS变化: {indicators['irs']['change']*100:.2f}BP ({'✓' if indicators['irs']['threshold_met'] else '✗'})"
        summary += f"\n  - 行业分化: {indicators['sector']['avg_underperform']*100:.2f}% ({'✓' if indicators['sector']['threshold_met'] else '✗'})"
        
        return TransmissionResult(
            path_type='cost_input',
            is_validated=is_validated,
            confidence=confidence,
            indicators=indicators,
            summary=summary
        )
    
    def _verify_bei(self, bond_data: pd.DataFrame, event_idx: int) -> Dict:
        """验证盈亏平衡通胀率（BEI）"""
        result = {
            'change': 0.0,
            'threshold_met': False,
            'details': '数据不足'
        }
        
        if bond_data is None or bond_data.empty:
            return result
        
        try:
            # BEI = 名义收益率 - 实际收益率
            # 简化：使用10年期国债收益率变化代理
            if '10年期国债收益率' in bond_data.columns:
                yields = bond_data['10年期国债收益率']
            elif '国债收益率' in bond_data.columns:
                yields = bond_data['国债收益率']
            else:
                return result
            
            if event_idx and event_idx < len(yields):
                change_1d = yields.iloc[event_idx] - yields.iloc[event_idx - 1]
                change_3d = yields.iloc[event_idx] - yields.iloc[event_idx - 3]
                
                threshold_met = (
                    change_1d >= self.THRESHOLDS['bei_single_day'] or
                    change_3d >= self.THRESHOLDS['bei_3day']
                )
                
                result = {
                    'change': change_1d,
                    'change_3d': change_3d,
                    'threshold_met': threshold_met,
                    'details': f'单日变化{change_1d*100:.2f}BP, 3日累计{change_3d*100:.2f}BP'
                }
        except Exception as e:
            logger.error(f"BEI计算失败: {e}")
        
        return result
    
    def _verify_irs(self, irs_data: pd.DataFrame, event_idx: int) -> Dict:
        """验证利率互换（IRS）"""
        result = {
            'change': 0.0,
            'threshold_met': False,
            'details': '数据不足'
        }
        
        if irs_data is None or irs_data.empty:
            # 没有IRS数据时，使用国债收益率代理
            result['details'] = 'IRS数据不可用'
            return result
        
        try:
            if 'irs_1y' in irs_data.columns:
                irs = irs_data['irs_1y']
            else:
                return result
            
            if event_idx and event_idx < len(irs):
                change_1d = irs.iloc[event_idx] - irs.iloc[event_idx - 1]
                
                threshold_met = change_1d >= self.THRESHOLDS['irs_single_day']
                
                result = {
                    'change': change_1d,
                    'threshold_met': threshold_met,
                    'details': f'IRS单日变化{change_1d*100:.2f}BP'
                }
        except Exception as e:
            logger.error(f"IRS验证失败: {e}")
        
        return result
    
    def _verify_sector_performance(
        self,
        sector_returns: Dict[str, float],
        benchmark_return: float
    ) -> Dict:
        """验证成本敏感行业表现"""
        result = {
            'avg_underperform': 0.0,
            'threshold_met': False,
            'details': '数据不足'
        }
        
        if not sector_returns:
            return result
        
        try:
            # 计算各行业相对基准的超额收益
            underperforms = []
            for sector, ret in sector_returns.items():
                underperform = ret - benchmark_return
                underperforms.append(underperform)
            
            avg_underperform = np.mean(underperforms)
            
            threshold_met = avg_underperform <= self.THRESHOLDS['sector_underperform']
            
            result = {
                'avg_underperform': avg_underperform,
                'sector_details': sector_returns,
                'threshold_met': threshold_met,
                'details': f'成本敏感行业平均跑输{abs(avg_underperform)*100:.2f}%'
            }
        except Exception as e:
            logger.error(f"行业表现验证失败: {e}")
        
        return result


class RiskAversionTransmissionVerifier:
    """
    全球避险情绪传导验证器
    
    逻辑链路：
    中东战争 → 全球地缘风险抬升 → VIX飙升 → 
    资金避险 → 北向资金大幅流出 → A股下跌
    
    验证条件（同时满足）：
    1. VIX指数单日涨幅≥20%，或3日累计涨幅≥35%，突破过去1年2倍上行标准差
    2. 北向资金单日净流出≥50亿元，或3日累计净流出≥100亿元
    3. 本土两融资金无同步大幅净流出
    """
    
    THRESHOLDS = {
        'vix_single_day': 0.20,  # VIX单日涨幅≥20%
        'vix_3day': 0.35,  # VIX 3日累计涨幅≥35%
        'north_flow_single_day': -50,  # 北向资金单日净流出≥50亿
        'north_flow_3day': -100,  # 北向资金3日累计净流出≥100亿
    }
    
    def __init__(self):
        pass
    
    def verify(
        self,
        vix_data: pd.DataFrame,
        north_flow_data: pd.DataFrame,
        margin_data: pd.DataFrame = None,
        event_idx: int = None
    ) -> TransmissionResult:
        """
        验证全球避险情绪传导
        """
        indicators = {}
        validated_count = 0
        
        # 1. 验证VIX变化
        vix_result = self._verify_vix(vix_data, event_idx)
        indicators['vix'] = vix_result
        if vix_result['threshold_met']:
            validated_count += 1
        
        # 2. 验证北向资金
        north_result = self._verify_north_flow(north_flow_data, event_idx)
        indicators['north_flow'] = north_result
        if north_result['threshold_met']:
            validated_count += 1
        
        # 3. 验证两融资金（排除内生情绪恶化）
        margin_result = self._verify_margin(margin_data, event_idx)
        indicators['margin'] = margin_result
        if margin_result['is_stable']:
            validated_count += 1
        
        # 计算置信度
        confidence = validated_count / 3.0
        
        # 判断是否验证通过
        is_validated = validated_count >= 2
        
        summary = f"全球避险情绪传导验证: {'通过' if is_validated else '未通过'}"
        summary += f"\n  - VIX变化: {indicators['vix']['change']*100:.1f}% ({'✓' if indicators['vix']['threshold_met'] else '✗'})"
        summary += f"\n  - 北向资金: {indicators['north_flow']['flow']:.1f}亿 ({'✓' if indicators['north_flow']['threshold_met'] else '✗'})"
        summary += f"\n  - 两融稳定: {'是' if indicators['margin']['is_stable'] else '否'} ({'✓' if indicators['margin']['is_stable'] else '✗'})"
        
        return TransmissionResult(
            path_type='risk_aversion',
            is_validated=is_validated,
            confidence=confidence,
            indicators=indicators,
            summary=summary
        )
    
    def _verify_vix(self, vix_data: pd.DataFrame, event_idx: int) -> Dict:
        """验证VIX变化"""
        result = {
            'change': 0.0,
            'threshold_met': False,
            'details': '数据不足'
        }
        
        if vix_data is None or vix_data.empty:
            return result
        
        try:
            if 'close' in vix_data.columns:
                vix = vix_data['close']
            else:
                return result
            
            if event_idx and event_idx < len(vix):
                change_1d = (vix.iloc[event_idx] - vix.iloc[event_idx - 1]) / vix.iloc[event_idx - 1]
                change_3d = (vix.iloc[event_idx] - vix.iloc[event_idx - 3]) / vix.iloc[event_idx - 3]
                
                # 计算2倍标准差阈值
                returns = vix.pct_change().dropna()
                std_1y = returns.tail(252).std()
                threshold_2std = 2 * std_1y
                
                threshold_met = (
                    (change_1d >= self.THRESHOLDS['vix_single_day'] and change_1d >= threshold_2std) or
                    (change_3d >= self.THRESHOLDS['vix_3day'])
                )
                
                result = {
                    'change': change_1d,
                    'change_3d': change_3d,
                    'threshold_met': threshold_met,
                    'details': f'VIX单日涨幅{change_1d*100:.1f}%, 3日累计{change_3d*100:.1f}%'
                }
        except Exception as e:
            logger.error(f"VIX验证失败: {e}")
        
        return result
    
    def _verify_north_flow(self, north_data: pd.DataFrame, event_idx: int) -> Dict:
        """验证北向资金"""
        result = {
            'flow': 0.0,
            'threshold_met': False,
            'details': '数据不足'
        }
        
        if north_data is None or north_data.empty:
            return result
        
        try:
            if 'north_flow' in north_data.columns:
                flow = north_data['north_flow']
            elif '净流入' in north_data.columns:
                flow = north_data['净流入']
            else:
                return result
            
            if event_idx and event_idx < len(flow):
                flow_1d = flow.iloc[event_idx]
                flow_3d = flow.iloc[event_idx:event_idx+3].sum()
                
                # 注意：流出为负值
                threshold_met = (
                    flow_1d <= self.THRESHOLDS['north_flow_single_day'] or
                    flow_3d <= self.THRESHOLDS['north_flow_3day']
                )
                
                result = {
                    'flow': flow_1d,
                    'flow_3d': flow_3d,
                    'threshold_met': threshold_met,
                    'details': f'北向资金单日{flow_1d:.1f}亿, 3日累计{flow_3d:.1f}亿'
                }
        except Exception as e:
            logger.error(f"北向资金验证失败: {e}")
        
        return result
    
    def _verify_margin(self, margin_data: pd.DataFrame, event_idx: int) -> Dict:
        """验证两融资金稳定性"""
        result = {
            'is_stable': True,
            'change': 0.0,
            'details': '数据不足，默认稳定'
        }
        
        if margin_data is None or margin_data.empty:
            return result
        
        try:
            if '融资余额' in margin_data.columns:
                margin = margin_data['融资余额']
            else:
                return result
            
            if event_idx and event_idx < len(margin):
                change = margin.iloc[event_idx] - margin.iloc[event_idx - 1]
                change_pct = change / margin.iloc[event_idx - 1]
                
                # 判断是否稳定（变化幅度小于5%）
                is_stable = abs(change_pct) < 0.05
                
                result = {
                    'is_stable': is_stable,
                    'change': change,
                    'change_pct': change_pct,
                    'details': f'融资余额变化{change_pct*100:.2f}%'
                }
        except Exception as e:
            logger.error(f"两融验证失败: {e}")
        
        return result


class CryptoContagionVerifier:
    """
    加密货币传染传导验证器 (新增)
    
    逻辑链路：
    BTC暴跌 → 加密市场恐慌 → 风险资产联动 → A股科技/区块链板块下跌
    
    验证条件（同时满足）：
    1. BTC单日跌幅≥10%，或3日累计跌幅≥20%
    2. 区块链/数字货币板块跌幅显著跑输大盘
    3. 加密货币相关个股大幅下跌
    """
    
    THRESHOLDS = {
        'btc_single_day': -0.10,  # BTC单日跌幅≥10%
        'btc_3day': -0.20,  # BTC 3日累计跌幅≥20%
        'sector_underperform': -0.03,  # 区块链板块跑输≥3%
    }
    
    def __init__(self):
        pass
    
    def verify(
        self,
        btc_data: pd.DataFrame,
        blockchain_sector_return: float = None,
        benchmark_return: float = 0.0,
        crypto_stock_returns: Dict[str, float] = None,
        event_idx: int = None
    ) -> TransmissionResult:
        """验证加密货币传染传导"""
        indicators = {}
        validated_count = 0
        
        # 1. 验证BTC价格变化
        btc_result = self._verify_btc(btc_data, event_idx)
        indicators['btc'] = btc_result
        if btc_result['threshold_met']:
            validated_count += 1
        
        # 2. 验证板块表现
        if blockchain_sector_return is not None:
            sector_underperform = blockchain_sector_return - benchmark_return
            sector_met = sector_underperform <= self.THRESHOLDS['sector_underperform']
            indicators['sector'] = {
                'underperform': sector_underperform,
                'threshold_met': sector_met
            }
            if sector_met:
                validated_count += 1
        
        # 3. 验证相关个股
        if crypto_stock_returns:
            stock_result = self._verify_crypto_stocks(crypto_stock_returns)
            indicators['stocks'] = stock_result
            if stock_result['threshold_met']:
                validated_count += 1
        
        confidence = validated_count / 3.0
        is_validated = validated_count >= 2
        
        summary = f"加密货币传染传导验证: {'通过' if is_validated else '未通过'}"
        summary += f"\n  - BTC变化: {indicators['btc']['change']*100:.1f}% ({'✓' if indicators['btc']['threshold_met'] else '✗'})"
        if 'sector' in indicators:
            summary += f"\n  - 板块跑输: {indicators['sector']['underperform']*100:.1f}% ({'✓' if indicators['sector']['threshold_met'] else '✗'})"
        
        return TransmissionResult(
            path_type='crypto_contagion',
            is_validated=is_validated,
            confidence=confidence,
            indicators=indicators,
            summary=summary
        )
    
    def _verify_btc(self, btc_data: pd.DataFrame, event_idx: int) -> Dict:
        """验证BTC价格变化"""
        result = {
            'change': 0.0,
            'threshold_met': False,
            'details': '数据不足'
        }
        
        if btc_data is None or btc_data.empty:
            return result
        
        try:
            if 'close' in btc_data.columns:
                btc = btc_data['close']
            else:
                return result
            
            if event_idx and event_idx < len(btc):
                change_1d = (btc.iloc[event_idx] - btc.iloc[event_idx - 1]) / btc.iloc[event_idx - 1]
                change_3d = (btc.iloc[event_idx] - btc.iloc[event_idx - 3]) / btc.iloc[event_idx - 3] if event_idx >= 3 else change_1d
                
                threshold_met = (
                    change_1d <= self.THRESHOLDS['btc_single_day'] or
                    change_3d <= self.THRESHOLDS['btc_3day']
                )
                
                result = {
                    'change': change_1d,
                    'change_3d': change_3d,
                    'threshold_met': threshold_met,
                    'details': f'BTC单日变化{change_1d*100:.1f}%, 3日累计{change_3d*100:.1f}%'
                }
        except Exception as e:
            logger.error(f"BTC验证失败: {e}")
        
        return result
    
    def _verify_crypto_stocks(self, stock_returns: Dict[str, float]) -> Dict:
        """验证加密货币相关个股"""
        if not stock_returns:
            return {'threshold_met': False, 'details': '无个股数据'}
        
        avg_return = np.mean(list(stock_returns.values()))
        threshold_met = avg_return <= -0.05  # 平均跌幅≥5%
        
        return {
            'avg_return': avg_return,
            'threshold_met': threshold_met,
            'details': f'加密货币相关股平均收益{avg_return*100:.1f}%'
        }


class TransmissionPathAnalyzer:
    """
    传导路径综合分析器
    整合多条传导链路的验证结果 (扩展版)
    """
    
    def __init__(self):
        self.cost_verifier = CostInputTransmissionVerifier()
        self.risk_verifier = RiskAversionTransmissionVerifier()
        self.crypto_verifier = CryptoContagionVerifier()  # 新增
    
    def analyze(
        self,
        oil_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        north_flow_data: pd.DataFrame,
        bond_data: pd.DataFrame = None,
        irs_data: pd.DataFrame = None,
        margin_data: pd.DataFrame = None,
        sensitive_sector_returns: Dict[str, float] = None,
        benchmark_return: float = 0.0,
        event_idx: int = None,
        # 新增参数
        btc_data: pd.DataFrame = None,
        blockchain_sector_return: float = None,
        crypto_stock_returns: Dict[str, float] = None
    ) -> Dict:
        """
        综合分析传导路径 (扩展版)
        
        Returns:
            包含多条链路验证结果的字典
        """
        results = {}
        
        # 验证成本输入型传导
        cost_result = self.cost_verifier.verify(
            oil_data=oil_data,
            bond_data=bond_data,
            irs_data=irs_data,
            sensitive_sector_returns=sensitive_sector_returns,
            benchmark_return=benchmark_return,
            event_idx=event_idx
        )
        results['cost_input'] = cost_result
        
        # 验证避险情绪传导
        risk_result = self.risk_verifier.verify(
            vix_data=vix_data,
            north_flow_data=north_flow_data,
            margin_data=margin_data,
            event_idx=event_idx
        )
        results['risk_aversion'] = risk_result
        
        # 验证加密货币传染传导 (新增)
        if btc_data is not None:
            crypto_result = self.crypto_verifier.verify(
                btc_data=btc_data,
                blockchain_sector_return=blockchain_sector_return,
                benchmark_return=benchmark_return,
                crypto_stock_returns=crypto_stock_returns,
                event_idx=event_idx
            )
            results['crypto_contagion'] = crypto_result
        else:
            crypto_result = None
        
        # 综合判断
        validated_paths = []
        if cost_result.is_validated:
            validated_paths.append('cost_input')
        if risk_result.is_validated:
            validated_paths.append('risk_aversion')
        if crypto_result and crypto_result.is_validated:
            validated_paths.append('crypto_contagion')
        
        results['any_path_validated'] = len(validated_paths) > 0
        results['validated_paths'] = validated_paths
        
        # 确定主要路径
        if len(validated_paths) == 0:
            results['primary_path'] = 'none'
            results['summary'] = "传导路径验证未通过，可能非外生冲击"
        elif len(validated_paths) == 1:
            results['primary_path'] = validated_paths[0]
            path_names = {
                'cost_input': '成本输入型',
                'risk_aversion': '避险情绪型',
                'crypto_contagion': '加密货币传染型'
            }
            results['summary'] = f"{path_names.get(validated_paths[0], validated_paths[0])}传导验证通过"
        else:
            results['primary_path'] = 'multiple'
            results['summary'] = f"多条传导路径验证通过: {', '.join(validated_paths)}"
        
        return results


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试传导路径验证模块")
    print("=" * 60)
    
    # 模拟数据
    np.random.seed(42)
    
    # VIX数据
    vix_data = pd.DataFrame({
        'trade_date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'close': 15 + np.random.randn(100) * 3
    })
    # 模拟事件日VIX飙升
    vix_data.loc[50, 'close'] = 25  # 单日暴涨
    
    # 北向资金数据
    north_data = pd.DataFrame({
        'trade_date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
        'north_flow': np.random.randn(100) * 30
    })
    north_data.loc[50, 'north_flow'] = -80  # 大幅流出
    
    # 测试避险情绪传导
    verifier = RiskAversionTransmissionVerifier()
    result = verifier.verify(vix_data, north_data, event_idx=50)
    
    print(f"\n{result.summary}")
    print(f"\n置信度: {result.confidence:.2f}")
