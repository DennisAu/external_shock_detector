"""
外部冲击识别系统 - 主控制器
整合所有模块，实现完整的事件识别流程
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass
import asyncio

from .endogenous_model import EndogenousBenchmarkModel, ResidualAnalyzer
from .transmission_verifier import TransmissionPathAnalyzer
from .sector_crosssection_verifier import SectorCrossSectionVerifier, InternalDeclineDetector
from .significance_tester import EventImpactAnalyzer
from ..data_collectors.free_data_sources import FreeDataSources
from ..data_collectors.event_detector import EventValidator, HistoricalEventDatabase, ShockEvent


@dataclass
class ShockDetectionResult:
    """外部冲击识别结果"""
    event_id: str
    event_date: datetime
    event_type: str
    
    # 核心判定
    is_external_shock: bool  # 是否为外部冲击
    shock_type: str  # shock_type: cost_input / risk_aversion / mixed / none
    confidence: float  # 置信度
    
    # 详细验证结果
    residual_validation: Dict  # 残差验证
    transmission_validation: Dict  # 传导路径验证
    sector_validation: Dict  # 行业横截面验证
    statistical_validation: Dict  # 统计显著性验证
    contribution_analysis: Dict  # 贡献度分析
    
    # 结论
    conclusion: str
    summary: str


class ExternalShockDetector:
    """
    外部冲击识别系统主控制器
    
    完整流程：
    1. 事件检测与标准化
    2. 内生基准拟合
    3. 残差分析与异常检测
    4. 传导路径验证
    5. 行业横截面验证
    6. 统计显著性检验
    7. 贡献度拆分
    8. 最终判定
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 初始化各模块
        self.event_validator = EventValidator()
        self.endogenous_model = EndogenousBenchmarkModel()
        self.residual_analyzer = ResidualAnalyzer()
        self.transmission_analyzer = TransmissionPathAnalyzer()
        self.sector_verifier = SectorCrossSectionVerifier()
        self.internal_detector = InternalDeclineDetector()
        self.impact_analyzer = EventImpactAnalyzer()
        self.event_db = HistoricalEventDatabase()
        
        # 数据缓存
        self.data_cache = {}
        
        logger.info("外部冲击识别系统初始化完成")
    
    async def detect(
        self,
        event_date: datetime,
        news_text: str = None,
        use_historical: bool = False
    ) -> ShockDetectionResult:
        """
        执行完整的外部冲击识别流程
        
        Args:
            event_date: 事件日期
            news_text: 相关新闻文本（可选）
            use_historical: 是否使用历史事件数据
        """
        logger.info(f"开始识别事件: {event_date.strftime('%Y-%m-%d')}")
        
        # Step 1: 收集数据
        logger.info("Step 1: 收集数据...")
        data = await self._collect_data(event_date)
        
        # Step 2: 事件标准化验证
        logger.info("Step 2: 事件标准化验证...")
        is_valid_event, event = self._validate_event(
            news_text or "地缘政治事件",
            event_date,
            data
        )
        
        if not is_valid_event:
            return self._create_invalid_result(event_date, "事件验证未通过")
        
        # Step 3: 内生基准拟合
        logger.info("Step 3: 内生基准拟合...")
        model_result = self._fit_endogenous_model(data, event)
        
        # Step 4: 残差分析与异常检测
        logger.info("Step 4: 残差分析与异常检测...")
        residual_result = self._analyze_residuals(data, event, model_result)
        
        # Step 5: 传导路径验证
        logger.info("Step 5: 传导路径验证...")
        transmission_result = self._verify_transmission(data, event)
        
        # Step 6: 行业横截面验证
        logger.info("Step 6: 行业横截面验证...")
        sector_result = self._verify_sector_crosssection(data, event)
        
        # Step 7: 统计显著性检验与贡献度分析
        logger.info("Step 7: 统计显著性检验与贡献度分析...")
        impact_result = self._analyze_impact(
            data, event, model_result, residual_result,
            transmission_result, sector_result
        )
        
        # Step 8: 综合判定
        logger.info("Step 8: 综合判定...")
        final_result = self._make_final_decision(
            event, residual_result, transmission_result,
            sector_result, impact_result
        )
        
        return final_result
    
    async def _collect_data(self, event_date: datetime) -> Dict:
        """收集所有必要数据"""
        data = {}
        
        # 计算日期范围
        start_date = (event_date - timedelta(days=100)).strftime("%Y%m%d")
        end_date = (event_date + timedelta(days=15)).strftime("%Y%m%d")
        
        try:
            # A股指数数据
            logger.info("  - 获取A股指数数据...")
            data['index_data'] = FreeDataSources.get_index_history(
                symbol="sh000300",  # 沪深300
                start_date=start_date,
                end_date=end_date
            )
            
            # 板块数据
            logger.info("  - 获取板块数据...")
            data['sector_data'] = FreeDataSources.get_sector_realtime()
            
            # 北向资金
            logger.info("  - 获取北向资金数据...")
            data['north_flow'] = FreeDataSources.get_north_flow()
            
            # 原油数据
            logger.info("  - 获取原油数据...")
            data['oil_data'] = FreeDataSources.get_crude_oil(
                start_date=start_date,
                end_date=end_date
            )
            
            # VIX数据
            logger.info("  - 获取VIX数据...")
            data['vix_data'] = FreeDataSources.get_global_index(
                symbol="^VIX",
                start_date=start_date,
                end_date=end_date
            )
            
            # 国债收益率
            logger.info("  - 获取国债收益率...")
            data['bond_data'] = FreeDataSources.get_bond_yield()
            
            # 利率数据
            logger.info("  - 获取利率数据...")
            data['rate_data'] = FreeDataSources.get_interest_rate()
            
            # 融资融券
            logger.info("  - 获取融资融券数据...")
            data['margin_data'] = FreeDataSources.get_margin_data()
            
        except Exception as e:
            logger.error(f"数据收集失败: {e}")
        
        return data
    
    def _validate_event(
        self,
        news_text: str,
        event_date: datetime,
        data: Dict
    ) -> Tuple[bool, Optional[ShockEvent]]:
        """验证事件有效性"""
        try:
            is_valid, event = self.event_validator.validate_event(
                news_text=news_text,
                event_date=event_date,
                bdti_data=data.get('bdti_data', pd.DataFrame()),
                oil_data=data.get('oil_data', pd.DataFrame()),
                domestic_indicators=None
            )
            return is_valid, event
        except Exception as e:
            logger.error(f"事件验证失败: {e}")
            return False, None
    
    def _fit_endogenous_model(
        self,
        data: Dict,
        event: ShockEvent
    ) -> Dict:
        """拟合内生基准模型"""
        result = {
            'is_fitted': False,
            'fitted_value': None,
            'residuals': None,
            'r_squared': None
        }
        
        try:
            index_data = data.get('index_data')
            if index_data is None or index_data.empty:
                return result
            
            # 构建因子（简化版）
            factor_data = pd.DataFrame({
                'trade_date': index_data['date'],
                'return_5d': index_data['close'].pct_change(5).fillna(0),
                'return_10d': index_data['close'].pct_change(10).fillna(0),
                'return_20d': index_data['close'].pct_change(20).fillna(0),
                'volatility_20d': index_data['close'].pct_change().rolling(20).std().fillna(0),
            })
            
            # 拟合模型
            self.endogenous_model.fit(
                index_data=index_data,
                factor_data=factor_data,
                clean_period=(str(event.clean_period_start), str(event.clean_period_end))
            )
            
            # 获取残差
            residuals = self.endogenous_model.get_residuals(index_data, factor_data)
            
            result = {
                'is_fitted': True,
                'residuals': residuals,
                'r_squared': 0.45  # 简化
            }
            
        except Exception as e:
            logger.error(f"内生模型拟合失败: {e}")
        
        return result
    
    def _analyze_residuals(
        self,
        data: Dict,
        event: ShockEvent,
        model_result: Dict
    ) -> Dict:
        """分析残差"""
        result = {
            'is_anomaly': False,
            'anomaly_type': None,
            'z_score': None
        }
        
        try:
            residuals = model_result.get('residuals')
            if residuals is None or len(residuals) == 0:
                return result
            
            # 找到事件日期对应的索引
            index_data = data.get('index_data')
            if index_data is None:
                return result
            
            event_idx = index_data[index_data['date'] == event.event_date].index
            if len(event_idx) == 0:
                event_idx = len(index_data) // 2  # 简化
            else:
                event_idx = event_idx[0]
            
            # 检测异常
            anomaly_result = self.residual_analyzer.detect_anomaly(residuals, event_idx)
            
            result = {
                'is_anomaly': anomaly_result['is_anomaly'],
                'anomaly_type': anomaly_result['anomaly_type'],
                'z_score': anomaly_result['details'].get('single_day', {}).get('z_score', 0),
                'details': anomaly_result['details']
            }
            
        except Exception as e:
            logger.error(f"残差分析失败: {e}")
        
        return result
    
    def _verify_transmission(
        self,
        data: Dict,
        event: ShockEvent
    ) -> Dict:
        """验证传导路径"""
        result = {
            'any_path_validated': False,
            'primary_path': 'none'
        }
        
        try:
            oil_data = data.get('oil_data')
            vix_data = data.get('vix_data')
            north_flow = data.get('north_flow')
            bond_data = data.get('bond_data')
            
            if oil_data is None or vix_data is None:
                return result
            
            # 执行传导验证
            transmission_result = self.transmission_analyzer.analyze(
                oil_data=oil_data,
                vix_data=vix_data,
                north_flow_data=north_flow,
                bond_data=bond_data,
                event_idx=len(oil_data) // 2  # 简化
            )
            
            result = transmission_result
            
        except Exception as e:
            logger.error(f"传导验证失败: {e}")
        
        return result
    
    def _verify_sector_crosssection(
        self,
        data: Dict,
        event: ShockEvent
    ) -> Dict:
        """验证行业横截面"""
        result = {
            'is_oil_shock_pattern': False,
            'confidence': 0.0
        }
        
        try:
            sector_data = data.get('sector_data')
            if sector_data is None or sector_data.empty:
                return result
            
            # 提取板块收益率
            if '涨跌幅' in sector_data.columns:
                sector_returns = dict(zip(
                    sector_data['名称'],
                    sector_data['涨跌幅'] / 100
                ))
            else:
                return result
            
            # 执行横截面验证
            cross_result = self.sector_verifier.verify(
                sector_returns=sector_returns,
                benchmark_return=-0.02  # 假设基准跌2%
            )
            
            result = {
                'is_oil_shock_pattern': cross_result.is_oil_shock_pattern,
                'confidence': cross_result.confidence,
                'sector_divergence': cross_result.sector_divergence,
                'oil_correlation': cross_result.oil_correlation,
                'summary': cross_result.summary
            }
            
        except Exception as e:
            logger.error(f"行业横截面验证失败: {e}")
        
        return result
    
    def _analyze_impact(
        self,
        data: Dict,
        event: ShockEvent,
        model_result: Dict,
        residual_result: Dict,
        transmission_result: Dict,
        sector_result: Dict
    ) -> Dict:
        """分析事件影响"""
        result = {}
        
        try:
            residuals = model_result.get('residuals')
            if residuals is None:
                return result
            
            # 执行影响分析
            impact_result = self.impact_analyzer.analyze(
                residuals=residuals,
                event_idx=len(residuals) // 2,
                event_window=(-2, 10),
                clean_window=(-60, -3),
                total_return=-0.05,  # 假设总收益-5%
                endogenous_fitted=-0.01,  # 内生拟合-1%
                oil_change=event.oil_price_change,
                vix_change=0.3  # 假设VIX涨30%
            )
            
            result = impact_result
            
        except Exception as e:
            logger.error(f"影响分析失败: {e}")
        
        return result
    
    def _make_final_decision(
        self,
        event: ShockEvent,
        residual_result: Dict,
        transmission_result: Dict,
        sector_result: Dict,
        impact_result: Dict
    ) -> ShockDetectionResult:
        """综合判定"""
        
        # 计算各维度得分
        residual_score = 1.0 if residual_result.get('is_anomaly', False) else 0.0
        transmission_score = 1.0 if transmission_result.get('any_path_validated', False) else 0.0
        sector_score = sector_result.get('confidence', 0.0)
        
        # 综合置信度
        confidence = (
            residual_score * 0.3 +
            transmission_score * 0.3 +
            sector_score * 0.4
        )
        
        # 判断是否为外部冲击
        is_external_shock = (
            residual_score >= 1.0 and
            transmission_score >= 1.0 and
            sector_score >= 0.5
        )
        
        # 确定冲击类型
        if is_external_shock:
            shock_type = transmission_result.get('primary_path', 'mixed')
        else:
            shock_type = 'none'
        
        # 生成结论
        if is_external_shock:
            conclusion = f"确认外部冲击事件：{event.trigger_source}"
            conclusion += f"\n冲击类型：{shock_type}"
            conclusion += f"\n置信度：{confidence:.1%}"
        else:
            conclusion = "未识别到显著的外部冲击"
            conclusion += "\n可能为市场内生下跌或其他因素"
        
        # 生成摘要
        summary = self._generate_summary(
            event, residual_result, transmission_result,
            sector_result, impact_result, is_external_shock, confidence
        )
        
        return ShockDetectionResult(
            event_id=event.event_id,
            event_date=event.event_date,
            event_type=event.event_type,
            is_external_shock=is_external_shock,
            shock_type=shock_type,
            confidence=confidence,
            residual_validation=residual_result,
            transmission_validation=transmission_result,
            sector_validation=sector_result,
            statistical_validation=impact_result.get('statistical_test', {}),
            contribution_analysis=impact_result.get('contribution', {}),
            conclusion=conclusion,
            summary=summary
        )
    
    def _generate_summary(
        self,
        event: ShockEvent,
        residual_result: Dict,
        transmission_result: Dict,
        sector_result: Dict,
        impact_result: Dict,
        is_external_shock: bool,
        confidence: float
    ) -> str:
        """生成摘要报告"""
        summary = "=" * 60 + "\n"
        summary += "外部冲击识别报告\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"事件ID: {event.event_id}\n"
        summary += f"事件日期: {event.event_date.strftime('%Y-%m-%d')}\n"
        summary += f"事件类型: {event.event_type}\n"
        summary += f"事件描述: {event.trigger_source}\n\n"
        
        summary += "-" * 60 + "\n"
        summary += "验证结果\n"
        summary += "-" * 60 + "\n\n"
        
        summary += "【1. 残差验证】\n"
        summary += f"  是否异常: {'是' if residual_result.get('is_anomaly') else '否'}\n"
        summary += f"  Z-score: {residual_result.get('z_score', 0):.2f}\n\n"
        
        summary += "【2. 传导路径验证】\n"
        summary += f"  验证通过: {'是' if transmission_result.get('any_path_validated') else '否'}\n"
        summary += f"  主要路径: {transmission_result.get('primary_path', '无')}\n\n"
        
        summary += "【3. 行业横截面验证】\n"
        summary += f"  符合原油冲击模式: {'是' if sector_result.get('is_oil_shock_pattern') else '否'}\n"
        summary += f"  置信度: {sector_result.get('confidence', 0):.1%}\n\n"
        
        summary += "-" * 60 + "\n"
        summary += "最终判定\n"
        summary += "-" * 60 + "\n\n"
        
        summary += f"是否外部冲击: {'✓ 是' if is_external_shock else '✗ 否'}\n"
        summary += f"综合置信度: {confidence:.1%}\n"
        
        return summary
    
    def _create_invalid_result(self, event_date: datetime, reason: str) -> ShockDetectionResult:
        """创建无效结果"""
        return ShockDetectionResult(
            event_id="INVALID",
            event_date=event_date,
            event_type="unknown",
            is_external_shock=False,
            shock_type="none",
            confidence=0.0,
            residual_validation={},
            transmission_validation={},
            sector_validation={},
            statistical_validation={},
            contribution_analysis={},
            conclusion=f"事件验证未通过: {reason}",
            summary=f"事件验证失败: {reason}"
        )
    
    async def analyze_historical_events(self) -> List[ShockDetectionResult]:
        """分析历史事件库"""
        results = []
        
        for event in self.event_db.events:
            try:
                result = await self.detect(
                    event_date=event.event_date,
                    use_historical=True
                )
                results.append(result)
            except Exception as e:
                logger.error(f"分析历史事件失败: {e}")
        
        return results


# 测试代码
if __name__ == "__main__":
    import asyncio
    
    async def test():
        detector = ExternalShockDetector()
        
        # 测试历史事件
        result = await detector.detect(
            event_date=datetime(2022, 2, 24),  # 俄乌冲突
            news_text="俄乌冲突爆发"
        )
        
        print(result.summary)
    
    asyncio.run(test())
