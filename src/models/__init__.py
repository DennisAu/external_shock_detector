"""
核心模型模块
"""
from .endogenous_model import (
    EndogenousBenchmarkModel,
    EndogenousFactorBuilder,
    ResidualAnalyzer,
    ModelResult
)
from .transmission_verifier import (
    TransmissionPathAnalyzer,
    CostInputTransmissionVerifier,
    RiskAversionTransmissionVerifier,
    TransmissionResult
)
from .sector_crosssection_verifier import (
    SectorCrossSectionVerifier,
    InternalDeclineDetector,
    SectorCrossSectionResult
)
from .significance_tester import (
    StatisticalSignificanceTester,
    ContributionAnalyzer,
    EventImpactAnalyzer,
    ContributionResult
)
from .shock_detector_system import (
    ExternalShockDetector,
    ShockDetectionResult
)

__all__ = [
    # 内生模型
    'EndogenousBenchmarkModel',
    'EndogenousFactorBuilder',
    'ResidualAnalyzer',
    'ModelResult',
    
    # 传导验证
    'TransmissionPathAnalyzer',
    'CostInputTransmissionVerifier',
    'RiskAversionTransmissionVerifier',
    'TransmissionResult',
    
    # 行业验证
    'SectorCrossSectionVerifier',
    'InternalDeclineDetector',
    'SectorCrossSectionResult',
    
    # 统计检验
    'StatisticalSignificanceTester',
    'ContributionAnalyzer',
    'EventImpactAnalyzer',
    'ContributionResult',
    
    # 系统主控
    'ExternalShockDetector',
    'ShockDetectionResult'
]
