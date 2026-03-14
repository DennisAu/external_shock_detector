"""
统计显著性检验与冲击贡献度量化模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression


@dataclass
class ContributionResult:
    """贡献度拆分结果"""
    endogenous_contribution: float  # 内生因素贡献度
    exogenous_contribution: float  # 外生事件贡献度
    random_noise: float  # 随机扰动
    total_change: float  # 总涨跌幅
    is_external_dominated: bool  # 是否外生主导
    is_internal_dominated: bool  # 是否内生主导
    conclusion: str  # 结论描述


class StatisticalSignificanceTester:
    """
    统计显著性检验器
    
    包括：
    1. 残差T检验：检验事件窗口期残差是否显著不同于洁净期
    2. F检验：检验残差波动率是否显著放大
    """
    
    def __init__(self):
        pass
    
    def t_test_residuals(
        self,
        event_residuals: np.ndarray,
        clean_residuals: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        T检验：检验事件窗口期残差是否显著小于洁净期
        
        H0: 事件窗口期残差 = 洁净期残差均值
        H1: 事件窗口期残差 < 洁净期残差均值（单侧检验）
        
        Args:
            event_residuals: 事件窗口期残差
            clean_residuals: 洁净期残差
            alpha: 显著性水平
        """
        # 计算统计量
        event_mean = np.mean(event_residuals)
        clean_mean = np.mean(clean_residuals)
        event_std = np.std(event_residuals)
        clean_std = np.std(clean_residuals)
        
        # 单样本T检验（检验事件残差是否显著偏离0）
        t_stat_1sample, p_value_1sample = stats.ttest_1samp(event_residuals, 0)
        
        # 双样本T检验（检验事件期与洁净期是否有显著差异）
        t_stat_2sample, p_value_2sample = stats.ttest_ind(
            event_residuals, clean_residuals
        )
        
        # 单侧检验p值（事件期残差是否显著小于洁净期）
        if event_mean < clean_mean:
            p_value_one_sided = p_value_2sample / 2
        else:
            p_value_one_sided = 1 - p_value_2sample / 2
        
        return {
            'event_mean': event_mean,
            'clean_mean': clean_mean,
            'event_std': event_std,
            'clean_std': clean_std,
            't_statistic_1sample': t_stat_1sample,
            'p_value_1sample': p_value_1sample,
            't_statistic_2sample': t_stat_2sample,
            'p_value_2sample': p_value_2sample,
            'p_value_one_sided': p_value_one_sided,
            'is_significant': p_value_one_sided < alpha,
            'alpha': alpha
        }
    
    def f_test_volatility(
        self,
        event_residuals: np.ndarray,
        clean_residuals: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        F检验：检验事件窗口期残差波动率是否显著放大
        
        H0: 事件期方差 = 洁净期方差
        H1: 事件期方差 > 洁净期方差
        """
        event_var = np.var(event_residuals, ddof=1)
        clean_var = np.var(clean_residuals, ddof=1)
        
        # F统计量
        f_stat = event_var / clean_var
        
        # 计算p值（单侧检验）
        dfn = len(event_residuals) - 1
        dfd = len(clean_residuals) - 1
        p_value = 1 - stats.f.cdf(f_stat, dfn, dfd)
        
        return {
            'event_variance': event_var,
            'clean_variance': clean_var,
            'f_statistic': f_stat,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'alpha': alpha
        }
    
    def comprehensive_test(
        self,
        event_residuals: np.ndarray,
        clean_residuals: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        综合检验
        """
        t_result = self.t_test_residuals(event_residuals, clean_residuals, alpha)
        f_result = self.f_test_volatility(event_residuals, clean_residuals, alpha)
        
        # 综合判断
        is_significant = t_result['is_significant'] or f_result['is_significant']
        
        return {
            't_test': t_result,
            'f_test': f_result,
            'is_significant': is_significant,
            'summary': self._generate_summary(t_result, f_result, is_significant)
        }
    
    def _generate_summary(self, t_result: Dict, f_result: Dict, is_significant: bool) -> str:
        """生成检验摘要"""
        if is_significant:
            summary = "统计检验通过：事件窗口期残差显著异常\n"
        else:
            summary = "统计检验未通过：事件窗口期残差未显著异常\n"
        
        summary += f"  - T检验: t={t_result['t_statistic_2sample']:.3f}, p={t_result['p_value_one_sided']:.4f}\n"
        summary += f"  - F检验: F={f_result['f_statistic']:.3f}, p={f_result['p_value']:.4f}"
        
        return summary


class ContributionAnalyzer:
    """
    冲击贡献度分析器
    
    将指数累计涨跌幅拆分为三部分：
    累计涨跌幅 = 内生因素贡献 + 目标外生事件贡献 + 随机扰动
    """
    
    def __init__(self):
        pass
    
    def decompose(
        self,
        total_return: float,
        endogenous_fitted_return: float,
        oil_price_change: float,
        vix_change: float,
        residual: float
    ) -> ContributionResult:
        """
        拆分贡献度
        
        Args:
            total_return: 指数累计涨跌幅
            endogenous_fitted_return: 内生模型拟合的累计涨跌幅
            oil_price_change: 原油价格变化
            vix_change: VIX变化
            residual: 累计残差
        """
        # 1. 内生因素贡献 = 内生模型拟合值
        endogenous_contribution = endogenous_fitted_return
        
        # 2. 外生事件贡献 = 用原油和VIX变化对残差回归
        # 简化模型：exogenous = residual * contribution_ratio
        # 其中 contribution_ratio 由原油和VIX变化幅度决定
        
        # 计算外生冲击强度
        exogenous_intensity = self._calculate_exogenous_intensity(
            oil_price_change, vix_change
        )
        
        # 外生贡献 = 残差 * 外生强度占比
        exogenous_contribution = residual * exogenous_intensity
        
        # 3. 随机扰动 = 残差 - 外生贡献
        random_noise = residual - exogenous_contribution
        
        # 4. 归一化贡献度（转换为百分比）
        if abs(total_return) > 0.001:
            endogenous_pct = endogenous_contribution / total_return
            exogenous_pct = exogenous_contribution / total_return
            random_pct = random_noise / total_return
        else:
            endogenous_pct = endogenous_contribution
            exogenous_pct = exogenous_contribution
            random_pct = random_noise
        
        # 5. 判断主导因素
        is_external_dominated = exogenous_pct >= 0.6 and endogenous_pct < 0.2
        is_internal_dominated = endogenous_pct >= 0.7 and exogenous_pct < 0.2
        
        # 6. 生成结论
        conclusion = self._generate_conclusion(
            endogenous_pct, exogenous_pct, is_external_dominated, is_internal_dominated
        )
        
        return ContributionResult(
            endogenous_contribution=endogenous_pct,
            exogenous_contribution=exogenous_pct,
            random_noise=random_pct,
            total_change=total_return,
            is_external_dominated=is_external_dominated,
            is_internal_dominated=is_internal_dominated,
            conclusion=conclusion
        )
    
    def _calculate_exogenous_intensity(
        self,
        oil_change: float,
        vix_change: float
    ) -> float:
        """
        计算外生冲击强度
        
        返回值在 [0, 1] 之间，表示残差中有多少比例可归因于外生事件
        """
        # 原油冲击强度（根据历史数据校准）
        if oil_change > 0.15:  # 暴涨15%以上
            oil_intensity = 0.9
        elif oil_change > 0.10:
            oil_intensity = 0.8
        elif oil_change > 0.06:
            oil_intensity = 0.7
        elif oil_change > 0.03:
            oil_intensity = 0.5
        else:
            oil_intensity = 0.3
        
        # VIX冲击强度
        if vix_change > 0.50:  # 飙升50%以上
            vix_intensity = 0.9
        elif vix_change > 0.35:
            vix_intensity = 0.8
        elif vix_change > 0.20:
            vix_intensity = 0.6
        else:
            vix_intensity = 0.4
        
        # 综合强度（取较高值，因为两者都会影响）
        intensity = max(oil_intensity, vix_intensity)
        
        return min(intensity, 1.0)
    
    def _generate_conclusion(
        self,
        endogenous_pct: float,
        exogenous_pct: float,
        is_external: bool,
        is_internal: bool
    ) -> str:
        """生成结论"""
        if is_external:
            return f"完全判定为外围事件主导的下跌（外生贡献{exogenous_pct*100:.1f}%），非市场自身悲观预期"
        elif is_internal:
            return f"完全判定为内生悲观预期主导的下跌（内生贡献{endogenous_pct*100:.1f}%）"
        else:
            return f"内生下跌与外生冲击共振（内生{endogenous_pct*100:.1f}%，外生{exogenous_pct*100:.1f}%）"


class EventImpactAnalyzer:
    """
    事件影响综合分析器
    整合统计检验和贡献度分析
    """
    
    def __init__(self):
        self.tester = StatisticalSignificanceTester()
        self.analyzer = ContributionAnalyzer()
    
    def analyze(
        self,
        residuals: pd.Series,
        event_idx: int,
        event_window: Tuple[int, int],
        clean_window: Tuple[int, int],
        total_return: float,
        endogenous_fitted: float,
        oil_change: float,
        vix_change: float
    ) -> Dict:
        """
        综合分析事件影响
        
        Args:
            residuals: 残差序列
            event_idx: 事件日索引
            event_window: 事件窗口（相对索引）
            clean_window: 洁净期窗口（相对索引）
            total_return: 窗口期累计收益率
            endogenous_fitted: 内生模型拟合的累计收益
            oil_change: 原油价格变化
            vix_change: VIX变化
        """
        results = {}
        
        # 1. 提取事件期和洁净期残差
        event_residuals = residuals.iloc[event_idx + event_window[0]: event_idx + event_window[1] + 1].values
        clean_residuals = residuals.iloc[event_idx + clean_window[0]: event_idx + clean_window[1] + 1].values
        
        # 2. 统计显著性检验
        stat_result = self.tester.comprehensive_test(event_residuals, clean_residuals)
        results['statistical_test'] = stat_result
        
        # 3. 贡献度拆分
        cumulative_residual = event_residuals.sum()
        contrib_result = self.analyzer.decompose(
            total_return=total_return,
            endogenous_fitted_return=endogenous_fitted,
            oil_price_change=oil_change,
            vix_change=vix_change,
            residual=cumulative_residual
        )
        results['contribution'] = contrib_result
        
        # 4. 综合判定
        results['final_conclusion'] = self._generate_final_conclusion(
            stat_result, contrib_result
        )
        
        return results
    
    def _generate_final_conclusion(
        self,
        stat_result: Dict,
        contrib_result: ContributionResult
    ) -> str:
        """生成最终结论"""
        # 构建结论报告
        conclusion = "=" * 50 + "\n"
        conclusion += "外部冲击识别最终结论\n"
        conclusion += "=" * 50 + "\n\n"
        
        # 统计检验结论
        conclusion += "【统计显著性检验】\n"
        conclusion += stat_result['summary'] + "\n\n"
        
        # 贡献度拆分
        conclusion += "【贡献度拆分】\n"
        conclusion += f"  内生因素贡献: {contrib_result.endogenous_contribution*100:.1f}%\n"
        conclusion += f"  外生事件贡献: {contrib_result.exogenous_contribution*100:.1f}%\n"
        conclusion += f"  随机扰动: {contrib_result.random_noise*100:.1f}%\n\n"
        
        # 最终判定
        conclusion += "【最终判定】\n"
        conclusion += contrib_result.conclusion + "\n"
        
        return conclusion


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试统计显著性检验与贡献度拆分")
    print("=" * 60)
    
    # 模拟数据
    np.random.seed(42)
    
    # 洁净期残差（均值为0）
    clean_residuals = np.random.randn(50) * 0.01
    
    # 事件期残差（负向显著）
    event_residuals = np.random.randn(10) * 0.02 - 0.03  # 负向偏移
    
    # 统计检验
    tester = StatisticalSignificanceTester()
    test_result = tester.comprehensive_test(event_residuals, clean_residuals)
    
    print(f"\n{test_result['summary']}")
    
    # 贡献度拆分
    analyzer = ContributionAnalyzer()
    contrib = analyzer.decompose(
        total_return=-0.08,  # 下跌8%
        endogenous_fitted_return=-0.02,  # 内生模型拟合-2%
        oil_price_change=0.10,  # 原油涨10%
        vix_change=0.30,  # VIX涨30%
        residual=-0.06  # 残差-6%
    )
    
    print(f"\n贡献度拆分:")
    print(f"  内生贡献: {contrib.endogenous_contribution*100:.1f}%")
    print(f"  外生贡献: {contrib.exogenous_contribution*100:.1f}%")
    print(f"  随机扰动: {contrib.random_noise*100:.1f}%")
    print(f"\n{contrib.conclusion}")
