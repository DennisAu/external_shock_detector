"""
内生基准拟合模型
核心算法：滚动窗口多元线性回归
目标：拟合A股内生走势，计算残差用于识别外生冲击
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """模型结果数据类"""
    fitted_value: float  # 拟合值
    actual_value: float  # 实际值
    residual: float  # 残差
    residual_std: float  # 残差标准差
    z_score: float  # 残差Z-score
    r_squared: float  # 模型R²
    is_significant: bool  # 是否显著异常
    contribution: Dict[str, float]  # 各因子贡献度


class EndogenousFactorBuilder:
    """
    内生因子构建器
    构建纯A股内生驱动因子
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def build_factors(
        self,
        index_data: pd.DataFrame,
        margin_data: pd.DataFrame = None,
        bond_data: pd.DataFrame = None,
        rate_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        构建内生因子
        
        Args:
            index_data: 指数数据（用于计算市场情绪）
            margin_data: 融资融券数据
            bond_data: 国债收益率数据
            rate_data: 银行间利率数据
        
        Returns:
            包含所有内生因子的DataFrame
        """
        factors = pd.DataFrame()
        factors['trade_date'] = index_data['trade_date']
        
        # ========== 1. 市场内生情绪因子 ==========
        
        # 两融余额占比变化
        if margin_data is not None and not margin_data.empty:
            # 计算两融余额/流通市值
            factors['margin_ratio'] = self._calculate_margin_ratio(margin_data)
        
        # 换手率变化（20日移动平均）
        if 'turnover' in index_data.columns:
            factors['turnover_ma20'] = index_data['turnover'].rolling(20).mean().pct_change()
        elif 'volume' in index_data.columns and 'amount' in index_data.columns:
            # 近似计算换手率
            factors['turnover_ma20'] = (index_data['amount'] / index_data['volume']).rolling(20).mean().pct_change()
        
        # ========== 2. 本土流动性因子 ==========
        
        # 10年期国债收益率变化
        if bond_data is not None and not bond_data.empty:
            factors['bond_yield_10y'] = self._extract_bond_yield(bond_data, '10年')
        
        # DR007（银行间7天回购利率）
        if rate_data is not None and not rate_data.empty:
            factors['dr007'] = self._extract_rate(rate_data, '7天')
        
        # ========== 3. 市场技术因子 ==========
        
        # 动量因子（5日、10日、20日收益率）
        factors['return_5d'] = index_data['close'].pct_change(5)
        factors['return_10d'] = index_data['close'].pct_change(10)
        factors['return_20d'] = index_data['close'].pct_change(20)
        
        # 波动率因子（20日滚动标准差）
        factors['volatility_20d'] = index_data['close'].pct_change().rolling(20).std()
        
        # ========== 4. 资金流向因子 ==========
        
        # 北向资金变化（如果有）
        if 'north_flow' in index_data.columns:
            factors['north_flow_ma5'] = index_data['north_flow'].rolling(5).mean()
        
        # ========== 填充缺失值 ==========
        factors = factors.fillna(method='ffill').fillna(method='bfill')
        
        return factors
    
    def _calculate_margin_ratio(self, margin_data: pd.DataFrame) -> pd.Series:
        """计算两融余额占比"""
        # 简化计算：使用融资余额变化率
        if '融资余额' in margin_data.columns:
            return margin_data['融资余额'].pct_change()
        return pd.Series()
    
    def _extract_bond_yield(self, bond_data: pd.DataFrame, term: str) -> pd.Series:
        """提取特定期限国债收益率"""
        for col in bond_data.columns:
            if term in str(col):
                return bond_data[col].pct_change()
        return pd.Series()
    
    def _extract_rate(self, rate_data: pd.DataFrame, term: str) -> pd.Series:
        """提取特定期限利率"""
        for col in rate_data.columns:
            if term in str(col):
                return rate_data[col].pct_change()
        return pd.Series()


class EndogenousBenchmarkModel:
    """
    内生基准拟合模型
    
    使用滚动窗口多元线性回归，拟合A股内生走势
    核心公式：R_t = α + Σβ_i * X_i,t + ε_t
    
    其中：
    - R_t: 沪深300指数日度涨跌幅
    - X_i,t: 纯A股内生驱动因子
    - ε_t: 残差（外生冲击代理变量）
    """
    
    # 模型参数
    ROLLING_WINDOW = 60  # 滚动窗口（60个交易日）
    MIN_R_SQUARED = 0.40  # 最小R²阈值
    MAX_VIF = 5  # 最大VIF阈值
    
    def __init__(self):
        self.factor_builder = EndogenousFactorBuilder()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    def fit(
        self,
        index_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        clean_period: Tuple[str, str] = None
    ) -> 'EndogenousBenchmarkModel':
        """
        拟合模型
        
        Args:
            index_data: 指数数据（包含trade_date, close等列）
            factor_data: 因子数据
            clean_period: 洁净期（避免事件污染）
        """
        # 筛选洁净期数据
        if clean_period:
            start_date, end_date = clean_period
            mask = (factor_data['trade_date'] >= start_date) & (factor_data['trade_date'] <= end_date)
            factor_data = factor_data[mask]
            mask = (index_data['trade_date'] >= start_date) & (index_data['trade_date'] <= end_date)
            index_data = index_data[mask]
        
        # 计算收益率
        y = index_data['close'].pct_change().dropna()
        
        # 准备特征
        feature_cols = [col for col in factor_data.columns if col != 'trade_date']
        X = factor_data[feature_cols].iloc[1:]  # 对齐收益率（去掉第一行）
        
        # 处理缺失值
        X = X.fillna(0)
        y = y.iloc[:len(X)]
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 计算VIF（方差膨胀因子）
        vif_scores = self._calculate_vif(X_scaled)
        valid_features = [f for f, vif in zip(feature_cols, vif_scores) if vif < self.MAX_VIF]
        
        logger.info(f"有效因子数量: {len(valid_features)}/{len(feature_cols)}")
        
        # 使用有效特征重新拟合
        valid_idx = [feature_cols.index(f) for f in valid_features]
        X_valid = X_scaled[:, valid_idx]
        
        # 拟合模型
        self.model = LinearRegression()
        self.model.fit(X_valid, y)
        self.feature_names = valid_features
        self.is_fitted = True
        
        # 计算R²
        y_pred = self.model.predict(X_valid)
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        
        logger.info(f"模型拟合完成，R² = {r_squared:.4f}")
        
        return self
    
    def predict(
        self,
        index_data: pd.DataFrame,
        factor_data: pd.DataFrame
    ) -> ModelResult:
        """
        预测并计算残差
        
        Returns:
            ModelResult: 包含拟合值、残差、显著性等
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合，请先调用fit()")
        
        # 准备数据
        y = index_data['close'].pct_change().iloc[-1]  # 最新一日收益率
        X = factor_data[[col for col in self.feature_names if col in factor_data.columns]].iloc[-1:]
        X = X.fillna(0)
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测
        fitted_value = self.model.predict(X_scaled)[0]
        residual = y - fitted_value
        
        # 计算残差标准差和Z-score
        # 需要历史残差
        residuals = self._get_historical_residuals(index_data, factor_data)
        residual_std = np.std(residuals)
        z_score = residual / residual_std if residual_std > 0 else 0
        
        # 判断显著性
        is_significant = abs(z_score) > 2.0  # 2倍标准差
        
        # 计算因子贡献度
        contribution = {}
        for i, name in enumerate(self.feature_names):
            contribution[name] = self.model.coef_[i] * X_scaled[0, i]
        
        # 计算R²
        r_squared = self.model.score(X_scaled, [y])
        
        return ModelResult(
            fitted_value=fitted_value,
            actual_value=y,
            residual=residual,
            residual_std=residual_std,
            z_score=z_score,
            r_squared=r_squared,
            is_significant=is_significant,
            contribution=contribution
        )
    
    def get_residuals(
        self,
        index_data: pd.DataFrame,
        factor_data: pd.DataFrame
    ) -> pd.Series:
        """
        获取历史残差序列
        """
        return self._get_historical_residuals(index_data, factor_data)
    
    def _get_historical_residuals(
        self,
        index_data: pd.DataFrame,
        factor_data: pd.DataFrame
    ) -> pd.Series:
        """计算历史残差"""
        y = index_data['close'].pct_change().dropna()
        X = factor_data[[col for col in self.feature_names if col in factor_data.columns]]
        X = X.fillna(0).iloc[1:]
        
        # 对齐
        min_len = min(len(y), len(X))
        y = y.iloc[:min_len]
        X = X.iloc[:min_len]
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        residuals = y.values - y_pred
        return pd.Series(residuals)
    
    def _calculate_vif(self, X: np.ndarray) -> List[float]:
        """
        计算方差膨胀因子（VIF）
        用于检测多重共线性
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_scores = []
        n_features = X.shape[1]
        
        # 如果特征太多，简化计算
        if n_features > 20:
            return [1.0] * n_features
        
        try:
            for i in range(n_features):
                vif = variance_inflation_factor(X, i)
                vif_scores.append(vif if not np.isinf(vif) else 100)
        except:
            vif_scores = [1.0] * n_features
        
        return vif_scores
    
    def analyze_residual_pattern(
        self,
        residuals: pd.Series,
        event_window: Tuple[int, int] = None
    ) -> Dict:
        """
        分析残差模式
        
        Args:
            residuals: 残差序列
            event_window: 事件窗口（相对于T0的偏移，如(-2, 10)）
        
        Returns:
            分析结果字典
        """
        result = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis(),
        }
        
        # 统计显著性检验
        # H0: 残差均值为0
        t_stat, p_value = stats.ttest_1samp(residuals, 0)
        result['t_statistic'] = t_stat
        result['p_value'] = p_value
        result['is_significant'] = p_value < 0.05
        
        # 如果有事件窗口，分析窗口内残差
        if event_window:
            start, end = event_window
            if 0 <= start and end < len(residuals):
                window_residuals = residuals.iloc[start:end+1]
                result['window_mean'] = window_residuals.mean()
                result['window_cumsum'] = window_residuals.sum()
                result['window_significance'] = abs(window_residuals.sum()) > 3 * residuals.std()
        
        return result


class ResidualAnalyzer:
    """
    残差分析器
    用于识别外生冲击
    """
    
    # 阈值设定
    THRESHOLDS = {
        'single_day_z': -2.0,  # 单日残差Z-score阈值
        'cumsum_3d_z': -3.0,  # 3日累计残差Z-score阈值
        'cumsum_5d_z': -4.0,  # 5日累计残差Z-score阈值
    }
    
    def __init__(self):
        pass
    
    def detect_anomaly(
        self,
        residuals: pd.Series,
        event_date_idx: int
    ) -> Dict:
        """
        检测异常残差
        
        Args:
            residuals: 残差序列
            event_date_idx: 事件日期在序列中的索引
        
        Returns:
            检测结果
        """
        result = {
            'is_anomaly': False,
            'anomaly_type': None,
            'details': {}
        }
        
        # 计算基准统计量
        std = residuals.std()
        
        # 单日检测
        if event_date_idx < len(residuals):
            single_day_residual = residuals.iloc[event_date_idx]
            single_day_z = single_day_residual / std
            
            result['details']['single_day'] = {
                'residual': single_day_residual,
                'z_score': single_day_z,
                'threshold_met': single_day_z < self.THRESHOLDS['single_day_z']
            }
        
        # 3日累计检测
        if event_date_idx + 2 < len(residuals):
            cumsum_3d = residuals.iloc[event_date_idx:event_date_idx+3].sum()
            cumsum_3d_z = cumsum_3d / (std * np.sqrt(3))
            
            result['details']['cumsum_3d'] = {
                'value': cumsum_3d,
                'z_score': cumsum_3d_z,
                'threshold_met': cumsum_3d_z < self.THRESHOLDS['cumsum_3d_z']
            }
        
        # 5日累计检测
        if event_date_idx + 4 < len(residuals):
            cumsum_5d = residuals.iloc[event_date_idx:event_date_idx+5].sum()
            cumsum_5d_z = cumsum_5d / (std * np.sqrt(5))
            
            result['details']['cumsum_5d'] = {
                'value': cumsum_5d,
                'z_score': cumsum_5d_z,
                'threshold_met': cumsum_5d_z < self.THRESHOLDS['cumsum_5d_z']
            }
        
        # 判断是否异常
        if result['details'].get('single_day', {}).get('threshold_met', False):
            result['is_anomaly'] = True
            result['anomaly_type'] = 'single_day_shock'
        elif result['details'].get('cumsum_3d', {}).get('threshold_met', False):
            result['is_anomaly'] = True
            result['anomaly_type'] = 'persistent_shock'
        
        return result
    
    def compare_with_clean_period(
        self,
        residuals: pd.Series,
        event_window: Tuple[int, int],
        clean_window: Tuple[int, int]
    ) -> Dict:
        """
        比较事件窗口与洁净期的残差
        
        用于区分内生下跌 vs 外生冲击
        """
        event_residuals = residuals.iloc[event_window[0]:event_window[1]+1]
        clean_residuals = residuals.iloc[clean_window[0]:clean_window[1]+1]
        
        # T检验
        t_stat, p_value = stats.ttest_ind(event_residuals, clean_residuals)
        
        # F检验（方差）
        f_stat = event_residuals.var() / clean_residuals.var()
        f_p_value = stats.f.cdf(f_stat, len(event_residuals)-1, len(clean_residuals)-1)
        
        return {
            'event_mean': event_residuals.mean(),
            'clean_mean': clean_residuals.mean(),
            'event_std': event_residuals.std(),
            'clean_std': clean_residuals.std(),
            't_statistic': t_stat,
            't_p_value': p_value,
            'f_statistic': f_stat,
            'f_p_value': f_p_value,
            'is_significantly_different': p_value < 0.05
        }


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试内生基准拟合模型")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # 模拟指数数据
    index_data = pd.DataFrame({
        'trade_date': dates,
        'close': 3000 + np.cumsum(np.random.randn(200) * 20),
        'volume': np.random.randint(1000000, 5000000, 200),
        'amount': np.random.randint(10000000, 50000000, 200)
    })
    
    # 模拟因子数据
    factor_data = pd.DataFrame({
        'trade_date': dates,
        'margin_ratio': np.random.randn(200) * 0.01,
        'bond_yield_10y': np.random.randn(200) * 0.001,
        'dr007': np.random.randn(200) * 0.001,
        'return_5d': index_data['close'].pct_change(5).fillna(0),
        'volatility_20d': index_data['close'].pct_change().rolling(20).std().fillna(0)
    })
    
    # 拟合模型
    model = EndogenousBenchmarkModel()
    model.fit(index_data, factor_data, clean_period=('2023-01-01', '2023-06-01'))
    
    # 预测
    result = model.predict(index_data, factor_data)
    print(f"\n预测结果:")
    print(f"  拟合值: {result.fitted_value:.6f}")
    print(f"  实际值: {result.actual_value:.6f}")
    print(f"  残差: {result.residual:.6f}")
    print(f"  Z-score: {result.z_score:.4f}")
    print(f"  是否显著: {result.is_significant}")
    
    print("\n" + "=" * 60)
