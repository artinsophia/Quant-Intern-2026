"""
因子IC值计算模块
计算信息系数(Information Coefficient)评估因子预测能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


class ICAnalyzer:
    """因子IC值分析器"""

    def __init__(self, forward_periods: List[int] = [1, 5, 10, 20]):
        """
        初始化IC分析器

        Args:
            forward_periods: 未来收益计算周期列表
        """
        self.forward_periods = forward_periods

    def calculate_forward_returns(
        self, prices: pd.Series, periods: List[int]
    ) -> pd.DataFrame:
        """
        计算未来收益率

        Args:
            prices: 价格序列
            periods: 未来周期列表

        Returns:
            DataFrame, 每列为不同周期的未来收益率
        """
        returns_df = pd.DataFrame(index=prices.index)

        for period in periods:
            # 计算未来period期的收益率
            future_prices = prices.shift(-period)
            returns = (future_prices - prices) / prices
            returns_df[f"return_{period}"] = returns

        return returns_df

    def calculate_ic(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        method: str = "spearman",
    ) -> float:
        """
        计算单个因子的IC值

        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益率序列
            method: 相关系数计算方法 ('pearson'或'spearman')

        Returns:
            IC值
        """
        # 对齐数据，去除NaN
        valid_idx = factor_values.notna() & forward_returns.notna()
        factor_clean = factor_values[valid_idx]
        returns_clean = forward_returns[valid_idx]

        if len(factor_clean) < 10:
            return np.nan

        if method == "pearson":
            ic, _ = stats.pearsonr(factor_clean, returns_clean)
        elif method == "spearman":
            ic, _ = stats.spearmanr(factor_clean, returns_clean)
        else:
            raise ValueError(f"不支持的method: {method}")

        return ic

    def analyze_factor_ic(
        self,
        factor_df: pd.DataFrame,
        price_series: pd.Series,
        factor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        分析多个因子的IC值

        Args:
            factor_df: 因子值DataFrame，索引为时间
            price_series: 价格序列
            factor_names: 要分析的因子名称列表，None表示分析所有列

        Returns:
            IC分析结果DataFrame
        """
        if factor_names is None:
            factor_names = factor_df.columns.tolist()

        # 计算未来收益率
        forward_returns = self.calculate_forward_returns(
            price_series, self.forward_periods
        )

        # 对齐时间索引
        common_idx = factor_df.index.intersection(forward_returns.index)
        factor_df = factor_df.loc[common_idx]
        forward_returns = forward_returns.loc[common_idx]

        results = []

        for factor_name in factor_names:
            factor_values = factor_df[factor_name]

            for period in self.forward_periods:
                return_col = f"return_{period}"
                future_returns = forward_returns[return_col]

                # 计算IC值
                ic_pearson = self.calculate_ic(factor_values, future_returns, "pearson")
                ic_spearman = self.calculate_ic(
                    factor_values, future_returns, "spearman"
                )

                results.append(
                    {
                        "factor": factor_name,
                        "period": period,
                        "ic_pearson": ic_pearson,
                        "ic_spearman": ic_spearman,
                        "abs_ic_pearson": abs(ic_pearson),
                        "abs_ic_spearman": abs(ic_spearman),
                        "sample_size": len(factor_values.dropna()),
                    }
                )

        return pd.DataFrame(results)

    def calculate_ic_series(
        self,
        factor_df: pd.DataFrame,
        price_series: pd.Series,
        factor_name: str,
        period: int = 1,
        rolling_window: int = 20,
    ) -> pd.Series:
        """
        计算滚动IC序列

        Args:
            factor_df: 因子值DataFrame
            price_series: 价格序列
            factor_name: 因子名称
            period: 未来收益周期
            rolling_window: 滚动窗口大小

        Returns:
            滚动IC序列
        """
        # 计算未来收益率
        forward_returns = self.calculate_forward_returns(price_series, [period])
        return_col = f"return_{period}"

        # 对齐数据
        common_idx = factor_df.index.intersection(forward_returns.index)
        factor_values = factor_df.loc[common_idx, factor_name]
        future_returns = forward_returns.loc[common_idx, return_col]

        # 计算滚动IC
        ic_series = pd.Series(index=factor_values.index, dtype=float)

        for i in range(rolling_window, len(factor_values)):
            window_start = i - rolling_window
            window_end = i

            factor_window = factor_values.iloc[window_start:window_end]
            returns_window = future_returns.iloc[window_start:window_end]

            ic = self.calculate_ic(factor_window, returns_window, "spearman")
            ic_series.iloc[i] = ic

        return ic_series

    def calculate_ic_decay(
        self,
        factor_df: pd.DataFrame,
        price_series: pd.Series,
        factor_name: str,
        max_period: int = 30,
    ) -> pd.DataFrame:
        """
        计算IC衰减曲线

        Args:
            factor_df: 因子值DataFrame
            price_series: 价格序列
            factor_name: 因子名称
            max_period: 最大衰减周期

        Returns:
            IC衰减结果
        """
        periods = list(range(1, max_period + 1))

        # 计算未来收益率
        forward_returns = self.calculate_forward_returns(price_series, periods)

        # 对齐数据
        common_idx = factor_df.index.intersection(forward_returns.index)
        factor_values = factor_df.loc[common_idx, factor_name]
        forward_returns = forward_returns.loc[common_idx]

        decay_results = []

        for period in periods:
            return_col = f"return_{period}"
            future_returns = forward_returns[return_col]

            ic = self.calculate_ic(factor_values, future_returns, "spearman")

            decay_results.append({"period": period, "ic": ic, "abs_ic": abs(ic)})

        return pd.DataFrame(decay_results)


def calculate_factor_stats(ic_results: pd.DataFrame) -> pd.DataFrame:
    """
    计算因子统计指标

    Args:
        ic_results: IC分析结果DataFrame

    Returns:
        因子统计指标DataFrame
    """
    stats_list = []

    for factor in ic_results["factor"].unique():
        factor_data = ic_results[ic_results["factor"] == factor]

        # 基本统计
        mean_ic = factor_data["ic_spearman"].mean()
        std_ic = factor_data["ic_spearman"].std()
        mean_abs_ic = factor_data["abs_ic_spearman"].mean()

        # ICIR (信息比率)
        icir = mean_ic / std_ic if std_ic > 0 else 0

        # 胜率 (IC > 0的比例)
        win_rate = (factor_data["ic_spearman"] > 0).mean()

        # 稳定性 (IC序列的自相关性)
        if len(factor_data) > 1:
            stability = factor_data["ic_spearman"].autocorr()
        else:
            stability = np.nan

        stats_list.append(
            {
                "factor": factor,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "mean_abs_ic": mean_abs_ic,
                "icir": icir,
                "win_rate": win_rate,
                "stability": stability,
                "sample_periods": len(factor_data),
            }
        )

    return pd.DataFrame(stats_list)
