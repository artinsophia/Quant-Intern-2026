"""
因子分组回测模块
通过分组回测评估因子的区分能力和单调性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings


class GroupBacktester:
    """因子分组回测器"""

    def __init__(self, n_groups: int = 5, group_method: str = "quantile"):
        """
        初始化分组回测器

        Args:
            n_groups: 分组数量
            group_method: 分组方法 ('quantile'或'equal')
        """
        self.n_groups = n_groups
        self.group_method = group_method
        self.group_labels = [f"Group_{i + 1}" for i in range(n_groups)]

    def create_factor_groups(
        self, factor_values: pd.Series, date: pd.Timestamp = None
    ) -> pd.Series:
        """
        创建因子分组

        Args:
            factor_values: 因子值序列
            date: 分组日期（用于时间序列分组）

        Returns:
            分组标签序列
        """
        if self.group_method == "quantile":
            # 按分位数分组
            groups = pd.qcut(
                factor_values,
                q=self.n_groups,
                labels=self.group_labels,
                duplicates="drop",
            )
        elif self.group_method == "equal":
            # 等间距分组
            groups = pd.cut(factor_values, bins=self.n_groups, labels=self.group_labels)
        else:
            raise ValueError(f"不支持的group_method: {self.group_method}")

        return groups

    def calculate_group_returns(
        self,
        factor_df: pd.DataFrame,
        price_series: pd.Series,
        factor_name: str,
        forward_periods: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, pd.DataFrame]:
        """
        计算分组收益率

        Args:
            factor_df: 因子值DataFrame
            price_series: 价格序列
            factor_name: 因子名称
            forward_periods: 未来收益周期列表

        Returns:
            分组收益率结果字典
        """
        # 对齐数据
        common_idx = factor_df.index.intersection(price_series.index)
        factor_values = factor_df.loc[common_idx, factor_name]
        prices = price_series.loc[common_idx]

        # 初始化结果存储
        group_returns = {
            period: pd.DataFrame(index=factor_values.index)
            for period in forward_periods
        }
        group_labels_all = pd.Series(index=factor_values.index, dtype=str)

        # 对每个时间点进行分组
        for i in range(len(factor_values)):
            if i < self.n_groups:  # 需要足够的数据进行分组
                continue

            current_factor = factor_values.iloc[i]
            current_date = factor_values.index[i]

            # 使用历史数据计算分组
            hist_factor = factor_values.iloc[max(0, i - 100) : i]  # 使用最近100个数据点
            groups = self.create_factor_groups(hist_factor)

            # 确定当前因子值所在分组
            if pd.isna(current_factor):
                group_labels_all.iloc[i] = np.nan
            else:
                # 找到当前因子值在历史分位数中的位置
                quantiles = hist_factor.quantile(np.linspace(0, 1, self.n_groups + 1))
                for group_idx in range(self.n_groups):
                    if (
                        quantiles.iloc[group_idx]
                        <= current_factor
                        <= quantiles.iloc[group_idx + 1]
                    ):
                        group_labels_all.iloc[i] = self.group_labels[group_idx]
                        break

        # 计算各分组未来收益率
        for period in forward_periods:
            # 计算未来收益率
            future_prices = prices.shift(-period)
            returns = (future_prices - prices) / prices

            # 按分组计算收益率
            for group in self.group_labels:
                group_mask = group_labels_all == group
                group_returns[period][group] = returns.where(group_mask)

            # 计算多空组合收益率 (最高组 - 最低组)
            if len(self.group_labels) >= 2:
                long_group = self.group_labels[-1]  # 最高分组
                short_group = self.group_labels[0]  # 最低分组
                group_returns[period]["Long-Short"] = (
                    group_returns[period][long_group]
                    - group_returns[period][short_group]
                )

        return group_returns

    def analyze_group_performance(
        self, group_returns: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        分析分组表现

        Args:
            group_returns: 分组收益率字典

        Returns:
            分组表现分析结果
        """
        results = []

        for period, returns_df in group_returns.items():
            for column in returns_df.columns:
                returns_series = returns_df[column].dropna()

                if len(returns_series) == 0:
                    continue

                # 计算表现指标
                mean_return = returns_series.mean()
                std_return = returns_series.std()
                sharpe_ratio = (
                    mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
                )
                win_rate = (returns_series > 0).mean()
                max_drawdown = self.calculate_max_drawdown(returns_series)

                results.append(
                    {
                        "period": period,
                        "group": column,
                        "mean_return": mean_return,
                        "std_return": std_return,
                        "sharpe_ratio": sharpe_ratio,
                        "win_rate": win_rate,
                        "max_drawdown": max_drawdown,
                        "sample_size": len(returns_series),
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def calculate_max_drawdown(returns_series: pd.Series) -> float:
        """
        计算最大回撤

        Args:
            returns_series: 收益率序列

        Returns:
            最大回撤
        """
        if len(returns_series) == 0:
            return 0

        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()

    def calculate_monotonicity(
        self, group_performance: pd.DataFrame, factor_name: str
    ) -> Dict[str, float]:
        """
        计算因子单调性

        Args:
            group_performance: 分组表现结果
            factor_name: 因子名称

        Returns:
            单调性指标字典
        """
        monotonicity_results = {}

        for period in group_performance["period"].unique():
            period_data = group_performance[group_performance["period"] == period]

            # 只取原始分组 (排除多空组合)
            groups_data = period_data[
                period_data["group"].str.startswith("Group_")
            ].copy()

            if len(groups_data) < 3:  # 需要至少3个分组评估单调性
                continue

            # 按分组编号排序
            groups_data["group_num"] = (
                groups_data["group"].str.extract(r"(\d+)").astype(int)
            )
            groups_data = groups_data.sort_values("group_num")

            # 计算分组收益率的秩相关系数
            group_nums = groups_data["group_num"].values
            mean_returns = groups_data["mean_return"].values

            # 计算Spearman秩相关系数
            from scipy import stats

            spearman_corr, _ = stats.spearmanr(group_nums, mean_returns)

            # 计算单调性得分 (理想情况下，高分组应有更高收益)
            monotonicity_score = spearman_corr

            monotonicity_results[f"period_{period}"] = {
                "spearman_corr": spearman_corr,
                "monotonicity_score": monotonicity_score,
                "n_groups": len(groups_data),
            }

        return monotonicity_results

    def run_complete_analysis(
        self,
        factor_df: pd.DataFrame,
        price_series: pd.Series,
        factor_name: str,
        forward_periods: List[int] = [1, 5, 10, 20],
    ) -> Dict:
        """
        运行完整的因子分组分析

        Args:
            factor_df: 因子值DataFrame
            price_series: 价格序列
            factor_name: 因子名称
            forward_periods: 未来收益周期列表

        Returns:
            完整分析结果字典
        """
        # 计算分组收益率
        group_returns = self.calculate_group_returns(
            factor_df, price_series, factor_name, forward_periods
        )

        # 分析分组表现
        group_performance = self.analyze_group_performance(group_returns)

        # 计算单调性
        monotonicity = self.calculate_monotonicity(group_performance, factor_name)

        # 计算分组收益统计
        group_stats = {}
        for period in forward_periods:
            if period in group_returns:
                period_returns = group_returns[period]
                period_stats = {}

                for group in period_returns.columns:
                    returns = period_returns[group].dropna()
                    if len(returns) > 0:
                        period_stats[group] = {
                            "mean": returns.mean(),
                            "std": returns.std(),
                            "sharpe": returns.mean() / returns.std() * np.sqrt(252)
                            if returns.std() > 0
                            else 0,
                            "count": len(returns),
                        }

                group_stats[f"period_{period}"] = period_stats

        return {
            "group_returns": group_returns,
            "group_performance": group_performance,
            "monotonicity": monotonicity,
            "group_stats": group_stats,
            "factor_name": factor_name,
            "n_groups": self.n_groups,
        }
