import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings
from scipy import stats
from ..metrics.factor_metrics import FactorMetrics


class GroupTester:
    """
    因子分组测试器
    实现因子分层回测和分组分析
    """

    def __init__(
        self,
        factor_data: pd.Series,
        forward_returns: pd.Series,
        prices: Optional[pd.DataFrame] = None,
    ):
        """
        初始化分组测试器

        Parameters
        ----------
        factor_data : pd.Series
            因子数据，索引为(date, symbol)
        forward_returns : pd.Series
            未来收益数据
        prices : pd.DataFrame, optional
            价格数据，用于计算净值曲线
        """
        if not isinstance(factor_data.index, pd.MultiIndex):
            raise ValueError("factor_data must have MultiIndex (date, symbol)")

        if not factor_data.index.equals(forward_returns.index):
            raise ValueError("factor_data and forward_returns must have same index")

        self.factor_data = factor_data
        self.forward_returns = forward_returns
        self.prices = prices

        # 因子指标计算器
        self.metrics_calculator = FactorMetrics(factor_data, forward_returns, prices)

    def create_groups(
        self,
        n_groups: int = 5,
        method: str = "quantile",
        date: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """
        创建分组

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str, {'quantile', 'equal', 'zscore'}
            分组方法：
            - 'quantile': 分位数分组
            - 'equal': 等权分组
            - 'zscore': Z-score分组
        date : pd.Timestamp, optional
            指定日期，如果为None则使用所有日期

        Returns
        -------
        pd.Series
            分组标签，索引与输入数据一致
        """
        if date is not None:
            # 指定日期的分组
            mask = self.factor_data.index.get_level_values(0) == date
            factor_values = self.factor_data[mask]
        else:
            # 所有日期的分组（按日期分别分组）
            factor_values = self.factor_data

        # 初始化分组结果
        groups = pd.Series(index=factor_values.index, dtype=float)

        if date is not None:
            # 单日分组
            groups.loc[mask] = self._create_single_group(
                factor_values, n_groups, method
            )
        else:
            # 按日期分别分组
            dates = factor_values.index.get_level_values(0).unique()

            for d in dates:
                date_mask = factor_values.index.get_level_values(0) == d
                date_factor = factor_values[date_mask]

                if len(date_factor) >= n_groups:
                    date_groups = self._create_single_group(
                        date_factor, n_groups, method
                    )
                    groups.loc[date_mask] = date_groups

        return groups

    def _create_single_group(
        self, factor_values: pd.Series, n_groups: int, method: str
    ) -> pd.Series:
        """创建单日分组"""
        # 移除缺失值
        factor_valid = factor_values.dropna()
        if len(factor_valid) < n_groups:
            return pd.Series(index=factor_values.index, data=np.nan)

        if method == "quantile":
            # 分位数分组
            try:
                groups = pd.qcut(
                    factor_valid, n_groups, labels=False, duplicates="drop"
                )
            except ValueError:
                # 如果无法分位数分组，使用等权分组
                ranks = factor_valid.rank()
                group_size = len(factor_valid) // n_groups
                groups = (ranks - 1) // group_size
                groups[groups >= n_groups] = n_groups - 1

        elif method == "equal":
            # 等权分组
            ranks = factor_valid.rank()
            group_size = len(factor_valid) // n_groups
            groups = (ranks - 1) // group_size
            groups[groups >= n_groups] = n_groups - 1

        elif method == "zscore":
            # Z-score分组
            zscores = (factor_valid - factor_valid.mean()) / factor_valid.std()
            # 等距分组
            min_z = zscores.min()
            max_z = zscores.max()
            bins = np.linspace(min_z, max_z, n_groups + 1)
            groups = pd.cut(zscores, bins=bins, labels=False, include_lowest=True)

        # 重新索引到原始索引
        return groups.reindex(factor_values.index)

    def calculate_group_performance(
        self,
        n_groups: int = 5,
        method: str = "quantile",
        rebalance_freq: str = "D",
        long_short: bool = True,
    ) -> Dict:
        """
        计算分组表现

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str
            分组方法
        rebalance_freq : str, {'D', 'W', 'M'}
            再平衡频率
        long_short : bool
            是否计算多空组合

        Returns
        -------
        Dict
            分组表现数据
        """
        # 获取分组收益
        group_returns_df = self.metrics_calculator.calculate_group_returns(
            n_groups=n_groups, method=method, long_short=long_short
        )

        if group_returns_df.empty:
            return {}

        # 按再平衡频率重采样
        if rebalance_freq != "D":
            group_returns_df = group_returns_df.resample(rebalance_freq).mean()

        # 计算累计收益
        cum_returns_df = (1 + group_returns_df).cumprod() - 1

        # 计算各项指标
        performance = {}

        for col in group_returns_df.columns:
            returns = group_returns_df[col].dropna()
            if len(returns) == 0:
                continue

            # 基本统计
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe = mean_return / std_return if std_return != 0 else np.nan

            # 最大回撤
            cum_returns = cum_returns_df[col].dropna()
            if len(cum_returns) > 0:
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / (running_max + 1e-10)
                max_drawdown = drawdown.min()
            else:
                max_drawdown = np.nan

            # 胜率
            win_rate = (returns > 0).mean()

            # Calmar比率
            calmar = -mean_return / max_drawdown if max_drawdown < 0 else np.nan

            performance[col] = {
                "mean_return": mean_return,
                "std_return": std_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "calmar_ratio": calmar,
                "total_return": cum_returns.iloc[-1]
                if len(cum_returns) > 0
                else np.nan,
                "num_periods": len(returns),
            }

        return performance

    def create_long_short_portfolio(
        self,
        n_groups: int = 5,
        method: str = "quantile",
        top_group: int = 0,
        bottom_group: Optional[int] = None,
        weight_method: str = "equal",
    ) -> pd.Series:
        """
        创建多空组合

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str
            分组方法
        top_group : int
            做多组索引（0为最高组）
        bottom_group : int, optional
            做空组索引，如果为None则使用最低组
        weight_method : str, {'equal', 'factor_weighted'}
            权重分配方法

        Returns
        -------
        pd.Series
            多空组合收益时间序列
        """
        if bottom_group is None:
            bottom_group = n_groups - 1

        # 获取分组收益
        group_returns_df = self.metrics_calculator.calculate_group_returns(
            n_groups=n_groups, method=method, long_short=False
        )

        if group_returns_df.empty:
            return pd.Series(dtype=float)

        # 提取做多和做空组收益
        long_col = f"group_{top_group}"
        short_col = f"group_{bottom_group}"

        if (
            long_col not in group_returns_df.columns
            or short_col not in group_returns_df.columns
        ):
            raise ValueError(f"Groups {top_group} or {bottom_group} not found")

        long_returns = group_returns_df[long_col]
        short_returns = group_returns_df[short_col]

        # 计算多空组合收益
        long_short_returns = long_returns - short_returns

        return long_short_returns

    def calculate_group_turnover(
        self, n_groups: int = 5, method: str = "quantile"
    ) -> Dict:
        """
        计算分组换手率

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str
            分组方法

        Returns
        -------
        Dict
            分组换手率数据
        """
        # 获取所有日期的分组
        all_groups = self.create_groups(n_groups=n_groups, method=method)

        if all_groups.isna().all():
            return {}

        # 转换为面板格式
        groups_panel = all_groups.unstack()

        # 计算换手率
        turnover_results = {}

        for group in range(n_groups):
            group_turnover = []
            prev_date = None
            prev_symbols = set()

            for date in sorted(groups_panel.index):
                current_symbols = set(
                    groups_panel.columns[groups_panel.loc[date] == group].tolist()
                )

                if prev_date is not None:
                    # 计算换手率
                    union_size = len(prev_symbols.union(current_symbols))
                    if union_size > 0:
                        turnover = (
                            1
                            - len(prev_symbols.intersection(current_symbols))
                            / union_size
                        )
                    else:
                        turnover = np.nan
                    group_turnover.append(turnover)

                prev_date = date
                prev_symbols = current_symbols

            if group_turnover:
                turnover_series = pd.Series(
                    group_turnover, index=sorted(groups_panel.index)[1:]
                )
                turnover_results[f"group_{group}"] = {
                    "mean_turnover": turnover_series.mean(),
                    "std_turnover": turnover_series.std(),
                    "max_turnover": turnover_series.max(),
                    "turnover_series": turnover_series,
                }

        # 计算多空组合换手率
        if n_groups >= 2:
            top_turnover = turnover_results.get(f"group_0", {}).get(
                "turnover_series", pd.Series()
            )
            bottom_turnover = turnover_results.get(f"group_{n_groups - 1}", {}).get(
                "turnover_series", pd.Series()
            )

            if not top_turnover.empty and not bottom_turnover.empty:
                # 多空组合换手率为两组换手率的平均值
                long_short_turnover = (top_turnover + bottom_turnover) / 2
                turnover_results["long_short"] = {
                    "mean_turnover": long_short_turnover.mean(),
                    "std_turnover": long_short_turnover.std(),
                    "max_turnover": long_short_turnover.max(),
                    "turnover_series": long_short_turnover,
                }

        return turnover_results

    def run_comprehensive_test(
        self, n_groups: int = 5, method: str = "quantile", rebalance_freq: str = "D"
    ) -> Dict:
        """
        运行全面的分组测试

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str
            分组方法
        rebalance_freq : str
            再平衡频率

        Returns
        -------
        Dict
            全面的测试结果
        """
        results = {}

        # 1. 分组表现
        group_performance = self.calculate_group_performance(
            n_groups=n_groups, method=method, rebalance_freq=rebalance_freq
        )
        results["group_performance"] = group_performance

        # 2. 分组换手率
        turnover_results = self.calculate_group_turnover(
            n_groups=n_groups, method=method
        )
        results["turnover"] = turnover_results

        # 3. 多空组合
        long_short_returns = self.create_long_short_portfolio(
            n_groups=n_groups, method=method
        )
        if not long_short_returns.empty:
            # 计算多空组合指标
            ls_mean = long_short_returns.mean()
            ls_std = long_short_returns.std()
            ls_sharpe = ls_mean / ls_std if ls_std != 0 else np.nan
            ls_win_rate = (long_short_returns > 0).mean()

            results["long_short"] = {
                "mean_return": ls_mean,
                "std_return": ls_std,
                "sharpe_ratio": ls_sharpe,
                "win_rate": ls_win_rate,
                "returns_series": long_short_returns,
            }

        # 4. 分组收益的单调性检验
        group_returns_df = self.metrics_calculator.calculate_group_returns(
            n_groups=n_groups, method=method, long_short=False
        )
        if not group_returns_df.empty:
            # 计算各组的平均收益
            group_means = {}
            for i in range(n_groups):
                col = f"group_{i}"
                if col in group_returns_df.columns:
                    group_means[i] = group_returns_df[col].mean()

            # 单调性检验：计算Spearman相关系数
            if len(group_means) >= 3:
                group_indices = list(group_means.keys())
                group_values = [group_means[i] for i in group_indices]

                # 预期：高分组收益高，低分组收益低
                expected_order = list(range(len(group_indices)))  # 0, 1, 2, ...

                monotonicity = stats.spearmanr(expected_order, group_values)[0]
                results["monotonicity"] = {
                    "spearman_corr": monotonicity,
                    "group_means": group_means,
                }

        return results

    @staticmethod
    def compare_factors(
        factor_data_dict: Dict[str, pd.Series],
        forward_returns: pd.Series,
        n_groups: int = 5,
        method: str = "quantile",
    ) -> pd.DataFrame:
        """
        比较多个因子的分组表现

        Parameters
        ----------
        factor_data_dict : Dict[str, pd.Series]
            因子数据字典，{factor_name: factor_data}
        forward_returns : pd.Series
            未来收益数据
        n_groups : int
            分组数量
        method : str
            分组方法

        Returns
        -------
        pd.DataFrame
            因子比较结果
        """
        comparison_results = []

        for factor_name, factor_data in factor_data_dict.items():
            # 创建测试器
            tester = GroupTester(factor_data, forward_returns)

            # 运行测试
            try:
                results = tester.run_comprehensive_test(
                    n_groups=n_groups, method=method
                )

                # 提取关键指标
                key_metrics = {
                    "factor": factor_name,
                    "long_short_sharpe": results.get("long_short", {}).get(
                        "sharpe_ratio", np.nan
                    ),
                    "long_short_return": results.get("long_short", {}).get(
                        "mean_return", np.nan
                    ),
                    "monotonicity": results.get("monotonicity", {}).get(
                        "spearman_corr", np.nan
                    ),
                    "avg_turnover": results.get("turnover", {})
                    .get("long_short", {})
                    .get("mean_turnover", np.nan),
                }

                comparison_results.append(key_metrics)
            except Exception as e:
                print(f"Error testing factor {factor_name}: {e}")
                comparison_results.append(
                    {
                        "factor": factor_name,
                        "long_short_sharpe": np.nan,
                        "long_short_return": np.nan,
                        "monotonicity": np.nan,
                        "avg_turnover": np.nan,
                    }
                )

        return pd.DataFrame(comparison_results).set_index("factor")
