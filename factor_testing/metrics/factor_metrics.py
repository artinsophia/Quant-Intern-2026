import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
import warnings
from .ic_calculator import ICCalculator


class FactorMetrics:
    """
    因子综合指标计算器
    计算IR、换手率、衰减率、分组收益等综合指标
    """

    def __init__(
        self,
        factor_data: pd.Series,
        forward_returns: pd.Series,
        prices: Optional[pd.DataFrame] = None,
        groupby: Optional[pd.Series] = None,
    ):
        """
        初始化因子指标计算器

        Parameters
        ----------
        factor_data : pd.Series
            因子数据
        forward_returns : pd.Series
            未来收益数据
        prices : pd.DataFrame, optional
            价格数据，用于计算换手率等指标
        groupby : pd.Series, optional
            分组序列
        """
        self.factor_data = factor_data
        self.forward_returns = forward_returns
        self.prices = prices
        self.groupby = groupby

        # IC计算器
        self.ic_calculator = ICCalculator(factor_data, forward_returns, groupby)

    def calculate_ir(self, freq: str = "D", method: str = "pearson") -> float:
        """
        计算信息比率（IR）

        Parameters
        ----------
        freq : str
            时间频率
        method : str
            IC计算方法

        Returns
        -------
        float
            信息比率
        """
        ic_series = self.ic_calculator.calculate_ic_series(freq=freq, method=method)
        ic_series_valid = ic_series.dropna()

        if len(ic_series_valid) == 0:
            return np.nan

        ir = (
            ic_series_valid.mean() / ic_series_valid.std()
            if ic_series_valid.std() != 0
            else np.nan
        )
        return ir

    def calculate_turnover(
        self, n_groups: int = 5, method: str = "quantile", freq: str = "D"
    ) -> pd.Series:
        """
        计算因子换手率

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str, {'quantile', 'equal'}
            分组方法
        freq : str
            换手率计算频率

        Returns
        -------
        pd.Series
            换手率时间序列
        """
        if not isinstance(self.factor_data.index, pd.MultiIndex):
            raise ValueError("Turnover calculation requires MultiIndex (date, symbol)")

        # 获取日期
        dates = sorted(self.factor_data.index.get_level_values(0).unique())

        # 转换为面板数据
        factor_panel = self.factor_data.unstack()

        # 计算每个时间点的分组
        group_assignments = {}

        for date in dates:
            if date not in factor_panel.index:
                continue

            factor_values = factor_panel.loc[date]

            # 移除缺失值
            factor_valid = factor_values.dropna()
            if len(factor_valid) == 0:
                group_assignments[date] = pd.Series(
                    index=factor_values.index, data=np.nan
                )
                continue

            # 分组
            if method == "quantile":
                # 分位数分组
                groups = pd.qcut(
                    factor_valid, n_groups, labels=False, duplicates="drop"
                )
            elif method == "equal":
                # 等权分组
                ranks = factor_valid.rank()
                group_size = len(factor_valid) // n_groups
                groups = (ranks - 1) // group_size
                groups[groups >= n_groups] = n_groups - 1

            group_assignments[date] = groups.reindex(factor_values.index)

        # 计算换手率
        turnover_series = {}
        prev_date = None
        prev_groups = None

        for date in sorted(group_assignments.keys()):
            if prev_date is not None:
                # 计算相邻两个时间点的换手率
                current_groups = group_assignments[date]

                # 对齐标的
                common_symbols = prev_groups.index.intersection(current_groups.index)
                if len(common_symbols) == 0:
                    turnover_series[date] = np.nan
                    continue

                prev_groups_aligned = prev_groups[common_symbols]
                current_groups_aligned = current_groups[common_symbols]

                # 计算换手率：分组变化的标的比例
                turnover = (prev_groups_aligned != current_groups_aligned).mean()
                turnover_series[date] = turnover

            prev_date = date
            prev_groups = group_assignments[date]

        # 转换为指定频率
        turnover_df = pd.Series(turnover_series)
        if freq != "D":
            turnover_df = turnover_df.resample(freq).mean()

        return turnover_df

    def calculate_decay_rate(self, max_lag: int = 10, method: str = "pearson") -> Dict:
        """
        计算因子衰减率

        Parameters
        ----------
        max_lag : int
            最大滞后阶数
        method : str
            IC计算方法

        Returns
        -------
        Dict
            衰减率相关指标
        """
        ic_decay = self.ic_calculator.calculate_ic_decay(max_lag=max_lag, method=method)
        ic_decay_valid = ic_decay.dropna()

        if len(ic_decay_valid) == 0:
            return {"decay_series": ic_decay, "half_life": np.nan, "decay_rate": np.nan}

        # 计算半衰期（IC衰减到一半所需的滞后阶数）
        ic_0 = ic_decay_valid.iloc[0] if 1 in ic_decay_valid.index else np.nan
        if np.isnan(ic_0) or ic_0 == 0:
            half_life = np.nan
        else:
            # 找到IC衰减到一半的滞后阶数
            half_value = ic_0 / 2
            half_life_idx = np.where(
                np.abs(ic_decay_valid.values) <= np.abs(half_value)
            )[0]
            half_life = half_life_idx[0] + 1 if len(half_life_idx) > 0 else np.nan

        # 计算衰减率（指数衰减拟合）
        try:
            # 使用指数衰减模型拟合：IC(t) = IC(0) * exp(-λt)
            lags = ic_decay_valid.index.values.astype(float)
            ics = ic_decay_valid.values

            # 移除零值
            valid_mask = (ics != 0) & ~np.isnan(ics)
            if valid_mask.sum() > 1:
                lags_valid = lags[valid_mask]
                ics_valid = ics[valid_mask]

                # 线性回归：ln|IC| = ln|IC0| - λt
                log_ics = np.log(np.abs(ics_valid))
                slope, intercept = np.polyfit(lags_valid, log_ics, 1)
                decay_rate = -slope  # λ
            else:
                decay_rate = np.nan
        except:
            decay_rate = np.nan

        return {
            "decay_series": ic_decay,
            "half_life": half_life,
            "decay_rate": decay_rate,
        }

    def calculate_group_returns(
        self, n_groups: int = 5, method: str = "quantile", long_short: bool = True
    ) -> pd.DataFrame:
        """
        计算分组收益

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str, {'quantile', 'equal'}
            分组方法
        long_short : bool
            是否计算多空组合收益

        Returns
        -------
        pd.DataFrame
            分组收益数据
        """
        if not isinstance(self.factor_data.index, pd.MultiIndex):
            raise ValueError(
                "Group returns calculation requires MultiIndex (date, symbol)"
            )

        # 获取日期和标的
        dates = sorted(self.factor_data.index.get_level_values(0).unique())

        # 转换为面板数据
        factor_panel = self.factor_data.unstack()
        returns_panel = self.forward_returns.unstack()

        # 存储分组收益
        group_returns_dict = {f"group_{i}": [] for i in range(n_groups)}
        if long_short:
            group_returns_dict["long_short"] = []

        group_dates = []

        for date in dates:
            if date not in factor_panel.index or date not in returns_panel.index:
                continue

            factor_values = factor_panel.loc[date]
            return_values = returns_panel.loc[date]

            # 对齐数据
            common_symbols = factor_values.index.intersection(return_values.index)
            if len(common_symbols) == 0:
                continue

            factor_aligned = factor_values[common_symbols]
            return_aligned = return_values[common_symbols]

            # 移除缺失值
            valid_mask = factor_aligned.notna() & return_aligned.notna()
            factor_valid = factor_aligned[valid_mask]
            return_valid = return_aligned[valid_mask]

            if len(factor_valid) < n_groups:
                continue

            # 分组
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

            # 计算每组收益
            group_returns = {}
            for group in range(n_groups):
                group_mask = groups == group
                if group_mask.any():
                    group_return = return_valid[group_mask].mean()
                else:
                    group_return = np.nan
                group_returns[f"group_{group}"] = group_return

            # 计算多空组合收益（第一组做多，最后一组做空）
            if (
                long_short
                and f"group_0" in group_returns
                and f"group_{n_groups - 1}" in group_returns
            ):
                long_return = group_returns[f"group_0"]
                short_return = group_returns[f"group_{n_groups - 1}"]
                if not np.isnan(long_return) and not np.isnan(short_return):
                    group_returns["long_short"] = long_return - short_return
                else:
                    group_returns["long_short"] = np.nan

            # 保存结果
            for key, value in group_returns.items():
                group_returns_dict[key].append(value)
            group_dates.append(date)

        # 创建DataFrame
        result_df = pd.DataFrame(group_returns_dict, index=group_dates)
        result_df.index.name = "date"

        return result_df

    def calculate_all_metrics(
        self, n_groups: int = 5, freq: str = "D", method: str = "pearson"
    ) -> Dict:
        """
        计算所有因子指标

        Parameters
        ----------
        n_groups : int
            分组数量
        freq : str
            计算频率
        method : str
            IC计算方法

        Returns
        -------
        Dict
            所有因子指标
        """
        metrics = {}

        # IC相关指标
        ic_stats = self.ic_calculator.calculate_ic_stats(method=method, freq=freq)
        metrics.update({f"ic_{k}": v for k, v in ic_stats.items()})

        # IR
        metrics["ir"] = self.calculate_ir(freq=freq, method=method)

        # Rank IC
        rank_ic = self.ic_calculator.calculate_rank_ic()
        metrics["rank_ic"] = rank_ic

        # 衰减率
        decay_info = self.calculate_decay_rate(method=method)
        metrics.update(
            {
                "decay_half_life": decay_info["half_life"],
                "decay_rate": decay_info["decay_rate"],
            }
        )

        # 分组收益
        group_returns = self.calculate_group_returns(n_groups=n_groups)
        if not group_returns.empty:
            # 计算累计收益
            cum_returns = (1 + group_returns).cumprod() - 1

            metrics.update(
                {
                    "group_0_mean_return": group_returns["group_0"].mean(),
                    f"group_{n_groups - 1}_mean_return": group_returns[
                        f"group_{n_groups - 1}"
                    ].mean(),
                    "long_short_mean_return": group_returns.get(
                        "long_short", pd.Series([np.nan])
                    ).mean(),
                    "group_0_sharpe": group_returns["group_0"].mean()
                    / group_returns["group_0"].std()
                    if group_returns["group_0"].std() != 0
                    else np.nan,
                    f"group_{n_groups - 1}_sharpe": group_returns[
                        f"group_{n_groups - 1}"
                    ].mean()
                    / group_returns[f"group_{n_groups - 1}"].std()
                    if group_returns[f"group_{n_groups - 1}"].std() != 0
                    else np.nan,
                    "long_short_sharpe": group_returns.get(
                        "long_short", pd.Series([np.nan])
                    ).mean()
                    / group_returns.get("long_short", pd.Series([np.nan])).std()
                    if group_returns.get("long_short", pd.Series([np.nan])).std() != 0
                    else np.nan,
                }
            )

        # 换手率
        try:
            turnover = self.calculate_turnover(n_groups=n_groups, freq=freq)
            metrics["avg_turnover"] = turnover.mean()
            metrics["max_turnover"] = turnover.max()
        except:
            metrics["avg_turnover"] = np.nan
            metrics["max_turnover"] = np.nan

        return metrics

    @staticmethod
    def batch_calculate_metrics(
        factor_data: pd.DataFrame,
        forward_returns: pd.Series,
        n_groups: int = 5,
        freq: str = "D",
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        批量计算多个因子的所有指标

        Parameters
        ----------
        factor_data : pd.DataFrame
            因子数据
        forward_returns : pd.Series
            未来收益数据
        n_groups : int
            分组数量
        freq : str
            计算频率
        method : str
            IC计算方法

        Returns
        -------
        pd.DataFrame
            每个因子的指标数据
        """
        results = {}

        for factor_name in factor_data.columns:
            calculator = FactorMetrics(factor_data[factor_name], forward_returns)
            metrics = calculator.calculate_all_metrics(
                n_groups=n_groups, freq=freq, method=method
            )
            results[factor_name] = metrics

        return pd.DataFrame(results).T
