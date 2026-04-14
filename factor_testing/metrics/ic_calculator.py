import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
import warnings


class ICCalculator:
    """
    IC（信息系数）计算器
    计算因子与未来收益的相关性指标
    """

    def __init__(
        self,
        factor_data: pd.Series,
        forward_returns: pd.Series,
        groupby: Optional[pd.Series] = None,
    ):
        """
        初始化IC计算器

        Parameters
        ----------
        factor_data : pd.Series
            因子数据，索引为(date, symbol)或date
        forward_returns : pd.Series
            未来收益数据，索引必须与factor_data一致
        groupby : pd.Series, optional
            分组序列，用于分组计算IC
        """
        if not factor_data.index.equals(forward_returns.index):
            raise ValueError("factor_data and forward_returns must have same index")

        self.factor_data = factor_data
        self.forward_returns = forward_returns
        self.groupby = groupby

        # 移除缺失值
        self.valid_mask = factor_data.notna() & forward_returns.notna()
        if self.groupby is not None:
            self.valid_mask = self.valid_mask & self.groupby.notna()

        self.factor_valid = factor_data[self.valid_mask]
        self.returns_valid = forward_returns[self.valid_mask]

        if self.groupby is not None:
            self.groupby_valid = self.groupby[self.valid_mask]

    def calculate_ic(
        self, method: str = "pearson", group_ic: bool = False
    ) -> Union[float, pd.Series]:
        """
        计算IC值

        Parameters
        ----------
        method : str, {'pearson', 'spearman', 'kendall'}
            相关性计算方法
        group_ic : bool
            是否计算分组IC

        Returns
        -------
        float or pd.Series
            IC值，如果group_ic=True则返回每个分组的IC
        """
        if method not in ["pearson", "spearman", "kendall"]:
            raise ValueError("method must be one of 'pearson', 'spearman', 'kendall'")

        if group_ic:
            if self.groupby is None:
                raise ValueError("groupby must be provided for group IC calculation")

            return self._calculate_group_ic(method)
        else:
            return self._calculate_overall_ic(method)

    def _calculate_overall_ic(self, method: str) -> float:
        """计算整体IC"""
        if len(self.factor_valid) < 2:
            return np.nan

        if method == "pearson":
            ic = stats.pearsonr(self.factor_valid, self.returns_valid)[0]
        elif method == "spearman":
            ic = stats.spearmanr(self.factor_valid, self.returns_valid)[0]
        elif method == "kendall":
            ic = stats.kendalltau(self.factor_valid, self.returns_valid)[0]

        return ic

    def _calculate_group_ic(self, method: str) -> pd.Series:
        """计算分组IC"""
        group_ics = {}

        for group in self.groupby_valid.unique():
            mask = self.groupby_valid == group
            group_factor = self.factor_valid[mask]
            group_returns = self.returns_valid[mask]

            if len(group_factor) < 2:
                group_ics[group] = np.nan
                continue

            if method == "pearson":
                ic = stats.pearsonr(group_factor, group_returns)[0]
            elif method == "spearman":
                ic = stats.spearmanr(group_factor, group_returns)[0]
            elif method == "kendall":
                ic = stats.kendalltau(group_factor, group_returns)[0]

            group_ics[group] = ic

        return pd.Series(group_ics)

    def calculate_rank_ic(self, group_ic: bool = False) -> Union[float, pd.Series]:
        """
        计算Rank IC（Spearman相关系数）

        Parameters
        ----------
        group_ic : bool
            是否计算分组Rank IC

        Returns
        -------
        float or pd.Series
            Rank IC值
        """
        return self.calculate_ic(method="spearman", group_ic=group_ic)

    def calculate_ic_series(
        self, freq: str = "D", method: str = "pearson"
    ) -> pd.Series:
        """
        计算时间序列IC

        Parameters
        ----------
        freq : str
            时间频率，如'D'（日）、'W'（周）、'M'（月）
        method : str
            相关性计算方法

        Returns
        -------
        pd.Series
            时间序列IC，索引为日期
        """
        # 提取日期信息
        if isinstance(self.factor_data.index, pd.MultiIndex):
            dates = self.factor_data.index.get_level_values(0)[self.valid_mask]
        else:
            dates = self.factor_data.index[self.valid_mask]

        # 按频率分组
        if freq == "D":
            # 日频，直接按日期分组
            date_groups = dates
        else:
            # 其他频率，需要重采样
            date_series = pd.Series(index=dates, data=dates)
            date_groups = date_series.resample(freq).asfreq().index

        # 计算每个时间段的IC
        ic_series = {}

        for period in date_groups.unique():
            if pd.isna(period):
                continue

            if freq == "D":
                mask = dates == period
            else:
                # 对于非日频，需要找到属于该时间段的所有日期
                if isinstance(period, pd.Period):
                    period_start = period.start_time
                    period_end = period.end_time
                else:
                    # 假设period是时间戳
                    period_start = period
                    if freq == "W":
                        period_end = period_start + pd.Timedelta(days=7)
                    elif freq == "M":
                        period_end = period_start + pd.offsets.MonthEnd()
                    else:
                        period_end = period_start

                mask = (dates >= period_start) & (dates < period_end)

            period_factor = self.factor_valid[mask]
            period_returns = self.returns_valid[mask]

            if len(period_factor) < 2:
                ic_series[period] = np.nan
                continue

            if method == "pearson":
                ic = stats.pearsonr(period_factor, period_returns)[0]
            elif method == "spearman":
                ic = stats.spearmanr(period_factor, period_returns)[0]
            elif method == "kendall":
                ic = stats.kendalltau(period_factor, period_returns)[0]

            ic_series[period] = ic

        return pd.Series(ic_series).sort_index()

    def calculate_ic_decay(
        self, max_lag: int = 10, method: str = "pearson"
    ) -> pd.Series:
        """
        计算IC衰减

        Parameters
        ----------
        max_lag : int
            最大滞后阶数
        method : str
            相关性计算方法

        Returns
        -------
        pd.Series
            IC衰减序列，索引为滞后阶数
        """
        # 需要按日期对齐数据
        if not isinstance(self.factor_data.index, pd.MultiIndex):
            raise ValueError("IC decay calculation requires MultiIndex (date, symbol)")

        # 获取日期和标的
        dates = sorted(self.factor_data.index.get_level_values(0).unique())
        symbols = sorted(self.factor_data.index.get_level_values(1).unique())

        # 转换为面板数据
        factor_panel = self.factor_data.unstack()
        returns_panel = self.forward_returns.unstack()

        # 计算不同滞后阶数的IC
        ic_decay = {}

        for lag in range(1, max_lag + 1):
            # 对齐因子和滞后收益
            aligned_factor = factor_panel.iloc[:-lag] if lag > 0 else factor_panel
            aligned_returns = returns_panel.iloc[lag:] if lag > 0 else returns_panel

            # 确保日期对齐
            common_dates = aligned_factor.index.intersection(aligned_returns.index)
            if len(common_dates) == 0:
                ic_decay[lag] = np.nan
                continue

            aligned_factor = aligned_factor.loc[common_dates]
            aligned_returns = aligned_returns.loc[common_dates]

            # 展平数据
            factor_flat = aligned_factor.stack()
            returns_flat = aligned_returns.stack()

            # 移除缺失值
            valid_mask = factor_flat.notna() & returns_flat.notna()
            factor_valid = factor_flat[valid_mask]
            returns_valid = returns_flat[valid_mask]

            if len(factor_valid) < 2:
                ic_decay[lag] = np.nan
                continue

            # 计算IC
            if method == "pearson":
                ic = stats.pearsonr(factor_valid, returns_valid)[0]
            elif method == "spearman":
                ic = stats.spearmanr(factor_valid, returns_valid)[0]
            elif method == "kendall":
                ic = stats.kendalltau(factor_valid, returns_valid)[0]

            ic_decay[lag] = ic

        return pd.Series(ic_decay)

    def calculate_ic_stats(self, method: str = "pearson", freq: str = "D") -> Dict:
        """
        计算IC统计指标

        Parameters
        ----------
        method : str
            相关性计算方法
        freq : str
            时间序列IC的频率

        Returns
        -------
        Dict
            IC统计指标
        """
        # 整体IC
        overall_ic = self.calculate_ic(method=method)

        # 时间序列IC
        ic_series = self.calculate_ic_series(freq=freq, method=method)
        ic_series_valid = ic_series.dropna()

        # 统计指标
        stats_dict = {
            "ic": overall_ic,
            "ic_mean": ic_series_valid.mean(),
            "ic_std": ic_series_valid.std(),
            "ic_ir": ic_series_valid.mean() / ic_series_valid.std()
            if ic_series_valid.std() != 0
            else np.nan,
            "ic_positive_rate": (ic_series_valid > 0).mean(),
            "ic_skew": ic_series_valid.skew(),
            "ic_kurtosis": ic_series_valid.kurtosis(),
            "ic_min": ic_series_valid.min(),
            "ic_max": ic_series_valid.max(),
            "ic_t_stat": overall_ic * np.sqrt(len(ic_series_valid))
            if not np.isnan(overall_ic)
            else np.nan,
            "ic_p_value": stats.ttest_1samp(ic_series_valid, 0).pvalue
            if len(ic_series_valid) > 1
            else np.nan,
        }

        return stats_dict

    @staticmethod
    def batch_calculate_ic(
        factor_data: pd.DataFrame, forward_returns: pd.Series, method: str = "pearson"
    ) -> pd.Series:
        """
        批量计算多个因子的IC

        Parameters
        ----------
        factor_data : pd.DataFrame
            因子数据，每列为一个因子
        forward_returns : pd.Series
            未来收益数据
        method : str
            相关性计算方法

        Returns
        -------
        pd.Series
            每个因子的IC值
        """
        if not factor_data.index.equals(forward_returns.index):
            raise ValueError("factor_data and forward_returns must have same index")

        ic_results = {}

        for factor_name in factor_data.columns:
            calculator = ICCalculator(factor_data[factor_name], forward_returns)
            ic = calculator.calculate_ic(method=method)
            ic_results[factor_name] = ic

        return pd.Series(ic_results)
