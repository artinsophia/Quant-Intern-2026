import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from scipy import stats
import warnings


class FactorPreprocessor:
    """
    因子数据预处理类
    提供因子标准化、去极值、中性化等预处理功能
    """

    @staticmethod
    def winsorize(
        factor_data: pd.Series,
        method: str = "quantile",
        limits: Union[float, Tuple[float, float]] = 0.05,
        groupby: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        去极值处理

        Parameters
        ----------
        factor_data : pd.Series
            因子数据
        method : str, {'quantile', 'sigma', 'mad'}
            去极值方法：
            - 'quantile': 分位数法
            - 'sigma': 标准差法
            - 'mad': 中位数绝对偏差法
        limits : float or tuple
            极值限制，对于quantile为分位数，对于sigma/mad为倍数
        groupby : pd.Series, optional
            分组序列，用于分组去极值

        Returns
        -------
        pd.Series
            去极值后的因子数据
        """
        if method not in ["quantile", "sigma", "mad"]:
            raise ValueError("method must be one of 'quantile', 'sigma', 'mad'")

        if groupby is None:
            # 整体去极值
            return FactorPreprocessor._winsorize_single(factor_data, method, limits)
        else:
            # 分组去极值
            if not factor_data.index.equals(groupby.index):
                raise ValueError("factor_data and groupby must have same index")

            result = pd.Series(index=factor_data.index, dtype=float)
            for group in groupby.unique():
                mask = groupby == group
                group_data = factor_data[mask]
                if len(group_data) > 0:
                    result[mask] = FactorPreprocessor._winsorize_single(
                        group_data, method, limits
                    )

            return result

    @staticmethod
    def _winsorize_single(
        data: pd.Series, method: str, limits: Union[float, Tuple[float, float]]
    ) -> pd.Series:
        """单个序列去极值"""
        if method == "quantile":
            if isinstance(limits, (int, float)):
                limits = (limits, limits)
            lower_limit = data.quantile(limits[0])
            upper_limit = data.quantile(1 - limits[1])

        elif method == "sigma":
            if isinstance(limits, (int, float)):
                limits = (limits, limits)
            mean = data.mean()
            std = data.std()
            lower_limit = mean - limits[0] * std
            upper_limit = mean + limits[1] * std

        elif method == "mad":
            if isinstance(limits, (int, float)):
                limits = (limits, limits)
            median = data.median()
            mad = stats.median_abs_deviation(data.dropna(), scale="normal")
            lower_limit = median - limits[0] * mad
            upper_limit = median + limits[1] * mad

        # 应用极值限制
        result = data.copy()
        result[result < lower_limit] = lower_limit
        result[result > upper_limit] = upper_limit

        return result

    @staticmethod
    def standardize(
        factor_data: pd.Series,
        method: str = "zscore",
        groupby: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        标准化处理

        Parameters
        ----------
        factor_data : pd.Series
            因子数据
        method : str, {'zscore', 'rank', 'minmax'}
            标准化方法：
            - 'zscore': Z-score标准化
            - 'rank': 排名标准化
            - 'minmax': 最小最大标准化
        groupby : pd.Series, optional
            分组序列，用于分组标准化

        Returns
        -------
        pd.Series
            标准化后的因子数据
        """
        if method not in ["zscore", "rank", "minmax"]:
            raise ValueError("method must be one of 'zscore', 'rank', 'minmax'")

        if groupby is None:
            # 整体标准化
            return FactorPreprocessor._standardize_single(factor_data, method)
        else:
            # 分组标准化
            if not factor_data.index.equals(groupby.index):
                raise ValueError("factor_data and groupby must have same index")

            result = pd.Series(index=factor_data.index, dtype=float)
            for group in groupby.unique():
                mask = groupby == group
                group_data = factor_data[mask]
                if len(group_data) > 0:
                    result[mask] = FactorPreprocessor._standardize_single(
                        group_data, method
                    )

            return result

    @staticmethod
    def _standardize_single(data: pd.Series, method: str) -> pd.Series:
        """单个序列标准化"""
        if method == "zscore":
            mean = data.mean()
            std = data.std()
            if std == 0:
                return pd.Series(0, index=data.index)
            return (data - mean) / std

        elif method == "rank":
            # 排名标准化到[-1, 1]
            ranks = data.rank()
            n = len(ranks)
            if n <= 1:
                return pd.Series(0, index=data.index)
            return 2 * (ranks - 1) / (n - 1) - 1

        elif method == "minmax":
            min_val = data.min()
            max_val = data.max()
            if max_val == min_val:
                return pd.Series(0, index=data.index)
            return 2 * (data - min_val) / (max_val - min_val) - 1

    @staticmethod
    def neutralize(
        factor_data: pd.Series, exposure_data: pd.DataFrame, method: str = "linear"
    ) -> pd.Series:
        """
        因子中性化处理

        Parameters
        ----------
        factor_data : pd.Series
            因子数据
        exposure_data : pd.DataFrame
            暴露度数据，索引必须与factor_data一致
        method : str, {'linear', 'rank'}
            中性化方法：
            - 'linear': 线性回归残差
            - 'rank': 排名回归残差

        Returns
        -------
        pd.Series
            中性化后的因子数据
        """
        if not factor_data.index.equals(exposure_data.index):
            raise ValueError("factor_data and exposure_data must have same index")

        # 移除缺失值
        valid_mask = factor_data.notna() & exposure_data.notna().all(axis=1)
        if not valid_mask.any():
            return pd.Series(np.nan, index=factor_data.index)

        factor_valid = factor_data[valid_mask]
        exposure_valid = exposure_data[valid_mask]

        if method == "linear":
            # 线性回归中性化
            from sklearn.linear_model import LinearRegression

            X = exposure_valid.values
            y = factor_valid.values

            if len(y) <= X.shape[1]:
                warnings.warn(
                    "Not enough samples for regression, returning original factor"
                )
                return factor_data

            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)

        elif method == "rank":
            # 排名回归中性化
            from sklearn.linear_model import LinearRegression

            # 对因子和暴露度进行排名
            factor_rank = factor_valid.rank()
            exposure_rank = exposure_valid.rank(axis=0)

            X = exposure_rank.values
            y = factor_rank.values

            if len(y) <= X.shape[1]:
                warnings.warn(
                    "Not enough samples for regression, returning original factor"
                )
                return factor_data

            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)

        # 创建结果序列
        result = pd.Series(index=factor_data.index, dtype=float)
        result[valid_mask] = residuals

        return result

    @staticmethod
    def fill_missing(
        factor_data: pd.Series,
        method: str = "mean",
        groupby: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        缺失值填充

        Parameters
        ----------
        factor_data : pd.Series
            因子数据
        method : str, {'mean', 'median', 'zero', 'ffill', 'bfill'}
            填充方法
        groupby : pd.Series, optional
            分组序列，用于分组填充

        Returns
        -------
        pd.Series
            填充后的因子数据
        """
        if method not in ["mean", "median", "zero", "ffill", "bfill"]:
            raise ValueError(
                "method must be one of 'mean', 'median', 'zero', 'ffill', 'bfill'"
            )

        if groupby is None:
            # 整体填充
            return FactorPreprocessor._fill_missing_single(factor_data, method)
        else:
            # 分组填充
            if not factor_data.index.equals(groupby.index):
                raise ValueError("factor_data and groupby must have same index")

            result = factor_data.copy()
            for group in groupby.unique():
                mask = groupby == group
                group_data = factor_data[mask]
                if len(group_data) > 0:
                    result[mask] = FactorPreprocessor._fill_missing_single(
                        group_data, method
                    )

            return result

    @staticmethod
    def _fill_missing_single(data: pd.Series, method: str) -> pd.Series:
        """单个序列缺失值填充"""
        if method == "mean":
            fill_value = data.mean()
        elif method == "median":
            fill_value = data.median()
        elif method == "zero":
            fill_value = 0
        elif method == "ffill":
            return data.ffill()
        elif method == "bfill":
            return data.bfill()

        return data.fillna(fill_value)

    @staticmethod
    def pipeline(factor_data: pd.Series, steps: List[Dict]) -> pd.Series:
        """
        预处理流水线

        Parameters
        ----------
        factor_data : pd.Series
            原始因子数据
        steps : List[Dict]
            预处理步骤列表，每个步骤为包含'name'和'params'的字典

        Returns
        -------
        pd.Series
            预处理后的因子数据
        """
        result = factor_data.copy()

        for step in steps:
            name = step["name"]
            params = step.get("params", {})

            if name == "winsorize":
                result = FactorPreprocessor.winsorize(result, **params)
            elif name == "standardize":
                result = FactorPreprocessor.standardize(result, **params)
            elif name == "neutralize":
                result = FactorPreprocessor.neutralize(result, **params)
            elif name == "fill_missing":
                result = FactorPreprocessor.fill_missing(result, **params)
            else:
                raise ValueError(f"Unknown preprocessing step: {name}")

        return result
