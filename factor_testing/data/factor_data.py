import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings
import os


class FactorData:
    """
    因子数据管理类
    负责加载、存储、预处理因子数据
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        初始化因子数据

        Parameters
        ----------
        data : pd.DataFrame, optional
            因子数据，格式为 MultiIndex (date, symbol) 或 (date)
            列名为因子名称
        """
        self.data = data
        self.factor_names = []
        self.dates = []
        self.symbols = []

        if data is not None:
            self._validate_and_setup(data)

    def _validate_and_setup(self, data: pd.DataFrame):
        """验证数据格式并设置属性"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # 检查索引格式
        if isinstance(data.index, pd.MultiIndex):
            if len(data.index.names) == 2:
                # (date, symbol) 格式
                self.dates = sorted(data.index.get_level_values(0).unique())
                self.symbols = sorted(data.index.get_level_values(1).unique())
            else:
                raise ValueError(
                    "MultiIndex must have exactly 2 levels: (date, symbol)"
                )
        else:
            # 单索引，假设为日期索引
            self.dates = sorted(data.index.unique())
            self.symbols = []

        self.factor_names = list(data.columns)
        self.data = data

    def load_from_csv(
        self,
        filepath: str,
        date_col: str = "date",
        symbol_col: Optional[str] = "symbol",
        factor_cols: Optional[List[str]] = None,
        index_cols: Optional[List[str]] = None,
    ):
        """
        从CSV文件加载因子数据

        Parameters
        ----------
        filepath : str
            CSV文件路径
        date_col : str
            日期列名
        symbol_col : str, optional
            标的列名，如果为None则假设单标的
        factor_cols : List[str], optional
            因子列名列表，如果为None则自动检测
        index_cols : List[str], optional
            索引列名列表，如果为None则使用date_col和symbol_col
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        # 确定因子列
        if factor_cols is None:
            # 排除索引列
            exclude_cols = [date_col]
            if symbol_col is not None:
                exclude_cols.append(symbol_col)
            factor_cols = [col for col in df.columns if col not in exclude_cols]

        # 设置索引
        if index_cols is None:
            index_cols = [date_col]
            if symbol_col is not None:
                index_cols.append(symbol_col)

        df = df.set_index(index_cols)

        # 只保留因子列
        df = df[factor_cols]

        self._validate_and_setup(df)
        return self

    def load_from_dict(self, data_dict: Dict[str, pd.DataFrame]):
        """
        从字典加载因子数据

        Parameters
        ----------
        data_dict : Dict[str, pd.DataFrame]
            字典格式：{factor_name: DataFrame}
            DataFrame索引为日期，列为标的
        """
        if not data_dict:
            raise ValueError("data_dict cannot be empty")

        # 获取所有因子名称
        factor_names = list(data_dict.keys())

        # 验证所有DataFrame的形状一致
        first_factor = list(data_dict.values())[0]
        dates = first_factor.index
        symbols = first_factor.columns

        for factor_name, df in data_dict.items():
            if not (df.index.equals(dates) and df.columns.equals(symbols)):
                raise ValueError(f"Factor {factor_name} has inconsistent shape")

        # 转换为MultiIndex格式
        data_list = []
        for factor_name in factor_names:
            df = data_dict[factor_name]
            # 堆叠数据
            stacked = df.stack()
            stacked.name = factor_name
            data_list.append(stacked)

        # 合并所有因子
        result = pd.concat(data_list, axis=1)

        self._validate_and_setup(result)
        return self

    def get_factor(self, factor_name: str) -> pd.Series:
        """
        获取单个因子数据

        Parameters
        ----------
        factor_name : str
            因子名称

        Returns
        -------
        pd.Series
            因子数据
        """
        if factor_name not in self.factor_names:
            raise ValueError(f"Factor {factor_name} not found")

        return self.data[factor_name]

    def get_factors(self, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取多个因子数据

        Parameters
        ----------
        factor_names : List[str], optional
            因子名称列表，如果为None则返回所有因子

        Returns
        -------
        pd.DataFrame
            因子数据
        """
        if factor_names is None:
            return self.data
        else:
            missing = [name for name in factor_names if name not in self.factor_names]
            if missing:
                raise ValueError(f"Factors not found: {missing}")
            return self.data[factor_names]

    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """获取日期范围"""
        if not self.dates:
            raise ValueError("No dates available")
        return min(self.dates), max(self.dates)

    def get_factor_stats(self, factor_name: str) -> Dict:
        """
        获取因子基本统计信息

        Parameters
        ----------
        factor_name : str
            因子名称

        Returns
        -------
        Dict
            统计信息字典
        """
        factor_data = self.get_factor(factor_name)

        stats = {
            "mean": factor_data.mean(),
            "std": factor_data.std(),
            "min": factor_data.min(),
            "max": factor_data.max(),
            "skew": factor_data.skew(),
            "kurtosis": factor_data.kurtosis(),
            "count": factor_data.count(),
            "missing_rate": factor_data.isna().sum() / len(factor_data),
        }

        return stats

    def add_factor(self, factor_name: str, factor_data: pd.Series):
        """
        添加新因子

        Parameters
        ----------
        factor_name : str
            因子名称
        factor_data : pd.Series
            因子数据，必须与现有数据索引一致
        """
        if factor_name in self.factor_names:
            warnings.warn(f"Factor {factor_name} already exists, overwriting")

        # 验证索引一致性
        if not factor_data.index.equals(self.data.index):
            raise ValueError("factor_data must have same index as existing data")

        self.data[factor_name] = factor_data
        if factor_name not in self.factor_names:
            self.factor_names.append(factor_name)

    def remove_factor(self, factor_name: str):
        """
        移除因子

        Parameters
        ----------
        factor_name : str
            因子名称
        """
        if factor_name not in self.factor_names:
            raise ValueError(f"Factor {factor_name} not found")

        self.data = self.data.drop(columns=[factor_name])
        self.factor_names.remove(factor_name)

    def to_panel(self, factor_name: str) -> pd.DataFrame:
        """
        将因子数据转换为面板格式 (date × symbol)

        Parameters
        ----------
        factor_name : str
            因子名称

        Returns
        -------
        pd.DataFrame
            面板格式数据
        """
        factor_data = self.get_factor(factor_name)

        if isinstance(factor_data.index, pd.MultiIndex):
            # 已经是MultiIndex，直接unstack
            return factor_data.unstack()
        else:
            # 单索引，假设为日期
            return pd.DataFrame({factor_name: factor_data})

    def save_to_csv(self, filepath: str):
        """
        保存因子数据到CSV文件

        Parameters
        ----------
        filepath : str
            保存路径
        """
        self.data.reset_index().to_csv(filepath, index=False)

    def __repr__(self) -> str:
        return (
            f"FactorData(factors={len(self.factor_names)}, "
            f"dates={len(self.dates)}, symbols={len(self.symbols)})"
        )
