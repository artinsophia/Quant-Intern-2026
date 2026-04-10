from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


class BaseModel(ABC):
    """基础模型接口"""

    def __init__(self, params: Dict[str, Any] = None):
        # 处理可能的元组情况（Jupyter notebook中字典末尾的逗号会创建元组）
        if (
            isinstance(params, tuple)
            and len(params) == 1
            and isinstance(params[0], dict)
        ):
            params = params[0]

        self.params = params or {}
        self.model = None

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ):
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测类别"""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """预测概率"""
        pass

    @abstractmethod
    def save(self, filename: str):
        """保存模型"""
        pass

    @abstractmethod
    def load(self, filename: str):
        """加载模型"""
        pass

    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性（可选）"""
        return pd.Series()

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.params.copy()
