from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from .early_stopping import create_early_stopping, EarlyStopping


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

        # 早停机制相关
        self.early_stopping = None
        self._setup_early_stopping()

    def _setup_early_stopping(self):
        """设置早停机制"""
        early_stopping_params = self.params.get("early_stopping")
        if early_stopping_params:
            # 获取模型类型
            model_type = self.__class__.__name__.lower().replace("model", "")
            if model_type == "xgboost":
                model_type = "xgboost"
            elif model_type == "linear":
                model_type = "linear"
            elif model_type == "ensemble":
                model_type = "ensemble"
            else:
                model_type = "generic"

            # 创建早停机制
            self.early_stopping = create_early_stopping(
                model_type, early_stopping_params
            )

            # 从参数中移除早停参数，避免传递给底层模型
            if "early_stopping" in self.params:
                del self.params["early_stopping"]

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
        params = self.params.copy()

        # 如果有早停机制，将其参数添加回去
        if self.early_stopping:
            early_stopping_params = {
                "patience": self.early_stopping.patience,
                "min_delta": self.early_stopping.min_delta,
                "mode": self.early_stopping.mode,
                "monitor": self.early_stopping.monitor,
                "baseline": self.early_stopping.baseline,
                "restore_best_weights": self.early_stopping.restore_best_weights,
                "verbose": self.early_stopping.verbose,
            }
            params["early_stopping"] = early_stopping_params

        return params

    def get_early_stopping_info(self) -> Optional[Dict[str, Any]]:
        """获取早停机制信息"""
        if self.early_stopping:
            return self.early_stopping.get_summary()
        return None

    def reset_early_stopping(self):
        """重置早停状态"""
        if self.early_stopping:
            self.early_stopping.reset()
