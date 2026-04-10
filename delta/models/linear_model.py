from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseModel


class LinearModel(BaseModel):
    """线性模型实现（Logistic回归）"""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)

        # 默认参数 - 更新以兼容sklearn 1.8+
        default_params = {
            "C": 1.0,
            "l1_ratio": 0,  # 0表示l2正则化，1表示l1正则化
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
        }

        # 合并参数
        self.model_params = {**default_params, **(params or {})}

        # 创建pipeline：缺失值填充 + 标准化 + Logistic回归
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),  # 用均值填充缺失值
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**self.model_params)),
            ]
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ):
        """训练线性模型"""
        # 线性模型通常不使用验证集进行early stopping
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测类别"""
        return pd.Series(self.pipeline.predict(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """预测概率"""
        proba = self.pipeline.predict_proba(X)
        return pd.DataFrame(proba, columns=[0, 1], index=X.index)

    def save(self, filename: str):
        """保存模型"""
        joblib.dump(self.pipeline, filename)

    def load(self, filename: str):
        """加载模型"""
        self.pipeline = joblib.load(filename)

    def get_feature_importance(self) -> pd.Series:
        """获取特征系数（重要性）"""
        if not hasattr(self.pipeline.named_steps["classifier"], "coef_"):
            return pd.Series()

        coef = self.pipeline.named_steps["classifier"].coef_[0]
        feature_names = (
            self.pipeline.feature_names_in_
            if hasattr(self.pipeline, "feature_names_in_")
            else None
        )

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        # 取绝对值作为重要性
        importance = np.abs(coef)
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

    def get_coefficients(self) -> pd.Series:
        """获取原始系数（带符号）"""
        if not hasattr(self.pipeline.named_steps["classifier"], "coef_"):
            return pd.Series()

        coef = self.pipeline.named_steps["classifier"].coef_[0]
        feature_names = (
            self.pipeline.feature_names_in_
            if hasattr(self.pipeline, "feature_names_in_")
            else None
        )

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        return pd.Series(coef, index=feature_names).sort_values(ascending=False)
