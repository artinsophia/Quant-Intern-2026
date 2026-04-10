import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseModel
import math

class XGBoostModel(BaseModel):
    """XGBoost模型实现"""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)

        # 默认参数
        default_params = {
            "n_estimators": 2000,
            "max_depth": 3,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 1,
        }

        # 合并参数
        self.model_params = {**default_params, **(params or {})}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ):
        """训练XGBoost模型"""

        # 计算类别权重
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        self.model_params["scale_pos_weight"] = scale_pos_weight

        # 创建模型
        self.model = xgb.XGBClassifier(**self.model_params)

        # 训练
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        else:
            self.model.fit(X_train, y_train, verbose=True)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测类别"""
        if self.model is None:
            raise ValueError("模型未训练")
        return pd.Series(self.model.predict(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        proba = self.model.predict_proba(X)
        return pd.DataFrame(proba, columns=[0, 1], index=X.index)

    def save(self, filename: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        joblib.dump(self.model, filename)

    def load(self, filename: str):
        """加载模型"""
        self.model = joblib.load(filename)

    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性"""
        if self.model is None:
            return pd.Series()

        importance = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

    def _calculate_scale_pos_weight(self, y_train: pd.Series) -> float:
        """计算类别权重"""
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()

        if pos_count > 0:
            return math.sqrt(neg_count / pos_count)
        return 1.0
