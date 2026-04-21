import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseModel
import math
from sklearn.metrics import precision_recall_curve


class XGBoostModel(BaseModel):
    """XGBoost模型实现"""

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)

        # 处理可能的元组情况（Jupyter notebook中字典末尾的逗号会创建元组）
        if (
            isinstance(params, tuple)
            and len(params) == 1
            and isinstance(params[0], dict)
        ):
            params = params[0]

        # 默认参数
        default_params = {
            "n_estimators": 2000,
            "max_depth": 3,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": 1,
        }

        # 合并参数
        self.model_params = {**default_params, **(params or {})}

        # 提取 beta 参数作为实例属性
        self.beta = self.model_params.pop("beta", 0.5)

        self.best_threshold = 0.5
        self.feature_names = None  # 存储特征名称

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """训练模型 - 兼容性增强版"""
        # 1. 计算权重
        self.model_params["scale_pos_weight"] = self._calculate_scale_pos_weight(
            y_train
        )

        self.model = xgb.XGBClassifier(**self.model_params)

        eval_set = [(X_valid, y_valid)]
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=100,
        )
        self._optimize_threshold(X_valid, y_valid)

        return self

    def predict(self, X):
        """使用优化后的阈值进行预测"""
        if self.model is None:
            raise ValueError("模型未训练")

        # 获取概率
        y_proba = self.model.predict_proba(X)[:, 1]
        # 根据最优阈值进行二分类
        y_pred = (y_proba >= self.best_threshold).astype(int)

        # 根据输入类型返回相应格式
        if isinstance(X, pd.DataFrame):
            return pd.Series(y_pred, index=X.index)
        else:
            # NumPy 数组输入，返回数组
            return y_pred

    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        proba = self.model.predict_proba(X)

        # 根据输入类型返回相应格式
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(proba, columns=[0, 1], index=X.index)
        else:
            # NumPy 数组输入，返回数组
            return proba

    def save(self, filename: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        joblib.dump(self.model, filename)

    def load(self, filename: str):
        """加载模型"""
        self.model = joblib.load(filename)

    def set_feature_names(self, feature_names):
        """设置特征名称"""
        self.feature_names = feature_names

    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性（默认使用gain）"""
        if self.model is None:
            return pd.Series()

        importance = self.model.feature_importances_

        # 使用存储的特征名称或从模型获取
        if self.feature_names is not None and len(self.feature_names) == len(
            importance
        ):
            feature_names = self.feature_names
        else:
            feature_names = self.model.get_booster().feature_names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance))]

        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

    def get_xgboost_importance(self) -> pd.DataFrame:
        """获取XGBoost的三种特征重要性（gain, weight, cover）"""
        if self.model is None:
            return pd.DataFrame()

        # 获取booster
        booster = self.model.get_booster()

        # 获取三种重要性
        importance_gain = booster.get_score(importance_type="gain")
        importance_weight = booster.get_score(importance_type="weight")
        importance_cover = booster.get_score(importance_type="cover")

        # 使用存储的特征名称或从模型获取
        if self.feature_names is not None:
            all_features = self.feature_names
        else:
            all_features = booster.feature_names
            if all_features is None:
                all_features = [
                    f"feature_{i}" for i in range(self.model.n_features_in_)
                ]

        # 创建DataFrame
        importance_data = []
        for feature in all_features:
            # 特征名称在XGBoost中可能以"f"开头
            xgb_feature = (
                f"f{all_features.index(feature)}"
                if feature in all_features
                else feature
            )

            importance_data.append(
                {
                    "feature": feature,
                    "gain": importance_gain.get(xgb_feature, 0.0),
                    "weight": importance_weight.get(xgb_feature, 0.0),
                    "cover": importance_cover.get(xgb_feature, 0.0),
                }
            )

        return pd.DataFrame(importance_data)

    def _calculate_scale_pos_weight(self, y_train: pd.Series) -> float:
        """
        采用平方根缩放，比单纯的 neg/pos 更稳健
        """
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        if pos_count > 0:
            return  neg_count / pos_count             # np.sqrt(neg_count / pos_count)
        return 1.0

    def _optimize_threshold(self, X_valid, y_valid):
        """
        在验证集上寻找最优阈值，极大化 F-beta score
        """
        y_proba = self.model.predict_proba(X_valid)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_valid, y_proba)

        # 计算 F-beta Score (beta=1 是 F1, beta>1 偏向召回率)
        f_scores = (
            (1 + self.beta**2)
            * (precisions * recalls)
            / ((self.beta**2 * precisions) + recalls + 1e-8)
        )

        best_idx = np.argmax(f_scores)
        # 最后一个 threshold 之后没有对应的 precision/recall，需要处理 index
        self.best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]

        print(
            f"阈值优化完成: Best Threshold={self.best_threshold:.4f}, F{self.beta}={f_scores[best_idx]:.4f}"
        )
