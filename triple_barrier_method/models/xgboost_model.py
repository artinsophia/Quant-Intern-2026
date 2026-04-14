import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base import BaseModel
import math
from sklearn.metrics import precision_recall_curve, fbeta_score
from sklearn.preprocessing import LabelEncoder


class XGBoostModel(BaseModel):
    """XGBoost模型实现 - 支持二分类和多分类"""

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
            "objective": "multi:softmax",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 1,
        }

        # 合并参数
        self.model_params = {**default_params, **(params or {})}

        # 提取 beta 参数作为实例属性
        self.beta = self.model_params.pop("beta", 0.8)

        self.best_threshold = 0.5
        self.num_class = 3

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """训练模型"""
        unique_labels = np.unique(y_train)
        self.num_class = len(unique_labels)

        if self.num_class == 2:
            self.model_params["objective"] = "binary:logistic"
            self.model_params["scale_pos_weight"] = self._calculate_scale_pos_weight(
                y_train
            )
        else:
            self.model_params["objective"] = "multi:softmax"
            self.model_params["num_class"] = self.num_class

        y_train_encoded = y_train
        if self.num_class > 2:
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            self.label_encoder = le

        early_stop_params = {}
        if self.early_stopping and X_valid is not None and y_valid is not None:
            early_stop_params = {
                "early_stopping_rounds": self.early_stopping.patience,
                "eval_metric": self._get_xgboost_eval_metric(),
            }
            self.early_stopping.reset()

        self.model = xgb.XGBClassifier(**self.model_params)

        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]
            self.model.fit(
                X_train,
                y_train_encoded,
                eval_set=eval_set,
                verbose=False,
                **early_stop_params,
            )
        else:
            self.model.fit(X_train, y_train_encoded, verbose=True)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测类别"""
        if self.model is None:
            raise ValueError("模型未训练")

        y_pred_encoded = self.model.predict(X)

        if hasattr(self, "label_encoder"):
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded.astype(int))
        else:
            y_pred = y_pred_encoded

        return pd.Series(y_pred, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        proba = self.model.predict_proba(X)
        if self.num_class == 2:
            return pd.DataFrame(proba, columns=[0, 1], index=X.index)
        else:
            return pd.DataFrame(proba, index=X.index)

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
        """
        采用平方根缩放，比单纯的 neg/pos 更稳健
        """
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        if pos_count > 0:
            # 使用平方根平滑。例如 1:100 的比例，权重为 10 而不是 100
            return math.sqrt(neg_count / pos_count)
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

    def _get_xgboost_eval_metric(self) -> str:
        """将早停监控指标转换为XGBoost评估指标"""
        if not self.early_stopping:
            return "error"

        monitor = self.early_stopping.monitor

        # 移除前缀（如 "validation_0-"）
        if "validation_0-" in monitor:
            metric = monitor.replace("validation_0-", "")
        elif "validation-" in monitor:
            metric = monitor.replace("validation-", "")
        else:
            metric = monitor

        # 映射到XGBoost支持的指标
        metric_map = {
            "error": "error",
            "logloss": "logloss",
            "auc": "auc",
            "merror": "merror",
            "mlogloss": "mlogloss",
            "mae": "mae",
            "mse": "mse",
            "rmse": "rmse",
            "mape": "mape",
        }

        return metric_map.get(metric, "error")
