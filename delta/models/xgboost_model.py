import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base import BaseModel
import math


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
            "n_jobs": -1,
            "verbosity": 1,
        }

        # 合并参数

        self.best_threshold = 0.5
        self.model_params = {**default_params, **(params or {})}

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """训练模型"""
        # 1. 直接计算稳健权重
        self.model_params["scale_pos_weight"] = self._calculate_scale_pos_weight(y_train)

        # 2. 早停设置 (保留你原来的逻辑)
        if self.early_stopping and X_valid is not None and y_valid is not None:
            self.model_params["early_stopping_rounds"] = self.early_stopping.patience
            self.model_params["eval_metric"] = self._get_xgboost_eval_metric()
            self.early_stopping.reset()

        # 3. 训练模型 (仅训练一次！)
        self.model = xgb.XGBClassifier(**self.model_params)
        
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
                callbacks=[self._xgboost_early_stopping_callback()] if self.early_stopping else None
            )
            # 4. 训练完成后，自动寻找最优阈值
            self._optimize_threshold(X_valid, y_valid, beta=1.2) # beta=1.2 稍微偏向召回率
        else:
            self.model.fit(X_train, y_train, verbose=True)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """使用优化后的阈值进行预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 获取概率
        y_proba = self.model.predict_proba(X)[:, 1]
        # 根据最优阈值进行二分类
        y_pred = (y_proba >= self.best_threshold).astype(int)
        
        return pd.Series(y_pred, index=X.index)

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
        """
        采用平方根缩放，比单纯的 neg/pos 更稳健
        """
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        if pos_count > 0:
            # 使用平方根平滑。例如 1:100 的比例，权重为 10 而不是 100
            return math.sqrt(neg_count / pos_count)
        return 1.0

    def _optimize_threshold(self, X_valid, y_valid, beta=1.0):
        """
        在验证集上寻找最优阈值，极大化 F-beta score
        """
        y_proba = self.model.predict_proba(X_valid)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_valid, y_proba)
        
        # 计算 F-beta Score (beta=1 是 F1, beta>1 偏向召回率)
        f_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-8)
        
        best_idx = np.argmax(f_scores)
        # 最后一个 threshold 之后没有对应的 precision/recall，需要处理 index
        self.best_threshold = thresholds[min(best_idx, len(thresholds)-1)]
        
        print(f"阈值优化完成: Best Threshold={self.best_threshold:.4f}, F{beta}={f_scores[best_idx]:.4f}")

    

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

    def _xgboost_early_stopping_callback(self):
        """创建XGBoost早停回调函数"""
        from xgboost.callback import EarlyStopping as XGBEarlyStopping

        # 获取早停参数
        patience = self.early_stopping.patience
        metric_name = self._get_xgboost_eval_metric()

        # 确定最大化还是最小化
        maximize = self.early_stopping.mode == "max"

        return XGBEarlyStopping(
            rounds=patience,
            metric_name=metric_name,
            maximize=maximize,
            min_delta=self.early_stopping.min_delta,
            save_best=self.early_stopping.restore_best_weights,
        )

    def get_training_history(self) -> Optional[pd.DataFrame]:
        """获取训练历史（包括早停信息）"""
        if self.model is None:
            return None

        # 获取XGBoost的训练历史
        evals_result = self.model.evals_result()
        if not evals_result:
            return None

        history_data = []
        for eval_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                for epoch, value in enumerate(values):
                    history_data.append(
                        {
                            "epoch": epoch,
                            "eval_set": eval_name,
                            "metric": metric_name,
                            "value": value,
                        }
                    )

        history_df = pd.DataFrame(history_data)

        # 添加早停信息
        if self.early_stopping and self.early_stopping.history:
            early_stop_history = self.early_stopping.get_history()
            if early_stop_history:
                # 将早停历史转换为DataFrame
                early_stop_df = pd.DataFrame(early_stop_history)
                history_df = pd.merge(
                    history_df,
                    early_stop_df,
                    left_on="epoch",
                    right_on="epoch",
                    how="left",
                )

        return history_df
