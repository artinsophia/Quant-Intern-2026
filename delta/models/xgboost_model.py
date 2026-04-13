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
        self.model_params = {**default_params, **(params or {})}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ):
        """训练XGBoost模型"""

        # 计算自适应类别权重
        scale_pos_weight = self._calculate_adaptive_scale_pos_weight(
            X_train, y_train, X_valid, y_valid
        )
        self.model_params["scale_pos_weight"] = scale_pos_weight

        # 如果有早停机制，设置XGBoost的早停参数
        if self.early_stopping and X_valid is not None and y_valid is not None:
            # 设置XGBoost的早停参数
            self.model_params["early_stopping_rounds"] = self.early_stopping.patience
            self.model_params["eval_metric"] = self._get_xgboost_eval_metric()

            # 重置早停状态
            self.early_stopping.reset()

            print(
                f"启用早停机制: patience={self.early_stopping.patience}, "
                f"monitor={self.early_stopping.monitor}"
            )

        # 创建模型
        self.model = xgb.XGBClassifier(**self.model_params)

        # 训练
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]

            # 如果有早停机制，使用自定义回调
            if self.early_stopping:
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    verbose=True,
                    callbacks=[self._xgboost_early_stopping_callback()],
                )
            else:
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
        """计算基础类别权重"""
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()

        if pos_count > 0:
            return neg_count / pos_count / 2  # math.sqrt(neg_count / pos_count)
        return 1.0

    def _calculate_adaptive_scale_pos_weight(self, X_train, y_train, X_valid, y_valid):
        """计算自适应类别权重，根据验证集表现优化

        目标：找到既能保证一定召回率（比如>0.3）又能最大化精确率的权重
        """
        # 基础权重计算
        base_weight = self._calculate_scale_pos_weight(y_train)

        # 如果没有验证集，返回基础权重
        if X_valid is None or y_valid is None:
            print(f"使用基础类别权重: {base_weight:.4f}")
            return base_weight

        # 尝试不同的权重因子
        weight_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
        best_weight = base_weight
        best_score = -1
        best_recall = 0

        print("优化类别权重...")

        for factor in weight_factors:
            current_weight = base_weight * factor

            # 创建临时模型测试权重
            temp_params = self.model_params.copy()
            temp_params["scale_pos_weight"] = current_weight
            temp_params["n_estimators"] = 1000  # 使用较少的树进行快速测试

            temp_model = xgb.XGBClassifier(**temp_params)
            temp_model.fit(X_train, y_train)

            # 在验证集上评估
            y_pred = temp_model.predict(X_valid)
            y_pred_proba = temp_model.predict_proba(X_valid)[:, 1]

            # 计算召回率和精确率
            from sklearn.metrics import recall_score, precision_score

            recall = recall_score(y_valid, y_pred)
            precision = precision_score(y_valid, y_pred, zero_division=0)

            # 计算综合得分：在保证一定召回率的前提下最大化精确率
            # 目标召回率设为0.3
            target_recall = 0.3
            if recall >= target_recall:
                # 召回率达标，使用精确率作为主要指标
                score = precision
            else:
                # 召回率不达标，惩罚得分
                score = precision * (recall / target_recall)

            print(
                f"  权重={current_weight:.4f} (因子={factor:.2f}): "
                f"召回率={recall:.4f}, 精确率={precision:.4f}, 得分={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_weight = current_weight
                best_recall = recall

        print(
            f"最佳类别权重: {best_weight:.4f} (召回率={best_recall:.4f}, 得分={best_score:.4f})"
        )
        return best_weight

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
