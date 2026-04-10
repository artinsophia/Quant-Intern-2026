import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from .base import BaseModel
import joblib


class EnsembleModel(BaseModel):
    """集成学习模型，支持软投票和硬投票"""

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
            "voting": "soft",  # "soft" 或 "hard"
            "weights": None,  # 模型权重列表，None表示等权重
            "models": [],  # 模型列表，每个元素为(model_type, model_params)元组
        }

        # 合并参数
        self.ensemble_params = {**default_params, **(params or {})}

        # 初始化模型列表
        self.models: List[BaseModel] = []
        self.model_types = []
        self.model_params_list = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ):
        """训练集成模型"""
        from .factory import ModelFactory

        # 清空现有模型
        self.models = []
        self.model_types = []
        self.model_params_list = []

        # 获取模型配置
        models_config = self.ensemble_params.get("models", [])
        if not models_config:
            raise ValueError("必须提供至少一个模型配置")

        # 训练每个基础模型
        for i, model_config in enumerate(models_config):
            if isinstance(model_config, tuple) and len(model_config) == 2:
                model_type, model_params = model_config
            elif isinstance(model_config, dict):
                model_type = model_config.get("type")
                model_params = model_config.get("params", {})
            else:
                raise ValueError(f"模型配置格式错误: {model_config}")

            # 创建模型
            model = ModelFactory.create_model(model_type, model_params)

            # 训练模型
            print(f"训练模型 {i + 1}/{len(models_config)}: {model_type}")
            model.fit(X_train, y_train, X_valid, y_valid)

            # 保存模型信息
            self.models.append(model)
            self.model_types.append(model_type)
            self.model_params_list.append(model_params)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测类别"""
        if not self.models:
            raise ValueError("模型未训练")

        voting_method = self.ensemble_params.get("voting", "soft")
        weights = self.ensemble_params.get("weights")

        if voting_method == "hard":
            return self._hard_vote(X, weights)
        else:  # soft voting
            return self._soft_vote(X, weights)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """预测概率"""
        if not self.models:
            raise ValueError("模型未训练")

        voting_method = self.ensemble_params.get("voting", "soft")
        weights = self.ensemble_params.get("weights")

        if voting_method == "hard":
            # 硬投票时返回类别概率（基于投票比例）
            return self._hard_vote_proba(X, weights)
        else:
            # 软投票时返回加权平均概率
            return self._soft_vote_proba(X, weights)

    def save(self, filename: str):
        """保存集成模型"""
        if not self.models:
            raise ValueError("模型未训练")

        # 保存模型信息
        save_data = {
            "ensemble_params": self.ensemble_params,
            "model_types": self.model_types,
            "model_params_list": self.model_params_list,
            "models": self.models,
        }
        joblib.dump(save_data, filename)

    def load(self, filename: str):
        """加载集成模型"""
        save_data = joblib.load(filename)
        self.ensemble_params = save_data["ensemble_params"]
        self.model_types = save_data["model_types"]
        self.model_params_list = save_data["model_params_list"]
        self.models = save_data["models"]

    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性（所有模型的平均重要性）"""
        if not self.models:
            return pd.Series()

        # 收集所有模型的特征重要性
        all_importances = []
        for i, model in enumerate(self.models):
            importance = model.get_feature_importance()
            if not importance.empty:
                all_importances.append(importance)

        if not all_importances:
            return pd.Series()

        # 对齐特征并计算平均重要性
        all_features = set()
        for imp in all_importances:
            all_features.update(imp.index)

        # 创建DataFrame并计算平均重要性
        importance_df = pd.DataFrame(index=list(all_features))
        for i, imp in enumerate(all_importances):
            importance_df[f"model_{i}"] = imp

        # 计算平均重要性（用0填充缺失值）
        avg_importance = importance_df.fillna(0).mean(axis=1)
        return avg_importance.sort_values(ascending=False)

    def _hard_vote(self, X: pd.DataFrame, weights: List[float] = None) -> pd.Series:
        """硬投票"""
        # 收集所有模型的预测
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred.values)

        predictions = np.array(predictions)  # shape: (n_models, n_samples)

        # 应用权重
        if weights is not None and len(weights) == len(self.models):
            # 为每个样本计算加权投票
            weighted_votes = np.zeros((predictions.shape[1], 2))  # (n_samples, 2)
            for i, weight in enumerate(weights):
                for j in range(predictions.shape[1]):
                    weighted_votes[j, int(predictions[i, j])] += weight
            final_predictions = np.argmax(weighted_votes, axis=1)
        else:
            # 简单多数投票
            final_predictions = np.round(np.mean(predictions, axis=0)).astype(int)

        return pd.Series(final_predictions, index=X.index)

    def _soft_vote(self, X: pd.DataFrame, weights: List[float] = None) -> pd.Series:
        """软投票"""
        # 获取概率预测
        proba = self._soft_vote_proba(X, weights)
        # 选择概率较高的类别
        return pd.Series((proba[1] > 0.5).astype(int), index=X.index)

    def _hard_vote_proba(
        self, X: pd.DataFrame, weights: List[float] = None
    ) -> pd.DataFrame:
        """硬投票的概率（基于投票比例）"""
        # 收集所有模型的预测
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred.values)

        predictions = np.array(predictions)  # shape: (n_models, n_samples)

        # 计算每个类别的投票比例
        n_samples = predictions.shape[1]
        proba = np.zeros((n_samples, 2))

        if weights is not None and len(weights) == len(self.models):
            # 加权投票
            for i, weight in enumerate(weights):
                for j in range(n_samples):
                    proba[j, int(predictions[i, j])] += weight
            # 归一化
            proba = proba / np.sum(weights)
        else:
            # 等权重投票
            for i in range(len(self.models)):
                for j in range(n_samples):
                    proba[j, int(predictions[i, j])] += 1
            # 归一化
            proba = proba / len(self.models)

        return pd.DataFrame(proba, columns=[0, 1], index=X.index)

    def _soft_vote_proba(
        self, X: pd.DataFrame, weights: List[float] = None
    ) -> pd.DataFrame:
        """软投票的概率（加权平均）"""
        # 收集所有模型的概率预测
        all_proba = []
        for model in self.models:
            proba = model.predict_proba(X)
            all_proba.append(proba.values)

        all_proba = np.array(all_proba)  # shape: (n_models, n_samples, 2)

        # 应用权重
        if weights is not None and len(weights) == len(self.models):
            # 加权平均
            weights_array = np.array(weights).reshape(-1, 1, 1)
            weighted_proba = all_proba * weights_array
            avg_proba = np.sum(weighted_proba, axis=0) / np.sum(weights)
        else:
            # 简单平均
            avg_proba = np.mean(all_proba, axis=0)

        return pd.DataFrame(avg_proba, columns=[0, 1], index=X.index)

    def get_model_info(self) -> pd.DataFrame:
        """获取所有模型的信息"""
        if not self.models:
            return pd.DataFrame()

        info = []
        for i, (model_type, model_params, model) in enumerate(
            zip(self.model_types, self.model_params_list, self.models)
        ):
            info.append(
                {
                    "model_index": i,
                    "model_type": model_type,
                    "params": str(model_params),
                    "has_model": model.model is not None
                    if hasattr(model, "model")
                    else True,
                }
            )

        return pd.DataFrame(info)
