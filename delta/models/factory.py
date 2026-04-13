from typing import Dict, Any, Type, Optional
from .base import BaseModel
from .xgboost_model import XGBoostModel
from .linear_model import LinearModel
from .ensemble_model import EnsembleModel
from .early_stopping import get_default_early_stopping_params


class ModelFactory:
    """模型工厂类"""

    # 注册的模型类型
    _model_registry = {
        "xgboost": XGBoostModel,
        "linear": LinearModel,
        "ensemble": EnsembleModel,
    }

    @classmethod
    def create_model(cls, model_type: str, params: Dict[str, Any] = None) -> BaseModel:
        """创建指定类型的模型

        Args:
            model_type: 模型类型，如 'xgboost', 'linear'
            params: 模型参数

        Returns:
            BaseModel实例

        Raises:
            ValueError: 如果模型类型不支持
        """
        model_class = cls._model_registry.get(model_type.lower())
        if model_class is None:
            available_models = list(cls._model_registry.keys())
            raise ValueError(
                f"不支持的模型类型: {model_type}. 可用类型: {available_models}"
            )

        # 处理可能的元组情况（Jupyter notebook中字典末尾的逗号会创建元组）
        if (
            isinstance(params, tuple)
            and len(params) == 1
            and isinstance(params[0], dict)
        ):
            params = params[0]

        return model_class(params)

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]):
        """注册新的模型类型

        Args:
            name: 模型类型名称
            model_class: 模型类，必须继承自BaseModel
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"模型类必须继承自BaseModel")

        cls._model_registry[name.lower()] = model_class

    @classmethod
    def get_available_models(cls) -> list:
        """获取所有可用的模型类型"""
        return list(cls._model_registry.keys())

    @classmethod
    def get_default_params(
        cls, model_type: str, include_early_stopping: bool = False
    ) -> Dict[str, Any]:
        """获取指定模型类型的默认参数

        Args:
            model_type: 模型类型
            include_early_stopping: 是否包含早停参数

        Returns:
            默认参数字典
        """
        # 不同模型的默认参数
        default_params = {
            "xgboost": {
                "n_estimators": 2000,
                "max_depth": 3,
                "learning_rate": 0.01,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 1,
            },
            "linear": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
            },
            "ensemble": {
                "voting": "soft",
                "weights": None,
                "models": [
                    ("xgboost", {"n_estimators": 1000, "max_depth": 3}),
                    ("linear", {"C": 1.0}),
                ],
            },
        }

        params = default_params.get(model_type.lower(), {}).copy()

        # 如果需要，添加早停参数
        if include_early_stopping:
            early_stopping_params = get_default_early_stopping_params(model_type)
            params["early_stopping"] = early_stopping_params

        return params

    @classmethod
    def get_default_early_stopping_params(cls, model_type: str) -> Dict[str, Any]:
        """获取指定模型类型的默认早停参数

        Args:
            model_type: 模型类型

        Returns:
            默认早停参数字典
        """
        return get_default_early_stopping_params(model_type)
