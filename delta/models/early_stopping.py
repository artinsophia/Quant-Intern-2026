import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
import warnings


class EarlyStopping:
    """通用的早停机制工具类

    支持多种早停策略，包括：
    1. 基于验证集损失的早停
    2. 基于验证集指标的早停（如准确率、F1分数等）
    3. 基于训练损失的早停
    4. 基于训练时间的早停

    兼容各种机器学习框架和自定义训练循环。
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        monitor: str = "val_loss",
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """初始化早停机制

        Args:
            patience: 容忍没有改进的轮数
            min_delta: 被视为改进的最小变化量
            mode: 'min' 表示监控指标越小越好，'max' 表示越大越好
            monitor: 监控的指标名称，如 'val_loss', 'val_accuracy', 'train_loss' 等
            baseline: 基线值，如果指标达不到基线值，会立即停止
            restore_best_weights: 是否在早停时恢复最佳权重
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.best_weights = None
        self.history = []

        # 验证参数
        if mode not in ["min", "max"]:
            raise ValueError(f"mode 必须是 'min' 或 'max', 但得到 {mode}")
        if patience < 0:
            raise ValueError(f"patience 必须是非负数, 但得到 {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta 必须是非负数, 但得到 {min_delta}")

    def __call__(self, epoch: int, metrics: Dict[str, float], model=None) -> bool:
        """检查是否应该早停

        Args:
            epoch: 当前训练轮数
            metrics: 包含监控指标的字典
            model: 可选的模型对象，用于保存/恢复权重

        Returns:
            bool: 如果应该停止训练返回True，否则返回False
        """
        if self.monitor not in metrics:
            warnings.warn(
                f"监控指标 '{self.monitor}' 不在 metrics 中: {list(metrics.keys())}"
            )
            return False

        current_score = metrics[self.monitor]

        # 保存历史记录
        self.history.append(
            {
                "epoch": epoch,
                "score": current_score,
                "metrics": metrics.copy(),
            }
        )

        # 检查基线
        if self.baseline is not None:
            if self.mode == "min" and current_score > self.baseline:
                if self.verbose:
                    print(
                        f"指标 {self.monitor}={current_score:.6f} 超过基线 {self.baseline:.6f}"
                    )
                return True
            elif self.mode == "max" and current_score < self.baseline:
                if self.verbose:
                    print(
                        f"指标 {self.monitor}={current_score:.6f} 低于基线 {self.baseline:.6f}"
                    )
                return True

        # 初始化最佳分数
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = self._get_model_weights(model)
            return False

        # 检查是否有改进
        if self._is_improvement(current_score, self.best_score):
            if self.verbose:
                improvement = abs(current_score - self.best_score)
                print(
                    f"轮数 {epoch}: {self.monitor} 从 {self.best_score:.6f} 改进到 {current_score:.6f} "
                    f"(改进: {improvement:.6f})"
                )
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = self._get_model_weights(model)
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"轮数 {epoch}: {self.monitor} 没有改进 "
                    f"(最佳: {self.best_score:.6f}, 当前: {current_score:.6f}, "
                    f"耐心: {self.counter}/{self.patience})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"早停在轮数 {epoch}, 最佳轮数 {self.best_epoch}")
                if (
                    model is not None
                    and self.restore_best_weights
                    and self.best_weights is not None
                ):
                    self._restore_model_weights(model, self.best_weights)
                return True

        return False

    def _is_improvement(self, current: float, best: float) -> bool:
        """检查当前分数是否比最佳分数有改进"""
        if self.mode == "min":
            return current < best - self.min_delta
        else:  # mode == "max"
            return current > best + self.min_delta

    def _get_model_weights(self, model) -> Any:
        """获取模型权重

        这个方法需要根据具体的模型框架进行重写。
        默认实现返回模型的 state_dict 或等效结构。
        """
        if hasattr(model, "state_dict"):
            return model.state_dict()
        elif hasattr(model, "get_weights"):
            return model.get_weights()
        elif hasattr(model, "coef_") and hasattr(model, "intercept_"):
            # 对于 scikit-learn 线性模型
            return {
                "coef": model.coef_,
                "intercept": model.intercept_,
            }
        else:
            # 如果无法获取权重，返回 None
            return None

    def _restore_model_weights(self, model, weights: Any):
        """恢复模型权重

        这个方法需要根据具体的模型框架进行重写。
        """
        if weights is None:
            return

        if hasattr(model, "load_state_dict"):
            model.load_state_dict(weights)
        elif hasattr(model, "set_weights"):
            model.set_weights(weights)
        elif isinstance(weights, dict) and "coef" in weights and "intercept" in weights:
            # 对于 scikit-learn 线性模型
            if hasattr(model, "coef_"):
                model.coef_ = weights["coef"]
            if hasattr(model, "intercept_"):
                model.intercept_ = weights["intercept"]

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.best_weights = None
        self.history = []

    def get_best_epoch(self) -> Optional[int]:
        """获取最佳轮数"""
        return self.best_epoch

    def get_best_score(self) -> Optional[float]:
        """获取最佳分数"""
        return self.best_score

    def get_history(self) -> List[Dict]:
        """获取训练历史"""
        return self.history.copy()

    def get_summary(self) -> Dict[str, Any]:
        """获取早停摘要"""
        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "early_stopped": self.early_stop,
            "final_epoch": len(self.history) - 1 if self.history else None,
            "patience": self.patience,
            "monitor": self.monitor,
            "mode": self.mode,
        }


class XGBoostEarlyStopping(EarlyStopping):
    """XGBoost专用的早停机制"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        monitor: str = "validation_0-error",
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """初始化XGBoost早停

        Args:
            monitor: XGBoost评估指标名称，如 'validation_0-error', 'validation_0-logloss',
                    'validation_0-auc', 'validation_0-merror' 等
        """
        super().__init__(
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            monitor=monitor,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
            verbose=verbose,
        )

    def _get_model_weights(self, model) -> Any:
        """获取XGBoost模型权重"""
        if hasattr(model, "save_raw"):
            # 对于XGBoost模型，返回原始模型数据
            return model.save_raw()
        return None

    def _restore_model_weights(self, model, weights: Any):
        """恢复XGBoost模型权重"""
        if weights is not None and hasattr(model, "load_raw"):
            model.load_raw(weights)


class SKLearnEarlyStopping(EarlyStopping):
    """scikit-learn模型的早停机制

    注意：scikit-learn模型通常不支持增量训练，
    这个类主要用于自定义训练循环或支持增量训练的模型。
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        monitor: str = "val_loss",
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            monitor=monitor,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
            verbose=verbose,
        )


def create_early_stopping(
    model_type: str,
    early_stopping_params: Optional[Dict[str, Any]] = None,
) -> Optional[EarlyStopping]:
    """创建适合指定模型类型的早停机制

    Args:
        model_type: 模型类型，如 'xgboost', 'linear', 'ensemble'
        early_stopping_params: 早停参数

    Returns:
        EarlyStopping实例或None（如果不支持早停）
    """
    if early_stopping_params is None:
        return None

    params = early_stopping_params.copy()

    # 根据模型类型选择适当的早停类
    if model_type.lower() == "xgboost":
        # XGBoost默认监控验证集错误率
        if "monitor" not in params:
            params["monitor"] = "validation_0-error"
        if "mode" not in params:
            params["mode"] = "min"
        return XGBoostEarlyStopping(**params)

    elif model_type.lower() in ["linear", "logistic", "sklearn"]:
        # scikit-learn模型
        if "monitor" not in params:
            params["monitor"] = "val_loss"
        if "mode" not in params:
            params["mode"] = "min"
        return SKLearnEarlyStopping(**params)

    else:
        # 通用早停机制
        if "monitor" not in params:
            params["monitor"] = "val_loss"
        if "mode" not in params:
            params["mode"] = "min"
        return EarlyStopping(**params)


def get_default_early_stopping_params(model_type: str) -> Dict[str, Any]:
    """获取指定模型类型的默认早停参数

    Args:
        model_type: 模型类型

    Returns:
        默认早停参数字典
    """
    defaults = {
        "xgboost": {
            "patience": 50,
            "min_delta": 0.001,
            "mode": "min",
            "monitor": "validation_0-error",
            "restore_best_weights": True,
            "verbose": True,
        },
        "linear": {
            "patience": 10,
            "min_delta": 0.0001,
            "mode": "min",
            "monitor": "val_loss",
            "restore_best_weights": True,
            "verbose": True,
        },
        "ensemble": {
            "patience": 20,
            "min_delta": 0.0005,
            "mode": "min",
            "monitor": "val_loss",
            "restore_best_weights": True,
            "verbose": True,
        },
    }

    return defaults.get(
        model_type.lower(),
        {
            "patience": 10,
            "min_delta": 0.001,
            "mode": "min",
            "monitor": "val_loss",
            "restore_best_weights": True,
            "verbose": True,
        },
    )
