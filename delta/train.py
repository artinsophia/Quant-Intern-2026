from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,
    average_precision_score,
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .models.factory import ModelFactory


def train_model(X_train, y_train, X_valid, y_valid, param_dict):
    """训练模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_valid: 验证特征
        y_valid: 验证标签
        param_dict: 参数字典，包含model_type等参数

    Returns:
        训练好的模型实例
    """
    # 获取模型类型，默认为xgboost以保持向后兼容
    model_type = param_dict.get("model_type", "xgboost")

    # 获取模型特定参数
    model_params = param_dict.get("model_params", {})

    # 处理可能的元组情况（Jupyter notebook中字典末尾的逗号会创建元组）
    if (
        isinstance(model_params, tuple)
        and len(model_params) == 1
        and isinstance(model_params[0], dict)
    ):
        model_params = model_params[0]

    print(f"训练 {model_type} 模型...")

    # 使用模型工厂创建模型
    model = ModelFactory.create_model(model_type, model_params)

    # 训练模型
    model.fit(X_train, y_train, X_valid, y_valid)

    # 打印特征重要性
    if hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
        if not importance.empty:
            print("\n特征重要性（前10个）:")
            print(importance.head(10))

    return model

def evaluate_model(model, X_test, y_test, show_plots=False):
    """评估模型性能

    Args:
        model: 模型实例
        X_test: 测试特征
        y_test: 测试标签
        show_plots: 是否展示图表，默认为False

    Returns:
        准确率
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 添加PR曲线和概率分布图
    if hasattr(model, "predict_proba"):
        y_pred_proba_df = model.predict_proba(X_test)
        # 获取正类的概率（第二列）
        y_pred_proba = (
            y_pred_proba_df.iloc[:, 1].values
            if hasattr(y_pred_proba_df, "iloc")
            else y_pred_proba_df[:, 1]
        )
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        print(f"\nPR曲线AUC: {pr_auc:.4f}")
        print(f"平均精度 (AP): {avg_precision:.4f}")

        # 计算预测概率的统计信息
        print(f"\n预测概率统计:")
        print(f"  均值: {y_pred_proba.mean():.4f}")
        print(f"  标准差: {y_pred_proba.std():.4f}")
        print(f"  最小值: {y_pred_proba.min():.4f}")
        print(f"  25%分位数: {np.percentile(y_pred_proba, 25):.4f}")
        print(f"  中位数: {np.median(y_pred_proba):.4f}")
        print(f"  75%分位数: {np.percentile(y_pred_proba, 75):.4f}")
        print(f"  最大值: {y_pred_proba.max():.4f}")

        # 按真实标签分组统计
        y_true_0 = y_pred_proba[y_test == 0]
        y_true_1 = y_pred_proba[y_test == 1]
        print(f"\n按真实标签分组的预测概率统计:")
        print(
            f"  标签0 (负类): 均值={y_true_0.mean():.4f}, 标准差={y_true_0.std():.4f}"
        )
        print(
            f"  标签1 (正类): 均值={y_true_1.mean():.4f}, 标准差={y_true_1.std():.4f}"
        )

        if show_plots:
            # 创建包含两个子图的图形
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 子图1: PR曲线
            ax1 = axes[0]
            ax1.plot(
                recall,
                precision,
                "b-",
                linewidth=2,
                label=f"PR curve (AUC = {pr_auc:.3f})",
            )
            ax1.plot([0, 1], [1, 0], "k--", alpha=0.5, label="Random")
            ax1.set_xlabel("Recall")
            ax1.set_ylabel("Precision")
            ax1.set_title("Precision-Recall Curve")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best")

            # 子图2: 预测概率分布图
            ax2 = axes[1]

            # 绘制直方图
            bins = np.linspace(0, 1, 21)
            ax2.hist(
                y_true_0,
                bins=bins,
                alpha=0.5,
                label="True Label 0",
                color="red",
                edgecolor="black",
            )
            ax2.hist(
                y_true_1,
                bins=bins,
                alpha=0.5,
                label="True Label 1",
                color="blue",
                edgecolor="black",
            )

            # 添加垂直线表示阈值0.5
            ax2.axvline(
                x=0.5,
                color="green",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Threshold 0.5",
            )

            ax2.set_xlabel("Predicted Probability")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Predicted Probability Distribution by True Label")
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    return accuracy


def save_model(model, filename):
    """保存模型

    Args:
        model: 模型实例
        filename: 保存文件名
    """
    model.save(filename)
    print(f"模型已保存到: {filename}")


def load_model(filename, model_type="xgboost", model_params=None):
    """加载模型

    Args:
        filename: 模型文件名
        model_type: 模型类型（用于创建模型实例）
        model_params: 模型参数

    Returns:
        加载的模型实例
    """
    # 创建模型实例
    model = ModelFactory.create_model(model_type, model_params)

    # 加载模型权重
    model.load(filename)
    print(f"模型已从 {filename} 加载")

    return model


def get_trade_dates():
    trade_dates = [
        "20260105",
        "20260106",
        "20260107",
        "20260108",
        "20260109",
        "20260112",
        "20260113",
        "20260114",
        "20260115",
        "20260116",
        "20260119",
        "20260120",
        "20260121",
        "20260122",
        "20260123",
        "20260126",
        "20260127",
        "20260128",
        "20260129",
        "20260130",
        "20260202",
        "20260203",
        "20260204",
        "20260205",
        "20260206",
        "20260209",
        "20260210",
        "20260211",
        "20260212",
        "20260213",
        "20260224",
        "20260225",
        "20260226",
        "20260227",
        "20260302",
        "20260303",
        "20260304",
        "20260305",
        "20260306",
        "20260309",
        "20260310",
        "20260311",
        "20260312",
        "20260313",
        "20260316",
        "20260317",
        "20260318",
        "20260319",
        "20260320",
        "20260323",
        "20260324",
        "20260325",
        "20260326",
        "20260327",
    ]
    return trade_dates


def split_dates(trade_dates, train_days=35, valid_days=9, test_days=10):
    train_dates = trade_dates[:train_days]
    valid_dates = trade_dates[train_days : train_days + valid_days]
    test_dates = trade_dates[
        train_days + valid_days : train_days + valid_days + test_days
    ]

    print(f"训练集: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)}天)")
    print(f"验证集: {valid_dates[0]} ~ {valid_dates[-1]} ({len(valid_dates)}天)")
    print(f"测试集: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")

    return train_dates, valid_dates, test_dates
