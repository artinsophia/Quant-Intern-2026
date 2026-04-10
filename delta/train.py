from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

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


def evaluate_model(model, X_test, y_test):
    """评估模型性能

    Args:
        model: 模型实例
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        准确率
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
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
