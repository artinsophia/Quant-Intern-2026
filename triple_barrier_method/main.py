import sys

sys.path.append("/home/jovyan/base_demo")

from .features import create_feature
from .data_processing import samples_from_dates, create_y
from .train import (
    train_model,
    evaluate_model,
    save_model,
    load_model,
    get_trade_dates,
    split_dates,
)
from .strategy import StrategyDemo


def main(model_type="xgboost", model_params=None):
    """主函数

    Args:
        model_type: 模型类型，可选 'xgboost' 或 'linear'
        model_params: 模型特定参数

    Returns:
        model: 训练好的模型
        strategy: 策略实例
    """
    instrument_id = "511520"

    param_dict = {
        "name": "TBM",
        "instrument_id": instrument_id,
        "trade_ymd": "20260319",
        "x_window": 300,
        "y_window": 300,
        "stride": 60,
        "k_up": 3,
        "k_down": 3,
        "model_type": model_type,
        "model_params": model_params or {},
    }

    trade_dates = get_trade_dates()
    print(f"总交易日数量: {len(trade_dates)}")
    print(f"交易日范围: {trade_dates[0]} ~ {trade_dates[-1]}")

    train_dates, valid_dates, test_dates = split_dates(trade_dates)

    print("\n生成训练集样本...")
    X_train, y_train = samples_from_dates(
        train_dates, instrument_id, param_dict, create_feature, create_y
    )
    print(f"训练集样本: X={X_train.shape}, y={y_train.shape}")
    if len(y_train) > 0:
        print(f"标签分布:\n{y_train.value_counts()}")

    print("\n生成验证集样本...")
    X_valid, y_valid = samples_from_dates(
        valid_dates, instrument_id, param_dict, create_feature, create_y
    )
    print(f"验证集样本: X={X_valid.shape}, y={y_valid.shape}")
    if len(y_valid) > 0:
        print(f"标签分布:\n{y_valid.value_counts()}")

    print("\n生成测试集样本...")
    X_test, y_test = samples_from_dates(
        test_dates, instrument_id, param_dict, create_feature, create_y
    )
    print(f"测试集样本: X={X_test.shape}, y={y_test.shape}")
    if len(y_test) > 0:
        print(f"标签分布:\n{y_test.value_counts()}")

    print("\n训练模型...")
    model = train_model(X_train, y_train, X_valid, y_valid, param_dict)

    print("\n评估模型...")
    accuracy = evaluate_model(model, X_test, y_test)

    model_filename = f"tbm_{model_type}_model_{instrument_id}.joblib"
    save_model(model, model_filename)

    print("\n创建策略实例...")
    strategy = StrategyDemo(model, param_dict)
    print(f"策略已创建: {strategy.name} (使用{model_type}模型)")

    return model, strategy


if __name__ == "__main__":
    model, strategy = main()
