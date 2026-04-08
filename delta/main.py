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


def main():
    instrument_id = "518880"

    param_dict = {
        "instrument_id": instrument_id,
        "trade_ymd": "20260319",
        "short_window": 60,
        "long_window": 300,
        "open_threshold": 2,
        "close_threshold": 0,
        "confidence_threshold": 0.4,
        "name": "delta_v1",
        "trailing_stop_pct": 0.001,
        "y_window": 300,
        "stride": 1,
        "k_up": 3,
        "k_down": 3,
    }
    param_dict["x_window"] = max(param_dict["short_window"], param_dict["long_window"])

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

    model_filename = f"delta_model_{instrument_id}.joblib"
    save_model(model, model_filename)

    print("\n创建策略实例...")
    strategy = StrategyDemo(model, param_dict)
    print(f"策略已创建: {strategy.name}")

    return model, strategy


if __name__ == "__main__":
    model, strategy = main()
