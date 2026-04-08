import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from .data_processing import samples_from_dates, create_y
from .features import create_feature


def train_model(X_train, y_train, X_valid, y_valid, param_dict):
    # 计算类别权重以处理不平衡数据
    scale_pos_weight_value = (
        (y_train == 0).sum() / (y_train != 0).sum() if (y_train != 0).sum() > 0 else 1.0
    )

    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight_value,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    eval_set = [(X_valid, y_valid)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"模型已保存到: {filename}")


def load_model(filename):
    model = joblib.load(filename)
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
