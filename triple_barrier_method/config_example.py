"""Triple Barrier Method 模型配置示例文件

这个文件展示了如何配置不同的模型参数。
"""

# XGBoost模型配置 - 三分类
XGBOOST_CONFIG = {
    "model_type": "xgboost",
    "model_params": {
        "n_estimators": 1000,
        "max_depth": 3,
        "learning_rate": 0.003,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softmax",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,
        "num_class": 3,
    },
}


# Triple Barrier 相关参数配置示例
TRIPLE_BARRIER_CONFIG = {
    "name": "TBM",
    "instrument_id": "511520",
    "trade_ymd": "20260319",
    "x_window": 300,
    "y_window": 300,
    "stride": 60,
    "k_up": 3,
    "k_down": 3,
}


# 策略参数配置示例
STRATEGY_CONFIG = {
    "x_window": 300,
    "open_confidence": 0.7,
    "name": "TBM",
}
