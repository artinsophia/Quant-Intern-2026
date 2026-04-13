#!/usr/bin/env python3
"""测试早停机制和验证集logloss图"""

import sys

sys.path.append("/home/jovyan/work/tactics_demo")

import numpy as np
import pandas as pd
from delta.models.xgboost_model import XGBoostModel

# 创建测试数据
np.random.seed(42)
n_samples = 1000
n_features = 10

# 训练数据
X_train = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"feature_{i}" for i in range(n_features)],
)
y_train = pd.Series(np.random.randint(0, 2, n_samples))

# 验证数据
X_valid = pd.DataFrame(
    np.random.randn(200, n_features),
    columns=[f"feature_{i}" for i in range(n_features)],
)
y_valid = pd.Series(np.random.randint(0, 2, 200))

# 测试早停参数
params = {
    "n_estimators": 500,  # 设置较大的树数量，测试早停
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 1,
    "beta": 1,
    # 早停参数
    "early_stopping": {
        "patience": 20,  # 20轮没有改善就停止
        "min_delta": 0.001,
        "mode": "min",
        "monitor": "validation_0-logloss",
        "baseline": None,
        "restore_best_weights": True,
        "verbose": 1,
    },
}

print("创建XGBoost模型...")
model = XGBoostModel(params)

print("训练模型（应该触发早停）...")
model.fit(X_train, y_train, X_valid, y_valid)

print("\n模型训练完成！")
print(f"模型参数: {model.model_params}")

# 测试预测
print("\n测试预测...")
y_pred = model.predict(X_valid)
print(f"预测结果形状: {y_pred.shape}")
print(f"预测结果示例: {y_pred[:10].values}")

# 测试概率预测
print("\n测试概率预测...")
y_proba = model.predict_proba(X_valid)
print(f"概率预测形状: {y_proba.shape}")
print(f"概率预测示例:\n{y_proba.head()}")

print("\n测试完成！")
