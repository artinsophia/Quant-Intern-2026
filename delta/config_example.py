"""模型配置示例文件

这个文件展示了如何配置不同的模型参数。
可以将这些配置复制到main.py的param_dict中。
"""

# XGBoost模型配置
XGBOOST_CONFIG = {
    "model_type": "xgboost",
    "model_params": {
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
}

# 线性模型配置（Logistic回归）- 更新以兼容sklearn 1.8+
LINEAR_CONFIG = {
    "model_type": "linear",
    "model_params": {
        "C": 1.0,  # 正则化强度，越小正则化越强
        "l1_ratio": 0,  # 0表示l2正则化，1表示l1正则化，0-1之间表示弹性网络
        "solver": "lbfgs",  # 优化算法：'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced",  # 自动平衡类别权重
    },
}

# 强正则化的线性模型（防止过拟合）
LINEAR_STRONG_REG_CONFIG = {
    "model_type": "linear",
    "model_params": {
        "C": 0.1,  # 更强的正则化
        "l1_ratio": 0,  # l2正则化
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": 42,
        "class_weight": "balanced",
    },
}

# L1正则化的线性模型（特征选择）
LINEAR_L1_CONFIG = {
    "model_type": "linear",
    "model_params": {
        "C": 0.5,
        "l1_ratio": 1,  # l1正则化
        "solver": "liblinear",  # L1正则化需要使用liblinear或saga
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced",
    },
}

# 使用示例
if __name__ == "__main__":
    print("可用配置:")
    print("1. XGBoost配置:", XGBOOST_CONFIG["model_type"])
    print("2. 线性模型配置:", LINEAR_CONFIG["model_type"])
    print("3. 强正则化线性模型:", LINEAR_STRONG_REG_CONFIG["model_type"])
    print("4. L1正则化线性模型:", LINEAR_L1_CONFIG["model_type"])

    print("\n使用示例（在main.py中）:")
    print("""
# 使用XGBoost模型
param_dict.update(XGBOOST_CONFIG)

# 或使用线性模型（已更新以兼容sklearn 1.8+）
param_dict.update(LINEAR_CONFIG)

# 然后在main函数中调用
model, strategy = main()
    """)

    print("\n注意：线性模型配置已更新以兼容sklearn 1.8+：")
    print("- 使用 'l1_ratio' 参数代替 'penalty'")
    print("- l1_ratio=None: l2正则化")
    print("- l1_ratio=1: l1正则化")
    print("- l1_ratio=0.5: 弹性网络（l1和l2混合）")
