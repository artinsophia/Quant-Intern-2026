# 早停机制使用指南

## 概述

本项目为 `@tactics_demo/delta/models/` 添加了通用的早停机制，支持多种模型类型，包括：
- XGBoost模型
- 线性模型（Logistic回归）
- 集成模型
- 未来可能添加的其他模型类型

## 快速开始

### 1. 基本使用

```python
from delta.models.factory import ModelFactory

# 创建带早停的XGBoost模型
params = {
    "n_estimators": 2000,  # 设置较大的迭代次数
    "early_stopping": {
        "patience": 50,           # 容忍没有改进的轮数
        "min_delta": 0.001,       # 最小改进量
        "mode": "min",            # 监控指标越小越好
        "monitor": "validation_0-error",  # 监控验证集错误率
        "verbose": True,          # 打印早停信息
    }
}

model = ModelFactory.create_model("xgboost", params)
model.fit(X_train, y_train, X_valid, y_valid)  # 需要提供验证集
```

### 2. 使用默认早停参数

```python
from delta.models.factory import ModelFactory

# 获取带早停的默认参数
params = ModelFactory.get_default_params("xgboost", include_early_stopping=True)
model = ModelFactory.create_model("xgboost", params)
```

### 3. 集成模型中的早停

```python
params = {
    "voting": "soft",
    "early_stopping": {
        "patience": 20,
        "min_delta": 0.0005,
        "mode": "min",
        "monitor": "val_loss",
    },
    "models": [
        ("xgboost", {"n_estimators": 1000}),
        ("linear", {"C": 1.0}),
    ]
}

ensemble_model = ModelFactory.create_model("ensemble", params)
ensemble_model.fit(X_train, y_train, X_valid, y_valid)
```

## 早停参数说明

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `patience` | int | 模型相关 | 容忍没有改进的轮数 |
| `min_delta` | float | 模型相关 | 被视为改进的最小变化量 |
| `mode` | str | 'min' | 'min'表示指标越小越好，'max'表示越大越好 |
| `monitor` | str | 模型相关 | 监控的指标名称 |
| `baseline` | float | None | 基线值，达不到基线会立即停止 |
| `restore_best_weights` | bool | True | 是否恢复最佳权重 |
| `verbose` | bool | True | 是否打印早停信息 |

### 各模型默认早停参数

```python
from delta.models.early_stopping import get_default_early_stopping_params

# XGBoost
xgb_params = get_default_early_stopping_params("xgboost")
# {'patience': 50, 'min_delta': 0.001, 'mode': 'min', 'monitor': 'validation_0-error', ...}

# 线性模型
linear_params = get_default_early_stopping_params("linear")
# {'patience': 10, 'min_delta': 0.0001, 'mode': 'min', 'monitor': 'val_loss', ...}

# 集成模型
ensemble_params = get_default_early_stopping_params("ensemble")
# {'patience': 20, 'min_delta': 0.0005, 'mode': 'min', 'monitor': 'val_loss', ...}
```

## XGBoost特定配置

### 支持的监控指标

XGBoost早停支持以下监控指标：
- `validation_0-error`: 验证集错误率（默认）
- `validation_0-logloss`: 验证集对数损失
- `validation_0-auc`: 验证集AUC
- `validation_0-merror`: 多分类错误率
- `validation_0-mlogloss`: 多分类对数损失

### 示例：监控AUC

```python
params = {
    "n_estimators": 2000,
    "early_stopping": {
        "patience": 30,
        "min_delta": 0.0001,
        "mode": "max",  # AUC越大越好
        "monitor": "validation_0-auc",
    }
}
```

## 获取训练信息

### 1. 获取早停摘要

```python
# 训练后获取早停信息
early_stop_info = model.get_early_stopping_info()
print(early_stop_info)
# {'best_epoch': 125, 'best_score': 0.1234, 'early_stopped': True, ...}
```

### 2. 获取训练历史（XGBoost）

```python
# 仅XGBoost模型支持
if hasattr(model, 'get_training_history'):
    history = model.get_training_history()
    print(history.head())
```

### 3. 集成模型的子模型信息

```python
# 获取集成模型中所有子模型的早停信息
if hasattr(ensemble_model, 'get_early_stopping_info_all'):
    all_info = ensemble_model.get_early_stopping_info_all()
    print(all_info)
```

## 高级用法

### 1. 自定义早停类

```python
from delta.models.early_stopping import EarlyStopping

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def _is_improvement(self, current: float, best: float) -> bool:
        # 自定义改进判断逻辑
        if self.mode == "min":
            return current < best - self.min_delta * self.custom_param
        else:
            return current > best + self.min_delta * self.custom_param

# 使用自定义早停
custom_es = CustomEarlyStopping(
    patience=10,
    min_delta=0.001,
    mode="min",
    monitor="val_loss",
    custom_param=2.0  # 自定义参数
)
```

### 2. 手动早停控制

```python
from delta.models.early_stopping import create_early_stopping

# 创建早停实例
early_stopping = create_early_stopping("xgboost", {
    "patience": 10,
    "min_delta": 0.001,
    "mode": "min",
    "monitor": "val_loss",
})

# 在训练循环中手动调用
for epoch in range(100):
    # ... 训练代码 ...
    
    # 计算验证指标
    val_metrics = {"val_loss": current_loss}
    
    # 检查是否应该早停
    if early_stopping(epoch, val_metrics, model=model):
        print(f"早停在轮次 {epoch}")
        break
```

### 3. 重置早停状态

```python
# 重置早停状态，用于多次训练
model.reset_early_stopping()
```

## 注意事项

### 1. 验证集要求
- 早停机制需要验证集才能工作
- 如果没有提供验证集，早停将被忽略

### 2. 线性模型限制
- 线性模型（Logistic回归）通常不支持增量训练
- 早停参数会被接受但会显示警告
- 实际训练时使用标准的一次性训练

### 3. 集成模型
- 集成模型的早停参数会传递给所有子模型
- 每个子模型独立进行早停判断
- 可以通过 `get_early_stopping_info_all()` 查看所有子模型的早停信息

### 4. 性能考虑
- 早停可以显著减少训练时间
- 但设置过小的 `patience` 可能导致过早停止
- 建议根据数据集大小和模型复杂度调整参数

## 故障排除

### 1. 早停不工作
- 检查是否提供了验证集
- 检查 `monitor` 参数是否正确
- 检查 `mode` 参数是否与监控指标匹配

### 2. 过早停止
- 增加 `patience` 值
- 减小 `min_delta` 值
- 检查验证集是否具有代表性

### 3. 从未停止
- 检查 `n_estimators` 是否设置过大
- 检查模型是否持续改进
- 考虑增加 `min_delta` 或减小 `patience`

## 兼容性说明

### 向后兼容
- 现有代码无需修改即可继续工作
- 如果不提供早停参数，模型行为不变
- 早停参数完全可选

### 未来扩展
- 早停机制设计为可扩展的
- 新模型类型只需继承 `BaseModel` 即可自动支持
- 可以通过 `ModelFactory.register_model()` 注册新模型

## 示例代码

完整的示例代码请参考：
- `test_early_stop_simple.py`: 基本使用示例
- 项目中的其他训练脚本