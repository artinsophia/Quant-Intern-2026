# Delta模块多模型架构扩展

## 概述

本扩展为delta模块添加了多模型支持，允许用户轻松切换和比较不同的机器学习模型。当前支持XGBoost和线性模型（Logistic回归），并设计了可扩展的架构以便未来添加更多模型类型。

## 新文件结构

```
delta/
├── __init__.py                 # 更新：导出新模型类
├── main.py                     # 更新：支持模型类型参数
├── strategy.py                 # 不变
├── data_processing.py          # 不变
├── features.py                 # 不变
├── models/                     # 新增：模型目录
│   ├── __init__.py            # 导出模型类
│   ├── base.py                # 基础模型接口
│   ├── xgboost_model.py       # XGBoost模型实现
│   ├── linear_model.py        # 线性模型实现
│   └── factory.py             # 模型工厂
├── train.py                    # 重构：使用模型工厂
├── config_example.py          # 新增：配置示例
├── test_models.py             # 新增：测试脚本
├── usage_example.py           # 新增：使用示例
└── demo.ipynb                 # 现有演示文件
```

## 核心组件

### 1. 基础模型接口 (`models/base.py`)
定义了所有模型必须实现的统一接口：
- `fit()`: 训练模型
- `predict()`: 预测类别
- `predict_proba()`: 预测概率
- `save()`/`load()`: 保存加载模型
- `get_feature_importance()`: 获取特征重要性

### 2. 模型实现
- **XGBoostModel**: 基于原有XGBoost实现，保持向后兼容
- **LinearModel**: 新的线性模型实现，使用标准化+Logistic回归pipeline

### 3. 模型工厂 (`models/factory.py`)
统一创建和管理模型：
```python
from delta.models.factory import ModelFactory

# 创建模型
model = ModelFactory.create_model("xgboost", params={...})
model = ModelFactory.create_model("linear", params={...})

# 查看可用模型
available = ModelFactory.get_available_models()  # ['xgboost', 'linear']
```

### 4. 重构的train模块
- `train_model()`: 现在接受`model_type`参数
- `load_model()`: 需要指定模型类型以正确加载

## 使用方法

### 基本使用

```python
from delta import main

# 1. 使用默认XGBoost模型（保持向后兼容）
model, strategy = main()

# 2. 使用线性模型
model, strategy = main(model_type="linear")

# 3. 使用自定义参数
model_params = {"n_estimators": 1000, "max_depth": 4}
model, strategy = main(model_type="xgboost", model_params=model_params)
```

### 直接使用模型工厂

```python
from delta.models.factory import ModelFactory
from delta.models.xgboost_model import XGBoostModel
from delta.models.linear_model import LinearModel

# 创建模型
xgb_model = ModelFactory.create_model("xgboost")
linear_model = ModelFactory.create_model("linear")

# 训练模型
xgb_model.fit(X_train, y_train, X_valid, y_valid)

# 预测
predictions = linear_model.predict(X_test)
probabilities = linear_model.predict_proba(X_test)
```

### 使用配置文件

```python
import config_example

# 使用预定义配置
param_dict.update(config_example.XGBOOST_CONFIG)
# 或
param_dict.update(config_example.LINEAR_CONFIG)
```

## 模型比较示例

```python
def compare_models():
    """比较不同模型性能"""
    configs = [
        ("xgboost_default", {"model_type": "xgboost"}),
        ("linear_default", {"model_type": "linear"}),
        ("linear_l1", {"model_type": "linear", "model_params": {"penalty": "l1"}}),
    ]
    
    results = {}
    for name, config in configs:
        print(f"训练 {name}...")
        model, strategy = main(**config)
        # 进行回测和性能评估
        results[name] = {
            "accuracy": evaluate_model(model, X_test, y_test),
            "feature_importance": model.get_feature_importance(),
        }
    
    return results
```

## 扩展新模型

要添加新模型类型：

1. 在`models/`目录下创建新模型文件，继承`BaseModel`
2. 在`models/factory.py`中注册新模型
3. 在`models/__init__.py`中导出新模型类

示例：
```python
# 1. 创建新模型类
class RandomForestModel(BaseModel):
    def __init__(self, params=None):
        super().__init__(params)
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**(params or {}))
    
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model.fit(X_train, y_train)
        return self
    
    # 实现其他必要方法...

# 2. 在factory.py中注册
ModelFactory.register_model("random_forest", RandomForestModel)
```

## 优势

1. **灵活性**: 轻松切换不同模型类型
2. **可扩展性**: 易于添加新模型
3. **统一接口**: 所有模型使用相同API
4. **向后兼容**: 现有代码无需修改
5. **便于比较**: 标准化接口便于模型比较

## 测试

运行测试脚本验证功能：
```bash
cd /home/jovyan/work/tactics_demo/delta
/opt/conda/bin/python test_models.py
```

## 未来扩展方向

1. **更多模型类型**: 随机森林、神经网络、LightGBM等
2. **集成学习**: 模型集成和堆叠
3. **超参数优化**: 自动调参框架
4. **模型解释**: SHAP值、部分依赖图等
5. **模型监控**: 性能衰减检测和模型更新