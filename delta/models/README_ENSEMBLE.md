# 集成学习模型 (EnsembleModel)

## 概述

`EnsembleModel` 是一个支持软投票和硬投票的集成学习模型，可以组合多个基础模型进行预测，提高模型的稳定性和准确性。

## 特性

- **两种投票方式**: 支持软投票(soft)和硬投票(hard)
- **权重支持**: 可以为每个基础模型分配不同的权重
- **多种基础模型**: 支持XGBoost、线性模型等
- **特征重要性**: 提供所有基础模型的平均特征重要性
- **模型保存/加载**: 支持完整的模型序列化
- **工厂模式**: 通过`ModelFactory`统一创建

## 使用方法

### 1. 创建集成模型

```python
from models.factory import ModelFactory

# 软投票集成 (XGBoost + 线性模型)
ensemble_params = {
    "voting": "soft",  # "soft" 或 "hard"
    "weights": [0.6, 0.4],  # 可选，None表示等权重
    "models": [
        ("xgboost", {
            "n_estimators": 1000,
            "max_depth": 3,
            "learning_rate": 0.01,
        }),
        ("linear", {
            "C": 1.0,
            "class_weight": "balanced",
        }),
    ],
}

ensemble_model = ModelFactory.create_model("ensemble", ensemble_params)
```

### 2. 硬投票集成示例

```python
# 硬投票集成 (多个XGBoost模型)
ensemble_params_hard = {
    "voting": "hard",
    "weights": None,  # 等权重
    "models": [
        ("xgboost", {"n_estimators": 800, "max_depth": 3}),
        ("xgboost", {"n_estimators": 1200, "max_depth": 4}),
        ("xgboost", {"n_estimators": 600, "max_depth": 5}),
    ],
}
```

### 3. 训练模型

```python
# 训练集成模型
ensemble_model.fit(X_train, y_train, X_valid, y_valid)
```

### 4. 预测

```python
# 预测类别
predictions = ensemble_model.predict(X_test)

# 预测概率
probabilities = ensemble_model.predict_proba(X_test)
```

### 5. 获取特征重要性

```python
# 获取所有基础模型的平均特征重要性
feature_importance = ensemble_model.get_feature_importance()
print("Top 10特征重要性:")
print(feature_importance.head(10))
```

### 6. 保存和加载模型

```python
# 保存模型
ensemble_model.save("ensemble_model.pkl")

# 加载模型
loaded_model = ModelFactory.create_model("ensemble")
loaded_model.load("ensemble_model.pkl")
```

## 投票方式说明

### 软投票 (Soft Voting)

- **原理**: 对每个基础模型的概率预测进行加权平均
- **输出**: 加权平均后的概率，然后选择概率较高的类别
- **优点**: 考虑模型的不确定性，通常更稳定
- **适用场景**: 基础模型都能输出概率预测

### 硬投票 (Hard Voting)

- **原理**: 对每个基础模型的类别预测进行投票
- **输出**: 多数票决定的类别
- **优点**: 简单直接，对异常值不敏感
- **适用场景**: 基础模型预测差异较大时

## 权重配置

权重参数 `weights` 是一个列表，长度必须与 `models` 列表相同：
- 如果为 `None`: 所有模型等权重
- 如果提供权重列表: 按权重进行加权投票
- 权重会自动归一化，不需要总和为1

## 在Delta策略中使用

在delta策略中，可以通过以下方式使用集成模型：

```python
class DeltaStrategy:
    def __init__(self, model=None, param_dict={}):
        self.__dict__.update(param_dict)
        
        # 创建集成模型
        if model is None:
            ensemble_params = {
                "voting": "soft",
                "weights": [0.6, 0.4],
                "models": [
                    ("xgboost", {"n_estimators": 1000, "max_depth": 3}),
                    ("linear", {"C": 1.0}),
                ],
            }
            self.model = ModelFactory.create_model("ensemble", ensemble_params)
        else:
            self.model = model
            
        self.position_last = 0
        self.prev_signal = 0
    
    def train_model(self, X_train, y_train, X_valid=None, y_valid=None):
        """训练集成模型"""
        self.model.fit(X_train, y_train, X_valid, y_valid)
    
    def predict_signal(self, features):
        """使用集成模型预测交易信号"""
        if self.model is None:
            return 0
        
        # 转换为DataFrame
        X = pd.DataFrame([features])
        
        # 预测概率
        proba = self.model.predict_proba(X)
        
        # 基于概率生成信号
        buy_prob = proba.iloc[0, 1]  # 类别1的概率
        sell_prob = proba.iloc[0, 0]  # 类别0的概率
        
        if buy_prob > 0.6:  # 买入阈值
            return 1
        elif sell_prob > 0.6:  # 卖出阈值
            return -1
        else:
            return 0
```

## 默认参数

通过工厂获取默认参数：

```python
default_params = ModelFactory.get_default_params("ensemble")
print(default_params)
```

输出：
```python
{
    "voting": "soft",
    "weights": None,
    "models": [
        ("xgboost", {"n_estimators": 1000, "max_depth": 3}),
        ("linear", {"C": 1.0}),
    ],
}
```

## 注意事项

1. **模型兼容性**: 所有基础模型必须实现`BaseModel`接口
2. **特征一致性**: 所有基础模型应使用相同的特征集
3. **内存使用**: 集成模型会保存所有基础模型，内存使用较多
4. **训练时间**: 需要训练所有基础模型，训练时间较长
5. **预测时间**: 需要所有基础模型进行预测，预测时间较长

## 性能优势

集成学习模型的主要优势：
1. **减少过拟合**: 多个模型的组合可以减少单个模型的过拟合
2. **提高稳定性**: 对异常值和噪声更鲁棒
3. **更好的泛化**: 通常比单个模型有更好的泛化能力
4. **不确定性估计**: 软投票提供概率估计，反映预测的不确定性