# 因子测试框架

用于评估交易因子预测能力和有效性的完整框架。

## 目录结构

```
factor_testing/
├── README.md                    # 本文档
├── __init__.py                  # 包初始化
├── test_factor_performance.py   # 主测试脚本
├── example_usage.py             # 使用示例
├── utils/                       # 工具模块
│   ├── __init__.py
│   └── data_loader.py           # 数据加载工具
├── metrics/                     # 评估指标模块
│   ├── __init__.py
│   ├── ic_analysis.py           # IC值分析
│   └── group_backtest.py        # 分组回测
└── visualization/               # 可视化模块
    ├── __init__.py
    └── plot_factors.py          # 因子可视化
```

## 功能特性

### 1. IC值分析
- **信息系数(IC)**: 衡量因子与未来收益的相关性
- **IC衰减**: 分析因子预测能力的持续性
- **ICIR**: IC信息比率，衡量因子稳定性
- **滚动IC**: 动态观察因子表现

### 2. 分组回测
- **分组方法**: 分位数分组、等间距分组
- **分组表现**: 计算各分组收益率、夏普比率、胜率
- **多空组合**: 构建最高组-最低组多空组合
- **单调性检验**: 验证因子区分能力

### 3. 可视化
- **IC分析图表**: 热力图、排名、衰减曲线、分布
- **分组表现图表**: 收益率、夏普比率、胜率
- **相关性矩阵**: 因子间相关性分析
- **综合报告**: 多图组合的综合评估

### 4. 数据支持
- **真实数据**: 支持从`base_tool`加载市场数据
- **模拟数据**: 内置模拟数据生成器
- **多日数据**: 支持加载多日数据合并分析

## 快速开始

### 安装依赖
```bash
# 确保在正确的Python环境
cd /home/jovyan/work/tactics_demo
```

### 运行示例
```bash
# 运行基础示例
cd factor_testing
python example_usage.py

# 运行快速测试（使用模拟数据）
python test_factor_performance.py --quick

# 运行完整测试（使用真实数据）
python test_factor_performance.py --instrument 511520 --start 20260202 --end 20260210
```

### 命令行参数
```bash
python test_factor_performance.py --help

选项:
  --instrument INSTRUMENT  标的ID (默认: 511520)
  --start START          开始日期 YYYYMMDD (默认: 20260202)
  --end END              结束日期 YYYYMMDD (默认: 20260210)
  --mock                 使用模拟数据
  --window WINDOW        特征窗口大小 (默认: 60)
  --step STEP            滑动步长 (默认: 10)
  --output OUTPUT        输出目录 (默认: ./factor_test_results)
  --quick                快速测试（使用模拟数据）
```

## 使用示例

### 1. 基础使用
```python
from utils.data_loader import DataLoader
from metrics.ic_analysis import ICAnalyzer
from metrics.group_backtest import GroupBacktester

# 初始化
data_loader = DataLoader()
ic_analyzer = ICAnalyzer(forward_periods=[1, 5, 10])
group_tester = GroupBacktester(n_groups=5)

# 加载数据
snap_slice = data_loader.load_snapshot_data('511520', '20260202')

# 提取特征
features_df = data_loader.extract_features_from_snapshots(snap_slice)

# IC分析
ic_results = ic_analyzer.analyze_factor_ic(features_df, price_series)

# 分组回测
group_analysis = group_tester.run_complete_analysis(
    features_df, price_series, 'best_bid'
)
```

### 2. 自定义分析
```python
# 自定义IC分析
custom_ic = ICAnalyzer(
    forward_periods=[1, 3, 5, 10, 20, 30]  # 更多周期
)

# 自定义分组
custom_group = GroupBacktester(
    n_groups=10,        # 更多分组
    group_method='equal' # 等间距分组
)

# 计算滚动IC
ic_series = custom_ic.calculate_ic_series(
    factor_df, price_series, 'factor_name',
    period=1, rolling_window=50
)

# 计算IC衰减
ic_decay = custom_ic.calculate_ic_decay(
    factor_df, price_series, 'factor_name',
    max_period=30
)
```

### 3. 可视化
```python
from visualization.plot_factors import FactorVisualizer

visualizer = FactorVisualizer(figsize=(12, 8))

# IC分析图表
ic_fig = visualizer.plot_ic_analysis(ic_results, "因子IC分析")

# 分组表现图表
group_fig = visualizer.plot_group_performance(
    group_performance, "因子分组表现"
)

# 相关性矩阵
corr_fig = visualizer.plot_factor_correlation(
    features_df, "因子相关性"
)

# 综合报告
comp_fig = visualizer.plot_comprehensive_report(
    ic_results, group_performance, features_df,
    "因子综合评估"
)
```

## 评估指标说明

### IC值 (Information Coefficient)
- **范围**: -1 到 1
- **解释**: 
  - > 0.05: 强正相关
  - 0.02-0.05: 中等正相关  
  - 0-0.02: 弱正相关
  - 0: 无相关性
  - < 0: 负相关

### ICIR (IC Information Ratio)
- **公式**: IC均值 / IC标准差
- **解释**: 
  - > 1.0: 优秀
  - 0.5-1.0: 良好
  - 0-0.5: 一般
  - < 0: 不稳定

### 分组回测指标
1. **平均收益率**: 分组平均收益
2. **夏普比率**: 风险调整后收益
3. **胜率**: 正收益比例
4. **最大回撤**: 最大亏损幅度
5. **单调性**: 分组收益是否单调

## 输出文件

测试完成后会在输出目录生成以下文件：

```
factor_test_results/
├── features.csv                 # 特征数据
├── ic_analysis.csv             # IC分析结果
├── factor_stats.csv            # 因子统计
├── group_performance_*.csv     # 分组表现结果
├── ic_analysis.png             # IC分析图表
├── factor_correlation.png      # 相关性矩阵
├── group_performance_*.png     # 分组表现图表
├── comprehensive_report.png    # 综合报告
└── test_report.txt            # 文本报告
```

## 与现有代码集成

### 1. 使用优化后的特征提取器
```python
# 导入优化后的特征提取器
sys.path.append('/home/jovyan/work/tactics_demo/delta')
from features import FeatureExtractor, create_feature

# 使用优化版本（已添加Hurst指数）
extractor = FeatureExtractor(snap_slice, short_window=60)
features = extractor.extract_all()  # 包含hurst_exponent
```

### 2. 在策略中使用因子测试
```python
class EnhancedStrategy:
    def __init__(self):
        self.factor_tester = FactorPerformanceTester()
        
    def evaluate_factors(self, snap_slice):
        # 提取特征
        features_df = self.data_loader.extract_features_from_snapshots(snap_slice)
        
        # 评估因子
        ic_results = self.ic_analyzer.analyze_factor_ic(features_df, price_series)
        
        # 选择最佳因子
        best_factor = self.select_best_factor(ic_results)
        
        return best_factor
```

## 最佳实践

### 1. 因子评估流程
1. **数据准备**: 加载清洗后的市场数据
2. **特征提取**: 计算所有候选因子
3. **IC分析**: 评估因子预测能力
4. **分组验证**: 验证因子区分能力
5. **稳定性检验**: 检查因子表现稳定性
6. **综合评估**: 结合多个指标选择因子

### 2. 避免常见问题
- **过拟合**: 使用样本外数据验证
- **幸存者偏差**: 考虑所有因子，不只是表现好的
- **数据窥探**: 避免使用未来信息
- **多重检验**: 对多个因子进行Bonferroni校正

### 3. 因子选择标准
1. **IC绝对值** > 0.02
2. **ICIR** > 0.5
3. **胜率** > 55%
4. **单调性**显著
5. **稳定性**高（滚动IC波动小）

## 扩展开发

### 添加新评估指标
```python
# 在metrics/目录下创建新模块
class NewMetricAnalyzer:
    def calculate_metric(self, factor_values, returns):
        # 实现新指标
        pass
```

### 添加新可视化
```python
# 在visualization/目录下扩展
class ExtendedVisualizer(FactorVisualizer):
    def plot_new_chart(self, data):
        # 实现新图表
        pass
```

### 集成机器学习
```python
# 可以扩展为ML因子评估
from sklearn.ensemble import RandomForestRegressor

class MLFactorEvaluator:
    def evaluate_with_ml(self, features, returns):
        # 使用机器学习评估因子重要性
        model = RandomForestRegressor()
        model.fit(features, returns)
        importance = model.feature_importances_
        return importance
```

## 故障排除

### 常见问题
1. **无法导入base_tool**
   - 检查路径: `/home/jovyan/base_demo`
   - 确保模块存在

2. **内存不足**
   - 减少数据量（使用`--window`和`--step`参数）
   - 使用模拟数据测试（`--mock`）

3. **可视化错误**
   - 确保matplotlib已安装
   - 检查中文字体配置

4. **IC值全部为0**
   - 检查数据对齐
   - 验证价格序列有效性
   - 确保有足够的数据点

### 调试建议
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据
print(f"数据长度: {len(snap_slice)}")
print(f"特征形状: {features_df.shape}")
print(f"价格序列: {price_series.head()}")

# 验证计算
test_ic = ic_analyzer.calculate_ic(
    features_df['best_bid'], 
    price_series.shift(-1), 
    'spearman'
)
print(f"测试IC值: {test_ic}")
```

## 更新日志

### v1.0.0 (2024-01-20)
- 初始版本发布
- 完整的IC分析框架
- 分组回测功能
- 丰富的可视化图表
- 模拟数据支持
- 与现有特征提取器集成

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目仅供研究使用。