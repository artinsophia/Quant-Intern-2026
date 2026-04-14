# 因子测试框架 (Factor Testing Framework)

一个完整的因子IC等指标计算框架，提供标准化的接口用于因子测试、分析和报告生成。

## 功能特性

- **因子数据管理**: 加载、存储、预处理因子数据
- **IC计算**: Pearson IC、Spearman Rank IC、Kendall Tau
- **综合指标**: IR（信息比率）、换手率、衰减率、分组收益
- **分组测试**: 分层回测、多空组合构建、单调性检验
- **可视化报告**: 自动生成图表和文本报告
- **批量处理**: 支持多个因子的批量测试和比较

## 安装

```bash
# 将factor_testing目录添加到Python路径
import sys
sys.path.append('/path/to/factor_testing')
```

## 依赖库

- pandas >= 1.0.0
- numpy >= 1.18.0
- scipy >= 1.4.0
- matplotlib >= 3.1.0
- seaborn >= 0.10.0
- scikit-learn >= 0.22.0 (用于中性化处理)

## 快速开始

### 1. 基本使用

```python
import pandas as pd
import numpy as np
from factor_testing import FactorData, FactorMetrics, GroupTester, ReportGenerator

# 准备数据（假设已有）
# factor_values: DataFrame, 索引为(date, symbol), 每列为一个因子
# forward_returns: Series, 索引与factor_values一致

# 创建因子数据对象
factor_data = FactorData(factor_values)

# 选择要测试的因子
test_factor = factor_data.get_factor('momentum')

# 计算因子指标
metrics_calculator = FactorMetrics(test_factor, forward_returns)
metrics = metrics_calculator.calculate_all_metrics(n_groups=5, freq='D')

print(f"IC: {metrics['ic']:.4f}")
print(f"IR: {metrics['ir']:.4f}")
print(f"多空组合夏普: {metrics['long_short_sharpe']:.4f}")
```

### 2. 分组测试

```python
# 运行分组测试
group_tester = GroupTester(test_factor, forward_returns)
group_results = group_tester.run_comprehensive_test(n_groups=5)

# 查看多空组合表现
if 'long_short' in group_results:
    ls = group_results['long_short']
    print(f"平均收益: {ls['mean_return'] * 100:.2f}%")
    print(f"夏普比率: {ls['sharpe_ratio']:.3f}")
    print(f"胜率: {ls['win_rate'] * 100:.1f}%")
```

### 3. 生成完整报告

```python
# 生成可视化报告
report_gen = ReportGenerator(
    factor_name='momentum',
    factor_data=test_factor,
    forward_returns=forward_returns
)

# 生成文本摘要
report_text = report_gen.generate_summary_report(
    n_groups=5, method='quantile', freq='D'
)
print(report_text)

# 保存完整报告（包含图表）
report_gen.save_report(
    output_dir='./factor_analysis_report',
    n_groups=5,
    method='quantile',
    freq='D'
)
```

## 核心模块

### FactorData - 因子数据管理

```python
from factor_testing import FactorData

# 从CSV加载
factor_data = FactorData().load_from_csv(
    'factors.csv',
    date_col='date',
    symbol_col='symbol',
    factor_cols=['momentum', 'value', 'quality']
)

# 从字典加载
factor_dict = {
    'momentum': momentum_df,  # DataFrame: date × symbol
    'value': value_df,
    'quality': quality_df
}
factor_data = FactorData().load_from_dict(factor_dict)

# 获取因子数据
momentum = factor_data.get_factor('momentum')
all_factors = factor_data.get_factors(['momentum', 'value'])

# 添加新因子
factor_data.add_factor('new_factor', new_factor_series)

# 获取统计信息
stats = factor_data.get_factor_stats('momentum')
```

### FactorPreprocessor - 因子预处理

```python
from factor_testing.utils import FactorPreprocessor

# 去极值
winsorized = FactorPreprocessor.winsorize(
    factor_series, method='quantile', limits=0.05
)

# 标准化
standardized = FactorPreprocessor.standardize(
    winsorized, method='zscore'
)

# 中性化（需要暴露度数据）
neutralized = FactorPreprocessor.neutralize(
    standardized, exposure_data, method='linear'
)

# 预处理流水线
preprocessing_steps = [
    {'name': 'winsorize', 'params': {'method': 'quantile', 'limits': 0.05}},
    {'name': 'fill_missing', 'params': {'method': 'mean'}},
    {'name': 'standardize', 'params': {'method': 'zscore'}}
]

processed = FactorPreprocessor.pipeline(factor_series, preprocessing_steps)
```

### ICCalculator - IC计算

```python
from factor_testing.metrics import ICCalculator

# 创建计算器
ic_calculator = ICCalculator(factor_series, forward_returns)

# 计算IC
ic_pearson = ic_calculator.calculate_ic(method='pearson')
ic_spearman = ic_calculator.calculate_ic(method='spearman')  # Rank IC
ic_kendall = ic_calculator.calculate_ic(method='kendall')

# 计算IC时间序列
ic_series = ic_calculator.calculate_ic_series(freq='D', method='pearson')

# 计算IC衰减
ic_decay = ic_calculator.calculate_ic_decay(max_lag=10, method='pearson')

# 计算IC统计指标
ic_stats = ic_calculator.calculate_ic_stats(method='pearson', freq='D')
```

### FactorMetrics - 综合指标

```python
from factor_testing.metrics import FactorMetrics

# 创建计算器
metrics_calculator = FactorMetrics(factor_series, forward_returns)

# 计算所有指标
all_metrics = metrics_calculator.calculate_all_metrics(
    n_groups=5, freq='D', method='pearson'
)

# 计算IR
ir = metrics_calculator.calculate_ir(freq='D', method='pearson')

# 计算换手率
turnover = metrics_calculator.calculate_turnover(n_groups=5, freq='D')

# 计算衰减率
decay_info = metrics_calculator.calculate_decay_rate(max_lag=10)

# 计算分组收益
group_returns = metrics_calculator.calculate_group_returns(n_groups=5)

# 批量计算多个因子
batch_metrics = FactorMetrics.batch_calculate_metrics(
    factor_df, forward_returns, n_groups=5, freq='D'
)
```

### GroupTester - 分组测试

```python
from factor_testing.analysis import GroupTester

# 创建测试器
group_tester = GroupTester(factor_series, forward_returns)

# 创建分组
groups = group_tester.create_groups(n_groups=5, method='quantile')

# 计算分组表现
performance = group_tester.calculate_group_performance(
    n_groups=5, method='quantile', rebalance_freq='D'
)

# 创建多空组合
long_short_returns = group_tester.create_long_short_portfolio(
    n_groups=5, method='quantile', top_group=0, bottom_group=4
)

# 计算分组换手率
turnover_results = group_tester.calculate_group_turnover(n_groups=5)

# 运行全面测试
comprehensive_results = group_tester.run_comprehensive_test(
    n_groups=5, method='quantile', rebalance_freq='D'
)

# 比较多个因子
factor_dict = {
    'factor1': factor1_series,
    'factor2': factor2_series,
    'factor3': factor3_series
}
comparison = GroupTester.compare_factors(
    factor_dict, forward_returns, n_groups=5
)
```

### ReportGenerator - 报告生成

```python
from factor_testing.analysis import ReportGenerator

# 创建报告生成器
report_gen = ReportGenerator(
    factor_name='momentum',
    factor_data=factor_series,
    forward_returns=forward_returns
)

# 生成图表
fig1 = report_gen.generate_factor_distribution_plot()
fig2 = report_gen.generate_ic_analysis_plot(freq='D', method='pearson')
fig3 = report_gen.generate_group_performance_plot(n_groups=5)
fig4 = report_gen.generate_turnover_analysis_plot(n_groups=5)

# 生成文本报告
report_text = report_gen.generate_summary_report(
    n_groups=5, method='quantile', freq='D', ic_method='pearson'
)

# 保存完整报告
report_gen.save_report(
    output_dir='./factor_report',
    n_groups=5,
    method='quantile',
    freq='D',
    ic_method='pearson'
)
```

## 数据格式要求

### 因子数据格式

推荐使用MultiIndex格式：

```python
# DataFrame格式，索引为(date, symbol)
factor_df = pd.DataFrame({
    'momentum': momentum_values,
    'value': value_values,
    'quality': quality_values
}, index=pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol']))
```

或者使用面板格式字典：

```python
# 字典格式，每个因子为date × symbol的DataFrame
factor_dict = {
    'momentum': momentum_panel,  # DataFrame: dates × symbols
    'value': value_panel,
    'quality': quality_panel
}
```

### 收益数据格式

```python
# Series格式，索引与因子数据一致
forward_returns = pd.Series(
    returns_values,
    index=pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol']),
    name='forward_return'
)
```

## 示例程序

运行示例程序查看完整功能：

```python
python factor_testing/example.py
```

## 与现有项目集成

如果已有因子数据和收益数据，可以这样集成：

```python
# 1. 导入模块
from factor_testing import FactorData, FactorMetrics, GroupTester, ReportGenerator

# 2. 准备数据（假设已有）
# factor_values: 因子值，DataFrame格式
# returns: 未来收益，Series格式

# 3. 创建因子数据对象
factor_data = FactorData(factor_values)

# 4. 选择要测试的因子
test_factor = factor_data.get_factor('your_factor_name')

# 5. 计算因子指标
metrics_calculator = FactorMetrics(test_factor, returns)
metrics = metrics_calculator.calculate_all_metrics(n_groups=5, freq='D')

# 6. 运行分组测试
group_tester = GroupTester(test_factor, returns)
group_results = group_tester.run_comprehensive_test(n_groups=5)

# 7. 生成报告
report_gen = ReportGenerator(
    factor_name='your_factor_name',
    factor_data=test_factor,
    forward_returns=returns
)

# 保存完整报告
report_gen.save_report(
    output_dir='./factor_analysis_report',
    n_groups=5,
    method='quantile',
    freq='D'
)
```

## 输出报告内容

保存的报告包含：

1. **图表文件**:
   - `factor_distribution.png`: 因子分布图
   - `ic_analysis.png`: IC分析图
   - `group_performance.png`: 分组表现图
   - `turnover_analysis.png`: 换手率分析图

2. **文本文件**:
   - `factor_report.txt`: 文本摘要报告

3. **数据文件**:
   - `factor_data.csv`: 因子数据
   - `forward_returns.csv`: 未来收益数据

## 注意事项

1. **数据对齐**: 确保因子数据和收益数据的索引完全一致
2. **缺失值处理**: 框架会自动处理缺失值，但建议先进行适当的预处理
3. **计算效率**: 对于大规模数据，建议使用适当的数据分块处理
4. **内存使用**: 分组测试和报告生成可能需要较多内存，特别是对于大量标的和长时间序列

## 扩展开发

框架设计为可扩展的，可以通过以下方式扩展功能：

1. **自定义指标**: 继承`FactorMetrics`类添加新的指标计算方法
2. **自定义分组方法**: 在`GroupTester`中添加新的分组逻辑
3. **自定义图表**: 继承`ReportGenerator`类添加新的可视化图表
4. **自定义报告格式**: 修改`ReportGenerator`的输出格式

## 许可证

MIT License