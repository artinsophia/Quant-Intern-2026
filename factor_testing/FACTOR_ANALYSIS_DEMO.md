# 因子测试框架应用示例

## 概述

本示例演示如何使用因子测试框架对518880（黄金ETF）的3月数据进行分析，计算IC、IR、分组测试等指标。

## 文件说明

1. **`factor_analysis_518880.py`** - 完整的Python脚本，可以直接运行
2. **`factor_analysis_demo.ipynb`** - Jupyter Notebook版本，适合交互式分析
3. **`factor_analysis_518880.ipynb`** - 详细的Jupyter Notebook（如果创建成功）

## 功能特性

### 1. 数据准备和特征提取
- 从delta模块的`features.py`读取特征提取函数
- 加载518880的3月数据（20260301-20260331）
- 使用滑动窗口提取特征

### 2. 因子数据管理
- 创建FactorData对象管理因子数据
- 支持MultiIndex格式（date × time_mark）
- 因子选择和数据对齐

### 3. 因子预处理
- 去极值处理（分位数法）
- 标准化处理（Z-score）
- 预处理流水线

### 4. IC分析
- Pearson IC计算
- Spearman Rank IC计算
- IC时间序列分析
- IC衰减分析
- IR（信息比率）计算

### 5. 综合指标计算
- 分组收益计算
- 多空组合构建
- 换手率分析
- 衰减率分析
- 批量因子指标计算

### 6. 分组测试
- 分层回测功能
- 分组换手率分析
- 单调性检验
- 多因子比较

### 7. 报告生成
- 因子分布可视化
- IC分析图表
- 分组表现图表
- 换手率分析图表
- 文本摘要报告

## 使用的因子

从delta模块的`features.py`中提取以下因子：

1. **volatility** - 波动率因子
2. **spread** - 买卖价差因子
3. **WAMP** - 加权平均中间价因子
4. **alpha_03** - 买卖量差因子
5. **alpha_04** - 交易频率因子
6. **alpha_05** - 买卖交易次数差因子
7. **hurst_exponent** - 赫斯特指数因子

## 使用方法

### 方法1：运行Python脚本
```bash
cd /home/jovyan/work/tactics_demo/factor_testing
python factor_analysis_518880.py
```

### 方法2：使用Jupyter Notebook
1. 打开Jupyter Notebook
2. 导航到`tactics_demo/factor_testing/`目录
3. 打开`factor_analysis_demo.ipynb`
4. 按顺序运行所有单元格

### 方法3：交互式使用
```python
# 在Python环境中导入模块
import sys
sys.path.append('/home/jovyan/work/tactics_demo/factor_testing')

from factor_testing import FactorData, FactorMetrics, GroupTester, ReportGenerator
from delta import create_feature, get_trade_dates
import base_tool

# 然后按照脚本中的步骤进行分析
```

## 输出结果

运行分析后，将生成：

### 1. 控制台输出
- 因子基本统计信息
- IC分析结果
- 综合指标比较
- 分组测试结果

### 2. 可视化图表
- 因子分布图
- IC分析图
- 分组表现图
- 换手率分析图

### 3. 保存的报告
在`./factor_analysis_report_*`目录中保存：
- `factor_distribution.png` - 因子分布图
- `ic_analysis.png` - IC分析图
- `group_performance.png` - 分组表现图
- `turnover_analysis.png` - 换手率分析图
- `factor_report.txt` - 文本摘要报告
- `factor_data.csv` - 因子数据
- `forward_returns.csv` - 未来收益数据

## 依赖要求

### Python包
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

### 项目依赖
- `base_tool`模块（位于`/home/jovyan/base_demo`）
- `delta`模块（位于`tactics_demo/delta/`）
- `factor_testing`框架（位于`tactics_demo/factor_testing/`）

## 自定义分析

### 修改分析参数
```python
# 修改标的和日期
instrument_id = '511520'  # 改为其他ETF
start_date = '20260101'   # 修改开始日期
end_date = '20260131'     # 修改结束日期

# 修改特征提取参数
param_dict = {
    'short_window': 30,   # 缩短窗口
    'long_window': 150,   # 缩短长期窗口
    'stride': 2          # 增加步长
}
```

### 添加新因子
```python
# 在特征提取后添加新因子
features_df['new_factor'] = features_df['alpha_03'] * features_df['volatility']

# 然后在因子选择中包含新因子
factor_columns.append('new_factor')
```

### 修改分组测试参数
```python
# 修改分组数量和方法
test_results = group_tester.run_comprehensive_test(
    n_groups=10,           # 增加分组数量
    method='equal',        # 改为等权分组
    rebalance_freq='W'     # 改为周度再平衡
)
```

## 故障排除

### 1. 导入错误
```
ModuleNotFoundError: No module named 'factor_testing'
```
解决方案：确保Python路径包含factor_testing目录
```python
import sys
sys.path.append('/home/jovyan/work/tactics_demo/factor_testing')
```

### 2. 数据加载失败
```
日期 20260301 特征提取出错: ...
```
解决方案：检查日期格式和数据可用性
- 确保日期格式为'YYYYMMDD'
- 检查base_tool模块是否正确安装
- 确认指定日期有交易数据

### 3. 内存不足
解决方案：减少数据量
- 增加`stride`参数减少样本数量
- 减少分析日期范围
- 选择部分因子进行分析

## 性能优化建议

### 1. 数据预处理
- 使用`stride`参数减少样本数量
- 提前过滤无效数据
- 使用适当的数据类型

### 2. 计算优化
- 批量计算多个因子
- 使用向量化操作
- 避免不必要的重复计算

### 3. 内存管理
- 及时释放不再使用的变量
- 使用分块处理大数据
- 选择适当的数据结构

## 扩展应用

### 1. 多标的分析
```python
instruments = ['518880', '511520', '511090']
for instrument_id in instruments:
    # 对每个标的进行分析
    pass
```

### 2. 时间序列分析
```python
# 按月分析
for month in ['202601', '202602', '202603']:
    start_date = month + '01'
    end_date = month + '31'
    # 按月分析
```

### 3. 因子组合
```python
# 创建因子组合
factor_combination = 0.5 * factor1 + 0.3 * factor2 + 0.2 * factor3
```

## 联系和支持

如有问题或建议，请参考：
- 因子测试框架文档：`README.md`
- 示例代码：`example.py`
- 框架验证：`verify_structure.py`

## 更新日志

### v1.0.0 (2026-04-14)
- 初始版本发布
- 完整的因子分析流程
- 支持518880 3月数据分析
- 提供Python脚本和Jupyter Notebook两种方式