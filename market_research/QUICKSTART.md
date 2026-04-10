# 快速开始指南

## 概述

本工具用于分析每日市场数据，统计波动、涨跌、成交量等指标，帮助量化交易策略的研究和评估。

## 文件结构

```
market_research/
├── daily_analysis.py          # 核心分析工具
├── market_analysis_demo.ipynb # Jupyter notebook示例
├── cli_tool.py               # 命令行工具
├── advanced_visualization.py # 高级可视化工具
├── test_analysis.py          # 测试脚本
├── verify_structure.py       # 项目验证脚本
├── README.md                 # 完整文档
└── QUICKSTART.md            # 本快速开始指南
```

## 快速开始

### 1. 基本使用（Python API）

```python
# 导入分析器
from daily_analysis import DailyMarketAnalyzer

# 初始化
analyzer = DailyMarketAnalyzer()

# 分析单个交易日
stats = analyzer.analyze_single_day('518880', '20260105')
if stats:
    print(f"涨跌幅: {stats['price_change_pct']:.2f}%")
    print(f"日内波幅: {stats['price_range_pct']:.2f}%")

# 分析多日数据
df_stats = analyzer.analyze_multiple_days('518880', '20260105', '20260110')

# 生成报告
report = analyzer.generate_summary_report(df_stats)
print(report)
```

### 2. 命令行使用

```bash
# 分析单个交易日
python cli_tool.py single 518880 20260105

# 分析多日数据
python cli_tool.py multi 518880 20260105 20260110 --output results.csv

# 批量分析
python cli_tool.py batch 511520 511090 518880 20260301 20260305
```

### 3. Jupyter Notebook

打开 `market_analysis_demo.ipynb` 查看完整示例。

## 核心功能

### 每日统计指标
- **价格指标**: 开盘价、收盘价、最高价、最低价、平均价、涨跌幅
- **波动率指标**: 日内波动率、价格波幅、最大回撤
- **成交量指标**: 总成交量、平均成交量、成交笔数
- **买卖价差**: 平均价差、最大最小价差（基点）
- **策略表现**: 最终盈利、最大回撤、仓位变化

### 多日分析
- 批量处理多个交易日
- 生成汇总统计报告
- 计算滚动统计指标
- 趋势分析和相关性分析

### 可视化
- 价格走势图
- 涨跌幅分布图
- 波动率热力图
- 成交量分析图
- 策略表现图表
- 综合仪表板

## 数据源

工具从 `/home/jovyan/work/backtest_result/` 读取回测结果文件，支持以下格式：
- `{标的}_{日期}_{策略}.pkl`
- `{标的}_{日期}_{策略}_pro.pkl`
- `{标的}_{日期}_{策略}_simple.pkl`

## 支持的标的

- `511520`: 国债ETF
- `511090`: 企业债ETF
- `518880`: 黄金ETF

## 示例输出

### 单日分析
```
标的: 518880
日期: 20260105
数据点: 23400
时段: 09:30:00 - 15:00:00
涨跌幅: 0.35%
日内波幅: 0.53%
策略盈利: 152.34
```

### 多日报告
```
============================================================
市场数据分析报告
============================================================

分析时间段: 20260105 至 20260110
标的数量: 1
总交易日数: 4

价格表现汇总:
  平均日涨跌幅: 0.28%
  日涨跌幅标准差: 0.42%
  最大单日涨幅: 0.75%
  最大单日跌幅: -0.15%
  上涨天数: 3
  下跌天数: 1
  平盘天数: 0

波动率汇总:
  平均日内波幅: 0.62%
  最大日内波幅: 0.85%
  最小日内波幅: 0.45%
  平均日内波动率(年化): 14.32%
============================================================
```

## 故障排除

### 1. 导入错误
```
ModuleNotFoundError: No module named 'pandas'
```
**解决方案**: 安装依赖
```bash
pip install pandas numpy matplotlib
```

### 2. 无数据错误
```
未找到数据
```
**解决方案**: 
- 检查回测结果文件是否存在
- 确认文件命名格式正确
- 验证标的代码和日期格式

### 3. 文件格式错误
```
未知的数据格式
```
**解决方案**: 
- 确保pickle文件包含正确格式的数据
- 检查文件是否损坏

## 下一步

1. 查看 `market_analysis_demo.ipynb` 获取完整示例
2. 阅读 `README.md` 了解详细功能
3. 尝试 `advanced_visualization.py` 的高级图表功能
4. 根据需要扩展 `daily_analysis.py` 添加自定义指标

## 获取帮助

- 查看代码注释和文档字符串
- 运行测试脚本: `python test_analysis.py`
- 验证项目结构: `python verify_structure.py`
- 参考示例notebook中的完整用法