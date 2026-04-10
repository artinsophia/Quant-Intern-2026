# 市场数据分析工具

本工具用于分析每日市场数据，统计波动、涨跌、成交量等指标，帮助量化交易策略的研究和评估。

## 功能特性

- **每日统计指标**: 计算开盘价、收盘价、最高价、最低价、平均价、涨跌幅等
- **波动率分析**: 计算日内波动率、价格波幅、最大回撤等
- **成交量分析**: 统计成交量、成交笔数等
- **买卖价差分析**: 分析买卖价差及其变化
- **策略表现评估**: 评估交易策略的盈利、回撤、仓位变化等
- **多日分析**: 支持多日数据的批量分析和趋势判断
- **可视化图表**: 生成直观的图表展示每日指标变化
- **命令行工具**: 提供便捷的命令行接口

## 安装依赖

```bash
pip install pandas numpy matplotlib
```

## 使用方法

### 1. Python API 使用

```python
from daily_analysis import DailyMarketAnalyzer

# 初始化分析器
analyzer = DailyMarketAnalyzer()

# 分析单个交易日
stats = analyzer.analyze_single_day('518880', '20260105')
if stats:
    print(f"涨跌幅: {stats['price_change_pct']:.2f}%")
    print(f"日内波幅: {stats['price_range_pct']:.2f}%")

# 分析多日数据
df_stats = analyzer.analyze_multiple_days('518880', '20260105', '20260110')

# 生成汇总报告
report = analyzer.generate_summary_report(df_stats)
print(report)

# 绘制图表
analyzer.plot_daily_metrics(df_stats)
```

### 2. 命令行工具使用

```bash
# 分析单个交易日
python cli_tool.py single 518880 20260105

# 分析多日数据并生成图表
python cli_tool.py multi 518880 20260105 20260110 --plot --output results.csv

# 批量分析多个标的
python cli_tool.py batch 511520 511090 518880 20260301 20260305

# 从CSV文件生成报告
python cli_tool.py report market_analysis_518880_20260105_20260110.csv
```

### 3. Jupyter Notebook 使用

打开 `market_analysis_demo.ipynb` 查看完整的使用示例。

## 输出指标说明

### 价格指标
- `open_price`: 开盘价
- `close_price`: 收盘价  
- `high_price`: 最高价
- `low_price`: 最低价
- `avg_price`: 平均价
- `price_change`: 价格变化
- `price_change_pct`: 价格变化百分比
- `price_std`: 价格标准差
- `price_range`: 价格范围
- `price_range_pct`: 价格范围百分比

### 波动率指标
- `intraday_volatility`: 日内波动率（年化）
- `max_intraday_drawdown`: 最大日内回撤

### 成交量指标
- `total_volume`: 总成交量
- `avg_volume`: 平均成交量
- `total_num_trades`: 总成交笔数
- `avg_num_trades`: 平均成交笔数

### 买卖价差指标
- `avg_spread`: 平均买卖价差
- `avg_spread_bps`: 平均买卖价差（基点）
- `max_spread`: 最大买卖价差
- `min_spread`: 最小买卖价差

### 策略表现指标
- `final_profit`: 最终盈利
- `max_profit`: 最大盈利
- `min_profit`: 最小盈利
- `profit_range`: 盈利范围
- `max_drawdown`: 最大回撤
- `position_changes`: 仓位变化次数
- `long_periods`: 多头时段数
- `short_periods`: 空头时段数
- `neutral_periods`: 中性时段数

## 数据源

工具从 `/home/jovyan/work/backtest_result/` 目录读取回测结果文件，支持以下格式：
- `{instrument_id}_{trade_ymd}_{strategy_name}.pkl`
- `{instrument_id}_{trade_ymd}_{strategy_name}_pro.pkl`
- `{instrument_id}_{trade_ymd}_{strategy_name}_simple.pkl`

## 支持的标的

- `511520`: 国债ETF
- `511090`: 企业债ETF  
- `518880`: 黄金ETF

## 示例输出

### 单日分析示例
```
标的: 518880
日期: 20260105
数据点: 23400
时段: 09:30:00 - 15:00:00

价格信息:
  开盘价: 4.5320
  收盘价: 4.5480
  最高价: 4.5520
  最低价: 4.5280
  平均价: 4.5402
  涨跌幅: 0.0160 (0.35%)
  价格标准差: 0.0052
  价格波幅: 0.0240 (0.53%)

波动率信息:
  日内波动率(年化): 12.45%
  最大日内回撤: -0.22%

买卖价差信息:
  平均价差: 0.0010 (2.20 bps)
  最大价差: 0.0020
  最小价差: 0.0005

策略表现信息:
  最终盈利: 152.34
  最大盈利: 185.67
  最小盈利: -23.45
  盈利范围: 209.12
  最大回撤: -45.23
  仓位变化次数: 8
  多头时段: 15600
  空头时段: 0
  中性时段: 7800
```

### 多日汇总报告示例
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

成交量汇总:
  平均每日: 1250000
  最大值: 1850000
  最小值: 850000

策略表现汇总:
  总盈利天数: 3
  总亏损天数: 1
  平均日盈利: 128.45
  累计总盈利: 513.80
  最大单日盈利: 185.67
  最大单日亏损: -45.23

============================================================
```

## 注意事项

1. 确保回测结果文件存在于 `/home/jovyan/work/backtest_result/` 目录
2. 数据文件应为pickle格式，包含时间序列数据
3. 建议使用Python 3.8+版本
4. 图表生成需要matplotlib库支持

## 扩展开发

如需添加新的分析指标，可以修改 `daily_analysis.py` 中的 `_calculate_daily_stats` 方法。

如需支持新的数据格式，可以修改 `analyze_single_day` 方法中的数据加载逻辑。