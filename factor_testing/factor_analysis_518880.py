#!/usr/bin/env python3
"""
因子测试框架应用示例 - 518880 3月数据分析

这个脚本演示如何使用因子测试框架对518880的3月数据进行分析，
计算IC、IR、分组测试等指标。

使用方法：
1. 可以直接运行此脚本：python factor_analysis_518880.py
2. 或复制代码到Jupyter Notebook中运行
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置显示选项
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 150)

# 设置绘图样式
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# 添加路径
sys.path.append("/home/jovyan/base_demo")
sys.path.append("/home/jovyan/work/tactics_demo")
sys.path.append("/home/jovyan/work/tactics_demo/factor_testing")

print("=" * 80)
print("因子测试框架应用示例 - 518880 3月数据分析")
print("=" * 80)

# 导入因子测试框架
try:
    from factor_testing import (
        FactorData,
        FactorPreprocessor,
        FactorMetrics,
        GroupTester,
        ReportGenerator,
    )

    print("✓ 因子测试框架导入成功")
except ImportError as e:
    print(f"✗ 导入因子测试框架失败: {e}")
    print("请确保factor_testing目录在Python路径中")
    sys.exit(1)

# 导入delta模块用于特征提取
try:
    from delta import (
        FeatureExtractor,
        create_feature,
        get_trade_dates,
        split_dates_by_range,
    )

    print("✓ delta模块导入成功")
except ImportError as e:
    print(f"✗ 导入delta模块失败: {e}")
    sys.exit(1)

# 导入基础工具
try:
    import base_tool

    print("✓ base_tool导入成功")
except ImportError as e:
    print(f"✗ 导入base_tool失败: {e}")
    sys.exit(1)

print("\n所有模块导入成功!")

# ============================================================================
# 3. 数据准备和特征提取
# ============================================================================

print("\n" + "=" * 80)
print("3. 数据准备和特征提取")
print("=" * 80)

# 设置参数
instrument_id = "518880"
start_date = "20260301"
end_date = "20260331"

# 特征提取参数
param_dict = {
    "instrument_id": instrument_id,
    "short_window": 60,
    "long_window": 300,
    "stride": 1,
}

print(f"分析标的: {instrument_id}")
print(f"分析期间: {start_date} 到 {end_date}")

# 获取交易日
all_trade_dates = get_trade_dates()

# 筛选3月数据
march_dates = [
    date for date in all_trade_dates if date >= start_date and date <= end_date
]

print(f"\n3月总交易日数量: {len(march_dates)}")
print(
    f"交易日: {march_dates[:5]}...{march_dates[-5:] if len(march_dates) > 10 else ''}"
)


def extract_features_for_date(trade_ymd, instrument_id, param_dict):
    """提取单日特征"""
    try:
        # 加载数据
        snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
        if not snap_list:
            return None

        # 提取特征
        features_list = []
        times_list = []

        short_window = param_dict.get("short_window", 60)
        stride = param_dict.get("stride", 1)

        # 滑动窗口提取特征
        for i in range(short_window, len(snap_list), stride):
            snap_slice = snap_list[i - short_window : i]
            features = create_feature(snap_slice, short_window)
            features_list.append(features)
            times_list.append(snap_list[i]["time_mark"])

        # 转换为DataFrame
        if features_list:
            df = pd.DataFrame(features_list)
            df["time_mark"] = times_list
            df["date"] = trade_ymd
            return df

    except Exception as e:
        print(f"日期 {trade_ymd} 特征提取出错: {e}")

    return None


# 提取所有日期的特征
print("\n开始提取特征...")

all_features = []
for trade_ymd in march_dates:
    print(f"处理日期: {trade_ymd}", end="\r")
    features_df = extract_features_for_date(trade_ymd, instrument_id, param_dict)
    if features_df is not None:
        all_features.append(features_df)

print("\n特征提取完成!")

# 合并所有特征
if all_features:
    features_df = pd.concat(all_features, ignore_index=True)
    print(f"\n特征数据形状: {features_df.shape}")
    print(f"特征列: {list(features_df.columns)}")

    # 显示前几行
    print("\n前5行数据:")
    print(features_df.head())
else:
    print("没有提取到特征数据!")
    sys.exit(1)

# ============================================================================
# 4. 创建未来收益数据
# ============================================================================

print("\n" + "=" * 80)
print("4. 创建未来收益数据")
print("=" * 80)


def calculate_forward_returns(features_df, forward_periods=60):
    """计算未来收益"""
    # 按日期和时间排序
    features_df = features_df.sort_values(["date", "time_mark"]).reset_index(drop=True)

    # 获取价格数据（使用WAMP作为价格代理）
    prices = features_df["WAMP"].values

    # 计算未来收益
    forward_returns = np.full(len(prices), np.nan)

    for i in range(len(prices) - forward_periods):
        current_price = prices[i]
        future_price = prices[i + forward_periods]

        if current_price > 0:
            # 计算对数收益
            forward_returns[i] = np.log(future_price / current_price)

    return forward_returns


# 计算未来收益（未来60秒）
forward_returns = calculate_forward_returns(features_df, forward_periods=60)

# 创建收益Series
returns_series = pd.Series(
    forward_returns,
    index=pd.MultiIndex.from_arrays(
        [features_df["date"], features_df["time_mark"]], names=["date", "time_mark"]
    ),
    name="forward_return",
)

# 移除NaN值
returns_series = returns_series.dropna()

print(f"未来收益数据形状: {returns_series.shape}")
print(f"\n收益统计:")
print(returns_series.describe())

# ============================================================================
# 5. 准备因子数据
# ============================================================================

print("\n" + "=" * 80)
print("5. 准备因子数据")
print("=" * 80)

# 选择要测试的因子
factor_columns = [
    "volatility",  # 波动率
    "spread",  # 买卖价差
    "WAMP",  # 加权平均中间价
    "alpha_03",  # 买卖量差
    "alpha_04",  # 交易频率
    "alpha_05",  # 买卖交易次数差
    "hurst_exponent",  # 赫斯特指数
]

# 创建因子DataFrame
factor_df = features_df[["date", "time_mark"] + factor_columns].copy()

# 设置MultiIndex
factor_df.set_index(["date", "time_mark"], inplace=True)

# 对齐因子和收益数据
common_index = factor_df.index.intersection(returns_series.index)
factor_df = factor_df.loc[common_index]
returns_series = returns_series.loc[common_index]

print(f"对齐后因子数据形状: {factor_df.shape}")
print(f"对齐后收益数据形状: {returns_series.shape}")

# 显示因子基本信息
print("\n因子描述统计:")
print(factor_df.describe())

# 创建FactorData对象
factor_data = FactorData(factor_df)
print(f"\nFactorData创建成功: {factor_data}")

# ============================================================================
# 6. 因子预处理
# ============================================================================

print("\n" + "=" * 80)
print("6. 因子预处理")
print("=" * 80)

# 选择一个因子进行预处理演示
test_factor_name = "volatility"
test_factor = factor_data.get_factor(test_factor_name)

print(f"原始 {test_factor_name} 因子统计:")
print(f"  均值: {test_factor.mean():.6f}")
print(f"  标准差: {test_factor.std():.6f}")
print(f"  最小值: {test_factor.min():.6f}")
print(f"  最大值: {test_factor.max():.6f}")

# 去极值处理
winsorized = FactorPreprocessor.winsorize(test_factor, method="quantile", limits=0.05)

print(f"\n去极值后 {test_factor_name} 因子统计:")
print(f"  均值: {winsorized.mean():.6f}")
print(f"  标准差: {winsorized.std():.6f}")
print(f"  最小值: {winsorized.min():.6f}")
print(f"  最大值: {winsorized.max():.6f}")

# 标准化处理
standardized = FactorPreprocessor.standardize(winsorized, method="zscore")

print(f"\n标准化后 {test_factor_name} 因子统计:")
print(f"  均值: {standardized.mean():.6f}")
print(f"  标准差: {standardized.std():.6f}")

# 预处理流水线
preprocessing_steps = [
    {"name": "winsorize", "params": {"method": "quantile", "limits": 0.05}},
    {"name": "fill_missing", "params": {"method": "mean"}},
    {"name": "standardize", "params": {"method": "zscore"}},
]

processed = FactorPreprocessor.pipeline(test_factor, preprocessing_steps)

print(f"\n流水线处理后 {test_factor_name} 因子统计:")
print(f"  均值: {processed.mean():.6f}")
print(f"  标准差: {processed.std():.6f}")

# ============================================================================
# 7. IC分析
# ============================================================================

print("\n" + "=" * 80)
print("7. IC分析")
print("=" * 80)

from factor_testing.metrics import ICCalculator

# 测试波动率因子
volatility_factor = factor_data.get_factor("volatility")

# 创建IC计算器
ic_calculator = ICCalculator(volatility_factor, returns_series)

# 计算Pearson IC
ic_pearson = ic_calculator.calculate_ic(method="pearson")
print(f"波动率因子 Pearson IC: {ic_pearson:.4f}")

# 计算Rank IC (Spearman)
ic_spearman = ic_calculator.calculate_ic(method="spearman")
print(f"波动率因子 Rank IC: {ic_spearman:.4f}")

# 计算IC时间序列（按日期）
ic_series = ic_calculator.calculate_ic_series(freq="D", method="pearson")

print(f"\nIC时间序列统计:")
print(f"  均值: {ic_series.mean():.4f}")
print(f"  标准差: {ic_series.std():.4f}")
print(
    f"  IR: {ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan:.4f}"
)
print(f"  正IC比例: {(ic_series > 0).mean() * 100:.1f}%")

# 计算IC衰减
ic_decay = ic_calculator.calculate_ic_decay(max_lag=10, method="pearson")

print(f"\nIC衰减（前5阶）:")
for lag in range(1, 6):
    if lag in ic_decay.index:
        print(f"  滞后{lag}阶: {ic_decay[lag]:.4f}")

# 计算IC统计指标
ic_stats = ic_calculator.calculate_ic_stats(method="pearson", freq="D")

print(f"\nIC统计指标:")
for key, value in ic_stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# ============================================================================
# 8. 综合指标计算
# ============================================================================

print("\n" + "=" * 80)
print("8. 综合指标计算")
print("=" * 80)

# 创建因子指标计算器
metrics_calculator = FactorMetrics(factor_data.get_factor("volatility"), returns_series)

# 计算所有指标
all_metrics = metrics_calculator.calculate_all_metrics(
    n_groups=5, freq="D", method="pearson"
)

print("波动率因子综合指标:")
print("=" * 50)

# 分组显示指标
metric_groups = {
    "IC相关指标": ["ic", "rank_ic", "ic_mean", "ic_std", "ir", "ic_positive_rate"],
    "分组表现指标": [
        "group_0_mean_return",
        "group_4_mean_return",
        "long_short_mean_return",
        "group_0_sharpe",
        "group_4_sharpe",
        "long_short_sharpe",
    ],
    "衰减和换手率": ["decay_half_life", "decay_rate", "avg_turnover", "max_turnover"],
}

for group_name, metric_keys in metric_groups.items():
    print(f"\n{group_name}:")
    print("-" * 30)
    for key in metric_keys:
        if key in all_metrics:
            value = all_metrics[key]
            if isinstance(value, float):
                # 根据指标类型格式化输出
                if "return" in key or "sharpe" in key:
                    print(f"  {key}: {value:.4f}")
                elif "rate" in key:
                    print(f"  {key}: {value * 100:.1f}%")
                elif key == "ic_positive_rate":
                    print(f"  {key}: {value * 100:.1f}%")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

# ============================================================================
# 9. 批量计算多个因子
# ============================================================================

print("\n" + "=" * 80)
print("9. 批量计算多个因子")
print("=" * 80)

# 获取所有因子数据
all_factors_df = factor_data.get_factors()

# 批量计算指标
batch_metrics = FactorMetrics.batch_calculate_metrics(
    all_factors_df, returns_series, n_groups=5, freq="D", method="pearson"
)

print("所有因子指标比较:")
print("=" * 60)

# 选择关键指标显示
key_metrics = [
    "ic",
    "ir",
    "rank_ic",
    "long_short_sharpe",
    "long_short_mean_return",
    "avg_turnover",
]

display_df = batch_metrics[key_metrics].copy()

# 格式化显示
for col in display_df.columns:
    if col in ["ic", "ir", "rank_ic", "long_short_sharpe"]:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN"
        )
    elif col == "long_short_mean_return":
        display_df[col] = display_df[col].apply(
            lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "NaN"
        )
    elif col == "avg_turnover":
        display_df[col] = display_df[col].apply(
            lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else "NaN"
        )

print(display_df)

# 按IR排序
print("\n按IR排序:")
print("-" * 60)
ir_sorted = batch_metrics.sort_values("ir", ascending=False)["ir"]
for factor, ir in ir_sorted.items():
    print(f"  {factor:20}: {ir:.4f}")

# 按多空组合夏普排序
print("\n按多空组合夏普排序:")
print("-" * 60)
sharpe_sorted = batch_metrics.sort_values("long_short_sharpe", ascending=False)[
    "long_short_sharpe"
]
for factor, sharpe in sharpe_sorted.items():
    print(f"  {factor:20}: {sharpe:.4f}")

# ============================================================================
# 10. 分组测试
# ============================================================================

print("\n" + "=" * 80)
print("10. 分组测试")
print("=" * 80)

# 选择IR最高的因子
best_factor_name = (
    batch_metrics["ir"].idxmax() if not batch_metrics.empty else "volatility"
)
best_factor = factor_data.get_factor(best_factor_name)

print(f"对最佳因子 '{best_factor_name}' 进行分组测试:")
print("=" * 60)

# 创建分组测试器
group_tester = GroupTester(best_factor, returns_series)

# 运行全面测试
test_results = group_tester.run_comprehensive_test(
    n_groups=5, method="quantile", rebalance_freq="D"
)

# 显示分组表现
if "group_performance" in test_results:
    print("\n分组表现:")
    print("-" * 40)

    for group_name, perf in test_results["group_performance"].items():
        print(f"\n{group_name}:")
        for metric, value in perf.items():
            if isinstance(value, float):
                if "return" in metric or "sharpe" in metric or "ratio" in metric:
                    print(f"  {metric}: {value:.4f}")
                elif "rate" in metric:
                    print(f"  {metric}: {value * 100:.1f}%")
                elif "drawdown" in metric:
                    print(f"  {metric}: {value * 100:.2f}%")
                else:
                    print(f"  {metric}: {value:.2f}")

# 显示换手率
if "turnover" in test_results:
    print("\n分组换手率:")
    print("-" * 40)

    if "long_short" in test_results["turnover"]:
        ls_turnover = test_results["turnover"]["long_short"]
        print(f"多空组合:")
        print(f"  平均换手率: {ls_turnover.get('mean_turnover', 0) * 100:.2f}%")
        print(f"  换手率标准差: {ls_turnover.get('std_turnover', 0) * 100:.2f}%")
        print(f"  最大换手率: {ls_turnover.get('max_turnover', 0) * 100:.2f}%")

# 显示单调性
if "monotonicity" in test_results:
    mono = test_results["monotonicity"]
    print(f"\n分组单调性 (Spearman相关系数): {mono.get('spearman_corr', 0):.4f}")

    if "group_means" in mono:
        print("各分组平均收益:")
        for group, mean_return in mono["group_means"].items():
            print(f"  分组{group}: {mean_return * 100:.2f}%")

# ============================================================================
# 11. 生成报告
# ============================================================================

print("\n" + "=" * 80)
print("11. 生成报告")
print("=" * 80)

# 选择要生成报告的因子
report_factor_name = best_factor_name
report_factor = factor_data.get_factor(report_factor_name)

print(f"生成 '{report_factor_name}' 因子分析报告...")

# 创建报告生成器
report_gen = ReportGenerator(
    factor_name=report_factor_name,
    factor_data=report_factor,
    forward_returns=returns_series,
)

# 生成文本摘要报告
report_text = report_gen.generate_summary_report(
    n_groups=5, method="quantile", freq="D", ic_method="pearson"
)

print("\n因子分析报告摘要:")
print("=" * 80)
print(report_text)

# 生成图表
print("\n生成分析图表...")

# 因子分布图
fig1 = report_gen.generate_factor_distribution_plot()
plt.suptitle(f"{report_factor_name} 因子分布分析", fontsize=14, fontweight="bold")
plt.show()

# IC分析图
fig2 = report_gen.generate_ic_analysis_plot(freq="D", method="pearson")
plt.suptitle(f"{report_factor_name} IC分析", fontsize=14, fontweight="bold")
plt.show()

# 分组表现图
fig3 = report_gen.generate_group_performance_plot(n_groups=5, method="quantile")
plt.suptitle(f"{report_factor_name} 分组表现分析", fontsize=14, fontweight="bold")
plt.show()

# 换手率分析图
fig4 = report_gen.generate_turnover_analysis_plot(n_groups=5, method="quantile")
plt.suptitle(f"{report_factor_name} 换手率分析", fontsize=14, fontweight="bold")
plt.show()

# 保存报告
output_dir = f"./factor_analysis_report_{report_factor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"\n保存完整报告到: {output_dir}")
report_gen.save_report(
    output_dir=output_dir, n_groups=5, method="quantile", freq="D", ic_method="pearson"
)

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)
print(f"分析报告已保存到: {output_dir}")
print(f"包含:")
print(f"  - 因子分布图: factor_distribution.png")
print(f"  - IC分析图: ic_analysis.png")
print(f"  - 分组表现图: group_performance.png")
print(f"  - 换手率分析图: turnover_analysis.png")
print(f"  - 文本报告: factor_report.txt")
print(f"  - 原始数据: factor_data.csv, forward_returns.csv")
