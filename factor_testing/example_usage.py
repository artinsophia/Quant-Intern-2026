"""
因子测试框架使用示例
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append("/home/jovyan/work/tactics_demo/factor_testing")
sys.path.append("/home/jovyan/work/tactics_demo/delta")

from utils.data_loader import DataLoader
from metrics.ic_analysis import ICAnalyzer, calculate_factor_stats
from metrics.group_backtest import GroupBacktester
from visualization.plot_factors import FactorVisualizer


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("因子测试框架 - 基础使用示例")
    print("=" * 60)

    # 1. 初始化组件
    data_loader = DataLoader()
    ic_analyzer = ICAnalyzer(forward_periods=[1, 5, 10])
    group_tester = GroupBacktester(n_groups=5)
    visualizer = FactorVisualizer()

    # 2. 加载数据（使用模拟数据）
    print("\n1. 加载模拟数据...")
    snap_slice = data_loader.create_mock_data(n_samples=2000)
    print(f"创建了 {len(snap_slice)} 条模拟记录")

    # 3. 提取特征
    print("\n2. 提取特征...")
    features_df = data_loader.extract_features_from_snapshots(
        snap_slice, window_size=60, step_size=5
    )

    if features_df.empty:
        print("错误: 无法提取特征")
        return

    print(f"提取了 {len(features_df)} 个特征样本")
    print(f"特征列: {list(features_df.columns)}")

    # 4. 准备价格序列
    print("\n3. 准备价格序列...")
    price_series = features_df["price"] if "price" in features_df.columns else None

    if price_series is None:
        # 从原始数据创建价格序列
        price_series = pd.Series(
            [s.get("price_last", np.nan) for s in snap_slice[60::5]]
        )
        price_series.index = features_df.index[: len(price_series)]

    print(f"价格序列长度: {len(price_series)}")

    # 5. IC值分析
    print("\n4. 进行IC值分析...")

    # 选择要分析的因子列
    exclude_cols = ["timestamp", "datetime", "price", "price_last", "num_trades"]
    factor_cols = [col for col in features_df.columns if col not in exclude_cols]

    ic_results = ic_analyzer.analyze_factor_ic(
        features_df[factor_cols], price_series, factor_cols
    )

    # 计算因子统计
    factor_stats = calculate_factor_stats(ic_results)

    print("\nTop 5 因子 (按平均绝对IC值):")
    top_5 = factor_stats.nlargest(5, "mean_abs_ic")
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(
            f"{i}. {row['factor']:15s} | "
            f"IC均值: {row['mean_ic']:6.4f} | "
            f"ICIR: {row['icir']:6.4f} | "
            f"胜率: {row['win_rate']:6.2%}"
        )

    # 6. 分组回测（对最佳因子）
    print("\n5. 对最佳因子进行分组回测...")
    best_factor = top_5.iloc[0]["factor"]
    print(f"分析因子: {best_factor}")

    group_analysis = group_tester.run_complete_analysis(
        features_df[factor_cols], price_series, best_factor, forward_periods=[1, 5, 10]
    )

    # 显示分组表现
    perf_df = group_analysis["group_performance"]
    print("\n分组表现 (周期=1):")
    period1_data = perf_df[perf_df["period"] == 1]
    for _, row in period1_data.iterrows():
        print(
            f"  {row['group']:12s} | "
            f"收益率: {row['mean_return']:7.4%} | "
            f"夏普: {row['sharpe_ratio']:6.4f} | "
            f"胜率: {row['win_rate']:6.2%}"
        )

    # 7. 可视化
    print("\n6. 生成可视化图表...")

    # IC分析图表
    ic_fig = visualizer.plot_ic_analysis(ic_results, "因子IC分析示例")
    ic_fig.suptitle("因子IC分析示例", fontsize=14)
    plt.show()

    # 分组表现图表
    group_fig = visualizer.plot_group_performance(
        group_analysis["group_performance"], f"因子 {best_factor} 分组表现"
    )
    plt.show()

    # 因子相关性矩阵
    corr_fig = visualizer.plot_factor_correlation(
        features_df[factor_cols].iloc[:100],  # 只取前100个样本
        "因子相关性矩阵示例",
    )
    plt.show()

    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


def example_advanced_usage():
    """高级使用示例"""
    print("\n" + "=" * 60)
    print("因子测试框架 - 高级使用示例")
    print("=" * 60)

    # 1. 自定义IC分析
    print("\n1. 自定义IC分析...")

    # 创建自定义的IC分析器
    custom_ic_analyzer = ICAnalyzer(
        forward_periods=[1, 3, 5, 10, 20]  # 更多的时间周期
    )

    # 2. 自定义分组回测
    print("\n2. 自定义分组回测...")

    custom_group_tester = GroupBacktester(
        n_groups=10,  # 更多分组
        group_method="equal",  # 等间距分组
    )

    # 3. 计算滚动IC序列
    print("\n3. 计算滚动IC序列...")

    # 创建一些模拟数据
    dates = pd.date_range("2023-01-01", periods=500, freq="H")
    np.random.seed(42)

    # 创建模拟因子和价格
    factor_values = pd.Series(np.random.randn(500).cumsum(), index=dates)
    price_values = pd.Series(100 + np.random.randn(500).cumsum() * 0.1, index=dates)

    # 计算滚动IC
    ic_series = custom_ic_analyzer.calculate_ic_series(
        pd.DataFrame({"test_factor": factor_values}),
        price_values,
        "test_factor",
        period=1,
        rolling_window=50,
    )

    print(f"滚动IC序列长度: {len(ic_series.dropna())}")
    print(f"滚动IC均值: {ic_series.mean():.4f}")
    print(f"滚动IC标准差: {ic_series.std():.4f}")

    # 4. 计算IC衰减
    print("\n4. 计算IC衰减...")

    ic_decay = custom_ic_analyzer.calculate_ic_decay(
        pd.DataFrame({"test_factor": factor_values}),
        price_values,
        "test_factor",
        max_period=30,
    )

    print("IC衰减曲线 (前10个周期):")
    for _, row in ic_decay.head(10).iterrows():
        print(f"  周期 {row['period']:2d}: IC = {row['ic']:.4f}")

    # 5. 可视化滚动IC
    print("\n5. 可视化滚动IC序列...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 价格序列
    ax1.plot(price_values.index, price_values.values, linewidth=1)
    ax1.set_ylabel("价格")
    ax1.set_title("价格序列")
    ax1.grid(True, alpha=0.3)

    # 滚动IC序列
    ax2.plot(ic_series.index, ic_series.values, linewidth=1, color="red")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax2.set_ylabel("滚动IC值")
    ax2.set_title("滚动IC序列 (窗口=50)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("高级示例完成!")
    print("=" * 60)


def example_factor_evaluation():
    """因子评估示例"""
    print("\n" + "=" * 60)
    print("因子评估流程示例")
    print("=" * 60)

    # 模拟多个因子
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")

    # 创建模拟因子
    factors = {
        "momentum": np.random.randn(n_samples).cumsum(),  # 动量因子
        "mean_reversion": np.sin(np.linspace(0, 10, n_samples))
        + np.random.randn(n_samples) * 0.1,  # 均值回归
        "volume_ratio": np.random.exponential(1, n_samples),  # 成交量比率
        "volatility": np.abs(np.random.randn(n_samples)),  # 波动率
        "spread": np.random.uniform(0, 0.1, n_samples),  # 价差
    }

    # 创建价格序列（因子应该能预测这个）
    true_alpha = (
        0.3 * factors["momentum"]
        - 0.2 * factors["mean_reversion"]
        + 0.1 * factors["volume_ratio"]
    )
    price = 100 + true_alpha.cumsum() + np.random.randn(n_samples) * 0.5

    # 创建DataFrame
    factor_df = pd.DataFrame(factors, index=dates)
    price_series = pd.Series(price, index=dates)

    # 评估因子
    ic_analyzer = ICAnalyzer(forward_periods=[1, 5, 10])
    ic_results = ic_analyzer.analyze_factor_ic(factor_df, price_series)

    # 计算因子统计
    factor_stats = calculate_factor_stats(ic_results)

    print("\n因子评估结果:")
    print("-" * 80)
    print(f"{'因子':15s} {'IC均值':>10s} {'IC绝对值':>10s} {'ICIR':>10s} {'胜率':>10s}")
    print("-" * 80)

    for _, row in factor_stats.iterrows():
        print(
            f"{row['factor']:15s} {row['mean_ic']:10.4f} {row['mean_abs_ic']:10.4f} "
            f"{row['icir']:10.4f} {row['win_rate']:10.2%}"
        )

    print("\n分析:")
    print("1. momentum因子应该表现最好（因为它是价格的主要驱动因素）")
    print("2. mean_reversion应该有负的IC（因为它是反向因子）")
    print("3. 其他因子可能表现随机")

    # 验证我们的假设
    momentum_ic = factor_stats[factor_stats["factor"] == "momentum"]["mean_ic"].values[
        0
    ]
    mean_rev_ic = factor_stats[factor_stats["factor"] == "mean_reversion"][
        "mean_ic"
    ].values[0]

    print(f"\n验证:")
    print(f"  • momentum IC值: {momentum_ic:.4f} (应该为正)")
    print(f"  • mean_reversion IC值: {mean_rev_ic:.4f} (应该为负)")

    if momentum_ic > 0 and mean_rev_ic < 0:
        print("  ✓ 结果符合预期!")
    else:
        print("  ✗ 结果不符合预期，可能需要检查数据或模型")

    print("\n" + "=" * 60)
    print("因子评估示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    print("因子测试框架示例脚本")
    print("选择要运行的示例:")
    print("1. 基础使用示例")
    print("2. 高级使用示例")
    print("3. 因子评估示例")
    print("4. 全部运行")

    choice = input("\n请输入选择 (1-4): ").strip()

    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_advanced_usage()
    elif choice == "3":
        example_factor_evaluation()
    elif choice == "4":
        example_basic_usage()
        example_advanced_usage()
        example_factor_evaluation()
    else:
        print("无效选择")
