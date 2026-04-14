#!/usr/bin/env python3
"""
快速演示因子测试框架的基本功能
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加路径
sys.path.append("/home/jovyan/work/tactics_demo/factor_testing")

print("=" * 80)
print("快速演示因子测试框架")
print("=" * 80)

try:
    # 导入框架
    from factor_testing import (
        FactorData,
        FactorPreprocessor,
        FactorMetrics,
        GroupTester,
        ReportGenerator,
    )

    print("✓ 因子测试框架导入成功")

    # 创建示例数据
    print("\n1. 创建示例数据...")

    # 创建日期和标的
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT"]

    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # 创建因子数据
    np.random.seed(42)
    factor_df = pd.DataFrame(
        {
            "momentum": np.random.randn(len(index)) * 0.1 + 0.05,
            "value": np.random.randn(len(index)) * 0.15 + 0.02,
            "quality": np.random.randn(len(index)) * 0.08 + 0.03,
        },
        index=index,
    )

    # 创建未来收益（与momentum因子正相关）
    forward_returns = factor_df["momentum"] * 0.3 + np.random.randn(len(index)) * 0.02

    print(f"因子数据形状: {factor_df.shape}")
    print(f"因子列: {list(factor_df.columns)}")
    print(f"收益数据形状: {forward_returns.shape}")

    # 创建FactorData对象
    print("\n2. 创建FactorData对象...")
    factor_data = FactorData(factor_df)
    print(f"FactorData: {factor_data}")

    # 因子预处理
    print("\n3. 因子预处理演示...")
    momentum_factor = factor_data.get_factor("momentum")

    # 去极值
    winsorized = FactorPreprocessor.winsorize(
        momentum_factor, method="quantile", limits=0.05
    )
    print(
        f"原始momentum范围: [{momentum_factor.min():.4f}, {momentum_factor.max():.4f}]"
    )
    print(f"去极值后范围: [{winsorized.min():.4f}, {winsorized.max():.4f}]")

    # 标准化
    standardized = FactorPreprocessor.standardize(winsorized, method="zscore")
    print(f"标准化后均值: {standardized.mean():.6f}, 标准差: {standardized.std():.6f}")

    # IC计算
    print("\n4. IC计算演示...")
    from factor_testing.metrics import ICCalculator

    ic_calculator = ICCalculator(momentum_factor, forward_returns)
    ic_pearson = ic_calculator.calculate_ic(method="pearson")
    ic_spearman = ic_calculator.calculate_ic(method="spearman")

    print(f"Momentum因子 Pearson IC: {ic_pearson:.4f}")
    print(f"Momentum因子 Rank IC: {ic_spearman:.4f}")

    # 综合指标计算
    print("\n5. 综合指标计算演示...")
    metrics_calculator = FactorMetrics(momentum_factor, forward_returns)
    all_metrics = metrics_calculator.calculate_all_metrics(n_groups=5, freq="D")

    print("关键指标:")
    print(f"  IC: {all_metrics.get('ic', np.nan):.4f}")
    print(f"  IR: {all_metrics.get('ir', np.nan):.4f}")
    print(f"  多空组合夏普: {all_metrics.get('long_short_sharpe', np.nan):.4f}")

    # 分组测试
    print("\n6. 分组测试演示...")
    group_tester = GroupTester(momentum_factor, forward_returns)
    test_results = group_tester.run_comprehensive_test(n_groups=5)

    if "long_short" in test_results:
        ls = test_results["long_short"]
        print("多空组合表现:")
        print(f"  平均收益: {ls.get('mean_return', 0) * 100:.2f}%")
        print(f"  夏普比率: {ls.get('sharpe_ratio', 0):.4f}")

    # 批量计算多个因子
    print("\n7. 批量计算多个因子...")
    batch_metrics = FactorMetrics.batch_calculate_metrics(
        factor_df, forward_returns, n_groups=5, freq="D"
    )

    print("因子比较:")
    print(batch_metrics[["ic", "ir", "rank_ic", "long_short_sharpe"]].round(4))

    # 报告生成
    print("\n8. 报告生成演示...")
    report_gen = ReportGenerator(
        factor_name="momentum",
        factor_data=momentum_factor,
        forward_returns=forward_returns,
    )

    # 生成文本报告
    report_text = report_gen.generate_summary_report(
        n_groups=5, method="quantile", freq="D"
    )
    print("报告摘要（前10行）:")
    print("\n".join(report_text.split("\n")[:10]))

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    print("\n框架功能验证:")
    print("  ✓ FactorData - 因子数据管理")
    print("  ✓ FactorPreprocessor - 因子预处理")
    print("  ✓ ICCalculator - IC计算")
    print("  ✓ FactorMetrics - 综合指标计算")
    print("  ✓ GroupTester - 分组测试")
    print("  ✓ ReportGenerator - 报告生成")
    print("  ✓ 批量计算 - 多因子分析")

    print("\n下一步:")
    print("  1. 运行 factor_analysis_518880.py 进行实际数据分析")
    print("  2. 打开 factor_analysis_demo.ipynb 进行交互式分析")
    print("  3. 查看 example.py 了解更多使用示例")

except Exception as e:
    print(f"\n❌ 演示过程中出错: {e}")
    import traceback

    traceback.print_exc()
