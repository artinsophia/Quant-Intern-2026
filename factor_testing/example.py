"""
因子测试框架使用示例
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factor_testing import (
    FactorData,
    FactorPreprocessor,
    FactorMetrics,
    GroupTester,
    ReportGenerator,
)


def create_sample_data():
    """
    创建示例数据
    """
    # 创建日期范围
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    symbols = [
        "AAPL",
        "GOOGL",
        "MSFT",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "JPM",
        "JNJ",
        "V",
    ]

    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # 创建因子数据（随机生成）
    np.random.seed(42)

    # 因子1: 动量因子（与未来收益正相关）
    momentum = np.random.randn(len(index)) * 0.1
    # 添加时间趋势
    for i, (date, symbol) in enumerate(index):
        momentum[i] += (date - dates[0]).days * 0.0001

    # 因子2: 价值因子（随机）
    value = np.random.randn(len(index)) * 0.15

    # 因子3: 质量因子（与未来收益弱相关）
    quality = np.random.randn(len(index)) * 0.08
    # 添加轻微的正相关性
    quality = quality * 0.7 + np.random.randn(len(index)) * 0.3

    # 创建未来收益（与动量因子正相关）
    forward_returns = momentum * 0.3 + np.random.randn(len(index)) * 0.02

    # 创建DataFrame
    factor_df = pd.DataFrame(
        {"momentum": momentum, "value": value, "quality": quality}, index=index
    )

    forward_returns_series = pd.Series(
        forward_returns, index=index, name="forward_return"
    )

    return factor_df, forward_returns_series


def example_basic_usage():
    """
    基本使用示例
    """
    print("=" * 80)
    print("因子测试框架 - 基本使用示例")
    print("=" * 80)

    # 1. 创建示例数据
    print("\n1. 创建示例数据...")
    factor_df, forward_returns = create_sample_data()

    print(f"因子数据形状: {factor_df.shape}")
    print(f"因子名称: {list(factor_df.columns)}")
    print(
        f"日期范围: {factor_df.index.get_level_values('date').min()} 到 {factor_df.index.get_level_values('date').max()}"
    )
    print(f"标的数量: {len(factor_df.index.get_level_values('symbol').unique())}")

    # 2. 创建FactorData对象
    print("\n2. 创建FactorData对象...")
    factor_data = FactorData(factor_df)
    print(factor_data)

    # 获取因子统计信息
    print("\n因子统计信息:")
    for factor_name in factor_data.factor_names:
        stats = factor_data.get_factor_stats(factor_name)
        print(f"\n{factor_name}:")
        print(f"  均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}")
        print(f"  偏度: {stats['skew']:.4f}, 峰度: {stats['kurtosis']:.4f}")
        print(f"  缺失率: {stats['missing_rate']:.2%}")

    # 3. 因子预处理
    print("\n3. 因子预处理示例...")
    momentum_factor = factor_data.get_factor("momentum")

    # 去极值
    winsorized = FactorPreprocessor.winsorize(
        momentum_factor, method="quantile", limits=0.05
    )
    print(f"去极值前范围: [{momentum_factor.min():.4f}, {momentum_factor.max():.4f}]")
    print(f"去极值后范围: [{winsorized.min():.4f}, {winsorized.max():.4f}]")

    # 标准化
    standardized = FactorPreprocessor.standardize(winsorized, method="zscore")
    print(f"标准化后均值: {standardized.mean():.6f}, 标准差: {standardized.std():.6f}")

    # 4. IC计算
    print("\n4. IC计算示例...")

    # 计算单个因子的IC
    from factor_testing.metrics import ICCalculator

    ic_calculator = ICCalculator(factor_data.get_factor("momentum"), forward_returns)

    # 计算Pearson IC
    ic_pearson = ic_calculator.calculate_ic(method="pearson")
    print(f"Momentum因子Pearson IC: {ic_pearson:.4f}")

    # 计算Rank IC
    ic_spearman = ic_calculator.calculate_ic(method="spearman")
    print(f"Momentum因子Rank IC: {ic_spearman:.4f}")

    # 计算IC时间序列
    ic_series = ic_calculator.calculate_ic_series(freq="M", method="pearson")
    print(f"\nIC时间序列（月频）:")
    print(f"  均值: {ic_series.mean():.4f}, 标准差: {ic_series.std():.4f}")
    print(
        f"  IR: {ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan:.4f}"
    )

    # 5. 综合指标计算
    print("\n5. 综合指标计算...")

    metrics_calculator = FactorMetrics(
        factor_data.get_factor("momentum"), forward_returns
    )

    all_metrics = metrics_calculator.calculate_all_metrics(
        n_groups=5, freq="M", method="pearson"
    )

    print("关键指标:")
    print(f"  IC: {all_metrics.get('ic', np.nan):.4f}")
    print(f"  IR: {all_metrics.get('ir', np.nan):.4f}")
    print(f"  Rank IC: {all_metrics.get('rank_ic', np.nan):.4f}")
    print(f"  多空组合夏普: {all_metrics.get('long_short_sharpe', np.nan):.4f}")

    # 6. 分组测试
    print("\n6. 分组测试...")

    group_tester = GroupTester(factor_data.get_factor("momentum"), forward_returns)

    group_results = group_tester.run_comprehensive_test(n_groups=5, method="quantile")

    if "long_short" in group_results:
        ls = group_results["long_short"]
        print("多空组合表现:")
        print(f"  平均收益: {ls.get('mean_return', np.nan) * 100:.2f}%")
        print(f"  夏普比率: {ls.get('sharpe_ratio', np.nan):.3f}")
        print(f"  胜率: {ls.get('win_rate', np.nan) * 100:.1f}%")

    # 7. 批量计算多个因子
    print("\n7. 批量计算多个因子...")

    batch_metrics = FactorMetrics.batch_calculate_metrics(
        factor_df, forward_returns, n_groups=5, freq="M"
    )

    print("\n因子比较:")
    print(batch_metrics[["ic", "ir", "rank_ic", "long_short_sharpe"]].round(4))

    return factor_df, forward_returns


def example_advanced_usage(factor_df, forward_returns):
    """
    高级使用示例
    """
    print("\n" + "=" * 80)
    print("因子测试框架 - 高级使用示例")
    print("=" * 80)

    # 1. 使用预处理流水线
    print("\n1. 预处理流水线示例...")

    momentum_factor = factor_df["momentum"]

    # 定义预处理步骤
    preprocessing_steps = [
        {"name": "winsorize", "params": {"method": "quantile", "limits": 0.05}},
        {"name": "fill_missing", "params": {"method": "mean"}},
        {"name": "standardize", "params": {"method": "zscore"}},
    ]

    processed_factor = FactorPreprocessor.pipeline(momentum_factor, preprocessing_steps)

    print(
        f"原始因子均值: {momentum_factor.mean():.6f}, 标准差: {momentum_factor.std():.6f}"
    )
    print(
        f"处理后因子均值: {processed_factor.mean():.6f}, 标准差: {processed_factor.std():.6f}"
    )

    # 2. 分组换手率分析
    print("\n2. 分组换手率分析...")

    group_tester = GroupTester(factor_df["momentum"], forward_returns)

    turnover_results = group_tester.calculate_group_turnover(
        n_groups=5, method="quantile"
    )

    if "long_short" in turnover_results:
        ls_turnover = turnover_results["long_short"]
        print("多空组合换手率:")
        print(f"  平均换手率: {ls_turnover.get('mean_turnover', np.nan) * 100:.2f}%")
        print(f"  换手率标准差: {ls_turnover.get('std_turnover', np.nan) * 100:.2f}%")
        print(f"  最大换手率: {ls_turnover.get('max_turnover', np.nan) * 100:.2f}%")

    # 3. 因子比较
    print("\n3. 因子比较分析...")

    # 创建因子数据字典
    factor_dict = {
        "momentum": factor_df["momentum"],
        "value": factor_df["value"],
        "quality": factor_df["quality"],
    }

    comparison_results = GroupTester.compare_factors(
        factor_dict, forward_returns, n_groups=5, method="quantile"
    )

    print("\n因子比较结果:")
    print(comparison_results.round(4))

    # 4. 生成完整报告
    print("\n4. 生成完整报告...")

    report_generator = ReportGenerator(
        factor_name="momentum",
        factor_data=factor_df["momentum"],
        forward_returns=forward_returns,
    )

    # 生成文本报告
    report_text = report_generator.generate_summary_report(
        n_groups=5, method="quantile", freq="M", ic_method="pearson"
    )

    print("报告摘要（前20行）:")
    print("\n".join(report_text.split("\n")[:20]))

    # 保存报告到文件（注释掉，实际使用时取消注释）
    # output_dir = './factor_test_report'
    # report_generator.save_report(
    #     output_dir=output_dir,
    #     n_groups=5,
    #     method='quantile',
    #     freq='M',
    #     ic_method='pearson'
    # )

    print("\n完整报告包含:")
    print("  - 因子分布图")
    print("  - IC分析图")
    print("  - 分组表现图")
    print("  - 换手率分析图")
    print("  - 文本摘要报告")
    print("  - 原始数据文件")


def example_integration_with_existing_project():
    """
    与现有项目集成示例
    """
    print("\n" + "=" * 80)
    print("与现有项目集成示例")
    print("=" * 80)

    print("\n假设您已有因子数据和收益数据，可以这样集成:")

    integration_code = """
# 1. 导入模块
import pandas as pd
import numpy as np
from factor_testing import FactorData, FactorMetrics, GroupTester, ReportGenerator

# 2. 准备数据（假设已有）
# factor_values: 因子值，DataFrame格式，索引为(date, symbol)，每列为一个因子
# returns: 未来收益，Series格式，索引与factor_values一致

# 3. 创建因子数据对象
factor_data = FactorData(factor_values)

# 4. 选择要测试的因子
test_factor = factor_data.get_factor('your_factor_name')

# 5. 计算因子指标
metrics_calculator = FactorMetrics(test_factor, returns)
metrics = metrics_calculator.calculate_all_metrics(n_groups=5, freq='D')

print(f"IC: {metrics['ic']:.4f}")
print(f"IR: {metrics['ir']:.4f}")
print(f"多空组合夏普: {metrics['long_short_sharpe']:.4f}")

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
"""

    print(integration_code)


if __name__ == "__main__":
    print("因子测试框架示例程序")
    print("=" * 80)

    try:
        # 运行基本示例
        factor_df, forward_returns = example_basic_usage()

        # 运行高级示例
        example_advanced_usage(factor_df, forward_returns)

        # 运行集成示例
        example_integration_with_existing_project()

        print("\n" + "=" * 80)
        print("示例程序运行完成！")
        print("=" * 80)
        print("\n主要功能:")
        print("1. FactorData: 因子数据加载和管理")
        print("2. FactorPreprocessor: 因子预处理（去极值、标准化、中性化等）")
        print("3. ICCalculator: IC计算（Pearson、Spearman、Kendall）")
        print("4. FactorMetrics: 综合指标计算（IR、换手率、衰减率等）")
        print("5. GroupTester: 分组测试和分层回测")
        print("6. ReportGenerator: 可视化图表和文本报告生成")

    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback

        traceback.print_exc()
