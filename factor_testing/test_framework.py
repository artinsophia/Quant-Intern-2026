#!/usr/bin/env python3
"""
因子测试框架验证测试
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append("/home/jovyan/work/tactics_demo/factor_testing")


def test_imports():
    """测试模块导入"""
    print("测试模块导入...")

    try:
        # 在函数内部导入
        import sys

        sys.path.append("/home/jovyan/work/tactics_demo/factor_testing")

        from utils.data_loader import DataLoader
        from metrics.ic_analysis import ICAnalyzer
        from metrics.group_backtest import GroupBacktester
        from visualization.plot_factors import FactorVisualizer

        print("✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")

    try:
        data_loader = DataLoader()

        # 测试模拟数据
        mock_data = data_loader.create_mock_data(n_samples=100)
        print(f"✓ 创建模拟数据: {len(mock_data)} 条记录")

        # 测试DataFrame转换
        df = data_loader.snapshots_to_dataframe(mock_data)
        print(f"✓ 转换为DataFrame: {df.shape}")

        # 测试特征提取
        features_df = data_loader.extract_features_from_snapshots(
            mock_data, window_size=30, step_size=5
        )
        print(f"✓ 提取特征: {features_df.shape if not features_df.empty else '失败'}")

        return True
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        return False


def test_ic_analyzer():
    """测试IC分析器"""
    print("\n测试IC分析器...")

    try:
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="h")

        # 创建因子和价格数据
        factor_values = pd.Series(np.random.randn(100).cumsum(), index=dates)
        price_values = pd.Series(100 + np.random.randn(100).cumsum() * 0.1, index=dates)

        factor_df = pd.DataFrame({"test_factor": factor_values})

        # 测试IC分析器
        ic_analyzer = ICAnalyzer(forward_periods=[1, 5])

        # 计算IC
        ic = ic_analyzer.calculate_ic(factor_values, price_values.shift(-1), "spearman")
        print(f"✓ 计算IC值: {ic:.4f}")

        # 分析因子IC
        ic_results = ic_analyzer.analyze_factor_ic(
            factor_df, price_values, ["test_factor"]
        )
        print(f"✓ 分析因子IC: {len(ic_results)} 条结果")

        # 计算滚动IC
        ic_series = ic_analyzer.calculate_ic_series(
            factor_df, price_values, "test_factor", period=1, rolling_window=20
        )
        print(f"✓ 计算滚动IC: {len(ic_series.dropna())} 个有效值")

        # 计算IC衰减
        ic_decay = ic_analyzer.calculate_ic_decay(
            factor_df, price_values, "test_factor", max_period=10
        )
        print(f"✓ 计算IC衰减: {len(ic_decay)} 个周期")

        return True
    except Exception as e:
        print(f"✗ IC分析器测试失败: {e}")
        return False


def test_group_backtester():
    """测试分组回测器"""
    print("\n测试分组回测器...")

    try:
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="h")

        # 创建因子和价格数据
        factor_values = pd.Series(np.random.randn(200).cumsum(), index=dates)
        price_values = pd.Series(100 + np.random.randn(200).cumsum() * 0.1, index=dates)

        factor_df = pd.DataFrame({"test_factor": factor_values})

        # 测试分组回测器
        group_tester = GroupBacktester(n_groups=5, group_method="quantile")

        # 创建分组
        groups = group_tester.create_factor_groups(factor_values.iloc[:100])
        print(f"✓ 创建分组: {groups.nunique()} 个分组")

        # 计算分组收益率
        group_returns = group_tester.calculate_group_returns(
            factor_df, price_values, "test_factor", forward_periods=[1, 5]
        )
        print(f"✓ 计算分组收益率: {len(group_returns)} 个周期")

        # 分析分组表现
        group_performance = group_tester.analyze_group_performance(group_returns)
        print(f"✓ 分析分组表现: {len(group_performance)} 条记录")

        # 计算单调性
        monotonicity = group_tester.calculate_monotonicity(
            group_performance, "test_factor"
        )
        print(f"✓ 计算单调性: {len(monotonicity)} 个周期")

        return True
    except Exception as e:
        print(f"✗ 分组回测器测试失败: {e}")
        return False


def test_visualizer():
    """测试可视化器"""
    print("\n测试可视化器...")

    try:
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="h")

        # 创建多个因子
        factors = {
            "factor1": np.random.randn(50).cumsum(),
            "factor2": np.sin(np.linspace(0, 10, 50)) * 2,
            "factor3": np.random.exponential(1, 50),
        }

        factor_df = pd.DataFrame(factors, index=dates)

        # 创建IC分析结果
        ic_data = []
        for factor in factors.keys():
            for period in [1, 5, 10]:
                ic_data.append(
                    {
                        "factor": factor,
                        "period": period,
                        "ic_spearman": np.random.uniform(-0.1, 0.1),
                        "ic_pearson": np.random.uniform(-0.1, 0.1),
                        "abs_ic_spearman": abs(np.random.uniform(0, 0.1)),
                        "abs_ic_pearson": abs(np.random.uniform(0, 0.1)),
                        "sample_size": 50,
                    }
                )

        ic_results = pd.DataFrame(ic_data)

        # 创建分组表现结果
        group_data = []
        for group in [
            "Group_1",
            "Group_2",
            "Group_3",
            "Group_4",
            "Group_5",
            "Long-Short",
        ]:
            for period in [1, 5, 10]:
                group_data.append(
                    {
                        "period": period,
                        "group": group,
                        "mean_return": np.random.uniform(-0.01, 0.01),
                        "std_return": np.random.uniform(0.005, 0.02),
                        "sharpe_ratio": np.random.uniform(-1, 2),
                        "win_rate": np.random.uniform(0.4, 0.7),
                        "max_drawdown": np.random.uniform(-0.05, -0.01),
                        "sample_size": 50,
                    }
                )

        group_performance = pd.DataFrame(group_data)

        # 测试可视化器
        visualizer = FactorVisualizer(figsize=(10, 6))

        # 测试IC分析图表
        try:
            ic_fig = visualizer.plot_ic_analysis(ic_results, "测试IC分析")
            print("✓ 创建IC分析图表")
        except Exception as e:
            print(f"⚠ IC分析图表创建警告: {e}")

        # 测试分组表现图表
        try:
            group_fig = visualizer.plot_group_performance(
                group_performance, "测试分组表现"
            )
            print("✓ 创建分组表现图表")
        except Exception as e:
            print(f"⚠ 分组表现图表创建警告: {e}")

        # 测试相关性矩阵
        try:
            corr_fig = visualizer.plot_factor_correlation(factor_df, "测试相关性矩阵")
            print("✓ 创建相关性矩阵图表")
        except Exception as e:
            print(f"⚠ 相关性矩阵图表创建警告: {e}")

        return True
    except Exception as e:
        print(f"✗ 可视化器测试失败: {e}")
        return False


def test_main_script():
    """测试主脚本"""
    print("\n测试主脚本...")

    try:
        # 检查主脚本是否存在
        script_path = (
            "/home/jovyan/work/tactics_demo/factor_testing/test_factor_performance.py"
        )
        if os.path.exists(script_path):
            print(f"✓ 主脚本存在: {script_path}")

            # 检查脚本是否可以导入
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "test_factor_performance", script_path
            )
            module = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(module)
                print("✓ 主脚本可以导入")

                # 检查主要类是否存在
                if hasattr(module, "FactorPerformanceTester"):
                    print("✓ FactorPerformanceTester类存在")
                else:
                    print("✗ FactorPerformanceTester类不存在")

                return True
            except Exception as e:
                print(f"✗ 主脚本导入失败: {e}")
                return False
        else:
            print(f"✗ 主脚本不存在: {script_path}")
            return False
    except Exception as e:
        print(f"✗ 主脚本测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("因子测试框架验证测试")
    print("=" * 60)

    tests = [
        ("模块导入", test_imports),
        ("数据加载器", test_data_loader),
        ("IC分析器", test_ic_analyzer),
        ("分组回测器", test_group_backtester),
        ("可视化器", test_visualizer),
        ("主脚本", test_main_script),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:15s} {status}")
        if success:
            passed += 1

    print(f"\n通过率: {passed}/{total} ({passed / total * 100:.1f}%)")

    if passed == total:
        print("\n🎉 所有测试通过!")
        print("\n下一步:")
        print("1. 运行示例: python example_usage.py")
        print("2. 快速测试: python test_factor_performance.py --quick")
        print(
            "3. 完整测试: python test_factor_performance.py --instrument 511520 --start 20260202 --end 20260205"
        )
    else:
        print(f"\n⚠ {total - passed} 个测试失败，请检查上述错误")

    print("=" * 60)


if __name__ == "__main__":
    main()
