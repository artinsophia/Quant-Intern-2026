#!/usr/bin/env python3
"""
测试市场数据分析工具
"""

import sys
import os
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))


def test_basic_functionality():
    """测试基本功能"""
    print("测试市场数据分析工具...")
    print("=" * 60)

    try:
        from daily_analysis import DailyMarketAnalyzer

        # 1. 测试分析器初始化
        print("1. 测试分析器初始化...")
        analyzer = DailyMarketAnalyzer()
        print("   ✓ 分析器初始化成功")

        # 2. 测试单日分析（使用已知存在的文件）
        print("\n2. 测试单日分析...")

        # 查找一个存在的回测文件
        data_dir = Path("/home/jovyan/work/backtest_result")
        pkl_files = list(data_dir.glob("*.pkl"))

        if pkl_files:
            # 从文件名解析标的和日期
            sample_file = pkl_files[0].name
            parts = sample_file.split("_")

            if len(parts) >= 3:
                instrument_id = parts[0]
                trade_ymd = parts[1]
                strategy_name = "_".join(parts[2:]).replace(".pkl", "")

                print(f"   测试文件: {sample_file}")
                print(f"   标的: {instrument_id}, 日期: {trade_ymd}")

                stats = analyzer.analyze_single_day(instrument_id, trade_ymd)

                if stats:
                    print(f"   ✓ 单日分析成功")
                    print(f"     数据点: {stats.get('data_points', 'N/A')}")
                    print(f"     交易时段: {stats.get('time_period', 'N/A')}")

                    if "price_change_pct" in stats:
                        print(f"     涨跌幅: {stats['price_change_pct']:.2f}%")

                    if "final_profit" in stats:
                        print(f"     策略盈利: {stats['final_profit']:.2f}")
                else:
                    print("   ✗ 单日分析失败（可能是数据格式问题）")
            else:
                print("   ⚠ 文件名格式不符合预期")
        else:
            print("   ⚠ 未找到回测结果文件")

        # 3. 测试多日分析
        print("\n3. 测试多日分析...")

        # 尝试分析最近几天的数据
        test_instruments = ["518880", "511520", "511090"]
        found_data = False

        for instrument in test_instruments:
            print(f"   尝试分析 {instrument}...")
            df_stats = analyzer.analyze_multiple_days(
                instrument, "20260301", "20260305"
            )

            if not df_stats.empty:
                print(f"   ✓ {instrument} 分析成功，找到 {len(df_stats)} 天数据")
                found_data = True

                # 测试报告生成
                report = analyzer.generate_summary_report(df_stats)
                print(f"     报告生成成功: {len(report.split('\\n'))} 行")
                break
            else:
                print(f"   ⚠ {instrument} 无数据")

        if not found_data:
            print("   ⚠ 所有测试标的均无数据")

        # 4. 测试命令行工具导入
        print("\n4. 测试命令行工具...")
        try:
            import cli_tool

            print("   ✓ 命令行工具导入成功")
        except ImportError as e:
            print(f"   ✗ 命令行工具导入失败: {e}")

        # 5. 测试可视化工具导入
        print("\n5. 测试可视化工具...")
        try:
            import advanced_visualization

            print("   ✓ 高级可视化工具导入成功")
        except ImportError as e:
            print(f"   ✗ 高级可视化工具导入失败: {e}")

        print("\n" + "=" * 60)
        print("测试完成！")

        # 提供使用建议
        print("\n使用建议:")
        print("1. 确保回测结果文件存在于 /home/jovyan/work/backtest_result/")
        print("2. 文件命名格式应为: {标的}_{日期}_{策略}.pkl")
        print("3. 支持的标的: 511520, 511090, 518880")
        print("4. 运行示例: python daily_analysis.py")
        print("5. 查看完整示例: jupyter notebook market_analysis_demo.ipynb")

        return True

    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_dependencies():
    """检查依赖"""
    print("检查依赖...")

    dependencies = ["pandas", "numpy", "matplotlib"]
    missing = []

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep}")
            missing.append(dep)

    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False

    print("所有依赖已安装")
    return True


def main():
    """主函数"""
    print("市场数据分析工具测试")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        print("\n请先安装缺失的依赖")
        return

    print()

    # 运行功能测试
    test_basic_functionality()


if __name__ == "__main__":
    main()
