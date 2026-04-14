#!/usr/bin/env python3
"""
因子性能测试主脚本
评估交易因子的预测能力和有效性
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append("/home/jovyan/work/tactics_demo/factor_testing")
sys.path.append("/home/jovyan/work/tactics_demo/delta")

from utils.data_loader import DataLoader
from metrics.ic_analysis import ICAnalyzer, calculate_factor_stats
from metrics.group_backtest import GroupBacktester
from visualization.plot_factors import FactorVisualizer


class FactorPerformanceTester:
    """因子性能测试器"""

    def __init__(self, output_dir: str = "./factor_test_results"):
        """
        初始化测试器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.data_loader = DataLoader()
        self.ic_analyzer = ICAnalyzer(forward_periods=[1, 5, 10, 20])
        self.group_backtester = GroupBacktester(n_groups=5, group_method="quantile")
        self.visualizer = FactorVisualizer(figsize=(12, 8))

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def run_test(
        self,
        instrument_id: str = "511520",
        start_ymd: str = "20260202",
        end_ymd: str = "20260210",
        use_mock_data: bool = False,
        window_size: int = 60,
        step_size: int = 10,
    ):
        """
        运行因子性能测试

        Args:
            instrument_id: 标的ID
            start_ymd: 开始日期
            end_ymd: 结束日期
            use_mock_data: 是否使用模拟数据
            window_size: 特征窗口大小
            step_size: 滑动步长
        """
        print("=" * 80)
        print("因子性能测试开始")
        print(f"标的: {instrument_id}")
        print(f"日期范围: {start_ymd} 到 {end_ymd}")
        print(f"窗口大小: {window_size}, 步长: {step_size}")
        print("=" * 80)

        # 1. 加载数据
        print("\n1. 加载数据...")
        if use_mock_data:
            print("使用模拟数据")
            snap_slice = self.data_loader.create_mock_data(n_samples=5000)
        else:
            print(f"加载真实数据: {instrument_id}")
            snap_slice = self.data_loader.load_multiple_days_data(
                instrument_id, start_ymd, end_ymd
            )

        if not snap_slice:
            print("错误: 无法加载数据")
            return

        print(f"共加载 {len(snap_slice)} 条快照记录")

        # 2. 提取特征
        print("\n2. 提取特征...")
        features_df = self.data_loader.extract_features_from_snapshots(
            snap_slice, window_size=window_size, step_size=step_size
        )

        if features_df.empty:
            print("错误: 无法提取特征")
            return

        print(f"提取了 {len(features_df)} 个特征样本")
        print(f"特征列: {list(features_df.columns)}")

        # 保存特征数据
        features_file = os.path.join(self.output_dir, "features.csv")
        features_df.to_csv(features_file)
        print(f"特征数据已保存到: {features_file}")

        # 3. 准备价格序列
        print("\n3. 准备价格序列...")
        price_df = self.data_loader.snapshots_to_dataframe(snap_slice)

        if "price_last" in features_df.columns:
            price_series = features_df["price_last"]
        elif "price" in features_df.columns:
            price_series = features_df["price"]
        else:
            # 从原始快照中提取价格
            price_series = pd.Series(
                [
                    s.get("price_last", np.nan)
                    for s in snap_slice[window_size::step_size]
                ],
                index=features_df.index[: len(snap_slice[window_size::step_size])],
            )

        print(f"价格序列长度: {len(price_series)}")

        # 4. IC值分析
        print("\n4. 进行IC值分析...")

        # 选择要分析的因子列（排除价格和时间列）
        exclude_cols = ["timestamp", "datetime", "price", "price_last", "num_trades"]
        factor_cols = [col for col in features_df.columns if col not in exclude_cols]

        print(f"分析 {len(factor_cols)} 个因子: {factor_cols}")

        ic_results = self.ic_analyzer.analyze_factor_ic(
            features_df[factor_cols], price_series, factor_cols
        )

        # 计算因子统计
        factor_stats = calculate_factor_stats(ic_results)

        # 保存IC分析结果
        ic_file = os.path.join(self.output_dir, "ic_analysis.csv")
        ic_results.to_csv(ic_file, index=False)
        stats_file = os.path.join(self.output_dir, "factor_stats.csv")
        factor_stats.to_csv(stats_file, index=False)

        print(f"IC分析结果已保存到: {ic_file}")
        print(f"因子统计已保存到: {stats_file}")

        # 5. 分组回测分析
        print("\n5. 进行分组回测分析...")

        # 选择IC值最高的几个因子进行详细分析
        top_factors = factor_stats.nlargest(5, "mean_abs_ic")["factor"].tolist()
        print(f"选择Top {len(top_factors)} 因子进行分组回测: {top_factors}")

        group_results = {}

        for factor_name in top_factors:
            print(f"  分析因子: {factor_name}")

            group_analysis = self.group_backtester.run_complete_analysis(
                features_df[factor_cols], price_series, factor_name
            )
            group_results[factor_name] = group_analysis

            # 保存分组表现结果
            perf_file = os.path.join(
                self.output_dir, f"group_performance_{factor_name}.csv"
            )
            group_analysis["group_performance"].to_csv(perf_file, index=False)

        # 6. 可视化
        print("\n6. 生成可视化图表...")

        # 6.1 IC分析图表
        ic_fig = self.visualizer.plot_ic_analysis(ic_results, "因子IC分析")
        ic_fig.savefig(
            os.path.join(self.output_dir, "ic_analysis.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(ic_fig)

        # 6.2 因子相关性矩阵
        corr_fig = self.visualizer.plot_factor_correlation(
            features_df[factor_cols], "因子相关性矩阵"
        )
        corr_fig.savefig(
            os.path.join(self.output_dir, "factor_correlation.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(corr_fig)

        # 6.3 分组表现图表（为每个Top因子）
        for factor_name in top_factors:
            if factor_name in group_results:
                group_fig = self.visualizer.plot_group_performance(
                    group_results[factor_name]["group_performance"],
                    f"因子 {factor_name} 分组表现",
                )
                group_fig.savefig(
                    os.path.join(
                        self.output_dir, f"group_performance_{factor_name}.png"
                    ),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(group_fig)

        # 6.4 综合报告
        if top_factors and top_factors[0] in group_results:
            comprehensive_fig = self.visualizer.plot_comprehensive_report(
                ic_results,
                group_results[top_factors[0]]["group_performance"],
                features_df[factor_cols],
                "因子综合评估报告",
            )
            comprehensive_fig.savefig(
                os.path.join(self.output_dir, "comprehensive_report.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(comprehensive_fig)

        # 7. 生成报告
        print("\n7. 生成测试报告...")
        self.generate_report(ic_results, factor_stats, group_results, top_factors)

        print("\n" + "=" * 80)
        print("因子性能测试完成!")
        print(f"所有结果已保存到: {self.output_dir}")
        print("=" * 80)

    def generate_report(
        self,
        ic_results: pd.DataFrame,
        factor_stats: pd.DataFrame,
        group_results: Dict,
        top_factors: List[str],
    ):
        """生成测试报告"""
        report_file = os.path.join(self.output_dir, "test_report.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("因子性能测试报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # 1. 总体统计
            f.write("1. 总体统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"分析因子数量: {len(factor_stats)}\n")
            f.write(f"平均IC值: {factor_stats['mean_ic'].mean():.4f}\n")
            f.write(f"平均绝对IC值: {factor_stats['mean_abs_ic'].mean():.4f}\n")
            f.write(f"平均ICIR: {factor_stats['icir'].mean():.4f}\n")
            f.write(f"平均胜率: {factor_stats['win_rate'].mean():.2%}\n\n")

            # 2. Top因子排名
            f.write("2. Top 10 因子排名 (按平均绝对IC值)\n")
            f.write("-" * 40 + "\n")
            top_10 = factor_stats.nlargest(10, "mean_abs_ic")
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(
                    f"{i:2d}. {row['factor']:15s} | "
                    f"IC均值: {row['mean_ic']:6.4f} | "
                    f"IC绝对值: {row['mean_abs_ic']:6.4f} | "
                    f"ICIR: {row['icir']:6.4f} | "
                    f"胜率: {row['win_rate']:6.2%}\n"
                )
            f.write("\n")

            # 3. 最差因子
            f.write("3. 最差 5 个因子 (按平均绝对IC值)\n")
            f.write("-" * 40 + "\n")
            bottom_5 = factor_stats.nsmallest(5, "mean_abs_ic")
            for i, (_, row) in enumerate(bottom_5.iterrows(), 1):
                f.write(
                    f"{i:2d}. {row['factor']:15s} | "
                    f"IC均值: {row['mean_ic']:6.4f} | "
                    f"IC绝对值: {row['mean_abs_ic']:6.4f} | "
                    f"ICIR: {row['icir']:6.4f}\n"
                )
            f.write("\n")

            # 4. Top因子详细分析
            f.write("4. Top因子详细分析\n")
            f.write("-" * 40 + "\n")

            for factor_name in top_factors:
                if factor_name in group_results:
                    analysis = group_results[factor_name]
                    f.write(f"\n因子: {factor_name}\n")

                    # IC衰减
                    if "monotonicity" in analysis:
                        f.write("  IC衰减分析:\n")
                        for period, stats in analysis["monotonicity"].items():
                            f.write(
                                f"    {period}: Spearman相关={stats['spearman_corr']:.4f}, "
                                f"单调性得分={stats['monotonicity_score']:.4f}\n"
                            )

                    # 分组表现
                    perf = analysis["group_performance"]
                    ls_perf = perf[perf["group"] == "Long-Short"]

                    if not ls_perf.empty:
                        f.write("  多空组合表现:\n")
                        for _, row in ls_perf.iterrows():
                            f.write(
                                f"    周期{row['period']}: "
                                f"收益率={row['mean_return']:.4%}, "
                                f"夏普={row['sharpe_ratio']:.4f}, "
                                f"胜率={row['win_rate']:.2%}\n"
                            )

            # 5. 建议
            f.write("\n5. 建议\n")
            f.write("-" * 40 + "\n")

            # 找出有效因子
            effective_factors = factor_stats[
                (factor_stats["mean_abs_ic"] > 0.02)
                & (factor_stats["icir"] > 0.5)
                & (factor_stats["win_rate"] > 0.55)
            ]

            if len(effective_factors) > 0:
                f.write("建议关注的因子:\n")
                for _, row in effective_factors.iterrows():
                    f.write(
                        f"  • {row['factor']}: IC={row['mean_ic']:.4f}, "
                        f"ICIR={row['icir']:.4f}, 胜率={row['win_rate']:.2%}\n"
                    )
            else:
                f.write("未找到显著有效的因子，建议:\n")
                f.write("  • 调整因子参数\n")
                f.write("  • 尝试不同的时间周期\n")
                f.write("  • 考虑市场环境因素\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"测试报告已生成: {report_file}")

    def quick_test(self):
        """快速测试（使用模拟数据）"""
        print("运行快速测试（使用模拟数据）...")
        self.run_test(
            instrument_id="511520",
            start_ymd="20260202",
            end_ymd="20260205",
            use_mock_data=True,
            window_size=60,
            step_size=5,
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="因子性能测试工具")
    parser.add_argument(
        "--instrument", type=str, default="511520", help="标的ID (默认: 511520)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="20260202",
        help="开始日期 YYYYMMDD (默认: 20260202)",
    )
    parser.add_argument(
        "--end", type=str, default="20260210", help="结束日期 YYYYMMDD (默认: 20260210)"
    )
    parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    parser.add_argument(
        "--window", type=int, default=60, help="特征窗口大小 (默认: 60)"
    )
    parser.add_argument("--step", type=int, default=10, help="滑动步长 (默认: 10)")
    parser.add_argument(
        "--output",
        type=str,
        default="./factor_test_results",
        help="输出目录 (默认: ./factor_test_results)",
    )
    parser.add_argument("--quick", action="store_true", help="快速测试（使用模拟数据）")

    args = parser.parse_args()

    tester = FactorPerformanceTester(output_dir=args.output)

    if args.quick:
        tester.quick_test()
    else:
        tester.run_test(
            instrument_id=args.instrument,
            start_ymd=args.start,
            end_ymd=args.end,
            use_mock_data=args.mock,
            window_size=args.window,
            step_size=args.step,
        )


if __name__ == "__main__":
    main()
