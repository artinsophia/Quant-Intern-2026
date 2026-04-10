#!/usr/bin/env python3
"""
市场数据分析命令行工具
"""

import argparse
import sys
import os
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
from daily_analysis import DailyMarketAnalyzer


def main():
    parser = argparse.ArgumentParser(description="市场数据分析工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 单日分析命令
    parser_single = subparsers.add_parser("single", help="分析单个交易日")
    parser_single.add_argument("instrument", help="标的代码，如 511520, 511090, 518880")
    parser_single.add_argument("date", help="交易日期，格式 YYYYMMDD")
    parser_single.add_argument("--strategy", default="delta_v1_simple", help="策略名称")

    # 多日分析命令
    parser_multi = subparsers.add_parser("multi", help="分析多日数据")
    parser_multi.add_argument("instrument", help="标的代码")
    parser_multi.add_argument("start_date", help="开始日期，格式 YYYYMMDD")
    parser_multi.add_argument("end_date", help="结束日期，格式 YYYYMMDD")
    parser_multi.add_argument("--strategy", default="delta_v1_simple", help="策略名称")
    parser_multi.add_argument("--output", help="输出CSV文件路径")
    parser_multi.add_argument("--plot", action="store_true", help="生成图表")

    # 批量分析命令
    parser_batch = subparsers.add_parser("batch", help="批量分析多个标的")
    parser_batch.add_argument("instruments", nargs="+", help="标的代码列表")
    parser_batch.add_argument("start_date", help="开始日期，格式 YYYYMMDD")
    parser_batch.add_argument("end_date", help="结束日期，格式 YYYYMMDD")
    parser_batch.add_argument(
        "--output-dir", default="/home/jovyan/work", help="输出目录"
    )

    # 报告命令
    parser_report = subparsers.add_parser("report", help="生成分析报告")
    parser_report.add_argument("csv_file", help="CSV数据文件路径")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    analyzer = DailyMarketAnalyzer()

    if args.command == "single":
        print(f"分析 {args.instrument} 在 {args.date} 的数据...")
        stats = analyzer.analyze_single_day(args.instrument, args.date, args.strategy)

        if stats:
            print(f"\n分析结果:")
            print(f"标的: {stats['instrument_id']}")
            print(f"日期: {stats['trade_date']}")
            print(f"数据点: {stats['data_points']}")
            print(f"时段: {stats['time_period']}")

            if "price_change_pct" in stats:
                print(f"涨跌幅: {stats['price_change_pct']:.2f}%")
                print(f"日内波幅: {stats['price_range_pct']:.2f}%")

            if "final_profit" in stats:
                print(f"策略盈利: {stats['final_profit']:.2f}")
        else:
            print("未找到数据")

    elif args.command == "multi":
        print(
            f"分析 {args.instrument} 从 {args.start_date} 到 {args.end_date} 的数据..."
        )
        df_stats = analyzer.analyze_multiple_days(
            args.instrument, args.start_date, args.end_date, args.strategy
        )

        if not df_stats.empty:
            print(f"\n成功分析 {len(df_stats)} 个交易日")

            # 生成报告
            report = analyzer.generate_summary_report(df_stats)
            print(report)

            # 保存结果
            if args.output:
                output_file = args.output
            else:
                output_file = f"/home/jovyan/work/market_analysis_{args.instrument}_{args.start_date}_{args.end_date}.csv"

            df_stats.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n详细数据已保存至: {output_file}")

            # 生成图表
            if args.plot:
                chart_file = output_file.replace(".csv", ".png")
                analyzer.plot_daily_metrics(df_stats, save_path=chart_file)
                print(f"图表已保存至: {chart_file}")
        else:
            print("未找到数据")

    elif args.command == "batch":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"批量分析 {len(args.instruments)} 个标的...")
        print(f"时间范围: {args.start_date} 到 {args.end_date}")
        print()

        for instrument in args.instruments:
            print(f"分析 {instrument}...")
            df_stats = analyzer.analyze_multiple_days(
                instrument, args.start_date, args.end_date
            )

            if not df_stats.empty:
                # 保存结果
                output_file = (
                    output_dir
                    / f"market_analysis_{instrument}_{args.start_date}_{args.end_date}.csv"
                )
                df_stats.to_csv(output_file, index=False, encoding="utf-8-sig")

                # 生成简要报告
                if "price_change_pct" in df_stats.columns:
                    avg_change = df_stats["price_change_pct"].mean()
                    print(f"  ✓ 平均日涨跌幅: {avg_change:.2f}%")
                    print(f"  ✓ 数据已保存至: {output_file}")
                else:
                    print(f"  ✓ 数据已保存至: {output_file}")
            else:
                print(f"  ✗ 无数据")

            print()

        print("批量分析完成")

    elif args.command == "report":
        csv_file = Path(args.csv_file)
        if not csv_file.exists():
            print(f"错误: 文件不存在 {csv_file}")
            return

        try:
            import pandas as pd

            df_stats = pd.read_csv(csv_file)

            print(f"从 {csv_file} 加载数据")
            print(f"数据行数: {len(df_stats)}")
            print()

            report = analyzer.generate_summary_report(df_stats)
            print(report)

            # 保存报告到文件
            report_file = csv_file.with_suffix(".txt")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n报告已保存至: {report_file}")

        except Exception as e:
            print(f"读取文件时出错: {e}")


if __name__ == "__main__":
    main()
