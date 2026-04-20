#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
延迟稳定性批量检验模块
输入延迟列表 [1,2,4,8,16]，分别作为delay_snap参数跑回测
每次调用multi_day_backtest.py中的summary得到一行，最终返回拼接好的数据框
"""

import sys
import os
import pandas as pd
from datetime import datetime

# 确保路径正确
sys.path.append("/home/jovyan/work/base_demo")
sys.path.append("/home/jovyan/work/tactics_demo/tools")


def batch_delay_stability_test(
    instrument_id,
    start_ymd,
    end_ymd,
    StrategyClass,
    model,
    param_dict,
    delay_list=[1, 2, 4, 8, 16],
    use_parallel=True,
    n_cores=4,
):
    """
    批量延迟稳定性检验

    参数:
    ----------
    instrument_id : str
        标的代码，如 '511520'
    start_ymd : str
        开始日期，格式 'YYYYMMDD'
    end_ymd : str
        结束日期，格式 'YYYYMMDD'
    StrategyClass : class
        策略类
    model : object or str
        模型对象或模型文件路径
    param_dict : dict
        策略参数字典
    delay_list : list
        延迟快照数列表，如 [1, 2, 4, 8, 16]
    use_parallel : bool
        是否使用并行回测（默认True）
    n_cores : int
        并行核心数（仅当use_parallel=True时有效）

    返回:
    ----------
    pd.DataFrame
        包含各延迟参数回测结果的汇总数据框
        每行对应一个延迟值，包含所有统计指标
    """

    print(f"开始批量延迟稳定性检验")
    print(f"标的: {instrument_id}, 日期范围: {start_ymd} - {end_ymd}")
    print(f"延迟参数列表: {delay_list}")
    print(f"使用{'并行' if use_parallel else '串行'}回测")

    all_results = []

    for i, delay_snaps in enumerate(delay_list):
        print(f"\n{'=' * 60}")
        print(f"进度: {i + 1}/{len(delay_list)} - 测试延迟: {delay_snaps}快照")
        print(f"{'=' * 60}")

        # 创建当前延迟的参数副本
        current_param = param_dict.copy()
        current_param["delay_snaps"] = delay_snaps

        try:
            if use_parallel:
                # 使用并行回测
                from parallel_backtest_simple import run_parallel_backtest

                # 运行并行回测
                daily_df = run_parallel_backtest(
                    instrument_id=instrument_id,
                    start_ymd=start_ymd,
                    end_ymd=end_ymd,
                    StrategyClass=StrategyClass,
                    model=model,
                    param_dict=current_param,
                    n_cores=n_cores,
                )
            else:
                # 使用串行回测
                from multi_day_backtest import backtest_multi_days

                daily_df = backtest_multi_days(
                    instrument_id=instrument_id,
                    start_ymd=start_ymd,
                    end_ymd=end_ymd,
                    StrategyClass=StrategyClass,
                    model=model,
                    param_dict=current_param,
                    official=False,
                    delay_snaps=delay_snaps,
                )

            if daily_df is not None and not daily_df.empty:
                # 获取汇总统计
                from multi_day_backtest import backtest_summary

                summary = backtest_summary(daily_df)

                if summary:
                    # 添加延迟参数
                    summary["delay_snaps"] = delay_snaps
                    summary["测试模式"] = "并行" if use_parallel else "串行"

                    # 计算额外指标
                    total_days = summary["测试天数"]
                    total_profits = summary["累计总盈亏"]
                    total_trades = summary["总成交次数"]

                    # 日均交易次数
                    summary["日均交易次数"] = (
                        round(total_trades / total_days, 2) if total_days > 0 else 0
                    )

                    # 交易胜率（基于交易次数）
                    if "profits" in daily_df.columns and "trades" in daily_df.columns:
                        # 计算盈利交易天数
                        profitable_days = daily_df[daily_df["profits"] > 0]
                        profitable_trades = (
                            profitable_days["trades"].sum()
                            if not profitable_days.empty
                            else 0
                        )
                        summary["交易胜率%"] = (
                            round(profitable_trades / total_trades * 100, 2)
                            if total_trades > 0
                            else 0
                        )

                    all_results.append(summary)
                    print(
                        f"延迟 {delay_snaps} 完成: 累计盈亏={summary['累计总盈亏']:.2f}, 胜率={summary['胜率(天)%']:.1f}%"
                    )
                else:
                    print(f"警告: 延迟 {delay_snaps} 未获取到有效汇总统计")
            else:
                print(f"警告: 延迟 {delay_snaps} 未获取到有效回测结果")

        except Exception as e:
            print(f"错误: 延迟 {delay_snaps} 回测失败: {e}")
            import traceback

            traceback.print_exc()

    if not all_results:
        print("所有延迟参数回测均失败")
        return None

    # 创建汇总数据框
    result_df = pd.DataFrame(all_results)

    # 重新排序列顺序
    columns_order = [
        "delay_snaps",
        "测试模式",
        "测试天数",
        "累计总盈亏",
        "日均盈亏",
        "总成交次数",
        "日均交易次数",
        "胜率(天)%",
        "交易胜率%",
        "盈亏比(日均)",
        "每笔交易平均盈利",
        "加权平均持仓时间(快照)",
        "最大单日盈利",
        "最大单日亏损",
    ]

    # 只保留存在的列
    columns_order = [col for col in columns_order if col in result_df.columns]

    # 添加其他列
    other_cols = [col for col in result_df.columns if col not in columns_order]
    final_columns = columns_order + other_cols

    result_df = result_df[final_columns]
    result_df = result_df.sort_values("delay_snaps").reset_index(drop=True)

    # 输出汇总报告
    print(f"\n{'=' * 80}")
    print("延迟稳定性检验汇总报告")
    print(f"{'=' * 80}")
    print(result_df.to_string(index=False))

    # 绘制延迟与绩效的关系图
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 延迟 vs 累计盈亏
        ax1 = axes[0, 0]
        ax1.plot(
            result_df["delay_snaps"],
            result_df["累计总盈亏"],
            "o-",
            linewidth=2,
            markersize=8,
        )
        ax1.set_xlabel("delay_snaps")
        ax1.set_ylabel("cum_profits")
        ax1.set_title("delay vs profits")
        ax1.grid(True, alpha=0.3)

        # 2. 延迟 vs 胜率
        ax2 = axes[0, 1]
        ax2.plot(
            result_df["delay_snaps"],
            result_df["胜率(天)%"],
            "s-",
            linewidth=2,
            markersize=8,
            color="green",
        )
        ax2.set_xlabel("delay_snaps")
        ax2.set_ylabel("win_rate(%)")
        ax2.set_title("delay_snaps vs win_rate")
        ax2.grid(True, alpha=0.3)

        # 3. 延迟 vs 日均交易次数
        ax3 = axes[1, 0]
        ax3.plot(
            result_df["delay_snaps"],
            result_df["日均交易次数"],
            "^-",
            linewidth=2,
            markersize=8,
            color="orange",
        )
        ax3.set_xlabel("delay_snap")
        ax3.set_ylabel("日均交易次数")
        ax3.set_title("delay_snap vs trade_count")
        ax3.grid(True, alpha=0.3)

        # 4. 延迟 vs 平均持仓时间
        ax4 = axes[1, 1]
        ax4.plot(
            result_df["delay_snaps"],
            result_df["加权平均持仓时间(快照)"],
            "d-",
            linewidth=2,
            markersize=8,
            color="red",
        )
        ax4.set_xlabel("delay_snap")
        ax4.set_ylabel("holding_time")
        ax4.set_title("delay_snap vs holding_time")
        ax4.grid(True, alpha=0.3)

        plt.suptitle(
            f"{instrument_id} delay_robust ({start_ymd} - {end_ymd})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"绘图失败: {e}")

    return result_df


def compare_delay_performance(result_df, baseline_delay=0):
    """
    比较不同延迟参数相对于基准的表现

    参数:
    ----------
    result_df : pd.DataFrame
        batch_delay_stability_test 返回的结果数据框
    baseline_delay : int
        基准延迟值（默认为0，即无延迟）

    返回:
    ----------
    pd.DataFrame
        包含相对表现比较的数据框
    """
    if result_df is None or result_df.empty:
        print("无有效结果数据")
        return None

    # 查找基准延迟
    baseline_row = result_df[result_df["delay_snaps"] == baseline_delay]

    if baseline_row.empty:
        print(f"警告: 未找到基准延迟 {baseline_delay}，使用第一个延迟作为基准")
        baseline_row = result_df.iloc[[0]]
        baseline_delay = baseline_row["delay_snaps"].iloc[0]

    baseline = baseline_row.iloc[0]

    comparison_data = []

    for _, row in result_df.iterrows():
        delay = row["delay_snaps"]

        if delay == baseline_delay:
            # 基准行
            comp = {
                "delay_snaps": delay,
                "累计盈亏_绝对值": row["累计总盈亏"],
                "累计盈亏_相对基准": 0,
                "胜率_绝对值": row["胜率(天)%"],
                "胜率_相对基准": 0,
                "日均交易_绝对值": row["日均交易次数"],
                "日均交易_相对基准": 0,
                "平均持仓_绝对值": row["加权平均持仓时间(快照)"],
                "平均持仓_相对基准": 0,
                "备注": "基准",
            }
        else:
            # 计算相对变化
            profit_change = row["累计总盈亏"] - baseline["累计总盈亏"]
            profit_change_pct = (
                (profit_change / abs(baseline["累计总盈亏"])) * 100
                if baseline["累计总盈亏"] != 0
                else 0
            )

            winrate_change = row["胜率(天)%"] - baseline["胜率(天)%"]

            trade_freq_change = row["日均交易次数"] - baseline["日均交易次数"]
            trade_freq_change_pct = (
                (trade_freq_change / baseline["日均交易次数"]) * 100
                if baseline["日均交易次数"] != 0
                else 0
            )

            holding_change = (
                row["加权平均持仓时间(快照)"] - baseline["加权平均持仓时间(快照)"]
            )
            holding_change_pct = (
                (holding_change / baseline["加权平均持仓时间(快照)"]) * 100
                if baseline["加权平均持仓时间(快照)"] != 0
                else 0
            )

            comp = {
                "delay_snaps": delay,
                "累计盈亏_绝对值": row["累计总盈亏"],
                "累计盈亏_相对基准": round(profit_change_pct, 2),
                "胜率_绝对值": row["胜率(天)%"],
                "胜率_相对基准": round(winrate_change, 2),
                "日均交易_绝对值": row["日均交易次数"],
                "日均交易_相对基准": round(trade_freq_change_pct, 2),
                "平均持仓_绝对值": row["加权平均持仓时间(快照)"],
                "平均持仓_相对基准": round(holding_change_pct, 2),
                "备注": f"vs {baseline_delay}快照",
            }

        comparison_data.append(comp)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("delay_snaps").reset_index(drop=True)

    print(f"\n{'=' * 80}")
    print(f"延迟参数相对表现比较 (基准: {baseline_delay}快照)")
    print(f"{'=' * 80}")
    print(comparison_df.to_string(index=False))

    return comparison_df


if __name__ == "__main__":
    # 示例用法
    print("延迟稳定性批量检验模块")
    print("使用方法:")
    print("""
    from delay_stability_test import batch_delay_stability_test, compare_delay_performance
    
    # 定义策略参数
    param_dict = {
        "name": "MyStrategy",
        "window": 20,
        "threshold": 0.5
    }
    
    # 运行批量延迟检验
    result_df = batch_delay_stability_test(
        instrument_id='511520',
        start_ymd='20250201',
        end_ymd='20250228',
        StrategyClass=MyStrategy,
        model='path/to/model.pkl',
        param_dict=param_dict,
        delay_list=[0, 1, 2, 4, 8, 16],
        use_parallel=True,
        n_cores=4
    )
    
    # 比较相对表现
    if result_df is not None:
        comparison_df = compare_delay_performance(result_df, baseline_delay=0)
    """)
