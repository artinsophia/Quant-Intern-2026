#!/usr/bin/env /opt/conda/bin/python3.13
"""
最简单的并行回测修复方案
直接使用multiprocessing，确保所有进程使用相同的Python环境
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import multiprocessing as mp
from multiprocessing import TimeoutError
from functools import partial

# 确保路径正确
sys.path.append("/home/jovyan/base_demo")
sys.path.append("/home/jovyan/work/tactics_demo/tools")

_model_obj = None
_strategy_class = None
_param_dict = None


def init_worker(model_path, StrategyClass, param_dict):
    global _model_obj, _strategy_class, _param_dict
    import joblib

    _model_obj = joblib.load(model_path)  # 只加载一次
    _strategy_class = StrategyClass
    _param_dict = param_dict


def worker_batch_days(date_batch, instrument_id, StrategyClass, model, param_dict):
    """
    批次处理函数 - 处理一批日期，避免反复加载模型
    """
    import sys
    import os

    # 确保路径正确
    sys.path.insert(0, "/home/jovyan/base_demo")
    sys.path.insert(0, "/home/jovyan/work/tactics_demo/tools")
    import base_tool
    from backtest_quick import backtest_quick
    import joblib

    try:
        # 策略名称
        strategy_name = param_dict.get("name", "strategy")

        # 加载模型（如果是路径）
        if isinstance(model, str):
            model_obj = joblib.load(model)
        else:
            model_obj = model

        batch_results = []

        for trade_ymd in date_batch:
            # 加载数据
            snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
            if not snap_list:
                print(f"日期 {trade_ymd} 无数据，跳过")
                continue

            # 生成信号
            strategy = StrategyClass(model_obj, param_dict)
            position_dict = {}
            for snap in snap_list:
                strategy.on_snap(snap)
                position_dict[snap["time_mark"]] = strategy.position_last

            # 运行回测
            profit_df = backtest_quick(
                instrument_id, trade_ymd, strategy_name, position_dict, remake=True
            )

            if (
                profit_df is not None
                and len(profit_df) > 0
                and "profits" in profit_df.columns
            ):
                # 统计当日交易次数
                trade_count = 0
                if "position" in profit_df.columns:
                    trade_count = (
                        (profit_df["position"].shift(1).fillna(0) == 0)
                        & (profit_df["position"] != 0)
                    ).sum()

                day_data = {
                    "trade_ymd": trade_ymd,
                    "profits": round(profit_df["profits"].iloc[-1], 2),
                    "trades": int(trade_count),
                }
                print(
                    f"日期 {trade_ymd} 完成，盈亏: {day_data['profits']:.2f}, 成交: {day_data['trades']}次"
                )
                batch_results.append(day_data)
            else:
                print(f"日期 {trade_ymd} 回测结果为空")

        return batch_results

    except Exception as e:
        print(f"批次处理出错: {e}")
        import traceback

        traceback.print_exc()
        return []


def worker_single_day(trade_ymd, instrument_id, StrategyClass, model, param_dict):
    """
    单日处理函数（保持向后兼容）
    """
    import sys
    import os

    # 确保路径正确
    sys.path.insert(0, "/home/jovyan/base_demo")
    sys.path.insert(0, "/home/jovyan/work/tactics_demo/tools")
    import base_tool
    from backtest_quick import backtest_quick
    import joblib

    try:
        # 策略名称
        strategy_name = param_dict.get("name", "strategy")

        # 加载数据
        snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
        if not snap_list:
            print(f"日期 {trade_ymd} 无数据，跳过")
            return None

        # 加载模型（如果是路径）
        if isinstance(model, str):
            model_obj = joblib.load(model)
        else:
            model_obj = model

        # 生成信号
        strategy = StrategyClass(model_obj, param_dict)
        position_dict = {}
        for snap in snap_list:
            strategy.on_snap(snap)
            position_dict[snap["time_mark"]] = strategy.position_last

        # 运行回测
        profit_df = backtest_quick(
            instrument_id, trade_ymd, strategy_name, position_dict, remake=True
        )

        if (
            profit_df is not None
            and len(profit_df) > 0
            and "profits" in profit_df.columns
        ):
            # 统计当日交易次数
            trade_count = 0
            if "position" in profit_df.columns:
                trade_count = (
                    (profit_df["position"].shift(1).fillna(0) == 0)
                    & (profit_df["position"] != 0)
                ).sum()

            day_data = {
                "trade_ymd": trade_ymd,
                "profits": round(profit_df["profits"].iloc[-1], 2),
                "trades": int(trade_count),
            }
            print(
                f"日期 {trade_ymd} 完成，盈亏: {day_data['profits']:.2f}, 成交: {day_data['trades']}次"
            )
            return day_data

    except Exception as e:
        print(f"日期 {trade_ymd} 出错: {e}")
        import traceback

        traceback.print_exc()

    return None


def backtest_multi_days_parallel_simple(
    instrument_id,
    start_ymd,
    end_ymd,
    StrategyClass,
    model,
    param_dict,
    n_processes=4,
    use_batch=True,
):
    """
    使用multiprocessing的简单并行回测，支持日期分批处理

    Parameters:
    -----------
    instrument_id : str
        标的代码
    start_ymd : str
        开始日期 (YYYYMMDD)
    end_ymd : str
        结束日期 (YYYYMMDD)
    StrategyClass : class
        策略类
    model : object or str
        模型对象或路径
    param_dict : dict
        策略参数
    n_processes : int
        进程数（也是批次数量）
    use_batch : bool
        是否使用批次处理模式，True时每个进程处理一个批次

    Returns:
    --------
    pd.DataFrame
        每日结果汇总
    """
    start_date = datetime.strptime(start_ymd, "%Y%m%d")
    end_date = datetime.strptime(end_ymd, "%Y%m%d")

    # 生成所有交易日列表
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    print(
        f"开始并行回测 {instrument_id}，共 {len(date_list)} 天，使用 {n_processes} 个进程"
    )
    print(f"Python版本: {sys.version}")

    if use_batch:
        # 自动计算批次大小：有多少进程就分多少批，平均分配
        batch_count = n_processes
        batch_size = max(1, len(date_list) // batch_count)
        if len(date_list) % batch_count != 0:
            batch_size += 1  # 向上取整

        print(f"使用批次处理模式，共 {batch_count} 批，每批约 {batch_size} 天")

        # 将日期列表分成批次
        date_batches = []
        for i in range(0, len(date_list), batch_size):
            batch = date_list[i : i + batch_size]
            date_batches.append(batch)

        print(f"实际分成 {len(date_batches)} 个批次")

        # 准备批次参数
        worker_args = [
            (batch, instrument_id, StrategyClass, model, param_dict)
            for batch in date_batches
        ]

        worker_func = worker_batch_days
    else:
        print("使用单日处理模式")
        # 准备单日参数
        worker_args = [
            (trade_ymd, instrument_id, StrategyClass, model, param_dict)
            for trade_ymd in date_list
        ]

        worker_func = worker_single_day

    # 使用进程池，设置maxtasksperchild防止内存泄漏
    results = []
    try:
        with mp.Pool(processes=n_processes, maxtasksperchild=5) as pool:
            # 使用starmap_async批量提交
            async_result = pool.starmap_async(worker_func, worker_args)

            # 等待结果，设置超时
            try:
                batch_results = async_result.get(timeout=600)  # 10分钟超时

                # 展开结果
                for result in batch_results:
                    if isinstance(result, list):
                        results.extend(result)
                    elif result is not None:
                        results.append(result)

            except TimeoutError:
                print("任务执行超时")
                pool.terminate()
                return None

    except KeyboardInterrupt:
        print("\n用户中断，正在终止进程池...")
        return None
    except Exception as e:
        print(f"并行执行出错: {e}")
        import traceback

        traceback.print_exc()
        return None

    # 过滤None结果
    all_day_summaries = [r for r in results if r is not None]

    if not all_day_summaries:
        print("无有效回测结果")
        return None

    # 汇总结果
    result_df = pd.DataFrame(all_day_summaries)
    result_df["trade_date"] = pd.to_datetime(result_df["trade_ymd"], format="%Y%m%d")
    result_df = result_df.sort_values("trade_date")

    # 绘制累计收益图
    plt.figure(figsize=(12, 6))
    cum_profit = result_df["profits"].cumsum()

    plt.plot(
        result_df["trade_date"],
        cum_profit,
        marker="o",
        markersize=4,
        linewidth=2,
        color="#2E86AB",
    )
    plt.fill_between(
        result_df["trade_date"],
        0,
        cum_profit,
        where=(cum_profit >= 0),
        color="green",
        alpha=0.1,
    )
    plt.fill_between(
        result_df["trade_date"],
        0,
        cum_profit,
        where=(cum_profit < 0),
        color="red",
        alpha=0.1,
    )

    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.title(
        f"{instrument_id} cumulative profits ({start_ymd}-{end_ymd}) [Simple Parallel]"
    )
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit")
    plt.grid(True, alpha=0.2)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.show()

    # 打印统计信息
    total_days = len(result_df)
    profitable_days = (result_df["profits"] > 0).sum()
    total_profit = cum_profit.iloc[-1]
    avg_profit_per_day = result_df["profits"].mean()
    win_rate = profitable_days / total_days * 100 if total_days > 0 else 0

    print("\n" + "=" * 50)
    print("回测统计结果:")
    print("=" * 50)
    print(f"总天数: {total_days}")
    print(f"盈利天数: {profitable_days} ({win_rate:.1f}%)")
    print(f"累计盈亏: {total_profit:.2f}")
    print(f"日均盈亏: {avg_profit_per_day:.2f}")
    print(f"总交易次数: {result_df['trades'].sum()}")
    print("=" * 50)

    return result_df


if __name__ == "__main__":
    # 测试代码
    print("This module should be imported, not run directly.")
    print(
        "Use: from parallel_backtest_simple import backtest_multi_days_parallel_simple"
    )
