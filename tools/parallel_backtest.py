import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback
import warnings

warnings.filterwarnings("ignore")

# 确保路径正确
sys.path.append("/home/jovyan/work/base_demo")
sys.path.append("/home/jovyan/work/tactics_demo/tools")


def backtest_single_day(args):
    """
    单日回测函数，用于并行处理
    """
    (
        instrument_id,
        trade_ymd,
        StrategyClass,
        model,
        param_dict,
        strategy_name,
        official,
    ) = args

    try:
        import base_tool

        if official:
            from base_tool import backtest_quick
        else:
            from backtest_quick import backtest_quick

        # 创建策略实例
        strategy = StrategyClass(model, param_dict)

        # 加载数据
        snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
        if not snap_list:
            return {
                "trade_ymd": trade_ymd,
                "error": "无数据",
                "profits": 0,
                "trades": 0,
            }

        # 生成信号
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
            # 提取当日最终状态
            last_row = profit_df.iloc[[-1]].copy()

            # 统计当日交易次数
            trade_count = 0
            if "position" in profit_df.columns:
                trade_count = (
                    (profit_df["position"].shift(1).fillna(0) == 0)
                    & (profit_df["position"] != 0)
                ).sum()
            else:
                trade_count = profit_df["trade"].iloc[-1]

            return {
                "trade_ymd": trade_ymd,
                "profits": round(last_row["profits"].values[0], 2),
                "trades": int(trade_count),
                "error": None,
            }
        else:
            return {
                "trade_ymd": trade_ymd,
                "error": "回测结果为空",
                "profits": 0,
                "trades": 0,
            }

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 100:
            error_msg = error_msg[:100] + "..."
        return {"trade_ymd": trade_ymd, "error": error_msg, "profits": 0, "trades": 0}


def backtest_multi_days_parallel(
    instrument_id,
    start_ymd,
    end_ymd,
    StrategyClass,
    model,
    param_dict,
    official=False,
    max_workers=None,
    chunk_size=1,
):
    """
    并行多天回测函数 - 使用多进程并行处理

    Parameters:
    -----------
    instrument_id : str
        交易品种ID
    start_ymd : str
        开始日期，格式：YYYYMMDD
    end_ymd : str
        结束日期，格式：YYYYMMDD
    StrategyClass : class
        策略类
    model : str or object
        模型路径或模型对象
    param_dict : dict
        策略参数
    official : bool
        是否使用官方backtest_quick
    max_workers : int or None
        最大工作进程数，None表示使用CPU核心数
    chunk_size : int
        每个进程处理的天数块大小
    """
    import base_tool

    start_date = datetime.strptime(start_ymd, "%Y%m%d")
    end_date = datetime.strptime(end_ymd, "%Y%m%d")
    strategy_name = param_dict.get("name", "strategy")

    # 生成所有交易日
    trade_dates = []
    current_date = start_date
    while current_date <= end_date:
        trade_dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    print(f"开始并行回测，共 {len(trade_dates)} 个交易日")
    print(f"使用 {max_workers if max_workers else mp.cpu_count()} 个进程")

    # 准备参数
    args_list = [
        (
            instrument_id,
            trade_ymd,
            StrategyClass,
            model,
            param_dict,
            strategy_name,
            official,
        )
        for trade_ymd in trade_dates
    ]

    all_day_summaries = []
    error_count = 0

    # 使用进程池并行处理
    if max_workers is None:
        max_workers = mp.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_date = {
            executor.submit(backtest_single_day, args): args[1] for args in args_list
        }

        # 处理完成的任务
        for future in as_completed(future_to_date):
            trade_ymd = future_to_date[future]
            try:
                result = future.result(timeout=300)  # 5分钟超时
                if result["error"]:
                    print(f"日期 {trade_ymd} 出错: {result['error']}")
                    error_count += 1
                else:
                    all_day_summaries.append(result)
                    print(
                        f"日期 {trade_ymd} 完成，盈亏: {result['profits']:.2f}, 成交: {result['trades']}次"
                    )
            except Exception as e:
                print(f"日期 {trade_ymd} 处理异常: {str(e)[:100]}")
                error_count += 1

    print(f"\n回测完成，成功: {len(all_day_summaries)} 天，失败: {error_count} 天")

    if not all_day_summaries:
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
    plt.title(f"{instrument_id} 并行回测累计收益 ({start_ymd}-{end_ymd})")
    plt.xlabel("日期")
    plt.ylabel("累计收益")
    plt.grid(True, alpha=0.2)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.show()

    return result_df


def backtest_multi_days_parallel_by_chunk(
    instrument_id,
    start_ymd,
    end_ymd,
    StrategyClass,
    model,
    param_dict,
    official=False,
    max_workers=None,
    days_per_chunk=5,
):
    """
    按块并行回测 - 每个进程处理连续的多天

    Parameters:
    -----------
    days_per_chunk : int
        每个进程处理的连续天数
    """
    import base_tool

    start_date = datetime.strptime(start_ymd, "%Y%m%d")
    end_date = datetime.strptime(end_ymd, "%Y%m%d")
    strategy_name = param_dict.get("name", "strategy")

    # 生成所有交易日
    trade_dates = []
    current_date = start_date
    while current_date <= end_date:
        trade_dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    # 按块分组
    date_chunks = []
    for i in range(0, len(trade_dates), days_per_chunk):
        chunk = trade_dates[i : i + days_per_chunk]
        date_chunks.append(chunk)

    print(
        f"开始块并行回测，共 {len(trade_dates)} 个交易日，分成 {len(date_chunks)} 个块"
    )
    print(
        f"每个块最多 {days_per_chunk} 天，使用 {max_workers if max_workers else mp.cpu_count()} 个进程"
    )

    # 准备参数
    args_list = []
    for chunk in date_chunks:
        args_list.append(
            (
                instrument_id,
                chunk,
                StrategyClass,
                model,
                param_dict,
                strategy_name,
                official,
            )
        )

    all_day_summaries = []
    error_count = 0

    # 使用进程池并行处理
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(date_chunks))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有块任务
        future_to_chunk = {
            executor.submit(backtest_chunk, args): i for i, args in enumerate(args_list)
        }

        # 处理完成的任务
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_results = future.result(timeout=600)  # 10分钟超时
                for result in chunk_results:
                    if result["error"]:
                        print(f"日期 {result['trade_ymd']} 出错: {result['error']}")
                        error_count += 1
                    else:
                        all_day_summaries.append(result)
            except Exception as e:
                print(f"块 {chunk_idx} 处理异常: {str(e)[:100]}")
                error_count += len(date_chunks[chunk_idx])

    print(f"\n回测完成，成功: {len(all_day_summaries)} 天，失败: {error_count} 天")

    if not all_day_summaries:
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
    plt.title(f"{instrument_id} 块并行回测累计收益 ({start_ymd}-{end_ymd})")
    plt.xlabel("日期")
    plt.ylabel("累计收益")
    plt.grid(True, alpha=0.2)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.tight_layout()
    plt.show()

    return result_df


def backtest_chunk(args):
    """
    处理一个日期块
    """
    (
        instrument_id,
        date_chunk,
        StrategyClass,
        model,
        param_dict,
        strategy_name,
        official,
    ) = args

    results = []
    for trade_ymd in date_chunk:
        try:
            result = backtest_single_day(
                (
                    instrument_id,
                    trade_ymd,
                    StrategyClass,
                    model,
                    param_dict,
                    strategy_name,
                    official,
                )
            )
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "trade_ymd": trade_ymd,
                    "error": str(e)[:100],
                    "profits": 0,
                    "trades": 0,
                }
            )

    return results


def compare_parallel_vs_serial(
    instrument_id,
    start_ymd,
    end_ymd,
    StrategyClass,
    model,
    param_dict,
    official=False,
    max_workers=None,
):
    """
    比较并行和串行回测的性能
    """
    import time
    from multi_day_backtest import backtest_multi_days as backtest_multi_days_serial

    print("=" * 60)
    print("开始性能比较测试")
    print("=" * 60)

    # 串行回测
    print("\n1. 串行回测...")
    start_time = time.time()
    serial_result = backtest_multi_days_serial(
        instrument_id, start_ymd, end_ymd, StrategyClass, model, param_dict, official
    )
    serial_time = time.time() - start_time
    print(f"串行回测用时: {serial_time:.2f} 秒")

    # 并行回测
    print("\n2. 并行回测...")
    start_time = time.time()
    parallel_result = backtest_multi_days_parallel(
        instrument_id,
        start_ymd,
        end_ymd,
        StrategyClass,
        model,
        param_dict,
        official,
        max_workers,
    )
    parallel_time = time.time() - start_time
    print(f"并行回测用时: {parallel_time:.2f} 秒")

    # 性能比较
    print("\n" + "=" * 60)
    print("性能比较结果")
    print("=" * 60)
    print(f"串行用时: {serial_time:.2f} 秒")
    print(f"并行用时: {parallel_time:.2f} 秒")
    print(f"加速比: {serial_time / parallel_time:.2f}x")
    print(f"时间节省: {(serial_time - parallel_time) / serial_time * 100:.1f}%")

    # 结果一致性检查
    if serial_result is not None and parallel_result is not None:
        serial_total = serial_result["profits"].sum()
        parallel_total = parallel_result["profits"].sum()
        print(f"\n结果一致性检查:")
        print(f"串行总收益: {serial_total:.2f}")
        print(f"并行总收益: {parallel_total:.2f}")
        print(f"收益差异: {abs(serial_total - parallel_total):.2f}")

        if abs(serial_total - parallel_total) < 0.01:
            print("✓ 结果一致")
        else:
            print("⚠ 结果有差异")

    return {
        "serial_time": serial_time,
        "parallel_time": parallel_time,
        "speedup": serial_time / parallel_time,
        "serial_result": serial_result,
        "parallel_result": parallel_result,
    }
