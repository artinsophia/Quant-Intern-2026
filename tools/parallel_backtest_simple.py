#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
极简且高可用的并行回测模块
确保所有子进程使用与父进程完全一致的 Python 环境和模块路径。
建议将此代码保存为 parallel_backtest.py，并在其他脚本/Notebook 中 import 使用。
"""

import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import math
import multiprocessing as mp
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gc

# ==========================================
# 1. 环境预设与加固 (非常关键)
# ==========================================
CUSTOM_PATHS = ["/home/jovyan/base_demo", "/home/jovyan/work/tactics_demo/tools"]


def _ensure_env():
    """确保当前进程（无论是父进程还是子进程）能找到自定义模块"""
    for path in CUSTOM_PATHS:
        if path not in sys.path:
            sys.path.insert(0, path)


# 立即为父进程注入路径
_ensure_env()


# ==========================================
# 2. 核心工作进程 (Worker)
# ==========================================
def worker_process(
    batch_dates, instrument_id, StrategyClass, model_path_or_obj, param_dict
):
    """
    处理被分配的一批日期。
    """

    # 子进程启动后第一件事：恢复系统路径
    _ensure_env()

    # 添加delta模块路径
    delta_path = "/home/jovyan/work/tactics_demo/delta"
    if delta_path not in sys.path:
        sys.path.insert(0, delta_path)

    # 路径就绪后，再安全地导入自定义模块
    try:
        import base_tool
        from backtest_quick import backtest_quick
        import joblib

        # 导入延迟开仓函数
        from single_day_backtest import delay_open_position
    except ImportError as e:
        print(f"[进程 {mp.current_process().name}] 模块导入失败: {e}，请检查路径。")
        return []

    # 加载模型（如果传的是路径，则进程内只加载一次，极其高效）
    model = (
        joblib.load(model_path_or_obj)
        if isinstance(model_path_or_obj, str)
        else model_path_or_obj
    )
    strategy_name = param_dict.get("name", "strategy")

    results = []

    # 遍历处理这批日期
    for trade_ymd in batch_dates:
        try:
            # 1. 加载快照数据
            snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
            if not snap_list:
                continue

            # 2. 实例化策略并生成信号
            strategy = StrategyClass(model, param_dict)
            position_dict = {}
            for snap in snap_list:
                strategy.on_snap(snap)
                position_dict[snap["time_mark"]] = strategy.position_last

            # 3. 应用开仓延迟
            delay_snaps = param_dict.get("delay_snaps", 0)
            if delay_snaps > 0:
                position_dict = delay_open_position(position_dict, delay_snaps)

            # 4. 执行单日回测
            profit_df = backtest_quick(
                instrument_id, trade_ymd, strategy_name, position_dict, remake=True
            )

            # 4. 提取并保存统计数据
            if (
                profit_df is not None
                and not profit_df.empty
                and "profits" in profit_df.columns
            ):
                trades = 0
                if "position" in profit_df.columns:
                    trades = (
                        (profit_df["position"].shift(1).fillna(0) == 0)
                        & (profit_df["position"] != 0)
                    ).sum()

                # 获取平均持仓时间
                avg_holding_ticks = 0
                if "holding" in profit_df.columns:
                    avg_holding_ticks = profit_df["holding"].iloc[-1]

                # 获取交易统计信息
                trade_stats = {}
                if hasattr(profit_df, "attrs") and "trades" in profit_df.attrs:
                    from backtest_quick import calculate_trade_stats

                    trade_stats = calculate_trade_stats(profit_df.attrs["trades"])

                day_data = {
                    "trade_ymd": trade_ymd,
                    "profits": round(profit_df["profits"].iloc[-1], 2),
                    "trades": int(trades),
                    "avg_holding_ticks": round(avg_holding_ticks, 2),
                }

                # 添加交易统计信息
                if trade_stats:
                    day_data.update(
                        {
                            "win_trades": trade_stats["win_trades"],
                            "loss_trades": trade_stats["loss_trades"],
                            "win_rate": round(trade_stats["win_rate"] * 100, 2),
                            "trade_pnl": round(trade_stats["total_pnl"], 2),
                            "avg_trade_pnl": round(trade_stats["avg_pnl_per_trade"], 2),
                        }
                    )
                else:
                    day_data.update(
                        {
                            "win_trades": 0,
                            "loss_trades": 0,
                            "win_rate": 0,
                            "trade_pnl": 0,
                            "avg_trade_pnl": 0,
                        }
                    )

                print(
                    f"[{mp.current_process().name}] 日期 {trade_ymd} 完成 | 盈亏: {day_data['profits']:.2f} | 成交: {day_data['trades']}次 | 胜率: {day_data['win_rate']:.1f}% | 平均持仓: {day_data['avg_holding_ticks']:.1f}快照"
                )
                results.append(day_data)

            del snap_list
            del strategy
            del position_dict
            del profit_df

            gc.collect()

        except Exception as e:
            print(f"[{mp.current_process().name}] 日期 {trade_ymd} 执行异常: {e}")

    return results


# ==========================================
# 3. 主控函数 (Master)
# ==========================================
def run_parallel_backtest(
    instrument_id, start_ymd, end_ymd, StrategyClass, model, param_dict, n_cores=4
):
    """
    主控函数：按核心数均分日期，并行执行回测并汇总结果。
    """
    # --- A. 环境与启动模式配置 ---

    print(f"-> Python解释器: {sys.executable}")
    print(f"-> Python版本: {sys.version}")

    try:
        mp.set_start_method("spawn", force=True)
        print("spawn模式启动")
    except RuntimeError:
        pass  # 如果已设置则忽略

    print(f"-> 当前 Python 解释器: {sys.executable}")

    # --- B. 准备日期切片 ---
    start_date = datetime.strptime(start_ymd, "%Y%m%d")
    end_date = datetime.strptime(end_ymd, "%Y%m%d")

    # 获取所有可能的日期
    all_possible_dates = []
    curr = start_date
    while curr <= end_date:
        all_possible_dates.append(curr.strftime("%Y%m%d"))
        curr += timedelta(days=1)

    if not all_possible_dates:
        print("-> 错误：日期范围为空")
        return None

    # 从delta/train.py获取有数据的交易日列表
    def extract_trade_dates_from_file():
        """直接从delta/train.py文件中提取交易日列表"""
        file_path = "/home/jovyan/work/tactics_demo/delta/train.py"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 查找get_trade_dates函数
            start_marker = "def get_trade_dates():"
            end_marker = "]"

            start_idx = content.find(start_marker)
            if start_idx == -1:
                return []

            # 找到函数体开始
            body_start = content.find("[", start_idx)
            if body_start == -1:
                return []

            # 找到列表结束
            bracket_count = 1
            i = body_start + 1
            while i < len(content) and bracket_count > 0:
                if content[i] == "[":
                    bracket_count += 1
                elif content[i] == "]":
                    bracket_count -= 1
                i += 1

            if bracket_count > 0:
                return []

            # 提取列表内容
            list_content = content[body_start:i]

            # 使用eval安全地解析列表（因为内容是硬编码的字符串列表）
            try:
                trade_dates = eval(list_content)
                if isinstance(trade_dates, list):
                    return trade_dates
            except:
                pass

            return []
        except Exception as e:
            print(f"-> 读取delta/train.py文件失败: {e}")
            return []

    available_dates = extract_trade_dates_from_file()
    if available_dates:
        print(f"-> 从delta/train.py获取到 {len(available_dates)} 个有数据的交易日")

        # 过滤掉没有数据的日期
        all_dates = [date for date in all_possible_dates if date in available_dates]
        print(f"-> 过滤后剩余 {len(all_dates)} 个有数据的交易日")

        # 打印被过滤掉的日期
        filtered_dates = [
            date for date in all_possible_dates if date not in available_dates
        ]
        if filtered_dates:
            print(
                f"-> 过滤掉 {len(filtered_dates)} 个无数据日期: {filtered_dates[:10]}{'...' if len(filtered_dates) > 10 else ''}"
            )
    else:
        print(f"-> 警告：无法从delta/train.py获取交易日列表")
        print("-> 将使用所有日期（不进行数据过滤）")
        all_dates = all_possible_dates

    # 核心切分逻辑：总天数 / 核心数，向上取整
    chunk_size = math.ceil(len(all_dates) / n_cores)
    date_batches = [
        all_dates[i : i + chunk_size] for i in range(0, len(all_dates), chunk_size)
    ]
    actual_cores = len(date_batches)

    print(f"-> 回测标的: {instrument_id} ({start_ymd} - {end_ymd})")
    print(f"-> 任务分配: 共 {len(all_dates)} 天，使用 {actual_cores} 个核心并行计算...")

    # --- C. 启动进程池 ---
    args_list = [
        (batch, instrument_id, StrategyClass, model, param_dict)
        for batch in date_batches
    ]

    pool = mp.Pool(processes=actual_cores)

    async_result = pool.starmap_async(worker_process, args_list)
    batch_results = async_result.get(timeout=200)  # 超时60s
    print("回测结束，强制终止所有子进程")
    pool.terminate()
    pool.join()

    # --- D. 整理与汇总 ---
    # 展平二维数组：将各个进程返回的嵌套列表合并成一个扁平列表
    final_results = [item for sublist in batch_results for item in sublist]

    if not final_results:
        print("-> 警告：未获取到有效回测结果。")
        return None

    df = pd.DataFrame(final_results)
    df["trade_date"] = pd.to_datetime(df["trade_ymd"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 计算统计指标
    cum_profits = df["profits"].cumsum()

    # 计算加权汇总的每手胜率
    total_trades = df["trades"].sum()
    total_win_trades = df["win_trades"].sum() if "win_trades" in df.columns else 0
    total_loss_trades = df["loss_trades"].sum() if "loss_trades" in df.columns else 0

    weighted_win_rate = 0
    if total_trades > 0:
        weighted_win_rate = total_win_trades / total_trades * 100

    # 打印汇总统计
    print("\n" + "=" * 60)
    print("多日回测汇总统计")
    print("=" * 60)
    print(f"测试天数: {len(df)}")
    print(f"累计总盈亏: {df['profits'].sum():.2f}")
    print(f"总成交次数: {total_trades}")
    print(f"盈利交易次数: {total_win_trades}")
    print(f"亏损交易次数: {total_loss_trades}")
    print(f"加权每手胜率: {weighted_win_rate:.2f}%")

    if "avg_trade_pnl" in df.columns:
        total_trade_pnl = (
            df["trade_pnl"].sum() if "trade_pnl" in df.columns else df["profits"].sum()
        )
        avg_trade_pnl = total_trade_pnl / total_trades if total_trades > 0 else 0
        print(f"每笔交易平均盈亏: {avg_trade_pnl:.2f}")

    # 计算加权平均持仓时间
    if total_trades > 0 and "avg_holding_ticks" in df.columns:
        weighted_avg_holding = (
            df["avg_holding_ticks"] * df["trades"]
        ).sum() / total_trades
        print(f"加权平均持仓时间: {weighted_avg_holding:.2f} 快照")

    print("=" * 60)

    # --- E. 绘制图表 ---
    plt.figure(figsize=(12, 6))
    plt.plot(
        df["trade_date"], cum_profits, "-o", markersize=4, linewidth=2, color="#2E86AB"
    )

    # 填充颜色表示正负收益
    plt.fill_between(
        df["trade_date"],
        0,
        cum_profits,
        where=(cum_profits >= 0),
        color="green",
        alpha=0.1,
    )
    plt.fill_between(
        df["trade_date"],
        0,
        cum_profits,
        where=(cum_profits < 0),
        color="red",
        alpha=0.1,
    )

    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title(f"{instrument_id} Cumulative Profits ({start_ymd} to {end_ymd})")
    plt.ylabel("Cumulative Profit")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df
