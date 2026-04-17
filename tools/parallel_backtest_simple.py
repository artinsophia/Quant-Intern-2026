#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
极简且高可用的并行回测模块
确保所有子进程使用与父进程完全一致的 Python 环境和模块路径。
建议将此代码保存为 parallel_backtest.py，并在其他脚本/Notebook 中 import 使用。
"""

import sys
import os
import math
import multiprocessing as mp
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================================
# 1. 环境预设与加固 (非常关键)
# ==========================================
CUSTOM_PATHS = [
    "/home/jovyan/base_demo",
    "/home/jovyan/work/tactics_demo/tools"
]

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
def worker_process(batch_dates, instrument_id, StrategyClass, model_path_or_obj, param_dict):
    """
    处理被分配的一批日期。
    """

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_TRADES'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # 子进程启动后第一件事：恢复系统路径
    _ensure_env()
    
    # 路径就绪后，再安全地导入自定义模块
    try:
        import base_tool
        from backtest_quick import backtest_quick
        import joblib
    except ImportError as e:
        print(f"[进程 {mp.current_process().name}] 模块导入失败: {e}，请检查路径。")
        return []

    # 加载模型（如果传的是路径，则进程内只加载一次，极其高效）
    model = joblib.load(model_path_or_obj) if isinstance(model_path_or_obj, str) else model_path_or_obj
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
            
            # 3. 执行单日回测
            profit_df = backtest_quick(
                instrument_id, trade_ymd, strategy_name, position_dict, remake=True
            )

            # 4. 提取并保存统计数据
            if profit_df is not None and not profit_df.empty and "profits" in profit_df.columns:
                trades = 0
                if "position" in profit_df.columns:
                    trades = ((profit_df["position"].shift(1).fillna(0) == 0) & (profit_df["position"] != 0)).sum()

                day_data = {
                    "trade_ymd": trade_ymd,
                    "profits": round(profit_df["profits"].iloc[-1], 2),
                    "trades": int(trades),
                }
                print(f"[{mp.current_process().name}] 日期 {trade_ymd} 完成 | 盈亏: {day_data['profits']:.2f}")
                results.append(day_data)
                
        except Exception as e:
            print(f"[{mp.current_process().name}] 日期 {trade_ymd} 执行异常: {e}")

    return results


# ==========================================
# 3. 主控函数 (Master)
# ==========================================
def run_parallel_backtest(instrument_id, start_ymd, end_ymd, StrategyClass, model, param_dict, n_cores=4):
    """
    主控函数：按核心数均分日期，并行执行回测并汇总结果。
    """
    # --- A. 环境与启动模式配置 ---
    # 尝试使用 fork 模式（Linux最佳实践，最完美继承环境内存和对象）
    try:
        mp.set_start_method('spawn', force=True)
        print("spawn模式启动")
    except RuntimeError:
        pass  # 如果已设置则忽略

    print(f"-> 当前 Python 解释器: {sys.executable}")
    
    # --- B. 准备日期切片 ---
    start_date = datetime.strptime(start_ymd, "%Y%m%d")
    end_date = datetime.strptime(end_ymd, "%Y%m%d")
    
    all_dates = []
    curr = start_date
    while curr <= end_date:
        all_dates.append(curr.strftime("%Y%m%d"))
        curr += timedelta(days=1)

    if not all_dates:
        print("-> 错误：日期范围为空")
        return None

    # 核心切分逻辑：总天数 / 核心数，向上取整
    chunk_size = math.ceil(len(all_dates) / n_cores)
    date_batches = [all_dates[i : i + chunk_size] for i in range(0, len(all_dates), chunk_size)]
    actual_cores = len(date_batches)
    
    print(f"-> 回测标的: {instrument_id} ({start_ymd} - {end_ymd})")
    print(f"-> 任务分配: 共 {len(all_dates)} 天，使用 {actual_cores} 个核心并行计算...")

    # --- C. 启动进程池 ---
    args_list = [(batch, instrument_id, StrategyClass, model, param_dict) for batch in date_batches]
    
    with mp.Pool(processes=actual_cores) as pool:
        # starmap 会阻塞并自动收集所有子进程的结果，不需要写复杂的异步超时逻辑
        batch_results = pool.starmap(worker_process, args_list)

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
    total_profit = cum_profits.iloc[-1]
    win_rate = (df["profits"] > 0).sum() / len(df) * 100 if len(df) > 0 else 0

    print("\n" + "=" * 45)
    print("回测统计结果汇总")
    print("=" * 45)
    print(f"总计天数: {len(df)} 天")
    print(f"盈利天数: {(df['profits'] > 0).sum()} ({win_rate:.1f}%)")
    print(f"累计盈亏: {total_profit:.2f}")
    print(f"日均盈亏: {df['profits'].mean():.2f}")
    print(f"总交易数: {df['trades'].sum()} 次")
    print("=" * 45)

    # --- E. 绘制图表 ---
    plt.figure(figsize=(12, 6))
    plt.plot(df["trade_date"], cum_profits, '-o', markersize=4, linewidth=2, color="#2E86AB")
    
    # 填充颜色表示正负收益
    plt.fill_between(df["trade_date"], 0, cum_profits, where=(cum_profits >= 0), color="green", alpha=0.1)
    plt.fill_between(df["trade_date"], 0, cum_profits, where=(cum_profits < 0), color="red", alpha=0.1)
    
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title(f"{instrument_id} Cumulative Profits ({start_ymd} to {end_ymd})")
    plt.ylabel("Cumulative Profit")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    print("提示：这是一个并行回测模块文件。")
    print("请将其保存为 parallel_backtest.py")
    print("并在你的业务脚本中使用: from parallel_backtest import run_parallel_backtest")