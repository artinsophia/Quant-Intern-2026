#!/usr/bin/env python3
import sys
import math
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import multiprocessing as mp
from datetime import datetime, timedelta

# 确保能找到你的自定义模块
sys.path.append("/home/jovyan/base_demo")
sys.path.append("/home/jovyan/work/tactics_demo/tools")

import base_tool
from backtest_quick import backtest_quick


def worker_process(batch_dates, instrument_id, StrategyClass, model_path_or_obj, param_dict):
    """
    核心工作进程：接收一批日期，连续跑完。
    这样每个进程只加载一次模型，极其高效。
    """
    # 1. 加载模型（如果传的是路径）
    model = joblib.load(model_path_or_obj) if isinstance(model_path_or_obj, str) else model_path_or_obj
    strategy_name = param_dict.get("name", "strategy")
    
    results = []
    
    # 2. 遍历该进程被分配的日期
    for trade_ymd in batch_dates:
        try:
            snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
            if not snap_list:
                continue

            # 实例化策略并灌入数据
            strategy = StrategyClass(model, param_dict)
            position_dict = {snap["time_mark"]: strategy.position_last for snap in snap_list if not strategy.on_snap(snap)}
            
            # 回测
            profit_df = backtest_quick(instrument_id, trade_ymd, strategy_name, position_dict, remake=True)

            if profit_df is not None and not profit_df.empty and "profits" in profit_df.columns:
                trades = 0
                if "position" in profit_df.columns:
                    trades = ((profit_df["position"].shift(1).fillna(0) == 0) & (profit_df["position"] != 0)).sum()

                day_data = {
                    "trade_ymd": trade_ymd,
                    "profits": round(profit_df["profits"].iloc[-1], 2),
                    "trades": int(trades),
                }
                print(f"[进程 {mp.current_process().name}] 日期 {trade_ymd} 完成, 盈亏: {day_data['profits']:.2f}")
                results.append(day_data)
                
        except Exception as e:
            print(f"日期 {trade_ymd} 异常跳过: {e}")

    return results


def run_parallel_backtest(instrument_id, start_ymd, end_ymd, StrategyClass, model, param_dict, n_cores=4):
    """
    主控函数：按核心数均分日期，并行执行回测。
    """
    # 1. 生成完整日期列表
    start_date = datetime.strptime(start_ymd, "%Y%m%d")
    end_date = datetime.strptime(end_ymd, "%Y%m%d")
    all_dates = [(start_date + timedelta(days=i)).strftime("%Y%m%d") 
                 for i in range((end_date - start_date).days + 1)]

    # 2. 核心逻辑：直接按核心数切分批次
    chunk_size = math.ceil(len(all_dates) / n_cores)
    date_batches = [all_dates[i : i + chunk_size] for i in range(0, len(all_dates), chunk_size)]
    
    actual_cores = len(date_batches)
    print(f"总计 {len(all_dates)} 天，切分为 {actual_cores} 个批次，并行回测中...")

    # 3. 构造参数并启动进程池
    args_list = [(batch, instrument_id, StrategyClass, model, param_dict) for batch in date_batches]
    
    # 使用 starmap 自动阻塞等待所有进程归来，省去了乱七八糟的 async/get 代码
    with mp.Pool(processes=actual_cores) as pool:
        batch_results = pool.starmap(worker_process, args_list)

    # 4. 展平二维数组 (List[List[dict]] -> List[dict])
    final_results = [item for sublist in batch_results for item in sublist]

    if not final_results:
        print("未获取到有效回测结果。")
        return None

    # 5. 整理数据并画图
    df = pd.DataFrame(final_results)
    df["trade_date"] = pd.to_datetime(df["trade_ymd"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    
    # 统计指标
    cum_profits = df["profits"].cumsum()
    total_profit = cum_profits.iloc[-1]
    win_rate = (df["profits"] > 0).sum() / len(df) * 100

    print("\n" + "=" * 40)
    print(f"回测完成 | 总天数: {len(df)} | 胜率: {win_rate:.1f}%")
    print(f"累计盈亏: {total_profit:.2f} | 总交易: {df['trades'].sum()}次")
    print("=" * 40)

    # 画图
    plt.figure(figsize=(12, 5))
    plt.plot(df["trade_date"], cum_profits, '-o', markersize=3, color="#2E86AB")
    plt.fill_between(df["trade_date"], 0, cum_profits, where=(cum_profits >= 0), color="green", alpha=0.1)
    plt.fill_between(df["trade_date"], 0, cum_profits, where=(cum_profits < 0), color="red", alpha=0.1)
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title(f"{instrument_id} Cumulative Profits")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df