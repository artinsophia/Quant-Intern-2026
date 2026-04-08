import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# 导入你刚才修改过的简易版 backtest_quick
from backtest_quick import backtest_quick

def single_day_backtest(instrument_id, trade_ymd, strategy, param_dict=None, figsize=(16, 8), title_suffix="", save_path=None):
    if param_dict is None: param_dict = {}
    strategy_name = param_dict.get("name", "strategy")

    # 1. 加载数据 (保持原样)
    import base_tool
    snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
    if not snap_list: return None

    # 2. 运行策略，收集仓位信号
    position_dict = {}
    price_history = []
    time_history = []
    position_history = []

    for snap in snap_list:
        strategy.on_snap(snap)
        t, p, pos = snap['time_mark'], snap['price_last'], strategy.position_last
        position_dict[t] = pos
        price_history.append(p)
        time_history.append(t)
        position_history.append(pos)

    # 3. 运行【新版】回测获取盈亏数据
    # 注意：这里的 profit_df["profits"] 已经是累计盈亏了
    profit_df = backtest_quick(instrument_id, trade_ymd, strategy_name, position_dict, remake=True)

    # 4. 准备绘图数据
    price_history = np.array(price_history)
    position_history = np.array(position_history)
    change_indices = np.where(np.diff(position_history, prepend=0) != 0)[0]
    segments = analyze_position_segments(position_history, price_history)

    # 5. 绘图 (调用下方的增强版函数或在此处直接画)
    # 此处省略具体绘图代码，逻辑同下...
    
    # 统计信息同步
    final_pnl = profit_df["profits"].iloc[-1] if profit_df is not None else 0
    
    result = {
        "final_pnl": final_pnl,
        "price_history": price_history,
        "position_history": position_history,
        "change_indices": change_indices,
        "segments": segments,
        "profit_df": profit_df,
    }
    return result

def plot_enhanced_backtest(instrument_id, trade_ymd, strategy, param_dict=None, figsize=(18, 10), title_suffix="", save_path=None):
    # 运行回测获取结果
    result = single_day_backtest(instrument_id, trade_ymd, strategy, param_dict)
    if result is None: return

    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]})
    
    # --- 图1：价格走势与持仓（逻辑保持不变） ---
    ax1 = axes[0]
    # ... (此处代码与你提供的画图逻辑完全一致) ...
    # 备注：由于你要求“画图不用改”，这里保留你原来的 ax1 绘制逻辑即可
    
    # --- 图2：累计盈亏（同步修改重点！） ---
    ax2 = axes[1]
    profit_df = result["profit_df"]
    
    if profit_df is not None and "profits" in profit_df.columns:
        # 【修改点】：直接取值，不再用 np.cumsum()，因为回测函数里已经 cumsum 过了
        cumulative_pnl = profit_df["profits"].values
        
        # 对应到时间轴（由于快照可能比回测记录多，取对应的索引长度）
        x_pnl = range(len(cumulative_pnl))
        
        ax2.plot(x_pnl, cumulative_pnl, "b-", linewidth=2, label="Net Cumulative P&L")
        ax2.fill_between(x_pnl, 0, cumulative_pnl, where=(cumulative_pnl >= 0), alpha=0.2, color="green")
        ax2.fill_between(x_pnl, 0, cumulative_pnl, where=(cumulative_pnl < 0), alpha=0.2, color="red")
        
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("Account P&L")
        ax2.grid(True, alpha=0.3)
        
        # 标注终点
        final_val = cumulative_pnl[-1]
        ax2.annotate(f"Final Net: {final_val:.2f}", 
                     xy=(len(cumulative_pnl)-1, final_val), 
                     xytext=(10, 0), textcoords="offset points",
                     fontweight='bold', color='darkblue')
    else:
        ax2.text(0.5, 0.5, "No P&L Data Available", ha="center")

    # --- 统计汇总文字 ---
    if profit_df is not None:
        total_pnl = profit_df["profits"].iloc[-1]
        # 统计成交次数：仓位变动次数
        trade_count = (profit_df['position'].diff().fillna(0) != 0).sum()
        
        stats_text = (
            f"Total P&L: {total_pnl:.2f} | "
            f"Trade Count: {trade_count} | "
            f"Strategy: {param_dict.get('name', 'N/A')}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=12, 
                 bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show()