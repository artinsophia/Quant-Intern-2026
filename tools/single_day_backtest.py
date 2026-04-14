import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys


def single_day_backtest(
    instrument_id,
    trade_ymd,
    StrategyClass,
    model,
    param_dict=None,
    figsize=(16, 8),
    title_suffix="",
    save_path=None,
    official=False,
):
    if param_dict is None:
        param_dict = {}
    strategy_name = param_dict.get("name", "strategy")

    # 1. 加载数据 (保持原样)
    import base_tool

    if official == True:
        from base_tool import backtest_quick
    else:
        from backtest_quick import backtest_quick

    snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
    if not snap_list:
        return None

    # 2. 运行策略，收集仓位信号
    strategy = StrategyClass(model, param_dict)
    position_dict = {}
    price_history = []
    time_history = []
    position_history = []

    for snap in snap_list:
        strategy.on_snap(snap)
        t, p, pos = snap["time_mark"], snap["price_last"], strategy.position_last
        position_dict[t] = pos
        price_history.append(p)
        time_history.append(t)
        position_history.append(pos)

    # 3. 运行【新版】回测获取盈亏数据
    # 注意：这里的 profit_df["profits"] 已经是累计盈亏了
    profit_df = backtest_quick(
        instrument_id, trade_ymd, strategy_name, position_dict, remake=1
    )

    # 4. 准备绘图数据
    price_history = np.array(price_history)
    position_history = np.array(position_history)
    change_indices = np.where(np.diff(position_history, prepend=0) != 0)[0]
    segments = analyze_position_segments(position_history, price_history)

    # 5. 绘图 (调用下方的增强版函数或在此处直接画)
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制价格曲线
    x = range(len(price_history))
    ax.plot(x, price_history, "k-", linewidth=1.5, alpha=0.7, label="Price")

    # 标注持仓段（用颜色区分）
    for seg in segments:
        color = get_position_color(seg["position"])
        alpha = 0.2 + 0.1 * min(abs(seg["pnl"]) / max(0.01, np.std(price_history)), 1.0)

        ax.axvspan(seg["start_idx"], seg["end_idx"], alpha=alpha, color=color)

        # 标注盈亏（如果持仓段足够长）
        if seg["position"] != 0 and seg["end_idx"] - seg["start_idx"] > 10:
            mid_idx = (seg["start_idx"] + seg["end_idx"]) // 2
            pnl_text = f"{seg['pnl'] * 100:.3f}"
            ax.text(
                mid_idx,
                price_history[mid_idx],
                pnl_text,
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    # 标注变仓节点
    for idx in change_indices:
        if idx < len(price_history):
            # 根据持仓变化方向选择标记
            change_type = position_history[idx] - (
                position_history[idx - 1] if idx > 0 else 0
            )
            marker = get_change_marker(change_type)
            color = get_position_color(position_history[idx])

            label = None
            if change_type > 0:
                label = "Open Long" if idx == change_indices[0] else ""
            elif change_type < 0:
                label = "Open Short" if idx == change_indices[0] else ""
            else:
                label = "Close" if idx == change_indices[0] else ""

            ax.plot(
                idx,
                price_history[idx],
                marker=marker,
                markersize=10,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1,
                label=label,
            )

    ax.set_xlabel("Time Index")
    ax.set_ylabel("Price")
    ax.set_title(f"{instrument_id} - {trade_ymd} {strategy_name} {title_suffix}")
    ax.grid(True, alpha=0.3)

    # 添加图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.3, label="Long Position"),
        Patch(facecolor="green", alpha=0.3, label="Short Position"),
        Patch(facecolor="gray", alpha=0.3, label="No Position"),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=10,
            label="Open Long",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=10,
            label="Open Short",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=10,
            label="Close Position",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # --- 添加左下角统计信息 ---
    if profit_df is not None and "profits" in profit_df.columns:
        total_pnl = profit_df["profits"].iloc[-1]
        # 统计成交次数：仓位变动次数
        trade_count = 0
        if "position" in profit_df.columns:
            trade_count = (
                (profit_df["position"].shift(1).fillna(0) == 0)
                & (profit_df["position"] != 0)
            ).sum()
        else:
            trade_count = profit_df["trade"].iloc[-1]
        avg_pnl_per_trade = total_pnl / max(trade_count, 1) if trade_count > 0 else 0

        stats_text = (
            f"Total P&L: {total_pnl:.2f} | "
            f"Avg P&L/Trade: {avg_pnl_per_trade:.2f} | "
            f"Trade Count: {trade_count}"
        )
        fig.text(
            0.02,
            0.02,
            stats_text,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

    plt.tight_layout()

    # 统计信息同步
    final_pnl = profit_df["profits"].iloc[-1] if profit_df is not None else 0

    result = {
        "final_pnl": final_pnl,
        "price_history": price_history,
        "position_history": position_history,
        "change_indices": change_indices,
        "segments": segments,
        "profit_df": profit_df,
        "strategy_name": strategy_name,
    }
    return result


def get_position_color(position):
    """根据持仓状态返回颜色"""
    if position == 1:  # 多头
        return "red"
    elif position == -1:  # 空头
        return "green"
    else:  # 空仓
        return "gray"


def get_change_marker(change_type):
    """根据持仓变化类型返回标记符号"""
    if change_type > 0:  # 开多
        return "^"
    elif change_type < 0:  # 开空
        return "v"
    else:  # 平仓
        return "o"


def analyze_position_segments(position_history, price_history):
    """
    分析持仓段并计算每段的盈亏（绘图辅助函数）

    Args:
        position_history: 持仓历史数组 [0, 1, 1, 0, -1, ...]
        price_history: 价格历史数组

    Returns:
        list: 持仓段列表，每个元素包含开始/结束索引、持仓方向、该段价格盈亏
    """
    segments = []
    current_position = 0
    segment_start = 0
    entry_price = 0

    # 将输入转为 numpy 数组确保兼容性
    position_history = np.array(position_history)
    price_history = np.array(price_history)

    for i in range(len(position_history)):
        if i == 0:
            current_position = position_history[i]
            segment_start = i
            entry_price = price_history[i]
            continue

        # 当持仓发生变化时（开仓、平仓、反手）
        if position_history[i] != current_position:
            # 如果之前有持仓，记录上一个持仓段
            if current_position != 0:
                exit_price = price_history[i - 1]
                # 计算该段的简单价格差盈亏
                pnl = (
                    (exit_price - entry_price)
                    if current_position == 1
                    else (entry_price - exit_price)
                )

                segments.append(
                    {
                        "start_idx": segment_start,
                        "end_idx": i - 1,
                        "position": current_position,
                        "pnl": pnl,
                    }
                )

            # 开启新的一段
            current_position = position_history[i]
            segment_start = i
            entry_price = price_history[i]

    # 处理最后一段未平仓的情况
    if current_position != 0:
        exit_price = price_history[-1]
        pnl = (
            (exit_price - entry_price)
            if current_position == 1
            else (entry_price - exit_price)
        )
        segments.append(
            {
                "start_idx": segment_start,
                "end_idx": len(position_history) - 1,
                "position": current_position,
                "pnl": pnl,
            }
        )

    return segments


def plot_delta_history(
    instrument_id,
    trade_ymd,
    StrategyClass,
    model,
    param_dict=None,
    figsize=(16, 8),
    title_suffix="",
    save_path=None,
    official=False,
):
    """绘制delta随时间变化的图，并标出变仓点"""
    if param_dict is None:
        param_dict = {}
    strategy_name = param_dict.get("name", "strategy")

    import base_tool

    if official == True:
        from base_tool import backtest_quick
    else:
        from backtest_quick import backtest_quick

    snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
    if not snap_list:
        return None

    from collections import deque
    import itertools

    strategy = StrategyClass(model, param_dict)
    position_dict = {}
    delta_history = []
    zscore_history = []
    price_history = []
    time_history = []
    position_history = []

    standard_num = param_dict.get("standard_num", 10)
    x_window = param_dict.get("x_window", 300)
    short_window = param_dict.get("short_window", 60)

    delta_buffer = deque(maxlen=x_window)

    for snap in snap_list:
        strategy.on_snap(snap)

        delta = sum(vol for _, vol in snap["buy_trade"][:standard_num]) - sum(
            vol for _, vol in snap["sell_trade"][:standard_num]
        )
        delta_buffer.append(delta)

        if len(delta_buffer) < x_window:
            zscore_val = 0.0
        else:
            recent_delta = list(
                itertools.islice(
                    delta_buffer,
                    max(0, len(delta_buffer) - short_window),
                    None,
                )
            )
            mean = np.mean(recent_delta)
            std = np.std(recent_delta)
            if std == 0:
                zscore_val = 0.0
            else:
                zscore_val = (recent_delta[-1] - mean) / std

        t, p, pos = snap["time_mark"], snap["price_last"], strategy.position_last
        position_dict[t] = pos
        delta_history.append(delta)
        zscore_history.append(zscore_val)
        price_history.append(p)
        time_history.append(t)
        position_history.append(pos)

    profit_df = backtest_quick(
        instrument_id, trade_ymd, strategy_name, position_dict, remake=1
    )

    delta_history = np.array(delta_history)
    zscore_history = np.array(zscore_history)
    price_history = np.array(price_history)
    position_history = np.array(position_history)
    change_indices = np.where(np.diff(position_history, prepend=0) != 0)[0]

    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(zscore_history))
    ax.plot(x, zscore_history, "b-", linewidth=1.5, alpha=0.7, label="Delta Z-Score")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=2, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=-2, color="green", linestyle="--", linewidth=1, alpha=0.5)

    for idx in change_indices:
        if idx < len(zscore_history):
            change_type = position_history[idx] - (
                position_history[idx - 1] if idx > 0 else 0
            )
            marker = get_change_marker(change_type)
            color = get_position_color(position_history[idx])

            label = None
            if change_type > 0:
                label = "Open Long" if idx == change_indices[0] else ""
            elif change_type < 0:
                label = "Open Short" if idx == change_indices[0] else ""
            else:
                label = "Close" if idx == change_indices[0] else ""

            ax.plot(
                idx,
                zscore_history[idx],
                marker=marker,
                markersize=10,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1,
                label=label,
            )

    ax.set_xlabel("Time Index")
    ax.set_ylabel("Delta Z-Score")
    ax.set_title(f"{instrument_id} - {trade_ymd} {strategy_name} Delta {title_suffix}")
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="blue",
            linewidth=1.5,
            label="Delta",
        ),
        Patch(facecolor="red", alpha=0.3, label="Long Position"),
        Patch(facecolor="green", alpha=0.3, label="Short Position"),
        Patch(facecolor="gray", alpha=0.3, label="No Position"),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=10,
            label="Open Long",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=10,
            label="Open Short",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=10,
            label="Close Position",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    if profit_df is not None and "profits" in profit_df.columns:
        total_pnl = profit_df["profits"].iloc[-1]
        trade_count = 0
        if "position" in profit_df.columns:
            trade_count = (
                (profit_df["position"].shift(1).fillna(0) == 0)
                & (profit_df["position"] != 0)
            ).sum()
        else:
            trade_count = profit_df["trade"].iloc[-1]
        avg_pnl_per_trade = total_pnl / max(trade_count, 1) if trade_count > 0 else 0

        stats_text = (
            f"Total P&L: {total_pnl:.2f} | "
            f"Avg P&L/Trade: {avg_pnl_per_trade:.2f} | "
            f"Trade Count: {trade_count}"
        )
        fig.text(
            0.02,
            0.02,
            stats_text,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Delta plot saved to {save_path}")

    final_pnl = profit_df["profits"].iloc[-1] if profit_df is not None else 0

    result = {
        "final_pnl": final_pnl,
        "delta_history": zscore_history,
        "price_history": price_history,
        "position_history": position_history,
        "change_indices": change_indices,
        "profit_df": profit_df,
        "strategy_name": strategy_name,
    }
    return result
