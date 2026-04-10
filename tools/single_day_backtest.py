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
            trade_count = profit_df['trade'].iloc[-1]
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


def plot_enhanced_backtest(
    instrument_id,
    trade_ymd,
    StrategyClass,
    param_dict=None,
    figsize=(18, 10),
    title_suffix="",
    save_path=None,
):
    # 运行回测获取结果
    result = single_day_backtest(
        instrument_id, trade_ymd, StrategyClass, None, param_dict
    )
    if result is None:
        return

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- 图1：价格走势与持仓（逻辑保持不变） ---
    ax1 = axes[0]
    price_history = result["price_history"]
    position_history = result["position_history"]
    segments = result["segments"]
    change_indices = result["change_indices"]
    x = range(len(price_history))

    # 绘制价格曲线
    ax1.plot(x, price_history, "k-", linewidth=1.5, alpha=0.7, label="Price")

    # 标注持仓段（用颜色区分）
    for seg in segments:
        color = get_position_color(seg["position"])
        alpha = 0.2 + 0.1 * min(abs(seg["pnl"]) / max(0.01, np.std(price_history)), 1.0)

        ax1.axvspan(seg["start_idx"], seg["end_idx"], alpha=alpha, color=color)

        # 标注盈亏（如果持仓段足够长）
        if seg["position"] != 0 and seg["end_idx"] - seg["start_idx"] > 10:
            mid_idx = (seg["start_idx"] + seg["end_idx"]) // 2
            pnl_text = f"{seg['pnl']:.3f}"
            ax1.text(
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
            change_type = position_history[idx] - (
                position_history[idx - 1] if idx > 0 else 0
            )
            marker = get_change_marker(change_type)
            color = get_position_color(position_history[idx])

            ax1.plot(
                idx,
                price_history[idx],
                marker=marker,
                markersize=8,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1,
            )

    ax1.set_xlabel("Time Index")
    ax1.set_ylabel("Price")
    ax1.set_title(
        f"{instrument_id} - {trade_ymd} {result['strategy_name']} {title_suffix}"
    )
    ax1.grid(True, alpha=0.3)

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
            markersize=8,
            label="Open Long",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=8,
            label="Open Short",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label="Close Position",
        ),
    ]
    ax1.legend(handles=legend_elements, loc="upper left")
    # --- 图2：累计盈亏（同步修改重点！） ---
    ax2 = axes[1]
    profit_df = result["profit_df"]

    if profit_df is not None and "profits" in profit_df.columns:
        # 【修改点】：直接取值，不再用 np.cumsum()，因为回测函数里已经 cumsum 过了
        cumulative_pnl = profit_df["profits"].values

        # 对应到时间轴（由于快照可能比回测记录多，取对应的索引长度）
        x_pnl = range(len(cumulative_pnl))

        ax2.plot(x_pnl, cumulative_pnl, "b-", linewidth=2, label="Net Cumulative P&L")
        ax2.fill_between(
            x_pnl,
            0,
            cumulative_pnl,
            where=(cumulative_pnl >= 0),
            alpha=0.2,
            color="red",
        )
        ax2.fill_between(
            x_pnl,
            0,
            cumulative_pnl,
            where=(cumulative_pnl < 0),
            alpha=0.2,
            color="green",
        )

        ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("Account P&L")
        ax2.grid(True, alpha=0.3)

        # 标注终点
        final_val = cumulative_pnl[-1]
        ax2.annotate(
            f"Final Net: {final_val:.2f}",
            xy=(len(cumulative_pnl) - 1, final_val),
            xytext=(10, 0),
            textcoords="offset points",
            fontweight="bold",
            color="darkblue",
        )
    else:
        ax2.text(0.5, 0.5, "No P&L Data Available", ha="center")

    # --- 统计汇总文字 ---
    if profit_df is not None and "profits" in profit_df.columns:
        total_pnl = profit_df["profits"].iloc[-1]
        # 统计成交次数：仓位变动次数
        trade_count = 0
        if "position" in profit_df.columns:
            trade_count = (
                (profit_df["position"].shift(1).fillna(0) == 0)
                & (profit_df["position"] != 0)
            ).sum()

        stats_text = (
            f"Total P&L: {total_pnl:.2f} | "
            f"Trade Count: {trade_count} | "
            f"Strategy: {param_dict.get('name', 'N/A')}"
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
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


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
