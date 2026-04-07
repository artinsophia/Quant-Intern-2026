import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append("/home/jovyan/work/base_demo")
try:
    import base_tool

    BASE_TOOL_AVAILABLE = True
except ImportError:
    BASE_TOOL_AVAILABLE = False
    print("警告: base_tool 模块不可用，将使用本地替代函数")

# 导入本地替代函数
sys.path.append("/home/jovyan/work/tactics_demo/tools")
from backtest_quick import backtest_quick


def single_day_backtest(
    instrument_id,
    trade_ymd,
    strategy,
    param_dict=None,
    figsize=(16, 8),
    title_suffix="",
    save_path=None,
):
    """
    单日回测可视化函数

    功能：
    1. 运行单日回测，获取持仓变化和盈亏数据
    2. 绘制价格随时间走势图
    3. 标注变仓节点（开仓/平仓点）
    4. 用不同颜色标注持仓段的盈亏情况

    Args:
        instrument_id: 合约ID，如 '511520'
        trade_ymd: 交易日期，如 '20260319'
        strategy: 策略对象（需有 on_snap 方法）
        param_dict: 策略参数字典，需包含 'name' 键
        figsize: 图形大小
        title_suffix: 标题后缀
        save_path: 保存路径（可选）

    Returns:
        dict: 包含回测结果和持仓数据的字典
    """

    if param_dict is None:
        param_dict = {}

    strategy_name = param_dict.get("name", "strategy")

    # 1. 加载数据
    try:
        if BASE_TOOL_AVAILABLE:
            snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
        else:
            print(f"错误: 无法加载 {trade_ymd} 的快照数据，base_tool 不可用")
            return None

        if not snap_list or len(snap_list) == 0:
            print(f"日期 {trade_ymd} 无数据")
            return None
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

    # 2. 运行策略，记录持仓变化
    position_dict = {}
    position_history = []  # 记录每个时间点的持仓
    price_history = []  # 记录每个时间点的价格
    time_history = []  # 记录每个时间点的时间戳
    time_str_history = []  # 记录每个时间点的时间字符串

    for snap in snap_list:
        strategy.on_snap(snap)
        time_mark = snap.get("time_mark", 0)
        position_last = strategy.position_last
        price_last = snap.get("price_last", 0)
        time_hms = snap.get("time_hms", "")

        position_dict[time_mark] = position_last
        position_history.append(position_last)
        price_history.append(price_last)
        time_history.append(time_mark)
        time_str_history.append(time_hms)

    # 3. 运行回测获取盈亏数据
    data_file = f"/home/jovyan/work/backtest_result/{instrument_id}_{trade_ymd}_{strategy_name}.pkl"
    if os.path.exists(data_file):
        os.remove(data_file)

    try:
        # 使用本地替代函数
        profit_df = backtest_quick(
            instrument_id, trade_ymd, strategy_name, position_dict
        )
    except Exception as e:
        print(f"回测执行失败: {e}")
        profit_df = None

    # 4. 准备绘图数据
    price_history = np.array(price_history)
    position_history = np.array(position_history)

    # 找到持仓变化的点
    position_changes = np.diff(position_history, prepend=0)
    change_indices = np.where(position_changes != 0)[0]

    # 5. 分析持仓段
    segments = analyze_position_segments(position_history, price_history)

    # 6. 创建图形（单图，省略持仓变化图）
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
            pnl_text = f"{seg['pnl']:.3f}"
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
        Patch(facecolor="green", alpha=0.3, label="Long Position"),
        Patch(facecolor="red", alpha=0.3, label="Short Position"),
        Patch(facecolor="gray", alpha=0.3, label="No Position"),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=10,
            label="Open Long",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="red",
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

    plt.tight_layout()

    # 7. 显示回测结果摘要
    if profit_df is not None and len(profit_df) > 0:
        # 计算交易次数（更准确的方法）
        # 每个完整交易：开仓+平仓 = 2个变仓点
        # 但需要排除从非0到非0的变化（如直接从1到-1）
        complete_trades = 0
        trade_states = []

        for i in range(len(change_indices)):
            idx = change_indices[i]
            if i == 0:
                prev_pos = 0
            else:
                prev_idx = change_indices[i - 1]
                prev_pos = position_history[prev_idx]

            current_pos = position_history[idx]

            # 开仓：从0到非0
            if prev_pos == 0 and current_pos != 0:
                trade_states.append(
                    {"type": "open", "position": current_pos, "idx": idx}
                )
            # 平仓：从非0到0
            elif prev_pos != 0 and current_pos == 0:
                trade_states.append({"type": "close", "position": prev_pos, "idx": idx})
            # 换仓：从非0到另一个非0（如1到-1）
            elif prev_pos != 0 and current_pos != 0 and prev_pos != current_pos:
                # 这算作平仓+开仓两个交易
                trade_states.append({"type": "close", "position": prev_pos, "idx": idx})
                trade_states.append(
                    {"type": "open", "position": current_pos, "idx": idx}
                )

        # 计算完整交易数（开仓必须有对应的平仓）
        open_trades = [t for t in trade_states if t["type"] == "open"]
        close_trades = [t for t in trade_states if t["type"] == "close"]
        complete_trades = min(len(open_trades), len(close_trades))

        final_pnl = (
            profit_df["profits"].iloc[-1] if "profits" in profit_df.columns else 0
        )
        # 乘以100，因为backtest_quick.py中的盈亏是以1股为单位计算的，但实际交易是100股
        final_pnl *= 100
        avg_trade_pnl = final_pnl / complete_trades if complete_trades > 0 else 0

        summary_text = (
            f"Complete Trades: {complete_trades} | "
            f"Position Changes: {len(change_indices)} | "
            f"Final P&L: {final_pnl:.2f} | "
            f"Avg Trade P&L: {avg_trade_pnl:.2f}"
        )

        fig.text(
            0.02,
            0.02,
            summary_text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # 8. 返回结果
    result = {
        "instrument_id": instrument_id,
        "trade_ymd": trade_ymd,
        "strategy_name": strategy_name,
        "snap_count": len(snap_list),
        "position_changes": len(change_indices),
        "complete_trades": complete_trades if "complete_trades" in locals() else 0,
        "price_history": price_history,
        "position_history": position_history,
        "time_history": time_history,
        "time_str_history": time_str_history,
        "change_indices": change_indices.tolist(),
        "segments": segments,
        "profit_df": profit_df,
    }

    return result


def get_position_color(position):
    """根据持仓状态返回颜色"""
    if position == 1:  # 多头
        return "green"
    elif position == -1:  # 空头
        return "red"
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
    分析持仓段并计算每段的盈亏

    Args:
        position_history: 持仓历史数组
        price_history: 价格历史数组

    Returns:
        list: 持仓段列表，每个元素为 (start_idx, end_idx, position, entry_price, exit_price, pnl)
    """
    segments = []
    current_position = 0
    segment_start = 0
    entry_price = 0

    for i in range(len(position_history)):
        if i == 0:
            current_position = position_history[i]
            segment_start = i
            entry_price = price_history[i] if price_history[i] > 0 else 0
            continue

        # 持仓变化：开仓或平仓
        if position_history[i] != current_position:
            # 结束上一个持仓段
            if current_position != 0 and segment_start < i:
                exit_price = (
                    price_history[i - 1]
                    if price_history[i - 1] > 0
                    else price_history[i]
                )

                # 计算盈亏（简化版：价格差乘以持仓方向）
                if current_position == 1:  # 多头
                    pnl = exit_price - entry_price
                elif current_position == -1:  # 空头
                    pnl = entry_price - exit_price
                else:
                    pnl = 0

                segments.append(
                    {
                        "start_idx": segment_start,
                        "end_idx": i - 1,
                        "position": current_position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "duration": i - segment_start,
                    }
                )

            # 开始新的持仓段
            current_position = position_history[i]
            segment_start = i
            entry_price = price_history[i] if price_history[i] > 0 else 0

    # 处理最后一个持仓段
    if current_position != 0 and segment_start < len(position_history) - 1:
        exit_price = price_history[-1] if price_history[-1] > 0 else price_history[-2]

        if current_position == 1:  # 多头
            pnl = exit_price - entry_price
        elif current_position == -1:  # 空头
            pnl = entry_price - exit_price
        else:
            pnl = 0

        segments.append(
            {
                "start_idx": segment_start,
                "end_idx": len(position_history) - 1,
                "position": current_position,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "duration": len(position_history) - segment_start,
            }
        )

    return segments


def plot_enhanced_backtest(
    instrument_id,
    trade_ymd,
    strategy,
    param_dict=None,
    figsize=(18, 10),
    title_suffix="",
    save_path=None,
):
    """
    增强版单日回测可视化 - 包含持仓段盈亏分析和累计盈亏曲线

    功能：
    1. 运行单日回测并可视化
    2. 分析每个持仓段的盈亏
    3. 用颜色深浅表示盈亏大小
    4. 显示累计盈亏曲线
    5. 显示详细的交易统计

    Args:
        参数同 single_day_backtest

    Returns:
        dict: 包含详细分析结果的字典
    """

    # 运行基础回测
    result = single_day_backtest(
        instrument_id,
        trade_ymd,
        strategy,
        param_dict,
        (figsize[0], figsize[1] * 0.6),
        title_suffix,
        None,
    )

    if result is None:
        return None

    # 创建增强图形（两图：价格+持仓段，累计盈亏）
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]}
    )

    # 图1：价格走势和持仓段（复用single_day_backtest的逻辑但简化）
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
        Patch(facecolor="green", alpha=0.3, label="Long Position"),
        Patch(facecolor="red", alpha=0.3, label="Short Position"),
        Patch(facecolor="gray", alpha=0.3, label="No Position"),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=8,
            label="Open Long",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="red",
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

    # 图2：累计盈亏（如果可用）
    ax2 = axes[1]

    if result["profit_df"] is not None and "profits" in result["profit_df"].columns:
        profit_data = result["profit_df"]["profits"].values
        # 乘以100，因为backtest_quick.py中的盈亏是以1股为单位计算的，但实际交易是100股
        profit_data = profit_data * 100
        cumulative_pnl = np.cumsum(profit_data)

        # 绘制累计盈亏曲线
        trade_indices = range(len(cumulative_pnl))
        ax2.plot(
            trade_indices, cumulative_pnl, "b-", linewidth=2, label="Cumulative P&L"
        )

        # 填充正负区域
        ax2.fill_between(
            trade_indices,
            0,
            cumulative_pnl,
            where=(cumulative_pnl >= 0),
            alpha=0.3,
            color="green",
        )
        ax2.fill_between(
            trade_indices,
            0,
            cumulative_pnl,
            where=(cumulative_pnl < 0),
            alpha=0.3,
            color="red",
        )

        ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax2.set_xlabel("Trade Index")
        ax2.set_ylabel("Cumulative P&L")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")

        # 标注最终盈亏
        if len(cumulative_pnl) > 0:
            final_pnl = cumulative_pnl[-1]
            ax2.annotate(
                f"Final: {final_pnl:.2f}",
                xy=(len(cumulative_pnl) - 1, final_pnl),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
            )
    else:
        # 如果没有盈亏数据，显示持仓段统计
        if segments:
            segment_pnls = [seg["pnl"] for seg in segments]
            colors = ["green" if pnl >= 0 else "red" for pnl in segment_pnls]

            ax2.bar(range(len(segments)), segment_pnls, color=colors, alpha=0.7)
            ax2.set_xlabel("Trade Segment")
            ax2.set_ylabel("Segment P&L")
            ax2.set_title("Individual Trade Segment P&L")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1)

            # 标注统计信息
            if len(segments) > 0:
                total_pnl = sum(segment_pnls)
                ax2.annotate(
                    f"Total: {total_pnl:.3f}",
                    xy=(len(segments) - 1, segment_pnls[-1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "No trade data available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
            ax2.set_axis_off()

    plt.tight_layout()

    # 添加总体统计信息
    if segments:
        # 优先使用profit_df中的盈亏数据（已乘以100）
        if result["profit_df"] is not None and "profits" in result["profit_df"].columns:
            total_pnl = result["profit_df"]["profits"].sum() * 100
        else:
            total_pnl = sum(seg["pnl"] for seg in segments)

        winning_trades = sum(1 for seg in segments if seg["pnl"] > 0)
        losing_trades = sum(1 for seg in segments if seg["pnl"] < 0)
        win_rate = winning_trades / len(segments) * 100 if segments else 0

        stats_text = (
            f"Complete Trades: {result.get('complete_trades', len(segments))} | "
            f"Win Rate: {win_rate:.1f}% | "
            f"Total P&L: {total_pnl:.3f} | "
            f"Avg Trade P&L: {total_pnl / len(segments):.3f}"
        )

        fig.text(
            0.02,
            0.02,
            stats_text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
        )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return result


# 测试函数
if __name__ == "__main__":
    print("单日回测可视化工具")
    print("主要函数:")
    print("1. single_day_backtest_visualization() - 基础可视化")
    print("2. plot_enhanced_backtest() - 增强版可视化")
    print("3. analyze_position_segments() - 分析持仓段")

    # 示例策略类
    class TestStrategy:
        def __init__(self):
            self.position_last = 0

        def on_snap(self, snap):
            # 简单示例：价格高于某个阈值时做多，低于时做空
            price = snap.get("price_last", 0)
            if price > 100.5:
                self.position_last = 1
            elif price < 100.0:
                self.position_last = -1
            else:
                self.position_last = 0
