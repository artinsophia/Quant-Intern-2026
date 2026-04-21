import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 确保路径正确
sys.path.append("/home/jovyan/work/base_demo")
sys.path.append("/home/jovyan/work/tactics_demo/tools")


def backtest_multi_days(
    instrument_id,
    start_ymd,
    end_ymd,
    StrategyClass,
    model,
    param_dict,
    official=False,
    delay_snaps=0,
):
    """
    多天回测函数 - 适配简易向量化版 backtest_quick


    """
    import base_tool

    if official == True:
        from base_tool import backtest_quick
    else:
        from backtest_quick import backtest_quick

    start_date = datetime.strptime(start_ymd, "%Y%m%d")
    end_date = datetime.strptime(end_ymd, "%Y%m%d")
    strategy_name = param_dict.get("name", "strategy")

    all_day_summaries = []
    current_date = start_date

    while current_date <= end_date:
        trade_ymd = current_date.strftime("%Y%m%d")
        strategy = StrategyClass(model, param_dict)
        try:
            snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
            if not snap_list:
                print(f"日期 {trade_ymd} 无数据，跳过")
                current_date += timedelta(days=1)
                continue

            # 2. 生成信号
            position_dict = {}
            for snap in snap_list:
                strategy.on_snap(snap)
                position_dict[snap["time_mark"]] = strategy.position_last

            # 3. 应用开仓延迟
            if delay_snaps > 0:
                from single_day_backtest import delay_open_position

                position_dict = delay_open_position(position_dict, delay_snaps)

            # 4. 运行回测 (remake=True 确保不读旧逻辑缓存)
            profit_df = backtest_quick(
                instrument_id, trade_ymd, strategy_name, position_dict, remake=True
            )

            if (
                profit_df is not None
                and len(profit_df) > 0
                and "profits" in profit_df.columns
            ):
                # 4. 提取当日最终状态
                # 新版 profits 已经是累加好的，最后一行即当日总盈亏
                last_row = profit_df.iloc[[-1]].copy()

                # 统计当日交易次数 (仓位变动次数)
                trade_count = 0
                if "position" in profit_df.columns:
                    trade_count = (
                        (profit_df["position"].shift(1).fillna(0) == 0)
                        & (profit_df["position"] != 0)
                    ).sum()
                else:
                    trade_count = profit_df["trade"].iloc[-1]
                # 获取平均持仓时间（快照数）
                avg_holding_ticks = 0
                if "holding" in profit_df.columns:
                    avg_holding_ticks = profit_df["holding"].iloc[-1]

                # 获取交易统计信息
                win_trades = 0
                loss_trades = 0
                win_rate = 0
                trade_pnl = 0
                avg_trade_pnl = 0

                if hasattr(profit_df, "attrs") and "trades" in profit_df.attrs:
                    from backtest_quick import calculate_trade_stats

                    trade_stats = calculate_trade_stats(profit_df.attrs["trades"])
                    win_trades = trade_stats["win_trades"]
                    loss_trades = trade_stats["loss_trades"]
                    win_rate = round(trade_stats["win_rate"] * 100, 2)
                    trade_pnl = round(trade_stats["total_pnl"], 2)
                    avg_trade_pnl = round(trade_stats["avg_pnl_per_trade"], 2)

                # 构造当日摘要
                day_data = {
                    "trade_ymd": trade_ymd,
                    "profits": round(last_row["profits"].values[0], 2),
                    "trades": int(trade_count),
                    "avg_holding_ticks": round(avg_holding_ticks, 2),
                    "win_trades": win_trades,
                    "loss_trades": loss_trades,
                    "win_rate": win_rate,
                    "trade_pnl": trade_pnl,
                    "avg_trade_pnl": avg_trade_pnl,
                }
                all_day_summaries.append(day_data)
                print(
                    f"日期 {trade_ymd} 完成，盈亏: {day_data['profits']:.2f}, 成交: {day_data['trades']}次, 胜率: {day_data['win_rate']:.1f}%, 平均持仓: {day_data['avg_holding_ticks']:.1f}快照"
                )

        except (SystemExit, Exception) as e:
            print(f"日期 {trade_ymd} 出错: {e}")
            # 继续处理下一天
            pass

        current_date += timedelta(days=1)

    if not all_day_summaries:
        return None

    # 5. 汇总 DataFrame
    result_df = pd.DataFrame(all_day_summaries)
    result_df["trade_date"] = pd.to_datetime(result_df["trade_ymd"], format="%Y%m%d")
    result_df = result_df.sort_values("trade_date")

    # 6. 绘制累计收益图
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
    plt.title(f"{instrument_id} cumulative profits ({start_ymd}-{end_ymd})")
    plt.grid(True, alpha=0.2)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.show()

    return result_df


def backtest_summary(daily_df):
    """
    多天结果综合统计
    """
    if daily_df is None or len(daily_df) == 0:
        return None

    # 基础指标
    total_profits = daily_df["profits"].sum()
    total_trades = daily_df["trades"].sum()
    total_days = len(daily_df)

    win_days = daily_df[daily_df["profits"] > 0]
    loss_days = daily_df[daily_df["profits"] < 0]

    # 盈亏比计算
    avg_win = win_days["profits"].mean() if len(win_days) > 0 else 0
    avg_loss = abs(loss_days["profits"].mean()) if len(loss_days) > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss != 0 else 0

    # 加权平均持仓时间计算（按交易次数加权）
    weighted_avg_holding = 0
    if total_trades > 0 and "avg_holding_ticks" in daily_df.columns:
        # 按每日交易次数加权计算平均持仓时间
        weighted_sum = (daily_df["avg_holding_ticks"] * daily_df["trades"]).sum()
        weighted_avg_holding = weighted_sum / total_trades

    # 每手胜率统计
    total_win_trades = 0
    total_loss_trades = 0
    weighted_win_rate = 0
    avg_trade_pnl = 0

    if "win_trades" in daily_df.columns and "loss_trades" in daily_df.columns:
        total_win_trades = daily_df["win_trades"].sum()
        total_loss_trades = daily_df["loss_trades"].sum()
        if total_trades > 0:
            weighted_win_rate = total_win_trades / total_trades * 100

    if "trade_pnl" in daily_df.columns:
        total_trade_pnl = daily_df["trade_pnl"].sum()
        avg_trade_pnl = total_trade_pnl / total_trades if total_trades > 0 else 0
    else:
        avg_trade_pnl = total_profits / total_trades if total_trades > 0 else 0

    summary = {
        "测试天数": total_days,
        "累计总盈亏": round(total_profits, 2),
        "总成交次数": int(total_trades),
        "盈利交易次数": int(total_win_trades),
        "亏损交易次数": int(total_loss_trades),
        "加权每手胜率%": round(weighted_win_rate, 2),
        "日均盈亏": round(total_profits / total_days, 2),
        "胜率(天)%": round(len(win_days) / total_days * 100, 2),
        "盈亏比(日均)": round(profit_factor, 2),
        "最大单日盈利": round(daily_df["profits"].max(), 2),
        "最大单日亏损": round(daily_df["profits"].min(), 2),
        "每笔交易平均盈利": round(avg_trade_pnl, 2),
        "加权平均持仓时间(快照)": round(weighted_avg_holding, 2),
    }

    return summary
