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
    instrument_id, start_ymd, end_ymd, StrategyClass, model, param_dict, official=False
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

            # 3. 运行回测 (remake=True 确保不读旧逻辑缓存)
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
                    trade_count = profit_df['trade'].iloc[-1]
                # 构造当日摘要
                day_data = {
                    "trade_ymd": trade_ymd,
                    "profits": last_row["profits"].values[0],
                    "trades": int(trade_count),
                }
                all_day_summaries.append(day_data)
                print(
                    f"日期 {trade_ymd} 完成，盈亏: {day_data['profits']:.2f}, 成交: {day_data['trades']}次"
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

    summary = {
        "测试天数": total_days,
        "累计总盈亏": round(total_profits, 2),
        "总成交次数": int(total_trades),
        "日均盈亏": round(total_profits / total_days, 2),
        "胜率(天)%": round(len(win_days) / total_days * 100, 2),
        "盈亏比(日均)": round(profit_factor, 2),
        "最大单日盈利": round(daily_df["profits"].max(), 2),
        "最大单日亏损": round(daily_df["profits"].min(), 2),
        "每笔交易平均盈利": round(total_profits / total_trades, 2)
        if total_trades > 0
        else 0,
    }

    return summary
