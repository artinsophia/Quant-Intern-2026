import sys
import os
from datetime import datetime, timedelta

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
import pandas as pd


def backtest_multi_days(instrument_id, start_ymd, end_ymd, strategy, param_dict):
    """
    多天回测函数

    Args:
        instrument_id: 合约ID，如 '511520'
        start_ymd: 开始日期，如 '20260319'
        end_ymd: 结束日期，如 '20260325'
        strategy: 策略对象
        param_dict: 策略参数字典，需包含 'name' 键

    Returns:
        多天回测结果DataFrame
    """

    start_date = datetime.strptime(start_ymd, "%Y%m%d")
    end_date = datetime.strptime(end_ymd, "%Y%m%d")

    all_results = []

    current_date = start_date
    while current_date <= end_date:
        trade_ymd = current_date.strftime("%Y%m%d")

        try:
            if BASE_TOOL_AVAILABLE:
                snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
            else:
                # 尝试从其他位置加载数据
                print(f"警告: 无法加载 {trade_ymd} 的快照数据，base_tool 不可用")
                current_date += timedelta(days=1)
                continue

            if not snap_list or len(snap_list) == 0:
                print(f"日期 {trade_ymd} 无数据，跳过")
                current_date += timedelta(days=1)
                continue

            strategy_name = param_dict.get("name", "strategy")

            data_file = f"/home/jovyan/work/backtest_result/{instrument_id}_{trade_ymd}_{strategy_name}.pkl"
            if os.path.exists(data_file):
                os.remove(data_file)

            position_dict = {}
            for snap in snap_list:
                strategy.on_snap(snap)
                position_dict[snap["time_mark"]] = strategy.position_last

            # 使用本地替代函数
            profit = backtest_quick(
                instrument_id, trade_ymd, strategy_name, position_dict
            )

            if profit is not None and len(profit) > 0:
                # 查找汇总行（time_mark为"汇总"的行）
                summary_rows = profit[profit["time_mark"] == "汇总"]
                if len(summary_rows) > 0:
                    day_summary = summary_rows.iloc[[0]].copy()
                else:
                    # 如果没有汇总行，取最后一行
                    day_summary = profit.iloc[[-1]].copy()

                day_summary["trade_ymd"] = trade_ymd
                all_results.append(day_summary)

                # 获取profits值，处理可能的NaN
                profits_value = (
                    day_summary["profits"].values[0]
                    if "profits" in day_summary.columns
                    else 0
                )
                if pd.isna(profits_value):
                    profits_value = 0

                print(f"日期 {trade_ymd} 回测完成，当日盈亏: {profits_value:.2f}")
            else:
                print(f"日期 {trade_ymd} 无交易记录")

        except (Exception, SystemExit) as e:
            print(f"日期 {trade_ymd} 处理失败: {e}，跳过")

        current_date += timedelta(days=1)

    if not all_results:
        print("所有日期均无有效数据")
        return None

    result_df = pd.concat(all_results, ignore_index=True)
    cols = ["trade_ymd"] + [c for c in result_df.columns if c != "trade_ymd"]
    result_df = result_df[cols]

    # ==================== 绘制累计收益图 ====================
    if result_df is not None and len(result_df) > 0:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 准备数据
        result_df["trade_date"] = pd.to_datetime(
            result_df["trade_ymd"], format="%Y%m%d"
        )
        result_df = result_df.sort_values("trade_date")
        cumulative_profit = result_df["profits"].cumsum()

        # 创建图形
        plt.figure(figsize=(12, 6))
        plt.plot(
            result_df["trade_date"],
            cumulative_profit,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=4,
            color="#2E86AB",
        )
        plt.axhline(y=0, color="gray", linestyle="--", linewidth=1)

        # 正负区域填充
        plt.fill_between(
            result_df["trade_date"],
            0,
            cumulative_profit,
            where=(cumulative_profit >= 0),
            color="#A3C4BC",
            alpha=0.3,
            interpolate=True,
        )
        plt.fill_between(
            result_df["trade_date"],
            0,
            cumulative_profit,
            where=(cumulative_profit < 0),
            color="#D95F5F",
            alpha=0.3,
            interpolate=True,
        )

        # 标签和标题
        strategy_name = param_dict.get("name", "strategy")
        plt.title(
            f"{instrument_id}  {strategy_name} cumulative ({start_ymd} ~ {end_ymd})",
            fontsize=14,
        )
        plt.xlabel("date", fontsize=12)
        plt.ylabel("cumulative profit", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()  # 自动旋转日期标签

        # 标注最后一个点
        last_date = result_df["trade_date"].iloc[-1]
        last_cum = cumulative_profit.iloc[-1]
        plt.annotate(
            f"{last_cum:.2f}",
            xy=(last_date, last_cum),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

        plt.show()

    # ======================================================

    return result_df


def backtest_summary(daily_df):
    """
    汇总多天回测结果

    Args:
        daily_df: backtest_multi_days 返回的每日结果DataFrame

    Returns:
        汇总统计字典
    """
    if daily_df is None or len(daily_df) == 0:
        return None

    total_days = len(daily_df)

    # 累计盈亏
    total_profits = daily_df["profits"].sum()

    max_profit = daily_df["profits"].max()
    min_profit = daily_df["profits"].min()

    win_days = (daily_df["profits"] > 0).sum()
    loss_days = (daily_df["profits"] < 0).sum()
    flat_days = total_days - win_days - loss_days

    win_rate = win_days / total_days * 100 if total_days > 0 else 0
    avg_profit = daily_df["profits"].mean()

    # 计算盈亏比
    avg_win = daily_df[daily_df["profits"] > 0]["profits"].mean() if win_days > 0 else 0
    avg_loss = (
        abs(daily_df[daily_df["profits"] < 0]["profits"].mean()) if loss_days > 0 else 0
    )
    profit_factor = avg_win / avg_loss if avg_loss != 0 else 0

    # 计算交易次数和平均交易盈利（如果数据中有交易次数信息）
    total_trades = 0
    avg_trade_profit = 0

    # 检查是否有交易次数列
    if "trade" in daily_df.columns:
        total_trades = daily_df["trade"].sum()
        if total_trades > 0:
            avg_trade_profit = total_profits / total_trades
    else:
        # 如果没有交易次数列，尝试从其他列推断或设为0
        total_trades = 0
        avg_trade_profit = 0

    summary = {
        "交易天数": total_days,
        "累计盈亏": round(total_profits, 2),
        "最大单日盈利": round(max_profit, 2),
        "最大单日亏损": round(min_profit, 2),
        "盈利天数": int(win_days),
        "亏损天数": int(loss_days),
        "平盘天数": int(flat_days),
        "胜率(%)": round(win_rate, 2),
        "日均盈亏": round(avg_profit, 2) if not pd.isna(avg_profit) else 0,
        "盈亏比": round(profit_factor, 2),
        "交易次数": int(total_trades),
        "平均交易盈利": round(avg_trade_profit, 2)
        if not pd.isna(avg_trade_profit)
        else 0,
    }

    return summary
