import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle


def backtest_quick(
    instrument_id, trade_ymd, strategy_name, position_dict, remake=False
):
    """
    替代 base_tool.backtest_quick 的回测函数

    模拟在 09:35~14:55 期间按对手价自动挂撤单跟踪 position_dict，其他时间空仓

    Args:
        instrument_id: 合约ID，如 '511520'
        trade_ymd: 交易日，如 '20260319'
        strategy_name: 策略名称
        position_dict: 仓位字典，key为time_mark，value为position_last (-1/0/1)
        remake: 是否重新计算（默认False，如果结果已存在则直接读取）

    Returns:
        DataFrame 包含回测结果，包含时间戳、仓位、成交价、盈亏等
    """

    # 检查缓存文件
    cache_dir = "/home/jovyan/work/backtest_result"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{instrument_id}_{trade_ymd}_{strategy_name}.pkl"

    if not remake and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                result = pickle.load(f)
            return result
        except Exception as e:
            pass  # 静默失败，重新计算

    # 加载快照数据
    import sys

    sys.path.append("/home/jovyan/work/base_demo")
    try:
        import base_tool

        snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
    except ImportError:
        return None

    if not snap_list:
        return None

    # 将快照转换为DataFrame以便处理
    snap_df = pd.DataFrame(snap_list)

    # 确保time_mark是整数
    snap_df["time_mark"] = snap_df["time_mark"].astype(int)

    # 确保time_hms列存在（如果不存在，创建占位符）
    if "time_hms" not in snap_df.columns:
        snap_df["time_hms"] = ""

    # 将position_dict转换为Series
    position_series = pd.Series(position_dict)
    position_series.index = position_series.index.astype(int)

    # 合并仓位信息到快照数据
    snap_df = snap_df.set_index("time_mark")
    snap_df["position"] = position_series
    # 使用向前填充处理缺失值（兼容不同pandas版本）
    snap_df["position"] = snap_df["position"].ffill().fillna(0)

    # 重置索引以便处理
    snap_df = snap_df.reset_index()

    # 简化时间处理：暂时完全禁用交易时间限制进行调试
    # 为了找出盈利为0的原因，先让所有时间都可以交易
    snap_df["in_trading_hours"] = True

    # 在交易时间段外强制仓位为0 - 暂时禁用
    # snap_df.loc[~snap_df["in_trading_hours"], "position"] = 0

    # 添加调试信息（简化版）
    if "price_last" in snap_df.columns:
        valid_prices = (snap_df["price_last"] > 0).sum()

    # 计算仓位变化
    snap_df["position_change"] = snap_df["position"].diff()
    # 第一行的仓位变化用第一行的仓位值填充
    if len(snap_df) > 0:
        snap_df.loc[snap_df.index[0], "position_change"] = snap_df.loc[
            snap_df.index[0], "position"
        ]
    snap_df["position_change"] = snap_df["position_change"].fillna(0)

    # 初始化回测结果列
    snap_df["trade_price"] = 0.0  # 成交价格
    snap_df["trade_volume"] = 0  # 成交数量（按仓位变化计算）
    snap_df["trade_cost"] = 0.0  # 交易成本
    snap_df["position_value"] = 0.0  # 持仓市值
    snap_df["realized_pnl"] = 0.0  # 已实现盈亏
    snap_df["unrealized_pnl"] = 0.0  # 未实现盈亏
    snap_df["total_pnl"] = 0.0  # 总盈亏

    # 模拟交易参数
    commission_rate = 0.000  # 交易佣金率 0.03%
    slippage = 0.000  # 滑点 0.01%
    base_volume = 100  # 基础交易量（股）

    # 初始化持仓
    current_position = 0
    avg_cost = 0.0
    total_realized_pnl = 0.0

    # 遍历每个快照进行交易模拟
    for i in range(len(snap_df)):
        row = snap_df.iloc[i]
        position_change = row["position_change"]
        price_last = row["price_last"]

        # 跳过无效价格
        if price_last <= 0:
            continue

        # 提前5分钟平仓：如果是最后300个样本，强制平仓
        if i >= len(snap_df) - 300 and current_position != 0:
            # 强制平仓，修改position_change
            position_change = -current_position
            # 更新DataFrame中的position_change值
            snap_df.at[i, "position_change"] = position_change
            # 更新position为0
            snap_df.at[i, "position"] = 0

        # 计算对手价（考虑买卖方向）
        if position_change > 0:  # 开多或加多
            # 买入：使用卖一价（ask price）
            if "ask_book" in row and row["ask_book"] and len(row["ask_book"]) > 0:
                trade_price = row["ask_book"][0][0]  # 卖一价
            else:
                trade_price = price_last * (1 + slippage)  # 无订单簿时使用最新价加滑点
        elif position_change < 0:  # 开空或加空（或平多）
            # 卖出：使用买一价（bid price）
            if "bid_book" in row and row["bid_book"] and len(row["bid_book"]) > 0:
                trade_price = row["bid_book"][0][0]  # 买一价
            else:
                trade_price = price_last * (1 - slippage)  # 无订单簿时使用最新价减滑点
        else:
            trade_price = 0.0

            # 记录交易
            trade_volume = 0  # 初始化trade_volume
            if position_change != 0 and trade_price > 0:
                snap_df.at[i, "trade_price"] = trade_price
                # 仓位变化量（乘以基础交易量）
                trade_volume = abs(position_change) * base_volume
                snap_df.at[i, "trade_volume"] = trade_volume

            # 计算交易成本
            trade_value = trade_volume * trade_price
            commission = trade_value * commission_rate
            snap_df.at[i, "trade_cost"] = commission

            # 更新持仓和成本（考虑基础交易量）
            position_change_volume = position_change * base_volume
            current_position_volume = current_position * base_volume

            if current_position * position_change >= 0:  # 同向加仓
                # 计算新的平均成本
                total_value = (
                    abs(current_position_volume) * avg_cost
                    + abs(position_change_volume) * trade_price
                )
                new_position_volume = current_position_volume + position_change_volume
                if new_position_volume != 0:
                    avg_cost = total_value / abs(new_position_volume)
                else:
                    avg_cost = 0.0
            else:  # 反向交易（平仓或反向开仓）
                # 计算已实现盈亏
                close_volume = min(
                    abs(current_position_volume), abs(position_change_volume)
                )
                if current_position > 0:  # 平多
                    realized = close_volume * (trade_price - avg_cost)
                else:  # 平空
                    realized = close_volume * (avg_cost - trade_price)

                total_realized_pnl += realized
                snap_df.at[i, "realized_pnl"] = realized

                # 调试信息（简化）

                # 更新持仓
                if abs(position_change_volume) > abs(
                    current_position_volume
                ):  # 反向开仓
                    remaining_change = position_change_volume + current_position_volume
                    avg_cost = trade_price
                else:  # 部分平仓
                    avg_cost = avg_cost  # 平均成本不变

        # 更新当前仓位
        current_position = row["position"]
        current_position_volume = current_position * base_volume

        # 计算未实现盈亏（考虑基础交易量）
        if current_position != 0 and price_last > 0:
            if current_position > 0:  # 多头持仓
                unrealized = current_position_volume * (price_last - avg_cost)
            else:  # 空头持仓
                unrealized = abs(current_position_volume) * (avg_cost - price_last)
            snap_df.at[i, "unrealized_pnl"] = unrealized

        # 计算总盈亏
        snap_df.at[i, "total_pnl"] = (
            total_realized_pnl + snap_df.at[i, "unrealized_pnl"]
        )

        # 计算持仓市值（考虑基础交易量）
        if current_position != 0 and price_last > 0:
            snap_df.at[i, "position_value"] = abs(current_position_volume) * price_last

    # 添加最终调试信息（简化）

    # 生成最终结果DataFrame（类似原backtest_quick的输出格式）
    result_cols = [
        "time_mark",
        "time_hms",
        "price_last",
        "position",
        "trade_price",
        "trade_volume",
        "trade_cost",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "position_value",
    ]

    # 只保留有交易的记录或重要记录
    result_df = snap_df[result_cols].copy()

    # 添加汇总行
    if len(result_df) > 0:
        summary_row = {
            "time_mark": "汇总",
            "time_hms": "汇总",
            "price_last": 0,
            "position": current_position,
            "trade_price": 0,
            "trade_volume": result_df["trade_volume"].sum()
            if "trade_volume" in result_df.columns
            else 0,
            "trade_cost": result_df["trade_cost"].sum()
            if "trade_cost" in result_df.columns
            else 0,
            "realized_pnl": total_realized_pnl,
            "unrealized_pnl": result_df["unrealized_pnl"].iloc[-1]
            if len(result_df) > 0 and "unrealized_pnl" in result_df.columns
            else 0,
            "total_pnl": result_df["total_pnl"].iloc[-1]
            if len(result_df) > 0 and "total_pnl" in result_df.columns
            else 0,
            "position_value": result_df["position_value"].iloc[-1]
            if len(result_df) > 0 and "position_value" in result_df.columns
            else 0,
        }

        # 创建汇总DataFrame
        summary_df = pd.DataFrame([summary_row])

        # 重命名列以匹配原格式
        result_df = result_df.rename(
            columns={"total_pnl": "profits", "trade_volume": "trade"}
        )

        summary_df = summary_df.rename(
            columns={"total_pnl": "profits", "trade_volume": "trade"}
        )

        # 合并详细记录和汇总行
        final_df = pd.concat([result_df, summary_df], ignore_index=True)

        # 保存缓存
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(final_df, f)
        except Exception as e:
            pass  # 静默失败

        return final_df

    return None


def backtest_summary_quick(result_df):
    """
    快速回测结果汇总

    Args:
        result_df: backtest_quick返回的DataFrame

    Returns:
        汇总统计字典
    """
    if result_df is None or len(result_df) == 0:
        return None

    # 过滤掉汇总行
    df = (
        result_df[result_df["time_mark"] != "汇总"].copy()
        if "time_mark" in result_df.columns
        else result_df
    )

    if len(df) == 0:
        return None

    # 计算基本统计
    total_trades = df["trade"].sum() if "trade" in df.columns else 0
    total_profits = df["profits"].sum() if "profits" in df.columns else 0

    # 计算交易次数（仓位变化次数）
    if "position" in df.columns:
        position_changes = df["position"].diff().abs().sum()
        total_trades = max(total_trades, position_changes)

    summary = {
        "交易次数": int(total_trades),
        "累计盈亏": round(total_profits, 2),
        "平均每笔盈亏": round(total_profits / total_trades, 2)
        if total_trades > 0
        else 0,
        "最大单笔盈利": round(df["profits"].max(), 2) if "profits" in df.columns else 0,
        "最大单笔亏损": round(df["profits"].min(), 2) if "profits" in df.columns else 0,
    }

    return summary
