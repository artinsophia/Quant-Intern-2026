import pandas as pd
import numpy as np
import os
import pickle

import pandas as pd
import numpy as np
import os
import pickle


def backtest_quick(
    instrument_id, trade_ymd, strategy_name, position_dict, remake=False
):
    """
    严谨对价版回测：买入看卖一，卖出看买一，向量化计算
    """
    cache_dir = "/home/jovyan/work/backtest_result"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{instrument_id}_{trade_ymd}_{strategy_name}_pro.pkl"

    if not remake and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # 1. 数据加载 (假设 snap_list 已获取)
    import sys

    sys.path.append("/home/jovyan/work/base_demo")
    try:
        import base_tool

        snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
    except:
        return None
    if not snap_list:
        return None

    # 2. 高效预处理：在转换为DF前提取买卖一价 (比之后用 apply 快)
    processed_snaps = []
    for s in snap_list:
        processed_snaps.append(
            {
                "time_mark": s["time_mark"],
                "price_last": s["price_last"],
                "bid_1": s["bid_book"][0][0] if s["bid_book"] else s["price_last"],
                "ask_1": s["ask_book"][0][0] if s["ask_book"] else s["price_last"],
            }
        )

    df = pd.DataFrame(processed_snaps)
    df["time_mark"] = df["time_mark"].astype(int)
    df = df.sort_values("time_mark").set_index("time_mark")

    # 3. 映射仓位
    pos_ser = pd.Series(position_dict)
    pos_ser.index = pos_ser.index.astype(int)
    df["target_pos"] = pos_ser
    df["pos"] = df["target_pos"].shift(1).ffill().fillna(0)

    # 强制尾盘平仓 (最后10分钟)
    df.iloc[-600:, df.columns.get_loc("pos")] = 0

    # 4. 【核心计算】对价成交逻辑
    base_volume = 100

    # 计算仓位变化
    df["pos_diff"] = df["pos"].diff().fillna(0)

    # 确定成交执行价：
    # 如果 pos_diff > 0 (买入)，用 ask_1
    # 如果 pos_diff < 0 (卖出)，用 bid_1
    # 如果 pos_diff = 0 (无交易)，价格不影响现金流
    df["exec_price"] = df["price_last"]  # 默认值
    df.loc[df["pos_diff"] > 0, "exec_price"] = df["ask_1"]
    df.loc[df["pos_diff"] < 0, "exec_price"] = df["bid_1"]

    # 5. 计算盈亏 (现金流法)
    # 交易发生的现金支出/收入 (买入为负，卖出为正)
    # 注意：这里可以轻松加入交易规费和滑点
    fee_rate = 0.000  # 假设万一手续费

    df["trade_cash_flow"] = -df["pos_diff"] * df["exec_price"] * base_volume
    df["transaction_fee"] = (
        df["pos_diff"].abs() * df["exec_price"] * base_volume * fee_rate
    )

    # 累计现金流
    df["cum_cash"] = (df["trade_cash_flow"] - df["transaction_fee"]).cumsum()

    # 当前持仓市值 (按 price_last 估值)
    df["inventory_value"] = df["pos"] * df["price_last"] * base_volume

    # 最终动态净盈亏
    df["profits"] = df["cum_cash"] + df["inventory_value"]

    # 6. 计算每笔交易的盈亏
    # 识别交易发生的时间点 (开仓和平仓)
    trade_points = df[df["pos_diff"] != 0].copy()

    # 初始化交易记录
    trades = []
    current_position = 0
    entry_price = 0
    entry_time = 0

    for idx, row in df.iterrows():
        if row["pos_diff"] != 0:
            # 这是交易发生点
            if current_position == 0:
                # 开仓
                current_position = row["pos"]
                entry_price = row["exec_price"]
                entry_time = idx
            else:
                # 平仓或反手
                exit_price = row["exec_price"]
                exit_time = idx

                # 计算这笔交易的盈亏
                if current_position == 1:  # 多头平仓
                    pnl = (exit_price - entry_price) * base_volume
                else:  # 空头平仓
                    pnl = (entry_price - exit_price) * base_volume

                # 记录交易
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "position": current_position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "win": pnl > 0,
                    }
                )

                # 如果 pos_diff 与当前仓位符号相同，说明是反手，需要记录新的开仓
                if row["pos"] != 0:
                    current_position = row["pos"]
                    entry_price = row["exec_price"]
                    entry_time = idx
                else:
                    current_position = 0

    # 7. 格式化输出
    result_df = df.reset_index()[["time_mark", "price_last", "pos", "profits"]]
    result_df.columns = ["time_mark", "price_last", "position", "profits"]
    result_df["holding"] = get_avg_holding_ticks(df)

    # 添加交易统计信息
    result_df.attrs["trades"] = trades

    with open(cache_file, "wb") as f:
        pickle.dump(result_df, f)
    return result_df


def get_avg_holding_ticks(df):
    # 1. 找到所有非零仓位的行
    # 创建一个辅助列，判断是否在持仓（不等于0即为持仓）
    holding_mask = df["pos"] != 0

    if not holding_mask.any():
        return 0

    # 2. 识别连续持仓的区间
    # 当 is_holding 的状态发生变化时，累加生成 ID
    # 这样每一段连续的持仓（或连续的空仓）都会有一个唯一的 block_id
    holding_blocks = (holding_mask != holding_mask.shift()).cumsum()

    # 3. 过滤出仅属于"持仓中"的 block
    # 并统计每个 block 包含的行数 (size)
    holding_durations = holding_blocks[holding_mask].value_counts()

    # 4. 计算平均行数（快照数）
    return holding_durations.mean()


def calculate_trade_stats(trades):
    """
    计算交易统计信息
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_pnl_per_trade": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
        }

    total_trades = len(trades)
    win_trades = sum(1 for trade in trades if trade["win"])
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    total_pnl = sum(trade["pnl"] for trade in trades)
    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0

    win_pnls = [trade["pnl"] for trade in trades if trade["win"]]
    loss_pnls = [trade["pnl"] for trade in trades if not trade["win"]]

    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = abs(sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0
    profit_factor = avg_win / avg_loss if avg_loss != 0 else 0

    return {
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }
