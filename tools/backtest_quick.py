import pandas as pd
import numpy as np
import os
import pickle

def backtest_quick(instrument_id, trade_ymd, strategy_name, position_dict, remake=False):
    """
    简化版回测：使用向量化计算，只关注最终总盈亏
    """
    # 1. 缓存管理
    cache_dir = "/home/jovyan/work/backtest_result"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{instrument_id}_{trade_ymd}_{strategy_name}_simple.pkl"
    
    if not remake and os.path.exists(cache_file):
        with open(cache_file, "rb") as f: return pickle.load(f)

    # 2. 数据加载
    import sys
    sys.path.append("/home/jovyan/work/base_demo")
    try:
        import base_tool
        snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
    except: return None
    if not snap_list: return None

    # 3. 数据预处理
    df = pd.DataFrame(snap_list)
    df["time_mark"] = df["time_mark"].astype(int)
    df = df.sort_values("time_mark").reset_index(drop=True)

    # 4. 映射仓位信号 (将稀疏的信号填充到每一秒的快照上)
    pos_ser = pd.Series(position_dict)
    pos_ser.index = pos_ser.index.astype(int)
    
    df = df.set_index("time_mark")
    df["pos"] = pos_ser
    df["pos"] = df["pos"].ffill().fillna(0) # 信号前向填充
    
    # 强制尾盘平仓逻辑 (最后5分钟)
    df.iloc[-300:, df.columns.get_loc("pos")] = 0

    # 5. 【核心计算】盈亏与费用
    base_volume = 100        # 每手股数
    slippage_price = 0.00  # 固定滑点(价格绝对值，如1分钱)
    fee_rate = 0.000    # 手续费率 0.03%

    # A. 计算价格变动产生的收益 (当前持仓 * 下一刻价格变动)
    # 使用 shift(1) 是因为这秒的价格变动是由上一秒持有的仓位决定的
    df['price_diff'] = df['price_last'].diff().fillna(0)
    df['pnl_raw'] = df['pos'].shift(1).fillna(0) * df['price_diff'] * base_volume

    # B. 计算交易成本 (仓位变化时产生)
    # pos_diff > 0 表示买入/加仓/平空，pos_diff < 0 表示卖出/减仓/平多
    df['pos_diff'] = df['pos'].diff().fillna(0).abs()
    
    # 交易额 = 变化量 * 价格 * 基础量
    trade_value = df['pos_diff'] * df['price_last'] * base_volume
    # 成本 = 手续费 + 滑点损失
    df['costs'] = (trade_value * fee_rate) + (df['pos_diff'] * slippage_price * base_volume)

    # C. 计算累计净盈亏
    df['net_pnl_step'] = df['pnl_raw'] - df['costs']
    df['cum_pnl'] = df['net_pnl_step'].cumsum()

    # 6. 整理输出格式
    result_df = df.reset_index()[['time_mark', 'price_last', 'pos', 'cum_pnl']]
    result_df.columns = ['time_mark', 'price_last', 'position', 'profits']

    # 保存缓存
    with open(cache_file, "wb") as f: pickle.dump(result_df, f)
    
    return result_df

def backtest_summary_quick(result_df):
    """
    极简统计
    """
    if result_df is None or len(result_df) == 0: return None
    
    # 注意：累计盈亏直接取最后一行，不要再 sum() 了！
    final_profit = result_df['profits'].iloc[-1]
    
    # 统计交易次数：仓位不等于前一时刻的次数
    trade_count = (result_df['position'].diff().fillna(0) != 0).sum()
    
    summary = {
        "总成交次数": int(trade_count),
        "最终总盈亏": round(final_profit, 2),
        "最大回撤": round((result_df['profits'].cummax() - result_df['profits']).max(), 2)
    }
    return summary