import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mplfinance.original_flavor import candlestick_ohlc

def plot_kline(snap_list, title="K Plot", figsize=(16, 8), save_path=None, freq="1min"):
    """
    Draw candlestick chart
    
    Parameters:
    -----------
    snap_list : list[dict]
        Snapshot data list
    title : str
        Chart title
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path
    freq : str
        K-line frequency: "1min", "5min", "15min", "30min", "1h"
    """
    if not snap_list:
        print("Empty data")
        return
    
    df_list = []
    for snap in snap_list:
        if snap.get('price_last', 0) == 0:
            continue
        
        price = snap.get('price_last', 0)
        price_low = snap.get('price_low', 0)
        
        if price_low == 0:
            price_low = price
        
        df_list.append({
            'time_mark': snap.get('time_mark', 0),
            'open': snap.get('price_open', price),
            'high': snap.get('price_high', price),
            'low': price_low,
            'close': price
        })
    
    if not df_list:
        print("No valid data")
        return
    
    df = pd.DataFrame(df_list)
    
    df['datetime'] = pd.to_datetime(df['time_mark'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    df_ohlc = df.resample(freq).agg(agg_rules)
    df_ohlc = df_ohlc[df_ohlc['close'] > 0]
    df_ohlc.loc[df_ohlc['open'] == 0, 'open'] = df_ohlc.loc[df_ohlc['open'] == 0, 'close']
    df_ohlc.dropna(inplace=True)
    df_ohlc.reset_index(inplace=True)
    
    print(f"Total {len(df_ohlc)} K-lines with freq={freq}")
    
    df_ohlc['idx'] = range(len(df_ohlc))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    candlestick_ohlc(ax, df_ohlc[['idx', 'open', 'high', 'low', 'close']].values, 
                     width=0.6, colorup='red', colordown='green', alpha=0.8)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.set_xticks(df_ohlc['idx'][::max(1, len(df_ohlc)//10)])
    ax.set_xticklabels(df_ohlc['datetime'].dt.strftime('%H:%M')[::max(1, len(df_ohlc)//10)], rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return df_ohlc



import math
from typing import List, Dict, Any

def get_daily_statistics(snap_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    根据全天1s快照数据计算当日统计指标
    :param snap_list: 包含全天快照的列表，每个元素是一个字典
    :return: 包含统计结果的字典
    """
    
    # --- 1. 数据清洗与预处理 ---
    # 过滤掉完全没有成交且价格为0的无效快照（通常是午休或开盘前）
    # 注意：如果 price_last 为 0，说明该秒无成交，我们保留它用于时间对齐，但在计算价格变动时需特殊处理
    valid_prices = []
    total_buy_volume = 0
    total_sell_volume = 0
    total_turnover = 0.0
    
    # 用于计算波动率的收益率序列
    log_returns = []
    
    last_price = 0.0

    print(f"开始处理数据，总快照数: {len(snap_list)}")

    for snap in snap_list:
        # 提取关键字段
        p_last = snap.get('price_last', 0.0)
        buy_vol = snap.get('buy_trade', 0)
        sell_vol = snap.get('sell_trade', 0)
        
        # 累加成交量和成交额 (用于VWAP)
        # 只有当价格有效时才计算金额
        if p_last > 0:
            current_vol = buy_vol + sell_vol
            total_turnover += p_last * current_vol
            total_buy_volume += buy_vol
            total_sell_volume += sell_vol
            
            # 记录有效价格用于波动率计算
            if last_price > 0:
                # 计算对数收益率: ln(Pt / Pt-1)
                r = math.log(p_last / last_price)
                log_returns.append(r)
            
            valid_prices.append(p_last)
            last_price = p_last

    # --- 2. 计算统计指标 ---
    
    # A. 成交量加权平均价
    total_volume = total_buy_volume + total_sell_volume
    vwap = total_turnover / total_volume if total_volume > 0 else 0.0
    
    # B. 买卖压力 (主动买入占比)
    buy_ratio = total_buy_volume / total_volume if total_volume > 0 else 0.5
    
    # C. 波动率 (年化)
    # 假设A股一天交易约 14400 秒 (4小时)
    trading_seconds_per_year = 14400 * 240 
    volatility = 0.0
    
    if len(log_returns) > 1:
        # 1. 计算标准差
        mean_return = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_return) ** 2 for r in log_returns) / (len(log_returns) - 1)
        std_dev = math.sqrt(variance)
        
        # 2. 年化 (sqrt(240) 或 sqrt(交易秒数))
        # 这里计算的是基于秒级数据的日内波动率推演到年
        volatility = std_dev * math.sqrt(trading_seconds_per_year)
    
    # D. 价格区间
    high_price = max(valid_prices) if valid_prices else 0.0
    low_price = min(valid_prices) if valid_prices else 0.0
    open_price = valid_prices[0] if valid_prices else 0.0
    close_price = valid_prices[-1] if valid_prices else 0.0

    # --- 3. 返回结果 ---
    stats = {
        "date": snap_list[0]['time_hms'][:10] if snap_list else "Unknown", # 假设time_hms包含日期
        "vwap": round(vwap, 4),
        "volatility_annualized": round(volatility, 4),
        "total_volume": total_volume,
        "total_turnover": round(total_turnover, 2),
        "buy_ratio": round(buy_ratio, 4), # > 0.5 表示多头强势
        "price_range": {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        },
        "valid_snapshots": len(valid_prices)
    }
    
    return stats