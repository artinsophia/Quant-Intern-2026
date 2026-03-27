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