import matplotlib.pyplot as plt
import numpy as np

def extract_volume_from_price_volume_data(data):
    """
    从 [price, volume] 格式的数据中提取成交量
    """
    total_volume = 0
    if isinstance(data, (list, np.ndarray)):
        for item in data:
            if isinstance(item, (list, np.ndarray)) and len(item) >= 2:
                volume = float(item[1]) if len(item) > 1 else 0
                total_volume += volume
            elif isinstance(item, (int, float)):
                total_volume += float(item)
    elif isinstance(data, (int, float)):
        total_volume = float(data)
    return total_volume

def calculate_volume_from_snaps(snap_list):
    """
    从snap_list计算成交量数据
    """
    total_volumes = []
    buy_volumes = []
    sell_volumes = []
    net_volumes = []
    
    for snap in snap_list:
        buy_trade_data = snap.get('buy_trade', [])
        sell_trade_data = snap.get('sell_trade', [])
        
        buy_vol = extract_volume_from_price_volume_data(buy_trade_data)
        sell_vol = extract_volume_from_price_volume_data(sell_trade_data)
        
        total_vol = buy_vol + sell_vol
        net_vol = buy_vol - sell_vol
        
        total_volumes.append(total_vol)
        buy_volumes.append(buy_vol)
        sell_volumes.append(sell_vol)
        net_volumes.append(net_vol)
    
    return (
        np.array(total_volumes),
        np.array(buy_volumes),
        np.array(sell_volumes),
        np.array(net_volumes)
    )

def remove_long_zero_segments(cumulative_volume, prices, max_zero_length=10, min_volume_threshold=0.1):
    """
    移除过长的0交易段（如午盘休市）
    """
    cumulative_volume = np.array(cumulative_volume)
    prices = np.array(prices)
    
    if len(cumulative_volume) < 2:
        return cumulative_volume, prices, []
    
    volume_changes = np.diff(cumulative_volume)
    
    zero_segments = []
    current_segment_start = None
    
    for i in range(len(volume_changes)):
        is_small_change = abs(volume_changes[i]) < min_volume_threshold
        
        if is_small_change:
            if current_segment_start is None:
                current_segment_start = i
        else:
            if current_segment_start is not None:
                segment_length = i - current_segment_start
                if segment_length > max_zero_length:
                    zero_segments.append((current_segment_start, i))
                current_segment_start = None
    
    if current_segment_start is not None:
        segment_length = len(volume_changes) - current_segment_start
        if segment_length > max_zero_length:
            zero_segments.append((current_segment_start, len(volume_changes)))
    
    if not zero_segments:
        return cumulative_volume, prices, []
    
    mask = np.ones(len(cumulative_volume), dtype=bool)
    
    for start, end in zero_segments:
        mask[start+1:end+1] = False
    
    filtered_cumulative_volume = cumulative_volume[mask]
    filtered_prices = prices[mask]
    
    return filtered_cumulative_volume, filtered_prices, zero_segments

def plot_price_analysis(snap_list, title="", figsize=(18, 8), 
                              save_path=None, smooth=False, window_size=7,
                              remove_zero_segments=True, max_zero_length=10,
                              min_volume_threshold=0.1):
    """
    简化版价格分析图：只保留两张图
    
    Parameters:
    -----------
    snap_list : list[dict]
        快照数据列表，包含 price_last, buy_trade, sell_trade 等字段
    title : str
        图表标题
    figsize : tuple
        图形大小
    save_path : str, optional
        保存路径
    smooth : bool, optional
        是否平滑曲线
    window_size : int, optional
        平滑窗口大小
    remove_zero_segments : bool, optional
        是否移除过长的0交易段（如午盘休市）
    max_zero_length : int, optional
        允许的最大连续0增长长度，超过此长度的段将被移除
    min_volume_threshold : float, optional
        最小成交量变化阈值，小于此值被认为是0变化
    """
    
    if not snap_list:
        print("空数据")
        return
    
    prices = []
    time_marks = []
    
    for snap in snap_list:
        price_data = snap.get('price_last', 0)
        
        if isinstance(price_data, (list, np.ndarray)):
            if len(price_data) > 0:
                price = float(price_data[-1])
            else:
                continue
        else:
            price = float(price_data) if price_data else 0
        
        if price == 0:
            continue
            
        time_mark = snap.get('time_mark', 0)
            
        prices.append(price)
        time_marks.append(time_mark)
    
    if len(prices) < 2:
        print("数据点不足")
        return
    
    prices = np.array(prices)
    
    total_volumes, buy_volumes, sell_volumes, net_volumes = calculate_volume_from_snaps(snap_list)
    
    cumulative_total_volume = np.cumsum(total_volumes)
    
    if smooth:
        prices_smooth = np.convolve(prices, np.ones(window_size)/window_size, mode='same')
        prices_smooth = np.convolve(prices_smooth, np.ones(3)/3, mode='same')
    else:
        prices_smooth = prices
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    x = range(len(prices))
    
    # 图1: 价格 vs 时间
    axes[0].plot(x, prices, 'b-', linewidth=1, alpha=0.5, label='price')
    if smooth:
        axes[0].plot(x, prices_smooth, 'b-', linewidth=2, label='smoothed price')
    axes[0].set_xlabel('time series')
    axes[0].set_ylabel('price')
    axes[0].set_title(f'{title} - price vs time series')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 检查数据长度一致性
    if len(cumulative_total_volume) != len(prices_smooth):
        min_len = min(len(cumulative_total_volume), len(prices_smooth))
        cumulative_total_volume = cumulative_total_volume[:min_len]
        prices_for_volume = prices_smooth[:min_len]
    else:
        prices_for_volume = prices_smooth
    
    # 处理第二张图：价格 vs 累积总成交量
    if remove_zero_segments:
        filtered_cumulative_volume, filtered_prices, removed_segments = remove_long_zero_segments(
            cumulative_total_volume, prices_for_volume, 
            max_zero_length=max_zero_length,
            min_volume_threshold=min_volume_threshold
        )
        
        plot_data_volume = filtered_cumulative_volume
        plot_data_prices = filtered_prices
    else:
        removed_segments = []
        plot_data_volume = cumulative_total_volume
        plot_data_prices = prices_for_volume
    
    # 图2: 价格 vs 累积总成交量
    axes[1].plot(plot_data_volume, plot_data_prices, 'g-', linewidth=2)
    axes[1].set_xlabel('cumulative total volume')
    axes[1].set_ylabel('price')
    axes[1].set_title(f'{title} - price vs cumulative total volume')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    print("简化版价格分析绘图工具")
    print("只保留两张图：价格 vs 时间，价格 vs 累积总成交量")