import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append('/home/jovyan/work/base_demo')
import base_tool


def single_day_backtest_visualization(instrument_id, trade_ymd, strategy, param_dict=None, 
                                      figsize=(16, 10), title_suffix="", save_path=None):
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
    
    strategy_name = param_dict.get('name', 'strategy')
    
    # 1. 加载数据
    try:
        snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
        if not snap_list or len(snap_list) == 0:
            print(f"日期 {trade_ymd} 无数据")
            return None
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None
    
    # 2. 运行策略，记录持仓变化
    position_dict = {}
    position_history = []  # 记录每个时间点的持仓
    price_history = []     # 记录每个时间点的价格
    time_history = []      # 记录每个时间点的时间戳
    time_str_history = []  # 记录每个时间点的时间字符串
    
    for snap in snap_list:
        strategy.on_snap(snap)
        time_mark = snap.get('time_mark', 0)
        position_last = strategy.position_last
        price_last = snap.get('price_last', 0)
        time_hms = snap.get('time_hms', '')
        
        position_dict[time_mark] = position_last
        position_history.append(position_last)
        price_history.append(price_last)
        time_history.append(time_mark)
        time_str_history.append(time_hms)
    
    # 3. 运行回测获取盈亏数据
    data_file = f'/home/jovyan/work/backtest_result/{instrument_id}_{trade_ymd}_{strategy_name}.pkl'
    if os.path.exists(data_file):
        os.remove(data_file)
    
    try:
        profit_df = base_tool.backtest_quick(instrument_id, trade_ymd, strategy_name, position_dict)
    except Exception as e:
        print(f"回测执行失败: {e}")
        profit_df = None
    
    # 4. 准备绘图数据
    price_history = np.array(price_history)
    position_history = np.array(position_history)
    
    # 找到持仓变化的点
    position_changes = np.diff(position_history, prepend=0)
    change_indices = np.where(position_changes != 0)[0]
    
    # 5. 创建图形
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # 主图：价格走势和持仓段
    ax1 = axes[0]
    
    # 绘制价格曲线
    x = range(len(price_history))
    ax1.plot(x, price_history, 'k-', linewidth=1.5, alpha=0.7, label='Price')
    
    # 标注持仓段
    current_position = 0
    segment_start = 0
    segment_color = 'gray'  # 默认无持仓颜色
    
    for i in range(len(position_history)):
        if i == 0:
            current_position = position_history[i]
            segment_start = i
            segment_color = get_position_color(current_position)
            continue
        
        if position_history[i] != current_position or i == len(position_history) - 1:
            # 绘制上一个持仓段
            segment_end = i if i == len(position_history) - 1 else i - 1
            if segment_end > segment_start:  # 确保有足够的点
                ax1.axvspan(segment_start, segment_end, alpha=0.2, color=segment_color, 
                           label=f'Position {current_position}' if segment_start == 0 else "")
            
            # 更新当前持仓
            current_position = position_history[i]
            segment_start = i
            segment_color = get_position_color(current_position)
    
    # 标注变仓节点
    for idx in change_indices:
        if idx < len(price_history):
            # 根据持仓变化方向选择标记
            change_type = position_history[idx] - (position_history[idx-1] if idx > 0 else 0)
            marker = get_change_marker(change_type)
            color = get_position_color(position_history[idx])
            
            ax1.plot(idx, price_history[idx], marker=marker, markersize=10, 
                    color=color, markeredgecolor='black', markeredgewidth=1,
                    label=f'{"Open" if change_type != 0 else "Close"} {position_history[idx]}' 
                    if idx == change_indices[0] else "")
    
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Price')
    ax1.set_title(f'{instrument_id} - {trade_ymd} {strategy_name} {title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # 副图：持仓变化
    ax2 = axes[1]
    
    # 绘制持仓线
    ax2.plot(x, position_history, 'b-', linewidth=2, alpha=0.8, label='Position')
    ax2.fill_between(x, 0, position_history, alpha=0.3, color='blue')
    
    # 标注变仓点
    for idx in change_indices:
        if idx < len(price_history):
            ax2.plot(idx, position_history[idx], 'ro', markersize=8, 
                    markeredgecolor='black', markeredgewidth=1)
    
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Position')
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short (-1)', 'Flat (0)', 'Long (+1)'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    # 6. 显示回测结果摘要
    if profit_df is not None and len(profit_df) > 0:
        # 添加文本摘要
        total_trades = len(change_indices) // 2  # 每个完整交易有2个变仓点
        final_pnl = profit_df['profits'].iloc[-1] if 'profits' in profit_df.columns else 0
        
        summary_text = f"Trades: {total_trades}, Final P&L: {final_pnl:.2f}"
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 7. 返回结果
    result = {
        'instrument_id': instrument_id,
        'trade_ymd': trade_ymd,
        'strategy_name': strategy_name,
        'snap_count': len(snap_list),
        'position_changes': len(change_indices),
        'price_history': price_history,
        'position_history': position_history,
        'time_history': time_history,
        'time_str_history': time_str_history,
        'change_indices': change_indices.tolist(),
        'profit_df': profit_df
    }
    
    return result


def get_position_color(position):
    """根据持仓状态返回颜色"""
    if position == 1:  # 多头
        return 'green'
    elif position == -1:  # 空头
        return 'red'
    else:  # 空仓
        return 'gray'


def get_change_marker(change_type):
    """根据持仓变化类型返回标记符号"""
    if change_type > 0:  # 开多
        return '^'
    elif change_type < 0:  # 开空
        return 'v'
    else:  # 平仓
        return 'o'


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
                exit_price = price_history[i-1] if price_history[i-1] > 0 else price_history[i]
                
                # 计算盈亏（简化版：价格差乘以持仓方向）
                if current_position == 1:  # 多头
                    pnl = exit_price - entry_price
                elif current_position == -1:  # 空头
                    pnl = entry_price - exit_price
                else:
                    pnl = 0
                
                segments.append({
                    'start_idx': segment_start,
                    'end_idx': i-1,
                    'position': current_position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'duration': i - segment_start
                })
            
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
        
        segments.append({
            'start_idx': segment_start,
            'end_idx': len(position_history) - 1,
            'position': current_position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'duration': len(position_history) - segment_start
        })
    
    return segments


def plot_enhanced_backtest(instrument_id, trade_ymd, strategy, param_dict=None, 
                          figsize=(18, 12), title_suffix="", save_path=None):
    """
    增强版单日回测可视化 - 包含持仓段盈亏分析
    
    功能：
    1. 运行单日回测并可视化
    2. 分析每个持仓段的盈亏
    3. 用颜色深浅表示盈亏大小
    4. 显示详细的交易统计
    
    Args:
        参数同 single_day_backtest_visualization
    
    Returns:
        dict: 包含详细分析结果的字典
    """
    
    # 运行基础回测可视化
    result = single_day_backtest_visualization(
        instrument_id, trade_ymd, strategy, param_dict, figsize, title_suffix, None
    )
    
    if result is None:
        return None
    
    # 分析持仓段
    segments = analyze_position_segments(
        result['position_history'], 
        result['price_history']
    )
    
    # 创建增强图形
    fig, axes = plt.subplots(3, 1, figsize=figsize, 
                            gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 图1：价格走势和持仓段（带盈亏颜色）
    ax1 = axes[0]
    price_history = result['price_history']
    position_history = result['position_history']
    x = range(len(price_history))
    
    # 绘制价格曲线
    ax1.plot(x, price_history, 'k-', linewidth=1.5, alpha=0.7, label='Price')
    
    # 绘制带盈亏颜色的持仓段
    for seg in segments:
        color = get_position_color(seg['position'])
        alpha = 0.2 + 0.3 * min(abs(seg['pnl']) / max(0.01, np.std(price_history)), 1.0)
        
        ax1.axvspan(seg['start_idx'], seg['end_idx'], alpha=alpha, color=color)
        
        # 标注盈亏
        mid_idx = (seg['start_idx'] + seg['end_idx']) // 2
        if seg['position'] != 0 and seg['end_idx'] - seg['start_idx'] > 10:
            pnl_text = f"{seg['pnl']:.3f}"
            ax1.text(mid_idx, price_history[mid_idx], pnl_text, 
                    fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # 标注变仓节点
    for idx in result['change_indices']:
        if idx < len(price_history):
            change_type = position_history[idx] - (position_history[idx-1] if idx > 0 else 0)
            marker = get_change_marker(change_type)
            color = get_position_color(position_history[idx])
            
            ax1.plot(idx, price_history[idx], marker=marker, markersize=10, 
                    color=color, markeredgecolor='black', markeredgewidth=1)
    
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Price')
    ax1.set_title(f'{instrument_id} - {trade_ymd} {result["strategy_name"]} {title_suffix}')
    ax1.grid(True, alpha=0.3)
    
    # 图2：持仓变化
    ax2 = axes[1]
    ax2.plot(x, position_history, 'b-', linewidth=2, alpha=0.8, label='Position')
    ax2.fill_between(x, 0, position_history, alpha=0.3, color='blue')
    
    for idx in result['change_indices']:
        if idx < len(price_history):
            ax2.plot(idx, position_history[idx], 'ro', markersize=8, 
                    markeredgecolor='black', markeredgewidth=1)
    
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Position')
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short (-1)', 'Flat (0)', 'Long (+1)'])
    ax2.grid(True, alpha=0.3)
    
    # 图3：累计盈亏（如果可用）
    ax3 = axes[2]
    
    if result['profit_df'] is not None and 'profits' in result['profit_df'].columns:
        profit_data = result['profit_df']['profits'].values
        cumulative_pnl = np.cumsum(profit_data)
        
        ax3.plot(range(len(cumulative_pnl)), cumulative_pnl, 'g-', linewidth=2, label='Cumulative P&L')
        ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                        where=(cumulative_pnl >= 0), alpha=0.3, color='green')
        ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                        where=(cumulative_pnl < 0), alpha=0.3, color='red')
        
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax3.set_xlabel('Trade Index')
        ax3.set_ylabel('Cumulative P&L')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        # 如果没有盈亏数据，显示持仓段统计
        if segments:
            segment_pnls = [seg['pnl'] for seg in segments]
            segment_positions = [seg['position'] for seg in segments]
            
            colors = ['green' if pnl >= 0 else 'red' for pnl in segment_pnls]
            ax3.bar(range(len(segments)), segment_pnls, color=colors, alpha=0.7)
            ax3.set_xlabel('Trade Segment')
            ax3.set_ylabel('Segment P&L')
            ax3.set_title('Individual Trade Segment P&L')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        else:
            ax3.text(0.5, 0.5, 'No trade segments', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_axis_off()
    
    plt.tight_layout()
    
    # 添加统计信息
    if segments:
        total_pnl = sum(seg['pnl'] for seg in segments)
        winning_trades = sum(1 for seg in segments if seg['pnl'] > 0)
        losing_trades = sum(1 for seg in segments if seg['pnl'] < 0)
        win_rate = winning_trades / len(segments) * 100 if segments else 0
        
        stats_text = (f"Total Trades: {len(segments)} | "
                     f"Win Rate: {win_rate:.1f}% | "
                     f"Total P&L: {total_pnl:.3f} | "
                     f"Avg P&L: {total_pnl/len(segments):.3f}")
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # 更新结果
    result['segments'] = segments
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
            price = snap.get('price_last', 0)
            if price > 100.5:
                self.position_last = 1
            elif price < 100.0:
                self.position_last = -1
            else:
                self.position_last = 0