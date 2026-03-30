import sys
import os
from datetime import datetime, timedelta

sys.path.append('/home/jovyan/work/base_demo')
import base_tool
import pandas as pd


def backtest_multi_days(instrument_id, start_ymd, end_ymd, strategy_class, param_dict):
    """
    多天回测函数
    
    Args:
        instrument_id: 合约ID，如 '511520'
        start_ymd: 开始日期，如 '20260319'
        end_ymd: 结束日期，如 '20260325'
        strategy_class: 策略类，如 StrategyDemo
        param_dict: 策略参数字典
    
    Returns:
        多天回测结果DataFrame
    """
    start_date = datetime.strptime(start_ymd, '%Y%m%d')
    end_date = datetime.strptime(end_ymd, '%Y%m%d')
    
    all_results = []
    
    current_date = start_date
    while current_date <= end_date:
        trade_ymd = current_date.strftime('%Y%m%d')
        
        try:
            snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
            
            if not snap_list or len(snap_list) == 0:
                print(f"日期 {trade_ymd} 无数据，跳过")
                current_date += timedelta(days=1)
                continue
            
            strategy_name = param_dict.get('name', 'strategy')
            
            data_file = f'/home/jovyan/work/backtest_result/{instrument_id}_{trade_ymd}_{strategy_name}.pkl'
            if os.path.exists(data_file):
                os.remove(data_file)
            
            strategy = strategy_class(param_dict)
            
            position_dict = {}
            for snap in snap_list:
                strategy.on_snap(snap)
                position_dict[snap['time_mark']] = strategy.position_last
            
            profit = base_tool.backtest_quick(instrument_id, trade_ymd, strategy_name, position_dict)
            
            if profit is not None and len(profit) > 0:
                day_summary = profit.iloc[[-1]].copy()
                day_summary['trade_ymd'] = trade_ymd
                all_results.append(day_summary)
                print(f"日期 {trade_ymd} 回测完成，当日盈亏: {day_summary['profits'].values[0]:.2f}")
            else:
                print(f"日期 {trade_ymd} 无交易记录")
                
        except (Exception, SystemExit) as e:
            print(f"日期 {trade_ymd} 处理失败: {e}，跳过")
        
        current_date += timedelta(days=1)
    
    if not all_results:
        print("所有日期均无有效数据")
        return None
    
    result_df = pd.concat(all_results, ignore_index=True)
    cols = ['trade_ymd'] + [c for c in result_df.columns if c != 'trade_ymd']
    result_df = result_df[cols]

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
    
    # --- 修正点 1: 累计盈亏应该是求和，而不是取最后一个值 ---
    # 假设 daily_df['profits'] 存储的是每天的独立盈亏
    total_profits = daily_df['profits'].sum()
    
    max_profit = daily_df['profits'].max()
    min_profit = daily_df['profits'].min()
    
    win_days = (daily_df['profits'] > 0).sum()
    loss_days = (daily_df['profits'] < 0).sum()
    # 平本天数
    flat_days = total_days - win_days - loss_days
    
    win_rate = win_days / total_days * 100 if total_days > 0 else 0
    
    # --- 修正点 2: 日均盈亏直接求均值，不需要 diff ---
    # 因为传入的数据本身就是每日的切片盈亏，不是累计曲线
    avg_profit = daily_df['profits'].mean()
    
    # 计算盈亏比 (可选优化)
    avg_win = daily_df[daily_df['profits'] > 0]['profits'].mean() if win_days > 0 else 0
    avg_loss = abs(daily_df[daily_df['profits'] < 0]['profits'].mean()) if loss_days > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss != 0 else 0

    summary = {
        '交易天数': total_days,
        '累计盈亏': round(total_profits, 2),
        '最大单日盈利': round(max_profit, 2),
        '最大单日亏损': round(min_profit, 2),
        '盈利天数': int(win_days),
        '亏损天数': int(loss_days),
        '平盘天数': int(flat_days),
        '胜率(%)': round(win_rate, 2),
        '日均盈亏': round(avg_profit, 2) if not pd.isna(avg_profit) else 0,
        '盈亏比': round(profit_factor, 2) 
    }
    
    return summary