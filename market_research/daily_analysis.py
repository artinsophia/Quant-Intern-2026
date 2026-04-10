import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


class DailyMarketAnalyzer:
    """
    每日市场数据分析器
    统计每天的波动、涨跌、成交量等指标
    """

    def __init__(self, data_dir="/home/jovyan/work/backtest_result"):
        self.data_dir = Path(data_dir)

    def analyze_single_day(
        self, instrument_id, trade_ymd, strategy_name="delta_v1_simple"
    ):
        """
        分析单日市场数据

        Parameters:
        -----------
        instrument_id : str
            标的代码，如 '511520', '511090', '518880'
        trade_ymd : str
            交易日期，格式 'YYYYMMDD'
        strategy_name : str
            策略名称，用于查找对应的回测结果文件

        Returns:
        --------
        dict : 包含每日统计指标的字典
        """
        # 查找对应的回测结果文件
        pattern = f"{instrument_id}_{trade_ymd}_{strategy_name}*.pkl"
        matching_files = list(self.data_dir.glob(pattern))

        if not matching_files:
            # 尝试其他可能的策略名称
            patterns = [
                f"{instrument_id}_{trade_ymd}_*.pkl",
                f"{instrument_id}_{trade_ymd}*.pkl",
            ]
            for pattern in patterns:
                matching_files = list(self.data_dir.glob(pattern))
                if matching_files:
                    break

        if not matching_files:
            print(f"未找到 {instrument_id} 在 {trade_ymd} 的数据文件")
            return None

        # 使用第一个匹配的文件
        file_path = matching_files[0]

        try:
            # 加载数据
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # 检查数据类型
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict) and "time_mark" in data:
                # 如果是字典，转换为DataFrame
                df = pd.DataFrame(data)
            else:
                print(f"未知的数据格式: {type(data)}")
                return None

            # 确保有时间戳列
            if "time_mark" not in df.columns:
                print("数据中缺少 'time_mark' 列")
                return None

            # 转换为datetime
            df["datetime"] = pd.to_datetime(df["time_mark"], unit="ms")
            df.set_index("datetime", inplace=True)

            # 计算基本统计指标
            stats = self._calculate_daily_stats(df, instrument_id, trade_ymd)

            return stats

        except Exception as e:
            print(f"分析数据时出错: {e}")
            return None

    def _calculate_daily_stats(self, df, instrument_id, trade_ymd):
        """
        计算每日统计指标
        """
        stats = {
            "instrument_id": instrument_id,
            "trade_date": trade_ymd,
            "data_points": len(df),
            "time_period": f"{df.index[0].strftime('%H:%M:%S')} - {df.index[-1].strftime('%H:%M:%S')}",
        }

        # 价格相关指标
        if "price_last" in df.columns:
            price_series = df["price_last"]

            # 基本价格统计
            stats["open_price"] = float(price_series.iloc[0])
            stats["close_price"] = float(price_series.iloc[-1])
            stats["high_price"] = float(price_series.max())
            stats["low_price"] = float(price_series.min())
            stats["avg_price"] = float(price_series.mean())

            # 涨跌幅
            stats["price_change"] = stats["close_price"] - stats["open_price"]
            stats["price_change_pct"] = (
                (stats["price_change"] / stats["open_price"] * 100)
                if stats["open_price"] != 0
                else 0
            )

            # 波动率指标
            stats["price_std"] = float(price_series.std())
            stats["price_range"] = stats["high_price"] - stats["low_price"]
            stats["price_range_pct"] = (
                (stats["price_range"] / stats["open_price"] * 100)
                if stats["open_price"] != 0
                else 0
            )

            # 日内波动
            returns = price_series.pct_change().dropna()
            if len(returns) > 0:
                stats["intraday_volatility"] = float(
                    returns.std() * np.sqrt(252)
                )  # 年化波动率
                stats["max_intraday_drawdown"] = float(
                    (price_series / price_series.cummax() - 1).min() * 100
                )

        # 成交量相关指标（如果可用）
        volume_columns = ["volume", "num_trades", "buy_volume", "sell_volume"]
        for col in volume_columns:
            if col in df.columns:
                stats[f"total_{col}"] = float(df[col].sum())
                stats[f"avg_{col}"] = float(df[col].mean())

        # 买卖价差指标（如果可用）
        if all(col in df.columns for col in ["bid_1", "ask_1"]):
            df["spread"] = df["ask_1"] - df["bid_1"]
            df["spread_bps"] = (df["spread"] / df["price_last"]) * 10000  # 基点

            stats["avg_spread"] = float(df["spread"].mean())
            stats["avg_spread_bps"] = float(df["spread_bps"].mean())
            stats["max_spread"] = float(df["spread"].max())
            stats["min_spread"] = float(df["spread"].min())

        # 策略表现指标（如果可用）
        if "profits" in df.columns:
            profits = df["profits"]
            stats["final_profit"] = float(profits.iloc[-1])
            stats["max_profit"] = float(profits.max())
            stats["min_profit"] = float(profits.min())
            stats["profit_range"] = stats["max_profit"] - stats["min_profit"]

            # 计算最大回撤
            cumulative_max = profits.cummax()
            drawdown = profits - cumulative_max
            stats["max_drawdown"] = float(drawdown.min())

        if "position" in df.columns:
            position = df["position"]
            stats["position_changes"] = int((position.diff().abs() > 0).sum())
            stats["long_periods"] = int((position > 0).sum())
            stats["short_periods"] = int((position < 0).sum())
            stats["neutral_periods"] = int((position == 0).sum())

        return stats

    def analyze_multiple_days(
        self, instrument_id, start_ymd, end_ymd, strategy_name="delta_v1_simple"
    ):
        """
        分析多日市场数据

        Returns:
        --------
        pd.DataFrame : 包含每日统计指标的DataFrame
        """
        start_date = datetime.strptime(start_ymd, "%Y%m%d")
        end_date = datetime.strptime(end_ymd, "%Y%m%d")

        all_stats = []
        current_date = start_date

        while current_date <= end_date:
            trade_ymd = current_date.strftime("%Y%m%d")
            print(f"分析 {instrument_id} - {trade_ymd}...")

            stats = self.analyze_single_day(instrument_id, trade_ymd, strategy_name)
            if stats:
                all_stats.append(stats)

            current_date += timedelta(days=1)

        if not all_stats:
            return pd.DataFrame()

        # 转换为DataFrame
        df_stats = pd.DataFrame(all_stats)

        # 按日期排序
        if "trade_date" in df_stats.columns:
            df_stats["trade_date_dt"] = pd.to_datetime(
                df_stats["trade_date"], format="%Y%m%d"
            )
            df_stats = df_stats.sort_values("trade_date_dt")
            df_stats = df_stats.drop("trade_date_dt", axis=1)

        return df_stats

    def generate_summary_report(self, df_stats):
        """
        生成汇总报告
        """
        if df_stats.empty:
            return "无数据可用"

        report = []
        report.append("=" * 60)
        report.append("市场数据分析报告")
        report.append("=" * 60)

        # 总体统计
        report.append(
            f"\n分析时间段: {df_stats['trade_date'].iloc[0]} 至 {df_stats['trade_date'].iloc[-1]}"
        )
        report.append(f"标的数量: {df_stats['instrument_id'].nunique()}")
        report.append(f"总交易日数: {len(df_stats)}")

        # 价格表现汇总
        if "price_change_pct" in df_stats.columns:
            report.append("\n价格表现汇总:")
            report.append(f"  平均日涨跌幅: {df_stats['price_change_pct'].mean():.2f}%")
            report.append(
                f"  日涨跌幅标准差: {df_stats['price_change_pct'].std():.2f}%"
            )
            report.append(f"  最大单日涨幅: {df_stats['price_change_pct'].max():.2f}%")
            report.append(f"  最大单日跌幅: {df_stats['price_change_pct'].min():.2f}%")
            report.append(f"  上涨天数: {(df_stats['price_change_pct'] > 0).sum()}")
            report.append(f"  下跌天数: {(df_stats['price_change_pct'] < 0).sum()}")
            report.append(f"  平盘天数: {(df_stats['price_change_pct'] == 0).sum()}")

        # 波动率汇总
        if "price_range_pct" in df_stats.columns:
            report.append("\n波动率汇总:")
            report.append(f"  平均日内波幅: {df_stats['price_range_pct'].mean():.2f}%")
            report.append(f"  最大日内波幅: {df_stats['price_range_pct'].max():.2f}%")
            report.append(f"  最小日内波幅: {df_stats['price_range_pct'].min():.2f}%")

        if "intraday_volatility" in df_stats.columns:
            report.append(
                f"  平均日内波动率(年化): {df_stats['intraday_volatility'].mean():.2f}%"
            )

        # 成交量汇总
        volume_cols = [
            col
            for col in df_stats.columns
            if "total_volume" in col or "total_num_trades" in col
        ]
        for col in volume_cols:
            if col in df_stats.columns:
                col_name = col.replace("total_", "").replace("_", " ").title()
                report.append(f"\n{col_name}汇总:")
                report.append(f"  平均每日: {df_stats[col].mean():.0f}")
                report.append(f"  最大值: {df_stats[col].max():.0f}")
                report.append(f"  最小值: {df_stats[col].min():.0f}")

        # 策略表现汇总（如果可用）
        if "final_profit" in df_stats.columns:
            report.append("\n策略表现汇总:")
            report.append(f"  总盈利天数: {(df_stats['final_profit'] > 0).sum()}")
            report.append(f"  总亏损天数: {(df_stats['final_profit'] < 0).sum()}")
            report.append(f"  平均日盈利: {df_stats['final_profit'].mean():.2f}")
            report.append(f"  累计总盈利: {df_stats['final_profit'].sum():.2f}")
            report.append(f"  最大单日盈利: {df_stats['final_profit'].max():.2f}")
            report.append(f"  最大单日亏损: {df_stats['final_profit'].min():.2f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def plot_daily_metrics(self, df_stats, save_path=None):
        """
        绘制每日指标图表
        """
        if df_stats.empty:
            print("无数据可绘制")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(
            f"市场数据每日指标分析 - {df_stats['instrument_id'].iloc[0]}", fontsize=16
        )

        # 1. 日涨跌幅
        if "price_change_pct" in df_stats.columns:
            ax = axes[0, 0]
            ax.bar(range(len(df_stats)), df_stats["price_change_pct"])
            ax.axhline(y=0, color="r", linestyle="-", alpha=0.3)
            ax.set_title("日涨跌幅 (%)")
            ax.set_xlabel("交易日")
            ax.set_ylabel("涨跌幅 (%)")
            ax.grid(True, alpha=0.3)

        # 2. 日内波幅
        if "price_range_pct" in df_stats.columns:
            ax = axes[0, 1]
            ax.bar(range(len(df_stats)), df_stats["price_range_pct"], color="orange")
            ax.set_title("日内价格波幅 (%)")
            ax.set_xlabel("交易日")
            ax.set_ylabel("波幅 (%)")
            ax.grid(True, alpha=0.3)

        # 3. 成交量
        volume_cols = [
            col
            for col in df_stats.columns
            if "total_volume" in col or "total_num_trades" in col
        ]
        if volume_cols:
            ax = axes[1, 0]
            for col in volume_cols[:2]:  # 最多显示两个成交量指标
                ax.plot(
                    range(len(df_stats)), df_stats[col], label=col.replace("total_", "")
                )
            ax.set_title("成交量")
            ax.set_xlabel("交易日")
            ax.set_ylabel("数量")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. 买卖价差
        if "avg_spread_bps" in df_stats.columns:
            ax = axes[1, 1]
            ax.plot(range(len(df_stats)), df_stats["avg_spread_bps"], color="green")
            ax.set_title("平均买卖价差 (bps)")
            ax.set_xlabel("交易日")
            ax.set_ylabel("价差 (bps)")
            ax.grid(True, alpha=0.3)

        # 5. 策略盈利
        if "final_profit" in df_stats.columns:
            ax = axes[2, 0]
            colors = ["green" if x > 0 else "red" for x in df_stats["final_profit"]]
            ax.bar(range(len(df_stats)), df_stats["final_profit"], color=colors)
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax.set_title("策略日盈利")
            ax.set_xlabel("交易日")
            ax.set_ylabel("盈利")
            ax.grid(True, alpha=0.3)

        # 6. 仓位变化
        if "position_changes" in df_stats.columns:
            ax = axes[2, 1]
            ax.bar(range(len(df_stats)), df_stats["position_changes"], color="purple")
            ax.set_title("每日仓位变化次数")
            ax.set_xlabel("交易日")
            ax.set_ylabel("变化次数")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"图表已保存至: {save_path}")

        plt.show()


def main():
    """
    主函数 - 示例用法
    """
    analyzer = DailyMarketAnalyzer()

    # 示例：分析单个交易日
    print("分析单个交易日...")
    stats = analyzer.analyze_single_day("518880", "20260105")
    if stats:
        print(f"单日分析结果:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # 示例：分析多日数据
    print("\n分析多日数据...")
    df_stats = analyzer.analyze_multiple_days("518880", "20260105", "20260110")

    if not df_stats.empty:
        # 生成报告
        report = analyzer.generate_summary_report(df_stats)
        print(report)

        # 保存结果
        output_file = "/home/jovyan/work/market_analysis_results.csv"
        df_stats.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n详细数据已保存至: {output_file}")

        # 绘制图表
        analyzer.plot_daily_metrics(df_stats)


if __name__ == "__main__":
    main()
