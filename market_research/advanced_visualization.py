import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class MarketVisualizer:
    """
    高级市场数据可视化工具
    """

    def __init__(self, style="seaborn"):
        """
        初始化可视化工具

        Parameters:
        -----------
        style : str
            图表样式，可选 'seaborn', 'ggplot', 'classic', 'dark_background'
        """
        plt.style.use(style)
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def plot_comprehensive_dashboard(self, df_stats, title=None, save_path=None):
        """
        绘制综合仪表板

        Parameters:
        -----------
        df_stats : pd.DataFrame
            包含每日统计指标的DataFrame
        title : str, optional
            图表标题
        save_path : str, optional
            保存路径
        """
        if df_stats.empty:
            print("无数据可绘制")
            return

        if title is None:
            title = f"市场数据分析仪表板 - {df_stats['instrument_id'].iloc[0]}"

        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)

        # 1. 价格走势图
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_price_trend(ax1, df_stats)

        # 2. 涨跌幅分布图
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_returns_distribution(ax2, df_stats)

        # 3. 波动率热力图
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_volatility_heatmap(ax3, df_stats)

        # 4. 成交量分析
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_volume_analysis(ax4, df_stats)

        # 5. 相关性矩阵
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_correlation_matrix(ax5, df_stats)

        # 6. 策略表现分析
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_strategy_performance(ax6, df_stats)

        # 7. 时间序列分解
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_time_series_decomposition(ax7, df_stats)

        plt.suptitle(title, fontsize=20, y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"仪表板已保存至: {save_path}")

        plt.show()

    def _plot_price_trend(self, ax, df_stats):
        """绘制价格趋势图"""
        if "close_price" not in df_stats.columns:
            return

        dates = pd.to_datetime(df_stats["trade_date"], format="%Y%m%d")

        ax.plot(
            dates,
            df_stats["close_price"],
            "o-",
            linewidth=2,
            markersize=6,
            color=self.colors[0],
            label="收盘价",
        )

        if "open_price" in df_stats.columns:
            ax.plot(
                dates,
                df_stats["open_price"],
                "s--",
                linewidth=1,
                markersize=4,
                color=self.colors[1],
                alpha=0.7,
                label="开盘价",
            )

        ax.fill_between(
            dates,
            df_stats["low_price"]
            if "low_price" in df_stats.columns
            else df_stats["close_price"],
            df_stats["high_price"]
            if "high_price" in df_stats.columns
            else df_stats["close_price"],
            alpha=0.2,
            color=self.colors[0],
        )

        ax.set_title("价格走势", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期")
        ax.set_ylabel("价格")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_returns_distribution(self, ax, df_stats):
        """绘制涨跌幅分布图"""
        if "price_change_pct" not in df_stats.columns:
            return

        returns = df_stats["price_change_pct"]

        # 直方图
        n, bins, patches = ax.hist(
            returns,
            bins=15,
            alpha=0.7,
            color=self.colors[1],
            edgecolor="black",
            linewidth=0.5,
        )

        # 添加正态分布曲线
        from scipy.stats import norm

        mu, std = norm.fit(returns)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(
            x,
            p * len(returns) * (bins[1] - bins[0]),
            "r-",
            linewidth=2,
            label=f"正态分布\nμ={mu:.2f}%, σ={std:.2f}%",
        )

        # 添加统计信息
        stats_text = f"""
        均值: {returns.mean():.2f}%
        标准差: {returns.std():.2f}%
        偏度: {returns.skew():.2f}
        峰度: {returns.kurtosis():.2f}
        正收益天数: {(returns > 0).sum()}
        负收益天数: {(returns < 0).sum()}
        """

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_title("涨跌幅分布", fontsize=14, fontweight="bold")
        ax.set_xlabel("涨跌幅 (%)")
        ax.set_ylabel("频数")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    def _plot_volatility_heatmap(self, ax, df_stats):
        """绘制波动率热力图"""
        volatility_metrics = []

        # 收集波动率相关指标
        for col in ["price_range_pct", "price_std", "intraday_volatility"]:
            if col in df_stats.columns:
                volatility_metrics.append(col)

        if not volatility_metrics:
            return

        # 创建热力图数据
        heatmap_data = df_stats[volatility_metrics].T

        # 标准化数据
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(heatmap_data.T).T

        im = ax.imshow(normalized_data, aspect="auto", cmap="YlOrRd")

        # 设置坐标轴
        ax.set_xticks(range(len(df_stats)))
        ax.set_xticklabels([d[-4:] for d in df_stats["trade_date"]], rotation=45)
        ax.set_yticks(range(len(volatility_metrics)))
        ax.set_yticklabels([self._get_metric_name(m) for m in volatility_metrics])

        # 添加数值标签
        for i in range(len(volatility_metrics)):
            for j in range(len(df_stats)):
                value = heatmap_data.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if normalized_data[i, j] > 0.5 else "black",
                    fontsize=8,
                )

        ax.set_title("波动率热力图", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="标准化值")

    def _plot_volume_analysis(self, ax, df_stats):
        """绘制成交量分析图"""
        volume_cols = [
            col
            for col in df_stats.columns
            if "total_volume" in col or "total_num_trades" in col
        ]

        if not volume_cols:
            return

        dates = pd.to_datetime(df_stats["trade_date"], format="%Y%m%d")

        # 绘制成交量
        for i, col in enumerate(volume_cols[:2]):  # 最多显示两个成交量指标
            ax.plot(
                dates,
                df_stats[col],
                "o-",
                linewidth=2,
                markersize=4,
                color=self.colors[i],
                label=self._get_metric_name(col),
            )

        ax.set_title("成交量分析", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期")
        ax.set_ylabel("数量")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # 添加第二y轴显示价格（如果可用）
        if "close_price" in df_stats.columns:
            ax2 = ax.twinx()
            ax2.plot(
                dates,
                df_stats["close_price"],
                "s--",
                linewidth=1,
                markersize=3,
                color=self.colors[2],
                alpha=0.7,
                label="收盘价",
            )
            ax2.set_ylabel("价格", color=self.colors[2])
            ax2.tick_params(axis="y", labelcolor=self.colors[2])

            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_correlation_matrix(self, ax, df_stats):
        """绘制相关性矩阵"""
        # 选择数值型列
        numeric_cols = df_stats.select_dtypes(include=[np.number]).columns.tolist()

        # 移除可能不是指标的列
        exclude_cols = ["data_points", "time_period_num"]
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        if len(numeric_cols) < 2:
            return

        # 计算相关性矩阵
        corr_matrix = df_stats[numeric_cols].corr()

        # 绘制热力图
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        # 设置坐标轴
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(
            [self._get_metric_name(col) for col in numeric_cols],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_yticklabels(
            [self._get_metric_name(col) for col in numeric_cols], fontsize=8
        )

        # 添加数值标签
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                value = corr_matrix.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > 0.5 else "black",
                    fontsize=7,
                )

        ax.set_title("指标相关性矩阵", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="相关系数")

    def _plot_strategy_performance(self, ax, df_stats):
        """绘制策略表现分析图"""
        if "final_profit" not in df_stats.columns:
            return

        dates = pd.to_datetime(df_stats["trade_date"], format="%Y%m%d")
        profits = df_stats["final_profit"]

        # 创建子图
        ax2 = ax.twinx()

        # 绘制盈利柱状图
        colors = ["green" if p > 0 else "red" for p in profits]
        bars = ax.bar(
            range(len(profits)), profits, color=colors, alpha=0.7, label="日盈利"
        )

        # 绘制累计盈利曲线
        cumulative_profit = profits.cumsum()
        ax2.plot(
            range(len(profits)),
            cumulative_profit,
            "o-",
            linewidth=2,
            markersize=6,
            color=self.colors[2],
            label="累计盈利",
        )

        ax.set_title("策略表现分析", fontsize=14, fontweight="bold")
        ax.set_xlabel("交易日")
        ax.set_ylabel("日盈利", color=self.colors[0])
        ax.tick_params(axis="y", labelcolor=self.colors[0])

        ax2.set_ylabel("累计盈利", color=self.colors[2])
        ax2.tick_params(axis="y", labelcolor=self.colors[2])

        # 添加零线
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # 添加统计信息
        total_profit = profits.sum()
        win_rate = (profits > 0).sum() / len(profits) * 100
        avg_win = profits[profits > 0].mean() if (profits > 0).any() else 0
        avg_loss = profits[profits < 0].mean() if (profits < 0).any() else 0

        stats_text = f"""
        总盈利: {total_profit:.2f}
        胜率: {win_rate:.1f}%
        平均盈利: {avg_win:.2f}
        平均亏损: {avg_loss:.2f}
        盈亏比: {abs(avg_win / avg_loss):.2f if avg_loss != 0 else 'N/A'}
        """

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        ax.grid(True, alpha=0.3)

    def _plot_time_series_decomposition(self, ax, df_stats):
        """绘制时间序列分解图"""
        if "close_price" not in df_stats.columns or len(df_stats) < 5:
            return

        from statsmodels.tsa.seasonal import seasonal_decompose

        try:
            # 创建时间序列
            dates = pd.to_datetime(df_stats["trade_date"], format="%Y%m%d")
            ts = pd.Series(df_stats["close_price"].values, index=dates)

            # 进行季节性分解（假设没有季节性，使用加法模型）
            result = seasonal_decompose(ts, model="additive", period=1)

            # 绘制分解结果
            ax.plot(
                result.observed,
                "o-",
                linewidth=2,
                markersize=4,
                color=self.colors[0],
                label="观测值",
            )
            ax.plot(
                result.trend,
                "s-",
                linewidth=2,
                markersize=4,
                color=self.colors[1],
                label="趋势",
            )
            ax.plot(
                result.resid,
                "^-",
                linewidth=1,
                markersize=3,
                color=self.colors[2],
                alpha=0.7,
                label="残差",
            )

            ax.set_title("时间序列分解", fontsize=14, fontweight="bold")
            ax.set_xlabel("日期")
            ax.set_ylabel("价格")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

            # 格式化x轴日期
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=max(1, len(dates) // 10))
            )
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        except Exception as e:
            print(f"时间序列分解失败: {e}")
            # 回退到简单的时间序列图
            ax.plot(
                dates,
                df_stats["close_price"],
                "o-",
                linewidth=2,
                markersize=6,
                color=self.colors[0],
            )
            ax.set_title("价格时间序列", fontsize=14, fontweight="bold")
            ax.set_xlabel("日期")
            ax.set_ylabel("价格")
            ax.grid(True, alpha=0.3)

            # 格式化x轴日期
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=max(1, len(dates) // 10))
            )
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _get_metric_name(self, metric):
        """获取指标的中文名称"""
        metric_names = {
            "price_change_pct": "涨跌幅(%)",
            "price_range_pct": "日内波幅(%)",
            "price_std": "价格标准差",
            "intraday_volatility": "日内波动率(%)",
            "total_volume": "总成交量",
            "total_num_trades": "总成交笔数",
            "avg_spread_bps": "平均价差(bps)",
            "final_profit": "最终盈利",
            "close_price": "收盘价",
            "open_price": "开盘价",
            "high_price": "最高价",
            "low_price": "最低价",
            "avg_price": "平均价",
            "position_changes": "仓位变化",
            "max_drawdown": "最大回撤",
        }

        return metric_names.get(metric, metric.replace("_", " ").title())

    def plot_rolling_statistics(self, df_stats, window=5, save_path=None):
        """
        绘制滚动统计图表

        Parameters:
        -----------
        df_stats : pd.DataFrame
            包含每日统计指标的DataFrame
        window : int
            滚动窗口大小
        save_path : str, optional
            保存路径
        """
        if df_stats.empty or len(df_stats) < window:
            print("数据不足，无法计算滚动统计")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f"滚动统计 (窗口={window}天)", fontsize=16)

        dates = pd.to_datetime(df_stats["trade_date"], format="%Y%m%d")

        # 1. 滚动平均涨跌幅
        if "price_change_pct" in df_stats.columns:
            ax = axes[0, 0]
            rolling_mean = df_stats["price_change_pct"].rolling(window=window).mean()
            rolling_std = df_stats["price_change_pct"].rolling(window=window).std()

            ax.plot(
                dates, df_stats["price_change_pct"], "o-", alpha=0.5, label="日涨跌幅"
            )
            ax.plot(dates, rolling_mean, "r-", linewidth=2, label=f"{window}日移动平均")
            ax.fill_between(
                dates,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.2,
                color="red",
                label=f"±{window}日标准差",
            )

            ax.set_title("滚动平均涨跌幅")
            ax.set_xlabel("日期")
            ax.set_ylabel("涨跌幅 (%)")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # 2. 滚动波动率
        if "price_range_pct" in df_stats.columns:
            ax = axes[0, 1]
            rolling_vol = df_stats["price_range_pct"].rolling(window=window).mean()

            ax.plot(
                dates, df_stats["price_range_pct"], "o-", alpha=0.5, label="日内波幅"
            )
            ax.plot(dates, rolling_vol, "r-", linewidth=2, label=f"{window}日移动平均")

            ax.set_title("滚动平均波动率")
            ax.set_xlabel("日期")
            ax.set_ylabel("波幅 (%)")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # 3. 滚动成交量
        volume_cols = [col for col in df_stats.columns if "total_volume" in col]
        if volume_cols:
            ax = axes[1, 0]
            col = volume_cols[0]
            rolling_volume = df_stats[col].rolling(window=window).mean()

            ax.plot(dates, df_stats[col], "o-", alpha=0.5, label="日成交量")
            ax.plot(
                dates, rolling_volume, "r-", linewidth=2, label=f"{window}日移动平均"
            )

            ax.set_title("滚动平均成交量")
            ax.set_xlabel("日期")
            ax.set_ylabel("成交量")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # 4. 滚动相关性
        if all(col in df_stats.columns for col in ["price_change_pct", "total_volume"]):
            ax = axes[1, 1]

            # 计算滚动相关性
            rolling_corr = (
                df_stats["price_change_pct"]
                .rolling(window=window)
                .corr(
                    df_stats[volume_cols[0]]
                    if volume_cols
                    else df_stats["total_volume"]
                )
            )

            ax.plot(dates, rolling_corr, "o-", linewidth=2)
            ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)

            ax.set_title("价格与成交量滚动相关性")
            ax.set_xlabel("日期")
            ax.set_ylabel("相关系数")
            ax.grid(True, alpha=0.3)

        # 5. 滚动夏普比率（如果可用）
        if (
            "final_profit" in df_stats.columns
            and "price_change_pct" in df_stats.columns
        ):
            ax = axes[2, 0]

            # 简化夏普比率计算
            rolling_return = df_stats["price_change_pct"].rolling(window=window).mean()
            rolling_vol = df_stats["price_change_pct"].rolling(window=window).std()
            sharpe_ratio = (
                rolling_return / rolling_vol.replace(0, np.nan) * np.sqrt(252)
            )

            ax.plot(dates, sharpe_ratio, "o-", linewidth=2)
            ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            ax.axhline(y=1, color="g", linestyle="--", alpha=0.5, label="夏普=1")

            ax.set_title("滚动夏普比率")
            ax.set_xlabel("日期")
            ax.set_ylabel("夏普比率")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # 6. 滚动最大回撤
        if "final_profit" in df_stats.columns:
            ax = axes[2, 1]

            # 计算滚动最大回撤
            rolling_dd = []
            for i in range(len(df_stats)):
                if i < window:
                    rolling_dd.append(np.nan)
                else:
                    window_profits = df_stats["final_profit"].iloc[i - window : i]
                    cummax = window_profits.cummax()
                    drawdown = (
                        (window_profits - cummax) / cummax.replace(0, np.nan) * 100
                    )
                    rolling_dd.append(drawdown.min())

            ax.plot(dates, rolling_dd, "o-", linewidth=2)

            ax.set_title("滚动最大回撤")
            ax.set_xlabel("日期")
            ax.set_ylabel("最大回撤 (%)")
            ax.grid(True, alpha=0.3)

        # 格式化所有x轴
        for i in range(3):
            for j in range(2):
                ax = axes[i, j]
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                ax.xaxis.set_major_locator(
                    mdates.DayLocator(interval=max(1, len(dates) // 10))
                )
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"滚动统计图表已保存至: {save_path}")

        plt.show()


def main():
    """示例用法"""
    # 示例数据（实际使用时需要从文件加载）
    dates = pd.date_range("2026-01-05", periods=20, freq="D")
    np.random.seed(42)

    sample_data = {
        "instrument_id": ["518880"] * 20,
        "trade_date": [d.strftime("%Y%m%d") for d in dates],
        "close_price": 100 + np.cumsum(np.random.randn(20) * 2),
        "open_price": 100 + np.cumsum(np.random.randn(20) * 2) - np.random.rand(20),
        "high_price": 100
        + np.cumsum(np.random.randn(20) * 2)
        + np.abs(np.random.randn(20)),
        "low_price": 100
        + np.cumsum(np.random.randn(20) * 2)
        - np.abs(np.random.randn(20)),
        "price_change_pct": np.random.randn(20) * 1.5,
        "price_range_pct": np.abs(np.random.randn(20)) * 2,
        "total_volume": np.random.randint(1000000, 5000000, 20),
        "final_profit": np.cumsum(np.random.randn(20) * 100),
    }

    df_stats = pd.DataFrame(sample_data)

    # 计算衍生指标
    df_stats["price_std"] = df_stats["close_price"].rolling(5).std()
    df_stats["intraday_volatility"] = df_stats["price_range_pct"] * np.sqrt(252)
    df_stats["avg_price"] = (df_stats["high_price"] + df_stats["low_price"]) / 2

    # 创建可视化器
    visualizer = MarketVisualizer(style="seaborn")

    # 绘制综合仪表板
    print("生成综合仪表板...")
    visualizer.plot_comprehensive_dashboard(df_stats, title="示例市场数据分析")

    # 绘制滚动统计
    print("\n生成滚动统计图表...")
    visualizer.plot_rolling_statistics(df_stats, window=5)


if __name__ == "__main__":
    main()
