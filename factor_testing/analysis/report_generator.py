import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import warnings
from datetime import datetime
import io
from ..metrics.factor_metrics import FactorMetrics
from ..analysis.group_test import GroupTester


class ReportGenerator:
    """
    因子测试报告生成器
    生成可视化图表和文本报告
    """

    def __init__(
        self,
        factor_name: str,
        factor_data: pd.Series,
        forward_returns: pd.Series,
        prices: Optional[pd.DataFrame] = None,
    ):
        """
        初始化报告生成器

        Parameters
        ----------
        factor_name : str
            因子名称
        factor_data : pd.Series
            因子数据
        forward_returns : pd.Series
            未来收益数据
        prices : pd.DataFrame, optional
            价格数据
        """
        self.factor_name = factor_name
        self.factor_data = factor_data
        self.forward_returns = forward_returns
        self.prices = prices

        # 创建计算器
        self.metrics_calculator = FactorMetrics(factor_data, forward_returns, prices)
        self.group_tester = GroupTester(factor_data, forward_returns, prices)

        # 设置绘图样式
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def generate_factor_distribution_plot(
        self, figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        生成因子分布图

        Parameters
        ----------
        figsize : Tuple[int, int]
            图形大小

        Returns
        -------
        plt.Figure
            因子分布图
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. 因子值直方图
        factor_values = self.factor_data.dropna()
        axes[0, 0].hist(factor_values, bins=50, edgecolor="black", alpha=0.7)
        axes[0, 0].set_title(f"{self.factor_name} Distribution")
        axes[0, 0].set_xlabel("Factor Value")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(
            factor_values.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {factor_values.mean():.4f}",
        )
        axes[0, 0].legend()

        # 2. QQ图（正态性检验）
        from scipy import stats

        stats.probplot(factor_values, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("QQ Plot (Normal Test)")

        # 3. 时间序列均值
        if isinstance(self.factor_data.index, pd.MultiIndex):
            # 按日期计算均值
            dates = self.factor_data.index.get_level_values(0).unique()
            date_means = []
            for date in sorted(dates):
                date_mask = self.factor_data.index.get_level_values(0) == date
                date_mean = self.factor_data[date_mask].mean()
                date_means.append(date_mean)

            axes[1, 0].plot(sorted(dates), date_means, marker="o", markersize=3)
            axes[1, 0].set_title("Factor Mean Over Time")
            axes[1, 0].set_xlabel("Date")
            axes[1, 0].set_ylabel("Mean Factor Value")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. 缺失值统计
        if isinstance(self.factor_data.index, pd.MultiIndex):
            dates = self.factor_data.index.get_level_values(0).unique()
            missing_rates = []
            for date in sorted(dates):
                date_mask = self.factor_data.index.get_level_values(0) == date
                date_data = self.factor_data[date_mask]
                missing_rate = date_data.isna().mean()
                missing_rates.append(missing_rate)

            axes[1, 1].plot(
                sorted(dates), missing_rates, marker="o", markersize=3, color="orange"
            )
            axes[1, 1].set_title("Missing Rate Over Time")
            axes[1, 1].set_xlabel("Date")
            axes[1, 1].set_ylabel("Missing Rate")
            axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        return fig

    def generate_ic_analysis_plot(
        self,
        freq: str = "D",
        method: str = "pearson",
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """
        生成IC分析图

        Parameters
        ----------
        freq : str
            时间频率
        method : str
            IC计算方法
        figsize : Tuple[int, int]
            图形大小

        Returns
        -------
        plt.Figure
            IC分析图
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. IC时间序列
        ic_series = self.metrics_calculator.ic_calculator.calculate_ic_series(
            freq=freq, method=method
        )
        axes[0, 0].plot(ic_series.index, ic_series.values, marker="o", markersize=3)
        axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[0, 0].set_title(f"IC Time Series ({method})")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("IC")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 添加IC均值和标准差线
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        axes[0, 0].axhline(
            y=ic_mean,
            color="g",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {ic_mean:.4f}",
        )
        axes[0, 0].axhline(
            y=ic_mean + ic_std, color="g", linestyle=":", alpha=0.5, label=f"±1 STD"
        )
        axes[0, 0].axhline(y=ic_mean - ic_std, color="g", linestyle=":", alpha=0.5)
        axes[0, 0].legend()

        # 2. IC分布直方图
        axes[0, 1].hist(ic_series.dropna(), bins=30, edgecolor="black", alpha=0.7)
        axes[0, 1].set_title("IC Distribution")
        axes[0, 1].set_xlabel("IC")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].axvline(
            ic_mean, color="red", linestyle="--", label=f"Mean: {ic_mean:.4f}"
        )
        axes[0, 1].legend()

        # 3. IC衰减图
        ic_decay = self.metrics_calculator.ic_calculator.calculate_ic_decay(
            method=method
        )
        axes[1, 0].plot(ic_decay.index, ic_decay.values, marker="o", markersize=5)
        axes[1, 0].set_title("IC Decay")
        axes[1, 0].set_xlabel("Lag (Periods)")
        axes[1, 0].set_ylabel("IC")
        axes[1, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 累计IC图
        cum_ic = ic_series.cumsum()
        axes[1, 1].plot(cum_ic.index, cum_ic.values, marker="o", markersize=3)
        axes[1, 1].set_title("Cumulative IC")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Cumulative IC")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_group_performance_plot(
        self,
        n_groups: int = 5,
        method: str = "quantile",
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        生成分组表现图

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str
            分组方法
        figsize : Tuple[int, int]
            图形大小

        Returns
        -------
        plt.Figure
            分组表现图
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 获取分组收益数据
        group_returns_df = self.metrics_calculator.calculate_group_returns(
            n_groups=n_groups, method=method, long_short=True
        )

        if group_returns_df.empty:
            # 创建空图
            for ax in axes.flat:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            plt.tight_layout()
            return fig

        # 1. 分组累计收益
        cum_returns = (1 + group_returns_df).cumprod() - 1

        for i in range(n_groups):
            col = f"group_{i}"
            if col in cum_returns.columns:
                axes[0, 0].plot(
                    cum_returns.index, cum_returns[col], label=f"Group {i}", linewidth=2
                )

        if "long_short" in cum_returns.columns:
            axes[0, 0].plot(
                cum_returns.index,
                cum_returns["long_short"],
                label="Long-Short",
                linewidth=3,
                color="black",
                linestyle="--",
            )

        axes[0, 0].set_title("Cumulative Returns by Group")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Cumulative Return")
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 分组平均收益柱状图
        group_means = {}
        for i in range(n_groups):
            col = f"group_{i}"
            if col in group_returns_df.columns:
                group_means[f"Group {i}"] = (
                    group_returns_df[col].mean() * 100
                )  # 转换为百分比

        if group_means:
            groups = list(group_means.keys())
            means = list(group_means.values())

            bars = axes[0, 1].bar(
                groups, means, color=sns.color_palette("husl", len(groups))
            )
            axes[0, 1].set_title("Average Return by Group")
            axes[0, 1].set_xlabel("Group")
            axes[0, 1].set_ylabel("Average Return (%)")
            axes[0, 1].tick_params(axis="x", rotation=45)

            # 在柱子上添加数值
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                axes[0, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{mean:.2f}%",
                    ha="center",
                    va="bottom",
                )

        # 3. 分组收益箱线图
        group_data = []
        group_labels = []
        for i in range(n_groups):
            col = f"group_{i}"
            if col in group_returns_df.columns:
                group_data.append(
                    group_returns_df[col].dropna().values * 100
                )  # 转换为百分比
                group_labels.append(f"Group {i}")

        if group_data:
            bp = axes[1, 0].boxplot(group_data, labels=group_labels, patch_artist=True)
            axes[1, 0].set_title("Return Distribution by Group")
            axes[1, 0].set_xlabel("Group")
            axes[1, 0].set_ylabel("Return (%)")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)

            # 设置箱线图颜色
            colors = sns.color_palette("husl", len(group_data))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)

        # 4. 多空组合收益时间序列
        if "long_short" in group_returns_df.columns:
            ls_returns = group_returns_df["long_short"] * 100  # 转换为百分比
            axes[1, 1].plot(
                ls_returns.index, ls_returns.values, marker="o", markersize=3
            )
            axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.5)
            axes[1, 1].set_title("Long-Short Portfolio Returns")
            axes[1, 1].set_xlabel("Date")
            axes[1, 1].set_ylabel("Return (%)")
            axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

            # 添加均值和标准差线
            ls_mean = ls_returns.mean()
            ls_std = ls_returns.std()
            axes[1, 1].axhline(
                y=ls_mean,
                color="g",
                linestyle="--",
                alpha=0.7,
                label=f"Mean: {ls_mean:.2f}%",
            )
            axes[1, 1].axhline(y=ls_mean + ls_std, color="g", linestyle=":", alpha=0.5)
            axes[1, 1].axhline(y=ls_mean - ls_std, color="g", linestyle=":", alpha=0.5)
            axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def generate_turnover_analysis_plot(
        self,
        n_groups: int = 5,
        method: str = "quantile",
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        生成换手率分析图

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str
            分组方法
        figsize : Tuple[int, int]
            图形大小

        Returns
        -------
        plt.Figure
            换手率分析图
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 获取换手率数据
        turnover_results = self.group_tester.calculate_group_turnover(
            n_groups=n_groups, method=method
        )

        if not turnover_results:
            # 创建空图
            for ax in axes.flat:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            plt.tight_layout()
            return fig

        # 1. 分组平均换手率
        group_turnovers = {}
        for i in range(n_groups):
            key = f"group_{i}"
            if key in turnover_results:
                group_turnovers[f"Group {i}"] = (
                    turnover_results[key]["mean_turnover"] * 100
                )  # 转换为百分比

        if group_turnovers:
            groups = list(group_turnovers.keys())
            turnovers = list(group_turnovers.values())

            bars = axes[0, 0].bar(
                groups, turnovers, color=sns.color_palette("husl", len(groups))
            )
            axes[0, 0].set_title("Average Turnover by Group")
            axes[0, 0].set_xlabel("Group")
            axes[0, 0].set_ylabel("Turnover (%)")
            axes[0, 0].tick_params(axis="x", rotation=45)

            # 在柱子上添加数值
            for bar, turnover in zip(bars, turnovers):
                height = bar.get_height()
                axes[0, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{turnover:.1f}%",
                    ha="center",
                    va="bottom",
                )

        # 2. 换手率时间序列（多空组合）
        if "long_short" in turnover_results:
            ls_turnover = turnover_results["long_short"]["turnover_series"] * 100
            axes[0, 1].plot(
                ls_turnover.index, ls_turnover.values, marker="o", markersize=3
            )
            axes[0, 1].set_title("Long-Short Portfolio Turnover")
            axes[0, 1].set_xlabel("Date")
            axes[0, 1].set_ylabel("Turnover (%)")
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # 添加均值和标准差线
            ls_mean = ls_turnover.mean()
            ls_std = ls_turnover.std()
            axes[0, 1].axhline(
                y=ls_mean,
                color="g",
                linestyle="--",
                alpha=0.7,
                label=f"Mean: {ls_mean:.1f}%",
            )
            axes[0, 1].axhline(y=ls_mean + ls_std, color="g", linestyle=":", alpha=0.5)
            axes[0, 1].axhline(y=ls_mean - ls_std, color="g", linestyle=":", alpha=0.5)
            axes[0, 1].legend()

        # 3. 换手率分布
        if "long_short" in turnover_results:
            ls_turnover = turnover_results["long_short"]["turnover_series"] * 100
            axes[1, 0].hist(ls_turnover.dropna(), bins=30, edgecolor="black", alpha=0.7)
            axes[1, 0].set_title("Turnover Distribution (Long-Short)")
            axes[1, 0].set_xlabel("Turnover (%)")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].axvline(
                ls_mean, color="red", linestyle="--", label=f"Mean: {ls_mean:.1f}%"
            )
            axes[1, 0].legend()

        # 4. 换手率与收益的关系
        if (
            "long_short" in turnover_results
            and "long_short"
            in self.metrics_calculator.calculate_group_returns(
                n_groups=n_groups, method=method, long_short=True
            ).columns
        ):
            ls_turnover = turnover_results["long_short"]["turnover_series"]
            ls_returns = self.metrics_calculator.calculate_group_returns(
                n_groups=n_groups, method=method, long_short=True
            )["long_short"]

            # 对齐数据
            common_dates = ls_turnover.index.intersection(ls_returns.index)
            if len(common_dates) > 0:
                aligned_turnover = ls_turnover.loc[common_dates]
                aligned_returns = ls_returns.loc[common_dates]

                scatter = axes[1, 1].scatter(
                    aligned_turnover * 100, aligned_returns * 100, alpha=0.6, s=30
                )
                axes[1, 1].set_title("Turnover vs Return (Long-Short)")
                axes[1, 1].set_xlabel("Turnover (%)")
                axes[1, 1].set_ylabel("Return (%)")
                axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.5)
                axes[1, 1].axvline(
                    x=aligned_turnover.mean() * 100,
                    color="g",
                    linestyle="--",
                    alpha=0.5,
                )
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_summary_report(
        self,
        n_groups: int = 5,
        method: str = "quantile",
        freq: str = "D",
        ic_method: str = "pearson",
    ) -> str:
        """
        生成文本摘要报告

        Parameters
        ----------
        n_groups : int
            分组数量
        method : str
            分组方法
        freq : str
            计算频率
        ic_method : str
            IC计算方法

        Returns
        -------
        str
            文本报告
        """
        # 计算所有指标
        all_metrics = self.metrics_calculator.calculate_all_metrics(
            n_groups=n_groups, freq=freq, method=ic_method
        )

        # 运行分组测试
        group_test_results = self.group_tester.run_comprehensive_test(
            n_groups=n_groups, method=method
        )

        # 生成报告
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"FACTOR TEST REPORT: {self.factor_name}")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")

        # 1. 因子基本信息
        report_lines.append("1. FACTOR BASIC INFORMATION")
        report_lines.append("-" * 40)

        factor_stats = {
            "Observations": len(self.factor_data.dropna()),
            "Missing Rate": f"{self.factor_data.isna().mean() * 100:.2f}%",
            "Mean": f"{self.factor_data.mean():.6f}",
            "Std": f"{self.factor_data.std():.6f}",
            "Skewness": f"{self.factor_data.skew():.4f}",
            "Kurtosis": f"{self.factor_data.kurtosis():.4f}",
        }

        for key, value in factor_stats.items():
            report_lines.append(f"  {key:<20}: {value}")

        report_lines.append("")

        # 2. IC分析结果
        report_lines.append("2. IC ANALYSIS")
        report_lines.append("-" * 40)

        ic_stats = {
            "IC": f"{all_metrics.get('ic', np.nan):.4f}",
            "Rank IC": f"{all_metrics.get('rank_ic', np.nan):.4f}",
            "IC Mean": f"{all_metrics.get('ic_mean', np.nan):.4f}",
            "IC Std": f"{all_metrics.get('ic_std', np.nan):.4f}",
            "IR": f"{all_metrics.get('ir', np.nan):.4f}",
            "IC Positive Rate": f"{all_metrics.get('ic_positive_rate', np.nan) * 100:.1f}%",
            "IC t-stat": f"{all_metrics.get('ic_t_stat', np.nan):.2f}",
            "IC p-value": f"{all_metrics.get('ic_p_value', np.nan):.4f}",
        }

        for key, value in ic_stats.items():
            report_lines.append(f"  {key:<25}: {value}")

        report_lines.append("")

        # 3. 分组表现
        report_lines.append("3. GROUP PERFORMANCE")
        report_lines.append("-" * 40)

        if "group_performance" in group_test_results:
            perf = group_test_results["group_performance"]

            # 多空组合
            if "long_short" in perf:
                ls = perf["long_short"]
                report_lines.append("  Long-Short Portfolio:")
                report_lines.append(
                    f"    Mean Return: {ls.get('mean_return', np.nan) * 100:.2f}%"
                )
                report_lines.append(
                    f"    Std Return: {ls.get('std_return', np.nan) * 100:.2f}%"
                )
                report_lines.append(
                    f"    Sharpe Ratio: {ls.get('sharpe_ratio', np.nan):.3f}"
                )
                report_lines.append(
                    f"    Max Drawdown: {ls.get('max_drawdown', np.nan) * 100:.2f}%"
                )
                report_lines.append(
                    f"    Win Rate: {ls.get('win_rate', np.nan) * 100:.1f}%"
                )
                report_lines.append(
                    f"    Calmar Ratio: {ls.get('calmar_ratio', np.nan):.3f}"
                )

            # 分组单调性
            if "monotonicity" in group_test_results:
                mono = group_test_results["monotonicity"]
                report_lines.append(
                    f"  Monotonicity (Spearman): {mono.get('spearman_corr', np.nan):.4f}"
                )

        report_lines.append("")

        # 4. 换手率分析
        report_lines.append("4. TURNOVER ANALYSIS")
        report_lines.append("-" * 40)

        if (
            "turnover" in group_test_results
            and "long_short" in group_test_results["turnover"]
        ):
            turnover = group_test_results["turnover"]["long_short"]
            report_lines.append(
                f"  Average Turnover: {turnover.get('mean_turnover', np.nan) * 100:.2f}%"
            )
            report_lines.append(
                f"  Turnover Std: {turnover.get('std_turnover', np.nan) * 100:.2f}%"
            )
            report_lines.append(
                f"  Max Turnover: {turnover.get('max_turnover', np.nan) * 100:.2f}%"
            )

        report_lines.append("")

        # 5. 衰减分析
        report_lines.append("5. DECAY ANALYSIS")
        report_lines.append("-" * 40)

        decay_stats = {
            "Decay Half-Life": f"{all_metrics.get('decay_half_life', np.nan):.1f} periods",
            "Decay Rate": f"{all_metrics.get('decay_rate', np.nan):.4f}",
        }

        for key, value in decay_stats.items():
            report_lines.append(f"  {key:<25}: {value}")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def save_report(
        self,
        output_dir: str,
        n_groups: int = 5,
        method: str = "quantile",
        freq: str = "D",
        ic_method: str = "pearson",
    ):
        """
        保存完整报告（图表+文本）

        Parameters
        ----------
        output_dir : str
            输出目录
        n_groups : int
            分组数量
        method : str
            分组方法
        freq : str
            计算频率
        ic_method : str
            IC计算方法
        """
        import os

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成并保存图表
        figs = []
        fig_names = []

        # 1. 因子分布图
        fig1 = self.generate_factor_distribution_plot()
        figs.append(fig1)
        fig_names.append("factor_distribution.png")

        # 2. IC分析图
        fig2 = self.generate_ic_analysis_plot(freq=freq, method=ic_method)
        figs.append(fig2)
        fig_names.append("ic_analysis.png")

        # 3. 分组表现图
        fig3 = self.generate_group_performance_plot(n_groups=n_groups, method=method)
        figs.append(fig3)
        fig_names.append("group_performance.png")

        # 4. 换手率分析图
        fig4 = self.generate_turnover_analysis_plot(n_groups=n_groups, method=method)
        figs.append(fig4)
        fig_names.append("turnover_analysis.png")

        # 保存图表
        for fig, name in zip(figs, fig_names):
            fig_path = os.path.join(output_dir, name)
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        # 生成并保存文本报告
        report_text = self.generate_summary_report(
            n_groups=n_groups, method=method, freq=freq, ic_method=ic_method
        )

        report_path = os.path.join(output_dir, "factor_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 保存数据到CSV
        data_path = os.path.join(output_dir, "factor_data.csv")
        self.factor_data.to_csv(data_path)

        returns_path = os.path.join(output_dir, "forward_returns.csv")
        self.forward_returns.to_csv(returns_path)

        print(f"Report saved to: {output_dir}")
        print(f"  - Charts: {', '.join(fig_names)}")
        print(f"  - Text report: factor_report.txt")
        print(f"  - Data: factor_data.csv, forward_returns.csv")
