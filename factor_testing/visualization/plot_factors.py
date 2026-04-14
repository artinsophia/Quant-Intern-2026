"""
因子可视化模块
绘制因子分析结果图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class FactorVisualizer:
    """因子可视化器"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def plot_ic_analysis(
        self, ic_results: pd.DataFrame, title: str = "因子IC分析"
    ) -> plt.Figure:
        """
        绘制IC分析结果

        Args:
            ic_results: IC分析结果DataFrame
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)

        # 1. 各因子IC值热力图
        if "period" in ic_results.columns and "factor" in ic_results.columns:
            ic_pivot = ic_results.pivot(
                index="factor", columns="period", values="ic_spearman"
            )
            sns.heatmap(
                ic_pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=axes[0, 0]
            )
            axes[0, 0].set_title("各因子IC值热力图")
            axes[0, 0].set_xlabel("未来周期")
            axes[0, 0].set_ylabel("因子")

        # 2. 因子IC排名
        factor_stats = (
            ic_results.groupby("factor")["ic_spearman"]
            .agg(["mean", "std"])
            .reset_index()
        )
        factor_stats = factor_stats.sort_values("mean", ascending=False)

        axes[0, 1].barh(range(len(factor_stats)), factor_stats["mean"])
        axes[0, 1].set_yticks(range(len(factor_stats)))
        axes[0, 1].set_yticklabels(factor_stats["factor"])
        axes[0, 1].set_xlabel("平均IC值")
        axes[0, 1].set_title("因子IC值排名")
        axes[0, 1].axvline(x=0, color="r", linestyle="--", alpha=0.5)

        # 3. IC衰减曲线
        if "period" in ic_results.columns:
            period_ic = ic_results.groupby("period")["ic_spearman"].mean().reset_index()
            axes[1, 0].plot(
                period_ic["period"], period_ic["ic_spearman"], marker="o", linewidth=2
            )
            axes[1, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
            axes[1, 0].set_xlabel("未来周期")
            axes[1, 0].set_ylabel("平均IC值")
            axes[1, 0].set_title("IC衰减曲线")
            axes[1, 0].grid(True, alpha=0.3)

        # 4. IC分布直方图
        axes[1, 1].hist(
            ic_results["ic_spearman"].dropna(), bins=30, edgecolor="black", alpha=0.7
        )
        axes[1, 1].axvline(
            x=ic_results["ic_spearman"].mean(),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"均值: {ic_results['ic_spearman'].mean():.3f}",
        )
        axes[1, 1].set_xlabel("IC值")
        axes[1, 1].set_ylabel("频数")
        axes[1, 1].set_title("IC值分布")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_group_performance(
        self, group_performance: pd.DataFrame, title: str = "因子分组表现"
    ) -> plt.Figure:
        """
        绘制分组表现结果

        Args:
            group_performance: 分组表现DataFrame
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)

        # 筛选原始分组数据
        group_data = group_performance[
            group_performance["group"].str.startswith("Group_")
        ].copy()

        if len(group_data) == 0:
            return fig

        # 1. 分组收益率柱状图
        for period in group_data["period"].unique():
            period_data = group_data[group_data["period"] == period]
            axes[0, 0].bar(
                period_data["group"],
                period_data["mean_return"],
                label=f"Period {period}",
                alpha=0.7,
            )

        axes[0, 0].set_xlabel("分组")
        axes[0, 0].set_ylabel("平均收益率")
        axes[0, 0].set_title("分组平均收益率")
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. 分组夏普比率
        for period in group_data["period"].unique():
            period_data = group_data[group_data["period"] == period]
            axes[0, 1].bar(
                period_data["group"],
                period_data["sharpe_ratio"],
                label=f"Period {period}",
                alpha=0.7,
            )

        axes[0, 1].set_xlabel("分组")
        axes[0, 1].set_ylabel("夏普比率")
        axes[0, 1].set_title("分组夏普比率")
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. 多空组合表现
        ls_data = group_performance[group_performance["group"] == "Long-Short"]
        if len(ls_data) > 0:
            periods = ls_data["period"].values
            mean_returns = ls_data["mean_return"].values
            sharpe_ratios = ls_data["sharpe_ratio"].values

            x = np.arange(len(periods))
            width = 0.35

            axes[1, 0].bar(x - width / 2, mean_returns, width, label="平均收益率")
            axes[1, 0].bar(x + width / 2, sharpe_ratios, width, label="夏普比率")
            axes[1, 0].set_xlabel("未来周期")
            axes[1, 0].set_title("多空组合表现")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(periods)
            axes[1, 0].legend()
            axes[1, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)

        # 4. 分组胜率
        for period in group_data["period"].unique():
            period_data = group_data[group_data["period"] == period]
            axes[1, 1].bar(
                period_data["group"],
                period_data["win_rate"],
                label=f"Period {period}",
                alpha=0.7,
            )

        axes[1, 1].set_xlabel("分组")
        axes[1, 1].set_ylabel("胜率")
        axes[1, 1].set_title("分组胜率")
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_factor_correlation(
        self, factor_df: pd.DataFrame, title: str = "因子相关性矩阵"
    ) -> plt.Figure:
        """
        绘制因子相关性矩阵

        Args:
            factor_df: 因子值DataFrame
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] // 2))
        fig.suptitle(title, fontsize=16)

        # 计算相关性矩阵
        corr_matrix = factor_df.corr()

        # 1. 相关性热力图
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=axes[0],
        )
        axes[0].set_title("因子相关性热力图")

        # 2. 相关性分布
        corr_values = corr_matrix.values.flatten()
        corr_values = corr_values[~np.isnan(corr_values)]
        corr_values = corr_values[corr_values != 1.0]  # 移除对角线

        axes[1].hist(corr_values, bins=30, edgecolor="black", alpha=0.7)
        axes[1].axvline(
            x=np.mean(corr_values),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"均值: {np.mean(corr_values):.3f}",
        )
        axes[1].set_xlabel("相关系数")
        axes[1].set_ylabel("频数")
        axes[1].set_title("因子相关性分布")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_factor_tsne(
        self, factor_df: pd.DataFrame, title: str = "因子t-SNE降维可视化"
    ) -> Optional[plt.Figure]:
        """
        使用t-SNE对因子进行降维可视化

        Args:
            factor_df: 因子值DataFrame
            title: 图表标题

        Returns:
            matplotlib Figure对象或None
        """
        try:
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler

            # 数据预处理
            factor_data = factor_df.dropna()
            if len(factor_data) < 10:
                return None

            # 标准化
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(factor_data)

            # t-SNE降维
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(factor_data) - 1),
            )
            tsne_results = tsne.fit_transform(scaled_data)

            fig, ax = plt.subplots(figsize=(10, 8))

            # 绘制散点图
            scatter = ax.scatter(
                tsne_results[:, 0], tsne_results[:, 1], alpha=0.6, s=50
            )
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # 添加时间颜色映射
            if len(factor_data) > 0:
                cmap = plt.cm.viridis
                time_indices = np.arange(len(factor_data))
                norm = plt.Normalize(time_indices.min(), time_indices.max())
                scatter.set_array(time_indices)
                scatter.set_cmap(cmap)
                scatter.set_norm(norm)

                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("时间序列")

            plt.tight_layout()
            return fig

        except ImportError:
            warnings.warn("sklearn未安装，跳过t-SNE可视化")
            return None

    def plot_comprehensive_report(
        self,
        ic_results: pd.DataFrame,
        group_performance: pd.DataFrame,
        factor_df: pd.DataFrame,
        title: str = "因子综合评估报告",
    ) -> plt.Figure:
        """
        绘制综合评估报告

        Args:
            ic_results: IC分析结果
            group_performance: 分组表现结果
            factor_df: 因子值DataFrame

        Returns:
            matplotlib Figure对象
        """
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(title, fontsize=18, y=0.98)

        # 创建子图网格
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 因子IC排名 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        factor_stats = (
            ic_results.groupby("factor")["ic_spearman"]
            .agg(["mean", "std"])
            .reset_index()
        )
        factor_stats = factor_stats.sort_values("mean", ascending=False)
        ax1.barh(range(len(factor_stats)), factor_stats["mean"])
        ax1.set_yticks(range(len(factor_stats)))
        ax1.set_yticklabels(factor_stats["factor"])
        ax1.set_xlabel("平均IC值")
        ax1.set_title("因子IC值排名")
        ax1.axvline(x=0, color="r", linestyle="--", alpha=0.5)

        # 2. IC衰减曲线 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        if "period" in ic_results.columns:
            period_ic = ic_results.groupby("period")["ic_spearman"].mean().reset_index()
            ax2.plot(
                period_ic["period"], period_ic["ic_spearman"], marker="o", linewidth=2
            )
            ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            ax2.set_xlabel("未来周期")
            ax2.set_ylabel("平均IC值")
            ax2.set_title("IC衰减曲线")
            ax2.grid(True, alpha=0.3)

        # 3. 分组收益率 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        group_data = group_performance[
            group_performance["group"].str.startswith("Group_")
        ].copy()
        if len(group_data) > 0:
            period = group_data["period"].iloc[0]  # 取第一个周期
            period_data = group_data[group_data["period"] == period]
            ax3.bar(period_data["group"], period_data["mean_return"])
            ax3.set_xlabel("分组")
            ax3.set_ylabel("平均收益率")
            ax3.set_title(f"分组收益率 (Period {period})")
            ax3.tick_params(axis="x", rotation=45)

        # 4. 因子相关性热力图 (左下，跨2列)
        ax4 = fig.add_subplot(gs[1, :2])
        corr_matrix = factor_df.corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax4,
        )
        ax4.set_title("因子相关性矩阵")

        # 5. 多空组合表现 (右下)
        ax5 = fig.add_subplot(gs[1, 2])
        ls_data = group_performance[group_performance["group"] == "Long-Short"]
        if len(ls_data) > 0:
            periods = ls_data["period"].values
            sharpe_ratios = ls_data["sharpe_ratio"].values

            ax5.bar(range(len(periods)), sharpe_ratios)
            ax5.set_xlabel("未来周期")
            ax5.set_ylabel("夏普比率")
            ax5.set_title("多空组合夏普比率")
            ax5.set_xticks(range(len(periods)))
            ax5.set_xticklabels(periods)
            ax5.axhline(y=0, color="r", linestyle="--", alpha=0.5)

        # 6. IC值分布 (左下)
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.hist(
            ic_results["ic_spearman"].dropna(), bins=30, edgecolor="black", alpha=0.7
        )
        ax6.axvline(
            x=ic_results["ic_spearman"].mean(),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"均值: {ic_results['ic_spearman'].mean():.3f}",
        )
        ax6.set_xlabel("IC值")
        ax6.set_ylabel("频数")
        ax6.set_title("IC值分布")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. 分组夏普比率 (中下)
        ax7 = fig.add_subplot(gs[2, 1])
        if len(group_data) > 0:
            period = group_data["period"].iloc[0]
            period_data = group_data[group_data["period"] == period]
            ax7.bar(period_data["group"], period_data["sharpe_ratio"])
            ax7.set_xlabel("分组")
            ax7.set_ylabel("夏普比率")
            ax7.set_title(f"分组夏普比率 (Period {period})")
            ax7.tick_params(axis="x", rotation=45)

        # 8. 分组胜率 (右下)
        ax8 = fig.add_subplot(gs[2, 2])
        if len(group_data) > 0:
            period = group_data["period"].iloc[0]
            period_data = group_data[group_data["period"] == period]
            ax8.bar(period_data["group"], period_data["win_rate"])
            ax8.set_xlabel("分组")
            ax8.set_ylabel("胜率")
            ax8.set_title(f"分组胜率 (Period {period})")
            ax8.tick_params(axis="x", rotation=45)
            ax8.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig
