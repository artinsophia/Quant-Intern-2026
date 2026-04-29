from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RETURN_COLUMNS = [
    "profits",
    "profit",
    "returns",
    "return",
    "收益",
    "盈亏",
    "trade_pnl",
]


def _infer_return_column(df, return_col=None):
    if return_col is not None:
        if return_col not in df.columns:
            raise ValueError(f"列 {return_col!r} 不存在，可选列: {list(df.columns)}")
        return return_col

    for col in DEFAULT_RETURN_COLUMNS:
        if col in df.columns:
            return col

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) == 1:
        return numeric_columns[0]

    raise ValueError(
        "无法自动识别收益列，请显式传入 return_col。"
        f" 当前列: {list(df.columns)}"
    )


def _gaussian_kde_curve(values, points=300):
    values = np.asarray(values, dtype=float)
    if values.size < 2 or np.allclose(values.std(ddof=1), 0):
        return None, None

    std = values.std(ddof=1)
    bandwidth = 1.06 * std * values.size ** (-1 / 5)
    bandwidth = max(bandwidth, 1e-8)

    x_grid = np.linspace(values.min(), values.max(), points)
    diff = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diff**2).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2 * np.pi)
    return x_grid, density


def plot_return_distribution(
    csv_path=None,
    result_df=None,
    return_col=None,
    bins=30,
    figsize=(10, 6),
    title=None,
    save_path=None,
):
    """
    根据收益 CSV 绘制收益分布图。

    Parameters
    ----------
    csv_path : str or Path, optional
        收益结果 CSV 路径。
    result_df : pandas.DataFrame, optional
        直接传入收益结果表。与 csv_path 二选一，若同时提供则优先使用 result_df。
    return_col : str, optional
        收益列名。不传时自动识别，优先尝试 profits。
    bins : int, optional
        直方图分箱数。
    figsize : tuple, optional
        图像尺寸。
    title : str, optional
        图表标题。
    save_path : str or Path, optional
        若提供则保存图片。

    Returns
    -------
    tuple
        (fig, ax, stats_dict)
    """
    if result_df is not None:
        df = result_df.copy()
        data_name = "result_df"
    elif csv_path is not None:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        data_name = csv_path.stem
    else:
        raise ValueError("csv_path 和 result_df 至少需要提供一个")

    return_col = _infer_return_column(df, return_col=return_col)

    returns = pd.to_numeric(df[return_col], errors="coerce").dropna()
    if returns.empty:
        raise ValueError(f"列 {return_col!r} 没有可用于绘图的数值数据")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    counts, bin_edges, _ = ax.hist(
        returns,
        bins=bins,
        density=True,
        color="#5B8FF9",
        alpha=0.72,
        edgecolor="white",
        linewidth=1.0,
    )

    kde_x, kde_y = _gaussian_kde_curve(returns.values)
    if kde_x is not None:
        ax.plot(kde_x, kde_y, color="#1D39C4", linewidth=2.2)

    mean_value = float(returns.mean())
    median_value = float(returns.median())
    ax.axvline(mean_value, color="#F5222D", linestyle="--", linewidth=1.8, label=f"mean {mean_value:.2f}")
    ax.axvline(
        median_value,
        color="#13A8A8",
        linestyle="--",
        linewidth=1.8,
        label=f"median {median_value:.2f}",
    )

    used_title = title or f"Return Distribution: {data_name}"
    ax.set_title(used_title, fontsize=14, pad=12)
    ax.set_xlabel(return_col, fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    stats_dict = {
        "count": int(returns.shape[0]),
        "mean": mean_value,
        "median": median_value,
        "std": float(returns.std(ddof=1)) if returns.shape[0] > 1 else 0.0,
        "min": float(returns.min()),
        "max": float(returns.max()),
        "positive_ratio": float((returns > 0).mean()),
    }

    stats_text = (
        f"n={stats_dict['count']}  "
        f"std={stats_dict['std']:.2f}  "
        f"win={stats_dict['positive_ratio']:.1%}"
    )
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#595959",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d9d9d9", alpha=0.9),
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
    return fig, ax, stats_dict
