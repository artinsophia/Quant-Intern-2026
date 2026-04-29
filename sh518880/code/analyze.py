from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INSTRUMENT_ID = "518880"
START_YMD = "20251201"
END_YMD = "20260425"

BASE_CANDIDATES = [Path("/home/jovyan/work/base_demo"), Path("/home/jovyan/base_demo")]
for base_path in BASE_CANDIDATES:
    if base_path.exists() and str(base_path) not in sys.path:
        sys.path.append(str(base_path))

import base_tool  # type: ignore


@dataclass(frozen=True)
class SessionConfig:
    tick_size: float = 0.001
    depth_levels: int = 5
    time_bucket_minutes: int = 30
    timezone: str = "Asia/Shanghai"


CFG = SessionConfig()


def candidate_dates(start_ymd: str = START_YMD, end_ymd: str = END_YMD) -> list[str]:
    return [d.strftime("%Y%m%d") for d in pd.date_range(start=start_ymd, end=end_ymd, freq="B")]


def load_snaps(trade_ymd: str) -> list[dict]:
    with redirect_stdout(io.StringIO()):
        snap_list = base_tool.snap_list_load(INSTRUMENT_ID, trade_ymd)
    if not isinstance(snap_list, list):
        return []
    return snap_list


def valid_dates(dates: Iterable[str] | None = None) -> list[str]:
    if dates is None:
        dates = candidate_dates()
    return [trade_ymd for trade_ymd in dates if load_snaps(trade_ymd)]


def sum_trade_turnover(trades: list[list[float]]) -> float:
    return float(sum(price * volume for price, volume in trades))


def sum_trade_volume(trades: list[list[float]]) -> float:
    return float(sum(volume for _, volume in trades))


def sum_depth(book: list[list[float]], levels: int) -> float:
    return float(sum(volume for _, volume in book[:levels]))


def bucket_label(ts: pd.Timestamp, minutes: int = CFG.time_bucket_minutes) -> str:
    start = ts.floor(f"{minutes}min")
    end = start + pd.Timedelta(minutes=minutes)
    return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def build_day_frame(trade_ymd: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for snap in load_snaps(trade_ymd):
        bid_book = snap.get("bid_book") or []
        ask_book = snap.get("ask_book") or []
        if not bid_book or not ask_book:
            continue

        best_bid, best_bid_vol = bid_book[0]
        best_ask, best_ask_vol = ask_book[0]
        buy_trade = snap.get("buy_trade") or []
        sell_trade = snap.get("sell_trade") or []
        ts = (
            pd.to_datetime(int(snap["time_mark"]), unit="ms", utc=True)
            .tz_convert(CFG.timezone)
            .tz_localize(None)
        )

        rows.append(
            {
                "trade_ymd": trade_ymd,
                "timestamp": ts,
                "time_hms": ts.strftime("%H:%M:%S"),
                "spread_ticks": round((best_ask - best_bid) / CFG.tick_size),
                "mid_price": (best_bid + best_ask) / 2.0,
                "turnover": sum_trade_turnover(buy_trade) + sum_trade_turnover(sell_trade),
                "trade_volume": sum_trade_volume(buy_trade) + sum_trade_volume(sell_trade),
                "trade_count_delta": int(snap.get("num_trades", 0)),
                "l1_depth": float(best_bid_vol + best_ask_vol),
                "l5_depth": sum_depth(bid_book, CFG.depth_levels) + sum_depth(ask_book, CFG.depth_levels),
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["trade_count_delta"] = df["trade_count_delta"].diff().fillna(df["trade_count_delta"]).clip(lower=0)
    # Mid can move by half-tick when one side of the quote changes while the other stays.
    # Quantize to 0.5-tick steps instead of rounding straight to an integer tick.
    df["mid_change_ticks"] = np.rint(
        df["mid_price"].diff().fillna(0.0) / (CFG.tick_size / 2.0)
    ) / 2.0
    df["l1_depth_change"] = df["l1_depth"].diff().fillna(0.0)
    df["l5_depth_change"] = df["l5_depth"].diff().fillna(0.0)
    df["active_second"] = (df["turnover"] > 0).astype(int)
    df["bucket_30m"] = df["timestamp"].map(bucket_label)
    return df


def quantile_table(series: pd.Series, quantiles: Iterable[float]) -> pd.Series:
    q = series.quantile(list(quantiles))
    q.index = [f"q{int(x * 100):02d}" for x in q.index]
    return q


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    spread = (
        df["spread_ticks"]
        .value_counts(normalize=True)
        .sort_index()
        .rename("ratio")
        .reset_index()
        .rename(columns={"index": "spread_ticks"})
    )

    turnover = pd.concat(
        [
            quantile_table(df["turnover"], [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
            pd.Series({"mean": df["turnover"].mean(), "std": df["turnover"].std()}),
        ]
    ).to_frame("turnover")

    mid = pd.concat(
        [
            quantile_table(df["mid_change_ticks"], [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]),
            pd.Series(
                {
                    "mean": df["mid_change_ticks"].mean(),
                    "std": df["mid_change_ticks"].std(),
                    "zero_ratio": (df["mid_change_ticks"] == 0).mean(),
                    "one_tick_abs_ratio": (df["mid_change_ticks"].abs() == 1).mean(),
                }
            ),
        ]
    ).to_frame("mid_change_ticks")

    depth = pd.DataFrame(
        {
            "l1_depth": pd.concat(
                [
                    quantile_table(df["l1_depth"], [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]),
                    pd.Series({"mean": df["l1_depth"].mean(), "std": df["l1_depth"].std()}),
                ]
            ),
            "l5_depth": pd.concat(
                [
                    quantile_table(df["l5_depth"], [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]),
                    pd.Series({"mean": df["l5_depth"].mean(), "std": df["l5_depth"].std()}),
                ]
            ),
        }
    )

    depth_change = pd.DataFrame(
        {
            "l1_depth_change": pd.concat(
                [
                    quantile_table(df["l1_depth_change"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
                    pd.Series(
                        {
                            "mean": df["l1_depth_change"].mean(),
                            "std": df["l1_depth_change"].std(),
                            "abs_mean": df["l1_depth_change"].abs().mean(),
                        }
                    ),
                ]
            ),
            "l5_depth_change": pd.concat(
                [
                    quantile_table(df["l5_depth_change"], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
                    pd.Series(
                        {
                            "mean": df["l5_depth_change"].mean(),
                            "std": df["l5_depth_change"].std(),
                            "abs_mean": df["l5_depth_change"].abs().mean(),
                        }
                    ),
                ]
            ),
        }
    )

    activity = (
        df.groupby("bucket_30m", sort=False)
        .agg(
            seconds=("bucket_30m", "size"),
            turnover_sum=("turnover", "sum"),
            avg_turnover_per_sec=("turnover", "mean"),
            median_turnover_per_sec=("turnover", "median"),
            avg_trade_count_per_sec=("trade_count_delta", "mean"),
            active_second_ratio=("active_second", "mean"),
            avg_trade_volume_per_sec=("trade_volume", "mean"),
        )
        .reset_index()
    )
    activity["turnover_share"] = activity["turnover_sum"] / activity["turnover_sum"].sum()

    headline = pd.DataFrame(
        [
            {
                "instrument_id": INSTRUMENT_ID,
                "trade_days": int(df["trade_ymd"].nunique()),
                "snapshots": int(len(df)),
                "tick_size": CFG.tick_size,
                "one_tick_spread_ratio": (df["spread_ticks"] == 1).mean(),
                "two_tick_spread_ratio": (df["spread_ticks"] == 2).mean(),
                "avg_turnover_per_sec": df["turnover"].mean(),
                "median_turnover_per_sec": df["turnover"].median(),
                "mid_unchanged_ratio": (df["mid_change_ticks"] == 0).mean(),
                "avg_l5_depth": df["l5_depth"].mean(),
                "avg_abs_l5_depth_change": df["l5_depth_change"].abs().mean(),
            }
        ]
    )

    return {
        "headline": headline,
        "spread": spread,
        "turnover": turnover,
        "mid": mid,
        "depth": depth,
        "depth_change": depth_change,
        "activity": activity,
    }


def run() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    dates = valid_dates()
    if not dates:
        raise RuntimeError(f"No snapshot data found for {INSTRUMENT_ID}")
    df = pd.concat([build_day_frame(trade_ymd) for trade_ymd in dates], ignore_index=True)
    return df, summarize(df)


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path = DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=True)


def plot_overview(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> list[plt.Figure]:
    plt.style.use("seaborn-v0_8-whitegrid")
    figs: list[plt.Figure] = []

    spread = tables["spread"]
    spread = spread[spread["spread_ticks"] <= 10]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(spread["spread_ticks"].astype(str), spread["ratio"], color="#0f766e")
    ax.set_title("Spread (ticks)")
    ax.set_ylabel("ratio")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    positive = df.loc[df["turnover"] > 0, "turnover"]
    bins = np.geomspace(max(float(positive.min()), 1.0), float(positive.quantile(0.999)), 50)
    ax.hist(positive, bins=bins, color="#2563eb", alpha=0.85)
    ax.set_xscale("log")
    ax.set_title("Turnover per second")
    ax.set_xlabel("turnover")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    mid_dist = df["mid_change_ticks"].clip(-5, 5).value_counts(normalize=True).sort_index()
    ax.bar(mid_dist.index.astype(str), mid_dist.values, color="#7c3aed")
    ax.set_title("Mid-price change (ticks)")
    ax.set_ylabel("ratio")
    figs.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(df["l1_depth"], bins=50, color="#ea580c", alpha=0.85)
    axes[0].set_title("L1 depth")
    axes[1].hist(df["l5_depth"], bins=50, color="#16a34a", alpha=0.85)
    axes[1].set_title("L5 depth")
    figs.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(df["l1_depth_change"].clip(-500_000, 500_000), bins=60, color="#dc2626", alpha=0.85)
    axes[0].set_title("L1 depth change")
    axes[1].hist(df["l5_depth_change"].clip(-1_500_000, 1_500_000), bins=60, color="#0891b2", alpha=0.85)
    axes[1].set_title("L5 depth change")
    figs.append(fig)

    fig, ax1 = plt.subplots(figsize=(9, 4))
    activity = tables["activity"]
    ax1.bar(activity["bucket_30m"], activity["turnover_share"], color="#1d4ed8", alpha=0.8)
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylabel("turnover share")
    ax2 = ax1.twinx()
    ax2.plot(activity["bucket_30m"], activity["active_second_ratio"], color="#dc2626", marker="o")
    ax2.set_ylabel("active second ratio")
    ax1.set_title("Activity by time bucket")
    figs.append(fig)

    return figs


def main() -> None:
    df, tables = run()
    save_tables(tables)
    df.sample(min(len(df), 50_000), random_state=42).to_csv(DATA_DIR / "sample.csv", index=False)
    print(f"trade_days={tables['headline'].iloc[0]['trade_days']}")
    print(f"snapshots={len(df)}")
    print(f"saved_to={DATA_DIR}")


if __name__ == "__main__":
    main()
