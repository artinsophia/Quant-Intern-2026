from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INSTRUMENT_ID = "511090"
START_YMD = "20260101"
END_YMD = "20260428"

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


def quantile_table(series: pd.Series, quantiles: Iterable[float]) -> pd.Series:
    q = series.quantile(list(quantiles))
    q.index = [f"q{int(x * 100):02d}" for x in q.index]
    return q


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
        l1_depth = float(best_bid_vol + best_ask_vol)
        imbalance_l1 = 0.0 if l1_depth == 0 else float(best_bid_vol - best_ask_vol) / l1_depth

        rows.append(
            {
                "trade_ymd": trade_ymd,
                "timestamp": ts,
                "time_hms": ts.strftime("%H:%M:%S"),
                "price_last": float(snap.get("price_last", 0.0)),
                "mid_price": (best_bid + best_ask) / 2.0,
                "spread_ticks": round((best_ask - best_bid) / CFG.tick_size),
                "turnover": sum_trade_turnover(buy_trade) + sum_trade_turnover(sell_trade),
                "trade_volume": sum_trade_volume(buy_trade) + sum_trade_volume(sell_trade),
                "trade_count_cum": int(snap.get("num_trades", 0)),
                "bid1_vol": float(best_bid_vol),
                "ask1_vol": float(best_ask_vol),
                "l1_depth": l1_depth,
                "l5_depth": sum_depth(bid_book, CFG.depth_levels) + sum_depth(ask_book, CFG.depth_levels),
                "imbalance_l1": imbalance_l1,
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return df

    df["trade_count_delta"] = df["trade_count_cum"].diff().fillna(df["trade_count_cum"]).clip(lower=0)
    df["mid_change_ticks"] = df["mid_price"].diff().fillna(0.0) / CFG.tick_size
    df["last_price_change_ticks"] = df["price_last"].diff().fillna(0.0) / CFG.tick_size
    df["l1_depth_change"] = df["l1_depth"].diff().fillna(0.0)
    df["l5_depth_change"] = df["l5_depth"].diff().fillna(0.0)
    df["active_second"] = (df["turnover"] > 0).astype(int)
    df["bucket_30m"] = df["timestamp"].map(bucket_label)
    return df


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    headline = pd.DataFrame(
        [
            {
                "instrument_id": INSTRUMENT_ID,
                "start_ymd": df["trade_ymd"].min(),
                "end_ymd": df["trade_ymd"].max(),
                "trade_days": int(df["trade_ymd"].nunique()),
                "snapshots": int(len(df)),
                "tick_size": CFG.tick_size,
                "avg_price_last": df["price_last"].mean(),
                "avg_spread_ticks": df["spread_ticks"].mean(),
                "one_tick_spread_ratio": (df["spread_ticks"] == 1).mean(),
                "avg_turnover_per_sec": df["turnover"].mean(),
                "median_turnover_per_sec": df["turnover"].median(),
                "active_second_ratio": df["active_second"].mean(),
                "mid_unchanged_ratio": (df["mid_change_ticks"] == 0).mean(),
                "avg_l1_depth": df["l1_depth"].mean(),
                "avg_l5_depth": df["l5_depth"].mean(),
                "avg_abs_imbalance_l1": df["imbalance_l1"].abs().mean(),
            }
        ]
    )

    spread = (
        df["spread_ticks"]
        .value_counts(normalize=True)
        .sort_index()
        .rename("ratio")
        .reset_index()
        .rename(columns={"index": "spread_ticks"})
    )

    price = pd.DataFrame(
        {
            "price_last": pd.concat(
                [
                    quantile_table(df["price_last"], [0, 0.25, 0.5, 0.75, 1]),
                    pd.Series({"mean": df["price_last"].mean(), "std": df["price_last"].std()}),
                ]
            ),
            "mid_price": pd.concat(
                [
                    quantile_table(df["mid_price"], [0, 0.25, 0.5, 0.75, 1]),
                    pd.Series({"mean": df["mid_price"].mean(), "std": df["mid_price"].std()}),
                ]
            ),
        }
    )

    turnover = pd.concat(
        [
            quantile_table(df["turnover"], [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
            pd.Series(
                {
                    "mean": df["turnover"].mean(),
                    "std": df["turnover"].std(),
                    "positive_ratio": (df["turnover"] > 0).mean(),
                }
            ),
        ]
    ).to_frame("turnover")

    trade = pd.DataFrame(
        {
            "trade_volume": pd.concat(
                [
                    quantile_table(df["trade_volume"], [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]),
                    pd.Series({"mean": df["trade_volume"].mean(), "std": df["trade_volume"].std()}),
                ]
            ),
            "trade_count_delta": pd.concat(
                [
                    quantile_table(df["trade_count_delta"], [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]),
                    pd.Series({"mean": df["trade_count_delta"].mean(), "std": df["trade_count_delta"].std()}),
                ]
            ),
        }
    )

    price_change = pd.DataFrame(
        {
            "mid_change_ticks": pd.concat(
                [
                    quantile_table(df["mid_change_ticks"], [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]),
                    pd.Series(
                        {
                            "mean": df["mid_change_ticks"].mean(),
                            "std": df["mid_change_ticks"].std(),
                            "zero_ratio": (df["mid_change_ticks"] == 0).mean(),
                            "abs_ge_1_ratio": (df["mid_change_ticks"].abs() >= 1).mean(),
                        }
                    ),
                ]
            ),
            "last_price_change_ticks": pd.concat(
                [
                    quantile_table(
                        df["last_price_change_ticks"], [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
                    ),
                    pd.Series(
                        {
                            "mean": df["last_price_change_ticks"].mean(),
                            "std": df["last_price_change_ticks"].std(),
                            "zero_ratio": (df["last_price_change_ticks"] == 0).mean(),
                            "abs_ge_1_ratio": (df["last_price_change_ticks"].abs() >= 1).mean(),
                        }
                    ),
                ]
            ),
        }
    )

    depth = pd.DataFrame(
        {
            "bid1_vol": pd.concat(
                [
                    quantile_table(df["bid1_vol"], [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]),
                    pd.Series({"mean": df["bid1_vol"].mean(), "std": df["bid1_vol"].std()}),
                ]
            ),
            "ask1_vol": pd.concat(
                [
                    quantile_table(df["ask1_vol"], [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]),
                    pd.Series({"mean": df["ask1_vol"].mean(), "std": df["ask1_vol"].std()}),
                ]
            ),
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
            "imbalance_l1": pd.concat(
                [
                    quantile_table(df["imbalance_l1"], [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]),
                    pd.Series({"mean": df["imbalance_l1"].mean(), "std": df["imbalance_l1"].std()}),
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
            avg_trade_volume_per_sec=("trade_volume", "mean"),
            active_second_ratio=("active_second", "mean"),
            avg_spread_ticks=("spread_ticks", "mean"),
        )
        .reset_index()
    )
    activity["turnover_share"] = activity["turnover_sum"] / activity["turnover_sum"].sum()

    daily = (
        df.groupby("trade_ymd", sort=True)
        .agg(
            snapshots=("trade_ymd", "size"),
            avg_price_last=("price_last", "mean"),
            avg_spread_ticks=("spread_ticks", "mean"),
            turnover_sum=("turnover", "sum"),
            avg_turnover_per_sec=("turnover", "mean"),
            active_second_ratio=("active_second", "mean"),
            avg_l1_depth=("l1_depth", "mean"),
            avg_l5_depth=("l5_depth", "mean"),
            mid_volatility_ticks=("mid_change_ticks", lambda x: float(x.std())),
        )
        .reset_index()
    )

    return {
        "headline": headline,
        "spread": spread,
        "price": price,
        "turnover": turnover,
        "trade": trade,
        "price_change": price_change,
        "depth": depth,
        "activity": activity,
        "daily": daily,
    }


def run() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    dates = valid_dates()
    if not dates:
        raise RuntimeError(f"No snapshot data found for {INSTRUMENT_ID}")
    df = pd.concat([build_day_frame(trade_ymd) for trade_ymd in dates], ignore_index=True)
    return df, summarize(df), dates


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path = DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=True)


def main() -> None:
    df, tables, dates = run()
    save_tables(tables)
    df.sample(min(len(df), 50_000), random_state=42).to_csv(DATA_DIR / "sample.csv", index=False)
    pd.DataFrame({"trade_ymd": dates}).to_csv(DATA_DIR / "valid_dates.csv", index=False)
    print(f"trade_days={len(dates)}")
    print(f"first_trade_day={dates[0]}")
    print(f"last_trade_day={dates[-1]}")
    print(f"snapshots={len(df)}")
    print(f"saved_to={DATA_DIR}")


if __name__ == "__main__":
    main()
