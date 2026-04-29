from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


def bucket_label(ts: pd.Timestamp, minutes: int = CFG.time_bucket_minutes) -> str:
    start = ts.floor(f"{minutes}min")
    end = start + pd.Timedelta(minutes=minutes)
    return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def spread_bucket(spread_ticks: int) -> str:
    if spread_ticks <= 5:
        return str(spread_ticks)
    return "6+"


def build_frames(dates: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, float | int | str]] = []
    second_rows: list[dict[str, float | int | str]] = []

    for trade_ymd in dates:
        for snap in load_snaps(trade_ymd):
            bid_book = snap.get("bid_book") or []
            ask_book = snap.get("ask_book") or []
            if not bid_book or not ask_book:
                continue

            best_bid, best_bid_vol = bid_book[0]
            best_ask, best_ask_vol = ask_book[0]
            spread_ticks = round((best_ask - best_bid) / CFG.tick_size)
            ts = (
                pd.to_datetime(int(snap["time_mark"]), unit="ms", utc=True)
                .tz_convert(CFG.timezone)
                .tz_localize(None)
            )
            buy_trade = snap.get("buy_trade") or []
            sell_trade = snap.get("sell_trade") or []
            buy_second_volume = float(sum(volume for _, volume in buy_trade))
            sell_second_volume = float(sum(volume for _, volume in sell_trade))

            second_rows.append(
                {
                    "trade_ymd": trade_ymd,
                    "timestamp": ts,
                    "time_hms": ts.strftime("%H:%M:%S"),
                    "bucket_30m": bucket_label(ts),
                    "spread_ticks": spread_ticks,
                    "best_bid": float(best_bid),
                    "best_ask": float(best_ask),
                    "best_bid_vol": float(best_bid_vol),
                    "best_ask_vol": float(best_ask_vol),
                    "buy_second_volume": buy_second_volume,
                    "sell_second_volume": sell_second_volume,
                    "second_total_volume": buy_second_volume + sell_second_volume,
                    "active_second": int((buy_second_volume + sell_second_volume) > 0),
                }
            )

            for side, trades in (("buy", buy_trade), ("sell", sell_trade)):
                for price, volume in trades:
                    event_rows.append(
                        {
                            "trade_ymd": trade_ymd,
                            "timestamp": ts,
                            "time_hms": ts.strftime("%H:%M:%S"),
                            "bucket_30m": bucket_label(ts),
                            "side": side,
                            "price": float(price),
                            "volume": float(volume),
                            "spread_ticks": spread_ticks,
                        }
                    )

    events = pd.DataFrame(event_rows).sort_values(["timestamp", "side"]).reset_index(drop=True)
    seconds = pd.DataFrame(second_rows).sort_values("timestamp").reset_index(drop=True)
    return events, seconds


def quantile_table(series: pd.Series, quantiles: Iterable[float]) -> pd.Series:
    labels = {
        0.0: "q000",
        0.1: "q10",
        0.25: "q25",
        0.5: "q50",
        0.75: "q75",
        0.9: "q90",
        0.95: "q95",
        0.99: "q99",
        0.999: "q999",
        1.0: "q100",
    }
    quantile_list = list(quantiles)
    q = series.quantile(quantile_list)
    q.index = [labels[float(x)] for x in quantile_list]
    return q


def summarize(events: pd.DataFrame, seconds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    active_seconds = seconds.loc[seconds["active_second"] == 1].copy()
    total_event_volume = float(events["volume"].sum())

    headline = pd.DataFrame(
        [
            {
                "instrument_id": INSTRUMENT_ID,
                "trade_days": int(seconds["trade_ymd"].nunique()),
                "snapshots": int(len(seconds)),
                "active_seconds": int(len(active_seconds)),
                "active_second_ratio": float(seconds["active_second"].mean()),
                "event_count": int(len(events)),
                "buy_event_ratio": float((events["side"] == "buy").mean()),
                "sell_event_ratio": float((events["side"] == "sell").mean()),
                "total_event_volume": total_event_volume,
                "median_event_volume": float(events["volume"].median()),
                "p90_event_volume": float(events["volume"].quantile(0.9)),
                "p99_event_volume": float(events["volume"].quantile(0.99)),
                "median_active_second_volume": float(active_seconds["second_total_volume"].median()),
                "p90_active_second_volume": float(active_seconds["second_total_volume"].quantile(0.9)),
            }
        ]
    )

    quantile_rows: list[dict[str, float | str]] = []
    for scope, frame in {
        "all_events": events["volume"],
        "buy_events": events.loc[events["side"] == "buy", "volume"],
        "sell_events": events.loc[events["side"] == "sell", "volume"],
        "active_seconds": active_seconds["second_total_volume"],
    }.items():
        row = {"scope": scope}
        row.update(
            quantile_table(frame, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]).to_dict()
        )
        quantile_rows.append(row)
    quantiles = pd.DataFrame(quantile_rows)

    side_summary = (
        events.groupby("side")
        .agg(
            event_count=("volume", "size"),
            volume_sum=("volume", "sum"),
            volume_share=("volume", lambda s: s.sum() / total_event_volume),
            mean_volume=("volume", "mean"),
            median_volume=("volume", "median"),
            p90_volume=("volume", lambda s: s.quantile(0.9)),
            p99_volume=("volume", lambda s: s.quantile(0.99)),
        )
        .reset_index()
    )

    second_mix = seconds.assign(
        second_type=np.select(
            [
                (seconds["buy_second_volume"] > 0) & (seconds["sell_second_volume"] > 0),
                (seconds["buy_second_volume"] > 0) & (seconds["sell_second_volume"] == 0),
                (seconds["buy_second_volume"] == 0) & (seconds["sell_second_volume"] > 0),
            ],
            [
                "both_sides",
                "buy_only",
                "sell_only",
            ],
            default="no_trade",
        )
    )
    second_mix = (
        second_mix.groupby("second_type")
        .agg(
            seconds=("second_type", "size"),
            ratio=("second_type", lambda s: len(s) / len(seconds)),
            mean_second_volume=("second_total_volume", "mean"),
            median_second_volume=("second_total_volume", "median"),
            p90_second_volume=("second_total_volume", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )

    spread_summary = (
        events.assign(spread_bucket=lambda x: x["spread_ticks"].map(spread_bucket))
        .groupby("spread_bucket")
        .agg(
            event_count=("volume", "size"),
            volume_sum=("volume", "sum"),
            volume_share=("volume", lambda s: s.sum() / total_event_volume),
            mean_volume=("volume", "mean"),
            median_volume=("volume", "median"),
            p90_volume=("volume", lambda s: s.quantile(0.9)),
            p99_volume=("volume", lambda s: s.quantile(0.99)),
        )
        .reset_index()
    )

    time_bucket_summary = (
        events.groupby("bucket_30m", sort=False)
        .agg(
            event_count=("volume", "size"),
            volume_sum=("volume", "sum"),
            volume_share=("volume", lambda s: s.sum() / total_event_volume),
            mean_volume=("volume", "mean"),
            median_volume=("volume", "median"),
            p90_volume=("volume", lambda s: s.quantile(0.9)),
            p99_volume=("volume", lambda s: s.quantile(0.99)),
        )
        .reset_index()
    )

    round_lot = (
        events["volume"]
        .value_counts()
        .rename_axis("volume")
        .reset_index(name="event_count")
        .head(30)
        .sort_values("volume")
        .reset_index(drop=True)
    )
    round_lot["event_ratio"] = round_lot["event_count"] / len(events)
    round_lot["volume_sum"] = round_lot["volume"] * round_lot["event_count"]
    round_lot["volume_share"] = round_lot["volume_sum"] / total_event_volume

    tail_rows = []
    for quantile in [0.9, 0.95, 0.99, 0.999]:
        threshold = float(events["volume"].quantile(quantile))
        tail = events.loc[events["volume"] >= threshold, "volume"]
        tail_rows.append(
            {
                "quantile": quantile,
                "threshold_volume": threshold,
                "event_ratio": float(len(tail) / len(events)),
                "volume_share": float(tail.sum() / total_event_volume),
                "mean_tail_volume": float(tail.mean()),
            }
        )
    tail_summary = pd.DataFrame(tail_rows)

    positive = events.loc[events["volume"] > 0, "volume"]
    bins = np.unique(np.rint(np.geomspace(max(float(positive.min()), 1.0), float(positive.max()), 40)).astype(int))
    hist_counts, bin_edges = np.histogram(positive, bins=bins)
    hist = pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "event_count": hist_counts,
        }
    )
    hist = hist.loc[hist["event_count"] > 0].copy()
    hist["event_ratio"] = hist["event_count"] / len(events)

    return {
        "active_trade_volume_headline": headline,
        "active_trade_volume_quantiles": quantiles,
        "active_trade_volume_side": side_summary,
        "active_trade_volume_second_mix": second_mix,
        "active_trade_volume_spread": spread_summary,
        "active_trade_volume_time_bucket": time_bucket_summary,
        "active_trade_volume_round_lot": round_lot,
        "active_trade_volume_tail": tail_summary,
        "active_trade_volume_hist": hist,
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    dates = valid_dates()
    if not dates:
        raise RuntimeError(f"No snapshot data found for {INSTRUMENT_ID}")
    events, seconds = build_frames(dates)
    return events, seconds, summarize(events, seconds)


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path = DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    events, seconds, tables = run()
    save_tables(tables)
    print(f"trade_days={tables['active_trade_volume_headline'].iloc[0]['trade_days']}")
    print(f"event_count={len(events)}")
    print(f"active_seconds={len(seconds.loc[seconds['active_second'] == 1])}")
    print(f"saved_to={DATA_DIR}")


if __name__ == "__main__":
    main()
