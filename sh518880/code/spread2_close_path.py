from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


BASE_CANDIDATES = [Path("/home/jovyan/work/base_demo"), Path("/home/jovyan/base_demo")]
for base_path in BASE_CANDIDATES:
    if base_path.exists() and str(base_path) not in sys.path:
        sys.path.append(str(base_path))

import base_tool  # type: ignore


INSTRUMENT_ID = "518880"
START_YMD = "20251201"
END_YMD = "20260425"
TICK_SIZE = 0.001
TIMEZONE = "Asia/Shanghai"
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "spread2_close_path"
WINDOW_BEFORE = 5
WINDOW_AFTER = 5


def candidate_dates(start_ymd: str = START_YMD, end_ymd: str = END_YMD) -> list[str]:
    return [d.strftime("%Y%m%d") for d in pd.date_range(start=start_ymd, end=end_ymd, freq="B")]


def in_regular_session(ts: pd.Timestamp) -> bool:
    hm = ts.hour * 100 + ts.minute
    if 930 <= hm < 1130:
        return True
    if 1300 <= hm < 1500:
        return True
    return ts.strftime("%H:%M:%S") == "15:00:00"


def load_snaps(trade_ymd: str) -> list[dict]:
    with redirect_stdout(io.StringIO()):
        snap_list = base_tool.snap_list_load(INSTRUMENT_ID, trade_ymd)
    return snap_list if isinstance(snap_list, list) else []


def valid_dates(dates: list[str] | None = None) -> list[str]:
    dates = candidate_dates() if dates is None else dates
    return [trade_ymd for trade_ymd in dates if load_snaps(trade_ymd)]


def sum_volume(trades: list[list[float]] | None) -> float:
    trades = trades or []
    return float(sum(volume for _, volume in trades))


def sum_turnover(trades: list[list[float]] | None) -> float:
    trades = trades or []
    return float(sum(price * volume for price, volume in trades))


def depth_sum(book: list[list[float]] | None, levels: int) -> float:
    book = book or []
    return float(sum(volume for _, volume in book[:levels]))


def imbalance(bid_depth: float, ask_depth: float) -> float:
    denom = bid_depth + ask_depth
    if denom <= 0:
        return math.nan
    return float((bid_depth - ask_depth) / denom)


def bucket_30m(ts: pd.Timestamp) -> str:
    start = ts.floor("30min")
    end = start + pd.Timedelta(minutes=30)
    return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def classify_open(prev_bid: float, prev_ask: float, bid: float, ask: float) -> str:
    bid_move = round((bid - prev_bid) / TICK_SIZE)
    ask_move = round((ask - prev_ask) / TICK_SIZE)
    if ask_move > 0 and bid_move == 0:
        return "ask_retreat"
    if ask_move == 0 and bid_move < 0:
        return "bid_retreat"
    if ask_move > 0 and bid_move < 0:
        return "both_widen"
    if ask_move > 0 and bid_move > 0:
        return "both_up_ask_faster"
    if ask_move < 0 and bid_move < 0:
        return "both_down_bid_faster"
    if ask_move != 0 or bid_move != 0:
        return "other_reprice"
    return "unchanged"


def classify_close(prev_bid: float, prev_ask: float, bid: float, ask: float) -> str:
    bid_move = round((bid - prev_bid) / TICK_SIZE)
    ask_move = round((ask - prev_ask) / TICK_SIZE)
    if ask_move < 0 and bid_move == 0:
        return "ask_improve"
    if ask_move == 0 and bid_move > 0:
        return "bid_improve"
    if ask_move < 0 and bid_move > 0:
        return "both_improve"
    if ask_move < 0 and bid_move < 0:
        return "both_down_ask_faster"
    if ask_move > 0 and bid_move > 0:
        return "both_up_bid_faster"
    if ask_move != 0 or bid_move != 0:
        return "other_reprice"
    return "unchanged"


def entry_direction(entry_mechanism: str, entry_mid_move_half_tick: float) -> int:
    if entry_mechanism in {"ask_retreat", "both_up_ask_faster"}:
        return 1
    if entry_mechanism in {"bid_retreat", "both_down_bid_faster"}:
        return -1
    if entry_mid_move_half_tick > 0:
        return 1
    if entry_mid_move_half_tick < 0:
        return -1
    return 0


def resolution_path(direction: int, exit_bid_move: int, exit_ask_move: int) -> str:
    if direction > 0:
        rollback_amt = max(-exit_ask_move, 0)
        advance_amt = max(exit_bid_move, 0)
    elif direction < 0:
        rollback_amt = max(exit_bid_move, 0)
        advance_amt = max(-exit_ask_move, 0)
    else:
        rollback_amt = max(exit_bid_move, 0) + max(-exit_ask_move, 0)
        advance_amt = 0

    if rollback_amt > 0 and advance_amt == 0:
        return "rollback_close"
    if advance_amt > 0 and rollback_amt == 0:
        return "advance_close"
    if rollback_amt > 0 and advance_amt > 0:
        if rollback_amt > advance_amt:
            return "rollback_dominant"
        if advance_amt > rollback_amt:
            return "advance_dominant"
        return "mixed_close"
    return "other_close"


def price_path_label(direction: int, close_mid_from_pre_entry_tick: float) -> str:
    if abs(close_mid_from_pre_entry_tick) < 1e-9:
        return "flat"
    if direction == 0:
        return "neutral_entry"
    if close_mid_from_pre_entry_tick * direction > 0:
        return "continue"
    return "reverse"


def build_day_panel(trade_ymd: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for snap in load_snaps(trade_ymd):
        bid_book = snap.get("bid_book") or []
        ask_book = snap.get("ask_book") or []
        if not bid_book or not ask_book:
            continue

        best_bid, bid_l1 = bid_book[0]
        best_ask, ask_l1 = ask_book[0]
        bid_l3 = depth_sum(bid_book, 3)
        ask_l3 = depth_sum(ask_book, 3)
        bid_l5 = depth_sum(bid_book, 5)
        ask_l5 = depth_sum(ask_book, 5)
        buy_trade = snap.get("buy_trade") or []
        sell_trade = snap.get("sell_trade") or []

        ts = (
            pd.to_datetime(int(snap["time_mark"]), unit="ms", utc=True)
            .tz_convert(TIMEZONE)
            .tz_localize(None)
        )
        if not in_regular_session(ts):
            continue

        rows.append(
            {
                "trade_ymd": trade_ymd,
                "timestamp": ts,
                "time_hms": ts.strftime("%H:%M:%S"),
                "best_bid": float(best_bid),
                "best_ask": float(best_ask),
                "spread_ticks": float((best_ask - best_bid) / TICK_SIZE),
                "mid": float((best_bid + best_ask) / 2.0),
                "bid_l1": float(bid_l1),
                "ask_l1": float(ask_l1),
                "bid_l3": bid_l3,
                "ask_l3": ask_l3,
                "bid_l5": bid_l5,
                "ask_l5": ask_l5,
                "buy_vol": sum_volume(buy_trade),
                "sell_vol": sum_volume(sell_trade),
                "buy_turnover": sum_turnover(buy_trade),
                "sell_turnover": sum_turnover(sell_trade),
                "num_trades": float(snap.get("num_trades", 0)),
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return df

    df["spread_ticks"] = df["spread_ticks"].round().astype(int)
    df["mid_move_half_tick"] = np.rint(df["mid"].diff().fillna(0.0) / (TICK_SIZE / 2.0)).astype(int)
    df["bid_move_ticks"] = np.rint(df["best_bid"].diff().fillna(0.0) / TICK_SIZE).astype(int)
    df["ask_move_ticks"] = np.rint(df["best_ask"].diff().fillna(0.0) / TICK_SIZE).astype(int)
    df["active_vol"] = df["buy_vol"] + df["sell_vol"]
    df["active_turnover"] = df["buy_turnover"] + df["sell_turnover"]
    df["net_active_vol"] = df["buy_vol"] - df["sell_vol"]
    df["trade_count_delta"] = df["num_trades"].diff().fillna(df["num_trades"]).clip(lower=0.0)

    for level in (1, 3, 5):
        df[f"l{level}_imbalance"] = [
            imbalance(bid_depth, ask_depth)
            for bid_depth, ask_depth in zip(df[f"bid_l{level}"], df[f"ask_l{level}"])
        ]
        df[f"l{level}_depth"] = df[f"bid_l{level}"] + df[f"ask_l{level}"]

    df["prev_spread_ticks"] = df["spread_ticks"].shift(1)
    df["is_spread12_entry"] = (df["prev_spread_ticks"] == 1) & (df["spread_ticks"] == 2)
    df["time_bucket_30m"] = df["timestamp"].map(bucket_30m)

    prev_bid = df["best_bid"].shift(1)
    prev_ask = df["best_ask"].shift(1)
    df["entry_mechanism"] = [
        classify_open(pb, pa, b, a) if pd.notna(pb) and pd.notna(pa) else "start_of_day"
        for pb, pa, b, a in zip(prev_bid, prev_ask, df["best_bid"], df["best_ask"])
    ]
    return df


def build_panel(dates: list[str]) -> pd.DataFrame:
    frames = [build_day_panel(trade_ymd) for trade_ymd in dates]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_episode_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for trade_ymd, day_df in panel.groupby("trade_ymd", sort=False):
        day_df = day_df.reset_index(drop=True)
        entry_positions = day_df.index[day_df["is_spread12_entry"]].tolist()
        for entry_pos in entry_positions:
            end_pos = entry_pos
            while end_pos + 1 < len(day_df) and int(day_df.loc[end_pos + 1, "spread_ticks"]) == 2:
                end_pos += 1

            prev_pos = entry_pos - 1
            close_pos = end_pos + 1
            if prev_pos < 0 or close_pos >= len(day_df):
                continue
            if int(day_df.loc[close_pos, "spread_ticks"]) != 1:
                continue

            prev_row = day_df.loc[prev_pos]
            entry_row = day_df.loc[entry_pos]
            end_row = day_df.loc[end_pos]
            close_row = day_df.loc[close_pos]

            exit_bid_move = int(round((close_row["best_bid"] - end_row["best_bid"]) / TICK_SIZE))
            exit_ask_move = int(round((close_row["best_ask"] - end_row["best_ask"]) / TICK_SIZE))
            direction = entry_direction(
                str(entry_row["entry_mechanism"]),
                float(entry_row["mid_move_half_tick"]),
            )
            close_mid_from_pre_entry_tick = float((close_row["mid"] - prev_row["mid"]) / TICK_SIZE)
            episode_mid_from_pre_entry_tick = float((end_row["mid"] - prev_row["mid"]) / TICK_SIZE)
            entry_mid_from_pre_entry_tick = float((entry_row["mid"] - prev_row["mid"]) / TICK_SIZE)
            signed_episode_mid_from_pre_entry_tick = (
                episode_mid_from_pre_entry_tick * direction if direction != 0 else math.nan
            )
            signed_close_mid_from_pre_entry_tick = (
                close_mid_from_pre_entry_tick * direction if direction != 0 else math.nan
            )

            rows.append(
                {
                    "trade_ymd": trade_ymd,
                    "entry_time": str(entry_row["time_hms"]),
                    "close_time": str(close_row["time_hms"]),
                    "entry_time_bucket": str(entry_row["time_bucket_30m"]),
                    "duration_seconds": int(end_pos - entry_pos + 1),
                    "entry_mechanism": str(entry_row["entry_mechanism"]),
                    "entry_direction": int(direction),
                    "entry_mid_move_half_tick": float(entry_row["mid_move_half_tick"]),
                    "entry_mid_from_pre_entry_tick": entry_mid_from_pre_entry_tick,
                    "entry_l1_imbalance": float(entry_row["l1_imbalance"]),
                    "entry_l3_imbalance": float(entry_row["l3_imbalance"]),
                    "entry_l5_imbalance": float(entry_row["l5_imbalance"]),
                    "entry_l1_depth": float(entry_row["l1_depth"]),
                    "entry_l5_depth": float(entry_row["l5_depth"]),
                    "entry_active_turnover": float(entry_row["active_turnover"]),
                    "episode_active_turnover": float(day_df.loc[entry_pos:end_pos, "active_turnover"].sum()),
                    "episode_net_active_vol": float(day_df.loc[entry_pos:end_pos, "net_active_vol"].sum()),
                    "episode_trade_count_delta": float(day_df.loc[entry_pos:end_pos, "trade_count_delta"].sum()),
                    "episode_mid_from_pre_entry_tick": episode_mid_from_pre_entry_tick,
                    "signed_episode_mid_from_pre_entry_tick": signed_episode_mid_from_pre_entry_tick,
                    "close_mid_from_pre_entry_tick": close_mid_from_pre_entry_tick,
                    "signed_close_mid_from_pre_entry_tick": signed_close_mid_from_pre_entry_tick,
                    "close_mid_from_entry_tick": float((close_row["mid"] - entry_row["mid"]) / TICK_SIZE),
                    "exit_mechanism": classify_close(
                        float(end_row["best_bid"]),
                        float(end_row["best_ask"]),
                        float(close_row["best_bid"]),
                        float(close_row["best_ask"]),
                    ),
                    "exit_bid_move_ticks": exit_bid_move,
                    "exit_ask_move_ticks": exit_ask_move,
                    "exit_mid_move_half_tick": float(close_row["mid_move_half_tick"]),
                    "resolution_path": resolution_path(direction, exit_bid_move, exit_ask_move),
                    "price_path_label": price_path_label(direction, close_mid_from_pre_entry_tick),
                }
            )

    return pd.DataFrame(rows)


def build_resolution_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    summary = (
        episodes.groupby("resolution_path", observed=False)
        .agg(
            samples=("trade_ymd", "size"),
            trade_days=("trade_ymd", "nunique"),
            avg_duration_seconds=("duration_seconds", "mean"),
            median_duration_seconds=("duration_seconds", "median"),
            avg_entry_l1_imbalance=("entry_l1_imbalance", "mean"),
            avg_entry_l1_depth=("entry_l1_depth", "mean"),
            avg_close_mid_from_pre_entry_tick=("close_mid_from_pre_entry_tick", "mean"),
            avg_signed_close_mid_from_pre_entry_tick=("signed_close_mid_from_pre_entry_tick", "mean"),
            median_close_mid_from_pre_entry_tick=("close_mid_from_pre_entry_tick", "median"),
            continue_ratio=("price_path_label", lambda x: float((x == "continue").mean())),
            reverse_ratio=("price_path_label", lambda x: float((x == "reverse").mean())),
            flat_ratio=("price_path_label", lambda x: float((x == "flat").mean())),
        )
        .reset_index()
    )
    summary["sample_share"] = summary["samples"] / summary["samples"].sum()
    return summary.sort_values("samples", ascending=False)


def build_mechanism_table(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    table = (
        episodes.groupby(["entry_mechanism", "resolution_path"], observed=False)
        .agg(
            samples=("trade_ymd", "size"),
            avg_duration_seconds=("duration_seconds", "mean"),
            avg_close_mid_from_pre_entry_tick=("close_mid_from_pre_entry_tick", "mean"),
            avg_signed_close_mid_from_pre_entry_tick=("signed_close_mid_from_pre_entry_tick", "mean"),
            continue_ratio=("price_path_label", lambda x: float((x == "continue").mean())),
            reverse_ratio=("price_path_label", lambda x: float((x == "reverse").mean())),
        )
        .reset_index()
    )
    total = table.groupby("entry_mechanism")["samples"].transform("sum")
    table["within_entry_share"] = np.where(total > 0, table["samples"] / total, np.nan)
    return table.sort_values(["entry_mechanism", "samples"], ascending=[True, False])


def build_exit_table(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    table = (
        episodes.groupby("exit_mechanism", observed=False)
        .agg(
            samples=("trade_ymd", "size"),
            avg_duration_seconds=("duration_seconds", "mean"),
            avg_close_mid_from_pre_entry_tick=("close_mid_from_pre_entry_tick", "mean"),
            avg_signed_close_mid_from_pre_entry_tick=("signed_close_mid_from_pre_entry_tick", "mean"),
            continue_ratio=("price_path_label", lambda x: float((x == "continue").mean())),
            reverse_ratio=("price_path_label", lambda x: float((x == "reverse").mean())),
        )
        .reset_index()
    )
    table["sample_share"] = table["samples"] / table["samples"].sum()
    return table.sort_values("samples", ascending=False)


def build_event_study(panel: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or episodes.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for trade_ymd, day_df in panel.groupby("trade_ymd", sort=False):
        day_df = day_df.reset_index(drop=True)
        day_events = episodes[episodes["trade_ymd"] == trade_ymd]
        if day_events.empty:
            continue

        pos_lookup = {time_hms: idx for idx, time_hms in enumerate(day_df["time_hms"])}
        for _, event in day_events.iterrows():
            close_pos = pos_lookup.get(str(event["close_time"]))
            if close_pos is None:
                continue
            for rel in range(-WINDOW_BEFORE, WINDOW_AFTER + 1):
                idx = close_pos + rel
                if idx < 0 or idx >= len(day_df):
                    continue
                row = day_df.loc[idx]
                rows.append(
                    {
                        "resolution_path": str(event["resolution_path"]),
                        "price_path_label": str(event["price_path_label"]),
                        "relative_second": rel,
                        "spread_ticks": float(row["spread_ticks"]),
                        "mid_move_half_tick": float(row["mid_move_half_tick"]),
                        "l1_imbalance": float(row["l1_imbalance"]),
                        "l1_depth": float(row["l1_depth"]),
                        "active_turnover": float(row["active_turnover"]),
                    }
                )

    event_df = pd.DataFrame(rows)
    if event_df.empty:
        return event_df

    return (
        event_df.groupby(["resolution_path", "relative_second"], observed=False)
        .agg(
            samples=("spread_ticks", "size"),
            spread_ticks_mean=("spread_ticks", "mean"),
            mid_move_half_tick_mean=("mid_move_half_tick", "mean"),
            l1_imbalance_mean=("l1_imbalance", "mean"),
            l1_depth_mean=("l1_depth", "mean"),
            active_turnover_mean=("active_turnover", "mean"),
        )
        .reset_index()
    )


def save_table(df: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def main() -> None:
    dates = valid_dates()
    if not dates:
        raise RuntimeError(f"No snapshot data found for {INSTRUMENT_ID}")

    panel = build_panel(dates)
    if panel.empty:
        raise RuntimeError(f"No regular-session rows built for {INSTRUMENT_ID}")

    episodes = build_episode_table(panel)
    if episodes.empty:
        raise RuntimeError("No spread 1->2->1 episodes found")

    overview = pd.DataFrame(
        [
            {
                "trade_days": int(panel["trade_ymd"].nunique()),
                "panel_rows": int(len(panel)),
                "spread12_entries": int(panel["is_spread12_entry"].sum()),
                "close_to_1_episodes": int(len(episodes)),
                "same_second_close_ratio": float((episodes["duration_seconds"] == 1).mean()),
                "rollback_close_ratio": float(episodes["resolution_path"].isin(["rollback_close", "rollback_dominant"]).mean()),
                "advance_close_ratio": float(episodes["resolution_path"].isin(["advance_close", "advance_dominant"]).mean()),
                "continue_ratio": float((episodes["price_path_label"] == "continue").mean()),
                "reverse_ratio": float((episodes["price_path_label"] == "reverse").mean()),
            }
        ]
    )

    resolution_summary = build_resolution_summary(episodes)
    mechanism_table = build_mechanism_table(episodes)
    exit_table = build_exit_table(episodes)
    event_study = build_event_study(panel, episodes)

    save_table(overview, "overview")
    save_table(resolution_summary, "resolution_summary")
    save_table(mechanism_table, "resolution_by_entry_mechanism")
    save_table(exit_table, "exit_mechanism_summary")
    save_table(event_study, "close_event_study")
    save_table(episodes, "episode_samples")

    print(f"valid_days={len(dates)}")
    print(f"panel_rows={len(panel)}")
    print(f"spread12_entries={int(panel['is_spread12_entry'].sum())}")
    print(f"close_to_1_episodes={len(episodes)}")
    print()
    print("=== resolution summary ===")
    print(resolution_summary.to_string(index=False))
    print()
    print("=== exit mechanism summary ===")
    print(exit_table.to_string(index=False))
    print()
    print(f"saved_to={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
