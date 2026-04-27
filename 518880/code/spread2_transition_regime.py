from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
OUTPUT_DIR = ROOT_DIR / "data" / "spread2_transition_regime"
WINDOW_BEFORE = 5
WINDOW_AFTER = 10
CLUSTER_FEATURES = [
    "entry_l1_imbalance",
    "entry_l3_imbalance",
    "entry_l5_imbalance",
    "entry_l1_depth",
    "entry_l5_depth",
    "duration_seconds",
    "pre_active_turnover_3s",
    "pre_net_active_vol_3s",
    "episode_active_turnover",
    "episode_net_active_vol",
    "episode_abs_mid_move_half_tick_mean",
    "entry_side_code",
]


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


def classify_transition(prev_bid: float, prev_ask: float, bid: float, ask: float) -> str:
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


def entry_side_code(mechanism: str) -> float:
    if mechanism == "ask_retreat":
        return 1.0
    if mechanism == "bid_retreat":
        return -1.0
    if mechanism == "both_widen":
        return 0.0
    return 0.5


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
    df["net_active_turnover"] = df["buy_turnover"] - df["sell_turnover"]
    df["active_vol_imbalance"] = np.where(df["active_vol"] > 0, df["net_active_vol"] / df["active_vol"], 0.0)
    df["trade_count_delta"] = df["num_trades"].diff().fillna(df["num_trades"]).clip(lower=0.0)

    for level in (1, 3, 5):
        df[f"l{level}_imbalance"] = [
            imbalance(bid_depth, ask_depth)
            for bid_depth, ask_depth in zip(df[f"bid_l{level}"], df[f"ask_l{level}"])
        ]
        df[f"l{level}_depth"] = df[f"bid_l{level}"] + df[f"ask_l{level}"]
        df[f"bid_l{level}_change"] = df[f"bid_l{level}"].diff().fillna(0.0)
        df[f"ask_l{level}_change"] = df[f"ask_l{level}"].diff().fillna(0.0)

    df["prev_spread_ticks"] = df["spread_ticks"].shift(1)
    df["is_spread12_entry"] = (df["prev_spread_ticks"] == 1) & (df["spread_ticks"] == 2)
    df["is_spread2_entry"] = (df["prev_spread_ticks"] != 2) & (df["spread_ticks"] == 2)

    prev_bid = df["best_bid"].shift(1)
    prev_ask = df["best_ask"].shift(1)
    df["transition_mechanism"] = [
        classify_transition(pb, pa, b, a) if pd.notna(pb) and pd.notna(pa) else "start_of_day"
        for pb, pa, b, a in zip(prev_bid, prev_ask, df["best_bid"], df["best_ask"])
    ]
    df["time_bucket_30m"] = df["timestamp"].map(bucket_30m)

    return df


def build_panel(dates: list[str]) -> pd.DataFrame:
    frames = [build_day_panel(trade_ymd) for trade_ymd in dates]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_transition_summary(panel: pd.DataFrame) -> pd.DataFrame:
    entries = panel[panel["is_spread12_entry"]].copy()
    if entries.empty:
        return pd.DataFrame()

    summary = (
        entries.groupby("transition_mechanism", observed=False)
        .agg(
            samples=("trade_ymd", "size"),
            trade_days=("trade_ymd", "nunique"),
            avg_entry_l1_imbalance=("l1_imbalance", "mean"),
            avg_entry_l3_imbalance=("l3_imbalance", "mean"),
            avg_entry_l1_depth=("l1_depth", "mean"),
            avg_entry_l5_depth=("l5_depth", "mean"),
            avg_active_turnover=("active_turnover", "mean"),
            avg_net_active_vol=("net_active_vol", "mean"),
            avg_trade_count_delta=("trade_count_delta", "mean"),
            avg_mid_move_half_tick=("mid_move_half_tick", "mean"),
        )
        .reset_index()
    )
    summary["sample_share"] = summary["samples"] / summary["samples"].sum()
    return summary.sort_values("samples", ascending=False)


def build_transition_by_time(panel: pd.DataFrame) -> pd.DataFrame:
    entries = panel[panel["is_spread12_entry"]].copy()
    if entries.empty:
        return pd.DataFrame()
    table = (
        entries.groupby(["time_bucket_30m", "transition_mechanism"], observed=False)
        .agg(
            samples=("trade_ymd", "size"),
            avg_entry_l1_imbalance=("l1_imbalance", "mean"),
            avg_entry_l1_depth=("l1_depth", "mean"),
            avg_active_turnover=("active_turnover", "mean"),
        )
        .reset_index()
    )
    total = table.groupby("time_bucket_30m")["samples"].transform("sum")
    table["bucket_share"] = np.where(total > 0, table["samples"] / total, np.nan)
    return table


def build_event_study(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for trade_ymd, day_df in panel.groupby("trade_ymd", sort=False):
        day_df = day_df.reset_index(drop=True)
        entry_positions = day_df.index[day_df["is_spread12_entry"]].tolist()
        for pos in entry_positions:
            mechanism = str(day_df.loc[pos, "transition_mechanism"])
            for rel in range(-WINDOW_BEFORE, WINDOW_AFTER + 1):
                idx = pos + rel
                if idx < 0 or idx >= len(day_df):
                    continue
                row = day_df.loc[idx]
                rows.append(
                    {
                        "trade_ymd": trade_ymd,
                        "entry_pos": pos,
                        "transition_mechanism": mechanism,
                        "relative_second": rel,
                        "spread_ticks": float(row["spread_ticks"]),
                        "mid_move_half_tick": float(row["mid_move_half_tick"]),
                        "l1_imbalance": float(row["l1_imbalance"]),
                        "l1_depth": float(row["l1_depth"]),
                        "l5_depth": float(row["l5_depth"]),
                        "active_turnover": float(row["active_turnover"]),
                        "net_active_vol": float(row["net_active_vol"]),
                        "trade_count_delta": float(row["trade_count_delta"]),
                    }
                )

    event_df = pd.DataFrame(rows)
    if event_df.empty:
        return event_df

    return (
        event_df.groupby(["transition_mechanism", "relative_second"], observed=False)
        .agg(
            samples=("trade_ymd", "size"),
            spread_ticks_mean=("spread_ticks", "mean"),
            mid_move_half_tick_mean=("mid_move_half_tick", "mean"),
            l1_imbalance_mean=("l1_imbalance", "mean"),
            l1_depth_mean=("l1_depth", "mean"),
            l5_depth_mean=("l5_depth", "mean"),
            active_turnover_mean=("active_turnover", "mean"),
            net_active_vol_mean=("net_active_vol", "mean"),
            trade_count_delta_mean=("trade_count_delta", "mean"),
        )
        .reset_index()
    )


def compute_episode_ids(panel: pd.DataFrame) -> pd.Series:
    episode_ids = pd.Series(index=panel.index, dtype="object")
    for trade_ymd, idx in panel.groupby("trade_ymd").groups.items():
        day_idx = list(idx)
        spread2 = panel.loc[day_idx, "spread_ticks"] == 2
        starts = spread2 & ~spread2.shift(1, fill_value=False)
        start_counter = 0
        current_episode = None
        for local_pos, (global_idx, is_spread2, is_start) in enumerate(zip(day_idx, spread2, starts)):
            if not is_spread2:
                current_episode = None
                continue
            if is_start:
                start_counter += 1
                current_episode = f"{trade_ymd}_{start_counter:05d}"
            episode_ids.loc[global_idx] = current_episode
    return episode_ids


def build_episode_table(panel: pd.DataFrame) -> pd.DataFrame:
    spread2_panel = panel[panel["spread_ticks"] == 2].copy()
    if spread2_panel.empty:
        return pd.DataFrame()

    panel = panel.copy()
    panel["spread2_episode_id"] = compute_episode_ids(panel)
    spread2_panel = panel[panel["spread_ticks"] == 2].copy()

    rows: list[dict[str, float | int | str]] = []
    for episode_id, episode_df in spread2_panel.groupby("spread2_episode_id", sort=False):
        episode_df = episode_df.reset_index()
        start_global_idx = int(episode_df.loc[0, "index"])
        end_global_idx = int(episode_df.loc[len(episode_df) - 1, "index"])
        start_row = panel.loc[start_global_idx]
        prev_rows = panel.iloc[max(0, start_global_idx - 3) : start_global_idx]
        if len(prev_rows) and prev_rows["trade_ymd"].nunique() > 1:
            prev_rows = prev_rows[prev_rows["trade_ymd"] == start_row["trade_ymd"]]

        next_row = None
        if end_global_idx + 1 < len(panel):
            candidate_next = panel.iloc[end_global_idx + 1]
            if candidate_next["trade_ymd"] == start_row["trade_ymd"]:
                next_row = candidate_next

        exit_ret_ticks = float((episode_df.iloc[-1]["mid"] - episode_df.iloc[0]["mid"]) / TICK_SIZE)
        rows.append(
            {
                "spread2_episode_id": episode_id,
                "trade_ymd": str(start_row["trade_ymd"]),
                "entry_time": str(start_row["time_hms"]),
                "entry_time_bucket": str(start_row["time_bucket_30m"]),
                "duration_seconds": int(len(episode_df)),
                "transition_mechanism": str(start_row["transition_mechanism"]),
                "entry_side_code": entry_side_code(str(start_row["transition_mechanism"])),
                "entry_l1_imbalance": float(start_row["l1_imbalance"]),
                "entry_l3_imbalance": float(start_row["l3_imbalance"]),
                "entry_l5_imbalance": float(start_row["l5_imbalance"]),
                "entry_l1_depth": float(start_row["l1_depth"]),
                "entry_l5_depth": float(start_row["l5_depth"]),
                "pre_active_turnover_3s": float(prev_rows["active_turnover"].sum()) if len(prev_rows) else 0.0,
                "pre_net_active_vol_3s": float(prev_rows["net_active_vol"].sum()) if len(prev_rows) else 0.0,
                "pre_trade_count_delta_3s": float(prev_rows["trade_count_delta"].sum()) if len(prev_rows) else 0.0,
                "episode_active_turnover": float(episode_df["active_turnover"].sum()),
                "episode_net_active_vol": float(episode_df["net_active_vol"].sum()),
                "episode_trade_count_delta": float(episode_df["trade_count_delta"].sum()),
                "episode_l1_imbalance_mean": float(episode_df["l1_imbalance"].mean()),
                "episode_l1_depth_mean": float(episode_df["l1_depth"].mean()),
                "episode_l5_depth_mean": float(episode_df["l5_depth"].mean()),
                "episode_abs_mid_move_half_tick_mean": float(episode_df["mid_move_half_tick"].abs().mean()),
                "episode_mid_ret_tick": exit_ret_ticks,
                "episode_mid_ret_abs_tick": abs(exit_ret_ticks),
                "episode_up_ratio": float((episode_df["mid_move_half_tick"] > 0).mean()),
                "episode_down_ratio": float((episode_df["mid_move_half_tick"] < 0).mean()),
                "exit_next_spread": float(next_row["spread_ticks"]) if next_row is not None else math.nan,
                "exit_next_mid_move_half_tick": float(next_row["mid_move_half_tick"]) if next_row is not None else math.nan,
                "exit_next_mechanism": str(next_row["transition_mechanism"]) if next_row is not None else "end_of_day",
            }
        )

    return pd.DataFrame(rows)


def label_cluster(center: pd.Series, mechanism: str) -> str:
    if center["duration_seconds"] <= 2.5:
        duration_tag = "short"
    elif center["duration_seconds"] >= 6:
        duration_tag = "long"
    else:
        duration_tag = "medium"

    if center["entry_l1_imbalance"] >= 0.2:
        pressure_tag = "bid_pressure"
    elif center["entry_l1_imbalance"] <= -0.2:
        pressure_tag = "ask_pressure"
    else:
        pressure_tag = "balanced"

    if center["entry_l1_depth"] <= 20000:
        depth_tag = "thin"
    elif center["entry_l1_depth"] >= 60000:
        depth_tag = "thick"
    else:
        depth_tag = "middepth"

    return f"{mechanism}_{pressure_tag}_{depth_tag}_{duration_tag}"


def build_episode_clusters(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if episodes.empty or len(episodes) < 100:
        empty_summary = pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_label",
                "samples",
                "share",
                "duration_mean",
                "duration_median",
                "entry_l1_imbalance_mean",
                "entry_l1_depth_mean",
                "episode_active_turnover_mean",
                "episode_mid_ret_abs_tick_mean",
                "up_exit_ratio",
                "down_exit_ratio",
            ]
        )
        return episodes.assign(cluster_id=math.nan, cluster_label="insufficient_samples"), empty_summary

    n_clusters = 4 if len(episodes) >= 400 else 3
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=7, n_init=20)),
        ]
    )
    episodes = episodes.copy()
    episodes["cluster_id"] = pipe.fit_predict(episodes[CLUSTER_FEATURES])

    centers_scaled = pipe.named_steps["cluster"].cluster_centers_
    scaler = pipe.named_steps["scaler"]
    centers = pd.DataFrame(
        scaler.inverse_transform(centers_scaled),
        columns=CLUSTER_FEATURES,
    )
    centers.insert(0, "cluster_id", range(n_clusters))

    cluster_summary = (
        episodes.groupby("cluster_id", observed=False)
        .agg(
            samples=("spread2_episode_id", "size"),
            share=("spread2_episode_id", lambda x: len(x) / len(episodes)),
            duration_mean=("duration_seconds", "mean"),
            duration_median=("duration_seconds", "median"),
            entry_l1_imbalance_mean=("entry_l1_imbalance", "mean"),
            entry_l1_depth_mean=("entry_l1_depth", "mean"),
            episode_active_turnover_mean=("episode_active_turnover", "mean"),
            episode_mid_ret_abs_tick_mean=("episode_mid_ret_abs_tick", "mean"),
            up_exit_ratio=("exit_next_mid_move_half_tick", lambda x: float((x > 0).mean())),
            down_exit_ratio=("exit_next_mid_move_half_tick", lambda x: float((x < 0).mean())),
        )
        .reset_index()
    )

    mech_share = (
        episodes.pivot_table(
            index="cluster_id",
            columns="transition_mechanism",
            values="spread2_episode_id",
            aggfunc="count",
            fill_value=0,
        )
        .div(episodes.groupby("cluster_id")["spread2_episode_id"].count(), axis=0)
        .reset_index()
    )
    mech_share.columns = [
        col if col == "cluster_id" else f"mechanism_share_{col}"
        for col in mech_share.columns
    ]
    cluster_summary = cluster_summary.merge(mech_share, on="cluster_id", how="left")

    labels = []
    for _, center in centers.iterrows():
        cluster_id = int(center["cluster_id"])
        mech_cols = [col for col in cluster_summary.columns if col.startswith("mechanism_share_")]
        mech_row = cluster_summary.loc[cluster_summary["cluster_id"] == cluster_id, mech_cols]
        majority_mechanism = "mixed"
        if not mech_row.empty and len(mech_cols):
            majority_col = mech_row.iloc[0].astype(float).idxmax()
            majority_mechanism = majority_col.replace("mechanism_share_", "")
        labels.append(label_cluster(center, majority_mechanism))
    centers["cluster_label"] = labels
    cluster_summary = cluster_summary.merge(
        centers[["cluster_id", "cluster_label"]], on="cluster_id", how="left"
    )
    episodes = episodes.merge(centers[["cluster_id", "cluster_label"]], on="cluster_id", how="left")
    return episodes, cluster_summary.merge(centers, on=["cluster_id", "cluster_label"], how="left")


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

    transition_summary = build_transition_summary(panel)
    transition_by_time = build_transition_by_time(panel)
    event_study = build_event_study(panel)
    episodes = build_episode_table(panel)
    episodes, cluster_summary = build_episode_clusters(episodes)

    overview = pd.DataFrame(
        [
            {
                "trade_days": int(panel["trade_ymd"].nunique()),
                "panel_rows": int(len(panel)),
                "spread1_rows": int((panel["spread_ticks"] == 1).sum()),
                "spread2_rows": int((panel["spread_ticks"] == 2).sum()),
                "spread12_entries": int(panel["is_spread12_entry"].sum()),
                "spread2_episodes": int(episodes["spread2_episode_id"].nunique()) if not episodes.empty else 0,
                "spread2_ratio": float((panel["spread_ticks"] == 2).mean()),
            }
        ]
    )

    save_table(overview, "overview")
    save_table(transition_summary, "transition_summary")
    save_table(transition_by_time, "transition_by_time")
    save_table(event_study, "transition_event_study")
    save_table(episodes, "episode_samples")
    save_table(cluster_summary, "episode_cluster_summary")

    print(f"valid_days={len(dates)}")
    print(f"panel_rows={len(panel)}")
    print(f"spread12_entries={int(panel['is_spread12_entry'].sum())}")
    print(f"spread2_episodes={int(episodes['spread2_episode_id'].nunique()) if not episodes.empty else 0}")
    print()
    print("=== transition summary ===")
    print(transition_summary.to_string(index=False))
    if not cluster_summary.empty:
        print()
        print("=== spread2 cluster summary ===")
        display_cols = [
            "cluster_id",
            "cluster_label",
            "samples",
            "share",
            "duration_mean",
            "entry_l1_imbalance_mean",
            "entry_l1_depth_mean",
            "episode_mid_ret_abs_tick_mean",
        ]
        print(cluster_summary[display_cols].to_string(index=False))
    print()
    print(f"saved_to={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
