from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spread2_close_path import (
    OUTPUT_DIR as CLOSE_PATH_OUTPUT_DIR,
    TICK_SIZE,
    build_episode_table,
    build_panel,
    valid_dates,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "spread2_recovery_drift"
TIME_WINDOWS = (1, 2, 3, 5, 10, 20, 30, 60)
MOVE_COUNTS = (1, 2, 3, 4, 5, 8, 10)
FOCUS_PATHS = ("advance_close", "rollback_close")
EVENT_WINDOW_BEFORE = 10
EVENT_WINDOW_AFTER = 30
DRIFT_QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def load_or_build_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    episode_path = CLOSE_PATH_OUTPUT_DIR / "episode_samples.csv"
    if episode_path.exists():
        episodes = pd.read_csv(episode_path)
    else:
        episodes = build_episode_table(panel)

    if episodes.empty:
        raise RuntimeError("No spread 1->2->1 episodes found")
    return episodes[episodes["resolution_path"].isin(FOCUS_PATHS)].copy()


def _build_day_context(panel: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict[str, int]]]:
    day_context: dict[str, tuple[pd.DataFrame, dict[str, int]]] = {}
    for trade_ymd, day_df in panel.groupby("trade_ymd", sort=False):
        day_df = day_df.reset_index(drop=True).copy()
        day_df["nonzero_mid_move"] = day_df["mid_move_half_tick"] != 0
        pos_lookup = {str(time_hms): idx for idx, time_hms in enumerate(day_df["time_hms"])}
        day_context[str(trade_ymd)] = (day_df, pos_lookup)
    return day_context


def build_time_window_samples(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    horizons: tuple[int, ...] = TIME_WINDOWS,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    day_context = _build_day_context(panel)

    for event in episodes.itertuples(index=False):
        trade_ymd = str(event.trade_ymd)
        context = day_context.get(trade_ymd)
        if context is None:
            continue
        day_df, pos_lookup = context
        close_pos = pos_lookup.get(str(event.close_time))
        if close_pos is None:
            continue

        close_mid = float(day_df.loc[close_pos, "mid"])
        entry_direction = int(event.entry_direction)

        for horizon in horizons:
            future_pos = close_pos + horizon
            if future_pos >= len(day_df):
                continue

            future_mid = float(day_df.loc[future_pos, "mid"])
            drift_tick = (future_mid - close_mid) / TICK_SIZE
            signed_drift_tick = drift_tick * entry_direction if entry_direction != 0 else np.nan
            rows.append(
                {
                    "trade_ymd": trade_ymd,
                    "entry_time": str(event.entry_time),
                    "close_time": str(event.close_time),
                    "resolution_path": str(event.resolution_path),
                    "entry_direction": entry_direction,
                    "horizon_type": "time_seconds",
                    "horizon_value": int(horizon),
                    "elapsed_seconds": int(horizon),
                    "post_close_drift_tick": float(drift_tick),
                    "signed_post_close_drift_tick": float(signed_drift_tick),
                }
            )

    return pd.DataFrame(rows)


def build_move_count_samples(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    move_counts: tuple[int, ...] = MOVE_COUNTS,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    day_context = _build_day_context(panel)

    for event in episodes.itertuples(index=False):
        trade_ymd = str(event.trade_ymd)
        context = day_context.get(trade_ymd)
        if context is None:
            continue
        day_df, pos_lookup = context
        close_pos = pos_lookup.get(str(event.close_time))
        if close_pos is None:
            continue

        close_mid = float(day_df.loc[close_pos, "mid"])
        entry_direction = int(event.entry_direction)
        move_positions = day_df.index[(day_df.index > close_pos) & day_df["nonzero_mid_move"]].tolist()

        for move_count in move_counts:
            if len(move_positions) < move_count:
                continue

            future_pos = move_positions[move_count - 1]
            future_mid = float(day_df.loc[future_pos, "mid"])
            drift_tick = (future_mid - close_mid) / TICK_SIZE
            signed_drift_tick = drift_tick * entry_direction if entry_direction != 0 else np.nan
            rows.append(
                {
                    "trade_ymd": trade_ymd,
                    "entry_time": str(event.entry_time),
                    "close_time": str(event.close_time),
                    "resolution_path": str(event.resolution_path),
                    "entry_direction": entry_direction,
                    "horizon_type": "mid_moves",
                    "horizon_value": int(move_count),
                    "elapsed_seconds": int(future_pos - close_pos),
                    "post_close_drift_tick": float(drift_tick),
                    "signed_post_close_drift_tick": float(signed_drift_tick),
                }
            )

    return pd.DataFrame(rows)


def summarize_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()

    def pos_ratio(series: pd.Series) -> float:
        return float((series > 0).mean())

    def neg_ratio(series: pd.Series) -> float:
        return float((series < 0).mean())

    def zero_ratio(series: pd.Series) -> float:
        return float((series == 0).mean())

    summary = (
        samples.groupby(["horizon_type", "horizon_value", "resolution_path"], observed=False)
        .agg(
            samples=("trade_ymd", "size"),
            trade_days=("trade_ymd", "nunique"),
            avg_elapsed_seconds=("elapsed_seconds", "mean"),
            median_elapsed_seconds=("elapsed_seconds", "median"),
            mean_drift_tick=("post_close_drift_tick", "mean"),
            median_drift_tick=("post_close_drift_tick", "median"),
            mean_signed_drift_tick=("signed_post_close_drift_tick", "mean"),
            median_signed_drift_tick=("signed_post_close_drift_tick", "median"),
            continue_ratio=("signed_post_close_drift_tick", pos_ratio),
            reverse_ratio=("signed_post_close_drift_tick", neg_ratio),
            flat_ratio=("signed_post_close_drift_tick", zero_ratio),
        )
        .reset_index()
    )

    pivot = summary.pivot(
        index=["horizon_type", "horizon_value"],
        columns="resolution_path",
        values="mean_signed_drift_tick",
    )
    if set(FOCUS_PATHS).issubset(pivot.columns):
        edge = (pivot["rollback_close"] - pivot["advance_close"]).rename("rollback_minus_advance")
        summary = summary.merge(edge.reset_index(), on=["horizon_type", "horizon_value"], how="left")

    return summary.sort_values(["horizon_type", "horizon_value", "resolution_path"]).reset_index(drop=True)


def summarize_drift_quantiles(
    samples: pd.DataFrame,
    quantiles: tuple[float, ...] = DRIFT_QUANTILES,
) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    group_cols = ["horizon_type", "horizon_value", "resolution_path"]

    for group_key, group_df in samples.groupby(group_cols, observed=False):
        horizon_type, horizon_value, resolution_path = group_key
        signed = group_df["signed_post_close_drift_tick"].dropna()
        if signed.empty:
            continue

        row: dict[str, float | int | str] = {
            "horizon_type": str(horizon_type),
            "horizon_value": int(horizon_value),
            "resolution_path": str(resolution_path),
            "samples": int(len(signed)),
            "mean_signed_drift_tick": float(signed.mean()),
            "std_signed_drift_tick": float(signed.std(ddof=0)),
            "min_signed_drift_tick": float(signed.min()),
            "max_signed_drift_tick": float(signed.max()),
        }
        for q in quantiles:
            label = f"q{int(round(q * 100)):02d}_signed_drift_tick"
            row[label] = float(signed.quantile(q))
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["horizon_type", "horizon_value", "resolution_path"])
        .reset_index(drop=True)
    )


def summarize_drift_value_distribution(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()

    group_cols = ["horizon_type", "horizon_value", "resolution_path"]
    dist_source = samples.dropna(subset=["signed_post_close_drift_tick"]).copy()
    # Drift lives on a half-tick grid; rounding avoids float-key splits like 1.0 vs 0.9999999998.
    dist_source["signed_post_close_drift_tick"] = dist_source["signed_post_close_drift_tick"].round(6)
    dist = (
        dist_source
        .groupby(group_cols + ["signed_post_close_drift_tick"], observed=False)
        .agg(samples=("trade_ymd", "size"))
        .reset_index()
    )
    totals = (
        dist.groupby(group_cols, observed=False)["samples"]
        .sum()
        .rename("group_samples")
        .reset_index()
    )
    dist = dist.merge(totals, on=group_cols, how="left")
    dist["sample_ratio"] = dist["samples"] / dist["group_samples"]

    return dist.sort_values(
        ["horizon_type", "horizon_value", "resolution_path", "signed_post_close_drift_tick"]
    ).reset_index(drop=True)


def build_event_window_samples(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    window_before: int = EVENT_WINDOW_BEFORE,
    window_after: int = EVENT_WINDOW_AFTER,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    day_context = _build_day_context(panel)

    for event in episodes.itertuples(index=False):
        trade_ymd = str(event.trade_ymd)
        context = day_context.get(trade_ymd)
        if context is None:
            continue
        day_df, pos_lookup = context
        close_pos = pos_lookup.get(str(event.close_time))
        if close_pos is None:
            continue

        close_mid = float(day_df.loc[close_pos, "mid"])
        entry_direction = int(event.entry_direction)

        for rel in range(-window_before, window_after + 1):
            idx = close_pos + rel
            if idx < 0 or idx >= len(day_df):
                continue
            row = day_df.loc[idx]
            drift_tick = (float(row["mid"]) - close_mid) / TICK_SIZE
            signed_drift_tick = drift_tick * entry_direction if entry_direction != 0 else np.nan
            rows.append(
                {
                    "trade_ymd": trade_ymd,
                    "entry_time": str(event.entry_time),
                    "close_time": str(event.close_time),
                    "resolution_path": str(event.resolution_path),
                    "entry_direction": entry_direction,
                    "relative_second": int(rel),
                    "spread_ticks": float(row["spread_ticks"]),
                    "mid_move_half_tick": float(row["mid_move_half_tick"]),
                    "mid_drift_from_close_tick": float(drift_tick),
                    "signed_mid_drift_from_close_tick": float(signed_drift_tick),
                    "l1_imbalance": float(row["l1_imbalance"]),
                    "l1_depth": float(row["l1_depth"]),
                    "active_turnover": float(row["active_turnover"]),
                    "is_nonzero_mid_move": int(bool(row["nonzero_mid_move"])),
                }
            )

    return pd.DataFrame(rows)


def summarize_event_window(event_samples: pd.DataFrame) -> pd.DataFrame:
    if event_samples.empty:
        return pd.DataFrame()

    return (
        event_samples.groupby(["resolution_path", "relative_second"], observed=False)
        .agg(
            samples=("trade_ymd", "size"),
            spread_ticks_mean=("spread_ticks", "mean"),
            mid_move_half_tick_mean=("mid_move_half_tick", "mean"),
            mean_signed_mid_drift_tick=("signed_mid_drift_from_close_tick", "mean"),
            median_signed_mid_drift_tick=("signed_mid_drift_from_close_tick", "median"),
            l1_imbalance_mean=("l1_imbalance", "mean"),
            l1_depth_mean=("l1_depth", "mean"),
            active_turnover_mean=("active_turnover", "mean"),
            nonzero_mid_move_ratio=("is_nonzero_mid_move", "mean"),
        )
        .reset_index()
        .sort_values(["resolution_path", "relative_second"])
        .reset_index(drop=True)
    )


def save_table(df: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def main() -> None:
    dates = valid_dates()
    if not dates:
        raise RuntimeError("No snapshot data found for 518880")

    panel = build_panel(dates)
    if panel.empty:
        raise RuntimeError("No regular-session panel rows built for 518880")

    episodes = load_or_build_episodes(panel)
    time_samples = build_time_window_samples(panel, episodes)
    move_samples = build_move_count_samples(panel, episodes)
    all_samples = pd.concat([time_samples, move_samples], ignore_index=True)
    summary = summarize_samples(all_samples)
    quantile_summary = summarize_drift_quantiles(all_samples)
    value_distribution = summarize_drift_value_distribution(all_samples)
    event_samples = build_event_window_samples(panel, episodes)
    event_summary = summarize_event_window(event_samples)

    overview = pd.DataFrame(
        [
            {
                "trade_days": int(panel["trade_ymd"].nunique()),
                "panel_rows": int(len(panel)),
                "episodes_total": int(len(episodes)),
                "advance_close_samples": int((episodes["resolution_path"] == "advance_close").sum()),
                "rollback_close_samples": int((episodes["resolution_path"] == "rollback_close").sum()),
            }
        ]
    )

    save_table(overview, "overview")
    save_table(summary, "drift_summary")
    save_table(quantile_summary, "drift_quantile_summary")
    save_table(value_distribution, "drift_value_distribution")
    save_table(all_samples, "drift_samples")
    save_table(event_summary, "event_window_summary")
    save_table(event_samples, "event_window_samples")

    print(overview.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print()
    print(quantile_summary.to_string(index=False))
    print()
    print(event_summary.head(20).to_string(index=False))
    print()
    print(f"saved_to={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
