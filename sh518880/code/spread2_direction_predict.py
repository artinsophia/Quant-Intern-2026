from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
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
OUTPUT_DIR = ROOT_DIR / "data" / "spread2_direction"

FEATURES = [
    "l1_imbalance",
    "l3_imbalance",
    "l5_imbalance",
    "l1_depth",
    "l3_depth",
    "l5_depth",
    "depth_skew_l1",
    "depth_skew_l3",
    "depth_skew_l5",
    "active_vol",
    "active_turnover",
    "net_active_vol",
    "net_active_turnover",
    "active_vol_imbalance",
    "trade_count_delta",
    "bid_l1_change",
    "ask_l1_change",
    "bid_l3_change",
    "ask_l3_change",
    "bid_l5_change",
    "ask_l5_change",
    "l1_imbalance_change",
    "l3_imbalance_change",
    "l5_imbalance_change",
    "active_vol_imbalance_change",
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


def build_day_panel(trade_ymd: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
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
    df["active_vol"] = df["buy_vol"] + df["sell_vol"]
    df["active_turnover"] = df["buy_turnover"] + df["sell_turnover"]
    df["net_active_vol"] = df["buy_vol"] - df["sell_vol"]
    df["net_active_turnover"] = df["buy_turnover"] - df["sell_turnover"]
    df["active_vol_imbalance"] = np.where(
        df["active_vol"] > 0,
        df["net_active_vol"] / df["active_vol"],
        0.0,
    )
    df["trade_count_delta"] = df["num_trades"].diff().fillna(df["num_trades"]).clip(lower=0.0)

    for level in (1, 3, 5):
        df[f"l{level}_imbalance"] = [
            imbalance(bid_depth, ask_depth)
            for bid_depth, ask_depth in zip(df[f"bid_l{level}"], df[f"ask_l{level}"])
        ]
        df[f"l{level}_depth"] = df[f"bid_l{level}"] + df[f"ask_l{level}"]
        df[f"depth_skew_l{level}"] = df[f"bid_l{level}"] - df[f"ask_l{level}"]
        df[f"bid_l{level}_change"] = df[f"bid_l{level}"].diff().fillna(0.0)
        df[f"ask_l{level}_change"] = df[f"ask_l{level}"].diff().fillna(0.0)
        df[f"l{level}_imbalance_change"] = df[f"l{level}_imbalance"].diff().fillna(0.0)

    df["active_vol_imbalance_change"] = df["active_vol_imbalance"].diff().fillna(0.0)
    df["prev_spread_ticks"] = df["spread_ticks"].shift(1)
    df["is_spread2_entry"] = (df["spread_ticks"] == 2) & (df["prev_spread_ticks"] != 2)
    return df


def build_panel(dates: list[str]) -> pd.DataFrame:
    frames = [build_day_panel(trade_ymd) for trade_ymd in dates]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel["next_move_half_tick"] = math.nan
    panel["seconds_to_next_move"] = math.nan

    for _, idx in panel.groupby("trade_ymd").groups.items():
        day_df = panel.loc[list(idx)].reset_index()
        next_move = np.full(len(day_df), np.nan)
        wait_time = np.full(len(day_df), np.nan)
        next_nonzero = None
        next_direction = math.nan
        for pos in range(len(day_df) - 1, -1, -1):
            curr_move = day_df.loc[pos, "mid_move_half_tick"]
            if next_nonzero is not None:
                wait_time[pos] = float(next_nonzero - pos)
                next_move[pos] = next_direction
            if curr_move != 0:
                next_nonzero = pos
                next_direction = float(curr_move)
        panel.loc[day_df["index"], "next_move_half_tick"] = next_move
        panel.loc[day_df["index"], "seconds_to_next_move"] = wait_time

    panel["next_up"] = (panel["next_move_half_tick"] > 0).astype(float)
    return panel


def single_factor_auc(feature: pd.Series, label: pd.Series) -> float:
    valid = feature.notna() & label.notna()
    if valid.sum() < 50 or label[valid].nunique() < 2:
        return math.nan
    return float(roc_auc_score(label[valid], feature[valid]))


def evaluate_subset(df: pd.DataFrame, label_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        [
            {
                "label": label_name,
                "samples": int(len(df)),
                "trade_days": int(df["trade_ymd"].nunique()),
                "up_ratio": float(df["next_up"].mean()),
                "median_seconds_to_next_move": float(df["seconds_to_next_move"].median()),
                "avg_seconds_to_next_move": float(df["seconds_to_next_move"].mean()),
            }
        ]
    )

    single_rows: list[dict[str, float | str]] = []
    for feature in FEATURES:
        auc = single_factor_auc(df[feature], df["next_up"])
        single_rows.append(
            {
                "label": label_name,
                "feature": feature,
                "auc": auc,
                "edge_over_random": abs(auc - 0.5) if pd.notna(auc) else math.nan,
                "corr_with_label": float(df[feature].corr(df["next_up"])),
            }
        )
    single_table = pd.DataFrame(single_rows).sort_values(
        ["edge_over_random", "auc"], ascending=[False, False]
    )

    unique_days = sorted(df["trade_ymd"].unique())
    split_idx = max(1, int(len(unique_days) * 0.7))
    train_days = set(unique_days[:split_idx])
    test_days = set(unique_days[split_idx:])
    train_df = df[df["trade_ymd"].isin(train_days)].copy()
    test_df = df[df["trade_ymd"].isin(test_days)].copy()

    model_table = pd.DataFrame()
    if len(test_df) > 0 and train_df["next_up"].nunique() == 2 and test_df["next_up"].nunique() == 2:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        )
        pipeline.fit(train_df[FEATURES], train_df["next_up"])
        test_prob = pipeline.predict_proba(test_df[FEATURES])[:, 1]
        test_pred = (test_prob >= 0.5).astype(float)

        summary.loc[:, "train_days"] = len(train_days)
        summary.loc[:, "test_days"] = len(test_days)
        summary.loc[:, "train_samples"] = len(train_df)
        summary.loc[:, "test_samples"] = len(test_df)
        summary.loc[:, "test_auc"] = float(roc_auc_score(test_df["next_up"], test_prob))
        summary.loc[:, "test_accuracy"] = float(accuracy_score(test_df["next_up"], test_pred))
        summary.loc[:, "test_up_ratio"] = float(test_df["next_up"].mean())
        summary.loc[:, "test_avg_pred_prob"] = float(test_prob.mean())

        clf = pipeline.named_steps["clf"]
        coefs = clf.coef_[0]
        model_table = pd.DataFrame({"feature": FEATURES, "coef": coefs}).sort_values(
            "coef", ascending=False
        )

    return summary, single_table, model_table


def build_bucket_table(df: pd.DataFrame, feature: str, buckets: int = 10) -> pd.DataFrame:
    temp = df[[feature, "next_up", "seconds_to_next_move"]].dropna().copy()
    temp["bucket"] = pd.qcut(temp[feature], q=buckets, duplicates="drop")
    table = (
        temp.groupby("bucket", observed=False)
        .agg(
            samples=(feature, "size"),
            feature_mean=(feature, "mean"),
            next_up_ratio=("next_up", "mean"),
            median_wait=("seconds_to_next_move", "median"),
        )
        .reset_index()
    )
    table.insert(0, "feature", feature)
    return table


def save_table(df: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def main() -> None:
    dates = valid_dates()
    if not dates:
        raise RuntimeError(f"No snapshot data found for {INSTRUMENT_ID}")

    panel = build_panel(dates)
    spread2 = panel[(panel["spread_ticks"] == 2) & panel["next_move_half_tick"].notna()].copy()
    spread2_entry = spread2[spread2["is_spread2_entry"]].copy()

    all_summary, all_single, all_model = evaluate_subset(spread2, "spread2_all")
    entry_summary, entry_single, entry_model = evaluate_subset(spread2_entry, "spread2_entry")

    bucket_tables = pd.concat(
        [
            build_bucket_table(spread2, "l1_imbalance"),
            build_bucket_table(spread2, "l3_imbalance"),
            build_bucket_table(spread2, "l5_imbalance"),
            build_bucket_table(spread2_entry, "l1_imbalance").assign(label="spread2_entry"),
        ],
        ignore_index=True,
    )

    overview = pd.DataFrame(
        [
            {
                "trade_days": int(panel["trade_ymd"].nunique()),
                "rows": int(len(panel)),
                "spread2_rows": int(len(spread2)),
                "spread2_ratio": float((panel["spread_ticks"] == 2).mean()),
                "spread2_entry_rows": int(len(spread2_entry)),
                "spread2_entry_up_ratio": float(spread2_entry["next_up"].mean()),
                "spread2_all_up_ratio": float(spread2["next_up"].mean()),
            }
        ]
    )

    save_table(overview, "overview")
    save_table(pd.concat([all_summary, entry_summary], ignore_index=True), "summary")
    save_table(all_single, "single_factor_auc_all")
    save_table(entry_single, "single_factor_auc_entry")
    save_table(all_model, "logit_coef_all")
    save_table(entry_model, "logit_coef_entry")
    save_table(bucket_tables, "imbalance_bucket_view")

    print(f"valid_days={len(dates)}")
    print(f"panel_rows={len(panel)}")
    print(f"spread2_rows={len(spread2)}")
    print(f"spread2_entry_rows={len(spread2_entry)}")
    print()
    print("=== spread2_all summary ===")
    print(all_summary.to_string(index=False))
    print()
    print("=== spread2_entry summary ===")
    print(entry_summary.to_string(index=False))
    print()
    print("=== top single factors: spread2_all ===")
    print(all_single.head(10).to_string(index=False))
    print()
    print("=== top single factors: spread2_entry ===")
    print(entry_single.head(10).to_string(index=False))
    print()
    print(f"saved_to={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
