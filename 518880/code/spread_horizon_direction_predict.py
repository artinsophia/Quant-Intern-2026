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
HORIZONS = [5, 10, 30]
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "spread_horizon_direction"

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
    df["active_vol_imbalance"] = np.where(df["active_vol"] > 0, df["net_active_vol"] / df["active_vol"], 0.0)
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
    df["is_spread1_entry"] = (df["spread_ticks"] == 1) & (df["prev_spread_ticks"] != 1)
    df["is_spread2_entry"] = (df["spread_ticks"] == 2) & (df["prev_spread_ticks"] != 2)
    return df


def build_panel(dates: list[str]) -> pd.DataFrame:
    frames = [build_day_panel(trade_ymd) for trade_ymd in dates]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    for horizon in HORIZONS:
        lead_mid = panel.groupby("trade_ymd")["mid"].shift(-horizon)
        future_ret_tick = (lead_mid - panel["mid"]) / TICK_SIZE
        panel[f"mid_ret_tick_{horizon}s"] = future_ret_tick
        panel[f"mid_up_{horizon}s"] = (future_ret_tick > 0).astype(float)
        panel[f"mid_down_{horizon}s"] = (future_ret_tick < 0).astype(float)
        panel[f"mid_flat_{horizon}s"] = (future_ret_tick == 0).astype(float)
    return panel


def single_factor_auc(feature: pd.Series, label: pd.Series) -> float:
    valid = feature.notna() & label.notna()
    if valid.sum() < 50 or label[valid].nunique() < 2:
        return math.nan
    return float(roc_auc_score(label[valid], feature[valid]))


def evaluate_subset(df: pd.DataFrame, subset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float | int | str]] = []
    single_rows: list[dict[str, float | int | str]] = []
    model_rows: list[dict[str, float | int | str]] = []

    unique_days = sorted(df["trade_ymd"].dropna().unique())
    split_idx = max(1, int(len(unique_days) * 0.7))
    train_days = set(unique_days[:split_idx])
    test_days = set(unique_days[split_idx:])

    for horizon in HORIZONS:
        ret_col = f"mid_ret_tick_{horizon}s"
        horizon_df = df[df[ret_col].notna()].copy()
        if horizon_df.empty:
            continue

        sign_df = horizon_df[horizon_df[ret_col] != 0].copy()
        up_ratio_all = float((horizon_df[ret_col] > 0).mean())
        down_ratio_all = float((horizon_df[ret_col] < 0).mean())
        flat_ratio_all = float((horizon_df[ret_col] == 0).mean())

        row: dict[str, float | int | str] = {
            "subset": subset_name,
            "horizon_s": horizon,
            "samples_all": int(len(horizon_df)),
            "samples_nonzero": int(len(sign_df)),
            "trade_days": int(horizon_df["trade_ymd"].nunique()),
            "up_ratio_all": up_ratio_all,
            "down_ratio_all": down_ratio_all,
            "flat_ratio_all": flat_ratio_all,
            "up_ratio_nonzero": float((sign_df[ret_col] > 0).mean()) if len(sign_df) else math.nan,
            "avg_ret_tick": float(horizon_df[ret_col].mean()),
            "avg_abs_ret_tick": float(horizon_df[ret_col].abs().mean()),
        }

        if len(sign_df) >= 100 and sign_df["trade_ymd"].nunique() >= 2:
            sign_df["label_up"] = (sign_df[ret_col] > 0).astype(float)
            train_df = sign_df[sign_df["trade_ymd"].isin(train_days)].copy()
            test_df = sign_df[sign_df["trade_ymd"].isin(test_days)].copy()

            row["train_days"] = len(train_days)
            row["test_days"] = len(test_days)
            row["train_samples"] = len(train_df)
            row["test_samples"] = len(test_df)

            if (
                len(train_df) >= 100
                and len(test_df) >= 100
                and train_df["label_up"].nunique() == 2
                and test_df["label_up"].nunique() == 2
            ):
                pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=2000)),
                    ]
                )
                pipeline.fit(train_df[FEATURES], train_df["label_up"])
                test_prob = pipeline.predict_proba(test_df[FEATURES])[:, 1]
                test_pred = (test_prob >= 0.5).astype(float)

                row["test_auc"] = float(roc_auc_score(test_df["label_up"], test_prob))
                row["test_accuracy"] = float(accuracy_score(test_df["label_up"], test_pred))
                row["test_up_ratio_nonzero"] = float(test_df["label_up"].mean())
                row["test_avg_pred_prob"] = float(test_prob.mean())

                coef_df = pd.DataFrame(
                    {
                        "subset": subset_name,
                        "horizon_s": horizon,
                        "feature": FEATURES,
                        "coef": pipeline.named_steps["clf"].coef_[0],
                    }
                )
                model_rows.extend(coef_df.to_dict("records"))

            for feature in FEATURES:
                auc = single_factor_auc(sign_df[feature], sign_df["label_up"])
                single_rows.append(
                    {
                        "subset": subset_name,
                        "horizon_s": horizon,
                        "feature": feature,
                        "auc": auc,
                        "edge_over_random": abs(auc - 0.5) if pd.notna(auc) else math.nan,
                        "corr_with_label": float(sign_df[feature].corr(sign_df["label_up"])),
                    }
                )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    single = pd.DataFrame(single_rows)
    model = pd.DataFrame(model_rows)
    if not single.empty:
        single = single.sort_values(["subset", "horizon_s", "edge_over_random", "auc"], ascending=[True, True, False, False])
    if not model.empty:
        model = model.sort_values(["subset", "horizon_s", "coef"], ascending=[True, True, False])
    return summary, single, model


def build_bucket_view(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        ret_col = f"mid_ret_tick_{horizon}s"
        for feature in ["l1_imbalance", "l3_imbalance", "l5_imbalance"]:
            temp = df[[feature, ret_col]].dropna().copy()
            if temp.empty:
                continue
            temp["bucket"] = pd.qcut(temp[feature], q=10, duplicates="drop")
            table = (
                temp.groupby("bucket", observed=False)
                .agg(
                    samples=(feature, "size"),
                    feature_mean=(feature, "mean"),
                    up_ratio=(ret_col, lambda x: float((x > 0).mean())),
                    down_ratio=(ret_col, lambda x: float((x < 0).mean())),
                    flat_ratio=(ret_col, lambda x: float((x == 0).mean())),
                    mean_ret_tick=(ret_col, "mean"),
                )
                .reset_index()
            )
            table.insert(0, "subset", subset_name)
            table.insert(1, "horizon_s", horizon)
            table.insert(2, "feature", feature)
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_table(df: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def main() -> None:
    dates = valid_dates()
    if not dates:
        raise RuntimeError(f"No snapshot data found for {INSTRUMENT_ID}")

    panel = build_panel(dates)
    subsets = {
        "spread1_all": panel[panel["spread_ticks"] == 1].copy(),
        "spread1_entry": panel[panel["is_spread1_entry"]].copy(),
        "spread2_all": panel[panel["spread_ticks"] == 2].copy(),
        "spread2_entry": panel[panel["is_spread2_entry"]].copy(),
    }

    summary_list: list[pd.DataFrame] = []
    single_list: list[pd.DataFrame] = []
    model_list: list[pd.DataFrame] = []
    bucket_list: list[pd.DataFrame] = []

    overview_rows = []
    for subset_name, subset_df in subsets.items():
        overview_rows.append(
            {
                "subset": subset_name,
                "samples": int(len(subset_df)),
                "trade_days": int(subset_df["trade_ymd"].nunique()),
            }
        )
        summary, single, model = evaluate_subset(subset_df, subset_name)
        summary_list.append(summary)
        if not single.empty:
            single_list.append(single)
        if not model.empty:
            model_list.append(model)
        bucket_view = build_bucket_view(subset_df, subset_name)
        if not bucket_view.empty:
            bucket_list.append(bucket_view)

    overview = pd.DataFrame(overview_rows)
    summary = pd.concat(summary_list, ignore_index=True)
    single = pd.concat(single_list, ignore_index=True) if single_list else pd.DataFrame()
    model = pd.concat(model_list, ignore_index=True) if model_list else pd.DataFrame()
    bucket = pd.concat(bucket_list, ignore_index=True) if bucket_list else pd.DataFrame()

    save_table(overview, "overview")
    save_table(summary, "summary")
    save_table(single, "single_factor_auc")
    save_table(model, "logit_coef")
    save_table(bucket, "bucket_view")

    print(f"valid_days={len(dates)}")
    print(f"panel_rows={len(panel)}")
    print()
    print("=== summary ===")
    print(summary.to_string(index=False))
    if not single.empty:
        print()
        print("=== top factors by subset/horizon ===")
        for subset_name in subsets:
            for horizon in HORIZONS:
                top = single[(single["subset"] == subset_name) & (single["horizon_s"] == horizon)].head(5)
                if top.empty:
                    continue
                print(f"--- {subset_name} horizon={horizon}s ---")
                print(top.to_string(index=False))
                print()
    print(f"saved_to={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
