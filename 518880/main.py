from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

for path in [CURRENT_DIR, ROOT_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from data_processing import get_valid_trade_dates, load_snaps, samples_from_dates, split_dates
from strategy import StrategyDemo
from tools.backtest_quick import backtest_quick

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


DEFAULT_PARAMS: dict[str, Any] = {
    "instrument_id": "518880",
    "name": "mid_stall_v1",
    "tick_size": 0.001,
    "x_window": 60,
    "stall_seconds": 5,
    "horizon_seconds": 10,
    "target_ticks": 2,
    "stride": 1,
    "allow_no_touch": False,
    "prob_threshold": 0.6,
    "start_ymd": "20251201",
    "end_ymd": "20260425",
    "model_type": "xgboost",
}


def build_model(model_type: str):
    if model_type == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.6
    best_score = -1.0
    for threshold in np.arange(0.5, 0.81, 0.02):
        long_mask = y_prob >= threshold
        short_mask = y_prob <= 1.0 - threshold
        decided_mask = long_mask | short_mask
        if decided_mask.sum() == 0:
            continue

        y_pred = np.where(long_mask[decided_mask], 1, 0)
        score = accuracy_score(y_true[decided_mask], y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_dataset(name: str, model, X: np.ndarray, y: np.ndarray, threshold: float) -> None:
    if len(y) == 0:
        print(f"{name}: 空数据集")
        return

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    decided_mask = (y_prob >= threshold) | (y_prob <= 1.0 - threshold)
    decided_ratio = float(np.mean(decided_mask))

    print(f"\n{name} 评估")
    print(f"samples={len(y)}")
    print(f"up_ratio={np.mean(y == 1):.4f}")
    print(f"accuracy@0.5={accuracy_score(y, y_pred):.4f}")
    print(f"decided_ratio@{threshold:.2f}={decided_ratio:.4f}")

    if decided_mask.any():
        decided_pred = np.where(y_prob[decided_mask] >= threshold, 1, 0)
        decided_true = y[decided_mask]
        print(f"accuracy@decided={accuracy_score(decided_true, decided_pred):.4f}")
        print(classification_report(decided_true, decided_pred, digits=4))


def run_strategy_for_day(
    trade_ymd: str,
    artifact: dict[str, Any],
    param_dict: dict[str, Any],
) -> Any:
    snap_list = load_snaps(param_dict["instrument_id"], trade_ymd)
    if not snap_list:
        return None

    strategy = StrategyDemo(artifact, param_dict)
    position_dict = {}
    for snap in snap_list:
        strategy.on_snap(snap)
        position_dict[int(snap["time_mark"])] = int(strategy.position_last)

    return backtest_quick(
        instrument_id=param_dict["instrument_id"],
        trade_ymd=trade_ymd,
        strategy_name=param_dict["name"],
        position_dict=position_dict,
        remake=True,
    )


def main(param_overrides: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(DEFAULT_PARAMS)
    if param_overrides:
        params.update(param_overrides)

    trade_dates = get_valid_trade_dates(
        instrument_id=params["instrument_id"],
        start_ymd=params["start_ymd"],
        end_ymd=params["end_ymd"],
    )
    if len(trade_dates) < 3:
        raise RuntimeError("有效交易日不足，无法训练")

    train_dates, valid_dates, test_dates = split_dates(trade_dates)
    print(f"train_dates={train_dates[0]}~{train_dates[-1]} ({len(train_dates)})")
    print(f"valid_dates={valid_dates[0]}~{valid_dates[-1]} ({len(valid_dates)})")
    print(f"test_dates={test_dates[0]}~{test_dates[-1]} ({len(test_dates)})")

    X_train, y_train, feature_names = samples_from_dates(train_dates, params["instrument_id"], params)
    X_valid, y_valid, _ = samples_from_dates(valid_dates, params["instrument_id"], params)
    X_test, y_test, _ = samples_from_dates(test_dates, params["instrument_id"], params)

    if len(y_train) == 0 or len(y_valid) == 0:
        raise RuntimeError("训练集或验证集为空，当前参数下没有足够样本")

    model = build_model(params["model_type"])
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    threshold = find_best_threshold(y_valid, valid_prob)
    params["prob_threshold"] = threshold

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "threshold": threshold,
        "param_dict": params,
    }

    output_dir = CURRENT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{params['name']}_artifact.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nmodel_saved_to={artifact_path}")
    print(f"selected_threshold={threshold:.2f}")

    evaluate_dataset("train", model, X_train, y_train, threshold)
    evaluate_dataset("valid", model, X_valid, y_valid, threshold)
    evaluate_dataset("test", model, X_test, y_test, threshold)

    backtest_summaries: dict[str, float] = {}
    for trade_ymd in test_dates[-3:]:
        result_df = run_strategy_for_day(trade_ymd, artifact, params)
        if result_df is None or len(result_df) == 0:
            continue
        backtest_summaries[trade_ymd] = float(result_df["profits"].iloc[-1])
        print(f"backtest {trade_ymd}: pnl={backtest_summaries[trade_ymd]:.4f}")

    return artifact, backtest_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="518880 mid stall strategy trainer")
    parser.add_argument("--stall-seconds", type=int, default=DEFAULT_PARAMS["stall_seconds"])
    parser.add_argument("--horizon-seconds", type=int, default=DEFAULT_PARAMS["horizon_seconds"])
    parser.add_argument("--target-ticks", type=int, default=DEFAULT_PARAMS["target_ticks"])
    parser.add_argument("--x-window", type=int, default=DEFAULT_PARAMS["x_window"])
    parser.add_argument("--start-ymd", type=str, default=DEFAULT_PARAMS["start_ymd"])
    parser.add_argument("--end-ymd", type=str, default=DEFAULT_PARAMS["end_ymd"])
    parser.add_argument("--model-type", type=str, default=DEFAULT_PARAMS["model_type"])
    args = parser.parse_args()

    main(
        {
            "stall_seconds": args.stall_seconds,
            "horizon_seconds": args.horizon_seconds,
            "target_ticks": args.target_ticks,
            "x_window": args.x_window,
            "start_ymd": args.start_ymd,
            "end_ymd": args.end_ymd,
            "model_type": args.model_type,
        }
    )
