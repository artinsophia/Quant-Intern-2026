from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from features import create_feature, get_mid_price

BASE_DEMO_CANDIDATES = [
    Path("/home/jovyan/work/base_demo"),
    Path("/home/jovyan/base_demo"),
]

for base_path in BASE_DEMO_CANDIDATES:
    if base_path.exists() and str(base_path) not in sys.path:
        sys.path.append(str(base_path))

import base_tool  # type: ignore


def candidate_dates(start_ymd: str, end_ymd: str) -> list[str]:
    return [
        day.strftime("%Y%m%d")
        for day in pd.date_range(start=start_ymd, end=end_ymd, freq="B")
    ]


def load_snaps(instrument_id: str, trade_ymd: str) -> list[dict[str, Any]]:
    with redirect_stdout(io.StringIO()):
        snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
    return snap_list if isinstance(snap_list, list) else []


def get_valid_trade_dates(
    instrument_id: str,
    start_ymd: str,
    end_ymd: str,
) -> list[str]:
    dates = candidate_dates(start_ymd, end_ymd)
    valid_dates: list[str] = []
    for trade_ymd in dates:
        if load_snaps(instrument_id, trade_ymd):
            valid_dates.append(trade_ymd)
    return valid_dates


def split_dates(
    trade_dates: list[str],
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
) -> tuple[list[str], list[str], list[str]]:
    if len(trade_dates) < 3:
        raise ValueError("trade_dates 至少需要 3 天，才能切分 train/valid/test")

    train_end = max(1, int(len(trade_dates) * train_ratio))
    valid_end = max(train_end + 1, int(len(trade_dates) * (train_ratio + valid_ratio)))
    valid_end = min(valid_end, len(trade_dates) - 1)

    train_dates = trade_dates[:train_end]
    valid_dates = trade_dates[train_end:valid_end]
    test_dates = trade_dates[valid_end:]

    if not valid_dates:
        valid_dates = train_dates[-1:]
        train_dates = train_dates[:-1]
    if not test_dates:
        test_dates = valid_dates[-1:]
        valid_dates = valid_dates[:-1]

    return train_dates, valid_dates, test_dates


def is_stall_event(
    snap_list: list[dict[str, Any]],
    idx: int,
    stall_seconds: int,
) -> bool:
    if stall_seconds <= 0 or idx < stall_seconds - 1:
        return False

    current_mid = get_mid_price(snap_list[idx])
    if current_mid is None:
        return False

    start_idx = idx - stall_seconds + 1
    mids = [get_mid_price(snap_list[pos]) for pos in range(start_idx, idx + 1)]
    if any(mid is None for mid in mids):
        return False
    if any(mid != current_mid for mid in mids):
        return False

    if start_idx == 0:
        return True

    prev_mid = get_mid_price(snap_list[start_idx - 1])
    return prev_mid != current_mid


def create_label(
    future_snaps: list[dict[str, Any]],
    entry_mid: float,
    tick_size: float,
    target_ticks: int,
    allow_no_touch: bool = False,
) -> int | None:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")

    upper_barrier = entry_mid + target_ticks * tick_size
    lower_barrier = entry_mid - target_ticks * tick_size

    for snap in future_snaps:
        future_mid = get_mid_price(snap)
        if future_mid is None:
            continue
        if future_mid >= upper_barrier:
            return 1
        if future_mid <= lower_barrier:
            return 0

    if allow_no_touch:
        return 0
    return None


class TrainValidTest:
    def __init__(self, snap_list, param_dict):
        self.snap_list = list(snap_list)
        self.param_dict = dict(param_dict or {})
        self.instrument_id = self.param_dict["instrument_id"]
        self.x_window = int(self.param_dict.get("x_window", 60))
        self.stall_seconds = int(self.param_dict.get("stall_seconds", 5))
        self.horizon_seconds = int(self.param_dict.get("horizon_seconds", 10))
        self.target_ticks = int(self.param_dict.get("target_ticks", 2))
        self.tick_size = float(self.param_dict.get("tick_size", 0.001))
        self.stride = int(self.param_dict.get("stride", 1))
        self.allow_no_touch = bool(self.param_dict.get("allow_no_touch", False))

    def samples(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        feature_rows: list[dict[str, float]] = []
        labels: list[int] = []
        n = len(self.snap_list)

        for idx in range(self.x_window - 1, n - self.horizon_seconds, self.stride):
            if not is_stall_event(self.snap_list, idx, self.stall_seconds):
                continue

            feature_slice = self.snap_list[idx - self.x_window + 1 : idx + 1]
            entry_mid = get_mid_price(self.snap_list[idx])
            if entry_mid is None:
                continue

            future_snaps = self.snap_list[idx + 1 : idx + 1 + self.horizon_seconds]
            label = create_label(
                future_snaps=future_snaps,
                entry_mid=entry_mid,
                tick_size=self.tick_size,
                target_ticks=self.target_ticks,
                allow_no_touch=self.allow_no_touch,
            )
            if label is None:
                continue

            feature_rows.append(create_feature(feature_slice, tick_size=self.tick_size))
            labels.append(label)

        if not feature_rows:
            return np.array([]), np.array([]), []

        feature_names = list(feature_rows[0].keys())
        X = np.array([[row[name] for name in feature_names] for row in feature_rows])
        y = np.array(labels, dtype=int)
        return X, y, feature_names


def samples_from_dates(
    dates: list[str],
    instrument_id: str,
    param_dict: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    feature_names: list[str] = []

    for trade_ymd in dates:
        try:
            snap_list = load_snaps(instrument_id, trade_ymd)
            if not snap_list:
                print(f"{trade_ymd}: 无 snapshot，跳过")
                continue

            tvt = TrainValidTest(snap_list, param_dict)
            X_day, y_day, feature_names = tvt.samples()
            if len(y_day) == 0:
                print(f"{trade_ymd}: 无有效 stall 样本")
                continue

            X_all.append(X_day)
            y_all.append(y_day)
            pos_ratio = float(np.mean(y_day == 1))
            print(f"{trade_ymd}: 样本={len(y_day)}, up_ratio={pos_ratio:.3f}")
        except Exception as exc:
            print(f"{trade_ymd}: 处理失败 - {exc}")

    if not X_all:
        return np.array([]), np.array([]), feature_names

    return np.vstack(X_all), np.concatenate(y_all), feature_names
