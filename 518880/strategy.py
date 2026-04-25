from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pickle

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from features import create_feature, get_mid_price


class StrategyDemo:
    def __init__(self, model, param_dict=None) -> None:
        artifact = model
        if isinstance(model, (str, Path)):
            with open(model, "rb") as f:
                artifact = pickle.load(f)

        artifact_param_dict = {}
        if isinstance(artifact, dict):
            artifact_param_dict = dict(artifact.get("param_dict") or {})

        merged_param_dict = dict(artifact_param_dict)
        merged_param_dict.update(dict(param_dict or {}))
        self.__dict__.update(merged_param_dict)

        self.x_window = int(getattr(self, "x_window", 60))
        self.stall_seconds = int(getattr(self, "stall_seconds", 5))
        self.horizon_seconds = int(getattr(self, "horizon_seconds", 10))
        self.target_ticks = int(getattr(self, "target_ticks", 2))
        self.tick_size = float(getattr(self, "tick_size", 0.001))
        self.prob_threshold = float(getattr(self, "prob_threshold", 0.6))
        self.name = getattr(self, "name", "mid_stall_v1")

        self.model = artifact["model"] if isinstance(artifact, dict) else artifact
        self.feature_names = (
            artifact.get("feature_names", getattr(self, "feature_names", None))
            if isinstance(artifact, dict)
            else getattr(self, "feature_names", None)
        )
        self.model_threshold = (
            float(artifact.get("threshold", self.prob_threshold))
            if isinstance(artifact, dict)
            else self.prob_threshold
        )

        self.position_last = 0
        self.snap_buffer: deque[dict[str, Any]] = deque(maxlen=self.x_window)
        self.last_mid = None
        self.current_stall = 0
        self.stall_ready = False

        self.entry_mid = None
        self.holding_seconds = 0

    def _predict_up_probability(self) -> float | None:
        if len(self.snap_buffer) < self.x_window:
            return None

        feature_dict = create_feature(list(self.snap_buffer), tick_size=self.tick_size)
        if self.feature_names is None:
            self.feature_names = list(feature_dict.keys())

        X = np.array([[feature_dict[name] for name in self.feature_names]], dtype=float)
        proba = self.model.predict_proba(X)
        return float(proba[0, 1])

    def _update_stall_state(self, current_mid: float | None) -> None:
        if current_mid is None:
            self.current_stall = 0
            self.stall_ready = False
            self.last_mid = None
            return

        if self.last_mid is None or current_mid != self.last_mid:
            self.current_stall = 1
            self.stall_ready = True
        else:
            self.current_stall += 1

        self.last_mid = current_mid

    def _should_trigger(self) -> bool:
        return self.stall_ready and self.current_stall >= self.stall_seconds

    def _check_exit(self, current_mid: float | None) -> bool:
        if self.position_last == 0 or current_mid is None or self.entry_mid is None:
            return False

        self.holding_seconds += 1
        move_ticks = (current_mid - self.entry_mid) / self.tick_size

        if self.position_last == 1 and move_ticks >= self.target_ticks:
            return True
        if self.position_last == -1 and move_ticks <= -self.target_ticks:
            return True
        if self.position_last == 1 and move_ticks <= -self.target_ticks:
            return True
        if self.position_last == -1 and move_ticks >= self.target_ticks:
            return True
        if self.holding_seconds >= self.horizon_seconds:
            return True
        return False

    def _close_position(self) -> None:
        self.position_last = 0
        self.entry_mid = None
        self.holding_seconds = 0

    def on_snap(self, snap: dict[str, Any]) -> None:
        current_mid = get_mid_price(snap)
        self.snap_buffer.append(snap)
        self._update_stall_state(current_mid)

        if self._check_exit(current_mid):
            self._close_position()

        if self.position_last != 0:
            return

        if len(self.snap_buffer) < self.x_window:
            return
        if not self._should_trigger():
            return

        prob_up = self._predict_up_probability()
        self.stall_ready = False
        if prob_up is None or current_mid is None:
            return

        if prob_up >= self.model_threshold:
            self.position_last = 1
            self.entry_mid = current_mid
            self.holding_seconds = 0
        elif prob_up <= 1.0 - self.model_threshold:
            self.position_last = -1
            self.entry_mid = current_mid
            self.holding_seconds = 0
