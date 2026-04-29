from collections import deque
from typing import Any, Dict, Optional

import numpy as np


def trade_volume_sum(trades: Any) -> float:
    if not isinstance(trades, (list, tuple)):
        return 0.0

    total = 0.0
    for row in trades:
        if isinstance(row, (list, tuple)):
            if len(row) >= 2:
                vol = row[1]
            elif len(row) == 1:
                vol = row[0]
            else:
                continue
        elif isinstance(row, (int, float)):
            vol = row
        else:
            continue

        try:
            total += float(vol)
        except (TypeError, ValueError):
            continue
    return total


class RollingFactorExtractor:
    def __init__(self, param_dict: Optional[Dict[str, Any]] = None) -> None:
        if param_dict is None:
            param_dict = {}
        self.__dict__.update(param_dict)

        self.window_size = int(getattr(self, "window_size", 60))
        self.vol_floor = float(getattr(self, "vol_floor", 1e-6))
        self.factor_clip = float(getattr(self, "factor_clip", 5.0))

        self.alpha_buffer = deque(maxlen=self.window_size)
        self.factor_buffer = deque(maxlen=self.window_size)

    def close(self) -> None:
        self.alpha_buffer.clear()
        self.factor_buffer.clear()

    @staticmethod
    def _calc_imbalance(snap: Dict[str, Any]) -> float:
        buy_volume = trade_volume_sum(snap.get("buy_trade"))
        sell_volume = trade_volume_sum(snap.get("sell_trade"))
        total_volume = buy_volume + sell_volume
        if total_volume <= 0:
            return 0.0
        return float((buy_volume - sell_volume) / total_volume)

    def on_snap(self, snap: Dict[str, Any]) -> Optional[float]:
        alpha = self._calc_imbalance(snap)
        self.alpha_buffer.append(alpha)
        if len(self.alpha_buffer) < self.window_size:
            return None

        alpha_std = max(float(np.std(self.alpha_buffer)), self.vol_floor)
        alpha_centered = alpha - float(np.mean(self.alpha_buffer))
        factor = (alpha - alpha_centered) / alpha_std


        return float(factor)
