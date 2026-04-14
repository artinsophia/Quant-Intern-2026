from collections import deque
import itertools
import os
import pandas as pd
from typing import Dict, Any

from .features import create_feature, latest_zscore


class StrategyDemo:
    def __init__(self, model, param_dict=None) -> None:
        if param_dict is None:
            param_dict = {}
        self.__dict__.update(param_dict)

        data_file = f"/home/jovyan/work/backtest_result/{self.instrument_id}_{self.trade_ymd}_{self.name}.pkl"
        try:
            if os.path.exists(data_file):
                os.remove(data_file)
        except OSError as e:
            print(f"Warning: Could not delete file {data_file}: {e}")

        self.position_last = 0
        self.model = model

        self.feature_buffer = deque(maxlen=self.x_window)
        self.delta_buffer = deque(maxlen=self.x_window)

        self.trailing_stop_pct = param_dict.get("trailing_stop_pct", 0.002)
        self.max_favorable_price = 0

        self.prev_signal = 0

    def on_snap(self, snap: Dict[str, Any]) -> None:
        price = snap.get("price_last")
        if not price:
            return

        delta = sum(vol for _, vol in snap["buy_trade"][: self.standard_num]) - sum(
            vol for _, vol in snap["sell_trade"][: self.standard_num]
        )
        self.delta_buffer.append(delta)
        if len(self.delta_buffer) < self.x_window:
            return

        recent_delta = list(
            itertools.islice(
                self.delta_buffer,
                max(0, len(self.delta_buffer) - self.short_window),
                None,
            )
        )
        std_delta = latest_zscore(recent_delta)

        self.feature_buffer.append(snap)
        if len(self.feature_buffer) == self.x_window:
            feat_dict = create_feature(self.feature_buffer, self.short_window)
            features_df = pd.DataFrame([feat_dict])
            proba = self.model.predict_proba(features_df)
            # 获取类别1的概率，兼容DataFrame和numpy数组
            if hasattr(proba, "iloc"):
                prob = proba.iloc[0, 1]  # DataFrame
            else:
                prob = proba[0, 1]  # numpy数组
        else:
            return

        dynamic_pct = self.trailing_stop_pct * (1 + abs(std_delta) * self.k_pct)

        dynamic_pct = max(self.trailing_stop_pct, min(dynamic_pct, 4*self.trailing_stop_pct))

        current_signal = self.prev_signal

        if self.position_last == 1:
            self.max_favorable_price = max(self.max_favorable_price, price)
            pullback = (self.max_favorable_price - price) / self.max_favorable_price
            if pullback >= dynamic_pct:
                if prob is not None and prob > self.model.best_threshold + self.close_confidence and std_delta > self.open_threshold:
                    current_signal = 1
                else:
                    current_signal = 0

        elif self.position_last == -1:
            self.max_favorable_price = min(self.max_favorable_price, price)
            pullback = (price - self.max_favorable_price) / self.max_favorable_price
            if pullback >= dynamic_pct:
                if prob is not None and prob > self.model.best_threshold + self.close_confidence and std_delta < -self.open_threshold:
                    current_signal = -1
                else:
                    current_signal = 0

        if self.position_last == 0:
            if std_delta > self.open_threshold:
                current_signal = 1
            elif std_delta < -self.open_threshold:
                current_signal = -1

        if current_signal != self.prev_signal:
            if current_signal == 0:
                self.position_last = 0
                self.prev_signal = 0
                self.max_favorable_price = 0
            else:
                if prob is not None and prob > self.model.best_threshold + self.open_confidence:
                    self.position_last = current_signal
                    self.prev_signal = current_signal
                    self.max_favorable_price = price
