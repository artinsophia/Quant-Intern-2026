from collections import deque
import itertools
import os
from typing import Dict, Any
import joblib
from .features import create_feature, latest_zscore , FeatureExtractor
import numpy as np


class StrategyDemo:
    def __init__(self, model, param_dict=None) -> None:
        if param_dict is None:
            param_dict = {}
        self.__dict__.update(param_dict)

        self.position_last = 0

        # 检查model是否已经是加载的模型对象
        if isinstance(model, str):
            # 如果是文件路径，加载模型
            self.model = joblib.load(model)
        else:
            # 如果已经是模型对象，直接使用
            self.model = model

        self.feature_buffer = deque(maxlen=self.x_window)
        self.delta_buffer = deque(maxlen=self.x_window)
        self.price_buffer = deque(maxlen=self.vol_window)

        self.max_favorable_price = 0
        self.entry_price = 0
        self.prev_signal = 0
        self.holding_snap = 0

    def close(self):
        self.delta_buffer.clear()
        self.feature_buffer.clear()
        self.price_buffer.clear()

    def __del__(self):
        self.close()

    def on_snap(self, snap: Dict[str, Any]) -> None:
        price = snap.get("price_last")
        if not price:
            return
        self.price_buffer.append(price)
        if len(self.price_buffer) < self.vol_window:
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
            values = list(feat_dict.values())
            features = np.array([values])
            proba = self.model.predict_proba(features)
            # 获取类别1的概率，兼容DataFrame和numpy数组
            if hasattr(proba, "iloc"):
                prob = proba.iloc[0, 1]  # DataFrame
            else:
                prob = proba[0, 1]  # numpy数组
        else:
            return

        rolling_std = np.std(self.price_buffer)
        self.trailing_stop = rolling_std * self.atr_multiplier

        dynamic = self.trailing_stop
        current_signal = self.prev_signal


        # 目标止盈
        if self.position_last != 0:
            self.holding_snap += 1
            if self.holding_snap >= 1000:
                current_signal = 0

            if self.position_last == 1:
                if price - self.entry_price > self.k_up * rolling_std * self.gamma:
                    current_signal = 0
                if std_delta > self.reset_threshold and prob > self.model.best_threshold + self.reset_confidence:
                    self.holding_snap = 0
            elif self.position_last == -1:
                if self.entry_price - price > self.k_down * rolling_std * self.gamma:
                    current_signal = 0
                if std_delta < -self.reset_threshold and prob > self.model.best_threshold + self.reset_confidence:
                    self.holding_snap = 0



        # 回撤平仓或反向信号平仓
        if self.position_last == 1:
            self.max_favorable_price = max(self.max_favorable_price, price)
            pullback = self.max_favorable_price - price
            if pullback >= dynamic:
                current_signal = 0
            if std_delta < - self.close_threshold and prob > self.model.best_threshold + self.close_confidence:
                current_signal = 0

        elif self.position_last == -1:
            self.max_favorable_price = min(self.max_favorable_price, price)
            pullback = price - self.max_favorable_price
            if pullback >= dynamic:
                current_signal = 0
            if std_delta > self.close_threshold and prob > self.model.best_threshold + self.close_confidence:
                current_signal = 0
        
        # 开仓信号
        if self.position_last == 0:
            if std_delta > self.open_threshold:
                current_signal = 1
            elif std_delta < -self.open_threshold:
                current_signal = -1

        # 实际行为
        if current_signal != self.prev_signal:
            if current_signal == 0: # 平仓
                self.position_last = 0
                self.prev_signal = 0
                self.max_favorable_price = 0
                self.entry_price = 0
                self.holding_snap = 0
            else: # 开仓
                if (
                    prob is not None
                    and prob > self.model.best_threshold + self.open_confidence
                ):
                    self.position_last = current_signal
                    self.prev_signal = current_signal
                    self.max_favorable_price = price
                    self.entry_price = price
