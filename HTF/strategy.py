from typing import Any, Dict
from collections import deque
import numpy as np


class StrategyDemo:
    def __init__(self, model=None, param_dict=None) -> None:
        if param_dict is None:
            param_dict = {}

        self.__dict__.update(param_dict)

        self.price_buffer = deque(maxlen=self.price_window)
        self.imbalance_buffer = deque(maxlen=self.imbalance_window)

        self.position_last = 0

    def on_snap(self, snap: Dict[str, Any]) -> None:
        if snap["bid_book"] is None or snap["ask_book"] is None:
            return

        if len(snap["bid_book"]) == 0 or len(snap["ask_book"]) == 0:
            return

        best_bid_price, best_bid_volume = snap["bid_book"][0]
        best_ask_price, best_ask_volume = snap["ask_book"][0]

        if best_bid_volume + best_ask_volume <= 0:
            return

        mid_price = (best_bid_price + best_ask_price) / 2

        microprice = (
            best_ask_price * best_bid_volume
            + best_bid_price * best_ask_volume
        ) / (best_bid_volume + best_ask_volume)

        microprice_alpha = (microprice - mid_price) / mid_price

        self.price_buffer.append(microprice_alpha)

        bid_volume = sum(level[1] for level in snap["bid_book"][:self.depth])
        ask_volume = sum(level[1] for level in snap["ask_book"][:self.depth])

        if bid_volume + ask_volume <= 0:
            return

        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        self.imbalance_buffer.append(imbalance)

        if len(self.price_buffer) < self.price_window:
            return

        if len(self.imbalance_buffer) < self.imbalance_window:
            return

        alpha = (

        )


        self.trade_logic(alpha)

    def trade_logic(self, alpha: float) -> None:
        if self.position_last == 0:
            if alpha > self.open_threshold:
                self.position_last = 1
            elif alpha < -self.open_threshold:
                self.position_last = -1

        elif self.position_last > 0:
            if alpha < self.close_threshold:
                self.position_last = 0

        elif self.position_last < 0:
            if alpha > -self.close_threshold:
                self.position_last = 0
