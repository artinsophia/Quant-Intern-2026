import numpy as np
from typing import List, Dict, Any


class FeatureExtractor:
    def __init__(self, snap_slice: List[Dict[str, Any]], short_window: int = 60):
        if not snap_slice:
            raise ValueError("snap_slice cannot be empty")
        self.snap_slice = list(snap_slice)
        self.last = snap_slice[-1]
        self.bid_book = self.last.get("bid_book", [])
        self.ask_book = self.last.get("ask_book", [])
        self.short_window = short_window

        self.bid_volume = sum(
            vol for row in self.snap_slice for _, vol in row["buy_trade"]
        )
        self.ask_volume = sum(
            vol for row in self.snap_slice for _, vol in row["sell_trade"]
        )
        self.bid_volume_short = sum(
            vol
            for row in self.snap_slice[-self.short_window :]
            for _, vol in row["buy_trade"]
        )
        self.ask_volume_short = sum(
            vol
            for row in self.snap_slice[-self.short_window :]
            for _, vol in row["sell_trade"]
        )
        

    @staticmethod
    def _safe_get_level(book: List[tuple], idx: int = 0) -> tuple:
        if book and len(book) > idx:
            return book[idx]
        return (np.nan, 0)

    @property
    def best_bid(self) -> float:
        return self._safe_get_level(self.bid_book)[0]

    @property
    def best_ask(self) -> float:
        return self._safe_get_level(self.ask_book)[0]

    @property
    def spread(self) -> float:
        bid, ask = self.best_bid, self.best_ask
        if np.isnan(bid) or np.isnan(ask):
            return np.nan
        return (ask - bid) / bid 

    @property
    def volatility(self) -> float:
        prices = [
            snap["price_last"]
            for snap in self.snap_slice
            if snap.get("price_last") is not None
        ]
        if len(prices) < 2:
            return 0.0
        mean_price = np.mean(prices)
        if mean_price == 0:
            return 0.0
        return np.std(prices) / mean_price

    @property
    def wamp(self) -> float:
        bid_price, bid_vol = self._safe_get_level(self.bid_book)
        ask_price, ask_vol = self._safe_get_level(self.ask_book)
        numerator = bid_price * bid_vol + ask_price * ask_vol
        denominator = bid_vol + ask_vol
        if denominator == 0 or np.isnan(numerator):
            return 0.0
        return numerator / denominator

    @property
    def alpha_01(self) -> float:
        return self.bid_volume_short / self.bid_volume if self.bid_volume > 0 else 0.0

    @property
    def alpha_02(self) -> float:
        return self.ask_volume_short / self.ask_volume if self.ask_volume > 0 else 0.0

    @property
    def alpha_03(self) -> float:
        return (
            (self.bid_volume_short - self.ask_volume_short)
            / (self.bid_volume + self.ask_volume)
            if (self.bid_volume + self.ask_volume) > 0
            else 0.0
        )

    @property
    def alpha_04(self) -> float:
        if len(self.snap_slice) < self.short_window:
            return 0.0
        num = (
            self.snap_slice[-1]["num_trades"]
            - self.snap_slice[-self.short_window]["num_trades"]
        )
        return num / self.short_window if num > 0 else 0.0
    
    @property
    def alpha_05(self) -> float:
        buys = sum(len(row["buy_trade"]) for row in self.snap_slice[-self.short_window:])
        sells = sum(len(row["sell_trade"]) for row in self.snap_slice[-self.short_window:])
        return (buys - sells) / (buys + sells + 1e-9)
    
    @property
    def alpha_06(self) -> float:
        if len(self.snap_slice) < self.short_window:
            return 0.0

        start_price = self.snap_slice[-self.short_window].get("price_last")
        end_price = self.snap_slice[-1].get("price_last")
        
        if start_price is None or end_price is None or start_price == 0:
            return 0.0
            
        price_diff = abs(end_price - start_price) / start_price 
        total_vol = self.bid_volume_short + self.ask_volume_short
        
        return price_diff  / total_vol if total_vol > 0 else 0.0
    
    @property
    def alpha_07(self) -> float:
        total_vol = self.bid_volume_short + self.ask_volume_short
        if total_vol == 0:
            return 0.0
        
        oi = abs(self.bid_volume_short - self.ask_volume_short)
        return oi / total_vol

    def extract_all(self) -> Dict[str, Any]:
        return {
            "num_trades": self.last.get("num_trades", 0) - self.snap_slice[-2].get("num_trades", 0),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "volatility": self.volatility,
            "spread": self.spread,
            "WAMP": self.wamp,
            "alpha_01": self.alpha_01,
            "alpha_02": self.alpha_02,
            "alpha_03": self.alpha_03,
            "alpha_04": self.alpha_04,
            "alpha_05": self.alpha_05,
            "alpha_06": self.alpha_06,
            "alpha_07": self.alpha_07,
        }


def create_feature(
    snap_slice: List[Dict[str, Any]], short_window: int = 60
) -> Dict[str, Any]:
    return FeatureExtractor(snap_slice, short_window).extract_all()


def latest_zscore(samples):
    if len(samples) == 0:
        return 0.0
    mean = np.mean(samples)
    std = np.std(samples)
    if std == 0:
        return 0.0
    return (samples[-1] - mean) / std
