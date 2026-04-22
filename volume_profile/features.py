import numpy as np
from typing import List, Dict, Any


class FeatureExtractor:
    def __init__(self, snap_slice: List[Dict[str, Any]] , x_window: int = 300, short_window: int = 60):

        self.n_bins = x_window // short_window
        self.eps = 1e-12
        
        self.trade_count = np.array([s['num_trades'] for s in snap_slice])

        self.bid_volume = np.array([
            sum(v for _,v in s["buy_trade"]) for s in snap_slice])
        self.ask_volume = np.array([
            sum(v for _,v in s["sell_trade"]) for s in snap_slice])
        self.net_volume = self.bid_volume - self.ask_volume

        self.bid_bin = self.bid_volume.reshape(-1,short_window).sum(axis = 1)
        self.ask_bin = self.ask_volume.reshape(-1,short_window).sum(axis = 1)
        self.net_bin = self.bid_bin - self.ask_bin
        self.abs_net_bin = np.abs(self.net_bin)
        self.total_bin = self.bid_bin + self.ask_bin
        self.imbalance_bin = np.where(
            self.total_bin > 0,
            self.net_bin / self.total_bin,
            0.0
        )



    def _safe_div(self, a: float, b: float) -> float:
        return a / b if abs(b) > self.eps else 0.0

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        s = np.sum(x)
        if s <= self.eps:
            return np.zeros_like(x, dtype=float)
        return x / s
    
    # 集中度
    def concentration_max(self, x: np.ndarray) -> float:
        s = np.sum(x)
        if s <= self.eps:
            return 0.0
        return np.max(x) / s

    def concentration_hhi(self, x: np.ndarray) -> float:
        p = self._normalize(x)
        return np.sum(p ** 2)

    def concentration_entropy(self, x: np.ndarray) -> float:
        p = self._normalize(x)
        p = p[p > self.eps]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log(p))
    
    def concentration_features(self) -> Dict[str, float]:
        return {
            "conc_total_max": self.concentration_max(self.total_bin),
            "conc_total_hhi": self.concentration_hhi(self.total_bin),
            "conc_total_entropy": self.concentration_entropy(self.total_bin),

            "conc_absnet_max": self.concentration_max(self.abs_net_bin),
            "conc_absnet_hhi": self.concentration_hhi(self.abs_net_bin),
            "conc_absnet_entropy": self.concentration_entropy(self.abs_net_bin),

            "conc_buy_max": self.concentration_max(self.bid_bin),
            "conc_sell_max": self.concentration_max(self.ask_bin),
        }
    
    # 前后置差
    
    def front_back_diff(self, x: np.ndarray, normalize: bool = True) -> float:
        half = self.n_bins // 2
        front = np.sum(x[:half]) / half
        back = np.sum(x[half:]) / (self.n_bins - half)
        diff = back - front

        if not normalize:
            return diff

        denom = np.sum(np.abs(x))
        return self._safe_div(diff, denom)
    
    def front_back_features(self) -> Dict[str, float]:
        return {
            "fb_total": self.front_back_diff(self.total_bin, normalize=True),
            "fb_buy": self.front_back_diff(self.bid_bin, normalize=True),
            "fb_sell": self.front_back_diff(self.ask_bin, normalize=True),
            "fb_net": self.front_back_diff(self.net_bin, normalize=True),

            "fb_total_raw": self.front_back_diff(self.total_bin, normalize=False),
            "fb_buy_raw": self.front_back_diff(self.bid_bin, normalize=False),
            "fb_sell_raw": self.front_back_diff(self.ask_bin, normalize=False),
            "fb_net_raw": self.front_back_diff(self.net_bin, normalize=False),
        }
    
    # 峰值强度
    def peak_strength(self, x: np.ndarray) -> float:
        m = np.mean(x)
        if m <= self.eps:
            return 0.0
        return np.max(x) / m

    def peak_features(self) -> Dict[str, float]:
        abs_net_bin = np.abs(self.net_bin)

        return {
            "peak_total": self.peak_strength(self.total_bin),
            "peak_buy": self.peak_strength(self.bid_bin),
            "peak_sell": self.peak_strength(self.ask_bin),
            "peak_absnet": self.peak_strength(abs_net_bin),
        }
    

    def extract_shape_features(self) -> Dict[str, float]:
        feat = {}
        feat.update(self.concentration_features())
        feat.update(self.front_back_features())
        feat.update(self.peak_features())
        return feat
