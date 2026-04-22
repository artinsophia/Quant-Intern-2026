import numpy as np
from typing import List, Dict, Any


class FeatureExtractor:
    def __init__(self, snap_slice: List[Dict[str, Any]], short_window: int = 60):
        if not snap_slice:
            raise ValueError("snap_slice cannot be empty")


        self.last = snap_slice[-1]
        self.short_window = short_window

        self.bid_book = self.last['bid_book']
        self.ask_book = self.last['ask_book']

        self.snap_slice = list(snap_slice)

        self.prices = np.array([
            snap["price_last"]
            for snap in self.snap_slice
        ])
        self.bid_volume = np.array([
            sum(v for _,v in s["buy_trade"]) for s in snap_slice])
        self.ask_volume = np.array([
            sum(v for _,v in s["sell_trade"]) for s in snap_slice])
        self.trade_count = np.array([s['num_trades'] for s in snap_slice])

        

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
        return abs(ask - bid) / bid 

    @property
    def volatility(self) -> float:

        mean_price = np.mean(self.prices)
        if mean_price == 0:
            return 0.0
        return np.std(self.prices) / mean_price

    @property
    def wamp(self) -> float:
        bid_price, bid_vol = self._safe_get_level(self.bid_book)
        ask_price, ask_vol = self._safe_get_level(self.ask_book)
        numerator = bid_price * bid_vol + ask_price * ask_vol
        denominator = bid_vol + ask_vol
        if denominator == 0 or np.isnan(numerator):
            return 0.0
        return numerator / denominator
    
    
    # trade
    @property
    def alpha_04(self) -> float:
        if len(self.snap_slice) < self.short_window:
            return 0.0
        num = (
            self.trade_count[-1]
            - self.trade_count[-self.short_window]
        )
        return num / self.short_window if num > 0 else 0.0
    
    @property
    def alpha_05(self) -> float:
        buys = sum(len(row["buy_trade"]) for row in self.snap_slice[-self.short_window:])
        sells = sum(len(row["sell_trade"]) for row in self.snap_slice[-self.short_window:])
        return (buys - sells) / (buys + sells + 1e-9)
    
    

    

    def extract_all(self) -> Dict[str, Any]:
        # hurst , hurst_flag = calculate_hurst_exponent(self.prices)
        
        total_buy = np.sum(self.bid_volume)
        total_sell = np.sum(self.ask_volume)

        short_buy = np.sum(self.bid_volume[-self.short_window:])
        short_sell = np.sum(self.ask_volume[-self.short_window:])
        
        volume = total_buy + total_sell
        volume_short = short_buy + short_sell

        alpha_01 = short_buy / (total_buy + 1e-9)
        alpha_02 = short_sell / (total_sell + 1e-9)
        alpha_03 = abs(short_buy - short_sell) / (volume_short + 1e-9) # 消除选择偏差

        start_price = self.snap_slice[-self.short_window].get("price_last")
        end_price = self.snap_slice[-1].get("price_last")
        price_diff = abs(end_price - start_price) / (start_price + 1e-9) 
        alpha_06 = price_diff / (short_buy + short_sell + 1e-9)

        prices_diffs = np.diff(self.prices)
        total_abs_diff = np.sum(np.abs(prices_diffs))
        alpha_07 = np.abs(np.sum(prices_diffs)) / (total_abs_diff + 1e-9)

        return {
            "num_trades": self.last.get("num_trades", 0) - self.snap_slice[-2].get("num_trades", 0),
            "volatility": self.volatility,
            "spread": self.spread,
            "WAMP": self.wamp,
            "volume": volume,
            "alpha_01": alpha_01,
            "alpha_02": alpha_02,
            "alpha_03": alpha_03,
            "alpha_04": self.alpha_04,
            "alpha_05": self.alpha_05,
            "alpha_06": alpha_06,
            "alpha_07": alpha_07,
            # "hurst_exponent": hurst,
            # "hurst": hurst_flag
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

def calculate_hurst_exponent(prices: List[float], max_lag: int = 20) -> tuple[float, bool]:
    if len(prices) < 10:
        return 0.5

    returns = np.diff(np.log(prices))
    if len(returns) < 5:
        return 0.5

    lags = range(2, min(max_lag, len(returns)))
    tau = []

    for lag in lags:
        n_blocks = len(returns) // lag
        if n_blocks == 0:
            tau.append(0.0)
            continue

        # 将 returns 重塑为 (n_blocks, lag) 的矩阵，丢弃末尾不足 lag 的部分
        blocks = returns[:n_blocks * lag].reshape(n_blocks, lag)

        # 计算每行的均值和标准差
        std_block = np.std(blocks, axis=1, ddof=0)  # 总体标准差，与原代码一致

        # 找出标准差大于0的行
        valid_idx = np.where(std_block > 0)[0]
        if not np.any(valid_idx):
            tau.append(0.0)
            continue

        # 只保留有效行
        blocks_valid = blocks[valid_idx]
        std_valid = std_block[valid_idx]

        means = np.mean(blocks_valid, axis=1)[:,None]
        centered_blocks = blocks_valid - means

        # 计算每行的累积和（按行）
        cumsum_block = np.cumsum(centered_blocks, axis=1)

        R = np.max(cumsum_block,axis = 1) - np.min(cumsum_block,axis = 1)

        tau.append(np.log(np.mean(R / std_valid)))

    # 过滤掉无效的 tau 值（0 表示该 lag 无有效子区间）
    valid_tau = [(l, t) for l, t in zip(lags, tau) if t != 0.0]
    if len(valid_tau) < 2:
        return 0.5

    lags_log = np.log([l for l, _ in valid_tau])
    tau_vals = [t for _, t in valid_tau]

    # 线性回归
    hurst = np.polyfit(lags_log, tau_vals, 1)[0]
    hurst = max(0.0, min(1.0, hurst))
    return hurst , hurst > 0.5