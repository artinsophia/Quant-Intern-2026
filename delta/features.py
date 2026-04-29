import numpy as np
from collections import deque
from typing import List, Dict, Any


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


def trade_entry_count(trades: Any) -> int:
    if not isinstance(trades, (list, tuple)):
        return 0
    return len(trades)


def trade_volume_sum_topn(trades: Any, n: int) -> float:
    if not isinstance(trades, (list, tuple)):
        return 0.0
    return trade_volume_sum(trades[:n])


class FeatureExtractor:
    def __init__(self, snap_slice: List[Dict[str, Any]], short_window: int = 300):
        self.eps = 1e-8
        if not snap_slice:
            raise ValueError("snap_slice cannot be empty")
        self.snap_slice = list(snap_slice)
        self.last = self.snap_slice[-1]
        self.short_window = min(short_window, len(self.snap_slice))
        self.window_slice = slice(-self.short_window, None)

        self.bid_book = self.last["bid_book"]
        self.ask_book = self.last["ask_book"]

        size = len(self.snap_slice)
        self.prices = np.fromiter(
            (snap["price_last"] for snap in self.snap_slice),
            dtype=float,
            count=size,
        )
        self.bid_volume = np.fromiter(
            (trade_volume_sum(snap.get("buy_trade")) for snap in self.snap_slice),
            dtype=float,
            count=size,
        )
        self.ask_volume = np.fromiter(
            (trade_volume_sum(snap.get("sell_trade")) for snap in self.snap_slice),
            dtype=float,
            count=size,
        )
        self.trade_count = np.fromiter(
            (snap["num_trades"] for snap in self.snap_slice),
            dtype=float,
            count=size,
        )
        self.trade_buy_count = np.fromiter(
            (trade_entry_count(snap.get("buy_trade")) for snap in self.snap_slice),
            dtype=float,
            count=size,
        )
        self.trade_sell_count = np.fromiter(
            (trade_entry_count(snap.get("sell_trade")) for snap in self.snap_slice),
            dtype=float,
            count=size,
        )
        self.bid = np.fromiter(
            (
                snap["bid_book"][0][0] if snap["bid_book"] else np.nan
                for snap in self.snap_slice
            ),
            dtype=float,
            count=size,
        )
        self.ask = np.fromiter(
            (
                snap["ask_book"][0][0] if snap["ask_book"] else np.nan
                for snap in self.snap_slice
            ),
            dtype=float,
            count=size,
        )

        self.returns = np.log(self.prices[1:] / self.prices[:-1]) if size >= 2 else np.empty(0, dtype=float)
        self.mid_price = (self.bid + self.ask) / 2.0

        self.short_buy_volume = float(np.sum(self.bid_volume[self.window_slice]))
        self.short_sell_volume = float(np.sum(self.ask_volume[self.window_slice]))
        self.total_buy_volume = float(np.sum(self.bid_volume))
        self.total_sell_volume = float(np.sum(self.ask_volume))
        self.short_buy_count = float(np.sum(self.trade_buy_count[self.window_slice]))
        self.short_sell_count = float(np.sum(self.trade_sell_count[self.window_slice]))

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
        if not np.isfinite(bid) or not np.isfinite(ask):
            return 0.0
        return self._safe_div(abs(ask - bid), bid)

    @property
    def volatility(self) -> float:

        mean_price = np.mean(self.prices)
        return self._safe_div(np.std(self.prices), mean_price)
    
    def _safe_div(self, a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= self.eps:
            return 0.0
        out = a / b
        return out if np.isfinite(out) else 0.0

    

    def extract_all(self) -> Dict[str, Any]:
        # hurst , hurst_flag = calculate_hurst_exponent(self.prices)

        total_buy = self.total_buy_volume
        total_sell = self.total_sell_volume

        short_buy = self.short_buy_volume
        short_sell = self.short_sell_volume

        volume = total_buy + total_sell
        volume_short = short_buy + short_sell

        alpha_01 = self._safe_div(short_buy, total_buy)
        alpha_02 = self._safe_div(short_sell, total_sell)
        alpha_03 = self._safe_div(abs(short_buy - short_sell), volume_short)

        start_price = self.prices[-self.short_window]
        end_price = self.prices[-1]
        price_diff = self._safe_div(abs(end_price - start_price), start_price)

        alpha_06 = self._safe_div(price_diff, self._safe_div(volume - volume_short, volume))

        prices_diffs = np.diff(self.prices)
        total_abs_diff = np.sum(np.abs(prices_diffs))
        alpha_07 = self._safe_div(abs(np.sum(prices_diffs)), total_abs_diff)

        short_prices = self.prices[self.window_slice]
        lower_bound = short_prices.min()
        upper_bound = short_prices.max()
        RPP = self._safe_div(self.prices[-1] - lower_bound, upper_bound - lower_bound)

        total_sq = np.sum(self.returns ** 2)
        positive_windows = np.where(self.returns > 0, self.returns, 0)
        positive_sumsq = np.sum(positive_windows ** 2)
        alpha_08 = self._safe_div(positive_sumsq, total_sq)

        MPC = self._safe_div(
            self.mid_price[-1] - self.mid_price[-self.short_window],
            self.mid_price[-self.short_window],
        )

        pos = self.returns > 0
        positive_return_count = np.mean(pos)
        padded = np.concatenate(([False], pos, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        max_up_streak = int(np.max(ends - starts)) if len(starts) else 0
        

        return {
            "volatility": self.volatility,
            "alpha_03": alpha_03,
            "alpha_06": alpha_06,
            "alpha_07": alpha_07,
            "RPP": RPP,
            "alpha_08": alpha_08,
            "MPC": MPC,
            "positive_return_count": positive_return_count,
            "alpha_09": max_up_streak,
            # "hurst_exponent": hurst,
            # "hurst": hurst_flag
        }
    


def create_feature(
    snap_slice: List[Dict[str, Any]], short_window: int = 60
) -> Dict[str, Any]:
    return FeatureExtractor(snap_slice, short_window).extract_all()


class RollingFeatureExtractor:
    def __init__(self, window_size: int, short_window: int = 60):
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.eps = 1e-8
        self.window_size = window_size
        self.short_window = min(short_window, window_size)

        self.snaps = deque(maxlen=window_size)
        self.prices = deque(maxlen=window_size)
        self.trade_count = deque(maxlen=window_size)
        self.bid = deque(maxlen=window_size)
        self.ask = deque(maxlen=window_size)
        self.buy_volume = deque(maxlen=window_size)
        self.sell_volume = deque(maxlen=window_size)
        self.buy_count = deque(maxlen=window_size)
        self.sell_count = deque(maxlen=window_size)

        self.total_buy_volume = 0.0
        self.total_sell_volume = 0.0

    def append(self, snap: Dict[str, Any]) -> None:
        if len(self.snaps) == self.window_size:
            self._evict_oldest()

        buy_trade = snap.get("buy_trade")
        sell_trade = snap.get("sell_trade")
        buy_volume = self._trade_volume_sum(buy_trade)
        sell_volume = self._trade_volume_sum(sell_trade)

        bid_book = snap["bid_book"]
        ask_book = snap["ask_book"]

        self.snaps.append(snap)
        self.prices.append(float(snap["price_last"]))
        self.trade_count.append(float(snap["num_trades"]))
        self.bid.append(bid_book[0][0] if bid_book else np.nan)
        self.ask.append(ask_book[0][0] if ask_book else np.nan)
        self.buy_volume.append(buy_volume)
        self.sell_volume.append(sell_volume)
        self.buy_count.append(float(self._trade_entry_count(buy_trade)))
        self.sell_count.append(float(self._trade_entry_count(sell_trade)))

        self.total_buy_volume += buy_volume
        self.total_sell_volume += sell_volume

    def is_ready(self) -> bool:
        return len(self.snaps) == self.window_size

    def clear(self) -> None:
        self.snaps.clear()
        self.prices.clear()
        self.trade_count.clear()
        self.bid.clear()
        self.ask.clear()
        self.buy_volume.clear()
        self.sell_volume.clear()
        self.buy_count.clear()
        self.sell_count.clear()
        self.total_buy_volume = 0.0
        self.total_sell_volume = 0.0

    def extract_all(self) -> Dict[str, Any]:
        if not self.snaps:
            raise ValueError("feature window is empty")

        prices = np.asarray(self.prices, dtype=float)
        bid = np.asarray(self.bid, dtype=float)
        ask = np.asarray(self.ask, dtype=float)
        mid_price = (bid + ask) / 2.0
        returns = np.log(prices[1:] / prices[:-1]) if len(prices) >= 2 else np.empty(0, dtype=float)

        short_buy = float(sum(list(self.buy_volume)[-self.short_window:]))
        short_sell = float(sum(list(self.sell_volume)[-self.short_window:]))
        volume = self.total_buy_volume + self.total_sell_volume
        volume_short = short_buy + short_sell

        start_price = prices[-self.short_window]
        end_price = prices[-1]
        price_diff = self._safe_div(abs(end_price - start_price), start_price)

        prices_diffs = np.diff(prices)
        total_abs_diff = np.sum(np.abs(prices_diffs))
        short_prices = prices[-self.short_window:]
        lower_bound = short_prices.min()
        upper_bound = short_prices.max()
        total_sq = np.sum(returns ** 2)
        positive_windows = np.where(returns > 0, returns, 0)
        positive_sumsq = np.sum(positive_windows ** 2)

        short_buy_count = float(sum(list(self.buy_count)[-self.short_window:]))
        short_sell_count = float(sum(list(self.sell_count)[-self.short_window:]))
        last_snap = self.snaps[-1]
        prev_snap = self.snaps[-2] if len(self.snaps) >= 2 else None

        pos = returns > 0
        positive_return_count = np.mean(pos)
        padded = np.concatenate(([False], pos, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        max_up_streak = int(np.max(ends - starts)) if len(starts) else 0

        return {
            "volatility": self._safe_div(np.std(prices), np.mean(prices)),
            "alpha_03": self._safe_div(abs(short_buy - short_sell), volume_short),
            "alpha_06": self._safe_div(price_diff, self._safe_div(volume - volume_short, volume)),
            "alpha_07": self._safe_div(abs(np.sum(prices_diffs)), total_abs_diff),
            "RPP": self._safe_div(prices[-1] - lower_bound, upper_bound - lower_bound),
            "alpha_08": self._safe_div(positive_sumsq, total_sq),
            "MPC": self._safe_div(mid_price[-1] - mid_price[-self.short_window], mid_price[-self.short_window]),
            "positive_return_count": positive_return_count,
            "alpha_09": max_up_streak,
        }

    def _evict_oldest(self) -> None:
        self.total_buy_volume -= self.buy_volume[0]
        self.total_sell_volume -= self.sell_volume[0]
        self.snaps.popleft()
        self.prices.popleft()
        self.trade_count.popleft()
        self.bid.popleft()
        self.ask.popleft()
        self.buy_volume.popleft()
        self.sell_volume.popleft()
        self.buy_count.popleft()
        self.sell_count.popleft()

    @staticmethod
    def _safe_get_level(book: List[tuple], idx: int = 0) -> tuple:
        if book and len(book) > idx:
            return book[idx]
        return (np.nan, 0)

    @staticmethod
    def _trade_volume_sum(trades: Any) -> float:
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

    @staticmethod
    def _trade_entry_count(trades: Any) -> int:
        if not isinstance(trades, (list, tuple)):
            return 0
        return len(trades)

    def _spread(self, snap: Dict[str, Any]) -> float:
        bid, _ = self._safe_get_level(snap["bid_book"])
        ask, _ = self._safe_get_level(snap["ask_book"])
        if not np.isfinite(bid) or not np.isfinite(ask):
            return 0.0
        return self._safe_div(abs(ask - bid), bid)


    def _alpha_04(self) -> float:
        if len(self.trade_count) < 2:
            return 0.0
        num = self.trade_count[-1] - self.trade_count[-self.short_window]
        return num / self.short_window if num > 0 else 0.0

    def _safe_div(self, a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= self.eps:
            return 0.0
        out = a / b
        return out if np.isfinite(out) else 0.0


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
