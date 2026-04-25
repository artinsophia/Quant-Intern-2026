from __future__ import annotations

from typing import Any

import numpy as np


def safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or abs(float(denominator)) < 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def get_mid_price(snap: dict[str, Any]) -> float | None:
    bid_book = snap.get("bid_book") or []
    ask_book = snap.get("ask_book") or []
    if not bid_book or not ask_book:
        return None
    return (float(bid_book[0][0]) + float(ask_book[0][0])) / 2.0


def get_spread_ticks(snap: dict[str, Any], tick_size: float) -> float:
    bid_book = snap.get("bid_book") or []
    ask_book = snap.get("ask_book") or []
    if not bid_book or not ask_book or tick_size <= 0:
        return 0.0
    return (float(ask_book[0][0]) - float(bid_book[0][0])) / tick_size


def depth_sum(book: list[list[float]] | None, levels: int) -> float:
    book = book or []
    return float(sum(volume for _, volume in book[:levels]))


def imbalance(bid_depth: float, ask_depth: float) -> float:
    return safe_div(bid_depth - ask_depth, bid_depth + ask_depth)


def active_volumes(snap: dict[str, Any]) -> tuple[float, float]:
    buy_trade = snap.get("buy_trade") or []
    sell_trade = snap.get("sell_trade") or []
    buy_vol = float(sum(volume for _, volume in buy_trade))
    sell_vol = float(sum(volume for _, volume in sell_trade))
    return buy_vol, sell_vol


def active_turnovers(snap: dict[str, Any]) -> tuple[float, float]:
    buy_trade = snap.get("buy_trade") or []
    sell_trade = snap.get("sell_trade") or []
    buy_turnover = float(sum(price * volume for price, volume in buy_trade))
    sell_turnover = float(sum(price * volume for price, volume in sell_trade))
    return buy_turnover, sell_turnover


def consecutive_same_mid(snaps: list[dict[str, Any]]) -> int:
    if not snaps:
        return 0

    mids = [get_mid_price(snap) for snap in snaps]
    last_mid = mids[-1]
    if last_mid is None:
        return 0

    count = 0
    for mid in reversed(mids):
        if mid != last_mid:
            break
        count += 1
    return count


def create_feature(
    snap_slice: list[dict[str, Any]],
    tick_size: float = 0.001,
) -> dict[str, float]:
    if not snap_slice:
        raise ValueError("snap_slice cannot be empty")

    mids = np.array([get_mid_price(snap) or 0.0 for snap in snap_slice], dtype=float)
    last_prices = np.array(
        [float(snap.get("price_last") or 0.0) for snap in snap_slice],
        dtype=float,
    )
    spreads = np.array(
        [get_spread_ticks(snap, tick_size) for snap in snap_slice],
        dtype=float,
    )
    l1_bid = np.array(
        [depth_sum(snap.get("bid_book"), 1) for snap in snap_slice],
        dtype=float,
    )
    l1_ask = np.array(
        [depth_sum(snap.get("ask_book"), 1) for snap in snap_slice],
        dtype=float,
    )
    l3_bid = np.array(
        [depth_sum(snap.get("bid_book"), 3) for snap in snap_slice],
        dtype=float,
    )
    l3_ask = np.array(
        [depth_sum(snap.get("ask_book"), 3) for snap in snap_slice],
        dtype=float,
    )
    l5_bid = np.array(
        [depth_sum(snap.get("bid_book"), 5) for snap in snap_slice],
        dtype=float,
    )
    l5_ask = np.array(
        [depth_sum(snap.get("ask_book"), 5) for snap in snap_slice],
        dtype=float,
    )
    buy_vol = np.array([active_volumes(snap)[0] for snap in snap_slice], dtype=float)
    sell_vol = np.array([active_volumes(snap)[1] for snap in snap_slice], dtype=float)
    buy_turnover = np.array(
        [active_turnovers(snap)[0] for snap in snap_slice],
        dtype=float,
    )
    sell_turnover = np.array(
        [active_turnovers(snap)[1] for snap in snap_slice],
        dtype=float,
    )
    trade_count = np.array(
        [float(snap.get("num_trades", 0)) for snap in snap_slice],
        dtype=float,
    )

    l1_imbalance = np.array(
        [imbalance(bid_depth, ask_depth) for bid_depth, ask_depth in zip(l1_bid, l1_ask)],
        dtype=float,
    )
    l3_imbalance = np.array(
        [imbalance(bid_depth, ask_depth) for bid_depth, ask_depth in zip(l3_bid, l3_ask)],
        dtype=float,
    )
    l5_imbalance = np.array(
        [imbalance(bid_depth, ask_depth) for bid_depth, ask_depth in zip(l5_bid, l5_ask)],
        dtype=float,
    )

    stall_seconds = float(consecutive_same_mid(snap_slice))
    mid_range_ticks = safe_div(np.max(mids) - np.min(mids), tick_size)
    last_return_ticks = safe_div(last_prices[-1] - last_prices[0], tick_size)
    trade_count_delta = float(max(trade_count[-1] - trade_count[0], 0.0))

    cum_buy_vol = float(np.sum(buy_vol))
    cum_sell_vol = float(np.sum(sell_vol))
    cum_buy_turnover = float(np.sum(buy_turnover))
    cum_sell_turnover = float(np.sum(sell_turnover))
    cum_active_vol = cum_buy_vol + cum_sell_vol

    return {
        "stall_seconds": stall_seconds,
        "spread_ticks_last": float(spreads[-1]),
        "spread_ticks_mean": float(np.mean(spreads)),
        "mid_range_ticks": float(mid_range_ticks),
        "last_return_ticks": float(last_return_ticks),
        "l1_imbalance_last": float(l1_imbalance[-1]),
        "l3_imbalance_last": float(l3_imbalance[-1]),
        "l5_imbalance_last": float(l5_imbalance[-1]),
        "l1_imbalance_mean": float(np.mean(l1_imbalance)),
        "l3_imbalance_mean": float(np.mean(l3_imbalance)),
        "l5_imbalance_mean": float(np.mean(l5_imbalance)),
        "l1_imbalance_change": float(l1_imbalance[-1] - l1_imbalance[0]),
        "l5_imbalance_change": float(l5_imbalance[-1] - l5_imbalance[0]),
        "bid_l1_change": float(l1_bid[-1] - l1_bid[0]),
        "ask_l1_change": float(l1_ask[-1] - l1_ask[0]),
        "bid_l5_change": float(l5_bid[-1] - l5_bid[0]),
        "ask_l5_change": float(l5_ask[-1] - l5_ask[0]),
        "cum_buy_vol": cum_buy_vol,
        "cum_sell_vol": cum_sell_vol,
        "cum_net_active_vol": cum_buy_vol - cum_sell_vol,
        "cum_active_vol_imbalance": imbalance(cum_buy_vol, cum_sell_vol),
        "cum_buy_turnover": cum_buy_turnover,
        "cum_sell_turnover": cum_sell_turnover,
        "cum_net_active_turnover": cum_buy_turnover - cum_sell_turnover,
        "cum_trade_count": trade_count_delta,
        "active_seconds_ratio": float(np.mean((buy_vol + sell_vol) > 0)),
    }
