from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Dict


class StrategyDemo:
    def __init__(self, model=None, param_dict=None) -> None:
        if param_dict is None:
            param_dict = {}
        self.__dict__.update(param_dict)

        self.tick_size = getattr(self, "tick_size", 0.001)
        self.spread_entry_ticks = getattr(self, "spread_entry_ticks", 2)
        self.normal_spread_ticks = getattr(self, "normal_spread_ticks", 1)
        self.rollback_hold_seconds = getattr(self, "rollback_hold_seconds", 3)
        self.rollback_hold_mid_moves = getattr(self, "rollback_hold_mid_moves", 3)
        self.unresolved_max_hold_seconds = getattr(self, "unresolved_max_hold_seconds", 3)
        self.max_total_hold_seconds = getattr(self, "max_total_hold_seconds", 6)
        self.enter_during_spread2 = bool(getattr(self, "enter_during_spread2", False))
        self.allowed_entry_mechanisms = set(
            getattr(self, "allowed_entry_mechanisms", ("bid_retreat",))
        )
        self.blocked_time_buckets = set(getattr(self, "blocked_time_buckets", ("13:00-13:30",)))
        self.min_prev_l1_depth = float(getattr(self, "min_prev_l1_depth", 250000.0))
        self.min_entry_l1_depth = float(getattr(self, "min_entry_l1_depth", 350000.0))
        self.max_entry_duration_for_trade = int(getattr(self, "max_entry_duration_for_trade", 2))
        self.min_signed_close_from_pre_entry_tick = float(
            getattr(self, "min_signed_close_from_pre_entry_tick", 1.0)
        )

        self.position_last = 0
        self.model = model
        self.prev_signal = 0

        self.prev_bid = math.nan
        self.prev_ask = math.nan
        self.prev_mid = math.nan
        self.prev_l1_depth = math.nan
        self.prev_spread_ticks: int | None = None

        self.event_active = False
        self.trade_active = False
        self.entry_direction = 0
        self.entry_hold_seconds = 0
        self.post_close_hold_seconds = 0
        self.post_close_nonzero_mid_moves = 0
        self.event_start_time_mark: int | None = None
        self.spread2_end_bid = math.nan
        self.spread2_end_ask = math.nan
        self.pre_entry_mid = math.nan

    def _reset_event(self) -> None:
        self.position_last = 0
        self.prev_signal = 0
        self.event_active = False
        self.trade_active = False
        self.entry_direction = 0
        self.entry_hold_seconds = 0
        self.post_close_hold_seconds = 0
        self.post_close_nonzero_mid_moves = 0
        self.event_start_time_mark = None
        self.spread2_end_bid = math.nan
        self.spread2_end_ask = math.nan
        self.pre_entry_mid = math.nan

    def _spread_ticks(self, bid: float, ask: float) -> int:
        return int(round((ask - bid) / self.tick_size))

    def _entry_direction(self, bid: float, ask: float, mid: float) -> int:
        if math.isnan(self.prev_bid) or math.isnan(self.prev_ask) or math.isnan(self.prev_mid):
            return 0

        bid_move = int(round((bid - self.prev_bid) / self.tick_size))
        ask_move = int(round((ask - self.prev_ask) / self.tick_size))
        mid_move_half_tick = int(round((mid - self.prev_mid) / (self.tick_size / 2.0)))

        if ask_move > 0 and bid_move == 0:
            return 1
        if ask_move == 0 and bid_move < 0:
            return -1
        if ask_move > 0 and bid_move > 0:
            return 1
        if ask_move < 0 and bid_move < 0:
            return -1
        if mid_move_half_tick > 0:
            return 1
        if mid_move_half_tick < 0:
            return -1
        return 0

    def _resolution_path(self, close_bid: float, close_ask: float) -> str:
        if math.isnan(self.spread2_end_bid) or math.isnan(self.spread2_end_ask):
            return "other_close"

        exit_bid_move = int(round((close_bid - self.spread2_end_bid) / self.tick_size))
        exit_ask_move = int(round((close_ask - self.spread2_end_ask) / self.tick_size))

        if self.entry_direction > 0:
            rollback_amt = max(-exit_ask_move, 0)
            advance_amt = max(exit_bid_move, 0)
        elif self.entry_direction < 0:
            rollback_amt = max(exit_bid_move, 0)
            advance_amt = max(-exit_ask_move, 0)
        else:
            rollback_amt = max(exit_bid_move, 0) + max(-exit_ask_move, 0)
            advance_amt = 0

        if rollback_amt > 0 and advance_amt == 0:
            return "rollback_close"
        if advance_amt > 0 and rollback_amt == 0:
            return "advance_close"
        if rollback_amt > 0 and advance_amt > 0:
            return "mixed_close"
        return "other_close"

    def _time_bucket(self, snap: Dict[str, Any], time_mark: int) -> str:
        time_hms = str(snap.get("time_hms", ""))
        if len(time_hms) >= 5:
            hour = int(time_hms[0:2])
            minute = int(time_hms[3:5])
        else:
            dt = datetime.utcfromtimestamp(time_mark / 1000.0)
            hour = dt.hour
            minute = dt.minute

        bucket_minute = 0 if minute < 30 else 30
        end_hour = hour
        end_minute = bucket_minute + 30
        if end_minute == 60:
            end_hour += 1
            end_minute = 0
        return f"{hour:02d}:{bucket_minute:02d}-{end_hour:02d}:{end_minute:02d}"

    def _entry_mechanism(self, bid: float, ask: float) -> str:
        if math.isnan(self.prev_bid) or math.isnan(self.prev_ask):
            return "start_of_day"

        bid_move = int(round((bid - self.prev_bid) / self.tick_size))
        ask_move = int(round((ask - self.prev_ask) / self.tick_size))
        if ask_move > 0 and bid_move == 0:
            return "ask_retreat"
        if ask_move == 0 and bid_move < 0:
            return "bid_retreat"
        if ask_move > 0 and bid_move > 0:
            return "both_up_ask_faster"
        if ask_move < 0 and bid_move < 0:
            return "both_down_bid_faster"
        if ask_move > 0 and bid_move < 0:
            return "both_widen"
        if ask_move != 0 or bid_move != 0:
            return "other_reprice"
        return "unchanged"

    def _entry_filters_pass(
        self,
        snap: Dict[str, Any],
        time_mark: int,
        bid: float,
        ask: float,
        l1_depth: float,
    ) -> bool:
        mechanism = self._entry_mechanism(bid, ask)
        if mechanism not in self.allowed_entry_mechanisms:
            return False

        time_bucket = self._time_bucket(snap, time_mark)
        if time_bucket in self.blocked_time_buckets:
            return False

        if not math.isnan(self.prev_l1_depth) and self.prev_l1_depth < self.min_prev_l1_depth:
            return False
        if l1_depth < self.min_entry_l1_depth:
            return False
        return True

    def _mark_prev(self, bid: float, ask: float, mid: float, spread_ticks: int, l1_depth: float) -> None:
        self.prev_bid = bid
        self.prev_ask = ask
        self.prev_mid = mid
        self.prev_l1_depth = l1_depth
        self.prev_spread_ticks = spread_ticks

    def on_snap(self, snap: Dict[str, Any]) -> None:
        bid_book = snap.get("bid_book") or []
        ask_book = snap.get("ask_book") or []
        if not bid_book or not ask_book:
            self.position_last = 0
            return

        bid = float(bid_book[0][0])
        ask = float(ask_book[0][0])
        l1_depth = float(bid_book[0][1]) + float(ask_book[0][1])
        mid = (bid + ask) / 2.0
        spread_ticks = self._spread_ticks(bid, ask)
        time_mark = int(snap.get("time_mark", 0))

        if self.event_active:
            self.entry_hold_seconds += 1

            if spread_ticks == self.spread_entry_ticks:
                self.spread2_end_bid = bid
                self.spread2_end_ask = ask

            if self.trade_active:
                self.post_close_hold_seconds += 1
                if not math.isnan(self.prev_mid):
                    mid_move_half_tick = int(round((mid - self.prev_mid) / (self.tick_size / 2.0)))
                    if mid_move_half_tick != 0:
                        self.post_close_nonzero_mid_moves += 1

                if (
                    self.post_close_hold_seconds >= self.rollback_hold_seconds
                    or self.post_close_nonzero_mid_moves >= self.rollback_hold_mid_moves
                    or self.entry_hold_seconds >= self.max_total_hold_seconds
                ):
                    self._reset_event()
            elif spread_ticks == self.normal_spread_ticks:
                resolution_path = self._resolution_path(bid, ask)
                signed_close_from_pre_entry_tick = math.nan
                if not math.isnan(self.pre_entry_mid):
                    signed_close_from_pre_entry_tick = (
                        (mid - self.pre_entry_mid) / self.tick_size * self.entry_direction
                    )

                if (
                    resolution_path == "rollback_close"
                    and self.entry_hold_seconds <= self.max_entry_duration_for_trade
                    and signed_close_from_pre_entry_tick >= self.min_signed_close_from_pre_entry_tick
                ):
                    self.trade_active = True
                    self.position_last = self.entry_direction
                    self.prev_signal = self.entry_direction
                    self.post_close_hold_seconds = 0
                    self.post_close_nonzero_mid_moves = 0
                else:
                    self._reset_event()
            elif self.entry_hold_seconds >= self.max_total_hold_seconds:
                self._reset_event()
            elif self.entry_hold_seconds >= self.unresolved_max_hold_seconds:
                self._reset_event()

        if not self.event_active and self.prev_spread_ticks == self.normal_spread_ticks and spread_ticks == self.spread_entry_ticks:
            direction = self._entry_direction(bid, ask, mid)
            if direction != 0 and self._entry_filters_pass(snap, time_mark, bid, ask, l1_depth):
                self.event_active = True
                self.entry_direction = direction
                self.entry_hold_seconds = 0
                self.trade_active = False
                self.post_close_hold_seconds = 0
                self.post_close_nonzero_mid_moves = 0
                self.event_start_time_mark = time_mark
                self.spread2_end_bid = bid
                self.spread2_end_ask = ask
                self.pre_entry_mid = self.prev_mid
                if self.enter_during_spread2:
                    self.position_last = direction
                    self.prev_signal = direction
                else:
                    self.position_last = 0
                    self.prev_signal = 0

        if self.event_active and not self.trade_active and self.enter_during_spread2:
            self.position_last = self.entry_direction
            self.prev_signal = self.entry_direction
        elif not self.trade_active:
            self.position_last = 0
            self.prev_signal = 0
        elif self.position_last == 0:
            self.position_last = self.entry_direction
            self.prev_signal = self.entry_direction

        self._mark_prev(bid, ask, mid, spread_ticks, l1_depth)
