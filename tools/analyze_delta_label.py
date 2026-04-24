import argparse
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

sys.path.append("/home/jovyan/base_demo")
import base_tool

from delta.features import latest_zscore


def safe_mean(values):
    return float(sum(values) / len(values)) if values else float("nan")


def safe_median(values):
    return float(statistics.median(values)) if values else float("nan")


def compute_volatility(prices, end_idx, vol_window):
    start_idx = max(0, end_idx - vol_window)
    price_window = prices[start_idx:end_idx]
    if len(price_window) == 0:
        return 0.0
    mean_price = float(np.mean(price_window))
    if mean_price == 0:
        return 0.0
    return float(np.std(price_window) / mean_price)


def barrier_outcome(prices, start_idx, y_window, volatility, k_up, k_down):
    start_price = prices[start_idx]
    if (
        start_price is None
        or start_price == 0
        or (isinstance(start_price, float) and math.isnan(start_price))
    ):
        return {
            "up_hit": False,
            "down_hit": False,
            "t_up": None,
            "t_down": None,
            "label": 0,
            "up_barrier": None,
            "down_barrier": None,
        }

    up_barrier = start_price * (1 + volatility * k_up)
    down_barrier = start_price * (1 - volatility * k_down)
    t_up = None
    t_down = None

    stop_idx = min(len(prices), start_idx + y_window)
    for idx in range(start_idx + 1, stop_idx):
        price = prices[idx]
        if price is None or (isinstance(price, float) and math.isnan(price)):
            continue
        elapsed = idx - start_idx
        if t_up is None and price >= up_barrier:
            t_up = elapsed
        if t_down is None and price <= down_barrier:
            t_down = elapsed
        if t_up is not None and t_down is not None:
            break

    if t_up is not None and t_down is not None:
        label = 1 if t_up < t_down else -1
    elif t_up is not None:
        label = 1
    elif t_down is not None:
        label = -1
    else:
        label = 0

    return {
        "up_hit": t_up is not None,
        "down_hit": t_down is not None,
        "t_up": t_up,
        "t_down": t_down,
        "label": label,
        "up_barrier": up_barrier,
        "down_barrier": down_barrier,
    }


def eligible_indices(n_snaps, x_window, y_window):
    return range(x_window, n_snaps - y_window)


def load_day_arrays(instrument_id, trade_ymd):
    snap_list = base_tool.snap_list_load(instrument_id, trade_ymd)
    prices = np.array([row["price_last"] for row in snap_list], dtype=float)
    delta = np.array(
        [
            sum(vol for _, vol in row["buy_trade"])
            - sum(vol for _, vol in row["sell_trade"])
            for row in snap_list
        ],
        dtype=float,
    )
    return snap_list, prices, delta


def analyze_y_windows(
    instrument_id,
    dates,
    y_windows,
    short_window,
    x_window,
    vol_window,
    open_threshold,
    k_up,
    k_down,
):
    results = []
    max_y_window = max(y_windows)
    cached_days = []

    for trade_ymd in dates:
        snap_list, prices, delta = load_day_arrays(instrument_id, trade_ymd)
        if len(snap_list) < x_window + max_y_window:
            continue
        cached_days.append((trade_ymd, prices, delta, len(snap_list)))

    for y_window in y_windows:
        direction_stats = {
            1: defaultdict(list),
            -1: defaultdict(list),
        }
        daily_trigger_count = []
        raw_label_counter = Counter()
        binary_counter = Counter()
        touches = Counter()
        volatility_list = []
        barrier_pct_list = []
        total_eligible = 0
        total_days = 0

        for trade_ymd, prices, delta, n_snaps in cached_days:
            day_trigger_count = 0
            total_days += 1

            for i in eligible_indices(n_snaps, x_window, y_window):
                total_eligible += 1
                z = latest_zscore(delta[i - short_window : i])
                if z > open_threshold:
                    category = 1
                elif z < -open_threshold:
                    category = -1
                else:
                    continue

                day_trigger_count += 1
                volatility = compute_volatility(prices, i, vol_window)
                outcome = barrier_outcome(prices, i, y_window, volatility, k_up, k_down)
                raw_label = outcome["label"]
                binary_label = 1 if raw_label == category else 0

                raw_label_counter[raw_label] += 1
                binary_counter[binary_label] += 1
                volatility_list.append(volatility)
                barrier_pct_list.append(volatility * k_up)

                direction_stats[category]["binary"].append(binary_label)
                direction_stats[category]["volatility"].append(volatility)

                if outcome["up_hit"]:
                    direction_stats[category]["t_up"].append(outcome["t_up"])
                if outcome["down_hit"]:
                    direction_stats[category]["t_down"].append(outcome["t_down"])

                if raw_label == 0:
                    touches["none"] += 1
                    direction_stats[category]["raw_0"].append(1)
                elif raw_label == 1:
                    touches["up_first"] += 1
                    direction_stats[category]["raw_1"].append(1)
                elif raw_label == -1:
                    touches["down_first"] += 1
                    direction_stats[category]["raw_-1"].append(1)

            daily_trigger_count.append(day_trigger_count)

        total_triggered = sum(raw_label_counter.values())
        res = {
            "y_window": y_window,
            "days": total_days,
            "eligible_samples": total_eligible,
            "triggered_samples": total_triggered,
            "trigger_rate": total_triggered / total_eligible if total_eligible else float("nan"),
            "label1_rate": binary_counter[1] / total_triggered if total_triggered else float("nan"),
            "raw_up_first_rate": raw_label_counter[1] / total_triggered if total_triggered else float("nan"),
            "raw_down_first_rate": raw_label_counter[-1] / total_triggered if total_triggered else float("nan"),
            "raw_no_touch_rate": raw_label_counter[0] / total_triggered if total_triggered else float("nan"),
            "mean_volatility": safe_mean(volatility_list),
            "mean_barrier_pct": safe_mean(barrier_pct_list),
            "median_barrier_pct": safe_median(barrier_pct_list),
            "up_first_median_t": safe_median(
                direction_stats[1]["t_up"] + direction_stats[-1]["t_up"]
            ),
            "down_first_median_t": safe_median(
                direction_stats[1]["t_down"] + direction_stats[-1]["t_down"]
            ),
            "long_trigger_share": (
                len(direction_stats[1]["binary"]) / total_triggered if total_triggered else float("nan")
            ),
            "long_label1_rate": safe_mean(direction_stats[1]["binary"]),
            "short_label1_rate": safe_mean(direction_stats[-1]["binary"]),
            "avg_daily_trigger_count": safe_mean(daily_trigger_count),
        }

        for category, prefix in ((1, "long"), (-1, "short")):
            binary_values = direction_stats[category]["binary"]
            count = len(binary_values)
            res[f"{prefix}_count"] = count
            res[f"{prefix}_win_rate"] = safe_mean(binary_values)
            res[f"{prefix}_up_hit_median_t"] = safe_median(direction_stats[category]["t_up"])
            res[f"{prefix}_down_hit_median_t"] = safe_median(direction_stats[category]["t_down"])
            res[f"{prefix}_up_first_rate"] = (
                len(direction_stats[category]["raw_1"]) / count if count else float("nan")
            )
            res[f"{prefix}_down_first_rate"] = (
                len(direction_stats[category]["raw_-1"]) / count if count else float("nan")
            )
            res[f"{prefix}_no_touch_rate"] = (
                len(direction_stats[category]["raw_0"]) / count if count else float("nan")
            )

        results.append(res)

    return results


def analyze_open_thresholds(
    instrument_id,
    dates,
    thresholds,
    short_window,
    x_window,
):
    results = []
    cached_days = []
    for trade_ymd in dates:
        snap_list, prices, delta = load_day_arrays(instrument_id, trade_ymd)
        if len(snap_list) <= x_window:
            continue
        cached_days.append((trade_ymd, len(snap_list), delta))

    for threshold in thresholds:
        daily_counts = []
        daily_long_counts = []
        daily_short_counts = []
        daily_episode_counts = []
        gap_list = []
        run_length_list = []
        total_signals = 0
        total_eligible = 0
        long_signals = 0
        short_signals = 0
        total_episodes = 0
        day_with_signal = 0

        for trade_ymd, n_snaps, delta in cached_days:
            signal_indices = []
            long_count = 0
            short_count = 0
            prev_signal = 0
            episode_count = 0
            current_run = 0
            for i in range(x_window, n_snaps):
                total_eligible += 1
                z = latest_zscore(delta[i - short_window : i])
                current_signal = 0
                if z > threshold:
                    signal_indices.append(i)
                    long_count += 1
                    current_signal = 1
                elif z < -threshold:
                    signal_indices.append(i)
                    short_count += 1
                    current_signal = -1

                if current_signal != 0:
                    if prev_signal != current_signal:
                        episode_count += 1
                        if current_run > 0:
                            run_length_list.append(current_run)
                        current_run = 1
                    else:
                        current_run += 1
                else:
                    if current_run > 0:
                        run_length_list.append(current_run)
                        current_run = 0
                prev_signal = current_signal

            total = long_count + short_count
            total_signals += total
            long_signals += long_count
            short_signals += short_count
            total_episodes += episode_count
            daily_counts.append(total)
            daily_long_counts.append(long_count)
            daily_short_counts.append(short_count)
            daily_episode_counts.append(episode_count)
            if total > 0:
                day_with_signal += 1
            if len(signal_indices) >= 2:
                gap_list.extend(np.diff(signal_indices).tolist())
            if current_run > 0:
                run_length_list.append(current_run)

        total_hours = total_eligible / 3600 if total_eligible else float("nan")
        results.append(
            {
                "open_threshold": threshold,
                "days": len(cached_days),
                "eligible_samples": total_eligible,
                "signals": total_signals,
                "long_signals": long_signals,
                "short_signals": short_signals,
                "episodes": total_episodes,
                "signal_rate": total_signals / total_eligible if total_eligible else float("nan"),
                "signals_per_day": safe_mean(daily_counts),
                "signals_per_hour": total_signals / total_hours if total_hours else float("nan"),
                "long_per_day": safe_mean(daily_long_counts),
                "short_per_day": safe_mean(daily_short_counts),
                "episodes_per_day": safe_mean(daily_episode_counts),
                "days_with_signal_ratio": day_with_signal / len(cached_days) if cached_days else float("nan"),
                "median_gap_seconds": safe_median(gap_list),
                "mean_gap_seconds": safe_mean(gap_list),
                "median_run_seconds": safe_median(run_length_list),
                "mean_run_seconds": safe_mean(run_length_list),
            }
        )
    return results


def format_pct(value, digits=2):
    if value != value:
        return "nan"
    return f"{value * 100:.{digits}f}%"


def format_float(value, digits=2):
    if value != value:
        return "nan"
    return f"{value:.{digits}f}"


def print_y_window_summary(results):
    print("=== Y_WINDOW SENSITIVITY ===")
    header = (
        "y_window | triggered | label1_rate | up_first | down_first | no_touch | "
        "long_win | short_win | median_barrier | up_t | down_t"
    )
    print(header)
    for row in results:
        print(
            " | ".join(
                [
                    str(row["y_window"]),
                    str(row["triggered_samples"]),
                    format_pct(row["label1_rate"]),
                    format_pct(row["raw_up_first_rate"]),
                    format_pct(row["raw_down_first_rate"]),
                    format_pct(row["raw_no_touch_rate"]),
                    format_pct(row["long_win_rate"]),
                    format_pct(row["short_win_rate"]),
                    format_pct(row["median_barrier_pct"], 3),
                    format_float(row["up_first_median_t"], 1),
                    format_float(row["down_first_median_t"], 1),
                ]
            )
        )

    print("\n=== DIRECTION BREAKDOWN ===")
    header = (
        "y_window | long_count | long_up_first | long_down_first | long_no_touch | "
        "short_count | short_up_first | short_down_first | short_no_touch"
    )
    print(header)
    for row in results:
        print(
            " | ".join(
                [
                    str(row["y_window"]),
                    str(row["long_count"]),
                    format_pct(row["long_up_first_rate"]),
                    format_pct(row["long_down_first_rate"]),
                    format_pct(row["long_no_touch_rate"]),
                    str(row["short_count"]),
                    format_pct(row["short_up_first_rate"]),
                    format_pct(row["short_down_first_rate"]),
                    format_pct(row["short_no_touch_rate"]),
                ]
            )
        )


def print_threshold_summary(results):
    print("\n=== OPEN_THRESHOLD SENSITIVITY ===")
    header = (
        "open_threshold | signals | episodes | per_day | episodes/day | per_hour | long/day | short/day | "
        "signal_rate | days_with_signal | median_gap_s | median_run_s"
    )
    print(header)
    for row in results:
        print(
            " | ".join(
                [
                    format_float(row["open_threshold"], 2),
                    str(row["signals"]),
                    str(row["episodes"]),
                    format_float(row["signals_per_day"], 1),
                    format_float(row["episodes_per_day"], 1),
                    format_float(row["signals_per_hour"], 1),
                    format_float(row["long_per_day"], 1),
                    format_float(row["short_per_day"], 1),
                    format_pct(row["signal_rate"]),
                    format_pct(row["days_with_signal_ratio"]),
                    format_float(row["median_gap_seconds"], 1),
                    format_float(row["median_run_seconds"], 1),
                ]
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze delta label construction.")
    parser.add_argument("--instrument-id", default="511090")
    parser.add_argument("--date-start", default=None)
    parser.add_argument("--date-end", default=None)
    parser.add_argument("--date-source", default="all", choices=["all", "train", "test"])
    parser.add_argument("--short-window", type=int, default=60)
    parser.add_argument("--long-window", type=int, default=300)
    parser.add_argument("--vol-window", type=int, default=900)
    parser.add_argument("--open-threshold", type=float, default=2.0)
    parser.add_argument("--k-up", type=float, default=3.0)
    parser.add_argument("--k-down", type=float, default=3.0)
    parser.add_argument(
        "--y-windows",
        default="30,60,120,300,600,900",
        help="Comma separated y_window values.",
    )
    parser.add_argument(
        "--thresholds",
        default="1.0,1.5,2.0,2.5,3.0,3.5",
        help="Comma separated open_threshold values.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    date_map = base_tool.snap_ymds_load(ymd_type=args.date_source)
    dates = list(date_map[args.instrument_id])
    if args.date_start is not None:
        dates = [d for d in dates if d >= args.date_start]
    if args.date_end is not None:
        dates = [d for d in dates if d <= args.date_end]

    x_window = max(args.short_window, args.long_window)
    y_windows = [int(item) for item in args.y_windows.split(",") if item]
    thresholds = [float(item) for item in args.thresholds.split(",") if item]

    print(
        f"instrument_id={args.instrument_id}, dates={len(dates)}, "
        f"range={dates[0]}..{dates[-1]}, x_window={x_window}, short_window={args.short_window}, "
        f"vol_window={args.vol_window}, base_open_threshold={args.open_threshold}, "
        f"k_up={args.k_up}, k_down={args.k_down}"
    )

    y_results = analyze_y_windows(
        instrument_id=args.instrument_id,
        dates=dates,
        y_windows=y_windows,
        short_window=args.short_window,
        x_window=x_window,
        vol_window=args.vol_window,
        open_threshold=args.open_threshold,
        k_up=args.k_up,
        k_down=args.k_down,
    )
    print_y_window_summary(y_results)

    threshold_results = analyze_open_thresholds(
        instrument_id=args.instrument_id,
        dates=dates,
        thresholds=thresholds,
        short_window=args.short_window,
        x_window=x_window,
    )
    print_threshold_summary(threshold_results)


if __name__ == "__main__":
    main()
