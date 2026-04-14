import numpy as np
import pandas as pd
from typing import List, Dict, Any
import sys

sys.path.append("/home/jovyan/base_demo")
import base_tool

from .features import create_feature


class TrainValidTest:
    def __init__(self, snap_list, param_dict, feature_func, y_func):
        if param_dict is not None:
            self.__dict__.update(param_dict)

        if not hasattr(self, "x_window"):
            self.x_window = 1
        if not hasattr(self, "y_window"):
            self.y_window = 1

        self.snap_list = snap_list.copy()
        self.create_feature = feature_func
        self.create_y = y_func

    def samples(self):
        feature_records = []
        labels = []
        n = len(self.snap_list)
        stride = getattr(self, "stride", 1)

        for i in range(self.x_window, n - self.y_window, stride):
            x_dict = self.create_feature(self.snap_list[i - self.x_window : i])

            volatility = x_dict.get("volatility", 0.0)
            y_val = self.create_y(
                self.snap_list[i : i + self.y_window],
                volatility,
                self.k_up,
                self.k_down,
            )
            feature_records.append(x_dict)
            labels.append(y_val)

        if not feature_records:
            return pd.DataFrame(), pd.Series(dtype=float)

        X_all = pd.DataFrame(feature_records)
        y_all = pd.Series(labels)
        return X_all, y_all


def samples_from_dates(dates, instrument_id, param_dict, create_feature, create_y):
    X_all_list = []
    y_all_list = []

    for date in dates:
        try:
            snap_list = base_tool.snap_list_load(instrument_id, date)
            if len(snap_list) < param_dict["x_window"] + param_dict["y_window"]:
                print(f"{date}: 数据不足，跳过")
                continue
            tv = TrainValidTest(snap_list, param_dict, create_feature, create_y)
            X_day, y_day = tv.samples()
            X_all_list.append(X_day)
            y_all_list.append(y_day)
            print(f"{date}: 产生 {len(X_day)} 个样本")
        except Exception as e:
            print(f"{date}: 加载失败 - {e}")

    if X_all_list:
        X_total = pd.concat(X_all_list, axis=0, ignore_index=True)
        y_total = pd.concat(y_all_list, axis=0, ignore_index=True)
    else:
        X_total = pd.DataFrame()
        y_total = pd.Series()

    return X_total, y_total


def create_y(snap_slice, volatility, k_up, k_down):
    t_up = None
    t_down = None

    start = snap_slice[0]["price_last"]
    if start is None or start == 0 or pd.isna(start):
        return 0

    up = start * (1 + volatility * k_up)
    down = start * (1 - volatility * k_down)

    for i in range(1, len(snap_slice)):
        price = snap_slice[i]["price_last"]
        if price is None or pd.isna(price):
            continue

        if t_up is None and price >= up:
            t_up = i
        if t_down is None and price <= down:
            t_down = i

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

    return label
