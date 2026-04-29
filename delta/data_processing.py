import numpy as np
import math  # 用于替代 pd.isna
from typing import List, Dict, Any
import sys

sys.path.append("/home/jovyan/base_demo")
import base_tool

from .features import latest_zscore, create_feature


class TrainValidTest:
    def __init__(self, snap_list, param_dict, feature_func, y_func):
        if param_dict is not None:
            self.__dict__.update(param_dict)

        # 设置默认值
        self.x_window = getattr(self, "x_window", 1)
        self.y_window = getattr(self, "y_window", 1)
        self.short_window = getattr(self, "short_window", 5)
        self.long_window = getattr(self, "long_window", 20)
        self.vol_window = getattr(
            self, "vol_window", 900
        )  # 添加波动率窗口参数，默认900

        self.snap_list = snap_list.copy()
        self.create_feature = feature_func
        self.create_y = y_func

        # 使用列表推导式替代，保持纯 Python/NumPy 风格
        bid_arr = np.array(
            [sum(vol for _, vol in row["buy_trade"]) for row in self.snap_list]
        )
        ask_arr = np.array(
            [sum(vol for _, vol in row["sell_trade"]) for row in self.snap_list]
        )

        # delta 保持为列表或数组均可
        self.delta = (bid_arr - ask_arr).tolist()

        # 提取价格序列用于波动率计算
        self.prices = np.array([row["price_last"] for row in self.snap_list])

    def samples(self):
        feature_records = []
        labels = []
        n = len(self.snap_list)
        stride = getattr(self, "stride", 1)

        # 决策时点定义为“看到第 i 秒结束后的 snapshot 之后”。
        # 因此：
        # 1. 特征与 trigger 都可以使用 snap[i]
        # 2. label 必须从 snap[i + 1] 开始，避免把当前秒当作可交易未来
        for i in range(self.x_window - 1, n - self.y_window, stride):
            flag, category = self.trigger(i)
            if not flag:
                continue

            x_dict = self.create_feature(
                self.snap_list[i - self.x_window + 1 : i + 1], self.short_window
            )

            # 波动率窗口与特征时点对齐，包含当前秒末 snapshot。
            start_idx = max(0, i - self.vol_window + 1)
            price_window = self.prices[start_idx : i + 1]
            if len(price_window) > 0:
                mean_price = np.mean(price_window)
                if mean_price != 0:
                    volatility = np.std(price_window) / mean_price
                else:
                    volatility = 0.0
            else:
                volatility = 0.0
            y_val = self.create_y(
                self.snap_list[i + 1 : i + 1 + self.y_window],
                volatility,
                self.k_up,
                self.k_down,
                category,
            )
            feature_records.append(x_dict)
            labels.append(y_val)

        # 如果没有数据，返回空列表或空数组
        if not feature_records:
            return [], []

        # 不再转换为 DataFrame，直接返回 list of dicts 和 list
        # 如果后续模型需要矩阵，可以在 samples_from_dates 统一转换
        return feature_records, labels

    def trigger(self, time):
        start_idx = max(0, time - self.short_window + 1)
        std_delta = latest_zscore(self.delta[start_idx : time + 1])

        if std_delta > self.open_threshold:
            return True, 1
        elif std_delta < -self.open_threshold:
            return True, -1
        else:
            return False, 0


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

            # 使用 list.extend 替代 pd.concat，效率更高
            X_all_list.extend(X_day)
            y_all_list.extend(y_day)

            print(f"{date}: 产生 {len(X_day)} 个样本")
        except Exception as e:
            print(f"{date}: 加载失败 - {e}")

    # 如果需要最终交给模型（如 XGBoost/LightGBM）
    # 通常需要把 list of dicts 转换为二维 NumPy 数组
    if X_all_list:
        # 提取所有特征名（假设所有字典键一致）
        feature_names = list(X_all_list[0].keys())
        X_total = np.array([[row[col] for col in feature_names] for row in X_all_list])
        y_total = np.array(y_all_list)
    else:
        X_total = np.array([])
        y_total = np.array([])
        feature_names = []

    return X_total, y_total, feature_names


def create_y(snap_slice, volatility, k_up, k_down, category):
    t_up = None
    t_down = None

    start = snap_slice[0]["price_last"]

    # 使用 math.isnan 替代 pd.isna
    if start is None or start == 0 or (isinstance(start, float) and math.isnan(start)):
        return 0

    up = start * (1 + volatility * k_up)
    down = start * (1 - volatility * k_down)

    for i in range(1, len(snap_slice)):
        price = snap_slice[i]["price_last"]
        # 使用 math.isnan 替代 pd.isna
        if price is None or (isinstance(price, float) and math.isnan(price)):
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

    return 1 if category == label else 0
