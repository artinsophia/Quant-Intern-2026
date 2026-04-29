from collections import deque
import itertools
from typing import Dict, Any
import joblib
from .features import latest_zscore, RollingFeatureExtractor, trade_volume_sum_topn
import numpy as np


class StrategyDemo:
    def __init__(self, model, param_dict=None) -> None:
        if param_dict is None:
            param_dict = {}
        self.__dict__.update(param_dict)
        self.stop_tighten_start = getattr(self, "stop_tighten_start", 300)
        self.stop_tighten_step = getattr(self, "stop_tighten_step", 150)
        self.stop_tighten_factor = getattr(self, "stop_tighten_factor", 0.85)
        self.stop_tighten_floor = getattr(self, "stop_tighten_floor", 0.5)
        self.adverse_cusum_threshold = getattr(self, "adverse_cusum_threshold", 5.0)
        self.adverse_cusum_drift = getattr(self, "adverse_cusum_drift", 0.2)

        self.position_last = 0

        # 检查model是否已经是加载的模型对象
        if isinstance(model, str):
            # 如果是文件路径，加载模型
            self.model = joblib.load(model)
        else:
            # 如果已经是模型对象，直接使用
            self.model = model

        self.delta_buffer = deque(maxlen=self.x_window)
        self.price_buffer = deque(maxlen=self.vol_window)
        self.feature_extractor = RollingFeatureExtractor(self.x_window, self.short_window)

        self.max_favorable_price = 0
        self.entry_price = 0
        self.prev_signal = 0
        self.holding_snap = 0
        self.entry_alpha = 0
        self.alpha_exit_count = 0
        self.close_num = 0
        self.adverse_cusum = 0.0

    def close(self):
        if hasattr(self, "delta_buffer"):
            self.delta_buffer.clear()
        if hasattr(self, "price_buffer"):
            self.price_buffer.clear()
        feature_extractor = getattr(self, "feature_extractor", None)
        if feature_extractor is not None:
            feature_extractor.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def on_snap(self, snap: Dict[str, Any]) -> None:
        price = snap.get("price_last")
        bid_book = snap.get("bid_book") or []
        ask_book = snap.get("ask_book") or []

        buy = bid_book[0][0] if bid_book else price
        sell = ask_book[0][0] if ask_book else price

        if not price:
            return
        self.price_buffer.append(price)
        if len(self.price_buffer) < self.vol_window:
            return

        delta = trade_volume_sum_topn(
            snap.get("buy_trade"), self.standard_num
        ) - trade_volume_sum_topn(snap.get("sell_trade"), self.standard_num)
        self.delta_buffer.append(delta)
        if len(self.delta_buffer) < self.x_window:
            return

        recent_delta = list(
            itertools.islice(
                self.delta_buffer,
                max(0, len(self.delta_buffer) - self.short_window),
                None,
            )
        )
        std_delta = latest_zscore(recent_delta)

        self.feature_extractor.append(snap)
        if self.feature_extractor.is_ready():
            feat_dict = self.feature_extractor.extract_all()
            values = list(feat_dict.values())
            features = np.array([values])
            proba = self.model.predict_proba(features)
            # 获取类别1的概率，兼容DataFrame和numpy数组
            if hasattr(proba, "iloc"):
                prob = proba.iloc[0, 1]  # DataFrame
            else:
                prob = proba[0, 1]  # numpy数组
        else:
            return

        rolling_std = np.std(self.price_buffer)
        self.trailing_stop = rolling_std * self.atr_multiplier

        current_signal = self.prev_signal

        # 1 秒级 snapshot，持仓计数可直接视为秒数
        if self.position_last != 0:
            self.holding_snap += 1
        if self.holding_snap > self.y_window + 300:
            current_signal = 0
        if self.position_last == 1 and std_delta > self.open_threshold and prob > self.model.best_threshold + self.open_confidence:
            self.holding_snap = 0
        if self.position_last == -1 and std_delta < -self.open_threshold and prob > self.model.best_threshold + self.open_confidence:
            self.holding_snap = 0

        if self.position_last == 1:
            adverse_move = -std_delta - self.adverse_cusum_drift
            self.adverse_cusum = max(0.0, self.adverse_cusum + adverse_move)
            if self.adverse_cusum >= self.adverse_cusum_threshold:
                current_signal = 0
        elif self.position_last == -1:
            adverse_move = std_delta - self.adverse_cusum_drift
            self.adverse_cusum = max(0.0, self.adverse_cusum + adverse_move)
            if self.adverse_cusum >= self.adverse_cusum_threshold:
                current_signal = 0

        dynamic = self.trailing_stop
        if self.position_last != 0 and self.holding_snap > self.stop_tighten_start:
            tighten_levels = (self.holding_snap - self.stop_tighten_start) // self.stop_tighten_step + 1
            tighten_ratio = max(
                self.stop_tighten_floor,
                self.stop_tighten_factor ** tighten_levels,
            )
            dynamic *= tighten_ratio

        # 回撤平仓，持仓越久止损越紧
        if self.position_last == 1:
            self.max_favorable_price = max(self.max_favorable_price, sell)
            pullback = self.max_favorable_price - sell
            if pullback >= dynamic:
                current_signal = 0

        elif self.position_last == -1:
            self.max_favorable_price = min(self.max_favorable_price, buy)
            pullback = buy - self.max_favorable_price
            if pullback >= dynamic:
                current_signal = 0


        
        # 开仓信号
        if self.position_last == 0:
            if std_delta > self.open_threshold:
                current_signal = 1
            elif std_delta < -self.open_threshold:
                current_signal = -1
        
        # 实际行为
        if current_signal != self.prev_signal:
            if self.position_last != 0:
                self.position_last = 0
                self.prev_signal = 0
                self.max_favorable_price = 0
                self.entry_price = 0
                self.holding_snap = 0
                self.adverse_cusum = 0.0

            else: # 开仓
                if (
                    prob is not None
                    and prob > self.model.best_threshold + self.open_confidence
                ):
                    self.position_last = current_signal
                    self.prev_signal = current_signal
                    self.entry_price = price
                    self.holding_snap = 0
                    self.adverse_cusum = 0.0
                    if current_signal == 1:
                        self.max_favorable_price = sell
                    else:
                        self.max_favorable_price = buy

                        


# 各个标的需要全样本回测的参数
strategy_params = {
    '511090': 
    [
        {
        'instrument_id': '511090',
        'name': f'delta_v1',
        'stride': 1,

        'short_window': 300,
        'long_window': 600,
        'y_window': 600,
        'x_window': 600,

        'open_threshold': 3,  
        'open_confidence': 0,  

        'standard_num': 1000,

        'atr_multiplier': 4,
        'vol_window': 600,

        'adverse_cusum_threshold': 10,
        'adverse_cusum_drift': 3,

        'k_up': 3,
        'k_down': 3,
        "stop_tighten_start": 6000,
        'stop_tighten_step' : 60,
        'stop_tighten_factor': 0.9,
        'stop_tighten_floor': 0.75,
        'model_path': '/home/jovyan/work/model/delta_511090_volatility_16d_600_3.pkl'
        },
        {
        'instrument_id': '511090',
        'name': f'delta_v1',
        'stride': 1,

        'short_window': 300,
        'long_window': 600,
        'y_window': 600,
        'x_window': 600,

        'open_threshold': 3,  
        'open_confidence': 0,  

        'standard_num': 1000,

        'atr_multiplier': 4,
        'vol_window': 600,

        'adverse_cusum_threshold': 7,
        'adverse_cusum_drift': 3,

        'k_up': 3,
        'k_down': 3,
        "stop_tighten_start": 6000,
        'stop_tighten_step' : 60,
        'stop_tighten_factor': 0.9,
        'stop_tighten_floor': 0.75,
        'model_path': '/home/jovyan/work/model/delta_511090_volatility_16d_600_3.pkl'
        },

        {
        'instrument_id': '511090',
        'name': f'delta_v1',
        'stride': 1,

        'short_window': 300,
        'long_window': 600,
        'y_window': 600,
        'x_window': 600,

        'open_threshold': 3,  
        'open_confidence': 0,  

        'standard_num': 1000,

        'atr_multiplier': 4,
        'vol_window': 600,

        'adverse_cusum_threshold': 5,
        'adverse_cusum_drift': 3,

        'k_up': 3,
        'k_down': 3,
        "stop_tighten_start": 6000,
        'stop_tighten_step' : 60,
        'stop_tighten_factor': 0.9,
        'stop_tighten_floor': 0.75,
        'model_path': '/home/jovyan/work/model/delta_511090_volatility_16d_600_3.pkl'
        },

    ],
}


if __name__ == '__main__':
    pass