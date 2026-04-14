import os
import pandas as pd
from typing import Dict, Any

from .features import create_feature


class StrategyDemo:
    def __init__(self, model, param_dict=None) -> None:
        if param_dict is None:
            param_dict = {}
        self.__dict__.update(param_dict)

        data_file = f"/home/jovyan/work/backtest_result/{self.instrument_id}_{self.trade_ymd}_{self.name}.pkl"
        try:
            if os.path.exists(data_file):
                os.remove(data_file)
        except OSError as e:
            print(f"Warning: Could not delete file {data_file}: {e}")

        self.position_last = 0
        self.model = model

        self.feature_buffer = []

        self.prev_signal = 0

    def on_snap(self, snap: Dict[str, Any]) -> None:
        price = snap.get("price_last")

        if price == 0.0 or price is None:
            return

        self.feature_buffer.append(snap)

        if len(self.feature_buffer) < self.x_window:
            self.position_last = 0
            self.prev_signal = 0
            return

        if len(self.feature_buffer) > self.x_window:
            self.feature_buffer.pop(0)

        features_dict = create_feature(self.feature_buffer[-self.x_window :])
        X_pred = pd.DataFrame([features_dict])

        probas = self.model.predict_proba(X_pred)
        confidence = (
            probas.iloc[0].max() if hasattr(probas, "iloc") else probas[0].max()
        )
        prediction = self.model.predict(X_pred)[0]

        open_confidence = getattr(self, "open_confidence", 0.7)

        if confidence > open_confidence and prediction != self.prev_signal:
            self.position_last = int(prediction)
            self.prev_signal = int(prediction)

        return
