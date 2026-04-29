from typing import Any, Dict

from feature import RollingFactorExtractor


class StrategyDemo:
    def __init__(self, model=None, param_dict=None) -> None:
        if param_dict is None:
            param_dict = {}
        self.__dict__.update(param_dict)

        self.model = model

        self.window_size = int(getattr(self, "window_size", 60))

        # CUSUM 参数
        self.drift = float(getattr(self, "drift", 0.0))
        self.threshold = float(getattr(self, "threshold", 0.8))
        self.adaptive_threshold = bool(getattr(self, "adaptive_threshold", True))
        self.drift_sigma = float(getattr(self, "drift_sigma", self.drift))
        self.threshold_sigma = float(getattr(self, "threshold_sigma", self.threshold))
        self.entry_drift = float(getattr(self, "entry_drift", self.drift))
        self.exit_drift = float(getattr(self, "exit_drift", self.drift))
        self.entry_threshold = float(getattr(self, "entry_threshold", self.threshold))
        self.exit_threshold = float(getattr(self, "exit_threshold", self.threshold))
        self.entry_drift_sigma = float(
            getattr(self, "entry_drift_sigma", self.drift_sigma)
        )
        self.exit_drift_sigma = float(
            getattr(self, "exit_drift_sigma", self.drift_sigma)
        )
        self.entry_threshold_sigma = float(
            getattr(self, "entry_threshold_sigma", self.threshold_sigma)
        )
        self.exit_threshold_sigma = float(
            getattr(self, "exit_threshold_sigma", self.threshold_sigma)
        )
        self.vol_floor = float(getattr(self, "vol_floor", 1e-6))

        self.position_last = 0
        self.prev_signal = 0

        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.feature_extractor = RollingFactorExtractor(param_dict)

    def close(self) -> None:
        feature_extractor = getattr(self, "feature_extractor", None)
        if feature_extractor is not None:
            feature_extractor.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _reset_cusum(self) -> None:
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

    def _get_cusum_params(self, is_entry: bool) -> tuple[float, float]:
        if self.adaptive_threshold:
            if is_entry:
                return self.entry_drift_sigma, self.entry_threshold_sigma
            return self.exit_drift_sigma, self.exit_threshold_sigma

        if is_entry:
            return self.entry_drift, self.entry_threshold
        return self.exit_drift, self.exit_threshold

    def on_snap(self, snap: Dict[str, Any]) -> None:
        cusum_input = self.feature_extractor.on_snap(snap)
        if cusum_input is None:
            return

        if self.position_last == 0:
            drift, threshold = self._get_cusum_params(is_entry=True)
            self.cusum_pos = max(0.0, self.cusum_pos + cusum_input - drift)
            self.cusum_neg = min(0.0, self.cusum_neg + cusum_input + drift)

            if self.cusum_pos >= threshold:
                self.position_last = 1
                self.prev_signal = 1
                self.cusum_pos = 0.0

            elif abs(self.cusum_neg) >= threshold:
                self.position_last = -1
                self.prev_signal = -1
                self.cusum_neg = 0.0

            return

        drift, threshold = self._get_cusum_params(is_entry=False)
        self.cusum_pos = max(0.0, self.cusum_pos + cusum_input - drift)
        self.cusum_neg = min(0.0, self.cusum_neg + cusum_input + drift)

        if self.position_last == 1 and abs(self.cusum_neg) >= threshold:
            self.position_last = 0
            self.prev_signal = 0
            self._reset_cusum()

        elif self.position_last == -1 and self.cusum_pos >= threshold:
            self.position_last = 0
            self.prev_signal = 0
            self._reset_cusum()
