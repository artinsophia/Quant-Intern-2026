from collections import deque
from typing import Dict, Any

class StrategyDemo:
    def __init__(self, param_dict=None) -> None:
        self.__dict__.update(param_dict)
        self.prev_signal = 0
        self.feature_buffer = deque(maxlen=self.x_window)


    def close(self):
        self.feature_buffer.clear()

    def __del__(self):
        self.close()

    def on_snap(self, snap: Dict[str, Any]) -> None:
        self.feature_buffer.append(snap)
        if len(self.feature_buffer) < self.x_window:
            return

                        
