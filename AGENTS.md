# AGENT.md - tactics_demo

## Project Overview

**tactics_demo** is a quantitative trading strategy research and backtesting project focused on Chinese exchange-traded ETF/money market fund instruments. It implements multiple trading tactics using 1-second market snapshot data, with both rule-based technical strategies and XGBoost ML-enhanced approaches.

## Directory Structure

```
tactics_demo/
├── tools/                       # Shared utility modules
│   ├── __init__.py
│   ├── backtesting.py           # Multi-day backtesting framework + summary stats
│   ├── plot_price.py            # Price/volume visualization utilities
│   └── Kline.py                 # Candlestick (OHLC) chart plotting
├── simple_MA/                   # Moving Average crossover strategies
│   ├── test.ipynb               # Pure SMA crossover (no ML)
│   ├── demo1.ipynb              # EMA + XGBoost
│   ├── demo1_1.ipynb            # EMA + XGBoost using WAMP (micro-price)
│   └── demo2.ipynb              # Hull MA + XGBoost
├── bollinger_bands/             # Bollinger Bands mean-reversion
│   └── test.ipynb
├── triple_barrier_method/       # Triple Barrier Method labeling + ML
│   └── test.ipynb
└── two_models/                  # Volume delta signal + ensemble
    └── delta.ipynb
```

## External Dependencies

- **`base_tool`** module at `/home/jovyan/base_demo` — CRITICAL: provides `snap_list_load()` and `backtest_quick()`
- Python packages: `pandas`, `numpy`, `matplotlib`, `mplfinance`, `xgboost`, `scikit-learn`, `joblib`, `scipy`
- Output directory: `/home/jovyan/work/backtest_result/` for pickle result files

## Data Structure (snap dict)

Each 1-second snapshot loaded via `base_tool.snap_list_load(instrument_id, trade_ymd)`:

```python
snap = {
    'time_mark': int,         # Millisecond timestamp
    'time_hms': str,          # HH:MM:SS
    'price_open': float,      # First trade price (0.0 if no trade)
    'price_low': float,
    'price_high': float,
    'price_last': float,      # Latest trade price
    'price_vwap': float,
    'num_trades': int,        # Cumulative trade count
    'bid_book': [[price, vol], ...],
    'ask_book': [[price, vol], ...],
    'buy_trade': [[price, vol], ...],
    'sell_trade': [[price, vol], ...],
    # ... plus bid/ask insert/cancel fields
}
```

**Instruments:** 511520, 511090, 518880 (primary)

## Strategy Class Interface

Every strategy must follow this pattern:

```python
class StrategyDemo():
    def __init__(self, model=None, param_dict={}) -> None:
        self.__dict__.update(param_dict)
        self.position_last = 0  # -1=short, 0=flat, 1=long
        self.model = model
        self.prev_signal = 0

    def on_snap(self, snap: dict) -> None:
        # Process each 1-second snapshot
        # Update self.position_last based on strategy logic
        pass
```

## Multi-Day Backtesting

```python
import sys
sys.path.append('/home/jovyan/work/tactics_demo/tools')
from backtesting import backtest_multi_days, backtest_summary

result_df = backtest_multi_days(
    instrument_id='511520',
    start_ymd='20260202',
    end_ymd='20260320',
    strategy=strategy_instance,
    param_dict={'name': 'strategy_name'}
)
summary = backtest_summary(result_df)
```

