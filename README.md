# tactics_demo

量化交易策略研究与回测项目，面向中国场内 ETF / 货币类品种的 1 秒级 snapshot 数据。仓库同时包含：

- 研究型 notebook，用于快速试验信号与参数
- 可复用的 Python 模块，用于特征生成、训练、策略执行与回测
- 多天批量回测与结果汇总工具

当前主要关注的标的包括 `511520`、`511090`、`518880`。

## 项目目标

本项目用于验证几类高频/准高频交易思路：

- 规则策略：如均线、布林带等
- 机器学习增强策略：如 `delta`、`triple_barrier_method`
- 因子检验与滚动验证：如 `alpha_test`

输入数据来自外部 `base_tool.snap_list_load()`，核心输出是策略逐秒仓位、单日盈亏序列、多日汇总表现以及训练好的模型文件。

## 当前目录结构

```text
tactics_demo/
├── README.md
├── AGENTS.md
├── CHANGELOG.md
├── tools/                    # 回测、绘图、结果保存等公共工具
├── simple_MA/                # 均线策略 notebook
├── bollinger_bands/          # 布林带策略 notebook
├── delta/                    # volume delta + XGBoost 的模块化实现
├── triple_barrier_method/    # TBM 标签、特征、训练、策略
├── volume_profile/           # 成交量分布相关策略/特征实验
├── alpha_test/               # 因子标注与滚动验证
├── market_research/          # 临时研究 notebook
├── two_models/               # 早期双模型实验 notebook
├── order_book/               # 预留目录，当前为空
├── backtest_result/          # 回测结果输出目录（运行产物）
└── delay_result/             # 延迟测试结果目录（运行产物）
```

## 关键模块

### `tools/`

- `backtest_quick.py`：单日快速回测，读取 `position_dict` 并输出逐秒盈亏
- `single_day_backtest.py`：单日策略执行辅助逻辑
- `multi_day_backtest.py`：多天批量回测与汇总
- `parallel_backtest_simple.py`：并行回测辅助
- `delay_stability_test.py`：开仓延迟稳定性检验
- `plot_price.py`、`Kline.py`：绘图工具
- `result_saver.py`：结果持久化

### `delta/`

较完整的模块化策略目录，包含：

- `features.py`：特征构造
- `data_processing.py`：样本生成与标签处理
- `train.py`：训练、评估、模型存取
- `strategy.py`：在线策略逻辑
- `main.py`：训练入口示例
- `models/`：模型抽象与工厂

### `triple_barrier_method/`

结构与 `delta/` 类似，围绕 TBM 标签法组织训练与策略。

### 研究型目录

- `simple_MA/`
- `bollinger_bands/`
- `two_models/`
- `market_research/`
- `volume_profile/`

这些目录以 notebook 为主，适合原型验证，但工程化程度不一致。

## 外部依赖

### 1. 外部代码依赖

仓库依赖外部模块 `/home/jovyan/work/base_demo`。

当前代码中实际会通过：

```python
sys.path.append("/home/jovyan/work/base_demo")
import base_tool
```

来加载数据与部分基础工具。关键函数：

- `base_tool.snap_list_load(instrument_id, trade_ymd)`
- 某些场景下也会调用 `base_tool.backtest_quick`

如果这个目录不存在，训练和回测都无法运行。

### 2. Python 依赖

常用依赖包括：

- `pandas`
- `numpy`
- `matplotlib`
- `mplfinance`
- `xgboost`
- `scikit-learn`
- `joblib`
- `scipy`

仓库目前还没有 `requirements.txt` 或 `pyproject.toml`，需要手动保证环境一致。

## 数据格式

`base_tool.snap_list_load(instrument_id, trade_ymd)` 返回的 `snap_list` 中，每个 `snap` 大致包含：

```python
snap = {
    "time_mark": int,
    "time_hms": str,
    "price_open": float,
    "price_low": float,
    "price_high": float,
    "price_last": float,
    "price_vwap": float,
    "num_trades": int,
    "bid_book": [[price, vol], ...],
    "ask_book": [[price, vol], ...],
    "buy_trade": [[price, vol], ...],
    "sell_trade": [[price, vol], ...],
}
```

其中 `bid_book` / `ask_book` 是盘口，策略通常在 `on_snap()` 中逐条消费这些 snapshot。

## 策略接口约定

所有可接入 `multi_day_backtest.backtest_multi_days()` 的策略，建议遵循以下接口：

```python
class StrategyDemo:
    def __init__(self, model=None, param_dict=None):
        if param_dict is None:
            param_dict = {}
        self.__dict__.update(param_dict)
        self.position_last = 0  # -1=short, 0=flat, 1=long
        self.model = model
        self.prev_signal = 0

    def on_snap(self, snap: dict) -> None:
        # 根据最新 snapshot 更新 self.position_last
        pass
```

回测时，框架会逐条调用 `on_snap()`，并把每个时刻的 `position_last` 收集为 `position_dict`。

## 快速上手

### 单日回测

```python
import sys

sys.path.append("/home/jovyan/work/tactics_demo/tools")
from backtest_quick import backtest_quick

position_dict = {
    93000000: 0,
    93000001: 1,
    93000002: 1,
}

result_df = backtest_quick(
    instrument_id="511520",
    trade_ymd="20260319",
    strategy_name="demo_strategy",
    position_dict=position_dict,
    remake=True,
)
```

### 多天回测

```python
import sys

sys.path.append("/home/jovyan/work/tactics_demo/tools")
from multi_day_backtest import backtest_multi_days, backtest_summary
from delta.strategy import StrategyDemo

param_dict = {
    "name": "delta_v1",
    "short_window": 60,
    "long_window": 300,
    "open_threshold": 2,
    "close_threshold": 0,
    "confidence_threshold": 0.4,
}

result_df = backtest_multi_days(
    instrument_id="511520",
    start_ymd="20260202",
    end_ymd="20260320",
    StrategyClass=StrategyDemo,
    model=None,
    param_dict=param_dict,
    official=False,
    delay_snaps=0,
)

summary = backtest_summary(result_df)
print(summary)
```

### 训练模块化策略

示例入口：

- `python -m delta.main`
- `python -m triple_barrier_method.main`

这两个入口都会：

- 读取交易日列表
- 生成训练/验证/测试集
- 训练模型
- 评估并保存模型
- 构造策略实例

## 输出位置

运行结果默认写到以下目录：

- 仓库内 `backtest_result/`
- 仓库内 `delay_result/`

不同模块目前对输出路径的使用还不完全统一，运行前最好确认目标目录。

## 当前状态

项目目前处于“研究代码 + 部分工程化模块并存”的阶段：

- `delta/`、`triple_barrier_method/` 已经有较完整的模块结构
- `simple_MA/`、`bollinger_bands/` 等仍以 notebook 为主
- 顶层尚未提供统一依赖声明、测试框架与标准 CLI

因此更适合研究和迭代，不适合作为直接部署的生产系统。
