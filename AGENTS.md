# AGENTS.md - tactics_demo

## Project Overview

`tactics_demo` 是一个量化交易策略研究与回测仓库，面向中国场内 ETF / 货币类品种的 1 秒级 snapshot 数据。仓库同时包含：

- 研究型 notebook
- 模块化训练/推理代码
- 单日与多日回测工具
- 因子与延迟稳定性实验

当前代码形态是“研究项目逐步工程化”，不要假设所有目录都遵循同一套规范。

## Actual Directory Structure

```text
tactics_demo/
├── README.md
├── AGENTS.md
├── CHANGELOG.md
├── tools/
├── simple_MA/
├── bollinger_bands/
├── delta/
├── triple_barrier_method/
├── volume_profile/
├── alpha_test/
├── market_research/
├── two_models/
├── order_book/
├── backtest_result/
└── delay_result/
```

### Directory Roles

- `tools/`
  通用回测与绘图工具，是大多数策略目录复用的基础设施。

- `delta/`
  当前较完整的模块化策略目录，包含特征、样本生成、训练、策略、模型抽象。

- `triple_barrier_method/`
  基于 Triple Barrier Method 的训练与策略实现，结构接近 `delta/`。

- `volume_profile/`
  与成交量分布相关的特征和策略实验，模块化程度中等。

- `alpha_test/`
  因子打标与滚动验证。

- `simple_MA/`、`bollinger_bands/`、`two_models/`、`market_research/`
  主要为 notebook 驱动的研究目录，代码复用性有限。

- `backtest_result/`、`delay_result/`
  运行产物目录，不应视为源码的一部分。

- `order_book/`
  预留目录，当前为空。

## External Dependencies

### Critical external module

仓库严重依赖外部路径：

- `/home/jovyan/work/base_demo`

当前代码会直接通过 `sys.path.append("/home/jovyan/work/base_demo")` 加载 `base_tool`。关键能力包括：

- `snap_list_load(instrument_id, trade_ymd)`
- 某些场景下的 `backtest_quick()`

如果该路径不可用，以下功能会失效：

- snapshot 数据加载
- 训练样本生成
- 单日 / 多日回测

### Python packages

常用依赖：

- `pandas`
- `numpy`
- `matplotlib`
- `mplfinance`
- `xgboost`
- `scikit-learn`
- `joblib`
- `scipy`

仓库当前没有标准依赖清单文件，协作时不要假设环境可自动重建。

## Data Shape

每个 1 秒 snapshot 的常见结构：

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

代码中最常使用的字段包括：

- `time_mark`
- `price_last`
- `bid_book`
- `ask_book`

## Strategy Interface Contract

所有接入回测框架的策略都应尽量兼容以下接口：

```python
class StrategyDemo:
    def __init__(self, model=None, param_dict=None) -> None:
        if param_dict is None:
            param_dict = {}
        self.__dict__.update(param_dict)
        self.position_last = 0
        self.model = model
        self.prev_signal = 0

    def on_snap(self, snap: dict) -> None:
        pass
```

约定：

- `position_last` 取值为 `-1 / 0 / 1`
- 回测框架会逐秒调用 `on_snap()`
- `on_snap()` 负责更新 `position_last`
- 多天回测时，策略通常按“每天重新实例化”处理

## Backtesting Entry Points

### Single-day

常用公共入口：

- `tools/backtest_quick.py`
- `tools/single_day_backtest.py`

`backtest_quick()` 接收：

- `instrument_id`
- `trade_ymd`
- `strategy_name`
- `position_dict`
- `remake`

并返回包含 `time_mark`、`price_last`、`position`、`profits` 的结果表。

### Multi-day

`tools/multi_day_backtest.py` 提供：

- `backtest_multi_days(...)`
- `backtest_summary(...)`

典型流程：

1. 逐日加载 snapshot
2. 调用策略生成 `position_dict`
3. 进入单日回测
4. 汇总成多日 DataFrame
5. 输出累计收益图与摘要统计

## Training Entry Points

模块化训练入口主要有：

- `python -m delta.main`
- `python -m triple_barrier_method.main`

这些入口通常会：

1. 获取交易日列表
2. 划分 train / valid / test
3. 生成样本与标签
4. 训练模型
5. 评估并保存模型
6. 构造 `StrategyDemo`

## Working Conventions For Agents

### Prefer module code over notebooks

当任务涉及可复用逻辑、修 bug 或补测试时，优先修改：

- `tools/`
- `delta/`
- `triple_barrier_method/`
- `alpha_test/`
- `volume_profile/`

除非用户明确要求，否则不要把 notebook 当成主要实现载体。

### Do not assume docs are complete

虽然本文件和 `README.md` 已补充，但仓库仍处于快速演化阶段。做出结构性判断前，应以实际目录与源码为准。

### Treat output folders as generated artifacts

以下目录默认视为运行产物：

- `backtest_result/`
- `delay_result/`
- `__pycache__/`
- `.ipynb_checkpoints/`

除非用户明确要求，否则不要把这些目录当作需要维护的源码内容。


