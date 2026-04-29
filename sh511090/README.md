# sh511090

`sh511090/` 用于对 `511090` 的 1 秒级 snapshot 做基础统计分析，目录按“代码 / 数据 / notebook”拆分，避免 notebook 和分析逻辑混在一起。

## 目录约定

```text
sh511090/
├── README.md
├── code/          # 生成统计结果的 Python 脚本
├── data/          # 脚本输出的 CSV / sample 等分析产物
└── notebooks/     # 只读取 data/ 的展示型 notebook
```

约定如下：

- `code/` 放可复用分析脚本，不把核心逻辑写死在 notebook 里。
- `data/` 只放运行产物，例如 `headline.csv`、`spread.csv`、`sample.csv`、`valid_dates.csv`。
- `notebooks/` 默认只消费 `data/` 里的结果，保证 notebook 可以直接基于已跑出的代码产物运行。
- notebook 如果需要改分析口径，应先改 `code/`，重新产出 `data/`，再刷新 notebook。

## 当前分析范围

当前基础分析脚本是 [code/analyze.py](/home/jovyan/work/tactics_demo/sh511090/code/analyze.py)，默认分析区间：

- `start_ymd=20260101`
- `end_ymd=20260428`

脚本不会假设每个工作日都有数据，而是会自动筛选 `base_tool.snap_list_load("511090", trade_ymd)` 返回非空 snapshot 的交易日。

## 运行方式

先生成统计结果：

```bash
/opt/conda/bin/python sh511090/code/analyze.py
```

脚本会把以下文件写入 `sh511090/data/`：

- `headline.csv`
- `spread.csv`
- `price.csv`
- `turnover.csv`
- `trade.csv`
- `price_change.csv`
- `depth.csv`
- `activity.csv`
- `daily.csv`
- `sample.csv`
- `valid_dates.csv`

然后打开 [notebooks/basic_stats_from_csv.ipynb](/home/jovyan/work/tactics_demo/sh511090/notebooks/basic_stats_from_csv.ipynb) 直接运行即可。这个 notebook 只依赖 `data/` 下 CSV，不会再次拉原始 snapshot。

## 指标口径

当前基础分析覆盖：

- 价格：`price_last`、`mid_price`
- 价差：`spread_ticks`
- 成交：每秒 `turnover`、`trade_volume`、`trade_count_delta`
- 深度：`bid1_vol`、`ask1_vol`、`l1_depth`、`l5_depth`
- 盘口不平衡：`imbalance_l1 = (bid1_vol - ask1_vol) / (bid1_vol + ask1_vol)`
- 日内活跃度：按 30 分钟分桶统计成交额占比、活跃秒占比、平均 spread 等

## 依赖说明

脚本依赖外部 `base_demo`：

- `/home/jovyan/work/base_demo`
- `/home/jovyan/base_demo`

以及当前仓库环境里的：

- `pandas`

如果 `base_tool` 无法导入，说明当前 Python 环境和外部扩展模块不匹配，需要先切换到可用环境。
