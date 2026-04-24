# Delta Label Analysis For 511090

本文档整理了 `delta/data_processing.py` 中 `label` 构造方式在 `511090` 上的测试接口、运行命令和当前统计结果，供后续直接复用。

## 1. 相关代码位置

- 标签构造逻辑: [delta/data_processing.py](/home/jovyan/work/tactics_demo/delta/data_processing.py)
- 分析脚本: [tools/analyze_delta_label.py](/home/jovyan/work/tactics_demo/tools/analyze_delta_label.py)

当前 `delta` 的标签逻辑分两步：

1. `trigger(i)` 用 `delta` 的 `zscore` 判断当前是否触发信号，以及方向 `category in {1, -1}`
2. `create_y(...)` 在未来 `y_window` 内看价格先碰上障碍还是下障碍，然后压成二分类：

```python
return 1 if category == label else 0
```

也就是说，模型学到的不是三分类 `{-1, 0, 1}`，而是“当前触发方向在未来是否先到对应 barrier”。

## 2. 测试脚本接口

分析脚本入口：

```bash
/opt/conda/bin/python3.13 tools/analyze_delta_label.py --instrument-id 511090 --date-source all
```

### 2.1 参数说明

- `--instrument-id`
  标的代码，默认 `511090`
- `--date-source`
  交易日来源，可选 `all`、`train`、`test`
- `--date-start`
  起始日期，例如 `20251201`
- `--date-end`
  结束日期，例如 `20260409`
- `--short-window`
  `trigger` 里 `zscore` 的窗口，默认 `60`
- `--long-window`
  用于得到 `x_window=max(short_window,long_window)`，默认 `300`
- `--vol-window`
  计算 barrier 波动率的历史窗口，默认 `900`
- `--open-threshold`
  基准 `trigger` 阈值，默认 `2.0`
- `--k-up`
  上障碍系数，默认 `3.0`
- `--k-down`
  下障碍系数，默认 `3.0`
- `--y-windows`
  逗号分隔的未来窗口列表，默认 `30,60,120,300,600,900`
- `--thresholds`
  逗号分隔的 `open_threshold` 列表，默认 `1.0,1.5,2.0,2.5,3.0,3.5`

### 2.2 示例

只看 2026-02 到 2026-04：

```bash
/opt/conda/bin/python3.13 tools/analyze_delta_label.py \
  --instrument-id 511090 \
  --date-source all \
  --date-start 20260201 \
  --date-end 20260409
```

只改 `y_window` 网格：

```bash
/opt/conda/bin/python3.13 tools/analyze_delta_label.py \
  --instrument-id 511090 \
  --y-windows 120,300,600,900,1200
```

只改 `open_threshold` 网格：

```bash
/opt/conda/bin/python3.13 tools/analyze_delta_label.py \
  --instrument-id 511090 \
  --thresholds 1.5,2.0,2.5,3.0,3.5,4.0
```

改 barrier 尺度：

```bash
/opt/conda/bin/python3.13 tools/analyze_delta_label.py \
  --instrument-id 511090 \
  --k-up 2.0 \
  --k-down 2.0 \
  --y-windows 120,300,600,900
```

## 3. 输出字段解释

### 3.1 `Y_WINDOW SENSITIVITY`

- `triggered`
  满足 `open_threshold` 的样本数
- `label1_rate`
  当前二分类标签里 `1` 的比例，也就是“触发方向最终正确”的比例
- `up_first`
  原始 triple barrier 结果里，先碰上障碍的比例
- `down_first`
  原始 triple barrier 结果里，先碰下障碍的比例
- `no_touch`
  在 `y_window` 内上下障碍都没碰到的比例
- `long_win`
  多头触发后的 `label=1` 比例
- `short_win`
  空头触发后的 `label=1` 比例
- `median_barrier`
  barrier 距离中位数，单位是相对价格比例
- `up_t`
  所有上障碍触达样本的触达时间中位数，单位秒
- `down_t`
  所有下障碍触达样本的触达时间中位数，单位秒

### 3.2 `DIRECTION BREAKDOWN`

按多头、空头分别拆解：

- `long_up_first`
  多头触发后先碰上障碍的比例
- `long_down_first`
  多头触发后先碰下障碍的比例
- `long_no_touch`
  多头触发后未触障比例
- `short_up_first`
  空头触发后先碰上障碍的比例
- `short_down_first`
  空头触发后先碰下障碍的比例
- `short_no_touch`
  空头触发后未触障比例

### 3.3 `OPEN_THRESHOLD SENSITIVITY`

- `signals`
  逐秒触发点总数
- `episodes`
  连续同向触发区间只记一次后的事件数
- `per_day`
  日均逐秒触发点
- `episodes/day`
  日均 episode 数
- `per_hour`
  小时级触发频率
- `long/day`
  日均多头触发点
- `short/day`
  日均空头触发点
- `signal_rate`
  触发点占所有可评估秒级样本的比例
- `days_with_signal`
  有触发的交易日比例
- `median_gap_s`
  相邻触发点间隔中位数，单位秒
- `median_run_s`
  连续同向触发的持续时间中位数，单位秒

## 4. 本次测试配置

本次统计使用如下配置：

- 标的: `511090`
- 日期范围: `20250901` 到 `20260409`
- 交易日数量: `102`
- `short_window=60`
- `long_window=300`
- `x_window=300`
- `vol_window=900`
- `open_threshold=2.0`
- `k_up=3.0`
- `k_down=3.0`
- `y_windows=30,60,120,300,600,900`
- `thresholds=1.0,1.5,2.0,2.5,3.0,3.5`

执行命令：

```bash
/opt/conda/bin/python3.13 tools/analyze_delta_label.py --instrument-id 511090 --date-source all
```

## 5. 本次测试结果

### 5.1 Y window 敏感性

| y_window | triggered | label1_rate | up_first | down_first | no_touch | long_win | short_win | median_barrier | up_t | down_t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 94776 | 0.16% | 0.13% | 0.13% | 99.74% | 0.15% | 0.17% | 0.055% | 22.0 | 22.0 |
| 60 | 94537 | 0.79% | 0.63% | 0.76% | 98.61% | 0.72% | 0.87% | 0.055% | 43.0 | 44.0 |
| 120 | 94109 | 2.98% | 2.58% | 2.76% | 94.66% | 2.89% | 3.07% | 0.055% | 84.0 | 83.0 |
| 300 | 93037 | 10.97% | 9.88% | 10.67% | 79.44% | 10.50% | 11.45% | 0.055% | 180.0 | 180.0 |
| 600 | 91008 | 21.33% | 19.87% | 21.06% | 59.06% | 20.69% | 22.00% | 0.055% | 304.0 | 307.0 |
| 900 | 88978 | 27.82% | 26.34% | 27.42% | 46.24% | 27.13% | 28.55% | 0.055% | 398.0 | 397.0 |

### 5.2 多空方向拆解

| y_window | long_count | long_up_first | long_down_first | long_no_touch | short_count | short_up_first | short_down_first | short_no_touch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 48566 | 0.15% | 0.09% | 99.76% | 46210 | 0.11% | 0.17% | 99.71% |
| 60 | 48440 | 0.72% | 0.65% | 98.62% | 46097 | 0.53% | 0.87% | 98.61% |
| 120 | 48226 | 2.89% | 2.47% | 94.64% | 45883 | 2.25% | 3.07% | 94.68% |
| 300 | 47614 | 10.50% | 9.92% | 79.58% | 45423 | 9.24% | 11.45% | 79.31% |
| 600 | 46537 | 20.69% | 20.17% | 59.14% | 44471 | 19.02% | 22.00% | 58.98% |
| 900 | 45462 | 27.13% | 26.34% | 46.53% | 43516 | 25.51% | 28.55% | 45.94% |

### 5.3 Open threshold 敏感性

| open_threshold | signals | episodes | per_day | episodes/day | per_hour | long/day | short/day | signal_rate | days_with_signal | median_gap_s | median_run_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 253978 | 219481 | 2490.0 | 2151.8 | 635.6 | 1248.5 | 1241.5 | 17.66% | 100.00% | 3.0 | 1.0 |
| 1.50 | 152822 | 137927 | 1498.3 | 1352.2 | 382.5 | 760.3 | 737.9 | 10.62% | 100.00% | 5.0 | 1.0 |
| 2.00 | 94923 | 87914 | 930.6 | 861.9 | 237.6 | 476.9 | 453.7 | 6.60% | 100.00% | 9.0 | 1.0 |
| 2.50 | 59720 | 56286 | 585.5 | 551.8 | 149.5 | 302.5 | 283.0 | 4.15% | 100.00% | 15.0 | 1.0 |
| 3.00 | 38256 | 36573 | 375.1 | 358.6 | 95.7 | 194.1 | 180.9 | 2.66% | 100.00% | 25.0 | 1.0 |
| 3.50 | 24620 | 23796 | 241.4 | 233.3 | 61.6 | 125.8 | 115.6 | 1.71% | 100.00% | 46.0 | 1.0 |

## 6. 结果解读

### 6.1 当前 label 的主要问题

在 `511090` 上，当前配置下最主要的问题不是方向偏差，而是：

- 大量样本在 `y_window` 内根本触不到 barrier
- `y_window<=300` 时，这个问题尤其严重

例如：

- `y_window=300` 时，`no_touch=79.44%`
- `y_window=600` 时，`no_touch=59.06%`
- `y_window=900` 时，`no_touch=46.24%`

因此，当前二分类标签里大量 `0` 实际上并不是“方向判断错”，而是“未来窗口内根本没走到 barrier”。

### 6.2 barrier 尺度

当前配置下，`median_barrier` 约为 `0.055%`，大约 `5.5bp`。

这意味着：

- barrier 本身并不算特别远
- 但 `511090` 的短期价格波动和当前 `y_window` 组合下，仍然经常无法在未来窗口内完成触达

### 6.3 y_window 建议

如果保持 `k_up=k_down=3` 不变：

- `30/60/120` 太短，基本不可用
- `300` 仍然偏短，未触障比例过高
- `600-900` 是更合理的观察区间

如果业务上必须坚持较短持有窗口，比如 `300` 以内，则更合理的优化方向不是继续缩短 `y_window`，而是下调 `k_up/k_down`。

### 6.4 open_threshold 建议

`open_threshold` 对信号密度影响非常稳定：

- `2.0` 仍然很密，约 `930` 个触发点/天
- `2.5` 开始进入相对可控区间，约 `586` 个触发点/天
- `3.0` 更稀疏，约 `375` 个触发点/天

如果希望降低噪音、减少过多触发，通常可优先考虑：

- `open_threshold=2.5`
- `open_threshold=3.0`

## 7. 后续建议

如果后续要继续做参数筛选，建议按下面顺序推进：

1. 固定 `open_threshold`，扫描 `k_up/k_down`
2. 在合理 barrier 尺度上，再比较 `y_window=300/600/900`
3. 最后再把筛出来的组合接回真实回测，观察收益、换手和持仓时长

更具体地说：

- 如果目标是提高 `label=1` 占比，优先调小 `k_up/k_down`
- 如果目标是让 label 更符合中短线持有逻辑，优先比较 `y_window=600` 和 `900`
- 如果目标是降低信号噪声，优先看 `open_threshold=2.5` 到 `3.0`
