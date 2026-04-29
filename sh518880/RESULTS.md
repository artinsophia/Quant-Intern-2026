# 518880 主要发现

## Spread 2tick 的状态特征

- `spread=2 tick` 是一个相对复杂的状态，不适合简单视为 `spread=1 tick` 的放大版。
- 相较于 `spread=1 tick`，`spread=2 tick` 状态下盘口不平衡率更低。
- 仅用不平衡率去预测 `spread=2 tick` 时刻的价格方向，效果弱于 `spread=1 tick`。
- 但 `spread=2 tick` 往往对应未来更强的价格变动，说明它对“后续波动强度”更敏感，对“即时方向”反而没那么直接。
- 2tick状态几乎不能保留，会立即回到1tick。

## 1tick 扩到 2tick 的过渡现象

- 当 `spread` 从 `1 tick` 扩大到 `2 tick` 前，大约前 `2s` 的不平衡率会明显抬升。真正进入 `2 tick` 后，不平衡率又会回到相对正常的水平。
- 在扩大前，L1 深度会不断减小，扩大后 `1s`恢复正常水平。
- 在到达 `2 tick` 的那个时刻前 `2s`，主动成交量会出现明显跳升; 扩大后的交易量水平会略低于交易前水平。

## 扩大原因和结果

- 扩大主要是买盘卖盘最优价格中的一个被消耗，也有极少数同时耗尽的。结合主动成交量，应该是主动成交而非撤单。


## 当前解释

- `spread` 扩大更像是主动成交把最优价位吃空，而不只是被动挂单撤退。
- 因此，`1tick -> 2tick` 更应被理解为一次短时流动性被消耗的事件，而不是单纯的静态宽价差状态。

### 先看 spread 是怎么合回去的

- 从盘口动作看，`2tick` 回正并不是单边压倒性地靠“回撤”或“前进”完成：
  - `advance_close` 占比约 `51.5%`
  - `rollback_close` 占比约 `48.5%`
- 这里的口径是：
  - `rollback_close`：原先退开的那一边回来，spread 被修复
  - `advance_close`：另一边继续沿原方向跟进，spread 被重新压回 `1tick`


### 真正该看的：回正之后下一次新的 `mid` move

- 把“回正当秒”剥离掉，只看 `close` 之后下一次非零 `mid_move_half_tick` 的方向：
  - `advance_close`：同方向约 `38.9%`，反方向约 `61.1%`
  - `rollback_close`：同方向约 `63.5%`，反方向约 `36.5%`

### 更合理的解释

- `advance_close` 更像是：另一边追一下，把 spread 合回去，但这一下之后更容易短线回吐。
- `rollback_close` 更像是：先把 spread 修复回去，但修复完成后，下一次新的 `mid` move 反而更容易继续原方向。
- 所以：
  - 如果问题是“回正之后下一次新的价格动作是否继续原方向”，答案并不强，而且 `advance_close` 甚至偏反向，`rollback_close` 才偏继续。
  - 这就合理多了，而且很有意思

### 再往前走一步：看回正之后的 `mid_price drift`

- 只看“下一次新的 `mid` move`”还不够，因为它只给方向，不给偏移幅度。
- 更合理的补充是直接看 `close` 之后的 `mid_price drift`，而且要明确从 `close` 秒之后开始算，避免把“回正当秒”的机械性 `mid` 变化混进去。

### 两种 drift 构造方式

- 第一种：固定非常短的时间窗，看 `close` 之后 `2s / 3s / 5s` 的累计 `mid` 偏移。
- 第二种：不按秒数，而是按 `mid_price` 后面发生了几次非零移动来算，看第 `2 / 3 / 4 / 5` 次非零 `mid move` 时，相对 `close` 的累计偏移。
- 这里统一都看 signed drift：
  - 沿扩价原方向记为正
  - 反向记为负

### 时间窗 drift：`rollback` 持续偏正，`advance` 持续偏负

- `close + 2s`
  - `rollback_close`：约 `+0.159 tick`
  - `advance_close`：约 `-0.112 tick`
- `close + 3s`
  - `rollback_close`：约 `+0.181 tick`
  - `advance_close`：约 `-0.099 tick`
- `close + 5s`
  - `rollback_close`：约 `+0.196 tick`
  - `advance_close`：约 `-0.066 tick`

- 这说明：
  - `rollback_close` 之后，价格在接下来几秒内仍然偏向原方向延续
  - `advance_close` 之后，价格反而更容易出现短线回吐
- 两者的均值差大约一直在 `0.26-0.28 tick`，不算小。

### 按 `mid move` 次数计 drift：结论没变，而且更稳

- 到第 `2` 次非零 `mid move`
  - `rollback_close`：约 `+0.159 tick`
  - `advance_close`：约 `-0.057 tick`
- 到第 `3` 次非零 `mid move`
  - `rollback_close`：约 `+0.195 tick`
  - `advance_close`：约 `-0.042 tick`
- 到第 `4` 次非零 `mid move`
  - `rollback_close`：约 `+0.185 tick`
  - `advance_close`：约 `-0.014 tick`
- 到第 `5` 次非零 `mid move`
  - `rollback_close`：约 `+0.175 tick`
  - `advance_close`：约 `-0.004 tick`

- 这组结果比固定秒数更说明问题：
  - 即使不按物理时间，而是按“后面真正发生了多少次新的价格动作”来对齐，`rollback_close` 依然明显更偏 continuation
  - `advance_close` 则主要表现为很短 horizon 上的回吐，之后逐渐回到接近 `0`

### 一个更清楚的解释框架

- `advance_close` 更像是：
  - 另一边先追一下，把 `2tick` 压回 `1tick`
  - 但这一步本身更像补价或追价完成，后面新增的价格发现并不继续支持这个方向
- `rollback_close` 更像是：
  - 原先退开的那一边先回来，把 spread 修平
  - 但真正的价格压力并没有消失，后面的独立 `mid` 变化反而更容易继续原方向

- 所以如果问题是：
  - “哪种恢复类型后，后续 `mid_price drift` 更容易延续原方向？”
  - 答案很明确：`rollback_close` 更主导 continuation，`advance_close` 更主导短线 mean reversion / 回吐。

## 恢复后 drift 的分布结论

- 上面的均值结论不是“少数极端样本”硬拉出来的假象，分布本身就有明显偏斜。
- 这次额外看了：
  - `drift_quantile_summary.csv`
  - `drift_value_distribution.csv`
- 结果显示：
  - 很多窗口的中位数确实是 `0`
  - 但 `+1 tick / -1 tick` 附近的质量分布已经明显不对称
  - 也就是说，虽然大量样本停在 `0`，一旦发生新的价格偏移，`advance_close` 和 `rollback_close` 的方向分布并不一样


