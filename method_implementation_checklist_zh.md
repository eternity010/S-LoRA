# AAAI 论文方法实现核对清单

本文档用于核对当前 AAAI 论文稿中已经描述的方法模块是否与实际代码实现一致。建议在查看代码时逐项补充“实现位置、实际逻辑、参数设置、是否需要修改论文”。

## 0. 总体框架

**论文声称**

- 系统包含 centralized control plane 和多个 worker。
- 请求到达后，control plane 读取 worker 状态并立即做在线路由。
- 调度器使用 worker queue、adapter cache、capacity 等状态计算路由决策。

**需要在代码中确认**

- 请求进入调度器的位置。
- 调度器如何获取每个 worker 的状态。
- worker 状态是实时上报，还是由调度器本地维护。
- dispatch 后是否更新 queue、cache、RWPT 等状态。
- 路由决策是同步完成还是异步完成。

## 1. 请求属性提取

**论文声称**

每个请求表示为：

```latex
x=(a_x,l_x,r_x)
```

其中：

- `a_x`：目标 adapter。
- `l_x`：输入长度。
- `r_x`：目标 LoRA adapter 的 rank。

**需要在代码中确认**

- adapter id 从哪里读取。
- input length 是 prompt token 数、字符数，还是其他长度。
- LoRA rank 是从 adapter metadata 读取，还是从配置表写死。
- 实验中是否真的存在不同 rank 的 adapter。
- rank 信息是否在路由决策前可用。

## 2. Worker 状态维护

**论文声称**

每个 worker `W_i` 维护：

- 本地等待队列 `Q_i`。
- 本地 adapter cache `C_i`。
- base model instance。

**需要在代码中确认**

- queue length 如何统计。
- pending requests 是否可被 scheduler 读取。
- adapter cache hit/miss 如何判断。
- adapter cache 的状态是否准确同步到 control plane。
- 请求完成后 queue/cache 状态是否更新。

## 3. RWPT 负载建模

**论文声称**

```latex
RWPT_i=\sum_{x_j\in Q_i} l_{x_j}\cdot(1+\gamma r_{x_j})
```

RWPT 用于估计每个 worker 当前 pending load。

**需要在代码中确认**

- 是否真的为每个 worker 维护 `RWPT_i`。
- `RWPT_i` 是每次路由时扫描队列计算，还是增量维护。
- 请求加入队列后是否增加对应 RWPT。
- 请求完成后是否从 RWPT 中扣除。
- `gamma` 的实际取值是什么。
- 是否使用论文中的 `gamma=2/(3d)`，还是使用实验调参值。
- 如果使用调参值，论文中是否需要改成 “tunable coefficient” 为主。

## 4. 增量 RWPT

**论文声称**

```latex
\Delta RWPT(x)=l_x\cdot(1+\gamma r_x)
```

调度时比较的是候选 worker 接收该请求后的 post-assignment load。

**需要在代码中确认**

- 路由时是否计算 `\Delta RWPT(x)`。
- 是否使用 `RWPT_i + \Delta RWPT(x)` 比较候选 worker。
- 是否错误地只比较当前队列负载，而没有考虑新请求。
- `\Delta RWPT(x)` 是否与 worker 无关。

## 5. Cache-Aware Routing Score

**论文声称**

```latex
Score(x,W_i)=
w_l\cdot \frac{RWPT_i+\Delta RWPT(x)}{Cap_i}
-w_c\cdot H_i(x)
```

最终选择：

```latex
R(x)=\arg\min_{W_i\in\mathcal{W}} Score(x,W_i)
```

**需要在代码中确认**

- 是否遍历所有 worker 计算 score。
- `H_i(x)` 是否表示目标 adapter 是否已在 worker cache 中。
- cache hit 是否为 0/1 值。
- `w_l` 和 `w_c` 的具体值。
- 是否选择 score 最小的 worker。
- tie-breaking 如何处理。
- 是否存在其他优先规则覆盖 score，例如优先空队列、优先 cache hit 等。

## 6. Worker Capacity `Cap_i`

**论文声称**

- `Cap_i` 表示 worker 的 normalized serving capacity。
- 同构 worker 下，`Cap_i=1`。
- 异构 worker 下，可由 profiling 得到。

**需要在代码中确认**

- 当前实验是否全部使用同构 GPU，例如 3 张 RTX 3090。
- 代码中是否真的有 `Cap_i` 概念。
- 如果没有 capacity 概念，论文应强调当前 prototype 使用 homogeneous setting，因此 `Cap_i=1`。
- 如果未来加入异构实验，需要记录如何 profiling：
  - tokens/s；
  - requests/s；
  - prefill throughput；
  - normalized throughput。

## 7. Hotspot Adapter Handling

**论文声称**

论文当前描述了 hotspot adapter handling：

1. 检测 congested worker。
2. 找出该 worker 队列中贡献 RWPT 最大的 adapter。
3. 将该 adapter replica 放到低负载且有显存的 worker。

拥塞阈值：

```latex
T_{\mathrm{cong},i}=\frac{w_c}{w_l}\cdot Cap_i
```

热点 adapter：

```latex
a_i^*=\arg\max_a
\sum_{\substack{x_j\in Q_i\\ a_{x_j}=a}}
l_{x_j}\cdot(1+\gamma r_{x_j})
```

副本放置：

```latex
W^*=\arg\min_{W_k\in\mathcal{C}(a_i^*)}
\frac{RWPT_k}{Cap_k}
```

**需要重点在代码中确认**

- 是否真的实现了 hotspot detection。
- 是否真的计算 `T_{\mathrm{cong},i}`。
- 是否真的按 adapter 聚合队列中的 RWPT 贡献。
- 是否真的动态复制 adapter。
- 是否存在 replica 上限 `N_max`。
- 是否有 cooldown interval。
- replica creation 是同步还是异步。
- 如果没有完整实现，论文中的 4.4 需要降级为 prototype support 或 future extension。

## 8. Runtime and Memory Support

**论文声称**

每个 worker 有轻量本地 memory manager，用于：

- 记录 cached adapters。
- 记录 free memory。
- adapter 不在显存时加载。
- 显存不足时 evict low-reuse inactive adapters。
- 尽量保护当前队列中仍有 pending requests 的 adapter。

**需要在代码中确认**

- 是否真的有 adapter eviction。
- eviction 策略是什么：
  - LRU；
  - LFU；
  - random；
  - 手写规则；
  - 没有 eviction。
- 是否读取 free GPU memory。
- 是否保护当前队列中仍有 pending requests 的 adapter。
- adapter load/cache 状态是否会影响 routing score。
- 如果只是简单 cache admission/eviction，论文应避免写成完整 memory manager。

## 9. Baseline 相关实现

虽然实验部分尚未定稿，但方法一致性也需要确认 baseline 是否和论文口径一致。

**需要在代码中确认**

- Round-Robin 是否按请求轮转。
- Queue-Length baseline 是否选择 queue length 最小的 worker。
- Pending-Token baseline 是否选择 pending token 最小的 worker。
- RWPT baseline 是否只加 rank-aware load，不加 cache term。
- Full RWPT 是否同时包含 RWPT 和 cache-aware routing。
- 各 baseline 是否共享相同 worker/cache/arrival workload。

## 10. 当前最可能需要修改论文的风险点

### 高风险

- `4.4 Hotspot Adapter Handling`
  - 如果代码没有动态 adapter replica，就不能写成已实现机制。

- `4.5 Runtime and Memory Support`
  - 如果代码没有 eviction/memory manager，就需要弱化表述。

### 中风险

- `Cap_i`
  - 如果代码没有 capacity，论文应明确当前同构设置下全部为 `1`。

- `gamma`
  - 如果代码使用经验参数，而不是 `2/(3d)`，论文应强调 `gamma` 是 tunable coefficient。

### 低风险

- `RWPT_i`
  - 如果实现是等价的增量维护或扫描计算，都可以接受，只需要在实现章节说明。

- `H_i(x)`
  - 如果 cache hit 判断真实存在，论文表述基本安全。

## 11. 建议你去代码中优先找的入口

建议按以下顺序找代码：

1. Scheduler / router 类或函数。
2. Worker state / worker manager。
3. Adapter cache 状态维护。
4. Request 数据结构。
5. Baseline scheduler 实现。
6. Hotspot / replica / adapter placement 相关代码。
7. Memory eviction / GPU memory monitor 相关代码。

## 12. 填写模板

可以按下面格式逐项补充：

```markdown
### 模块名称

- 论文位置：
- 代码位置：
- 当前实现：
- 与论文是否一致：
- 需要修改代码：
- 需要修改论文：
- 备注：
```

