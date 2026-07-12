# ICWS 论文实验补充计划

本文档记录当前 ICWS 论文第 6 节仍建议补充的实验内容，用于后续完善实验评估与结果分析。

## 1 当前实验基础

当前已有实验结果主要包括：

1. `RWPT`、`token_count` 和 `queue_length` 三种负载指标在不同访问偏斜程度下的对比结果；
2. `alpha=0.1`、`alpha=0.3`、`alpha=0.8` 三组访问分布设置；
3. 单卡 S-LoRA 基线结果；
4. 部分 multi-worker round-robin 基线结果；
5. 吞吐量、平均首 token 延迟、P90 首 token 延迟、平均端到端延迟、P90 端到端延迟、缓存命中率等指标。

这些结果已经能够初步支撑“`RWPT` 相比粗粒度负载指标更适合多租户 LoRA 推理服务路由”的主张，但如果希望论文实验部分更加扎实，还需要补充若干关键实验。

## 2 必须补充的实验

### 2.1 补全 multi-worker round-robin 基线（已完成）

**目的：**  
补齐多 Worker 场景下最基本的请求分发基线，使论文能够区分“多 Worker 架构带来的收益”和“`RWPT` 路由带来的额外收益”。

**当前缺口：**

- 已有 `alpha=0.1` 和 `alpha=0.3` 的 round-robin 结果；
- 缺少 `alpha=0.8` 的 round-robin 结果；
- 当前 round-robin 实验时长为 120s，而主实验为 240s，最好统一实验设置。

**建议补充设置：**

| 策略 | alpha | req_rate | duration | num_workers |
|---|---:|---:|---:|---:|
| round-robin | 0.1 | 6.0 | 240s | 3 |
| round-robin | 0.3 | 6.0 | 240s | 3 |
| round-robin | 0.8 | 6.0 | 240s | 3 |

**论文中用途：**

- 用于整体性能对比；
- 说明简单均匀分发无法利用 Adapter 缓存亲和性与负载状态；
- 与 `RWPT-Based Routing` 共同证明“架构 + 路由”带来的收益。

**完成说明：**

- 已整理出 `240s` 的 round-robin 三组结果（`alpha=0.1 / 0.3 / 0.8`）；
- 导出文件位于 `benchmarks/exported_results/dp_roundrobin_baseline_240s.csv` 和 `benchmarks/exported_results/dp_roundrobin_baseline_240s.json`；
- 后续论文绘图与表格统计以导出文件为准。

### 2.2 关键实验重复运行

**目的：**  
降低单次运行偶然性，增强实验可信度。

**当前缺口：**

- 现有 JSONL 看起来每组配置只有一次运行结果；
- 缺少均值、标准差或误差条。

**建议补充设置：**

对以下核心配置至少重复 3 次：

| 策略 | alpha | req_rate | duration | 重复次数 |
|---|---:|---:|---:|---:|
| RWPT | 0.1 / 0.3 / 0.8 | 6.0 | 240s | 3 |
| token_count | 0.1 / 0.3 / 0.8 | 6.0 | 240s | 3 |
| queue_length | 0.1 / 0.3 / 0.8 | 6.0 | 240s | 3 |
| round-robin | 0.1 / 0.3 / 0.8 | 6.0 | 240s | 3 |

如果时间有限，优先重复：

1. `RWPT`;
2. `token_count`;
3. `round-robin`;
4. `queue_length`。

**论文中用途：**

- 图中加入误差条；
- 表格中报告 mean ± std；
- 增强“性能提升稳定存在”的说服力。

**当前进展（截至 2026-05-18）：**

- `RWPT` 在 `alpha=0.1 / 0.3 / 0.8` 下的 3 次重复实验已完成；
- `round-robin` 在 `alpha=0.1 / 0.3 / 0.8` 下的 3 次重复实验已完成；
- `token_count` 和 `queue_length` 的重复实验尚未开始。

**当前结论（RWPT vs round-robin）：**

- 从整体趋势看，`RWPT` 相比 `round-robin` 在 `alpha=0.3` 和 `alpha=0.8` 场景下表现更优，尤其在 `P90 latency`、`P90 TTFT` 和缓存命中率相关指标上优势更明显；
- 在 `alpha=0.1` 场景下，`RWPT` 并非所有指标都优于 `round-robin`，两者吞吐量接近，而 `round-robin` 的平均端到端延迟和平均 TTFT 略低；
- 因此更稳妥的论文表述应为：`RWPT` 在整体 QoS 上优于简单均匀分发，且在访问偏斜较弱、负载更复杂的场景下优势更加明显，而不宜表述为“所有场景和所有指标均全面优于 round-robin”。

## 3 强烈建议补充的实验

### 3.1 请求到达率压力变化实验

**目的：**  
验证不同负载压力下，`RWPT-Based Routing` 是否仍然能够改善 QoS。

**当前缺口：**

- 当前主要实验使用 `req_rate=6.0`；
- 缺少低负载、中负载和高负载条件下的趋势分析。

**建议补充设置：**

固定 `alpha=0.3`，改变请求到达率：

| 策略 | alpha | req_rate | duration | num_workers |
|---|---:|---:|---:|---:|
| RWPT | 0.3 | 4.0 | 240s | 3 |
| RWPT | 0.3 | 6.0 | 240s | 3 |
| RWPT | 0.3 | 8.0 | 240s | 3 |
| RWPT | 0.3 | 10.0 | 240s | 3 |
| token_count | 0.3 | 4.0 | 240s | 3 |
| token_count | 0.3 | 6.0 | 240s | 3 |
| token_count | 0.3 | 8.0 | 240s | 3 |
| token_count | 0.3 | 10.0 | 240s | 3 |

如果实验时间紧张，可以只比较 `RWPT` 与 `token_count`。

**论文中用途：**

- 展示系统在负载压力升高时的延迟变化；
- 支撑“高并发条件下 QoS 改善”的论文主张；
- 可画折线图：横轴 `req_rate`，纵轴 TTFT / P90 latency / throughput。

### 3.2 热点副本扩展消融实验

**目的：**  
验证第 4.3 节提出的热点 Adapter 副本扩展机制是否确实有帮助。

**当前缺口：**

- 当前方法章节已经写了热点副本扩展；
- 但现有结果中还没有明确的 `RWPT with replication` vs `RWPT without replication` 对比。

**建议补充设置：**

重点测试热点更明显的场景：

| 策略 | alpha | req_rate | duration | enable_replication |
|---|---:|---:|---:|---|
| RWPT | 0.1 | 6.0 | 240s | false |
| RWPT + replication | 0.1 | 6.0 | 240s | true |
| RWPT | 0.3 | 6.0 | 240s | false |
| RWPT + replication | 0.3 | 6.0 | 240s | true |

如果时间允许，可额外加入高压场景：

| 策略 | alpha | req_rate | duration | enable_replication |
|---|---:|---:|---:|---|
| RWPT | 0.1 | 8.0 | 240s | false |
| RWPT + replication | 0.1 | 8.0 | 240s | true |

**论文中用途：**

- 支撑第 4.3 节热点增强机制；
- 说明在热点集中场景下，副本扩展可缓解局部拥塞；
- 重点观察 P90 TTFT、P90 latency、缓存命中率和 Worker 分布变化。

**设计修正：复制触发阈值与 `routing_w2` 解耦**

当前主动复制机制的拥塞触发阈值由路由评分边界推导：

```text
EMA(RWPT / batch_max_tokens) > routing_w1 / routing_w2
```

这个设计的优点是零新增超参数，能够从“负载惩罚超过缓存命中奖励”的边界自然解释复制触发时机。但它也使 `routing_w2` 同时承担两个职责：

- 在请求级路由中控制负载惩罚强度；
- 在主动复制中控制拥塞触发阈值。

因此，如果后续为了优化路由而调整 `routing_w2`，主动复制触发时机也会被同步改变。例如 `routing_w1=1.0, routing_w2=4.0` 时，复制阈值会降到 `0.25`，可能比高负载 cache affinity 衰减更早触发，导致复制策略过于激进，也会增加论文实验解释难度。

最终论文向实现将复制触发阈值解耦为归一化后的固定语义阈值：

```text
replication_congestion_threshold = 1.0
```

即当 Worker 的 EMA RWPT 压力超过约一个 prefill batch 容量时触发主动复制：

```text
EMA(RWPT / batch_max_tokens) > 1.0
```

这个 `1.0` 不是需要搜索的额外调参，而是 RWPT 归一化后的自然单位：约一个 prefill batch 的积压工作量。这样可以保持三类控制逻辑各自独立：

- `routing_w2`：只影响请求级 RWPT 路由选择；
- `cache_affinity_decay_threshold`：只影响高压下缓存奖励衰减；
- `replication_congestion_threshold`：只影响主动复制触发时机。

**当前真实 Trace 主线设置：`routing_w2 = 3.0`**

在 Azure LLM + Azure Functions HTTP top100 融合 trace 的 6 rps 实验中，已对
`routing_w2 = 1.0 / 1.2 / 1.5 / 2.0 / 3.0` 以及
`2.5 / 3.0 / 3.5` 做过小范围搜索。综合吞吐、平均延迟和尾延迟后，当前论文主线
采用 `routing_w2 = 3.0`：

- 相比 round-robin，`RWPT(w2=3.0)` 保持约 70% 以上 adapter cache hit，并显著降低 P90 latency / P90 TTFT；
- 相比 `w2=1.0`，`w2=3.0` 的平均延迟差异较小，但尾延迟更稳定；
- `w2=3.5` 在部分单轮实验中平均指标更好，但 P90 TTFT 不如 `w2=3.0` 稳定。

因此，除非后续更大规模重复实验推翻该结论，真实 trace 主线对比中的
`adapter-aware + RWPT` 默认使用 `routing_w2 = 3.0`。

## 4 可选补充实验

### 4.1 路由权重敏感性实验

**目的：**  
验证缓存收益权重和负载惩罚权重对结果的影响。

**建议设置：**

固定 `alpha=0.3`、`req_rate=6.0`，改变 `routing_w2`：

| load_metric | routing_w1 | routing_w2 |
|---|---:|---:|
| RWPT | 1.0 | 0.5 |
| RWPT | 1.0 | 1.0 |
| RWPT | 1.0 | 1.5 |
| RWPT | 1.0 | 2.0 |

**论文中用途：**

- 说明方法对参数变化是否敏感；
- 可作为附加实验，不一定放主图。

### 4.2 Rank 分布变化实验

**目的：**  
验证 `RWPT` 对 LoRA Rank 异质性的感知能力。

**建议设置：**

如果实验框架支持，可构造不同 Rank 组合：

| Rank 组合 | 说明 |
|---|---|
| 16 / 64 | 当前主实验设置 |
| 8 / 64 | 更明显的 Rank 差异 |
| 16 / 32 / 64 | 多级 Rank 差异 |

**论文中用途：**

- 直接支撑“Rank 感知”的核心动机；
- 如果当前框架不方便改，可暂不补。

## 5 推荐优先级

如果时间有限，建议按以下顺序补：

1. **补全 round-robin 三组 alpha，且统一 duration=240s；**
2. **核心配置重复 3 次，至少覆盖 RWPT、token_count、round-robin；**
3. **补一个 req_rate 压力变化实验；**
4. **补 RWPT with/without replication 消融；**
5. **视时间决定是否做权重敏感性与 Rank 分布变化实验。**

## 6 第 6 节可对应的实验组织

补充完成后，第 6 节可以组织为：

1. `6.1 实验问题`：定义 RQ1/RQ2/RQ3/RQ4；
2. `6.2 实验设置`：说明模型、Adapter、请求生成、硬件环境；
3. `6.3 对比方法与评价指标`：说明 baselines 和 metrics；
4. `6.4 整体性能对比`：对比 S-LoRA、round-robin 和 RWPT；
5. `6.5 负载指标消融分析`：对比 RWPT、token_count、queue_length；
6. `6.6 不同负载与访问偏斜下的鲁棒性`：分析 alpha 和 req_rate；
7. `6.7 热点副本扩展效果分析`：如果补了 replication 实验，则单独成节。
