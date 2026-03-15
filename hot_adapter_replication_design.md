# 热门 Adapter 多副本机制 — 预思考

> **📅 创建**: 2026-03-14
> **📌 状态**: 设计完成，待代码实现
> **📋 前置**: 智能路由（RWPT + w2 甜点搜索 + alpha 鲁棒性实验）✅ 已完成

---

## 1. 问题定义

当前系统中，每个 adapter 在某一时刻通常只缓存在 1 个 worker 上。路由器通过 w1（cache affinity）倾向于将请求发往有缓存的 worker。当某个 adapter 成为热点时（Zipf 分布下 α 越小越明显），所有该 adapter 的请求集中到同一个 worker，导致：

- 该 worker 队列堆积，P90 TTFT 恶化
- 其他 worker 相对空闲，整体 GPU 利用率不均
- RWPT 的 w2 惩罚虽然能将部分请求分散到无缓存的 worker，但代价是冷启动加载延迟

核心矛盾：**缓存亲和性（w1）和负载均衡（w2）的 trade-off 是零和博弈，多副本是打破这个零和博弈的唯一手段**。

---

## 2. 目标

让热门 adapter 同时缓存在多个 worker 上，使路由器可以在多个有缓存的 worker 之间分流，同时享受缓存命中和负载均衡。

量化目标：
- 热点 adapter 的 P90 TTFT 降低 20%+（对比无副本的 RWPT）
- 整体吞吐量不退化（副本占用的显存不应导致冷 adapter 大量淘汰）
- 副本机制在流量模式变化时能自动收敛（热→冷时副本被回收）

---

## 3. 三个核心决策问题

### 3.1 什么时候复制？（触发条件）

**现有基础设施**：
- `WorkerState.pending_prefill_tokens`：每个 worker 的 RWPT 值（100ms 上报）
- 路由评分函数：`Score_i = w1·I(hit) - w2·RWPT_i/Capacity`
- 倒排索引 `adapter_to_workers`：adapter → 缓存该 adapter 的 worker 集合
- Router 维护的请求分发记录：知道每个请求的 adapter、input_len、目标 worker

#### 算力告警驱动的复制决策（Worker 视角 → 揪出元凶 → 全局护栏）

传统缓存复制基于访问频率或未命中率，在异构大模型推理中是滞后的——10 个短请求和 10 个长请求的频率相同，但对 worker 的压力完全不同。我们利用 RWPT 进行主动防御式复制。

**决策流程**：

```
Step 1: 算力拥塞检测（后台巡检线程，周期 500ms-2s）
  ┌─────────────────────────────────────────────────────────┐
  │ 遍历所有 Worker，检查 EMA(RWPT_i / Capacity)            │
  │ 若 EMA > T_congestion → 该 Worker 即将发生队头阻塞      │
  └──────────────────────────┬──────────────────────────────┘
                             │ 触发
                             ▼
Step 2: 揪出元凶（Identify Culprit）
  ┌─────────────────────────────────────────────────────────┐
  │ 遍历过载 Worker 的等待队列（通过 Router 分发记录估算），  │
  │ 找出对 RWPT 贡献最大的 adapter：                         │
  │   culprit = argmax_A Σ(input_len_j · (1+γ·r_j))        │
  │             for j in requests_of(A) on worker_i          │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
Step 3: 全局防爆护栏（副本上限 + 复制冷却期）
  ┌─────────────────────────────────────────────────────────┐
  │ 检查 culprit adapter 的全局副本数：                      │
  │   current_replicas = |{w | adapter ∈ Cache_w}|           │
  │ 检查 culprit adapter 的复制冷却状态：                    │
  │   time_since_last = now - last_replication_time(adapter)  │
  │                                                           │
  │ 若 current_replicas ≥ N_max → 放弃（保护长尾生存空间）   │
  │ 若 time_since_last < T_cooldown → 放弃（等待上次复制生效）│
  │ 否则 → 执行复制，记录 last_replication_time(adapter)=now  │
  └─────────────────────────────────────────────────────────┘
```

**T_congestion 的推导（零新增超参数）**：

T_congestion 不是独立超参数，而是从路由评分函数的决策边界推导：

```
T_congestion = (w1 / w2) × Capacity
```

物理含义：当 worker 的 RWPT 超过此值时，路由器对该 worker 上所有有缓存 adapter 的评分中，负载惩罚 `w2·RWPT/Capacity` 已经 ≥ 缓存收益 `w1`，路由器被迫放弃缓存命中。这就是复制的最佳时机。

以当前甜点值 w1=1.0, w2=1.0 为例：`T_congestion = 1.0 × Capacity`，即 RWPT 超过一个批次容量时触发。

**N_max 副本上限**：

单个 adapter 在全集群的最大副本数。建议 `N_max = num_workers - 1`（3 worker 集群中最多 2 个副本），强制为长尾 adapter 预留至少 1 个 worker 的显存空间，防止爆款 adapter "蝗虫过境"导致系统死锁。

**T_cooldown 复制冷却期**：

同一个 adapter 两次复制之间的最小间隔。建议 `T_cooldown = 5~10s`。

物理含义：一次复制从"发出指令"到"新副本加载完成并开始分流"的预期时间。在此窗口内，原 worker 的 EMA(RWPT) 几乎不会下降（新副本还没生效），如果不加冷却期，巡检线程会在每个周期（500ms-2s）重复触发 Step 1→2→3，对同一个 adapter 连续发出复制指令，几秒内把 N_max 打满。

注意：EMA 平滑无法替代冷却期——EMA 只能平滑瞬时尖峰，但如果 RWPT 持续高位（因为新副本还没生效），EMA 值也会持续高于 T_congestion。冷却期是唯一能阻止连续扩容的机制。

实现：`ReplicaManager` 维护 `last_replication_time: Dict[str, float]`（adapter → 上次复制时间戳），Step 3 中检查 `time.time() - last_replication_time.get(adapter, 0) < T_cooldown`。

**"揪出元凶"的实现路径**：

Router 目前只知道每个 worker 的 RWPT 总值，不知道队列中每个请求的明细。两种实现方式：

- Phase 1（Router 端估算）：Router 维护一个 per-worker 的请求分发记录（Router 知道它把哪些请求发给了哪个 worker，也知道每个请求的 adapter 和 input_len）。从分发记录中估算每个 adapter 对 RWPT 的贡献。估算值和实际值可能有偏差（Router 不知道哪些请求已被 Worker 处理完），但足够用于识别贡献最大的 adapter。
- Phase 2（Worker 端上报）：Worker 上报时额外带上"当前队列中 RWPT 贡献 top-K 的 adapter"信息。更准确，但需要改 WorkerState 和上报协议。

**抗抖动机制**：

RWPT 是 100ms 上报的瞬时值，可能有波动。为避免瞬时尖峰触发不必要的复制：
- 对 RWPT/Capacity 维护 EMA（指数移动平均，窗口约 500ms，即最近 5 次上报）
- 触发条件使用 `EMA(RWPT_i) > T_congestion` 而非瞬时值

### 3.2 复制到哪里？（目标 worker 选择）

候选 worker 需满足：
1. 尚未缓存该 adapter（`adapter ∉ Cache_i`）
2. 有足够显存余量（LoRA 使用率 < max_lora_ratio 的 80%）
3. 当前负载较轻

**选择策略**：在满足条件的候选中，选 RWPT/Capacity 最低的 worker（最空闲的）。这与路由评分函数的逻辑一致——副本应该放在路由器最可能选择的 worker 上。

**副本数量策略**：

每个决策周期只增加 1 个副本（渐进式）。理由：
- 复制后流量自然分散到新副本 worker，原 worker 的 RWPT 下降
- 如果 1 个副本不够，下一个巡检周期 RWPT 仍然超阈值，会继续复制（受 N_max 约束）
- 避免一次性过度复制浪费显存

### 3.3 什么时候回收？（副本淘汰）

流量模式会变化，副本需要能被回收。

**回收方式：复用现有淘汰机制（零新增代码路径）**

不主动回收。而是：
1. 副本加载后，给予一个初始保护期（如 30s），期间加入淘汰保护列表
2. 保护期结束后，移出保护列表，让本地淘汰机制自然处理
3. 如果该 adapter 仍然热门，它会积累使用次数和访问时间，评分自然升高，不会被淘汰
4. 如果该 adapter 变冷，使用次数停滞、时间衰减生效，评分下降，被正常淘汰

这样完全复用了现有的三维评分淘汰机制（使用频率 + 时间衰减 + 队列需求），不需要新的回收逻辑。

**可选的主动回收优化（Phase 2）**：

用 RWPT 信号 + 滞后系数防振荡：
```
对于被复制到 worker B 的 adapter A：
  如果 EMA(RWPT_B/Capacity) < T_congestion/Capacity × 0.5（滞后系数）
  且 adapter A 在 worker B 上的使用率低于平均水平
  则将 adapter A 从 worker B 的保护列表中移除
```

滞后系数（0.5）确保触发阈值 > 回收阈值，避免"复制→流量分散→RWPT 下降→回收→流量集中→RWPT 上升→再复制"的振荡

---

## 4. 现有代码改动点分析

### 4.1 需要新增的组件

| 组件 | 位置 | 职责 |
|------|------|------|
| `ReplicaManager` | `adapter_aware_router.py` 内部或独立模块 | 维护副本状态、触发复制/回收决策 |
| 预加载 RPC | `dp_manager.py` → `gpu_worker.py` | 向 worker 发送"主动加载 adapter"指令 |

### 4.2 需要修改的组件

| 组件 | 改动 | 说明 |
|------|------|------|
| `AdapterAwareRouter` | 集成 ReplicaManager，路由时考虑副本状态 | 核心改动 |
| `DataParallelRouterManager` | 新增预加载指令通道 | 目前 adapter 是被动加载的（请求来了才加载），需要支持主动加载 |
| `InferAdapter.select_eviction_candidates` | 尊重"副本保护列表" | 防止刚复制过去的热门 adapter 被本地淘汰策略干掉 |
| `WorkerStateReporter` | 可能需要上报更精确的显存余量 | `pool_used_ratio` 已有，可能够用 |
| `GPUWorker` | 新增处理预加载指令的逻辑 | 收到指令后主动加载 adapter 到显存 |

### 4.3 不需要改动的组件

- 评分函数 `calculate_score()`：不需要改，副本加载后 `cached_adapters` 自然包含该 adapter，w1 自动生效
- `WorkerState`：`cached_adapters` 已经是 Set，副本加载后自动出现在集合中
- 倒排索引 `adapter_to_workers`：状态上报时自动更新

---

## 5. 实现路径建议

### Phase 1：最小可行版本（验证机制）

目标：验证"RWPT 驱动的主动复制"能否降低 P90 TTFT。

改动范围最小化：
1. Router 周期性（如每 2s）扫描倒排索引 `adapter_to_workers`，对每个 adapter 检查其所有缓存 worker 的 `EMA(RWPT/Capacity)`
2. 若 `min(EMA) > w1/w2`（T_replicate），选 RWPT 最低的无缓存 worker，发送一个"空请求"（使用该 adapter 但 input 极短），触发被动加载
3. 在 `select_eviction_candidates` 中加入保护列表（保护期 30s），防止新副本被淘汰

这个方案不需要新的 RPC 接口，利用现有请求通道触发加载。缺点是"空请求"会占用一点推理资源，但作为 Phase 1 验证足够了。

### Phase 2：正式实现

1. 新增预加载 RPC 接口（`/preload_adapter`），worker 收到后只加载 adapter 权重，不执行推理
2. `ReplicaManager` 独立模块，维护 `{adapter: [worker_ids]}` 的副本映射
3. 动态副本数调整：根据请求率和 worker 数量计算期望副本数
4. 副本生命周期管理：冷却后自动降级为可淘汰

### Phase 3：优化（如果需要）

- 预测式复制：基于请求率趋势提前复制，而不是等到过载才触发
- 全局 adapter 放置优化：不仅考虑热门 adapter，还考虑 adapter 之间的共现关系
- 副本数与 w2 的联动：副本数增加后，w2 的最优值可能需要重新搜索

---

## 6. 实验设计

### 6.1 验证场景

最能体现多副本收益的场景：**小 α（0.1-0.3）+ 高请求率**。此时少数 adapter 占大量流量，单 worker 瓶颈最明显。

### 6.2 对比实验

| 配置 | 说明 |
|------|------|
| RWPT（无副本） | 当前最优基线，alpha-robustness-sla 实验的 RWPT 结果 |
| RWPT + 被动分散 | 现有的热点 Round-Robin（已实现） |
| RWPT + 主动复制 | 本次新增的多副本机制 |

### 6.3 核心指标

- P90 TTFT（最核心，直接反映用户体验）
- 热点 adapter 的 per-adapter P90 TTFT（细粒度验证）
- 整体吞吐量（确认不退化）
- 缓存命中率（预期提升，因为热门 adapter 在多个 worker 上都有缓存）
- 副本数量随时间的变化曲线（验证自适应性）
- 冷 adapter 的淘汰频率（确认副本没有过度挤占显存）

### 6.4 实验矩阵

| 变量 | 值 |
|------|-----|
| α | 0.1, 0.3, 0.8 |
| 副本策略 | 无副本 / 被动分散 / 主动复制 |
| req_rate | 6.0（与现有实验一致） |
| max_lora_ratio | 0.2（与现有实验一致） |

共 3 × 3 = 9 组实验。

---

## 7. 风险与注意事项

1. **显存压力**：每个副本占用一份 adapter 显存。在 max_lora_ratio=0.2（每 worker 约 18-19 个 adapter 槽位）的约束下，热门 adapter 占用多个槽位可能导致冷 adapter 被频繁淘汰，反而降低整体缓存命中率。需要监控冷 adapter 的淘汰频率。

2. **复制-回收振荡**：复制后流量分散 → RWPT 下降 → 低于回收阈值 → 回收 → 流量集中 → RWPT 上升 → 再复制。通过滞后系数（触发阈值 = w1/w2，回收阈值 = w1/w2 × 0.5）、保护期（30s）和复制冷却期（T_cooldown = 5~10s）三重机制防止振荡。

3. **连续扩容风暴**：新副本从发出指令到生效需要数秒，期间原 worker 的 RWPT 不会下降，巡检线程可能连续触发复制。通过 T_cooldown 复制冷却期解决——同一 adapter 两次复制之间必须间隔 T_cooldown 秒。

3. **与淘汰机制的交互**：主动复制的 adapter 刚到目标 worker 时，使用次数=0、时间衰减=1.0（刚加载），评分模型可能给出中等分数。保护期（30s）确保新副本有时间积累使用记录。保护期结束后，如果 adapter 确实热门，自然积累的使用次数会维持高分。

4. **RWPT 信号延迟**：RWPT 是 100ms 上报的，从检测到过载到完成复制（加载 adapter 权重）可能需要数百毫秒到数秒。这段时间内热点 worker 仍然过载。EMA 平滑会进一步增加延迟。这是可接受的——副本机制是中期优化（秒级），不是实时调度（毫秒级），实时调度由路由评分函数负责。

5. **T_replicate 与 w2 的联动**：多副本上线后，w2 的最优值可能变小（副本减轻了负载均衡压力）。T_replicate = w1/w2 会随之变大，意味着复制触发更保守。这是一个自洽的反馈：副本越多 → 需要的 w2 越小 → 触发阈值越高 → 不会过度复制。但如果重新搜索 w2，T_replicate 也会自动调整，无需手动干预。

6. **实验公平性**：对比实验中，"无副本"和"有副本"的 max_lora_ratio 应该相同，确保总显存预算一致。副本占用的显存是从同一个预算中扣除的。

---

## 8. 与现有工作的关系

- **S-LoRA**：原始系统无数据并行，无副本概念
- **Toppings (ATC 2025)**：提出 Rank-Aware 调度，但未涉及多副本
- **vLLM**：多实例部署本质上是全量复制（每个实例加载所有 adapter），不是选择性复制
- **Chameleon**：提出 adapter 放置优化，但是静态的（离线规划），不是动态的

本工作的差异化：**动态的、请求率驱动的、与路由评分函数协同的选择性副本机制**。
