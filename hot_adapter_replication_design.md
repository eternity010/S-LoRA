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

Router 目前只知道每个 worker 的 RWPT 总值，不知道队列中每个请求的明细。存在两种可选方案：

- ~~Phase 1（Router 端估算）~~：Router 维护 per-worker 请求分发记录，从中估算每个 adapter 对 RWPT 的贡献。优点是零改动 Worker 端；缺点是 Router 不知道哪些请求已被 Worker 处理完，估算存在偏差。
- **Phase 2（Worker 端上报）✅ 采用**：Worker 上报时额外带上"当前队列中 RWPT 贡献 top-K 的 adapter"信息。数据来自 Worker 的真实队列状态，精确度高。

**决策：直接采用 Phase 2，跳过 Phase 1。** 理由：
1. Phase 1 的估算虽然在热点压倒性场景下"够用"，但本质是用 Router 的发送记录"猜"Worker 的队列状态，信息源不对称。一步到位采用 Worker 端上报，避免后续从 Phase 1 重构到 Phase 2 的二次工程成本。
2. 改动范围可控：`WorkerState` 新增一个 `top_k_rwpt_adapters` 字段（如 `List[Tuple[adapter_id, rwpt_contribution]]`），Worker 在每次心跳时遍历当前等待队列按 adapter 聚合 RWPT 贡献、排序取 top-K（K=3~5 即可），随心跳上报。Router 端解析新字段后直接用于 Step 2 的 culprit 识别。
3. 对于 3 worker 规模的集群，Worker 端的聚合计算开销可忽略（队列长度通常在几十到几百量级，排序 top-K 是 O(n) 操作）。

**抗抖动机制**：

RWPT 是 100ms 上报的瞬时值，可能有波动。为避免瞬时尖峰触发不必要的复制：
- 对 RWPT/Capacity 维护 EMA（指数移动平均，窗口约 500ms，即最近 5 次上报）
- 触发条件使用 `EMA(RWPT_i) > T_congestion` 而非瞬时值

### 3.2 复制到哪里？（目标 worker 选择）

候选 worker 需满足两个硬性条件：
1. 尚未缓存该 adapter（`adapter ∉ Cache_i`）
2. 负载足够低（`EMA(RWPT_i/Capacity) < T_congestion × 0.9`）—— 全局过载熔断条件，确保目标 worker 有足够 headroom 吸收新增流量
3. 当前受保护的副本数未满（`protected_replica_count_i < max_protected_per_worker`，建议 2）—— 防止短时间内多个不同 adapter 的副本堆积在同一个 worker 上，导致保护期内显存无法释放

**为什么不检查显存余量**：在 ratio=0.2 的高压缓存环境下（每 worker 仅 18-19 个 adapter 槽位），系统运行几分钟后所有 worker 的显存使用率都会顶到 90%+ 高水位。如果要求"显存 < 80%"才允许复制，复制机制将永远无法触发。实际上不需要这个条件——复制执行时（无论是空请求触发还是 preload RPC），目标 worker 的 `load_adapters()` 内部会自动调用淘汰机制（`check_and_evict_by_threshold()`）腾出空间。被淘汰的一定是三维评分最低的冷 adapter（使用次数低 + 时间衰减大 + 队列需求为 0），而 `preserve_dirs` 保护机制确保正在使用的 adapter 不会被误杀。用一个冷 adapter 的槽位换一个热点 adapter 的副本，净收益为正。

**选择策略**：在满足条件的候选中，选 RWPT/Capacity 最低的 worker（最空闲的）。这与路由评分函数的逻辑一致——副本应该放在路由器最可能选择的 worker 上。复制完成后，路由器自然会把该 adapter 的部分流量导向新副本（因为它有缓存且负载低，评分最高）。

**无合格候选时的行为**：三个条件可能同时过滤掉所有 worker（典型场景：全局饱和），此时直接放弃复制。这就是弹性降级——不强行复制到不合格的 worker，因为那比不复制更糟（把压力从一个过载节点搬到另一个过载节点是零和博弈）。

**副本数量策略**：

每个决策周期只增加 1 个副本（渐进式）。理由：
- 复制后流量自然分散到新副本 worker，原 worker 的 RWPT 下降
- 如果 1 个副本不够，下一个巡检周期 RWPT 仍然超阈值，会继续复制（受 N_max + T_cooldown 约束）
- 避免一次性过度复制浪费显存

### 3.3 什么时候回收？（副本淘汰）

流量模式会变化，副本需要能被回收。

**回收方式：复用现有淘汰机制（零新增代码路径）**

不主动回收。而是：
1. 副本加载后，给予一个初始保护期（如 30s），期间加入淘汰保护列表
2. 保护期结束后，移出保护列表，让本地淘汰机制自然处理
3. 如果该 adapter 仍然热门，它会积累使用次数和访问时间，评分自然升高，不会被淘汰
4. 如果该 adapter 变冷，使用次数停滞、时间衰减生效，评分下降，被正常淘汰

这样完全复用了现有的三维评分淘汰机制，不需要新的回收逻辑。

**现有淘汰机制概览（`InferAdapter`，副本回收的基础设施）**：

三维评分模型——每个 adapter 的"重要性"分数，越高越不应被淘汰：

```
Score = 0.35 × S_usage + 0.35 × S_recency + 0.30 × S_pending

S_usage   = log(1+count_window) / log(1+max_count_window)  # 滑动窗口内使用频率，对数归一化
S_recency = e^(-Δt / 300)                                   # 时间衰减，5min 未访问降至 0.37
S_pending = log(1+pending) / log(1+max_pending)             # 队列需求，有等待请求的不应淘汰
```

**S_usage 滑动窗口改造（✅ 已实现）**：

原始实现使用累计计数器 `score_update_counter`，`max_count` 为全局历史最大值。长时间运行后，爆款 adapter 的 count 可能达到 10⁶，导致 `max_count` 无限膨胀，温 adapter（count=100）的 S_usage 被压缩至 `log(101)/log(10⁶) ≈ 0.33`，而 S_recency 是绝对归一化（刚被访问 1 次即为 1.0），三维评分实质退化为纯 LRU。

修复方案：将 S_usage 从累计计数器改为滑动窗口时间戳（窗口 = 300s，与 S_recency 的 τ 对齐）。每次批次更新时追加时间戳并剪枝过期条目，`max_count_window` 也仅统计窗口内的全局最大值。这样 S_usage 和 S_recency 在相同时间尺度上衡量"近期活跃度"，权重比例不再随运行时间失衡。

实现细节（`infer_adapter.py`）：
- 新增 `usage_timestamps: Dict[str, list]`（adapter → 窗口内访问时间戳列表）和 `usage_window: float = 300.0`
- `update_adapter_stats_batch()`：追加时间戳 + 剪枝过期 + 同步 `score_update_counter`（兼容字段）
- `calculate_adapter_score()`：S_usage 计数和 max_usage 均基于窗口内统计
- `load_adapters()`：新 adapter 初始化 `usage_timestamps = []`
- `offload_adapters()`：卸载时清理 `usage_timestamps`
- `score_update_counter` 保留为兼容字段，始终等于窗口内计数

三种触发模式：

| 触发时机 | 阈值 | 淘汰比例 | 场景 |
|---------|------|---------|------|
| 加载前（`load_adapters` 内部） | 90% | 固定 20% | 新 adapter 要进来，先腾空间 |
| 请求完成时 | 90% | 动态 20%-60% | 请求处理完毕，趁间隙清理 |
| 批次清空时 | 80% | 固定 70% | 整个批次跑完，大扫除 |

保护机制：
- `preserve_dirs`：当前批次正在使用的 adapter 硬保护，绝对不淘汰
- `prefetch_tag`：正在预取的 adapter 不淘汰
- 队列需求分数：间接保护即将使用的 adapter（有 pending 请求 → 评分高 → 不易被选中）

**副本保护期如何接入**：新副本加入 `preserve_dirs` 保护列表后，在 `select_eviction_candidates()` 中会被直接过滤掉，和当前批次 adapter 享受完全相同的硬保护。30s 后移出保护列表，此时如果 adapter 确实热门，积累的使用次数（S_usage 升高）和近期访问时间（S_recency 接近 1.0）会让评分自然维持高位；如果 adapter 已变冷，三维评分自然下降，被正常淘汰——这正是期望的行为。

**为什么新副本需要保护期**：刚复制过去的 adapter 存在"冷启动劣势"——使用次数=0（S_usage=0）、队列需求=0（S_pending=0），总评分极低。而目标 worker 上那些温冷 adapter 虽然不热门，但好歹有几次使用记录，评分更高。没有保护期的话，新副本可能在下一次淘汰触发时就被干掉。

**保护期的安全边界**：在 max_lora_ratio=0.2 的高压环境下，如果短时间内多个不同 adapter 的副本都落户到同一个 worker 并获得 30s 保护，可能导致淘汰器找不到可释放的候选。这通过目标选择条件 3（`protected_replica_count < max_protected_per_worker`）防止——每个 worker 同时受保护的副本数不超过 2 个，超出后新的复制请求会转向其他 worker。此外，每个副本落户后会增加该 worker 的流量和 RWPT，负载条件（条件 2）也会自然阻止后续副本继续堆积。

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

> **决策：直接采用一步到位方案，跳过"空请求"验证阶段。** 理由：空请求方案虽然改动小，但会引入推理资源占用和额外的工程债务（后续仍需重构为预加载 RPC）。预加载 RPC 的改动范围可控，且与 Worker 端上报元凶信息（Section 3.1 Phase 2）的改动可以一并完成，一次性打通整条链路。

### Phase 1：完整复制机制（一步到位）

目标：实现完整的"RWPT 驱动的主动复制"链路，包括 Worker 端元凶上报 + 预加载 RPC + 副本保护。

改动清单：
1. **Worker 端元凶上报**：`WorkerState` 新增 `top_k_rwpt_adapters: List[Tuple[str, float]]` 字段，Worker 每次心跳时遍历当前等待队列，按 adapter 聚合 RWPT 贡献（`Σ input_len × (1+γ×rank)`），排序取 top-K（K=3~5）随心跳上报
2. **预加载 RPC 接口**：新增 `/preload_adapter` 端点，Worker 收到后只加载 adapter 权重到显存，不执行推理。通过 `DataParallelRouterManager` 下发指令
3. **ReplicaManager 模块**：独立模块，维护 `{adapter: [worker_ids]}` 副本映射、`last_replication_time` 冷却状态、保护期计时器
4. **巡检线程**：Router 周期性（500ms-2s）扫描 Worker 上报的 `EMA(RWPT/Capacity)`，若超过 `T_congestion`，直接从上报的 `top_k_rwpt_adapters` 中取 culprit（无需 Router 端估算），执行 Step 2→3 的复制决策
5. **副本保护期**：在 `select_eviction_candidates` 中加入保护列表（保护期 30s），防止新副本被淘汰

~~备选方案（已放弃）：用"空请求"触发被动加载，不需要新 RPC 接口，但会占用推理资源且需要后续重构。~~

### Phase 2：优化与精细化

1. 动态副本数调整：根据请求率和 worker 数量计算期望副本数
2. 副本生命周期管理：冷却后自动降级为可淘汰
3. 主动回收（RWPT 信号 + 滞后系数防振荡）

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
