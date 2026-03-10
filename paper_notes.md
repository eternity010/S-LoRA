# 多租户 LoRA 推理系统论文思路整理

> **📅 最后更新**: 2026-03-10
> **📌 状态**: 基于已实现功能整理

---

## 📝 论文定位

基于 S-LoRA 系统，提出面向高并发多租户场景的**多 Worker 数据并行**推理架构，包含四个核心贡献：
1. **多 Worker 数据并行架构**：每个 GPU 独立运行完整模型实例，解决张量并行在高并发场景的可扩展性瓶颈
2. **Adapter 感知路由**：基于缓存亲和性和负载感知的智能请求分发
3. **Rank 感知路由**：考虑 LoRA Rank 相似性，优化异构批处理效率
4. **热门 Adapter 多副本**：检测热点并复制到多个 Worker，线性提升热点吞吐
5. **多维评分淘汰机制**：结合使用频率、时间衰减和队列需求的精细化显存管理

**术语说明**：
- 本文使用 "数据并行" 指代 "请求级数据并行"（Request-Level Data Parallelism）
- 与训练场景的数据并行不同，推理场景无需梯度同步
- 每个 Worker 独立处理不同请求，实现真正的线性可扩展

---

## ✅ 已实现功能清单

### 1. 多 Worker 数据并行架构 (✅ 已完成)
- `dp_manager.py`: 数据并行路由管理器
- `gpu_worker.py`: 独立 GPU Worker 进程（每个 Worker 运行完整模型）
- ZMQ 进程间通信
- 连续批处理支持

### 2. Adapter 感知路由 (✅ 已完成)
- `adapter_aware_router.py`: 智能路由器
  - 评分函数: `Score = w1·I(adapter∈Cache) - w2·QueueLen`
  - 缓存亲和性 + 负载感知
  - 热点 Adapter 检测与分散（被动分散已实现）
  - 冷启动处理
- `worker_state.py`: Worker 状态数据结构
- `worker_state_reporter.py`: Worker 状态上报器 (100ms 心跳)
- `worker_state_cache.py`: Router 端状态缓存
- 倒排索引: Adapter → Worker 映射
- ✅ **Rank 感知路由**：已支持 LoRA Rank 感知（详见 2.2）

### 2.1 热门 Adapter 多副本 (🔄 部分实现)
- ✅ 热点检测（请求率追踪）
- ✅ 被动分散（热点请求 Round-Robin 分配）
- ⏳ 主动复制（预加载到多个 Worker）- 待实现

### 2.2 Rank 感知路由 (✅ 已完成)
- ✅ Worker 上报当前批次的 Rank 分布（avg/min/max rank 追踪）
- ✅ 路由时考虑 Rank 相似性，避免大小 Rank 混合批处理
- ✅ 评分公式扩展：`Score = w1·Cache - w2·QueueLen - w3·RankMismatch`
- ✅ 命令行参数支持（`--routing-w3`, `--default-lora-rank`）
- 参考：Toppings (ATC 2025) 的 Rank-Aware 调度

### 3. Adapter 评分与淘汰机制 (✅ 已完成)
- `infer_adapter.py`: 三维评分模型
  - 使用频率 (对数归一化)
  - 时间衰减 (τ=300s)
  - 队列需求 (pending requests)
- 双模式淘汰:
  - 被动淘汰 (加载时触发)
  - 主动淘汰 (请求完成时触发，动态比例)
- `preserve_dirs` 保护机制

---

## 🏗️ 论文结构

### Title
"Scalable Multi-Tenant LoRA Serving with Multi-Worker Data Parallelism and Adapter-Aware Scheduling"

或更简洁：
"Efficient Multi-Tenant LoRA Serving: A Multi-Worker Approach with Adapter-Aware Routing"

### Abstract

多租户 LoRA 推理面临三大挑战：(1) 张量并行在高并发场景下 GPU 利用率低、通信开销大；(2) 默认轮询路由无视数据局部性，导致 Adapter 重复加载；(3) 热门 Adapter 单副本服务成为吞吐瓶颈；(4) 简单 LRU 淘汰无法感知队列需求，造成"淘汰-重加载"循环。

本文基于 S-LoRA 系统，提出面向高并发多租户场景的多 Worker 数据并行推理架构。主要贡献包括：(1) 多 Worker 数据并行架构，每个 GPU 独立运行完整模型实例，实现线性可扩展性；(2) Adapter 感知路由基于缓存亲和性和负载感知优化请求分发；(3) Rank 感知路由考虑 LoRA Rank 相似性，优化异构批处理效率；(4) 热门 Adapter 多副本机制实现热点吞吐量线性扩展；(5) 多维评分驱动的双模式淘汰机制实现精细化显存管理。

实验表明，相比原始 S-LoRA 张量并行模式，本系统在 3 GPU 配置下实现 X 倍吞吐量提升，缓存命中率从 30% 提升至 70%+，热点 Adapter 吞吐提升 N 倍，P99 延迟降低 X%。

---

### Section 1: Introduction

**1.1 Background**
- LoRA 在多租户场景的重要性（个性化模型、成本效益）
- 现有系统的局限性（S-LoRA 只支持张量并行）

**1.2 Challenges**
- Challenge 1: 张量并行可扩展性瓶颈（通信开销、显存浪费、次线性扩展）
- Challenge 2: 轮询路由无视缓存局部性（Adapter 重复加载、冷启动频繁）
- Challenge 3: 热门 Adapter 单副本瓶颈（长尾分布下热点成为系统瓶颈）
- Challenge 4: 简单淘汰策略无法感知队列需求（淘汰后立即重加载）

**1.3 Our Approach**
- 多 Worker 数据并行架构：每 GPU 独立运行完整模型实例，无需同步，线性扩展
- Adapter 感知路由：缓存亲和性 + 负载感知评分函数
- 热门 Adapter 多副本：检测热点并复制到多个 Worker，线性提升热点吞吐
- 多维评分淘汰：使用频率 + 时间衰减 + 队列需求

**1.4 Contributions**
1. 设计并实现多 Worker 数据并行推理架构，每个 Worker 独立运行完整模型，实现线性可扩展性
2. 提出 Adapter 感知路由算法，显著提升缓存命中率
3. 提出 Rank 感知路由算法，优化异构 Rank 批处理效率
4. 设计热门 Adapter 多副本机制，解决热点瓶颈问题
5. 设计多维评分驱动的双模式淘汰机制，优化显存管理
6. 全面的实验评估，验证系统在不同负载模式下的性能

---

### Section 2: Background and Motivation

**2.1 LoRA and Multi-Tenant Serving**
- LoRA 原理（低秩分解、参数高效微调）
- 多租户推理需求（数百 Adapter、高并发、低延迟）

**2.2 Existing Approaches**
- S-LoRA: 张量并行 + Unified Paging
- vLLM: 多实例部署
- 其他相关工作（Chameleon, Toppings, LoRAServe）

**2.3 Tensor Parallelism Limitations in Multi-Tenant Scenarios**

| 挑战 | 问题描述 | 影响 |
|------|---------|------|
| GPU 利用率 | 低并发时所有 GPU 协同处理少量请求 | 算力浪费 |
| 通信开销 | 每次推理需要 AllReduce | 25%+ 时间开销 |
| 显存利用率 | Adapter 空间受限于最小 GPU | 总容量 = 单卡容量 |
| 扩展性 | 次线性扩展 | 加卡收益递减 |

**2.4 Motivation for Data Parallelism**
- 高并发场景：并发数 >> GPU 数量
- 线性可扩展性：N 卡 ≈ N 倍吞吐
- 更大 Adapter 空间：总容量 = N × 单卡容量
- 热点可扩展：热门 Adapter 可复制到多 Worker 并行服务
- 与张量并行互补（非替代）

---

### Section 3: System Architecture

**3.1 Overview: Multi-Worker Data Parallelism**

与张量并行（模型切分到多 GPU 协同计算）不同，我们采用多 Worker 数据并行架构：
- 每个 GPU Worker 独立运行完整的模型实例
- Worker 之间无需同步，独立处理不同请求
- 通过智能路由实现负载均衡和缓存优化

```
┌─────────────────────────────────────────────────────────┐
│                    API Server                            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              DP Manager (Router)                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Adapter-Aware Router                            │    │
│  │  - 缓存亲和性评分                                │    │
│  │  - 负载感知                                      │    │
│  │  - 热点 Adapter 分散                             │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Worker State Cache                              │    │
│  │  - Adapter → Worker 倒排索引                     │    │
│  │  - 实时状态缓存                                  │    │
│  └─────────────────────────────────────────────────┘    │
└───┬─────────────┬─────────────┬─────────────────────────┘
    │             │             │
    ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│Worker 0 │  │Worker 1 │  │Worker 2 │
│ GPU 0   │  │ GPU 1   │  │ GPU 2   │
├─────────┤  ├─────────┤  ├─────────┤
│完整模型 │  │完整模型 │  │完整模型 │
│Adapter  │  │Adapter  │  │Adapter  │
│Cache    │  │Cache    │  │Cache    │
├─────────┤  ├─────────┤  ├─────────┤
│State    │  │State    │  │State    │
│Reporter │  │Reporter │  │Reporter │
└─────────┘  └─────────┘  └─────────┘
```

**关键特点：**
- 每个 Worker 是独立的推理引擎，拥有完整模型和独立的 Adapter 缓存
- Worker 之间通过 Router 协调，无直接通信
- 线性可扩展：N 个 Worker ≈ N 倍吞吐量

**3.2 DP Manager (Router)**
- 启动和管理多个 GPU Worker 进程
- 接收请求并根据路由策略分发
- 收集 Worker 状态信息
- ZMQ 进程间通信

**3.3 GPU Worker**
- 独立加载完整模型（每个 Worker 是独立的推理引擎）
- 本地 Adapter 缓存管理
- 连续批处理
- 状态上报（100ms 心跳）

---

### Section 4: Adapter-Aware Routing

**4.1 Motivation: Cache Inefficiency in Round-Robin**

```
Round-Robin 问题示例：
请求序列: [A, A, A] → 3 个 Worker
结果: Adapter A 在 3 个 Worker 上重复加载
理想: 全部路由到 Worker 0，复用缓存
```

**4.2 Scoring Function 演进**

三个版本对应消融实验中的三个 variant，均已实现（`load_metric` 参数控制）：

**V1: queue_length（✅ 已实现，消融基线）**

$Score_i(r) = w_1 \cdot \mathbf{I}(hit) - w_2 \cdot QueueLen_i$

负载项为队列请求数，无归一化，不感知 token 长度差异。

**V2: token_count（✅ 已实现，消融中间档）**

$Score_i(r) = w_1 \cdot \mathbf{I}(hit) - w_2 \cdot \frac{\sum_j input\_len_j}{Capacity_i}$

升级到 token 级，感知请求的 prompt 长度差异，但不感知 LoRA Rank 差异。与 rwpt 共享相同的 Capacity 归一化方式，无 fallback 机制。

**V3: rwpt（✅ 已实现，完整版，默认）**

核心思想：从"请求数"升级为"Rank 校准的 Token 级负载估算"。在多租户 LoRA 场景下，不同 Adapter 的 Rank 差异（如 8 vs 128）导致同样数量的 Token 在不同 Worker 上的实际计算代价截然不同。传统按请求数或 Token 数均衡的策略会产生"隐藏的掉队者"。

最终评分公式：

$Score_i(r) = w_1 \cdot \mathbf{I}(hit) - w_2 \cdot \frac{RWPT_i}{Capacity_i} - w_3 \cdot RankMismatch_i$

其中 RWPT（Rank-Weighted Pending Tokens）定义为：

$RWPT_i = \sum_{j \in WaitQueue_i} input\_len_j \cdot (1 + \gamma \cdot r_j)$

RWPT 仅基于等待队列中的 prefill token，不包含 decode 序列折算。早期版本曾包含 decode 折算项 `α · DecodeSeqs`（α ≈ 90），但实验发现该项导致 RWPT 出现巨大尖峰，引发路由抖动（routing thrashing）和 Worker 负载不均衡，已移除。

token_count 和 rwpt 均无 fallback 机制：当 pending tokens 为 0 时直接使用 0 值，不回退到 queue_length。早期 token_count 存在 fallback（当 `pending_raw_tokens == 0` 且 `queue_length > 0` 时回退到 `w2 * queue_length`），导致评分在两种不同量纲间跳变，路由决策不稳定，已移除（commit 3.24）。三个 variant 现在各自逻辑独立、量纲一致，确保消融实验的公平性。

| 符号 | 含义 |
|------|------|
| $input\_len_j$ | 等待队列中第 j 个请求的 prompt 长度 |
| $r_j$ | 第 j 个请求对应 Adapter 的 LoRA Rank |
| $\gamma$ | Rank 开销系数（从模型结构推导，非超参数） |
| $Capacity_i$ | 单次 prefill 批次最大 token 数（由 `max_total_token_num / 6` 自动推算），RWPT/Capacity ≈ 排队批次数 |

**γ 的理论推导（非超参数）**：

尽管早期文献仅对 Q 和 V 投影注入 LoRA，但现代 LLM 部署的默认配置通常对 Attention 的全部四个投影（Q, K, V, O）应用 LoRA 以保证模型能力。每个 LoRA 投影额外 FLOPs = 4dr（A 矩阵 2dr + B 矩阵 2dr），四个投影共 16dr。Base Model 每层每 token 总 FLOPs ≈ 24d²（Attention 8d² + MLP 16d²）。

计算量增比 = 16dr / 24d² = 2r/(3d)，即 γ·r，提取 r 后得：

$$\gamma = \frac{2}{3d}$$

对 LLaMA-7B（d=4096）：γ ≈ 1.63×10⁻⁴。`hidden_dim` 从模型 `config.json` 的 `hidden_size` 字段自动检测，无需手动指定：
```python
gamma = 2.0 / (3.0 * hidden_dim)  # 理论推导，从模型配置自动获取
```

**α 的运行时实测（保留用于未来扩展）**：

α = decode_per_step / prefill_per_token，表示生成一个 decode token 相对于处理一个 prefill token 的实际耗时比。当前版本的 RWPT 公式已移除 decode 折算项，但 α 的实测机制保留在 Worker 中，供未来副本放置决策等场景使用。

实测方案：Worker 启动时执行一次轻量 micro-benchmark，调用引擎真实推理接口（含 PagedAttention + KV Block 分配），测量 prefill 512 tokens 和 decode 1 step 的耗时，计算比值。

```python
# Worker 启动时自动实测，不需要手动指定
alpha = profile_decode_cost_alpha()
```

**本环境实测值（RTX 3090 × 3 + LLaMA-7B, d=4096）**：

| Worker | GPU | alpha |
|--------|-----|-------|
| Worker 0 | GPU-1 | 90.89 |
| Worker 1 | GPU-2 | 97.09 |
| Worker 2 | GPU-3 | 82.90 |

三卡中位数 α ≈ 90.89。物理含义：一个 decode step 的计算量约等于 prefill 91 个 token。当前 RWPT 公式已不使用 α，但该实测值保留供未来副本放置决策使用。

**非超参数汇总**：

公式中仅 w1、w2、w3 为超参数，其余参数均由系统自动获取：

| 参数 | 获取方式 | 说明 |
|------|---------|------|
| γ | 从模型 `config.json` 的 `hidden_size` 推导 | 数学推导，非经验值 |
| Capacity | 由 `max_total_token_num / 6` 自动推算（`batch_max_tokens`） | 单次 prefill 批次容量，RWPT/Capacity ≈ 排队批次数 |
| r_j | 各 adapter 的 `adapter_config.json` | 训练时确定 |
| input_len_j | Worker 实时采集 | 运行时状态量 |

> **注**：α（decode/prefill 时间比）和 active_decode_seqs 仍由 Worker 实测和上报，但当前 RWPT 公式不使用，保留供未来副本放置决策。

Rank 影响示例：Rank 8 额外 0.13%，Rank 32 额外 0.52%，Rank 128 额外 2.1%。
绝对比例虽小，但队列中数百请求累积后差异显著。

**路由示例**：
```
Worker A: 等待 5 个请求 (input_len=32, rank=8)
  RWPT_A = 5 × 32 × (1 + 0.000163 × 8) = 160.2

Worker B: 等待 1 个请求 (input_len=1024, rank=128)
  RWPT_B = 1 × 1024 × (1 + 0.000163 × 128) = 1045.4

按 QueueLen: A=5, B=1 → 选 B（错误）
按 RWPT:    A=160, B=1045 → 选 A（正确）
```

**消融实验设计**：

| 变体 | load_metric | 负载度量 | 对应公式 | Fallback |
|------|------------|---------|---------|----------|
| Variant A | `queue_length` | 请求数（V1 基线） | $w_2 \cdot QueueLen$ | N/A（无归一化） |
| Variant B | `token_count` | Token 数，无 rank 加权 | $w_2 \cdot \Sigma input\_len / Cap$ | 无（已移除） |
| Variant C (Ours) | `rwpt` | Rank 加权 Token 数（完整 V3） | $w_2 \cdot RWPT / Cap$ | 无（已移除） |

A→B 消融"token 粒度 vs 请求粒度"，B→C 消融"rank 感知 vs 无 rank 感知"。每一步只增加一个因素，形成干净的递进式消融。三个 variant 均无 fallback，量纲独立，确保对比公平。

重点验证指标：P95/P99 尾部延迟、Worker 间负载均衡度（Jain's Fairness Index）。

**w2 甜点搜索**：

移除 decode 折算项和 fallback 机制后，三个 variant 的负载值量纲一致，w2 搜索结果稳定可复现。

已完成三轮独立 w2 搜索实验（实验条件：3×RTX 3090, LLaMA-7B, 100 adapters, α=0.3, req_rate=6.0, duration=120s）：

| Metric | 搜索范围 | 实验套件 | 实验数 |
|--------|---------|---------|--------|
| rwpt | w2 ∈ [0.5, 5.0] | `rwpt-w2-search` | 10 |
| queue_length | w2 ∈ [0.05, 0.8] | `ql-w2-search` | 10 |
| token_count | w2 ∈ [0.3, 4.0] | `tc-w2-search` | 10 |

各 metric 综合最优 w2（吞吐-延迟综合评分）：

| Metric | Best w2 | Throughput | Avg Latency | First Token Latency | P90 Latency | Cache Hit Rate |
|--------|---------|-----------|-------------|--------------------|-----------|----|
| **rwpt** | **0.5** | **5.615** | **10.32s** | **5.39s** | **13.95s** | 81.0% |
| queue_length | 0.1 | 5.325 | 11.85s | 6.74s | 15.89s | 88.8% |
| token_count | 0.8 | 5.498 | 13.03s | 7.70s | 16.93s | 74.4% |

关键发现：
- RWPT 全面领先：吞吐最高（+5.4% vs queue_length），延迟最低（-12.9% vs queue_length, -20.8% vs token_count）
- queue_length 的最优 w2 极小（0.05~0.1），说明粗粒度负载信号加大权重反而破坏缓存亲和性
- token_count 虽然有 token 级粒度，但缺少 rank 加权导致负载估算不准确，表现反而不如 queue_length
- RWPT 的 rank 加权 + Capacity 归一化使其在更大的 w2 范围内保持稳定（w2=0.5~3.0 均表现良好）

数据文件：`benchmarks/routing_comparison_results/w2_search_data/`

**Worker 新增上报字段**：
- `pending_prefill_tokens`: 等待队列中 rank 加权后的 input_len 总和（RWPT 值）
- `pending_raw_tokens`: 等待队列中未加权的 input_len 总和
- `active_decode_seqs`: 当前 batch 中 decode 序列数（保留上报，当前路由公式不使用，供未来副本放置决策）
- `pool_used_ratio`: 内存池整体使用率（KV + LoRA 共享池）

**Rank 感知扩展（✅ 已实现）**：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| $w_3$ | Rank 不匹配惩罚权重 | 0.0（禁用），推荐 2.0-5.0 |
| $RankMismatch_i$ | 归一化 Rank 差异 | |


**4.3 Worker State Synchronization**
- Worker 定期上报状态（100ms 间隔）
- 事件触发上报（Adapter 加载/卸载）
- Router 维护倒排索引：Adapter → Worker 集合

**4.4 Hot Adapter Handling**
- 请求率追踪（滑动窗口）
- 热点检测阈值（默认 10 req/s）
- 热点 Adapter 分散到多个 Worker（避免单点瓶颈）

**4.5 Hot Adapter Replication（热门 Adapter 多副本）**

当单个 Worker 无法满足热门 Adapter 的吞吐需求时，主动将其复制到多个 Worker：

```
热点检测 → 触发复制 → 多 Worker 并行服务 → 吞吐量线性提升
```

| 策略 | 触发条件 | 复制目标 |
|------|---------|---------|
| 被动分散 | 热点请求到达时 | Round-Robin 分配到可用 Worker |
| 主动复制 | 请求率 > 阈值 | 预加载到空闲 Worker |

**收益分析：**
- 单 Worker 服务热点：吞吐受限于单卡
- N Worker 服务热点：吞吐 ≈ N × 单卡（线性扩展）
- 代价：占用 N 份显存，但热点 Adapter 通常值得

**4.6 Cold Start Handling**
- 新 Adapter 首次请求时，所有 Worker 都无缓存
- 策略：选择队列最短的 Worker
- 记录冷启动事件用于分析

---

### Section 5: Multi-Dimensional Adapter Eviction

**5.1 Scoring Model**

$$Score(a) = 0.35 \cdot S_{usage}(a) + 0.35 \cdot S_{recency}(a) + 0.3 \cdot S_{pending}(a)$$

| 维度 | 公式 | 说明 |
|------|------|------|
| 使用频率 | $\frac{\log(1+count_a)}{\log(1+\max_i count_i)}$ | 对数归一化，避免极端值压缩 |
| 时间衰减 | $e^{-\Delta t / \tau}$, $\tau=300s$ | 5 分钟未使用降至 0.37 |
| 队列需求 | $\frac{\log(1+pending_a)}{\log(1+\max_i pending_i)}$ | 等待队列中的请求数 |

**5.2 Dual-Mode Eviction**

| 模式 | 触发时机 | 阈值 | 淘汰比例 |
|------|---------|------|---------|
| 被动淘汰 | 加载新 Adapter 前 | 显存 ≥ 90% | 固定 20% |
| 主动淘汰 (请求完成) | 请求完成时 | 显存 ≥ 90% | 动态 20%-60% |
| 主动淘汰 (批次清空) | 批次完全清空 | 显存 ≥ 80% | 固定 70% |

**动态淘汰比例公式：**
$$ratio = \frac{usage - 0.9}{0.1} \times 0.4 + 0.2$$

**5.3 Protection Mechanisms**
- `preserve_dirs`: 当前批次使用的 Adapter 硬保护
- 队列感知: 通过 $S_{pending}$ 间接保护即将使用的 Adapter
- 连续批处理兼容: 淘汰只在请求完成间隙执行

---

### Section 6: Evaluation

**6.1 Experimental Setup**
- 硬件: 3 × RTX 3090 (24GB)
- 模型: Llama-7B
- Adapters: 50 个 (rank=8/16/32 混合)
- 负载模式: 均匀分布、长尾分布、突发流量

**6.2 Baselines**
- S-LoRA (张量并行, TP=3)
- S-LoRA-DP (数据并行, Round-Robin)
- S-LoRA-DP + Adapter-Aware Routing (本文)

**6.3 Metrics**
- Throughput (req/s)
- Latency (P50, P95, P99)
- Cache Hit Rate (%)
- Adapter Load/Eviction Count
- GPU Utilization (%)

**6.4 Experiments**

| 实验 | 目的 | 预期结果 |
|------|------|---------|
| E1: 吞吐量对比 | 验证数据并行可扩展性 | DP 吞吐量 ≈ 3× TP |
| E2: 缓存命中率 | 验证路由优化效果 | 30% → 70%+ |
| E3: 延迟分布 | 验证冷启动减少 | P99 降低 30-50% |
| E4: 热点吞吐量 | 验证多副本效果 | 热点吞吐 ≈ N× 单副本 |
| E5: 淘汰效率 | 验证评分模型效果 | 淘汰-重加载减少 50%+ |
| E6: 消融实验 | 分析各组件贡献 | 逐步叠加效果 |
| E7: 负载模式 | 验证鲁棒性 | 长尾分布优势最明显 |

---

### Section 7: Related Work

**7.1 LLM Serving Systems**
- vLLM, Orca, FlexGen

**7.2 LoRA Serving**
- S-LoRA, Chameleon, Toppings, LoRAServe

**7.3 Parallel Strategies**
- Tensor Parallelism, Data Parallelism, Pipeline Parallelism

**7.4 Cache and Memory Management**
- LRU, LFU, ARC, 机器学习驱动的缓存策略

---

### Section 8: Conclusion

本文提出了面向高并发多租户场景的数据并行 LoRA 推理系统。通过数据并行架构实现线性可扩展性，通过 Adapter 感知路由提升缓存命中率，通过热门 Adapter 多副本机制解决热点瓶颈，通过多维评分淘汰机制优化显存管理。实验表明，本系统在吞吐量、延迟和资源效率方面均显著优于基线系统。

**Future Work:**
- 混合并行策略（TP + DP）
- 预测式 Adapter 预加载
- 自适应副本数调整

---

## 📊 张量并行 vs 数据并行：对比分析

### 核心对比表

| 维度 | 张量并行 (TP) | 数据并行 (DP) | 高并发场景优势 |
|------|--------------|--------------|---------------|
| 模型大小支持 | 超大模型 ✅ | 中小模型 ✅ | TP |
| 通信开销 | 高（AllReduce）| 无 | **DP** |
| GPU 利用率 | 低-中 | 高 | **DP** |
| 显存效率 | 低（空间浪费）| 高 | **DP** |
| Adapter 容量 | 受限 | 大 | **DP** |
| 可扩展性 | 次线性 | 线性 | **DP** |

### 显存利用率对比（关键差异）

```
示例：Llama-7B (14GB), 3 × RTX 3090 (24GB)

张量并行（TP=3）：
- 模型参数：14GB / 3 ≈ 4.7GB（切分）
- KV Cache：~10GB（复制）
- Adapters：~2GB（复制）
→ 总 Adapter 空间：2GB × 3 = 6GB

数据并行（DP=3）：
- 模型参数：14GB（完整）
- KV Cache：~5GB（独立）
- Adapters：~5GB（独立）
→ 总 Adapter 空间：5GB × 3 = 15GB

结论：数据并行 Adapter 空间是张量并行的 2.5 倍
```

### 互补性说明

数据并行和张量并行是**互补**的，而非替代关系：
- 张量并行：适合超大模型、低并发场景
- 数据并行：适合中小模型、高并发多租户场景
- 混合并行：结合两者优势（后续优化方向）

---

## 🚀 后续优化方向

详见 `优化方向详解.md`，主要包括：

1. **热门 Adapter 主动复制** ⭐⭐⭐⭐⭐ - 多 Worker 并行服务热点，线性提升吞吐
2. ~~**Rank 感知路由**~~ ✅ 已完成 - Phase 2.5 实现
3. **混合并行策略** ⭐⭐⭐⭐⭐ - TP + DP 结合
4. **跨 Worker Adapter 协调** ⭐⭐⭐⭐ - 全局放置策略
5. **预测式 Adapter 预加载** ⭐⭐⭐⭐ - 减少冷启动
6. **QoS 保证和优先级调度** ⭐⭐⭐⭐ - 多租户 SLA

---

## 🎓 目标会议

**顶会 (Tier 1):**
- MLSys
- OSDI
- ATC
- EuroSys

**好会议 (Tier 2):**
- SoCC
- Middleware
- ICPP

---

## 📌 关键文件索引

| 文件 | 功能 |
|------|------|
| `slora/server/router/dp_manager.py` | 数据并行路由管理器 |
| `slora/server/router/gpu_worker.py` | GPU Worker 进程 |
| `slora/server/router/adapter_aware_router.py` | 智能路由器 |
| `slora/server/router/worker_state.py` | Worker 状态数据结构 |
| `slora/server/router/worker_state_reporter.py` | 状态上报器 |
| `slora/server/router/worker_state_cache.py` | 状态缓存 |
| `slora/server/router/model_infer/infer_adapter.py` | Adapter 评分与淘汰 |
| `slora/server/router/model_infer/model_rpc.py` | 模型 RPC 层 |
