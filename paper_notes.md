# 多租户 LoRA 推理系统论文思路整理

> **📅 最后更新**: 2026-03-31
> **📌 状态**: w2 甜点搜索 ✅ 全部完成 | alpha 鲁棒性实验 ✅ 完成（**w1/w2 主线，w3=0**）| 显式 **w3** 🔀 **备选、非主推** | 热门 Adapter **主动多副本 ⏳ 未实现**（仅有设计稿 + 路由层**被动分散**）

---

## 📝 论文定位

基于 S-LoRA 系统，提出面向高并发多租户场景的**多 Worker 数据并行**推理架构，包含四个核心贡献：
1. **多 Worker 数据并行架构**：每个 GPU 独立运行完整模型实例，解决张量并行在高并发场景的可扩展性瓶颈
2. **RWPT 感知路由**：Rank-Weighted Pending Tokens 将 LoRA Rank 异构性纳入负载估算，结合缓存亲和性实现 Pareto 最优调度（消融实验：queue_length → token_count → RWPT 递进验证）
3. **RWPT 驱动的热门 Adapter 主动复制（设计完成，系统未实现）**：机制与阈值推导已完成文档化；**工程实现与实验尚未完成**。当前线上等价能力仅为路由层**被动分散**（热点 Round-Robin）
4. **多维评分淘汰机制**：结合使用频率、时间衰减和队列需求的精细化显存管理

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
- **Rank 与路由（主线）**：通过 **RWPT 负载项**内嵌 \(r_j\) 与 \(\gamma\)，在 **w1/w2** 框架下完成「负载侧的 Rank 感知」；**本文与 benchmark 均不依赖显式 w3 项**（详见 2.1 备选说明）

### 2.1 显式 Rank 匹配项 w3（🔀 **备选 / 非主线**）
> **定位**：与 RWPT 正交的**可选扩展**——在评分中增加 \(-w_3\cdot RankMismatch\)（请求 rank vs Worker 当前批次 rank 分布）。代码与 CLI 曾打通，但增益有限、调参成本高，**已基本放弃作为论文与实验主要内容**；保留作附录或 future work 即可。

- ✅ Worker 可上报批次 Rank 分布（avg/min/max）；路由侧曾支持 **w3** 与 `--routing-w3` / `--default-lora-rank`
- ⏸️ **叙事与实验默认**：**w3 = 0（关闭）**；w2 搜索、alpha 鲁棒性等结果均在 **无 w3** 前提下取得
- 若写 related work 对比：Toppings (ATC 2025) 等「显式 rank-aware 调度」可与本文 **RWPT 负载建模**对照；**w3 批次匹配不是本文贡献重点**

### 3. 热门 Adapter 多副本 (⏳ **主动多副本未实现**)
- ✅ 热点检测（请求率追踪）
- ✅ 被动分散（热点请求 Round-Robin 分配）— **当前唯一已落地的「多 Worker 分担热点」手段**
- ✅ 主动复制**设计**完成（RWPT 驱动的算力告警式复制，详见 `hot_adapter_replication_design.md`）
- ⏳ 主动复制**代码与评测** — **未开发 / 未纳入本文实验**

### 4. Adapter 评分与淘汰机制 (✅ 已完成)
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

本文基于 S-LoRA 系统，提出面向高并发多租户场景的多 Worker 数据并行推理架构。主要贡献包括：(1) 多 Worker 数据并行架构，每个 GPU 独立运行完整模型实例，实现线性可扩展性；(2) Rank-Weighted Pending Tokens (RWPT) 感知路由，将 LoRA Rank 异构性纳入负载估算，消除队头阻塞；(3) **（设计层面）** RWPT 驱动的热门 Adapter 主动复制机制：利用路由评分函数的决策边界推导复制阈值，零新增超参数的自适应副本管理——**该条目的系统实现与实验验证仍为未来工作**；(4) 多维评分驱动的双模式淘汰机制实现精细化显存管理。

实验表明，相比原始 S-LoRA 张量并行模式，本系统在 3 GPU 配置下实现 X 倍吞吐量提升，缓存命中率从 30% 提升至 70%+，P90 TTFT 降低 X%。**主动多副本尚未实现，本文实验不包含「多副本热点吞吐 N×」类结论**；热点场景当前仅体现为路由层被动分散 + RWPT/消融与 alpha 鲁棒性结果。

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
- RWPT 感知路由：Rank-Weighted Pending Tokens 将 LoRA Rank 异构性纳入负载估算，结合缓存亲和性实现 Pareto 最优调度
- RWPT 驱动的热门 Adapter 主动复制（**设计提案**；**未实现**）：利用路由评分函数的决策边界作为复制触发阈值，Worker 视角揪出元凶 + N_max 全局护栏，零新增超参数；**与多维淘汰的协同已设计，代码与评测待完成**
- 多维评分淘汰：使用频率 + 时间衰减 + 队列需求，与副本保护期协同

**1.4 Contributions**
1. 设计并实现多 Worker 数据并行推理架构，每个 Worker 独立运行完整模型，实现线性可扩展性
2. 提出 RWPT（Rank-Weighted Pending Tokens）感知路由算法，将 LoRA Rank 异构性纳入负载估算，消除队头阻塞，在所有流量分布下取得 Pareto 最优的延迟-命中率 trade-off
3. **设计** RWPT 驱动的热门 Adapter 主动复制机制（**实现与系统评测为未来工作**），利用路由评分函数的决策边界推导复制阈值（T_congestion = w1/w2 × Capacity），目标为零新增超参数的自适应副本管理
4. 设计多维评分驱动的双模式淘汰机制，优化显存管理
5. 全面的实验评估，验证系统在不同负载模式下的性能

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

**本文主公式（实验与叙事均采用，$w_3=0$）：**

$Score_i(r) = w_1 \cdot \mathbf{I}(hit) - w_2 \cdot \frac{RWPT_i}{Capacity_i}$

**备选扩展（可选，默认关闭，非主线）：** 可叠加显式批次匹配惩罚 \(-w_3 \cdot RankMismatch_i\)（实现上独立于 `load_metric` 三 variant）。**本文不报告 w3 消融；benchmark 配置中 routing_w3 均为 0。**

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

**主线**仅需调 **w1、w2**（本文全部实验）；**w3** 为可选超参，默认 0，**不作为主要研究内容**。其余量均由系统自动获取：

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

A→B 消融"token 粒度 vs 请求粒度"，B→C 消融「**负载侧**是否把 LoRA rank 纳入 pending 工作量（RWPT vs 纯 token 计数）」。每一步只增加一个因素，形成干净的递进式消融。三个 variant 均无 fallback，量纲独立，确保对比公平。**说明**：这里的「rank 感知」指 **RWPT 内的 \(r_j\) 加权**，与 **显式 w3·RankMismatch** 正交；后者**未纳入** A/B/C 消融与本文图表。

重点验证指标：P95/P99 尾部延迟、Worker 间负载均衡度（Jain's Fairness Index）。

**w2 甜点搜索**：

移除 decode 折算项和 fallback 机制后，三个 variant 的负载值量纲一致，w2 搜索结果稳定可复现。

已完成三轮独立 w2 搜索实验（实验条件：3×RTX 3090, LLaMA-7B, 100 adapters, α=0.3, req_rate=6.0, duration=120s, **max_lora_ratio=0.2**）：

| Metric | 搜索范围 | 实验套件 | 状态 |
|--------|---------|---------|------|
| rwpt | w2 ∈ [0.5, 5.0] | `rwpt-w2-search` | ✅ |
| token_count | w2 ∈ [0.3, 4.0] | `tc-w2-search` | ✅ |
| queue_length | w2 ∈ [0.05, 0.8] | `ql-w2-search` | ✅ |

RWPT w2 搜索结果（ratio=0.2，淘汰机制已修复）：

| w2 | Throughput | P90 Latency | Cache HR |
|----|-----------|-------------|----------|
| 0.5 | 5.444 | 14.28s | 68.3% |
| 1.0 | 5.553 | 14.14s | 72.1% |
| 1.5 | 5.579 | 14.23s | 70.0% |
| 2.0 | 5.575 | 14.11s | 69.6% |
| **2.5** | **5.521** | **14.07s** | **70.0%** |
| 3.0 | 5.558 | 14.20s | 67.2% |
| 3.5 | 5.581 | 14.86s | 66.9% |
| 4.0 | 5.617 | 14.44s | 67.1% |
| 4.5 | 5.594 | 14.59s | 66.9% |
| 5.0 | 5.610 | 14.26s | 65.8% |

TC w2 搜索结果（ratio=0.2）：

| w2 | Throughput | P90 Latency | Cache HR |
|----|-----------|-------------|----------|
| 0.3 | 5.560 | 14.59s | 67.2% |
| 0.5 | 5.491 | 15.01s | 72.8% |
| 0.8 | 5.504 | 14.95s | 73.9% |
| 1.0 | 5.515 | 14.94s | 73.2% |
| **1.5** | **5.508** | **14.68s** | **75.4%** |
| **2.0** | **5.611** | **14.51s** | **73.6%** |
| 2.5 | 5.499 | 15.08s | 73.2% |
| 3.0 | 5.589 | 14.61s | 73.8% |
| 3.5 | 5.550 | 14.87s | 72.5% |
| 4.0 | 5.546 | 14.76s | 72.8% |

SLA 模型甜点值：RWPT w2=1.0, TC w2=1.5, QL w2=0.10（各自最优 w2 用于 alpha 鲁棒性对比）。

QL w2 搜索结果（ratio=0.2）：

| w2 | Throughput | P90 Latency | P90 TTFT | Cache HR |
|----|-----------|-------------|----------|----------|
| 0.05 | 5.513 | 19.97s | 11.75s | 63.2% |
| **0.10** | **5.504** | **15.83s** | **9.36s** | **66.7%** |
| 0.15 | 5.456 | 15.57s | 9.09s | 64.9% |
| 0.20 | 5.556 | 15.64s | 9.20s | 58.3% |
| 0.30 | 5.523 | 15.90s | 9.49s | 59.2% |
| 0.40 | 5.550 | 15.87s | 9.05s | 59.0% |
| 0.50 | 5.236 | 16.11s | 8.54s | 52.5% |
| 0.60 | 5.490 | 15.29s | 9.04s | 57.2% |
| 0.70 | 5.492 | 15.66s | 8.89s | 56.1% |
| 0.80 | 5.460 | 15.77s | 9.37s | 55.4% |

QL 特征：w2=0.05 时 P90 延迟飙升至 20s（负载惩罚过弱，缓存绑架），w2=0.10 为 SLA 约束下最优平衡点。

关键变化（相比旧 ratio=0.4 实验）：
- Cache HR 从 80-94% 降至 65-75%，淘汰机制正常工作
- 每 Worker 约 18-19 个 adapter（vs 旧 ratio=0.4 约 37 个），缓存压力显著增大
- 三种 metric 的性能差异在高压力下更明显

数据文件：`benchmarks/routing_comparison_results/<suite_name>/results.jsonl`
汇总工具：`benchmarks/routing_comparison_results/extract_w2_search_data.py`

**Alpha 鲁棒性实验（alpha-robustness-sla）**：

在 w2 甜点搜索确定各 metric 最优 w2 后，固定 w2 变化流量偏斜度 α，验证三种策略在不同流量分布下的鲁棒性。

实验条件：3×RTX 3090, LLaMA-7B, 100 adapters, req_rate=6.0, duration=240s, max_lora_ratio=0.2。
各 metric 使用 SLA 约束模型选出的最优 w2：RWPT w2=1.0, TC w2=1.5, QL w2=0.10。

| α | Metric | P90 TTFT (s) | Cache HR (%) | P90 Latency (s) | Throughput |
|---|--------|-------------|-------------|-----------------|-----------|
| 0.1 | Queue Length | 9.23 | 85.3 | 15.69 | 5.806 |
| 0.1 | Token Count | 8.79 | 88.0 | 15.16 | 5.799 |
| 0.1 | **RWPT** | **7.59** | **85.7** | **14.24** | **5.811** |
| 0.3 | Queue Length | 9.79 | 63.9 | 16.22 | 5.721 |
| 0.3 | Token Count | 8.97 | 72.6 | 15.56 | 5.771 |
| 0.3 | **RWPT** | **8.13** | **69.2** | **14.99** | **5.783** |
| 0.8 | Queue Length | 10.28 | 48.8 | 17.19 | 5.749 |
| 0.8 | Token Count | 9.86 | 54.6 | 16.73 | 5.756 |
| 0.8 | **RWPT** | **9.11** | **46.5** | **16.03** | **5.761** |

**核心发现**：

**1. RWPT 的尾延迟统治力（图 1：P90 TTFT 分组柱状图）**

RWPT 在所有 α 下均取得最低 P90 TTFT：
- α=0.1（极端热点）：7.59s vs TC 8.79s vs QL 9.23s，RWPT 比 QL 低 17.8%
- α=0.3（中等偏斜）：8.13s vs TC 8.97s vs QL 9.79s，RWPT 比 QL 低 17.0%
- α=0.8（长尾均匀）：9.11s vs TC 9.86s vs QL 10.28s，RWPT 比 QL 低 11.4%

无论流量分布如何变化，RWPT 始终保持最优尾延迟。**在 RWPT 负载项中对 rank 加权**的估算，有效缓解了纯队列/token 计数下「高 rank 请求拖尾」导致的队头阻塞（Head-of-Line Blocking）。**此结论与显式 w3 项无关**（实验配置 w3=0）。

**2. 有毒局部性（Toxic Locality）现象（图 2：Cache Hit Rate 分组柱状图）**

Token Count 在所有 α 下维持最高缓存命中率（88.0% / 72.6% / 54.6%），但这种"高命中率"并未转化为低延迟——反而伴随着最差的尾延迟表现（仅次于 QL）。

这揭示了一种"有毒局部性"：Token Count 不感知 Rank 差异，盲目追求缓存亲和性，将高 Rank 请求持续路由到已缓存该 Adapter 的 Worker，即使该 Worker 已经因高 Rank 计算负载而过载。结果是局部性越高，负载越不均衡，尾延迟越差。

RWPT 则展现了精确的"断臂求生"机制：以微小的缓存代价（相比 TC 低约 2-8 个百分点）换取显著的延迟收益（P90 TTFT 降低 13-18%）。RWPT 的 Rank 加权使其能够识别"缓存命中但计算代价高"的情况，主动将请求分散到负载更轻的 Worker。

**3. Pareto 前沿分析（图 3：Cache Hit Rate vs P90 TTFT 散点图）**

在 Cache Hit Rate × P90 TTFT 的二维 trade-off 空间中，RWPT 的数据点始终处于 Pareto 最优边界（左下角 = 低延迟 + 合理命中率）。

Token Count 和 Queue Length 的数据点被"缓存绑架"——它们要么追求高命中率但延迟恶化（TC），要么命中率和延迟都不理想（QL）。RWPT 打破了这种次优局面，表明 **在负载估计中显式计入 LoRA rank（RWPT）** 是实现 Pareto 最优延迟–命中率折中的关键；**非依赖额外的 w3 批次匹配项**。

图表文件：`benchmarks/routing_comparison_results/alpha-robustness-sla/`
- `fig1_p90_ttft.pdf` — P90 TTFT 分组柱状图
- `fig2_cache_hit_rate.pdf` — Cache Hit Rate 分组柱状图
- `fig3_pareto_scatter.pdf` — Pareto 散点图
- `plot_figures.py` — 图表生成脚本

**Worker 新增上报字段**：
- `pending_prefill_tokens`: 等待队列中 rank 加权后的 input_len 总和（RWPT 值）
- `pending_raw_tokens`: 等待队列中未加权的 input_len 总和
- `active_decode_seqs`: 当前 batch 中 decode 序列数（保留上报，当前路由公式不使用，供未来副本放置决策）
- `pool_used_ratio`: 内存池整体使用率（KV + LoRA 共享池）

**备选：显式 Rank 匹配项 w3（非主线，默认关闭）**：

| 参数 | 含义 | 本文采用 |
|------|------|----------|
| $w_3$ | 请求 rank 与 Worker 批次 rank 分布不匹配的惩罚权重 | **0（关闭）** |
| $RankMismatch_i$ | 归一化 rank 差异 | 不展开 |

实现上仍可通过 CLI 打开供复现或附录；**论文正文与主实验以 RWPT（w1/w2）为唯一路由叙事。**


**4.3 Worker State Synchronization**
- Worker 定期上报状态（100ms 间隔）
- 事件触发上报（Adapter 加载/卸载）
- Router 维护倒排索引：Adapter → Worker 集合

**4.4 Hot Adapter Handling**
- 请求率追踪（滑动窗口）
- 热点检测阈值（默认 10 req/s）
- 热点 Adapter 分散到多个 Worker（避免单点瓶颈）

**4.5 Cold Start Handling**
- 新 Adapter 首次请求时，所有 Worker 都无缓存
- 策略：选择队列最短的 Worker
- 记录冷启动事件用于分析

---

### Section 5: Hot Adapter Replication（热门 Adapter 多副本 — RWPT 驱动）

> **实现状态（与全文对齐）**：本节为**主动多副本**的机制设计与论文叙事；**当前仓库未实现**跨 Worker 主动加载/复制逻辑，运行时可用的仅为 **Section 4.4** 所述**热点被动分散**。下文「当副本存在时…」类表述均指**设计目标行为**，非本文已测功能。

**5.1 Motivation**

传统缓存复制基于访问频率或未命中率，在异构大模型推理中是滞后的。我们**设想**利用 RWPT 进行主动防御式复制（**待实现**）。

**5.2 RWPT-Driven Replication Decision Flow**

核心决策流程：

```
Step 1: 算力拥塞检测 — 后台巡检线程检查 EMA(RWPT_i/Capacity)
        若 EMA > T_congestion → 该 Worker 即将发生队头阻塞
                    │
Step 2: 揪出元凶 — 遍历过载 Worker 的等待队列，
        找出对 RWPT 贡献最大的 adapter (culprit)
                    │
Step 3: 全局防爆护栏 — 检查 culprit 的全局副本数
        若 ≥ N_max → 放弃（保护长尾生存空间）
        否则 → 复制到 RWPT 最低的无缓存 Worker
```

**5.3 Design Details**

**关键设计**：
- T_congestion = (w1/w2) × Capacity（零新增超参数，从路由评分函数决策边界推导）
- N_max = num_workers - 1（强制为长尾 adapter 预留显存）
- T_cooldown = 5~10s（复制冷却期，防止新副本生效前连续扩容打满 N_max）
- 渐进式复制：每个决策周期只增加 1 个副本
- 回收：复用现有三维评分淘汰机制，新副本有 30s 保护期
- 抗抖动：EMA 平滑 + 滞后系数（触发阈值 > 回收阈值）+ 复制冷却期

详细设计文档：`hot_adapter_replication_design.md`

---

### 📖 论文叙事主线：大脑 → 肌肉 → 免疫系统

> 以下是 Section 4（Routing）→ Section 5（Replication）→ Section 6（Eviction）的论文叙事设计，用于指导论文写作。
> 导师定调：顶会系统论文讲究"草蛇灰线，伏脉千里"——前一个机制的缺陷，必须是后一个机制的登场契机。

**核心隐喻**：RWPT 是大脑（传感器），多副本是肌肉（执行器），三维淘汰是免疫系统（护盾）。三者形成"发现瓶颈 → 打破瓶颈 → 解决副作用"的因果链，缺一不可。**当前实现进度**：大脑（RWPT 路由）与免疫系统（三维淘汰）已落地；**肌肉（主动多副本）仍为设计叙事，未编码实现**。

---

**第一幕：发现问题——缓存亲和性 vs 负载均衡的零和博弈**

多 Worker 数据并行架构下，路由器面临一个根本矛盾：把请求发给有缓存的 worker（高命中率但可能过载），还是发给空闲的 worker（低延迟但冷启动）。传统的 queue_length 路由只看请求数，完全忽略了 LoRA 场景下请求之间的计算代价差异（rank 8 vs rank 128）。

**第二幕：🧠 诊断——RWPT 让路由器"看见"真实负载（大脑与传感器）**

我们提出 RWPT（Rank-Weighted Pending Tokens），将 LoRA Rank 异构性纳入负载估算。消融实验（queue_length → token_count → RWPT）证明：
- token 粒度比请求粒度更准确（A→B）
- rank 加权比纯 token 更准确（B→C）
- RWPT 在所有流量分布下取得 Pareto 最优的延迟-命中率 trade-off

RWPT 是单副本拓扑下的绝对最优解——它能在拥塞边缘做出最完美的"断臂求生（放弃缓存以保延迟）"。alpha-robustness 实验揭示：RWPT 以微小的缓存代价（比 token_count 低 2-8%）换取显著的延迟收益（P90 TTFT 降低 13-18%）。

**🔗 衔接点 1：RWPT → 多副本（Section 4→5 过渡段，建议用于论文正文）**

> "然而，单副本路由的本质是'在既定物理拓扑下做最优妥协'。当单一 Worker 的绝对物理算力被爆款 Adapter 彻底击穿时，无论 RWPT 怎么妥协，P90 延迟都会不可避免地崩塌。RWPT 作为极其敏锐的'算力预言机（Compute Oracle）'，已经向我们发出了系统局部物理极限的明确告警。要彻底打破这一零和博弈，系统必须从'被动路由'走向'主动的拓扑重构'。"

**第三幕：💪 治疗——RWPT 驱动的动态多副本（肌肉与执行器）**

如果热门 adapter 同时缓存在多个 worker 上，路由器就不需要在"缓存命中"和"负载均衡"之间二选一了——两个 worker 都有缓存，选负载低的那个就行。零和博弈变成了正和博弈。

RWPT 天然提供了复制决策所需的全部信号：
- **什么时候复制**：当 worker 的 RWPT 超过路由评分函数的决策边界（T_congestion = w1/w2 × Capacity），说明路由器已经被迫放弃缓存命中，这就是复制的最佳时机
- **复制谁**：RWPT 贡献最大的 adapter（揪出元凶）
- **复制到哪**：RWPT 最低的无缓存 worker（负载熔断：EMA < T_congestion × 0.9，不检查显存——复制时淘汰机制自动腾空间）
- **什么时候回收**：RWPT 信号下降 + 现有淘汰机制自然回收

零新增超参数——T_congestion 从路由评分函数推导，N_max 从集群规模推导，T_cooldown 从物理加载时间推导。多副本是 RWPT 的数学边界自然推导出来的正向反馈环。

**🔗 衔接点 2：多副本 → 三维淘汰（Section 5→6 过渡段，建议用于论文正文）**

> "动态多副本机制虽然在理论上打破了算力孤岛，但在极度受限的显存环境（例如 ratio=0.2，每 Worker 仅 18-19 个 adapter 槽位）中，暴力的动态复制会引发分布式系统中臭名昭著的'缓存冷启动颠簸（Cold-Start Thrashing）'。新分配的副本在建立初期，由于缺乏历史热度积累（S_usage=0, S_pending=0），极易被底层的缓存淘汰器误杀，导致系统陷入'无尽搬家'的 PCIe 带宽灾难。因此，多副本拓扑重构的成功，必须仰赖于一个具备'前瞻预测能力'的底层缓存驱逐引擎作为坚实护盾。"

**第四幕：🛡️ 护盾——前瞻性三维缓存驱逐（免疫系统）**

传统 LFU/LRU 是"向后看"的——只统计历史访问频率和最近访问时间。在大模型异步推理中，这存在致命盲区：一个 adapter 可能历史访问次数为 0（刚被复制过来），但等待队列中有 50 个请求即将使用它。LRU 会毫不犹豫地淘汰它。

我们的三维评分引入了"向前看"的维度——S_pending（队列需求感知）：
- S_usage（滑动窗口频率）：近期活跃度，防止历史累计膨胀退化为纯 LRU
- S_recency（时间衰减）：最近访问时间
- S_pending（队列感知）：**前瞻性维度**，有等待请求的 adapter 不应被淘汰

三维评分与多副本的协同闭环：
- RWPT 负责把副本"生"出来（触发复制）
- 保护期（30s grace period）负责让它"活"过通信延迟（防止冷启动误杀）
- S_pending 评分负责让它在系统中真正"立"住脚跟（有请求排队 → 评分高 → 不被淘汰）

**第五幕：协同反馈环——诊断指导治疗，护盾保障治疗**

RWPT 路由、多副本、三维淘汰不是三个独立机制，而是一个协同系统：
- RWPT 是传感器（告诉系统哪里有问题）→ 触发多副本
- 多副本是执行器（解决问题）→ 依赖淘汰机制腾空间 + 保护新副本
- 三维淘汰是护盾（保障执行器不自爆）→ 淘汰冷 adapter 为新副本让路

自洽反馈环：副本越多 → 热点压力越小 → w2 最优值越小 → T_congestion 越高 → 复制触发越保守 → 不会过度复制。

**第六幕：弹性降级——系统的物理边界与生存能力**

多副本机制不能凭空创造算力。当所有 worker 都处于饱和状态时（如 6 req/s），全局不存在"算力洼地"可供复制，此时系统的行为是：

- 巡检线程检测到所有 worker 的 EMA(RWPT) 超过 T_congestion
- Step 2 能揪出元凶，但 Step 3 的目标选择找不到满足条件的候选 worker（所有 worker 的 RWPT 都过高）
- 复制机制静默退出，系统平滑退化为纯 RWPT 单副本路由

这不是缺陷，而是**设计意图**（**主动多副本实现后**）：系统在有余量时表现为多副本拓扑重构引擎（**预期** P90 TTFT 等尾延迟显著改善），在极限高压下自动休眠执行器，退化为已验证的 RWPT 单副本路由基线（**alpha-robustness 实验已覆盖单副本 RWPT 路由**）。论文中若同时写设计与实测，应明确分界，避免审稿人误以为多副本已评测。

**适用场景说明（针对未来实现与实验）**：主动多副本在「局部热点 + 全局有余量」场景下**理论上**收益最大（如 4 req/s + α=0.1）。当所有 worker 饱和（如 6 req/s）时，复制无法创造总算力；高负载可用于验证「不退化」——**上述负载数字为设计阶段实验规划，非本文已跑通的多副本数据**。

**叙事总结**：发现零和博弈 → 🧠 RWPT 诊断（单副本最优解，但触及物理极限）→ 💪 多副本打破极限（RWPT 驱动的拓扑重构）→ 🛡️ 三维淘汰护航（防止副本冷启动颠簸）→ 三者协同形成闭环 → 极限压力下平滑降级。每一步的缺陷自然引出下一步，不是拼凑的独立贡献，而是一条"发现瓶颈 → 打破瓶颈 → 解决副作用"的因果链。

---

### Section 6: Multi-Dimensional Adapter Eviction（前瞻性三维缓存驱逐 — 多副本的护盾）

> 传统 LFU/LRU 是"向后看"的——只统计历史访问频率和最近访问时间。在大模型异步推理 + 动态多副本的场景中，这存在致命盲区：一个刚被复制过来的热门 adapter，历史访问次数为 0（S_usage=0），但等待队列中可能有数十个请求即将使用它。纯 LRU 会毫不犹豫地淘汰它，导致"无尽搬家"的 PCIe 带宽灾难。本节介绍的三维评分模型引入了"向前看"的队列感知维度（S_pending），与保护期机制协同，为多副本拓扑重构提供坚实的底层护盾。

**6.1 Scoring Model**

$$Score(a) = 0.35 \cdot S_{usage}(a) + 0.35 \cdot S_{recency}(a) + 0.3 \cdot S_{pending}(a)$$

| 维度 | 公式 | 说明 |
|------|------|------|
| 使用频率 | $\frac{\log(1+count_a^{window})}{\log(1+\max_i count_i^{window})}$ | 滑动窗口（300s）内对数归一化，防止长期运行后 max_count 膨胀导致退化为纯 LRU |
| 时间衰减 | $e^{-\Delta t / \tau}$, $\tau=300s$ | 5 分钟未使用降至 0.37 |
| 队列需求 | $\frac{\log(1+pending_a)}{\log(1+\max_i pending_i)}$ | 等待队列中的请求数 |

**6.2 Dual-Mode Eviction**

| 模式 | 触发时机 | 阈值 | 淘汰比例 |
|------|---------|------|---------|
| 被动淘汰 | 加载新 Adapter 前 | 显存 ≥ 90% | 固定 20% |
| 主动淘汰 (请求完成) | 请求完成时 | 显存 ≥ 90% | 动态 20%-60% |
| 主动淘汰 (批次清空) | 批次完全清空 | 显存 ≥ 80% | 固定 70% |

**动态淘汰比例公式：**
$$ratio = \frac{usage - 0.9}{0.1} \times 0.4 + 0.2$$

**6.3 Protection Mechanisms**
- `preserve_dirs`: 当前批次使用的 Adapter 硬保护
- 队列感知: 通过 $S_{pending}$ 间接保护即将使用的 Adapter
- 连续批处理兼容: 淘汰只在请求完成间隙执行

---

### Section 7: Evaluation

**7.1 Experimental Setup**
- 硬件: 3 × RTX 3090 (24GB), 数据并行模式
- 模型: LLaMA-7B
- Adapters: 100 个（alpaca-lora-7b rank=16 + bactrian-x-llama-7b-lora rank=64 交替）
- max_lora_ratio: 0.2（每 Worker 约 18-19 个 adapter）
- max_total_token_num: 15000
- 淘汰阈值: 85%
- 负载模式: Power Law 分布 (α=0.3), req_rate=6.0

**7.2 Baselines**
- S-LoRA (张量并行, TP=3)
- S-LoRA-DP (数据并行, Round-Robin)
- S-LoRA-DP + Adapter-Aware Routing (本文)

**7.3 Metrics**
- Throughput (req/s)
- Latency (P50, P95, P99)
- Cache Hit Rate (%)
- Adapter Load/Eviction Count
- GPU Utilization (%)

**7.4 Experiments**

| 实验 | 目的 | 预期结果 |
|------|------|---------|
| E1: 吞吐量对比 | 验证数据并行可扩展性 | DP 吞吐量 ≈ 3× TP |
| E2: 缓存命中率 | 验证路由优化效果 | 30% → 70%+ |
| E3: 延迟分布 | 验证冷启动减少 | P99 降低 30-50% |
| E4: 热点吞吐量 | 验证主动多副本效果 | **未实现**；待复制落地后验证「热点吞吐 ≈ N× 单副本」 |
| E5: 淘汰效率 | 验证评分模型效果 | 淘汰-重加载减少 50%+ |
| E6: 消融实验 | 分析各组件贡献 | 逐步叠加效果 |
| E7: 负载模式 | 验证鲁棒性 | 长尾分布优势最明显 |

---

### Section 8: Related Work

**8.1 LLM Serving Systems**
- vLLM, Orca, FlexGen

**8.2 LoRA Serving**
- S-LoRA, Chameleon, Toppings, LoRAServe

**8.3 Parallel Strategies**
- Tensor Parallelism, Data Parallelism, Pipeline Parallelism

**8.4 Cache and Memory Management**
- LRU, LFU, ARC, 机器学习驱动的缓存策略

---

### Section 9: Conclusion

本文提出了面向高并发多租户场景的数据并行 LoRA 推理系统。通过数据并行架构实现线性可扩展性，通过 Adapter 感知路由（含 RWPT 与消融）提升缓存命中率与尾延迟表现，通过多维评分淘汰机制优化显存管理。**热门 Adapter 主动多副本仍为设计方案，尚未在系统中实现**；当前热点缓解依赖路由层被动分散与上述路由/淘汰机制。实验表明，本系统在已完成的评测范围内，吞吐量、延迟和资源效率相对基线有显著改善。

**Future Work:**
- **（备选）** 显式 w3 批次 rank 匹配：代码路径保留，曾观测增益有限；若需与 Toppings 等对照可写附录，**不作为主线**
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

1. **热门 Adapter 主动复制** ⏳ **未实现**（✅ 设计文档：`hot_adapter_replication_design.md`；❌ 代码与实验）
2. **RWPT（w1/w2）路由与消融** ✅ 主线已完成；**显式 w3** 🔀 备选（默认关闭，不主推）
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
