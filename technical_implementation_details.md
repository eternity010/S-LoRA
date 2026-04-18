# 技术实现细节文档

> **📅 创建时间**: 2026-04-16
> **📌 用途**: 供开题报告 4.4.1 技术可行性撰写参考

---

## 1. S-LoRA 具体改了哪些模块

### 新增文件（全部为本人开发）

| 文件 | 行数 | 功能 |
|------|------|------|
| `slora/server/router/dp_manager.py` | ~1700 | 数据并行路由管理器：Worker 进程管理、ZMQ 通信、请求路由、状态接收 |
| `slora/server/router/gpu_worker.py` | ~1700 | GPU Worker 进程：独立推理引擎、请求处理、状态上报、预加载处理 |
| `slora/server/router/adapter_aware_router.py` | ~1000 | 智能路由器：RWPT 评分函数、缓存亲和性、热点检测、倒排索引 |
| `slora/server/router/worker_state.py` | ~300 | 数据结构：WorkerState、RoutingStats、RoutingConfig |
| `slora/server/router/worker_state_reporter.py` | ~200 | Worker 端状态上报器（100ms 心跳） |
| `slora/server/router/worker_state_cache.py` | ~150 | Router 端状态缓存与健康监控 |
| `slora/server/router/replica_manager.py` | ~350 | 热门 Adapter 主动复制决策引擎 |
| `slora/server/router/round_robin_router.py` | ~50 | 轮询路由基线 |
| `slora/server/router/response_merger.py` | ~100 | 多 Worker 响应合并 |
| `benchmarks/routing_experiment/config.py` | ~120 | 实验配置数据类 |
| `benchmarks/routing_experiment/runner.py` | ~1200 | 实验执行框架（服务器管理、基准测试、结果收集） |
| `benchmarks/routing_experiment/suite.py` | ~280 | 预定义实验矩阵 |
| `benchmarks/routing_experiment/analyzer.py` | ~200 | 结果分析 |
| `benchmarks/routing_experiment/charts.py` | ~150 | 图表生成 |

### 修改的原有文件

| 文件 | 修改内容 |
|------|---------|
| `slora/server/router/model_infer/infer_adapter.py` | 新增三维评分模型（S_usage + S_recency + S_pending）、双模式淘汰、副本保护列表、滑动窗口使用统计 |
| `slora/server/router/model_infer/model_rpc.py` | 新增 `exposed_preload_adapter()` 预加载端点、淘汰触发 RPC |
| `slora/server/router/manager.py` | 新增数据并行模式分支，创建 DataParallelRouterManager |
| `slora/server/api_server.py` | 新增路由配置热更新 API、缓存重置 API、副本相关命令行参数 |
| `benchmarks/launch_server.py` | 新增数据并行、路由策略、副本相关的命令行参数传递 |

### 原始 S-LoRA 是什么样的

原始 S-LoRA 只有张量并行模式（TP）：多张 GPU 协同处理同一个请求，模型参数切分到各 GPU，每次推理需要 AllReduce 同步。没有多 Worker 概念，没有路由器，没有 Adapter 缓存亲和性，Adapter 淘汰用简单的"用完即弃"策略。

### 扩展成了什么

每张 GPU 独立运行完整模型实例（Worker），Worker 之间无需同步。Router 端根据 RWPT 评分函数决定请求路由到哪个 Worker，Worker 端通过心跳上报状态。热门 Adapter 可以被主动复制到多个 Worker 实现分流。Adapter 淘汰从"用完即弃"升级为三维评分模型。

---

## 2. Router 和 Worker 之间的 ZMQ 通信

### 通信拓扑

```
API Server (HTTP)
    ↓ ZMQ PUSH
HttpServerManager
    ↓ ZMQ PUSH (pickle 序列化)
DataParallelRouterManager (ZMQ PULL)
    ├─ ZMQ PUSH → Worker 0 (ZMQ PULL)  [请求转发，端口 50000]
    ├─ ZMQ PUSH → Worker 1 (ZMQ PULL)  [请求转发，端口 50001]
    └─ ZMQ PUSH → Worker 2 (ZMQ PULL)  [请求转发，端口 50002]
    
Worker 0/1/2 (ZMQ PUSH) → ResponseMerger (ZMQ PULL) → Detokenization → Client

Worker 0/1/2 (ZMQ PUSH) → Router (ZMQ PULL)  [状态上报，端口 52000]
```

### 心跳上报

每个 Worker 通过 `WorkerStateReporter` 每 100ms 发送一次状态心跳（ZMQ PUSH → Router 的 ZMQ PULL）。

心跳消息内容（JSON 格式）：
```json
{
    "type": "worker_state",
    "worker_id": 0,
    "cached_adapters": ["adapter_A", "adapter_B", ...],
    "queue_length": 5,
    "gpu_memory_free": 8000000000,
    "timestamp": 1710000000.0,
    "avg_rank": 32.0,
    "min_rank": 8,
    "max_rank": 64,
    "pending_prefill_tokens": 4500,
    "pending_raw_tokens": 4200,
    "active_decode_seqs": 12,
    "pool_used_ratio": 0.85,
    "top_k_rwpt_adapters": [["adapter_A", 3500.0], ["adapter_C", 1200.0]],
    "profiled_alpha": 90.89,
    "hidden_dim": 4096
}
```

Router 端收到心跳后：
1. 更新 `AdapterAwareRouter.worker_states[worker_id]`（用于路由评分）
2. 更新 `WorkerStateCache`（用于健康监控，超时 300ms 标记不健康）
3. 更新 `ReplicaManager`（用于 EMA 平滑、replica_map 同步、top_k 更新）

### 请求转发

Router 收到请求后，通过评分函数选择 Worker，然后通过对应 Worker 的 ZMQ PUSH socket 发送请求消息（JSON 格式）。

### 控制消息

Router → Worker 的控制消息通过同一个 ZMQ PUSH socket 发送，Worker 端根据 `type` 字段区分：
- `type: "request"` — 正常推理请求
- `type: "preload_adapter"` — 预加载 Adapter 指令（热门 Adapter 复制）
- `type: "reset_cache"` — 重置 Adapter 缓存（实验间清理）

---

## 3. RWPT 的输入量来自哪里

### 能采到的量

| 输入量 | 来源 | 采集方式 | 用途 |
|--------|------|---------|------|
| `pending_prefill_tokens` (RWPT) | Worker 等待队列 | 实时遍历 `req_queue.waiting_req_list`，计算 `Σ input_len × (1 + γ × rank)` | 路由评分的负载项 |
| `pending_raw_tokens` | Worker 等待队列 | 实时遍历，计算 `Σ input_len`（无 rank 加权） | 消融实验的 token_count 变体 |
| `queue_length` | Worker 等待队列 | `len(req_queue.waiting_req_list)` | 消融实验的 queue_length 变体 |
| `active_decode_seqs` | Worker 当前 batch | `len(current_batch.reqs)` | 保留上报，当前公式不使用 |
| `pool_used_ratio` | Worker 内存池 | `1 - (can_use_mem_size / tot_size)` | 淘汰阈值判断 |
| `cached_adapters` | Worker InferAdapter | `set(infer_adapter.adapter_dirs)` | 路由缓存亲和性、replica_map |
| `rank` (per adapter) | Adapter 配置文件 | 启动时从 `adapter_config.json` 读取 | RWPT 的 γ×r 项 |
| `hidden_dim` | 模型配置文件 | 启动时从 `config.json` 的 `hidden_size` 读取 | 计算 γ = 2/(3d) |
| `profiled_alpha` | Worker 启动时 profiling | micro-benchmark 测量 prefill/decode 耗时比 | 保留，当前不使用 |
| `top_k_rwpt_adapters` | Worker 等待队列 | 按 adapter 聚合 RWPT 贡献，取 top-5 | ReplicaManager 元凶识别 |

### 不能直接采到的量

| 量 | 原因 | 替代方案 |
|----|------|---------|
| GPU 算力利用率 | CUDA 没有提供实时算力利用率 API | 用 RWPT/Capacity 间接估算 |
| 单请求延迟预测 | 需要 profiling 模型，开销大 | 用 RWPT 作为延迟的代理指标 |
| 网络带宽占用 | 单机场景不涉及 | 多机扩展时需要 |

### γ 的来源（非超参数）

γ = 2/(3×hidden_dim)，从 LoRA 的 FLOPs 增量理论推导：
- LoRA 对 Q/K/V/O 四个投影各加 4dr FLOPs，共 16dr
- Base Model 每层每 token 总 FLOPs ≈ 24d²
- 计算量增比 = 16dr / 24d² = 2r/(3d)

对 LLaMA-7B（d=4096）：γ ≈ 1.63×10⁻⁴。从模型 `config.json` 自动检测，无需手动指定。

---

## 4. 热点 Adapter 副本扩展机制的实现程度

### 已完全实现 ✅

**阶段 1：Worker 端元凶信息上报**
- `GPUWorker._compute_top_k_rwpt_adapters(k=5)`：遍历等待队列，按 adapter 聚合 RWPT 贡献，取 top-K
- 集成到心跳上报链路：`_get_state_for_reporter()` → `WorkerStateReporter._create_state_message()` → ZMQ 发送

**阶段 2：预加载 RPC 通道**
- `ModelRpcServer.exposed_preload_adapter(adapter_dir, preserve_sec)`：加载权重 + 加入保护列表
- `ModelRpcClient.preload_adapter()`：客户端接口
- `InferAdapter.add_protection() / remove_protection() / get_protected_count()`：副本保护列表管理
- `InferAdapter.select_eviction_candidates()`：保护期内的 adapter 不出现在淘汰候选中
- `GPUWorker.handle_preload_adapter()`：Worker 端预加载处理
- `DataParallelRouterManager.send_preload_to_worker()`：通过 ZMQ 发送预加载指令

**阶段 3：ReplicaManager 核心模块**
- `ReplicaManager.__init__()`：初始化状态字典、计算 T_congestion
- `update_worker_state()`：心跳回调，更新 EMA、replica_map、worker_top_k
- `update_config()`：w1/w2 变更时重算 T_congestion
- `_detect_congested_workers()`：EMA > w1/w2 判定拥塞
- `_identify_culprit()`：取 top_k[0] 作为元凶
- `_check_guardrails()`：副本数 < N_max 且冷却期已过
- `_select_target_worker()`：未缓存 + EMA < 阈值×0.9 + 保护数未满，选 EMA 最低
- `patrol()`：完整决策管线，每周期最多 1 个复制动作
- `_start_patrol_loop()`：asyncio 后台巡检线程
- `get_stats()`：运行统计查询
- 集成到 `DataParallelRouterManager`：心跳同步、巡检线程启动、配置热更新同步
- `--enable-replication` 命令行开关

**阶段 4：实验配置**
- `ExperimentConfig` 新增 `enable_replication`、`cooldown_sec`、`patrol_interval_sec`、`protection_sec`
- `replication-comparison` 实验 suite（2 策略 × 3 alpha = 6 组）

### 待完成 ⏳

- 端到端实验验证（`replication-comparison` suite 已配置好，尚未跑）
- 实验结果分析和图表生成

### 属性测试覆盖

58 个属性测试全部通过，覆盖 12 个正确性属性：
- Property 1-2：WorkerState 序列化、Top-K RWPT 聚合
- Property 3-4：EMA 计算、T_congestion 公式
- Property 5-6：拥塞检测阈值、元凶识别
- Property 7：全局防爆护栏
- Property 8：目标 Worker 选择
- Property 9：每周期单副本
- Property 10：副本保护阻止淘汰
- Property 11：心跳同步 replica_map
- Property 12：路由器自动发现新副本

---

## 5. 异构感知部分的现状

### 已有的基础设施

- Worker 启动时自动 profiling `decode_cost_alpha`（prefill/decode 耗时比），通过心跳上报给 Router
- Worker 启动时自动检测 `hidden_dim`，通过心跳上报
- Router 取所有 Worker 上报的 `profiled_alpha` 中位数作为全局值
- `RoutingConfig` 支持 per-worker 的 `batch_max_tokens`（当前所有 Worker 使用相同值）

### 尚未实现

- 没有 per-worker 的算力系数（所有 Worker 假设同构）
- 没有 per-worker 的 Capacity 和 T_congestion
- 副本放置不考虑卡的能力差异
- 淘汰阈值不按显存大小自适应

### 如果要做异构感知

可以复用现有的 profiling 机制：
1. Worker 启动时 profiling 自己的 prefill 速度和 decode 速度
2. 通过心跳上报给 Router
3. Router 为每个 Worker 计算独立的 Capacity 和算力系数
4. 路由评分函数中加入 per-worker 的归一化
5. ReplicaManager 的目标选择考虑卡的能力

---

## 6. 可视化和实验部分的技术栈

### 实验框架

- **实验配置**: `ExperimentConfig` dataclass，支持 Cartesian product 生成实验矩阵
- **实验执行**: `ExperimentRunner` 类，管理服务器生命周期、基准测试、结果收集
- **服务器管理**: 通过 `subprocess` 启动 `launch_server.py`，支持服务器复用（只在配置 key 变化时重启）
- **基准测试**: 异步 HTTP 客户端（`aiohttp`），按 Poisson 过程发送请求，测量 per-request 延迟
- **断点续传**: JSON checkpoint 文件，中断后可 `--resume` 继续

### 指标收集

- **路由统计**: 通过 HTTP API `/routing_stats` 从运行中的服务器拉取（cache_hit_rate、worker_distribution 等）
- **配置热更新**: 通过 HTTP API `/update_routing_config` 动态修改 w1/w2/w3/load_metric（无需重启服务器）
- **缓存重置**: 通过 HTTP API `/reset_adapter_cache` 在实验间清理缓存
- **统计文件**: Router 每 5 秒写入 `/tmp/slora_routing_stats.json`（原子写入）

### 结果存储

- 格式: JSONL（每行一个实验结果）
- 路径: `benchmarks/routing_comparison_results/<suite_name>/results.jsonl`
- 内容: 实验配置 + 延迟分布（P50/P90/P95/P99）+ 吞吐量 + 缓存命中率 + Worker 分布

### 图表生成

- 工具: Matplotlib
- 已有图表: P90 TTFT 分组柱状图、Cache Hit Rate 分组柱状图、Pareto 散点图
- 脚本: `benchmarks/routing_comparison_results/alpha-robustness-sla/plot_figures.py`

---

## 7. 请求从进入 Router 到被 Worker 执行的完整流程

```
1. 客户端发送 HTTP POST /generate_stream {lora_dir, inputs, parameters}
   ↓
2. api_server.py 接收请求，构建 SamplingParams
   ↓
3. HttpServerManager 通过 ZMQ PUSH 发送 (adapter_dir, prompt_ids, sampling_params, request_id)
   ↓
4. DataParallelRouterManager 通过 ZMQ PULL 接收请求
   ↓
5. 调用 AdapterAwareRouter.select_worker(adapter_dir)：
   - 检查是否热点 Adapter（请求率 > 阈值）
   - 如果是热点：Round-Robin 分散
   - 如果不是：计算每个健康 Worker 的评分
     Score = w1 × I(adapter ∈ Cache_i) - w2 × (RWPT_i / Capacity)
   - 选择评分最高的 Worker
   ↓
6. 通过该 Worker 的 ZMQ PUSH socket 发送请求
   ↓
7. Worker 的 GPUWorker._receive_requests_loop() 通过 ZMQ PULL 接收
   ↓
8. 请求加入 Worker 本地的 req_queue.waiting_req_list
   ↓
9. Worker 的 _step() 循环从 waiting_req_list 取请求组成 batch
   ↓
10. 调用 model_rpc.load_adapters() 加载所需 Adapter 权重到 GPU
    - 如果显存不足，触发淘汰（三维评分选出低分 Adapter 卸载）
    ↓
11. 调用 model_rpc.prefill_batch() 执行 prefill（生成第一个 token）
    ↓
12. 循环调用 model_rpc.decode_batch() 执行 decode（逐 token 生成）
    ↓
13. 每生成一个 token，通过 ZMQ PUSH 发送到 ResponseMerger
    ↓
14. ResponseMerger 转发到 Detokenization 进程
    ↓
15. Detokenization 将 token_id 转为文本，通过 ZMQ 发送到 HttpServerManager
    ↓
16. HttpServerManager 通过 HTTP SSE 流式返回给客户端
```

**并行发生的后台任务：**
- Worker 每 100ms 上报状态心跳 → Router 更新路由状态
- ReplicaManager 每 1s 巡检一次 → 检测拥塞 → 触发副本复制
- Router 每 10s 输出统计日志
- Router 每 5s 检查配置更新文件和缓存重置触发文件
