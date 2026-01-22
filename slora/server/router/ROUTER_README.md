# Router 目录说明

本目录包含 S-LoRA 的请求调度和管理核心代码，支持**张量并行**和**数据并行**两种模式。

## 📁 文件结构

```
router/
├── manager.py                  # 张量并行调度管理器（原有）
├── dp_manager.py               # 数据并行路由管理器（新增）⭐
├── gpu_worker.py               # GPU Worker 进程（新增）⭐
├── round_robin_router.py       # 轮询路由器（新增）⭐
├── response_merger.py          # 响应合并器（新增）⭐
├── req_queue.py                # 基础请求队列（两种模式共用）
├── vtc_req_queue.py            # VTC 公平调度器
├── pets_req_queue.py           # PETS 预测调度器
├── peft_req_queue.py           # PEFT 基线调度器
├── cluster_req_queue.py        # 集群调度器
├── abort_req_queue.py          # 支持请求中止的调度器
├── stats.py                    # 性能统计工具
├── profiler.py                 # 性能分析器（用于 PETS）
└── model_infer/                # 模型推理相关
    ├── model_rpc.py            # 模型 RPC 客户端
    └── infer_adapter.py        # Adapter 推理管理
```

## 🔀 并行模式对比

S-LoRA 支持两种并行模式，根据硬件配置和性能需求选择：

| 特性 | 张量并行 (Tensor Parallel) | 数据并行 (Data Parallel) ⭐ |
|------|---------------------------|---------------------------|
| **核心管理器** | `manager.py` | `dp_manager.py` |
| **模型加载** | 模型切分到多个 GPU | 每个 GPU 加载完整模型 |
| **请求处理** | 所有 GPU 协同处理每个请求 | 每个 GPU 独立处理请求 |
| **吞吐量** | 中等 (1x) | 高 (2.8x ~ 3x) |
| **延迟** | 低 | 中等 |
| **显存需求** | 低（模型切分） | 高（每 GPU 一份模型） |
| **适用场景** | 大模型 + 低延迟 | 小模型 + 高吞吐量 |
| **GPU 通信** | 频繁（协同推理） | 无（独立推理） |
| **扩展性** | 受限于模型大小 | 易于扩展 |


## 🔧 核心文件详解

---

## 📦 张量并行模式（Tensor Parallel）

### 1. `manager.py` - 张量并行调度管理器

**功能**：张量并行模式的核心控制器

**核心类**：
- `RouterManager`: 管理整个请求调度流程

**主要职责**：
- 接收来自 HTTP 服务器的请求
- 管理全局请求队列和批次生成
- 协调多个 GPU 协同推理
- 处理 Prefill 和 Decode 阶段
- 管理 LoRA 适配器的加载/卸载

**关键方法**：
- `add_req()`: 添加新请求到队列
- `loop_for_fwd()`: 主事件循环
- `_step()`: 执行单步调度逻辑
- `abort()`: 中止指定请求

**调度器选择**：
```python
get_scheduler()  # 根据配置选择不同的调度器
```

**架构特点**：
```
RouterManager (全局管理器)
    ↓
ReqQueue (全局队列)
    ↓
MemoryManager (全局显存)
    ↓
┌──────┬──────┬──────┬──────┐
│GPU 0 │GPU 1 │GPU 2 │GPU 3 │
│(模型切分，协同处理每个请求)│
└──────┴──────┴──────┴──────┘
```

---

## 📦 数据并行模式（Data Parallel）⭐ 新增

### 1. `dp_manager.py` - 数据并行路由管理器

**功能**：数据并行模式的中央协调器

**核心类**：
- `DataParallelRouterManager`: 管理多个 GPU Worker 和请求路由

**主要职责**：
- 🖥️ **GPU 资源管理**：自动检测 GPU，分配 GPU ID
- 👷 **Worker 进程管理**：启动、监控、健康检查
- 🚦 **请求路由**：Round Robin 轮询分发请求
- 📡 **ZMQ 通信管理**：管理多个 socket，消息转发
- 💊 **健康检查**：定期检测 Worker 进程状态
- 📊 **统计监控**：吞吐量、负载分布、性能指标

**关键方法**：
- `_detect_gpus()`: 自动检测可用 GPU
- `_parse_gpu_ids()`: 解析 GPU ID 列表
- `start_workers()`: 启动所有 Worker 进程
- `route_request()`: 路由请求到 Worker
- `_check_worker_health()`: 健康检查后台任务
- `_print_statistics()`: 统计输出后台任务

**架构特点**：
```
DataParallelRouterManager (轻量级路由器)
    ↓ Round Robin
┌────────────┬────────────┬────────────┐
│ Worker 0   │ Worker 1   │ Worker 2   │
│ GPU 0      │ GPU 1      │ GPU 2      │
│            │            │            │
│ ReqQueue   │ ReqQueue   │ ReqQueue   │
│ MemManager │ MemManager │ MemManager │
│ 完整模型   │ 完整模型   │ 完整模型   │
│ 独立处理   │ 独立处理   │ 独立处理   │
└────────────┴────────────┴────────────┘
```

**不负责**：
- ❌ 请求队列管理（由 Worker 的 ReqQueue 负责）
- ❌ 显存管理（由 Worker 的 MemoryManager 负责）
- ❌ 模型推理（由 Worker 的 model_rpc 负责）
- ❌ Adapter 管理（由 Worker 的 InferAdapter 负责）

**启动示例**：
```bash
# 自动检测 GPU
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode data \
    --num-workers 4

# 指定 GPU
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode data \
    --num-workers 4 \
    --gpu-ids 0,1,2,3
```

---

### 2. `gpu_worker.py` - GPU Worker 进程

**功能**：数据并行模式下的独立推理单元

**核心类**：
- `GPUWorker`: 在指定 GPU 上处理推理请求

**主要职责**：
- 🖥️ **GPU 环境设置**：设置 CUDA_VISIBLE_DEVICES
- 📦 **模型加载**：加载完整的基座模型
- 📋 **请求队列管理**：复用 ReqQueue 进行批处理
- 🔧 **Adapter 管理**：加载/卸载 LoRA 适配器
- 💾 **显存管理**：独立的 MemoryManager 实例
- 🔄 **推理执行**：Prefill + Decode

**关键方法**：
- `_setup_gpu()`: 设置 GPU 环境
- `_setup_adapter_config()`: 初始化 Adapter 配置
- `_init_model_rpc()`: 初始化模型 RPC
- `_setup_request_queue()`: 初始化请求队列
- `_load_adapters()`: 加载 Adapters
- `_update_actual_adapter_usage()`: 更新实际显存占用
- `_process_requests()`: 处理请求批次（核心）
- `_infer_batch()`: 执行批次推理
- `run()`: 主循环

**显存管理机制**：
```python
# 每个 Worker 独立管理显存
Worker:
  ├─ ReqQueue (批处理逻辑)
  │   └─ generate_new_batch() 检查显存
  ├─ MemoryManager (统一内存池)
  │   ├─ KV Cache 区域
  │   └─ LoRA Adapter 区域
  └─ InferAdapter (Adapter 管理)
      └─ get_lora_memory_usage() 查询占用
```

**工作流程**：
```
1. 接收 ZMQ 请求
   ↓
2. 添加到 ReqQueue
   ↓
3. generate_new_batch() 生成批次
   ↓
4. 加载所需 Adapters
   ↓
5. 执行推理 (Prefill/Decode)
   ↓
6. 发送 ZMQ 响应
```

**设计理念**：
> 每个 Worker = 一个独立的"单 GPU 张量并行系统"

---

### 3. `round_robin_router.py` - 轮询路由器

**功能**：实现 Round Robin 请求分发策略

**核心类**：
- `RoundRobinRouter`: 轮询路由器

**调度策略**：
```
请求 1 → Worker 0
请求 2 → Worker 1
请求 3 → Worker 2
请求 4 → Worker 0  (循环)
请求 5 → Worker 1
...
```

**关键方法**：
- `select_worker()`: 选择下一个 Worker

**特点**：
- ✅ 简单高效
- ✅ 负载均衡
- ✅ 无状态（不考虑 Worker 负载）

**使用场景**：
- 数据并行模式的默认路由策略
- Worker 性能相近时效果最好

---

### 4. `response_merger.py` - 响应合并器

**功能**：收集所有 Worker 的响应并转发到 Detokenization

**核心类**：
- `ResponseMerger`: 响应合并器

**主要职责**：
- 📥 **接收响应**：从所有 Worker 接收 ZMQ 响应
- 🔄 **格式转换**：转换为 Detokenization 所需格式
- 📤 **转发响应**：发送到 Detokenization 进程

**通信拓扑**：
```
Worker 0 ──PUSH──┐
Worker 1 ──PUSH──┤
Worker 2 ──PUSH──┼──→ ResponseMerger (PULL) ──PUSH──→ Detokenization
Worker 3 ──PUSH──┘
```

**关键方法**：
- `run()`: 主循环，持续接收和转发响应

**特点**：
- ✅ 独立进程运行
- ✅ 异步处理
- ✅ 无状态转发

---

## 📦 共用组件（两种模式都使用）

**功能**：S-LoRA 的默认 FIFO 调度器

**核心类**：
- `ReqQueue`: 基础请求队列类

**调度策略**：
- **FIFO（先进先出）**：按请求到达顺序处理
- **内存感知**：检查 token 和适配器内存限制
- **批处理优化**：尽可能将请求打包成批次

**关键方法**：
- `generate_new_batch()`: 生成新的批处理批次
- `_can_add_new_req()`: 判断是否可以添加新请求
- `next_batch()`: 获取下一个预取批次

**使用场景**：
- 默认调度器（`--scheduler slora`）
- 其他调度器的基类

---


### 1. `req_queue.py` - 基础请求队列

**功能**：S-LoRA 的默认 FIFO 调度器（**两种模式共用**）

**核心类**：
- `ReqQueue`: 基础请求队列类

**调度策略**：
- **FIFO（先进先出）**：按请求到达顺序处理
- **内存感知**：检查 token 和适配器内存限制
- **批处理优化**：尽可能将请求打包成批次

**关键方法**：
- `generate_new_batch()`: 生成新的批处理批次
- `_can_add_new_req()`: 判断是否可以添加新请求
- `next_batch()`: 获取下一个预取批次

**显存计算公式**：
```python
可用空间 = max_total_tokens - 实际 LoRA 占用 - 当前批次新增 LoRA
需要空间 = max(所有请求的 KV Cache 并发需求)

能否添加 = 需要空间 < 可用空间
```

**使用场景**：
- **张量并行**：全局唯一的 ReqQueue 实例
- **数据并行**：每个 Worker 独立的 ReqQueue 实例

**代码复用**：
```python
# 张量并行 (manager.py)
self.req_queue = get_scheduler(input_params, adapter_dirs)

# 数据并行 (gpu_worker.py)
self.req_queue = ReqQueue(
    max_total_tokens=self.args.max_total_token_num,
    batch_max_tokens=self.args.batch_max_tokens,
    running_max_req_size=self.args.running_max_req_size
)
```

---

### 2. `vtc_req_queue.py` - VTC 公平调度器

**功能**：实现 VTC（Virtual Token Counter）公平调度算法

**核心类**：
- `VTCReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **公平性保证**：确保不同适配器/用户获得公平的服务
- **虚拟令牌计数**：使用虚拟令牌计数器跟踪服务量
- **权重支持**：支持为不同适配器设置权重

**关键特性**：
- 维护每个适配器的服务计数器
- 优先服务服务量较少的适配器
- 支持公平权重配置

**使用场景**：
- 需要公平性的多租户场景
- 不同适配器有不同优先级需求

**启用方式**：
```bash
--scheduler vtc_fair --fair-weights <权重列表>
```

---

### 3. `pets_req_queue.py` - PETS 预测调度器

**功能**：实现 PETS（Predictive Early Termination Scheduling）调度算法

**核心类**：
- `PETSReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **预测性调度**：基于性能模型预测请求延迟
- **任务内批处理**：将相同适配器的请求聚类批处理
- **动态规划优化**：使用 DP 算法优化批次组合

**关键特性**：
- 使用 `AlphaModel` 和 `BetaModel` 预测延迟
- 任务内批处理（intra-task batching）
- 任务间批处理（inter-task batching）

**使用场景**：
- 需要优化延迟的场景
- 有性能分析数据的场景

**启用方式**：
```bash
--scheduler pets --profile
```

---

### 4. `peft_req_queue.py` - PEFT 基线调度器

**功能**：用于 HuggingFace PEFT 基线对比

**核心类**：
- `PEFTReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **单适配器批处理**：每个批次只包含同一适配器的请求
- **适配器切换**：批次之间切换适配器权重

**使用场景**：
- 与 HuggingFace PEFT 进行性能对比
- 测试单适配器批处理性能

**启用方式**：
```bash
--scheduler peft
```

---

### 5. `cluster_req_queue.py` - 集群调度器

**功能**：限制批次中的适配器数量

**核心类**：
- `ClusterReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **适配器数量限制**：限制单个批次中的适配器数量
- **优先级策略**：优先使用当前批次已有的适配器

**关键特性**：
- 当批次适配器数量达到上限时，优先添加使用相同适配器的请求
- 减少适配器切换开销

**使用场景**：
- 需要控制批次复杂度的场景
- 减少适配器加载/卸载频率

**启用方式**：
```bash
--batch-num-adapters <数量>
```

---

### 6. `abort_req_queue.py` - 支持中止的调度器

**功能**：支持请求中止功能的调度器

**核心类**：
- `AbortReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **请求中止**：自动中止超时请求
- **动态批处理大小**：根据请求速率调整批处理大小
- **LIFO/FIFO 切换**：根据负载情况切换调度顺序

**关键特性**：
- 跟踪请求时间戳
- 使用 attainment function 判断请求是否超时
- 自适应批处理大小

**使用场景**：
- 需要支持请求取消的场景
- 高负载下的动态调度

**启用方式**：
```bash
--enable-abort
```

---

### 7. `stats.py` - 性能统计工具

**功能**：收集和输出性能统计信息

**核心类**：
- `Stats`: 性能统计类

**统计指标**：
- 总 token 吞吐量（prompt + generate）
- Prompt token 吞吐量
- Generate token 吞吐量

**使用方式**：
- 自动集成到 `RouterManager` 中
- 定期输出统计信息到控制台

---

### 8. `profiler.py` - 性能分析器

**功能**：为 PETS 调度器提供性能预测模型

**核心类**：
- `AlphaModel`: 基础模型延迟预测
- `BetaModel`: LoRA 适配器延迟预测

**功能**：
- 加载性能分析结果（`.pkl` 文件）
- 根据批次大小和序列长度预测延迟
- 支持插值和外推

**使用场景**：
- PETS 调度器需要性能预测时
- 性能分析和优化

---


## 🔄 调度流程对比

### 张量并行流程

```
1. HTTP 服务器接收请求
   ↓
2. RouterManager.add_req() 添加到全局队列
   ↓
3. RouterManager.loop_for_fwd() 主循环
   ↓
4. RouterManager._step() 执行调度
   ↓
5. ReqQueue.generate_new_batch() 生成批次
   ↓
6. 检查全局内存/适配器限制
   ↓
7. 加载适配器 → Prefill → Decode（多 GPU 协同）
   ↓
8. 返回结果并更新统计
```

### 数据并行流程 ⭐

```
1. HTTP 服务器接收请求
   ↓
2. DataParallelRouterManager 接收请求
   ↓
3. RoundRobinRouter 选择 Worker
   ↓
4. 通过 ZMQ 发送到选定的 Worker
   ↓
5. GPUWorker 接收请求并添加到本地 ReqQueue
   ↓
6. ReqQueue.generate_new_batch() 生成批次
   ↓
7. 检查本地内存/适配器限制
   ↓
8. 加载适配器 → Prefill → Decode（单 GPU 独立）
   ↓
9. 通过 ZMQ 发送响应到 ResponseMerger
   ↓
10. ResponseMerger 转发到 Detokenization
   ↓
11. 返回结果给用户
```

## 📊 调度器对比

| 调度器 | 策略 | 特点 | 适用场景 |
|--------|------|------|----------|
| **ReqQueue** | FIFO | 简单高效，默认选择 | 通用场景 |
| **VTCReqQueue** | 公平调度 | 保证公平性，支持权重 | 多租户场景 |
| **PETSReqQueue** | 预测调度 | 优化延迟，需要性能分析 | 延迟敏感场景 |
| **PEFTReqQueue** | 单适配器批处理 | 与 PEFT 对比 | 基线测试 |
| **ClusterReqQueue** | 适配器限制 | 控制批次复杂度 | 减少切换开销 |
| **AbortReqQueue** | 支持中止 | 动态调整，自动超时 | 需要请求取消 |


## 🎯 如何选择并行模式和调度器

### 选择并行模式

#### 使用张量并行（Tensor Parallel）
```bash
# 适用场景：
# - 模型太大，单张 GPU 放不下（如 Llama-70B）
# - 追求低延迟
# - GPU 数量有限

python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode tensor \
    --tp 4
```

#### 使用数据并行（Data Parallel）⭐
```bash
# 适用场景：
# - 模型能放入单张 GPU（如 Llama-7B）
# - 有多张 GPU 可用
# - 需要处理大量并发请求
# - 追求高吞吐量

python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode data \
    --num-workers 4 \
    --gpu-ids 0,1,2,3
```

### 选择调度器（适用于两种模式）

#### 默认场景
```bash
# 使用默认 FIFO 调度器（张量并行）
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode tensor \
    --tp 4

# 使用默认 FIFO 调度器（数据并行）
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode data \
    --num-workers 4
```

### 公平性需求
```bash
# 使用 VTC 公平调度器（张量并行）
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode tensor \
    --tp 4 \
    --scheduler vtc_fair \
    --fair-weights 1 2 3

# 使用 VTC 公平调度器（数据并行）
# 注意：数据并行模式下，每个 Worker 独立使用调度器
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode data \
    --num-workers 4 \
    --scheduler vtc_fair \
    --fair-weights 1 2 3
```

### 延迟优化
```bash
# 使用 PETS 预测调度器（需要先进行性能分析）
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode tensor \
    --tp 4 \
    --scheduler pets \
    --profile
```

### 控制适配器数量
```bash
# 限制批次中最多 10 个适配器
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode data \
    --num-workers 4 \
    --batch-num-adapters 10
```

### 支持请求中止
```bash
# 启用请求中止功能
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --parallel-mode data \
    --num-workers 4 \
    --enable-abort
```

## 🔍 代码复用说明

### ReqQueue 的复用

**张量并行**：
```python
# manager.py
class RouterManager:
    def __init__(self, ...):
        # 全局唯一的 ReqQueue
        self.req_queue = get_scheduler(input_params, adapter_dirs)
```

**数据并行**：
```python
# gpu_worker.py
class GPUWorker:
    def _setup_request_queue(self):
        # 每个 Worker 独立的 ReqQueue
        self.req_queue = ReqQueue(
            max_total_tokens=self.args.max_total_token_num,
            batch_max_tokens=self.args.batch_max_tokens,
            running_max_req_size=self.args.running_max_req_size
        )
```

**关键区别**：
- 张量并行：1 个 ReqQueue 实例（全局）
- 数据并行：N 个 ReqQueue 实例（每个 Worker 一个）
- 代码逻辑：完全相同

### MemoryManager 的复用

**张量并行**：
```python
# models/llama/model.py
class LlamaTpPartModel:
    def __init__(self, ...):
        # 全局唯一的 MemoryManager
        self.mem_manager = MemoryManager(
            size=max_total_token_num,
            dtype=torch.float16,
            ...
        )
```

**数据并行**：
```python
# gpu_worker.py
class GPUWorker:
    def _init_model_rpc(self):
        # 每个 Worker 的模型独立创建 MemoryManager
        # 通过 model_rpc.init_model() 初始化
        # self.model.mem_manager 就是独立的 MemoryManager
```

**关键区别**：
- 张量并行：1 个 MemoryManager 实例（全局）
- 数据并行：N 个 MemoryManager 实例（每个 Worker 一个）
- 代码逻辑：完全相同

## 🏗️ 架构对比图

### 张量并行架构

```
┌─────────────────────────────────────────────────────────┐
│                    API Server                            │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│            RouterManager (全局管理器)                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ReqQueue (全局唯一)                             │   │
│  │  MemoryManager (全局唯一)                        │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
    ┌──────┐    ┌──────┐    ┌──────┐
    │GPU 0 │    │GPU 1 │    │GPU 2 │
    │(模型切分，协同处理每个请求)│
    └──────┘    └──────┘    └──────┘
```

### 数据并行架构 ⭐

```
┌─────────────────────────────────────────────────────────┐
│                    API Server                            │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│   DataParallelRouterManager (轻量级路由器)                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  RoundRobinRouter (请求分发)                     │   │
│  │  ZMQ 通信管理                                    │   │
│  │  Worker 进程管理                                 │   │
│  │  健康检查 + 统计监控                             │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     ↓ Round Robin
        ┌────────────┼────────────┐
        ↓            ↓            ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ GPU Worker 0 │ │ GPU Worker 1 │ │ GPU Worker 2 │
│   (GPU 0)    │ │   (GPU 1)    │ │   (GPU 2)    │
│              │ │              │ │              │
│ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │ReqQueue  │ │ │ │ReqQueue  │ │ │ │ReqQueue  │ │
│ │(独立)    │ │ │ │(独立)    │ │ │ │(独立)    │ │
│ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │
│              │ │              │ │              │
│ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │MemManager│ │ │ │MemManager│ │ │ │MemManager│ │
│ │(独立)    │ │ │ │(独立)    │ │ │ │(独立)    │ │
│ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │
│              │ │              │ │              │
│  完整模型    │ │  完整模型    │ │  完整模型    │
│  独立处理    │ │  独立处理    │ │  独立处理    │
└──────────────┘ └──────────────┘ └──────────────┘
        ↓            ↓            ↓
        └────────────┼────────────┘
                     ↓
        ┌────────────────────────┐
        │   ResponseMerger       │
        │   (响应收集与转发)      │
        └────────────────────────┘
                     ↓
        ┌────────────────────────┐
        │   Detokenization       │
        └────────────────────────┘
```

## 📝 关键数据结构

位置：`slora/server/io_struct.py`

- `Req`: 单个请求对象
- `Batch`: 请求批次对象
- `BatchAbortReq`: 中止请求批次

## 🔗 相关文件

- `../api_server.py`: API 服务器入口
- `../io_struct.py`: 数据结构定义
- `../sampling_params.py`: 采样参数
- `model_infer/model_rpc.py`: 模型推理 RPC 客户端

