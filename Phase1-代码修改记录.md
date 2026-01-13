# Phase 1 数据并行基础框架 - 代码修改记录

本文档记录 Phase 1 实施过程中的所有代码修改。

## 修改记录

### 2025-01-12 - Task 1: 创建 Round Robin Router

**新增文件**:
1. `slora/server/router/round_robin_router.py`
   - 实现 `RoundRobinRouter` 类
   - 功能：轮询路由策略，按顺序循环分配请求到不同的 Worker
   - 特性：
     - 支持任意数量的 Worker
     - 线程安全（使用锁保护计数器）
     - 提供统计信息和重置功能
   - 核心方法：
     - `select_worker()`: 选择下一个 Worker
     - `reset()`: 重置计数器
     - `get_stats()`: 获取统计信息

---

### 2025-01-12 - Task 2.1: 创建 GPUWorker 类框架

**新增文件**:
1. `slora/server/router/gpu_worker.py`
   - 实现 `GPUWorker` 类的基础框架
   - 功能：在指定 GPU 上运行的独立推理实例
   - 核心方法：
     - `__init__()`: 初始化 Worker，接收 worker_id、gpu_id 和 args
     - `_setup_gpu()`: 设置 CUDA_VISIBLE_DEVICES 和 torch.cuda.set_device()

**实现细节**:
- 使用 `os.environ['CUDA_VISIBLE_DEVICES']` 限制可见 GPU
- 设置后，Worker 只能看到分配给它的 GPU（索引为 0）
- 添加了详细的日志输出和错误处理

---

### 2025-01-12 - Task 2.2: 实现 ZMQ 通信设置

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 添加 ZMQ 相关导入：`zmq`, `zmq.asyncio`
   - 实现 `_setup_zmq()` 方法
   - 功能：设置 Worker 的 ZMQ 通信
   - 核心方法：
     - `_setup_zmq(request_port, response_port)`: 设置 ZMQ 通信
       - 创建 PULL socket 并连接到 Router Manager
       - 创建 PUSH socket 并连接到 Response Merger

**实现细节**:
- 使用 `zmq.asyncio.Context()` 创建异步 context
- PULL socket 用于接收请求（多对一模式）
- PUSH socket 用于发送响应（一对一模式）
- 使用 `connect()` 而非 `bind()`，因为 Worker 是客户端

---

### 2025-01-12 - Task 2.3: 集成模型加载逻辑

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 添加模型相关导入：
     - `LlamaTpPartModel` (Llama 模型)
     - `Llama2TpPartModel` (Llama2 模型，支持 GQA)
     - `get_model_config` (模型配置工具)
   - 实现 `_load_model()` 方法
   - 功能：在指定 GPU 上加载完整的基座模型

**实现细节**:
- 复用现有的 `slora/common/basemodel/` 模型加载逻辑
- 数据并行模式参数：`tp_rank=0`, `world_size=1`（单 GPU 模式）
- 模型类型检测：通过 `num_key_value_heads` 判断是否为 Llama2（GQA）

---

### 2025-01-12 - Task 2.4: 实现请求处理循环

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 实现请求处理的核心方法：
     - `_receive_request()`: 接收请求
     - `_process_request()`: 处理请求
     - `_send_response()`: 发送响应
     - `run()`: 主循环
   - 功能：完整的请求-响应处理流程

**实现细节**:
- 从 ZMQ PULL socket 异步接收请求
- 解析请求消息（request_id, adapter_dir, prompt_ids, sampling_params）
- 执行推理（Phase 1 简化实现）
- 生成响应消息（包含 request_id, worker_id, output_ids, metadata, success, error）
- 通过 ZMQ PUSH socket 异步发送响应
- 无限循环，持续处理请求
- 完整的异常捕获和错误响应生成

**Phase 1 简化说明**:
- 推理逻辑使用占位实现（简单追加 token）
- Adapter 加载暂未实现（输出日志）
- 完整的推理逻辑将在后续任务中实现

---

### 2025-01-13 - Task 2.4: 集成 ReqQueue 请求管理

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 添加 ReqQueue 相关导入：`ReqQueue`, `Req`, `Batch`, `SamplingParams`
   - 添加请求队列管理属性：`req_queue`, `current_batch`, `lora_ranks`, `actual_adapter_size`
   - 实现新方法：
     - `_setup_request_queue()`: 初始化 ReqQueue
     - `_convert_to_req_object()`: 将 ZMQ 消息转换为 Req 对象
     - `_process_requests()`: 批处理请求（使用 ReqQueue）
   - 更新 `run()` 方法：支持批处理循环

**关键设计**:
- 复用张量并行的 ReqQueue：每个 GPU Worker 内部包含一个独立的 ReqQueue 实例
- 批处理管理：使用 ReqQueue 的 `generate_new_batch()` 方法生成批次
- 显存管理：ReqQueue 自动处理显存分配和 Adapter 调度

**实现细节**:
- 创建 ReqQueue 实例，配置 `max_total_tokens`, `batch_max_tokens`, `running_max_req_size`
- 将 ZMQ 消息格式转换为 Req 对象，解析采样参数
- 使用 ReqQueue 生成新批次，合并到当前批次，执行批量推理
- 非阻塞接收请求（使用 asyncio.wait_for 超时）

---

### 2025-01-13 - Task 2.7.1: 实现 Adapter Rank 配置

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 添加导入：`get_lora_config`
   - 实现 `_setup_adapter_config()` 方法
   - 功能：读取所有 adapter 的 rank 配置，用于 ReqQueue 显存管理

**关键设计**:
- 借鉴 manager.py：复用张量并行模式中的 Adapter rank 管理策略
- lora_ranks 字典：存储 adapter_dir -> rank 的映射关系
- None 键：表示无 adapter 的情况（base 模型），rank 为 0

**实现细节**:
- 初始化 `self.lora_ranks = {}` 字典
- 遍历所有 adapter 目录，调用 `get_lora_config()` 读取配置
- 提取 rank 值并存储到字典
- 错误处理：加载失败时使用默认 rank 值（8）
- 添加 `self.lora_ranks[None] = 0` 处理无 adapter 情况

---

### 2025-01-13 - Task 2.7.2: 实现实际内存占用跟踪

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 添加 `self.actual_adapter_memory_usage = 0` 属性
   - 实现 `_update_actual_adapter_usage()` 方法
   - 功能：通过 RPC 查询 LoRA 内存使用情况

**关键设计**:
- 借鉴 manager.py：复用张量并行模式中的实际内存占用跟踪策略
- actual_adapter_memory_usage：缓存实际的 adapter 内存占用（单位：cells）
- 保守估计：查询失败时保持当前值不变

**实现细节**:
- 检查是否启用了 LoRA 和是否有 model_rpc
- 通过 RPC 查询 `check_lora_memory()` 获取内存信息
- 计算所有 adapter_cells 的总和
- 同步更新 `actual_adapter_size`（用于 ReqQueue）
- 完整的错误处理

---

### 2025-01-13 - Task 2.7.3: 实现基本的 Adapter 加载/卸载

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 实现 `_load_adapters(adapter_dirs)` 方法
   - 更新 `_process_requests()` 方法：在生成新批次后加载所需的 adapters

**关键设计**:
- 借鉴 manager.py：复用张量并行模式中的 adapter 加载策略
- 加载时机：在 ReqQueue 生成新批次后，立即加载批次所需的 adapters
- 内存更新：加载后调用 `_update_actual_adapter_usage()` 更新实际占用

**实现细节**:
- 检查是否启用了 LoRA、是否有 model_rpc、adapter_dirs 是否为空
- 调用 RPC 的 `load_adapters(adapter_dirs)` 方法
- 调用 `_update_actual_adapter_usage()` 更新实际占用
- 错误处理：加载失败不终止服务，输出警告日志并继续运行

**Phase 1 简化说明**:
- 假设 `model_rpc` 已经初始化
- 暂不实现 adapter 卸载逻辑（Phase 1 只需要加载）

---

### 2025-01-13 - Task 2.7.4: 更新 ReqQueue 调用传递 Adapter 信息

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 更新 `_process_requests()` 方法：使用关键字参数传递 `actual_adapter_size`

**关键设计**:
- 显式参数传递：使用 `actual_adapter_size=self.actual_adapter_memory_usage` 作为关键字参数
- 与 manager.py 一致：保持与张量并行模式相同的调用方式

**实现细节**:
```python
new_batch = self.req_queue.generate_new_batch(
    self.current_batch,
    self.lora_ranks,
    actual_adapter_size=self.actual_adapter_memory_usage
)
```

**Adapter 管理完整流程**:
```
初始化
→ _setup_adapter_config() 读取 lora_ranks
→ actual_adapter_memory_usage = 0

批次处理
→ generate_new_batch(lora_ranks, actual_adapter_size)
→ ReqQueue 计算显存占用
→ 生成新批次
→ _load_adapters(batch.adapter_dirs)
→ RPC.load_adapters()
→ _update_actual_adapter_usage()
→ actual_adapter_memory_usage 更新
→ 执行推理
```

---

### 2025-01-13 - Task 3.1: 创建 DataParallelRouterManager 类框架

**新增文件**:
1. `slora/server/router/dp_manager.py`
   - 实现 `DataParallelRouterManager` 类框架
   - 功能：管理多个 GPU Worker 进程并路由请求
   - 核心方法：
     - `__init__()`: 初始化 Router Manager
     - `_detect_gpus()`: 自动检测可用 GPU 数量
     - `_parse_gpu_ids()`: 解析 GPU ID 列表
     - `_allocate_ports()`: 为每个 Worker 分配端口

**关键设计**:
- 多进程架构：每个 GPU Worker 运行在独立的进程中
- 灵活配置：支持自动检测 GPU 或手动指定
- 端口管理：为每个 Worker 分配唯一的通信端口（从 50000 开始递增）
- 路由集成：集成 RoundRobinRouter 进行请求分发

**实现细节**:
- 接收参数：args（包含 num_workers, gpu_ids 等）、router_port、response_port
- 初始化属性：num_workers、gpu_ids、workers、worker_ports、router
- 验证 GPU ID 数量与 Worker 数量匹配
- 使用 `torch.cuda.is_available()` 和 `torch.cuda.device_count()` 检测 GPU
- 解析逗号分隔的 GPU ID 字符串，验证有效性
- 完整的错误处理

---

### 2025-01-13 - Task 3.2: 实现 Worker 进程管理

**修改文件**:
1. `slora/server/router/dp_manager.py`
   - 添加 asyncio 导入
   - 实现 `start_workers()` 方法：启动所有 Worker 进程
   - 实现 `_start_worker()` 方法：启动单个 Worker 进程
   - 添加模块级函数 `run_gpu_worker_process()`: Worker 进程的入口函数

**关键设计**:
- 多进程架构：每个 Worker 运行在独立的进程中
- 进程管理：使用 multiprocessing.Process 创建和管理 Worker 进程
- 端口分配集成：在启动 Worker 前自动分配端口
- 就绪等待：Phase 1 使用固定等待时间（5秒），Phase 2 将实现心跳机制

**实现细节**:

1. **`start_workers()` 方法**（异步）:
   - 调用 `_allocate_ports()` 分配端口
   - 循环启动所有 Worker 进程
   - 等待 5 秒让 Worker 初始化
   - 输出启动日志

2. **`_start_worker()` 方法**:
   - 创建 multiprocessing.Process 实例
   - 目标函数：`run_gpu_worker_process`
   - 传递参数：worker_id, gpu_id, args, request_port, response_port
   - 设置进程名称：`GPUWorker-{worker_id}`
   - 启动进程并返回进程对象

3. **`run_gpu_worker_process()` 函数**（模块级）:
   - Worker 进程的入口函数
   - 创建 GPUWorker 实例
   - 设置 ZMQ 通信
   - Phase 1 简化：暂不加载模型和初始化请求队列
   - 运行 Worker 主循环（asyncio.run）
   - 完整的错误处理和日志输出

**进程启动流程**:
```
start_workers()
→ _allocate_ports() 分配端口
→ 循环调用 _start_worker(i, gpu_id)
  → 创建 Process(target=run_gpu_worker_process)
  → proc.start() 启动进程
  → 返回进程对象
→ asyncio.sleep(5) 等待就绪
→ 完成
```

**Phase 1 简化说明**:
- 使用固定等待时间（5秒）而非心跳机制
- Worker 进程中暂不加载模型和初始化请求队列
- 完整的模型加载和请求队列初始化将在后续任务中实现

---

### 2025-01-13 - Task 3.3: 实现 ZMQ 通信设置

**修改文件**:
1. `slora/server/router/dp_manager.py`
   - 添加 ZMQ 相关导入：`zmq`, `zmq.asyncio`
   - 实现 `_setup_zmq()` 方法
   - 功能：设置 Router Manager 的 ZMQ 通信

**关键设计**:
- **通信模式**：
  - API Server → Router Manager: PUSH/PULL
  - Router Manager → Workers: PUSH/PULL（每个 Worker 一个 PUSH socket）
- **Socket 类型**：
  - PULL socket：接收来自 API Server 的请求（多对一）
  - PUSH sockets：向每个 Worker 发送请求（一对多）

**实现细节**:

1. **`_setup_zmq()` 方法**:
   - 创建异步 ZMQ context：`zmq.asyncio.Context()`
   - 创建 PULL socket 并绑定到 router_port：
     - 用于接收来自 API Server 的请求
     - 使用 `bind()` 因为 Router Manager 是服务端
   - 为每个 Worker 创建 PUSH socket：
     - 绑定到对应的 worker_port
     - 用于向特定 Worker 发送请求
     - 使用 `bind()` 因为 Router Manager 是服务端
   - 输出详细的日志信息

2. **通信架构**:
```
API Server (PUSH)
    ↓
Router Manager (PULL) - router_port
    ↓
Router Manager (PUSH) - worker_ports[0..N]
    ↓
Workers (PULL) - 每个 Worker 监听自己的端口
```

3. **端口使用**:
   - `router_port`: Router Manager 接收 API Server 请求
   - `worker_ports[i]`: Router Manager 向 Worker i 发送请求
   - `response_port`: Workers 发送响应（将在 Response Merger 中使用）

**满足 Requirements**:
- Requirements 2.3（通过 ZMQ PUSH socket 发送请求消息）

---
### 2025-01-13 - Task 3.4: 实现请求路由逻辑

**修改文件**:
1. `slora/server/router/dp_manager.py`
   - 实现 `route_request()` 方法：路由请求到 Worker
   - 实现 `run()` 方法：主循环，持续接收和路由请求

**新增文件**:
1. `test/test_dp_manager_routing.py`
   - 测试 `route_request()` 方法的正确性
   - 测试 `run()` 主循环的功能
   - 测试轮询路由的公平性
   - 测试错误处理机制

**关键设计**:
- **路由策略**：使用 Round Robin Router 选择 Worker
- **异步通信**：使用 ZMQ 异步 API 发送请求
- **错误处理**：发送失败时记录错误日志并抛出异常
- **持续运行**：主循环无限运行，直到进程被终止

**实现细节**:

1. **`route_request()` 方法**（异步）:
   - 使用 `self.router.select_worker()` 选择 Worker
   - DEBUG 级别记录路由决策（Requirement 8.4）
   - 通过 `self.request_senders[worker_id].send_json(request)` 发送请求
   - 错误处理：捕获异常，记录错误日志（Requirement 2.5）
   - 抛出异常以便上层处理

2. **`run()` 方法**（异步）:
   - 输出启动日志，显示监听端口
   - 无限循环：
     - 从 `self.request_receiver.recv_json()` 接收请求
     - 调用 `await self.route_request(request)` 路由请求
   - 错误处理：
     - 捕获所有异常
     - 记录错误日志和堆栈跟踪
     - 继续处理下一个请求（不终止循环）

3. **路由流程**:
```
API Server 发送请求
    ↓
Router Manager.run() 接收请求
    ↓
Router Manager.route_request()
    ↓
Round Robin Router.select_worker() 选择 Worker
    ↓
ZMQ PUSH socket 发送请求到 Worker
    ↓
Worker 接收并处理请求
```

**测试覆盖**:

1. **`test_route_request_selects_worker`**:
   - 验证 route_request 能够选择 Worker
   - 验证只有一个 sender 被调用

2. **`test_route_request_round_robin_order`**:
   - 验证轮询顺序正确（0, 1, 2, 0, 1, 2）
   - 验证每个 Worker 收到的请求数量相等
   - **满足 Requirements 2.1, 2.2**

3. **`test_route_request_sends_correct_message`**:
   - 验证发送的消息包含所有必需字段
   - 验证消息内容与原始请求一致
   - **满足 Requirements 2.3, 2.4**

4. **`test_route_request_handles_error`**:
   - 验证发送失败时抛出异常
   - 验证错误日志被记录
   - **满足 Requirement 2.5**

5. **`test_run_receives_and_routes_requests`**:
   - 验证 run() 能够持续接收请求
   - 验证所有请求都被路由
   - **满足 Requirements 2.1, 2.3**

6. **`test_run_continues_on_error`**:
   - 验证单个请求失败不会终止主循环
   - 验证错误被记录但循环继续
   - **满足 Requirement 2.5**

**满足 Requirements**:
- Requirements 2.1（使用 Round Robin Router 选择下一个 Worker）
- Requirements 2.2（按照 Worker ID 的顺序循环分配请求）
- Requirements 2.3（通过 ZMQ PUSH socket 发送请求消息）
- Requirements 2.5（发送请求失败时记录错误日志）
- Requirements 8.4（DEBUG 级别记录路由决策）

**测试结果**:
- 所有 6 个测试用例通过
- 测试覆盖：路由选择、轮询顺序、消息格式、错误处理、主循环

---
### 2025-01-13 - Task 3.5: 实现主循环

**说明**:
Task 3.5 的实现已经在 Task 3.4 中完成。`run()` 方法在实现请求路由逻辑时一并实现，因为两者紧密相关。

**已实现功能**:
1. ✓ 持续接收来自 API Server 的请求
2. ✓ 调用路由器选择 Worker
3. ✓ 发送请求到选定的 Worker

**实现细节**（参见 Task 3.4）:
- `run()` 方法是一个异步无限循环
- 从 `self.request_receiver.recv_json()` 接收请求
- 调用 `await self.route_request(request)` 路由请求
- 完整的错误处理：捕获异常但继续运行
- 输出启动日志和错误日志

**测试覆盖**（参见 Task 3.4）:
- `test_run_receives_and_routes_requests`: 验证持续接收和路由功能
- `test_run_continues_on_error`: 验证错误恢复机制

**满足 Requirements**:
- Requirements 2.1（新请求到达时使用 Round Robin Router 选择 Worker）
- Requirements 2.3（通过 ZMQ 发送请求到选定的 Worker）

**测试结果**:
- 2/2 测试用例通过
- 验证了主循环的持续运行能力
- 验证了错误处理和恢复机制

---

### 2025-01-13 - Task 2.9.1: 实现模型 RPC 初始化

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 添加导入：`start_model_process`, `ModelRpcClient`, `InputParams`
   - 添加 `self.model_rpc = None` 属性
   - 实现 `_init_model_rpc()` 方法：初始化模型 RPC 连接
   - 更新 `_update_actual_adapter_usage()` 和 `_load_adapters()` 方法：简化 model_rpc 检查

2. `slora/server/router/dp_manager.py`
   - 更新 `run_gpu_worker_process()` 函数：
     - 添加 `worker._setup_request_queue()` 调用
     - 添加 `asyncio.run(worker._init_model_rpc())` 调用
     - 移除注释，启用完整的 Worker 初始化流程

**关键设计**:
- **RPC 架构**：使用 ModelRpcClient 连接到模型进程
- **单 GPU 模式**：world_size=1，不启动额外的 RPC 进程
- **参数传递**：创建 InputParams 对象传递给 init_model
- **初始化顺序**：ZMQ → ReqQueue → Model RPC → 主循环

**实现细节**:

1. **`_init_model_rpc()` 方法**（异步）:
   - 调用 `start_model_process(port=None, world_size=1)` 创建 ModelRpcClient
   - 单 GPU 模式下直接返回本地 ModelRpcServer 实例
   - 创建 InputParams 对象，从 args 中提取所有必需参数：
     - 基础参数：max_req_total_len, max_total_token_num, batch_max_tokens 等
     - LoRA 参数：pool_size_lora, no_lora, no_lora_compute 等
     - 调度参数：scheduler, prefetch, swap 等
     - 淘汰参数：evict_interval_threshold, evict_idle_threshold 等
   - 调用 `model_rpc.init_model()` 初始化模型：
     - rank_id=0, world_size=1（单 GPU 模式）
     - 传递 weight_dir, adapter_dirs, max_total_token_num
     - 传递 load_way, mode, input_params
     - prefetch_stream=None（数据并行不使用 prefetch）
   - 完整的错误处理和日志输出

2. **Worker 初始化流程**（在 `run_gpu_worker_process` 中）:
```
创建 GPUWorker 实例
    ↓
_setup_zmq() - 设置 ZMQ 通信
    ↓
_setup_request_queue() - 初始化 ReqQueue
    ↓
_init_model_rpc() - 初始化模型 RPC（异步）
    ↓ start_model_process(world_size=1)
    ↓ 创建 ModelRpcClient（本地 ModelRpcServer）
    ↓ 创建 InputParams 对象
    ↓ model_rpc.init_model() - 加载模型权重
    ↓
run() - 启动主循环（异步）
```

3. **InputParams 参数映射**:
   - 从 args 中提取参数，使用 getattr 提供默认值
   - 所有参数都有合理的默认值，确保兼容性
   - 关键参数：
     - max_total_token_num: KV cache 总大小
     - batch_max_tokens: 批次最大 token 数
     - running_max_req_size: 批次最大请求数
     - no_lora: 是否禁用 LoRA
     - evict_*_threshold/ratio: 淘汰策略参数

4. **model_rpc 检查简化**:
   - 从 `hasattr(self, 'model_rpc') or self.model_rpc is None` 简化为 `self.model_rpc is None`
   - 因为 `__init__` 中已经初始化 `self.model_rpc = None`

**满足 Requirements**:
- Requirements 1.4（在指定 GPU 上加载模型）
- Requirements 3.4（使用现有的模型推理逻辑处理请求）

**Phase 1 实现说明**:
- 使用 RPC 方式加载模型，而不是直接加载（Task 2.3 的 `_load_model()` 方法）
- 单 GPU 模式下，ModelRpcClient 直接包装本地 ModelRpcServer，不启动额外进程
- 完整的参数传递，确保模型正确初始化
- 为后续的推理逻辑（Task 2.9.2）奠定基础

**下一步**:
- Task 2.9.2: 实现实际推理逻辑（调用 model_rpc 执行推理）
- Task 2.9.3: 集成完整的批次管理（处理 EOS token，更新批次状态）

---
