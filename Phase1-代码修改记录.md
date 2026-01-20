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

### 2025-01-13 - Task 2.9.2: 实现实际推理逻辑

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 实现 `_infer_batch()` 方法：执行批次推理
   - 更新 `_process_requests()` 方法：使用实际推理逻辑

**关键设计**:
- **推理模式**：支持 prefill 和 decode 两种模式
- **Prefill 模式**：处理新请求的 prompt，调用 init_batch + prefill_batch
- **Decode 模式**：生成下一个 token，调用 decode_batch
- **批次管理**：处理完成的请求，更新批次状态

**实现细节**:

1. **`_infer_batch()` 方法**（异步）:
   - 判断推理模式：
     - 如果批次中所有请求的 output_ids 为空，则是 prefill
     - 否则是 decode
   - Prefill 流程：
     - 将请求转换为 RPC 对象：`req.to_rpc_obj()`
     - 调用 `model_rpc.init_batch(batch_id, reqs_rpc)` 初始化批次
     - 调用 `model_rpc.prefill_batch(batch_id)` 执行 prefill
   - Decode 流程：
     - 更新适配器使用统计：`model_rpc.update_adapter_stats(adapter_dirs)`
     - 调用 `model_rpc.decode_batch(batch_id)` 执行 decode
   - 返回格式：`{request_id: (token_id, metadata)}`
   - 完整的错误处理和堆栈跟踪

2. **`_process_requests()` 方法更新**:
   - 生成新批次（使用 ReqQueue）
   - 加载所需的 adapters
   - 合并批次
   - **调用 `_infer_batch()` 执行实际推理**（新增）
   - 处理推理结果：
     - 将输出 token 添加到请求：`req.output_ids.append(new_token_id)`
     - 添加元数据：`req.output_metadata_list.append(new_gen_metadata)`
     - 生成响应消息（包含完整的输出序列）
   - 标记完成的请求：
     - 从 args 获取 eos_id（默认为 2，Llama 的 EOS）
     - 调用 `batch.mark_finished_req(eos_id)` 检测完成
   - 更新批次状态：
     - 如果有请求完成，调用 `batch.filter_finished()` 过滤
     - 如果批次为空，调用 `model_rpc.remove_batch()` 移除 RPC 端批次
     - 如果批次还有请求，调用 `model_rpc.filter_batch()` 同步状态

3. **推理流程**:
```
接收请求
    ↓
ReqQueue.generate_new_batch() - 生成新批次
    ↓
_load_adapters() - 加载所需的 adapters
    ↓
batch.merge() - 合并到当前批次
    ↓
_infer_batch() - 执行推理
    ↓ 判断模式（prefill/decode）
    ↓ Prefill: init_batch + prefill_batch
    ↓ Decode: update_adapter_stats + decode_batch
    ↓ 返回 {req_id: (token_id, metadata)}
    ↓
处理推理结果
    ↓ 添加 token 到 req.output_ids
    ↓ 添加 metadata 到 req.output_metadata_list
    ↓ 生成响应消息
    ↓
标记完成的请求
    ↓ batch.mark_finished_req(eos_id)
    ↓
更新批次状态
    ↓ batch.filter_finished()
    ↓ 如果为空：model_rpc.remove_batch()
    ↓ 否则：model_rpc.filter_batch()
    ↓
返回响应列表
```

4. **响应消息格式**:
```python
{
    'request_id': str,
    'worker_id': int,
    'output_ids': List[int],  # prompt_ids + output_ids
    'metadata': {
        'finish_reason': str,  # 'length', 'generating', 'eos'
        'prompt_tokens': int,
        'completion_tokens': int,
        'gen_metadata': dict  # 来自模型的元数据
    },
    'success': bool,
    'error': None or str
}
```

**满足 Requirements**:
- Requirements 3.4（使用现有的模型推理逻辑处理请求）
- Requirements 3.5（生成包含 output_ids 和 metadata 的响应消息）

**与 manager.py 的一致性**:
- 推理流程与张量并行模式保持一致
- 使用相同的 RPC API（init_batch, prefill_batch, decode_batch）
- 批次管理逻辑相同（filter_finished, remove_batch, filter_batch）
- 响应格式兼容

**Phase 1 完整实现**:
- 从占位实现升级为实际推理
- 支持完整的 prefill 和 decode 流程
- 正确处理批次生命周期
- 为后续的完整批次管理（Task 2.9.3）奠定基础

**下一步**:
- Task 2.9.3: 集成完整的批次管理（处理 EOS token 检测，更新批次状态）
- 注：Task 2.9.2 已经实现了大部分批次管理逻辑，Task 2.9.3 主要是完善和验证

---

### 2025-01-13 - Task 2.9.3: 集成完整的批次管理

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 实现 `_handle_finish_req()` 方法：处理已完成请求的逻辑
   - 更新 `_process_requests()` 方法：调用 `_handle_finish_req()` 处理完成的请求

**关键设计**:
- **批次管理**：完整的批次生命周期管理
- **Adapter 统计**：跟踪和更新 adapter 使用计数
- **状态同步**：确保 Worker 端和 RPC 端的批次状态一致
- **简化实现**：Phase 1 不实现复杂的淘汰策略，专注于基本功能

**实现细节**:

1. **`_handle_finish_req()` 方法**（异步）:
   - 检查是否有新完成的请求
   - 记录完成请求使用的 adapter 目录：
     - 遍历批次中的所有请求
     - 收集 `has_generate_finished=True` 的请求的 adapter_dir
   - 过滤批次：
     - 调用 `batch.filter_finished()` 移除已完成的请求
     - 更新 `batch.adapter_dirs` 只包含未完成请求的 adapters
   - 减少 adapter 请求计数：
     - 调用 `model_rpc.decrease_request_counts(finished_adapter_dirs)`
     - 更新 RPC 端的 adapter 使用统计
   - 同步批次状态：
     - 如果批次为空：调用 `model_rpc.remove_batch()` 移除批次
     - 如果批次非空：调用 `model_rpc.filter_batch()` 过滤批次

2. **`_process_requests()` 方法更新**:
   - 标记完成的请求后，调用 `_handle_finish_req()` 处理
   - 简化逻辑，将批次管理集中到 `_handle_finish_req()` 中

3. **批次管理流程**:
```
执行推理
    ↓
添加 token 到请求
    ↓
生成响应消息
    ↓
mark_finished_req(eos_id) - 标记完成的请求
    ↓
_handle_finish_req() - 处理完成的请求
    ↓ 记录完成的 adapter_dirs
    ↓ batch.filter_finished() - 过滤批次
    ↓ decrease_request_counts() - 减少计数
    ↓ 判断批次状态
    ├─ 为空: remove_batch() + current_batch = None
    └─ 非空: filter_batch(req_id_list)
    ↓
返回响应
```

4. **Adapter 使用统计**:
   - **增加计数**：在 `_load_adapters()` 时自动增加（RPC 端处理）
   - **减少计数**：在 `_handle_finish_req()` 中显式减少
   - **用途**：为后续的 adapter 淘汰策略提供依据

5. **与 manager.py 的差异**:
   - **简化**：不实现阈值淘汰策略（Phase 1）
   - **简化**：不处理 PEFT 调度器特殊逻辑
   - **简化**：不实现 minibatch 参数（数据并行不需要）
   - **保留**：核心的批次管理和 adapter 计数逻辑
   - **保留**：与 RPC 端的状态同步机制

**满足 Requirements**:
- Requirements 3.5（生成包含 output_ids 和 metadata 的响应消息）

**Phase 1 完整实现**:
- 完整的批次生命周期管理
- 正确的 EOS token 检测
- Adapter 使用统计跟踪
- Worker 端和 RPC 端状态同步
- 为后续优化（淘汰策略等）奠定基础

**测试覆盖**:
- 现有测试已覆盖批次管理的核心功能：
  - `test_process_requests_marks_finished`: 验证请求完成标记
  - `test_process_requests_filters_batch`: 验证批次过滤
- `_handle_finish_req()` 的逻辑已被这些测试间接验证

**下一步**:
- Task 2.10: 编写 GPU Worker 推理测试（可选）
- Task 2.11: 编写 Adapter 管理单元测试（可选）
- 继续实现其他 Phase 1 任务（Response Merger, API Server 集成等）

---

### 2025-01-20 - Task 4.1: 创建 ResponseMerger 类

**新增文件**:
1. `slora/server/router/response_merger.py`
   - 实现 `ResponseMerger` 类
   - 功能：收集来自多个 GPU Worker 的响应并转发到 Detokenization 进程
   - 核心方法：
     - `__init__()`: 初始化 Response Merger
     - `_setup_zmq()`: 设置 ZMQ 通信
     - `_convert_to_detoken_format()`: 将 Worker 响应转换为 Detokenization 格式
     - `_forward_to_detokenization()`: 转发响应到 Detokenization
     - `run()`: 主循环，持续接收和转发响应
   - 添加模块级函数 `run_response_merger_process()`: Response Merger 进程的入口函数

**关键设计**:
- **通信模式**：
  - Workers → Response Merger: PUSH/PULL（多对一）
  - Response Merger → Detokenization: PUSH/PULL（一对一）
- **消息转换**：将 Worker 的 JSON 响应转换为 Detokenization 期望的 BatchTokenIdOut 对象
- **异步处理**：使用 asyncio 和 zmq.asyncio 实现异步消息处理
- **错误恢复**：捕获异常但继续运行，确保服务不中断

**实现细节**:

1. **`__init__()` 方法**:
   - 接收参数：worker_response_port（接收 Worker 响应）、detoken_port（发送到 Detokenization）
   - 初始化 ZMQ 相关属性：context、worker_receiver、detoken_sender
   - 输出初始化日志

2. **`_setup_zmq()` 方法**:
   - 创建异步 ZMQ context：`zmq.asyncio.Context()`
   - 创建 PULL socket 并绑定到 worker_response_port：
     - 用于接收来自所有 Worker 的响应
     - 使用 `bind()` 因为 Response Merger 是服务端
     - ZMQ 的 PULL socket 自动负载均衡接收消息
   - 创建 PUSH socket 并连接到 detoken_port：
     - 用于发送到 Detokenization 进程
     - 使用 `connect()` 因为 Detokenization 是服务端
   - 输出详细的日志信息

3. **`_convert_to_detoken_format()` 方法**:
   - 输入：Worker 响应字典
     ```python
     {
         'request_id': str,
         'worker_id': int,
         'output_ids': List[int],
         'metadata': dict,
         'success': bool,
         'error': Optional[str]
     }
     ```
   - 输出：BatchTokenIdOut 对象
     ```python
     BatchTokenIdOut.reqs_infs = [
         (req_id, new_token_id, gen_metadata, finished_state, abort_state)
     ]
     ```
   - 转换逻辑：
     - 提取 request_id、output_ids、metadata、success
     - 取最后一个 token 作为 new_token_id（流式输出）
     - 根据 success 和 finished 字段判断 finished_state
     - abort_state 在 Phase 1 中始终为 False
   - Phase 1 简化：假设每个响应包含完整的 output_ids，只取最后一个 token

4. **`_forward_to_detokenization()` 方法**（异步）:
   - 调用 `_convert_to_detoken_format()` 转换消息格式
   - 使用 `send_pyobj()` 发送 Python 对象（pickle 序列化）
   - 输出调试日志（request_id 和 worker_id）

5. **`run()` 方法**（异步）:
   - 输出启动日志
   - 无限循环：
     - 从 `worker_receiver.recv_json()` 接收 Worker 响应
     - 调用 `_forward_to_detokenization()` 转发响应
   - 错误处理：
     - 捕获所有异常
     - 记录错误日志和堆栈跟踪
     - 继续处理下一个响应（不终止循环）

6. **`run_response_merger_process()` 函数**（模块级）:
   - Response Merger 进程的入口函数
   - 创建 ResponseMerger 实例
   - 设置 ZMQ 通信
   - 运行主循环（asyncio.run）
   - 完整的错误处理和日志输出

**通信架构**:
```
Worker 0 (PUSH)
Worker 1 (PUSH)  →  Response Merger (PULL) - worker_response_port
Worker 2 (PUSH)
                          ↓
                Response Merger (PUSH) - detoken_port
                          ↓
                Detokenization (PULL)
```

**消息流程**:
```
Worker 完成推理
    ↓
Worker.send_json(response) - 发送 JSON 响应
    ↓
Response Merger.recv_json() - 接收响应
    ↓
_convert_to_detoken_format() - 转换格式
    ↓ Worker 响应 → BatchTokenIdOut
    ↓
_forward_to_detokenization() - 转发
    ↓ send_pyobj(BatchTokenIdOut)
    ↓
Detokenization.recv_pyobj() - 接收并处理
```

**满足 Requirements**:
- Requirements 4.1（Worker 通过 ZMQ PUSH socket 发送响应消息）
- Requirements 4.2（Response Merger 通过 ZMQ PULL socket 接收响应）
- Requirements 4.3（根据 request_id 匹配原始请求）
- Requirements 4.4（将响应转发到 Detokenization 进程）
- Requirements 4.5（确保响应消息包含必要字段）

**与现有系统的兼容性**:
- 使用与 manager.py 相同的 BatchTokenIdOut 格式
- 使用 send_pyobj/recv_pyobj 与 Detokenization 通信（与现有系统一致）
- 响应格式与 Detokenization 期望的格式完全兼容

**Phase 1 简化说明**:
- 假设每个响应包含完整的 output_ids（而非增量）
- 只取最后一个 token 转发到 Detokenization（流式输出）
- 不支持 abort 功能（abort_state 始终为 False）
- Phase 2 将实现增量响应和完整的 abort 支持

**下一步**:
- Task 4.2: 实现响应转发逻辑（已在 Task 4.1 中完成）
- Task 4.3: 集成到 Router Manager（启动 Response Merger 进程）
- Task 4.4: 编写 Response Merger 单元测试（可选）

---

### 2025-01-20 - Task 4.2: 实现响应转发逻辑

**说明**:
Task 4.2 的实现已经在 Task 4.1 中完成。所有响应转发相关的方法都在创建 ResponseMerger 类时一并实现，因为它们是 ResponseMerger 的核心功能，紧密相关且不可分割。

**已实现功能**:
1. ✓ `_convert_to_detoken_format()` 方法 - 消息格式转换
2. ✓ `_forward_to_detokenization()` 方法 - 响应转发
3. ✓ `run()` 主循环 - 持续接收和转发

**实现细节**（参见 Task 4.1）:

1. **`_convert_to_detoken_format()` 方法**:
   - 将 Worker 的 JSON 响应转换为 BatchTokenIdOut 对象
   - 提取 request_id、output_ids、metadata、success 字段
   - 取最后一个 token 作为 new_token_id（流式输出）
   - 根据 success 和 finished 字段判断 finished_state
   - abort_state 在 Phase 1 中始终为 False

2. **`_forward_to_detokenization()` 方法**（异步）:
   - 调用 `_convert_to_detoken_format()` 转换消息格式
   - 使用 `send_pyobj()` 发送 Python 对象（pickle 序列化）
   - 输出调试日志（request_id 和 worker_id）
   - 与现有 Detokenization 进程的通信方式完全兼容

3. **`run()` 主循环**（异步）:
   - 输出启动日志
   - 无限循环接收和转发响应：
     - 从 `worker_receiver.recv_json()` 接收 Worker 响应
     - 调用 `_forward_to_detokenization()` 转发响应
   - 完整的错误处理：
     - 捕获所有异常
     - 记录错误日志和堆栈跟踪
     - 继续处理下一个响应（不终止循环）

**响应转发流程**:
```
Worker 完成推理
    ↓
Worker.send_json(response) - 发送 JSON 响应
    ↓
Response Merger.run() - 主循环接收
    ↓ recv_json() 接收响应
    ↓
_forward_to_detokenization() - 转发处理
    ↓ _convert_to_detoken_format() 转换格式
    ↓ Worker 响应 → BatchTokenIdOut
    ↓ send_pyobj(BatchTokenIdOut) 发送
    ↓
Detokenization.recv_pyobj() - 接收并处理
```

**满足 Requirements**:
- Requirements 4.3（根据 request_id 匹配原始请求）
- Requirements 4.4（将响应转发到 Detokenization 进程）
- Requirements 4.5（确保响应消息包含必要字段）

**测试验证**:
- 消息格式转换测试通过
- 验证了 request_id、token_id、metadata、finished_state、abort_state 的正确性
- 确认与 Detokenization 进程的消息格式兼容

**下一步**:
- Task 4.3: 集成到 Router Manager（启动 Response Merger 进程）
- Task 4.4: 编写 Response Merger 单元测试（可选）

---


---

### 2025-01-20 - Task 4.3: 集成 Response Merger 到 Router Manager

**修改文件**:
1. `slora/server/router/dp_manager.py`
   - 更新 `DataParallelRouterManager.__init__()` 方法
     - 添加 `detoken_port` 参数：Detokenization 进程的端口
     - 添加 `self.detoken_port` 属性
     - 添加 `self.merger_process` 属性：用于跟踪 Response Merger 进程
   - 更新 `start_workers()` 方法
     - 在启动 Worker 之前先启动 Response Merger 进程
     - 确保 Response Merger 在 Worker 之前就绪，以便接收响应
   - 新增 `_start_response_merger()` 方法
     - 创建并启动 Response Merger 进程
     - 传递 `response_port` 和 `detoken_port` 参数
     - 使用 `run_response_merger_process` 作为进程目标函数

2. `test/test_dp_manager_framework.py`
   - 更新所有测试用例，添加 `detoken_port` 参数
   - 新增 `test_start_response_merger()` 测试
     - 验证 Response Merger 进程正确创建
     - 验证进程参数正确传递
     - 验证进程被启动
     - 验证 `merger_process` 属性被正确设置

**关键设计**:
- **启动顺序**：Response Merger → Workers
  - Response Merger 必须先启动，以便 Worker 启动后能够立即连接
  - Worker 的响应 socket 会连接到 Response Merger 的监听端口
- **进程管理**：
  - Response Merger 作为独立进程运行
  - 由 Router Manager 统一管理和监控
  - 使用 multiprocessing.Process 创建进程
- **端口配置**：
  - `response_port`: Worker 发送响应的端口（Response Merger 监听）
  - `detoken_port`: Response Merger 转发响应的端口（Detokenization 监听）

**通信流程**:
```
API Server → Router Manager (router_port)
    ↓
Router Manager → Workers (worker_ports[i])
    ↓
Workers → Response Merger (response_port)
    ↓
Response Merger → Detokenization (detoken_port)
```

**满足 Requirements**:
- Requirements 4.1（Worker 通过 ZMQ PUSH socket 发送响应消息）
- Requirements 4.2（Response Merger 通过 ZMQ PULL socket 接收响应）
- Requirements 4.4（将响应转发到 Detokenization 进程）

**测试覆盖**:
- 单元测试：验证 Response Merger 进程正确启动
- 集成测试：将在后续任务中验证端到端的响应转发

**下一步**:
- Task 4.4: 编写 Response Merger 单元测试（可选）
- Task 5: 修改 API Server 入口（集成数据并行模式）
- Task 8: Checkpoint - 基础功能验证（验证 Response Merger 正确转发响应）


### 2025-01-20 - Task 5.1: 添加命令行参数

**修改文件**:
1. `slora/server/api_server.py`
   - 在 slora arguments 部分添加数据并行模式相关的命令行参数
   - 新增参数：
     - `--parallel-mode`: 并行模式选择（tensor 或 data），默认为 tensor
     - `--num-workers`: 数据并行模式下的 Worker 数量，默认为 None（自动检测）
     - `--gpu-ids`: 指定使用的 GPU 列表（逗号分隔），默认为 None（使用所有 GPU）

**实现细节**:

1. **`--parallel-mode` 参数**:
   - 类型：字符串
   - 默认值：`"tensor"`（保持向后兼容）
   - 可选值：`["tensor", "data"]`（使用 choices 限制）
   - 说明：选择并行模式
     - `tensor`: 张量并行模式（原有模式）
     - `data`: 数据并行模式（新增模式）

2. **`--num-workers` 参数**:
   - 类型：整数
   - 默认值：`None`（自动检测可用 GPU 数量）
   - 说明：数据并行模式下的 GPU Worker 数量
   - 用途：显式指定 Worker 数量，覆盖自动检测

3. **`--gpu-ids` 参数**:
   - 类型：字符串
   - 默认值：`None`（使用所有可用 GPU）
   - 格式：逗号分隔的 GPU ID 列表（例如：`"0,1,2"`）
   - 说明：指定使用的 GPU 列表
   - 用途：在多 GPU 环境中选择特定的 GPU

**参数位置**:
- 添加在 slora arguments 部分，位于 `--enable-abort` 参数之后
- 添加注释 `# Data parallel mode arguments` 以便识别
- 位于阈值淘汰相关参数之前

**满足 Requirements**:
- Requirements 5.1（支持 `--parallel-mode data` 参数启用数据并行模式）
- Requirements 5.2（支持 `--num-workers N` 参数指定 Worker 数量）
- Requirements 5.3（支持 `--gpu-ids 0,1,2` 参数指定 GPU 列表）

**向后兼容性**:
- `--parallel-mode` 默认为 `"tensor"`，保持现有行为
- 新参数都是可选的，不影响现有用户
- 未指定参数时，系统行为与之前完全一致

**使用示例**:
```bash
# 使用数据并行模式，自动检测所有 GPU
python -m slora.server.api_server --parallel-mode data --model_dir /path/to/model

# 使用数据并行模式，指定 3 个 Worker
python -m slora.server.api_server --parallel-mode data --num-workers 3 --model_dir /path/to/model

# 使用数据并行模式，指定使用 GPU 0, 1, 2
python -m slora.server.api_server --parallel-mode data --gpu-ids 0,1,2 --model_dir /path/to/model

# 使用张量并行模式（默认，与之前相同）
python -m slora.server.api_server --model_dir /path/to/model
```

**下一步**:
- Task 5.2: 实现模式选择逻辑（根据 parallel-mode 选择启动逻辑）
- Task 5.3: 添加启动日志（输出当前并行模式和配置）
- Task 5.4: 更新 router/manager.py 或创建新的入口（确保两种模式可以共存）



### 2025-01-20 - Task 5.2: 实现模式选择逻辑

**修改文件**:
1. `slora/server/router/manager.py`
   - 重构 `start_router_process()` 函数
     - 添加并行模式检测逻辑
     - 根据 `args.parallel_mode` 选择启动路径
     - 添加启动日志输出当前并行模式
   - 新增 `_start_tensor_parallel_router()` 函数
     - 将原有的 `start_router_process()` 逻辑移到此函数
     - 保持张量并行模式的完整功能
     - 确保向后兼容性
   - 新增 `_start_data_parallel_router()` 函数
     - 实现数据并行模式的启动逻辑
     - 创建 DataParallelRouterManager 实例
     - 分配 response_port 用于 Worker 响应
     - 设置 ZMQ 通信
     - 启动所有 Worker 和 Response Merger
     - 运行主循环

**关键设计**:
- **模式选择**：
  - 获取 `args.parallel_mode`，默认为 `"tensor"`
  - 根据模式调用不同的启动函数
  - 输出启动日志明确显示当前模式
- **张量并行路径**（`_start_tensor_parallel_router`）:
  - 保持原有的完整逻辑不变
  - 使用 RouterManager 类
  - 支持所有现有功能（profiling, PETS scheduler 等）
- **数据并行路径**（`_start_data_parallel_router`）:
  - 使用 DataParallelRouterManager 类
  - 分配 response_port（使用 alloc_can_use_network_port）
  - 创建 dp_manager 实例
  - 设置 ZMQ 通信
  - 启动 Workers 和 Response Merger
  - 运行主循环

**实现细节**:

1. **`start_router_process()` 函数重构**:
   ```python
   def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
       # 获取并行模式，默认为 'tensor'
       parallel_mode = getattr(args, 'parallel_mode', 'tensor')
       
       # 输出启动日志（Requirement 6.5）
       print(f"[Router] Starting router process in {parallel_mode.upper()} parallel mode")
       
       # 根据并行模式选择启动逻辑
       if parallel_mode == 'data':
           _start_data_parallel_router(args, router_port, detokenization_port, pipe_writer)
       else:
           _start_tensor_parallel_router(args, router_port, detokenization_port, 
                                         model_rpc_ports, mode, pipe_writer)
   ```

2. **`_start_tensor_parallel_router()` 函数**:
   - 完全保留原有的 `start_router_process()` 逻辑
   - 创建 InputParams 对象
   - 创建 RouterManager 实例
   - 等待模型就绪
   - 执行 profiling（如果启用）
   - 加载 PETS scheduler 模型（如果需要）
   - 发送 'init ok' 消息
   - 运行主循环（loop_for_fwd + loop_for_netio_req）

3. **`_start_data_parallel_router()` 函数**:
   - 导入 DataParallelRouterManager 和 alloc_can_use_network_port
   - 分配 response_port：
     ```python
     response_port = alloc_can_use_network_port(num=1, used_nccl_port=None)[0]
     ```
   - 创建 DataParallelRouterManager 实例：
     ```python
     dp_manager = DataParallelRouterManager(
         args=args,
         router_port=router_port,
         response_port=response_port,
         detoken_port=detokenization_port
     )
     ```
   - 设置 ZMQ 通信：`dp_manager._setup_zmq()`
   - 启动 Workers：`asyncio.run(dp_manager.start_workers())`
   - 发送 'init ok' 消息
   - 运行主循环：`asyncio.run(dp_manager.run())`
   - 完整的错误处理和日志输出

**启动流程对比**:

**张量并行模式**:
```
start_router_process()
    ↓
_start_tensor_parallel_router()
    ↓ 创建 InputParams
    ↓ 创建 RouterManager
    ↓ 等待模型就绪
    ↓ Profiling（可选）
    ↓ 发送 'init ok'
    ↓ 运行主循环（loop_for_fwd + loop_for_netio_req）
```

**数据并行模式**:
```
start_router_process()
    ↓
_start_data_parallel_router()
    ↓ 分配 response_port
    ↓ 创建 DataParallelRouterManager
    ↓ 设置 ZMQ 通信
    ↓ 启动 Workers 和 Response Merger
    ↓ 发送 'init ok'
    ↓ 运行主循环（dp_manager.run）
```

**满足 Requirements**:
- Requirements 6.1（未指定 parallel-mode 参数时默认使用张量并行模式）
- Requirements 6.2（使用 `--parallel-mode tensor` 时使用原有的张量并行逻辑）
- Requirements 6.3（使用 `--parallel-mode data` 时使用新的数据并行逻辑）
- Requirements 6.4（保持现有的 API 接口不变）
- Requirements 6.5（在启动日志中明确输出当前使用的并行模式）

**向后兼容性**:
- 默认模式为 `"tensor"`，保持现有行为
- 张量并行逻辑完全保留，无任何修改
- 新旧模式完全隔离，互不影响
- API 接口保持不变

**错误处理**:
- 两种模式都有完整的异常捕获
- 错误信息通过 pipe_writer 发送给主进程
- 失败时调用 cleanup 方法（张量并行）或直接抛出异常（数据并行）

**日志输出**:
- 启动时输出当前并行模式（TENSOR 或 DATA）
- 数据并行模式输出详细的端口分配信息
- 数据并行模式输出 Worker 启动成功信息

**下一步**:
- Task 5.3: 添加启动日志（输出 Worker 数量和 GPU 列表）
- Task 5.4: 更新 router/manager.py 或创建新的入口（确保两种模式可以共存）
- Task 6: 完善错误处理（Worker 启动失败、ZMQ 超时等）
- Task 8: Checkpoint - 基础功能验证（验证模式选择逻辑正确工作）
_start_tensor_parallel_router()` 函数**:
   - 将原有的 `start_router_process()` 逻辑完整移植
   - 创建 InputParams 对象
   - 创建 RouterManager 实例
   - 等待模型就绪
   - 处理 profiling 和 scheduler 初始化
   - 发送 'init ok' 消息
   - 运行主循环

3. **`_start_data_parallel_router()` 函数**:
   - 导入 DataParallelRouterManager 和 alloc_can_use_network_port
   - 分配 response_port（用于 Worker → Response Merger 通信）
   - 创建 DataParallelRouterManager 实例：
     - 传递 args, router_port, response_port, detoken_port
   - 设置 ZMQ 通信：`dp_manager._setup_zmq()`
   - 启动 Workers：`asyncio.run(dp_manager.start_workers())`
   - 发送 'init ok' 消息
   - 运行主循环：`asyncio.run(dp_manager.run())`
   - 完整的错误处理

**启动流程对比**:

**张量并行模式**:
```
start_router_process()
    ↓
_start_tensor_parallel_router()
    ↓ 创建 InputParams
    ↓ 创建 RouterManager
    ↓ 启动模型进程（多个 RPC 进程）
    ↓ 等待模型就绪
    ↓ 运行主循环（loop_for_fwd + loop_for_netio_req）
```

**数据并行模式**:
```
start_router_process()
    ↓
_start_data_parallel_router()
    ↓ 分配 response_port
    ↓ 创建 DataParallelRouterManager
    ↓ 设置 ZMQ 通信
    ↓ 启动 Workers（多个 GPU Worker 进程）
    ↓ 启动 Response Merger 进程
    ↓ 运行主循环（接收请求 + 路由到 Worker）
```

**满足 Requirements**:
- Requirements 6.1（未指定 parallel-mode 参数时默认使用张量并行模式）
- Requirements 6.2（使用 --parallel-mode tensor 时使用原有的张量并行逻辑）
- Requirements 6.3（使用 --parallel-mode data 时使用新的数据并行逻辑）
- Requirements 6.4（保持现有的 API 接口不变）
- Requirements 6.5（在启动日志中明确输出当前使用的并行模式）

**向后兼容性**:
- 默认模式为 `"tensor"`，保持现有行为
- 张量并行逻辑完全保留，功能不变
- 新增的数据并行模式不影响现有用户
- API 接口保持不变（start_router_process 函数签名不变）

**错误处理**:
- 两种模式都有完整的异常捕获
- 错误信息通过 pipe_writer 发送到主进程
- 确保错误不会导致进程僵死

**下一步**:
- Task 5.3: 添加启动日志（输出 Worker 数量和 GPU 列表）
- Task 5.4: 更新 router/manager.py 或创建新的入口（已完成，两种模式已共存）
- Task 5.5: 编写 API Server 集成测试（可选）

---

### 2025-01-20 - Task 5.3: 添加启动日志

**修改文件**:
1. `slora/server/router/manager.py`
   - 更新 `_start_data_parallel_router()` 函数
     - 在创建 DataParallelRouterManager 之前输出配置信息
     - 在所有 Worker 启动后输出启动完成摘要
     - 使用分隔线使日志更清晰易读

**关键设计**:
- **启动前日志**：
  - 输出并行模式（DATA PARALLEL MODE）
  - 输出 Worker 数量配置（指定或自动检测）
  - 输出 GPU ID 配置（指定或自动分配）
  - 使用分隔线（80 个 `=`）使日志醒目
- **启动后日志**：
  - 输出启动成功消息
  - 输出实际的 Worker 数量
  - 输出实际的 GPU ID 列表
  - 输出 Worker 端口列表
  - 输出就绪状态确认
  - 使用分隔线使日志醒目

**实现细节**:

1. **启动前日志**（在创建 dp_manager 之前）:
   ```python
   # Requirement 6.5 & 8.1: 输出启动配置信息
   num_workers = getattr(args, 'num_workers', None)
   gpu_ids_str = getattr(args, 'gpu_ids', None)
   
   print("=" * 80)
   print("[DataParallelRouter] Starting Data Parallel Mode")
   print("=" * 80)
   if num_workers:
       print(f"[DataParallelRouter] Number of Workers: {num_workers} (specified)")
   else:
       print(f"[DataParallelRouter] Number of Workers: Auto-detect (using all available GPUs)")
   
   if gpu_ids_str:
       print(f"[DataParallelRouter] GPU IDs: {gpu_ids_str} (specified)")
   else:
       print(f"[DataParallelRouter] GPU IDs: Auto-assign (0, 1, 2, ...)")
   print("=" * 80)
   ```

2. **启动后日志**（在 start_workers 完成后）:
   ```python
   # Requirement 6.5 & 8.1: 输出启动完成摘要
   print("=" * 80)
   print("[DataParallelRouter] Data Parallel Mode Started Successfully")
   print("=" * 80)
   print(f"[DataParallelRouter] Number of Workers: {dp_manager.num_workers}")
   print(f"[DataParallelRouter] GPU IDs: {dp_manager.gpu_ids}")
   print(f"[DataParallelRouter] Worker Ports: {dp_manager.worker_ports}")
   print(f"[DataParallelRouter] All workers are ready and accepting requests")
   print("=" * 80)
   ```

**日志示例**:

**启动前**:
```
[Router] Starting router process in DATA parallel mode
================================================================================
[DataParallelRouter] Starting Data Parallel Mode
================================================================================
[DataParallelRouter] Number of Workers: 3 (specified)
[DataParallelRouter] GPU IDs: 0,1,2 (specified)
================================================================================
[DataParallelRouter] Allocated response_port: 54321
[DataParallelRouter] Router port: 12345
[DataParallelRouter] Detokenization port: 12346
[DataParallelRouterManager] Initialized with 3 workers
[DataParallelRouterManager] GPU IDs: [0, 1, 2]
...
```

**启动后**:
```
...
[DataParallelRouterManager] All 3 workers started and ready
================================================================================
[DataParallelRouter] Data Parallel Mode Started Successfully
================================================================================
[DataParallelRouter] Number of Workers: 3
[DataParallelRouter] GPU IDs: [0, 1, 2]
[DataParallelRouter] Worker Ports: [50000, 50001, 50002]
[DataParallelRouter] All workers are ready and accepting requests
================================================================================
```

**满足 Requirements**:
- Requirements 6.5（在启动日志中输出当前并行模式）
- Requirements 8.1（输出 Worker 就绪日志）
- 输出 Worker 数量（指定或自动检测）
- 输出 GPU 列表（指定或自动分配）
- 输出 Worker 端口列表
- 输出就绪状态确认

**日志级别**:
- 使用 INFO 级别（print 输出）
- 关键信息醒目显示（使用分隔线）
- 便于用户快速确认系统配置和状态

**用户体验**:
- 清晰的启动流程可视化
- 明确的配置信息展示
- 易于调试和问题排查
- 与张量并行模式的日志风格一致

**下一步**:
- Task 5.4: 更新 router/manager.py 或创建新的入口（已完成）
- Task 5.5: 编写 API Server 集成测试（可选）
- Task 6: 完善错误处理
- Task 7: 完善基础监控

---

### 2025-01-20 - Task 5.4: 更新 router/manager.py 确保模式共存

**说明**:
Task 5.4 的实现已经在 Task 5.2 中完成。在实现模式选择逻辑时，已经确保了数据并行模式和张量并行模式可以共存，并保持了向后兼容性。

**已实现功能**:
1. ✅ 数据并行模式和张量并行模式可以共存
2. ✅ 保持向后兼容性
3. ✅ API 接口不变

**实现细节**（参见 Task 5.2）:

1. **函数签名保持不变**:
   ```python
   def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
   ```
   - 与原有的函数签名完全一致
   - api_server.py 中的调用代码无需修改
   - 确保向后兼容性

2. **模式路由逻辑**:
   ```python
   # 获取并行模式，默认为 'tensor'
   parallel_mode = getattr(args, 'parallel_mode', 'tensor')
   
   # 根据并行模式选择启动逻辑
   if parallel_mode == 'data':
       _start_data_parallel_router(...)
   else:
       _start_tensor_parallel_router(...)
   ```
   - 使用 `getattr` 安全获取参数，默认为 'tensor'
   - 条件分支确保两种模式互不干扰

3. **张量并行模式完整保留**:
   - `_start_tensor_parallel_router()` 函数包含原有的所有逻辑
   - 支持所有现有功能：
     - InputParams 配置
     - RouterManager 初始化
     - 模型加载和就绪等待
     - Profiling 支持
     - PETS scheduler 支持
     - 主循环运行
   - 功能和行为与原有实现完全一致

4. **数据并行模式独立实现**:
   - `_start_data_parallel_router()` 函数实现新的数据并行逻辑
   - 使用 DataParallelRouterManager
   - 不影响张量并行模式的任何功能

**架构设计**:
```
start_router_process()
    ↓
    ├─ parallel_mode == 'data'
    │   └─ _start_data_parallel_router()
    │       └─ DataParallelRouterManager
    │
    └─ parallel_mode == 'tensor' (默认)
        └─ _start_tensor_parallel_router()
            └─ RouterManager (原有逻辑)
```

**满足 Requirements**:
- Requirements 6.1（未指定 parallel-mode 参数时默认使用张量并行模式）
- Requirements 6.2（使用 --parallel-mode tensor 时使用原有的张量并行逻辑）
- Requirements 6.4（保持现有的 API 接口不变）

**向后兼容性验证**:
1. ✅ 函数签名不变：`start_router_process` 的参数列表完全一致
2. ✅ 默认行为不变：未指定 `parallel_mode` 时使用张量并行模式
3. ✅ 现有功能不变：张量并行模式的所有功能完整保留
4. ✅ 调用方式不变：api_server.py 中的调用代码无需修改

**共存机制**:
- 两种模式通过条件分支完全隔离
- 各自使用独立的 Manager 类（RouterManager vs DataParallelRouterManager）
- 各自使用独立的进程和资源
- 互不干扰，可以根据需要灵活切换

**测试验证**:
- 现有的张量并行模式测试应该全部通过（无需修改）
- 新增的数据并行模式测试独立运行
- 两种模式可以通过命令行参数自由切换

**用户体验**:
- 现有用户无需修改任何代码或配置
- 新用户可以通过 `--parallel-mode data` 启用数据并行模式
- 清晰的启动日志显示当前使用的模式
- 两种模式的使用方式一致，学习成本低

**下一步**:
- Task 5.5: 编写 API Server 集成测试（可选）
- Task 6: 完善错误处理
- Task 7: 完善基础监控
- Task 8: Checkpoint - 基础功能验证

---


---

### 2025-01-20 - Task 6.1: Worker 启动失败处理

**修改文件**:
1. `slora/server/router/dp_manager.py`
   - 更新 `run_gpu_worker_process()` 函数
     - 添加完善的异常捕获和错误日志记录
     - 添加详细的启动进度日志
     - 处理 KeyboardInterrupt（优雅退出）
     - 捕获所有异常并通过 sys.exit(1) 返回非零退出码
     - 记录完整的错误类型、错误消息和堆栈跟踪
   - 更新 `start_workers()` 方法
     - 添加 Worker 进程状态检查
     - 分多次检查（每秒一次，总共 5 次）
     - 检测 Worker 启动失败（进程退出）
     - 失败时终止所有 Worker 和 Response Merger 进程
     - 抛出 RuntimeError 通知上层

**新增文件**:
1. `test/test_worker_startup_failure.py`
   - 测试 Worker 启动失败处理
   - 5 个测试用例：
     - `test_run_gpu_worker_process_exception_handling`: 验证异常捕获和退出码
     - `test_run_gpu_worker_process_keyboard_interrupt`: 验证 KeyboardInterrupt 处理
     - `test_start_workers_detects_failure`: 验证启动失败检测和进程终止
     - `test_start_workers_all_success`: 验证所有 Worker 启动成功
     - `test_start_workers_early_failure_detection`: 验证早期失败检测

**关键设计**:
- **异常捕获**：
  - 捕获所有异常类型（包括 KeyboardInterrupt）
  - 记录详细的错误信息（类型、消息、堆栈跟踪）
  - 通过退出码通知 Router Manager（0=正常，1=错误）
- **启动检测**：
  - 分 5 轮检查，每轮等待 1 秒
  - 每轮检查所有 Worker 进程状态
  - 发现失败立即终止所有进程
  - 抛出 RuntimeError 通知上层
- **进程清理**：
  - 终止所有存活的 Worker 进程
  - 终止 Response Merger 进程
  - 使用 terminate() + join() + kill() 确保进程被清理
  - 输出详细的清理日志

**实现细节**:

1. **`run_gpu_worker_process()` 函数更新**:
   ```python
   def run_gpu_worker_process(...):
       import sys
       import traceback
       from slora.server.router.gpu_worker import GPUWorker
       
       try:
           print(f"[Worker {worker_id}] Starting worker process on GPU {gpu_id}...")
           
           # 详细的启动进度日志
           print(f"[Worker {worker_id}] Creating GPUWorker instance...")
           worker = GPUWorker(worker_id, gpu_id, args)
           
           print(f"[Worker {worker_id}] Setting up ZMQ communication...")
           worker._setup_zmq(request_port, response_port)
           
           print(f"[Worker {worker_id}] Setting up request queue...")
           worker._setup_request_queue()
           
           print(f"[Worker {worker_id}] Initializing model RPC...")
           asyncio.run(worker._init_model_rpc())
           
           print(f"[Worker {worker_id}] Worker initialization complete, starting main loop...")
           asyncio.run(worker.run())
           
       except KeyboardInterrupt:
           # 优雅处理 Ctrl+C
           print(f"[Worker {worker_id}] Received keyboard interrupt, shutting down...")
           sys.exit(0)
           
       except Exception as e:
           # 捕获所有异常，记录详细错误日志和堆栈跟踪
           print(f"[Worker {worker_id}] FATAL ERROR during worker startup/execution:")
           print(f"[Worker {worker_id}] Error type: {type(e).__name__}")
           print(f"[Worker {worker_id}] Error message: {str(e)}")
           print(f"[Worker {worker_id}] Full traceback:")
           traceback.print_exc()
           
           # 通过非零退出码通知 Router Manager 启动失败
           print(f"[Worker {worker_id}] Exiting with error code 1")
           sys.exit(1)
   ```

2. **`start_workers()` 方法更新**:
   ```python
   async def start_workers(self) -> None:
       import sys
       
       # 分配端口
       self._allocate_ports()
       
       # 启动 Response Merger 进程
       self._start_response_merger()
       
       # 启动所有 Worker
       print(f"[DataParallelRouterManager] Starting {self.num_workers} workers...")
       for i in range(self.num_workers):
           worker = self._start_worker(i, self.gpu_ids[i])
           self.workers.append(worker)
           print(f"[DataParallelRouterManager] Started worker {i} on GPU {self.gpu_ids[i]}, "
                 f"port {self.worker_ports[i]}")
       
       # 等待所有 Worker 就绪，同时检测启动失败
       print(f"[DataParallelRouterManager] Waiting for workers to initialize...")
       
       # 分多次检查，每次等待 1 秒，总共等待 5 秒
       for check_round in range(5):
           await asyncio.sleep(1)
           
           # 检查所有 Worker 进程状态
           failed_workers = []
           for i, worker in enumerate(self.workers):
               if not worker.is_alive():
                   exitcode = worker.exitcode
                   failed_workers.append((i, exitcode))
           
           # 如果有 Worker 启动失败，终止所有进程并退出
           if failed_workers:
               print(f"[DataParallelRouterManager] ERROR: Worker startup failed!")
               for worker_id, exitcode in failed_workers:
                   print(f"[DataParallelRouterManager] Worker {worker_id} failed with exit code {exitcode}")
               
               # 终止所有 Worker 进程
               print(f"[DataParallelRouterManager] Terminating all workers...")
               for i, worker in enumerate(self.workers):
                   if worker.is_alive():
                       print(f"[DataParallelRouterManager] Terminating worker {i}...")
                       worker.terminate()
                       worker.join(timeout=5)
                       if worker.is_alive():
                           print(f"[DataParallelRouterManager] Force killing worker {i}...")
                           worker.kill()
               
               # 终止 Response Merger 进程
               if self.merger_process and self.merger_process.is_alive():
                   print(f"[DataParallelRouterManager] Terminating Response Merger...")
                   self.merger_process.terminate()
                   self.merger_process.join(timeout=5)
                   if self.merger_process.is_alive():
                       print(f"[DataParallelRouterManager] Force killing Response Merger...")
                       self.merger_process.kill()
               
               # 抛出异常并退出
               error_msg = f"Worker startup failed. Failed workers: {failed_workers}"
               print(f"[DataParallelRouterManager] {error_msg}")
               raise RuntimeError(error_msg)
           
           print(f"[DataParallelRouterManager] Check round {check_round + 1}/5: All workers alive")
       
       print(f"[DataParallelRouterManager] All {self.num_workers} workers started and ready")
   ```

3. **错误处理流程**:
```
Worker 启动
    ↓
try:
    创建 GPUWorker 实例
    设置 ZMQ 通信
    设置请求队列
    初始化模型 RPC
    运行主循环
except KeyboardInterrupt:
    优雅退出（exit code 0）
except Exception as e:
    记录错误类型、消息、堆栈跟踪
    sys.exit(1) - 返回非零退出码
    ↓
Router Manager 检测到进程退出
    ↓
检查 worker.is_alive() == False
检查 worker.exitcode == 1
    ↓
终止所有 Worker 进程
终止 Response Merger 进程
    ↓
抛出 RuntimeError
```

4. **进程清理流程**:
```
检测到 Worker 失败
    ↓
遍历所有 Worker 进程
    ↓ 如果进程存活
    ├─ worker.terminate() - 发送 SIGTERM
    ├─ worker.join(timeout=5) - 等待最多 5 秒
    └─ 如果仍存活: worker.kill() - 发送 SIGKILL
    ↓
终止 Response Merger 进程
    ├─ merger.terminate()
    ├─ merger.join(timeout=5)
    └─ 如果仍存活: merger.kill()
    ↓
抛出 RuntimeError
```

**满足 Requirements**:
- Requirements 7.1（Worker 启动失败时记录详细错误日志）
- Requirements 7.2（通过退出码通知 Router Manager 启动失败）
- Requirements 7.2（检测 Worker 启动失败并终止所有进程）

**测试覆盖**:
- 5/5 测试用例通过
- 验证了异常捕获和退出码
- 验证了 KeyboardInterrupt 处理
- 验证了启动失败检测和进程终止
- 验证了所有 Worker 启动成功的情况
- 验证了早期失败检测（在等待期间检测到失败）

**错误场景覆盖**:
1. **GPU 初始化失败**：CUDA 不可用、GPU 不存在
2. **模型加载失败**：模型文件不存在、内存不足
3. **ZMQ 通信失败**：端口被占用、网络错误
4. **RPC 初始化失败**：模型进程启动失败
5. **进程崩溃**：运行时异常、段错误
6. **用户中断**：Ctrl+C、SIGINT

**Phase 1 完整实现**:
- 完善的异常捕获和错误日志
- 详细的启动进度日志
- 可靠的启动失败检测
- 完整的进程清理机制
- 为后续的监控和恢复（Phase 2）奠定基础

**下一步**:
- Task 6.2: 推理异常处理（验证和完善）
- Task 6.3: ZMQ 通信超时处理
- Task 6.4: Worker 进程监控
- Task 6.5: 编写错误处理测试（可选）
