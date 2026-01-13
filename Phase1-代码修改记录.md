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
   - 核心方法：
     - `_load_model()`: 加载基座模型
       - 读取模型配置文件
       - 检测模型类型和特性（如 GQA）
       - 实例化对应的模型类

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
   - 特性：
     - 异步消息处理
     - 完整的错误处理
     - 请求 ID 一致性保证
     - Worker ID 追踪
     - 元数据生成

**实现细节**:

1. **`_receive_request()` 方法**:
   - 从 ZMQ PULL socket 异步接收请求
   - 返回 JSON 格式的请求消息
   - 阻塞等待直到收到请求

2. **`_process_request()` 方法**:
   - 解析请求消息（request_id, adapter_dir, prompt_ids, sampling_params）
   - 检查并加载 adapter（如果需要）
   - 执行推理（Phase 1 简化实现）
   - 生成响应消息，包含：
     - request_id: 与请求保持一致
     - worker_id: 当前 Worker 的 ID
     - output_ids: 输出 token IDs
     - metadata: 元数据（finish_reason, token 统计等）
     - success: 成功标志
     - error: 错误信息（如果失败）
   - 完整的异常捕获和错误响应生成

3. **`_send_response()` 方法**:
   - 通过 ZMQ PUSH socket 异步发送响应
   - 使用 JSON 格式

4. **`run()` 方法**:
   - 无限循环，持续处理请求
   - 输出 Worker 就绪日志
   - 按顺序执行：接收 → 处理 → 发送
   - 错误不会终止循环，保证服务持续运行

**Phase 1 简化说明**:
- 推理逻辑使用占位实现（简单追加 token）
- Adapter 加载暂未实现（输出日志）
- 完整的推理逻辑将在后续任务中实现
- 当前实现确保消息流通和架构正确性

**新增测试文件**:
1. `test/test_gpu_worker_request_processing.py`
   - GPU Worker 请求处理的单元测试
   - 测试覆盖：
     - 接收请求功能
     - 成功处理请求
     - 带 adapter 的请求处理
     - 错误处理
     - 发送响应
     - 响应消息格式验证
     - 请求 ID 一致性（Property 3）
     - Worker ID 正确性
     - 元数据生成
   - 所有测试通过 ✓ (9/9)

**验证结果**:
- 单元测试：9/9 通过
- 满足 Requirements 3.1（持续监听 ZMQ PULL socket 接收请求）
- 满足 Requirements 3.2（解析请求消息并提取必要的参数）
- 满足 Requirements 3.5（生成包含 output_ids 和 metadata 的响应消息）
- 满足 Requirements 8.3（输出 Worker 就绪日志）
- 满足 Property 3（请求响应一致性）
- 请求处理流程完整，错误处理健壮

---

### 2025-01-13 - Task 2.4: 集成 ReqQueue 请求管理

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 添加 ReqQueue 相关导入：
     - `ReqQueue` (请求队列管理)
     - `Req`, `Batch` (请求和批次对象)
     - `SamplingParams` (采样参数)
   - 添加请求队列管理属性：
     - `req_queue`: ReqQueue 实例
     - `current_batch`: 当前批次
     - `lora_ranks`: LoRA rank 信息（用于显存管理）
     - `actual_adapter_size`: 实际 adapter 占用的显存大小
   - 实现新方法：
     - `_setup_request_queue()`: 初始化 ReqQueue
     - `_convert_to_req_object()`: 将 ZMQ 消息转换为 Req 对象
     - `_process_requests()`: 批处理请求（使用 ReqQueue）
   - 更新 `run()` 方法：支持批处理循环

**关键设计**:
- **复用张量并行的 ReqQueue**：每个 GPU Worker 内部包含一个独立的 ReqQueue 实例
- **批处理管理**：使用 ReqQueue 的 `generate_new_batch()` 方法生成批次
- **显存管理**：ReqQueue 自动处理显存分配和 Adapter 调度
- **本质**：每个 Worker 是一个"单 GPU 的张量并行系统"

**实现细节**:

1. **`_setup_request_queue()` 方法**:
   - 创建 ReqQueue 实例
   - 配置参数：`max_total_tokens`, `batch_max_tokens`, `running_max_req_size`

2. **`_convert_to_req_object()` 方法**:
   - 将 ZMQ 消息格式转换为 Req 对象
   - 解析采样参数（SamplingParams）
   - 支持所有采样参数：do_sample, temperature, top_p, top_k, presence_penalty, frequency_penalty, max_new_tokens, ignore_eos, stop_sequences

3. **`_process_requests()` 方法**（批处理版本）:
   - 使用 ReqQueue 生成新批次
   - 合并到当前批次
   - 执行批量推理
   - 生成批量响应
   - 更新批次状态（移除已完成的请求）

4. **`run()` 方法更新**:
   - 非阻塞接收请求（使用 asyncio.wait_for 超时）
   - 将接收到的请求添加到 ReqQueue
   - 调用 `_process_requests()` 批处理
   - 发送所有响应

**向后兼容**:
- 保留 `_process_request()` 方法用于单请求处理（向后兼容）
- 新的批处理逻辑在 `_process_requests()` 中实现

---
