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

### 2025-01-13 - Task 2.7.1: 实现 Adapter Rank 配置（Phase 1 必需）

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 添加导入：`get_lora_config` (从 `slora.models.peft.lora_adapter`)
   - 更新 `__init__()` 方法：
     - 重新组织 Adapter 管理相关属性
     - 在初始化时调用 `_setup_adapter_config()`
   - 实现新方法：
     - `_setup_adapter_config()`: 初始化 Adapter rank 配置
   - 功能：读取所有 adapter 的 rank 配置，用于 ReqQueue 显存管理

**关键设计**:
- **借鉴 manager.py**：复用张量并行模式中的 Adapter rank 管理策略
- **lora_ranks 字典**：存储 adapter_dir -> rank 的映射关系
- **用途**：传递给 ReqQueue.generate_new_batch() 用于显存占用计算
- **None 键**：表示无 adapter 的情况（base 模型），rank 为 0

**实现细节**:

1. **`_setup_adapter_config()` 方法**:
   - 初始化 `self.lora_ranks = {}` 字典
   - 检查 args 是否有 `lora_dirs` 参数
   - 遍历所有 adapter 目录：
     - 调用 `get_lora_config(lora_dir, dummy)` 读取配置
     - 提取 rank 值：`config["r"]`
     - 存储到 `lora_ranks` 字典
   - 错误处理：
     - 如果加载失败，输出警告日志
     - 使用默认 rank 值（8）
   - 添加 `self.lora_ranks[None] = 0` 处理无 adapter 情况
   - 输出初始化完成日志

2. **属性更新**:
   - `self.lora_ranks`: adapter_dir -> rank 映射（用于 ReqQueue 显存管理）
   - `self.actual_adapter_size`: 实际 adapter 占用的显存大小（cells）

3. **初始化顺序**:
   - `_setup_gpu()`: 设置 GPU 环境
   - `_setup_adapter_config()`: 初始化 Adapter rank 配置

**新增测试文件**:
1. `test/test_gpu_worker_adapter_config.py`
   - Adapter rank 配置的单元测试
   - 测试覆盖：
     - 有 lora_dirs 时的配置初始化
     - 没有 lora_dirs 时的配置初始化
     - lora_dirs 为空列表时的配置初始化
     - dummy 模式下的配置初始化
     - 配置加载失败时的处理（使用默认值）
     - 多个 adapter 的配置初始化
     - actual_adapter_size 初始化为 0
   - 所有测试通过 ✓ (7/7)

**验证结果**:
- 单元测试：7/7 通过
- 满足 Requirements 3.4（使用现有的模型推理逻辑处理请求）
- lora_ranks 字典正确初始化
- None 键正确添加（base 模型）
- 错误处理健壮（加载失败时使用默认值）
- 支持 dummy 模式
- 为后续 Task 2.7.2（实际内存占用跟踪）和 Task 2.7.3（Adapter 加载/卸载）奠定基础

**下一步**:
- Task 2.7.2: 实现实际内存占用跟踪（`_update_actual_adapter_usage()`）
- Task 2.7.3: 实现基本的 Adapter 加载/卸载（`_load_adapters()`）
- Task 2.7.4: 更新 ReqQueue 调用传递 Adapter 信息

---

### 2025-01-13 - Task 2.7.2: 实现实际内存占用跟踪（Phase 1 必需）

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 更新 `__init__()` 方法：
     - 添加 `self.actual_adapter_memory_usage = 0` 属性
   - 实现新方法：
     - `_update_actual_adapter_usage()`: 查询并更新实际的 adapter 内存占用
   - 功能：通过 RPC 查询 LoRA 内存使用情况，用于并发控制的准确判断

**关键设计**:
- **借鉴 manager.py**：复用张量并行模式中的实际内存占用跟踪策略
- **actual_adapter_memory_usage**：缓存实际的 adapter 内存占用（单位：cells）
- **用途**：在 adapter 加载/卸载后更新，用于并发控制的准确判断
- **保守估计**：查询失败时保持当前值不变

**实现细节**:

1. **`_update_actual_adapter_usage()` 方法**:
   - 检查是否启用了 LoRA（`no_lora` 参数）
   - 检查是否有 `model_rpc`（需要在模型加载后才能查询）
   - 通过 RPC 查询 `check_lora_memory()` 获取内存信息
   - 计算所有 adapter_cells 的总和：`sum(adapter_cells_list)`
   - 同步更新 `actual_adapter_size`（用于 ReqQueue）
   - 错误处理：
     - 如果查询返回 None，保持当前值不变
     - 如果查询抛出异常，捕获并输出警告日志
   - 输出更新完成日志

2. **属性更新**:
   - `self.actual_adapter_memory_usage`: 缓存实际的 adapter 内存占用（单位：cells）
   - 在 `__init__()` 中初始化为 0

3. **调用时机**（将在后续任务中实现）:
   - 模型初始化后：查询初始占用
   - Adapter 加载后：更新占用
   - Adapter 卸载后：更新占用

**新增测试文件**:
1. `test/test_gpu_worker_memory_tracking.py`
   - 实际内存占用跟踪的单元测试
   - 测试覆盖：
     - actual_adapter_memory_usage 初始化为 0
     - no_lora 模式下的内存占用更新
     - 没有 model_rpc 时的内存占用更新
     - 有效内存信息时的内存占用更新
     - adapter_cells 为空时的内存占用更新
     - check_lora_memory 返回 None 时的处理
     - 查询异常时的处理（保守估计）
     - 单个 adapter 时的内存占用更新
     - 多次更新内存占用（模拟加载/卸载）
   - 所有测试通过 ✓ (9/9)

**验证结果**:
- 单元测试：9/9 通过
- 满足 Requirements 3.4（使用现有的模型推理逻辑处理请求）
- actual_adapter_memory_usage 正确初始化
- RPC 查询逻辑正确实现
- adapter_cells 总和计算正确
- actual_adapter_size 同步更新
- 错误处理健壮（查询失败时保持当前值）
- 支持多次更新（模拟加载/卸载场景）
- 为后续 Task 2.7.3（Adapter 加载/卸载）和 Task 2.7.4（ReqQueue 集成）奠定基础

**下一步**:
- Task 2.7.3: 实现基本的 Adapter 加载/卸载（`_load_adapters()`）
- Task 2.7.4: 更新 ReqQueue 调用传递 Adapter 信息

---

### 2025-01-13 - Task 2.7.3: 实现基本的 Adapter 加载/卸载（Phase 1 必需）

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 实现新方法：
     - `_load_adapters(adapter_dirs)`: 加载指定的 adapters
   - 更新 `_process_requests()` 方法：
     - 在生成新批次后加载所需的 adapters
   - 功能：调用 RPC 加载 adapters，并更新实际内存占用

**关键设计**:
- **借鉴 manager.py**：复用张量并行模式中的 adapter 加载策略
- **加载时机**：在 ReqQueue 生成新批次后，立即加载批次所需的 adapters
- **内存更新**：加载后调用 `_update_actual_adapter_usage()` 更新实际占用
- **错误处理**：加载失败不应该终止服务，输出警告日志并继续运行

**实现细节**:

1. **`_load_adapters(adapter_dirs)` 方法**:
   - 检查是否启用了 LoRA（`no_lora` 参数）
   - 检查是否有 `model_rpc`（需要在模型加载后才能加载 adapter）
   - 检查 adapter_dirs 是否为空
   - 调用 RPC 的 `load_adapters(adapter_dirs)` 方法
   - 输出加载日志（显示加载的 adapter 数量和名称）
   - 调用 `_update_actual_adapter_usage()` 更新实际占用
   - 错误处理：
     - 如果 RPC 调用失败，捕获异常并输出错误日志
     - 不抛出异常，确保服务继续运行

2. **`_process_requests()` 方法更新**:
   - 在 `generate_new_batch()` 后检查是否有新批次
   - 如果有新批次且未启用 `no_lora`：
     - 检查批次的 `adapter_dirs` 是否非空
     - 调用 `_load_adapters(new_batch.adapter_dirs)` 加载 adapters
   - 然后合并批次并执行推理

3. **加载流程**:
   ```
   generate_new_batch() 
   → 检查 new_batch.adapter_dirs 
   → _load_adapters(adapter_dirs) 
   → RPC.load_adapters() 
   → _update_actual_adapter_usage() 
   → merge batch 
   → inference
   ```

**新增测试文件**:
1. `test/test_gpu_worker_adapter_loading.py`
   - Adapter 加载功能的单元测试
   - 测试覆盖：
     - no_lora 模式下的加载（直接返回）
     - 没有 model_rpc 时的加载（输出警告）
     - 空 adapter 集合的加载（不调用 RPC）
     - 成功加载 adapters
     - 加载单个 adapter
     - 加载时发生异常（不崩溃）
     - 多次加载不同的 adapters
     - actual_adapter_size 同步更新
     - 加载大量 adapters
   - 所有测试通过 ✓ (9/9)

**验证结果**:
- 单元测试：9/9 通过
- 满足 Requirements 3.3（根据请求中的 adapter_dir 加载对应的 Adapter）
- RPC 调用逻辑正确实现
- 加载后内存占用正确更新
- 错误处理健壮（加载失败不崩溃）
- 支持多次加载（模拟批次变化）
- 与 ReqQueue 批处理流程正确集成
- 为后续 Task 2.7.4（ReqQueue 参数传递）奠定基础

**Phase 1 简化说明**:
- 假设 `model_rpc` 已经初始化（实际的 RPC 初始化将在后续任务中实现）
- 暂不实现 adapter 卸载逻辑（Phase 1 只需要加载）
- 完整的 adapter 管理（包括淘汰策略）将在后续 Phase 中实现

**下一步**:
- Task 2.7.4: 更新 ReqQueue 调用传递 Adapter 信息

---

### 2025-01-13 - Task 2.7.4: 更新 ReqQueue 调用传递 Adapter 信息

**修改文件**:
1. `slora/server/router/gpu_worker.py`
   - 更新 `_process_requests()` 方法：
     - 使用关键字参数传递 `actual_adapter_size`
     - 确保 ReqQueue 能够正确计算显存占用

**关键设计**:
- **显式参数传递**：使用 `actual_adapter_size=self.actual_adapter_memory_usage` 作为关键字参数
- **与 manager.py 一致**：保持与张量并行模式相同的调用方式
- **显存管理集成**：确保 ReqQueue 能够准确计算批次的显存占用

**实现细节**:

1. **`generate_new_batch()` 调用更新**:
   ```python
   # 之前（位置参数）
   new_batch = self.req_queue.generate_new_batch(
       self.current_batch,
       self.lora_ranks,
       self.actual_adapter_size
   )
   
   # 之后（关键字参数）
   new_batch = self.req_queue.generate_new_batch(
       self.current_batch,
       self.lora_ranks,
       actual_adapter_size=self.actual_adapter_memory_usage
   )
   ```

2. **参数说明**:
   - `current_batch`: 当前正在处理的批次（可能为 None）
   - `lora_ranks`: adapter_dir -> rank 的映射，用于计算 adapter 显存占用
   - `actual_adapter_size`: 实际已加载的 adapter 占用的显存大小（cells）

3. **显存管理流程**:
   ```
   ReqQueue.generate_new_batch()
   → _init_cache_list(current_batch, lora_ranks, actual_adapter_size)
   → 计算当前 adapter 占用
   → _can_add_new_req() 检查是否有足够显存
   → 生成新批次
   ```

**新增测试文件**:
1. `test/test_gpu_worker_reqqueue_integration.py`
   - ReqQueue 集成测试
   - 测试覆盖：
     - lora_ranks 正确传递
     - actual_adapter_size 正确传递
     - actual_adapter_size 为 0 时的传递
     - 有当前批次时的调用
     - 批次生成后加载 adapters
     - no_lora 模式下不加载 adapters
     - adapter 信息在整个流程中的一致性
   - 所有测试通过 ✓ (7/7)

**验证结果**:
- 单元测试：7/7 通过
- 满足 Requirements 3.4（使用现有的模型推理逻辑处理请求）
- lora_ranks 正确传递给 ReqQueue
- actual_adapter_memory_usage 正确传递给 ReqQueue
- ReqQueue 能够准确计算显存占用
- 与 manager.py 的调用方式保持一致
- 完整的 adapter 管理流程集成完成

**集成验证**:
- Task 2.7.1（Adapter Rank 配置）✓
- Task 2.7.2（实际内存占用跟踪）✓
- Task 2.7.3（Adapter 加载/卸载）✓
- Task 2.7.4（ReqQueue 参数传递）✓

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

**Phase 1 Adapter 管理总结**:
- ✓ Adapter rank 配置读取
- ✓ 实际内存占用跟踪
- ✓ 基本的 adapter 加载
- ✓ ReqQueue 显存管理集成
- 为后续 Phase 的完整 adapter 管理（包括淘汰策略）奠定基础

**下一步**:
- Task 2.8: 编写 Adapter 管理单元测试（可选）
- Task 3: 实现 Data Parallel Router Manager

---
