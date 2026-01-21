# Implementation Plan: Phase 1 数据并行基础框架

## Overview

本实施计划将 Phase 1 的设计转换为具体的编码任务。任务按照依赖关系组织，确保每个任务都可以在前置任务完成后独立执行。

## 实施规范

**开发环境**：
- Conda 环境：`conda activate slora`
- 所有开发和测试都在此环境下进行

**代码修改记录**：
- 完成每个任务后，将代码修改记录统一更新到 `/home/hzheng/S-LoRA/Phase1-代码修改记录.md`
- 记录内容包括：新增/修改的文件、代码功能简介
- 不创建额外的说明文档

**测试要求**：
- 每个任务完成后必须编写并运行单元测试
- 测试通过后才能标记任务为完成状态
- 测试命令：`pytest test/` 或 `pytest <specific_test_file>`

## 当前状态总结

**已完成的核心组件**（Tasks 1-7）:
- ✅ Round Robin Router（轮询路由器）
- ✅ GPU Worker 完整实现（ZMQ 通信、ReqQueue 集成、Adapter 管理、推理逻辑）
- ✅ DataParallelRouterManager（Worker 管理、请求路由、进程监控）
- ✅ Response Merger（响应收集与转发）
- ✅ API Server 集成（命令行参数、模式选择、启动日志）
- ✅ 错误处理（Worker 启动失败、推理异常、ZMQ 超时、进程监控）
- ✅ 基础监控（启动日志、请求统计、调试日志）
- ✅ 单元测试覆盖（20+ 个测试文件，覆盖核心功能）

**Task 8 进度**（基础功能验证 - Checkpoint）:
- ✅ 8.1: 准备测试环境（测试框架、环境验证）
- ✅ 8.2.1: 测试 Round Robin Router
- ✅ 8.2.2: 测试 GPU Worker 初始化
- ✅ 8.2.3: 测试 DataParallelRouterManager 初始化
- ✅ 8.3.1: 测试单个 Worker 进程启动
- ✅ 8.3.2: 测试多个 Worker 进程启动
- ✅ 8.3.3: 测试 Response Merger 进程启动
- ✅ 8.4.1: 测试 Manager → Worker 通信
- ✅ 8.4.2: 测试 Worker → Merger 通信
- ✅ 8.4.3: 测试完整通信链路
- ✅ 8.5.1: 测试单个请求处理（1 Worker）
- ✅ 8.5.2: 测试多个串行请求（1 Worker）
- ✅ 8.5.3: 测试并发请求（2 Workers）
- ✅ 8.5.4: 测试高并发（4 Workers）
- ✅ 8.6.1: 测试 Worker 启动失败
- ✅ 8.6.2: 测试 Worker 进程崩溃
- ✅ 8.6.3: 测试推理异常
- ⏳ 8.7: 监控和日志测试（待完成）
- ⏳ 8.8: 总结和报告（待完成）

**待实现的组件**（Tasks 9-12）:
- ⏳ 集成测试（端到端测试）
- ⏳ 性能测试（吞吐量、延迟基准）
- ⏳ 文档和示例
- ⏳ 最终验收

## Tasks

- [x] 1. 创建 Round Robin Router
  - 实现轮询路由逻辑
  - 支持任意数量的 Worker
  - _Requirements: 2.1, 2.2_

- [x]* 1.1 编写 Round Robin Router 单元测试
  - **Property 2: 轮询路由公平性**
  - **Validates: Requirements 2.1, 2.2**
  - 测试轮询顺序正确性
  - 测试不同 Worker 数量下的行为

- [x] 2. 实现 GPU Worker 基础框架
  - [x] 2.1 创建 GPUWorker 类框架
    - 实现 `__init__` 方法
    - 实现 GPU 环境设置
    - _Requirements: 1.2, 1.3_
  
  - [x] 2.2 实现 ZMQ 通信设置
    - 创建 PULL socket 接收请求
    - 创建 PUSH socket 发送响应
    - _Requirements: 2.3, 4.1_
  
  - [x] 2.3 集成模型加载逻辑
    - 复用 `slora/common/basemodel/` 代码
    - 在指定 GPU 上加载模型
    - _Requirements: 1.4_
  
  - [x] 2.4 集成 ReqQueue 请求管理
    - **关键设计**：复用张量并行的 `ReqQueue` 进行请求管理
    - 实现 `_setup_request_queue()` 方法
    - 导入并初始化 `ReqQueue`（来自 `slora/server/router/req_queue.py`）
    - 配置 `max_total_tokens`, `batch_max_tokens`, `running_max_req_size`
    - 实现 `_convert_to_req_object()` 方法（转换 ZMQ 消息为 Req 对象）
    - 更新 `_process_request()` 为 `_process_requests()` 使用批处理
    - _Requirements: 3.1, 3.2, 3.4_
    - _复用组件_: `ReqQueue.append()`, `ReqQueue.generate_new_batch()`
  
  - [x] 2.5 实现请求处理循环（基础版本）
    - 实现 `_receive_request()` 方法（接收 ZMQ 消息）
    - 实现 `_process_request()` 方法（单请求处理）
    - 实现 `_send_response()` 方法
    - 实现 `run()` 主循环
    - _Requirements: 3.1, 3.2, 3.5_
    - _注意_: 完整的批处理逻辑将在 Task 2.4 中实现

- [x]* 2.6 编写 GPU Worker 单元测试
  - 测试 GPU 环境设置
  - 测试 ZMQ socket 创建
  - 测试消息接收和发送
  - 测试请求处理流程

- [x] 2.7 实现 Adapter 管理（借鉴 manager.py）
  - **设计理念**：复用张量并行的 Adapter 管理策略
  
  - [x] 2.7.1 实现 Adapter Rank 配置（Phase 1 必需）
    - 导入 `get_lora_config` 函数
    - 在 `__init__` 中初始化 `self.lora_ranks` 字典
    - 遍历所有 adapter 目录，读取配置并存储 rank
    - 添加 `self.lora_ranks[None] = 0` 处理无 adapter 情况
    - _借鉴_: `manager.py` 的 lora_ranks 初始化逻辑
    - _Requirements: 3.4_
  
  - [x] 2.7.2 实现实际内存占用跟踪（Phase 1 必需）
    - 在 `__init__` 中初始化 `self.actual_adapter_memory_usage = 0`
    - 实现 `_update_actual_adapter_usage()` 方法
    - 通过 RPC 查询 `check_lora_memory()` 获取实际占用
    - 计算所有 adapter_cells 的总和
    - _借鉴_: `manager.py` 的 `_update_actual_adapter_usage()` 方法
    - _Requirements: 3.4_
  
  - [x] 2.7.3 实现基本的 Adapter 加载/卸载（Phase 1 必需）
    - 实现 `_load_adapters(adapter_dirs)` 方法
    - 调用 RPC 的 `load_adapters()` 加载 adapters
    - 加载后调用 `_update_actual_adapter_usage()` 更新占用
    - 在批次生成后加载所需的 adapters
    - _借鉴_: `manager.py` 的 adapter 加载逻辑
    - _Requirements: 3.3_
  
  - [x] 2.7.4 更新 ReqQueue 调用传递 Adapter 信息
    - 在 `generate_new_batch()` 调用中传递 `lora_ranks` 参数
    - 在 `generate_new_batch()` 调用中传递 `actual_adapter_size` 参数
    - 确保 ReqQueue 能够正确计算显存占用
    - _Requirements: 3.4_

- [x] 2. 实现 GPU Worker 基础框架
  - [x] 2.1 创建 GPUWorker 类框架
    - 实现 `__init__` 方法
    - 实现 GPU 环境设置
    - _Requirements: 1.2, 1.3_
  
  - [x] 2.2 实现 ZMQ 通信设置
    - 创建 PULL socket 接收请求
    - 创建 PUSH socket 发送响应
    - _Requirements: 2.3, 4.1_
  
  - [x] 2.3 集成模型加载逻辑
    - 复用 `slora/common/basemodel/` 代码
    - 在指定 GPU 上加载模型
    - _Requirements: 1.4_
  
  - [x] 2.4 集成 ReqQueue 请求管理
    - **关键设计**：复用张量并行的 `ReqQueue` 进行请求管理
    - 实现 `_setup_request_queue()` 方法
    - 导入并初始化 `ReqQueue`（来自 `slora/server/router/req_queue.py`）
    - 配置 `max_total_tokens`, `batch_max_tokens`, `running_max_req_size`
    - 实现 `_convert_to_req_object()` 方法（转换 ZMQ 消息为 Req 对象）
    - 更新 `_process_request()` 为 `_process_requests()` 使用批处理
    - _Requirements: 3.1, 3.2, 3.4_
    - _复用组件_: `ReqQueue.append()`, `ReqQueue.generate_new_batch()`
  
  - [x] 2.5 实现请求处理循环（基础版本）
    - 实现 `_receive_request()` 方法（接收 ZMQ 消息）
    - 实现 `_process_request()` 方法（单请求处理）
    - 实现 `_send_response()` 方法
    - 实现 `run()` 主循环
    - _Requirements: 3.1, 3.2, 3.5_
    - _注意_: 完整的批处理逻辑将在 Task 2.4 中实现

- [x]* 2.6 编写 GPU Worker 单元测试
  - 测试 GPU 环境设置
  - 测试 ZMQ socket 创建
  - 测试消息接收和发送
  - 测试请求处理流程

- [x] 2.7 实现 Adapter 管理（借鉴 manager.py）
  - **设计理念**：复用张量并行的 Adapter 管理策略
  
  - [x] 2.7.1 实现 Adapter Rank 配置（Phase 1 必需）
    - 导入 `get_lora_config` 函数
    - 在 `__init__` 中初始化 `self.lora_ranks` 字典
    - 遍历所有 adapter 目录，读取配置并存储 rank
    - 添加 `self.lora_ranks[None] = 0` 处理无 adapter 情况
    - _借鉴_: `manager.py` 的 lora_ranks 初始化逻辑
    - _Requirements: 3.4_
  
  - [x] 2.7.2 实现实际内存占用跟踪（Phase 1 必需）
    - 在 `__init__` 中初始化 `self.actual_adapter_memory_usage = 0`
    - 实现 `_update_actual_adapter_usage()` 方法
    - 通过 RPC 查询 `check_lora_memory()` 获取实际占用
    - 计算所有 adapter_cells 的总和
    - _借鉴_: `manager.py` 的 `_update_actual_adapter_usage()` 方法
    - _Requirements: 3.4_
  
  - [x] 2.7.3 实现基本的 Adapter 加载/卸载（Phase 1 必需）
    - 实现 `_load_adapters(adapter_dirs)` 方法
    - 调用 RPC 的 `load_adapters()` 加载 adapters
    - 加载后调用 `_update_actual_adapter_usage()` 更新占用
    - 在批次生成后加载所需的 adapters
    - _借鉴_: `manager.py` 的 adapter 加载逻辑
    - _Requirements: 3.3_
  
  - [x] 2.7.4 更新 ReqQueue 调用传递 Adapter 信息
    - 在 `generate_new_batch()` 调用中传递 `lora_ranks` 参数
    - 在 `generate_new_batch()` 调用中传递 `actual_adapter_size` 参数
    - 确保 ReqQueue 能够正确计算显存占用
    - _Requirements: 3.4_

- [x] 2.9 完善 GPU Worker 推理逻辑
  - [x] 2.9.1 实现模型 RPC 初始化
    - 添加 `_init_model_rpc()` 方法
    - 创建 ModelRpcClient 连接到模型进程
    - 在 Worker 启动时初始化 RPC 连接
    - _Requirements: 1.4, 3.4_
  
  - [x] 2.9.2 实现实际推理逻辑
    - 实现 `_infer_batch()` 方法
    - 调用 model_rpc 执行实际推理
    - 处理推理结果并更新批次状态
    - _Requirements: 3.4, 3.5_
  
  - [x] 2.9.3 集成完整的批次管理
    - 实现 `_handle_finish_req()` 方法（完整版）
    - 处理 EOS token 检测
    - 更新批次状态和 adapter 使用统计
    - _Requirements: 3.5_

- [x]* 2.10 编写 GPU Worker 推理测试
  - 测试模型 RPC 初始化
  - 测试实际推理流程
  - 测试批次管理逻辑

- [x]* 2.11 编写 Adapter 管理单元测试
  - 测试 lora_ranks 初始化
  - 测试实际内存占用查询
  - 测试 adapter 加载流程
  - 测试 ReqQueue 与 adapter 信息的集成

- [x] 3.  DataParallelRouterManager
  - [x] 3.1 创建 DataParallelRouterManager 类框架
    - 创建 `slora/server/router/dp_manager.py` 文件
    - 实现 `__init__` 方法
    - 实现 GPU 检测逻辑 `_detect_gpus()`
    - 实现 GPU ID 解析 `_parse_gpu_ids()`
    - 实现端口分配 `_allocate_ports()`
    - _Requirements: 1.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 3.2 实现 Worker 进程管理
    - 实现 `_start_worker()` 方法（启动单个 Worker 进程）
    - 实现 `start_workers()` 方法（启动所有 Worker）
    - 为每个 Worker 分配唯一的端口
    - 等待所有 Worker 就绪
    - _Requirements: 1.1, 1.2, 1.5_
  
  - [x] 3.3 实现 ZMQ 通信设置
    - 实现 `_setup_zmq()` 方法
    - 创建 PULL socket 接收来自 API Server 的请求
    - 为每个 Worker 创建 PUSH socket
    - _Requirements: 2.3_
  
  - [x] 3.4 实现请求路由逻辑
    - 集成 Round Robin Router
    - 实现 `route_request()` 方法
    - 根据路由器选择 Worker 并发送请求
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 3.5 实现主循环
    - 实现 `run()` 方法
    - 持续接收来自 API Server 的请求
    - 调用路由器选择 Worker
    - 发送请求到选定的 Worker
    - _Requirements: 2.1, 2.3_

- [x]* 3.6 编写 Router Manager 单元测试
  - 测试 GPU 检测逻辑
  - 测试 GPU ID 解析
  - 测试端口分配
  - 测试 Worker 启动
  - 测试请求路由
  - **Property 1: Worker 启动完整性**
  - **Validates: Requirements 1.1, 1.5**

- [x] 4. 实现 Response Merger
  - [x] 4.1 创建 ResponseMerger 类
    - 创建 `slora/server/router/response_merger.py` 文件
    - 实现 `__init__` 方法
    - 设置 ZMQ PULL socket（接收 Worker 响应）
    - 设置 ZMQ PUSH socket（发送到 Detokenization）
    - _Requirements: 4.1, 4.2_
  
  - [x] 4.2 实现响应转发逻辑
    - 实现 `_convert_to_detoken_format()` 方法
    - 实现 `_forward_to_detokenization()` 方法
    - 实现 `run()` 主循环
    - _Requirements: 4.3, 4.4, 4.5_
  
  - [x] 4.3 集成到 Router Manager
    - 在 `dp_manager.py` 中启动 Response Merger 进程
    - 确保 Worker 响应正确转发到 Detokenization
    - _Requirements: 4.1, 4.2_

- [ ]* 4.4 编写 Response Merger 单元测试
  - 测试消息格式转换
  - 测试响应转发
  - 测试与 Detokenization 的集成

- [x] 5. 修改 API Server 入口
  - [x] 5.1 添加命令行参数
    - 在 `slora/server/api_server.py` 中添加 `--parallel-mode` 参数
    - 添加 `--num-workers` 参数
    - 添加 `--gpu-ids` 参数
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [x] 5.2 实现模式选择逻辑
    - 修改 `start_router_process` 函数或创建新的启动函数
    - 根据 `parallel-mode` 选择启动逻辑：
      - `data`: 启动 DataParallelRouterManager
      - `tensor` 或默认: 使用原有的张量并行逻辑
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [x] 5.3 添加启动日志
    - 输出当前并行模式
    - 输出 Worker 数量和 GPU 列表
    - _Requirements: 6.5, 8.1_
  
  - [x] 5.4 更新 router/manager.py 或创建新的入口
    - 确保数据并行模式和张量并行模式可以共存
    - 保持向后兼容性
    - _Requirements: 6.1, 6.2, 6.4_

- [ ]* 5.5 编写 API Server 集成测试
  - 测试命令行参数解析
  - 测试模式选择逻辑
  - 测试启动日志输出

- [x] 6. 完善错误处理
  - [x] 6.1 Worker 启动失败处理
    - 在 `run_gpu_worker_process()` 中添加更完善的异常捕获
    - 记录详细错误日志和堆栈跟踪
    - 通过退出码通知 Router Manager
    - 在 Router Manager 中检测 Worker 启动失败
    - 实现方式：
      - 在 `run_gpu_worker_process()` 中捕获所有异常
      - 使用 `sys.exit(1)` 返回非零退出码
      - 在 `dp_manager.start_workers()` 中检查进程状态
      - 如果任何 Worker 启动失败，终止所有进程并退出
    - _Requirements: 7.1, 7.2_
  
  - [x] 6.2 推理异常处理（验证和完善）
    - 验证 Worker 中的推理异常捕获是否完整
    - 确保返回错误响应格式正确
    - 添加更详细的错误日志
    - 测试各种异常场景（CUDA OOM、模型错误等）
    - 实现方式：
      - 检查 `_infer_batch()` 和 `_process_requests()` 的异常处理
      - 确保所有异常都被捕获并转换为错误响应
      - 添加错误类型分类（OOM、模型错误、超时等）
      - 编写测试用例模拟各种异常场景
    - _Requirements: 7.3_
  
  - [x] 6.3 ZMQ 通信超时处理
    - 在 Router Manager 中设置 socket 超时
    - 在 Worker 中设置 socket 超时
    - 实现重试逻辑（最多 3 次）
    - 记录超时错误日志
    - 实现方式：
      - 使用 `socket.setsockopt(zmq.RCVTIMEO, 30000)` 设置接收超时
      - 使用 `socket.setsockopt(zmq.SNDTIMEO, 30000)` 设置发送超时
      - 在超时时捕获 `zmq.Again` 异常
      - 实现重试循环，记录每次重试
      - 超过重试次数后返回错误响应
    - _Requirements: 7.4_
  
  - [x] 6.4 Worker 进程监控
    - 在 Router Manager 中实现进程状态检查
    - 定期检测 Worker 进程是否存活
    - 检测 Worker 进程意外退出
    - 记录进程退出日志
    - 实现方式：
      - 在 `dp_manager.run()` 主循环中添加进程检查
      - 使用 `process.is_alive()` 检查进程状态
      - 使用 `process.exitcode` 获取退出码
      - 每 10 秒检查一次所有 Worker 进程
      - 发现进程退出时记录详细日志（worker_id、exitcode、时间）
    - _Requirements: 7.5_

- [ ]* 6.5 编写错误处理测试
  - 测试 Worker 启动失败场景
  - 测试推理异常处理
  - 测试 ZMQ 通信超时
  - 测试 Worker 进程崩溃恢复

- [x] 7. 完善基础监控
  - [x] 7.1 验证和完善启动日志
    - 确认 Worker 就绪日志已输出（已实现）
    - 确认 Router Manager 启动日志已输出（已实现）
    - 添加更详细的启动信息（GPU 型号、内存等）
    - 添加模型加载进度日志
    - 实现方式：
      - 在 Worker 启动时使用 `torch.cuda.get_device_properties()` 获取 GPU 信息
      - 输出 GPU 名称、总内存、计算能力
      - 在模型加载过程中添加进度日志
      - 输出模型大小、加载时间等信息
    - _Requirements: 8.1, 8.3_
  
  - [x] 7.2 实现请求统计
    - 在 Router Manager 中添加统计计数器
    - 统计总请求数、成功数、失败数
    - 每 10 秒输出统计信息
    - 添加每个 Worker 的请求分布统计
    - 添加平均延迟和吞吐量统计
    - 实现方式：
      - 在 `dp_manager.py` 中添加 `self.stats` 字典
      - 记录：total_requests, successful_requests, failed_requests
      - 记录每个 Worker 的请求计数：worker_request_counts
      - 使用 asyncio.create_task 创建后台统计任务
      - 每 10 秒输出统计摘要
    - _Requirements: 8.2_
  
  - [x] 7.3 验证和完善调试日志
    - 确认路由决策日志已输出（已实现）
    - 添加响应返回日志（DEBUG 级别）
    - 添加批次处理日志（DEBUG 级别）
    - 添加 Adapter 加载/卸载日志
    - 实现方式：
      - 在 `route_request()` 中已有 DEBUG 日志
      - 在 Response Merger 中添加响应接收日志
      - 在 Worker 的 `_process_requests()` 中添加批次信息日志
      - 在 `_load_adapters()` 中添加加载详情日志
      - 使用 Python logging 模块，设置 DEBUG 级别
    - _Requirements: 8.4, 8.5_

- [ ]* 7.4 编写监控测试
  - 测试启动日志输出
  - 测试请求统计功能
  - 测试调试日志输出

- [ ] 8. Checkpoint - 基础功能验证
  - **目标**: 阶段性验证所有核心组件能够正常工作
  - **方式**: 从简单到复杂，逐步验证每个组件
  
  - [x] 8.1 准备测试环境
    - [x] 8.1.1 确认开发环境
      - 激活 conda 环境：`conda activate slora`
      - 检查 Python 版本和依赖
      - 确认 GPU 可用：`nvidia-smi`
    
    - [x] 8.1.2 创建测试脚本目录
      - 创建 `test/phase1_manual/` 目录
      - 创建基础测试脚本框架
  
  - [x] 8.2 组件级测试（从底层到上层）
    
    - [x] 8.2.1 测试 Round Robin Router（最简单）
      - 创建 `test/phase1_manual/test_router.py`
      - 测试内容：
        - 创建 Router 实例
        - 测试 `select_worker()` 方法
        - 验证轮询顺序：0, 1, 2, 0, 1, 2...
        - 验证不同 Worker 数量
      - 预期结果：轮询顺序正确
      - 运行：`python test/phase1_manual/test_router.py`
    
    - [x] 8.2.2 测试 GPU Worker 初始化（不启动推理）
      - 创建 `test/phase1_manual/test_worker_init.py`
      - 测试内容：
        - 创建 GPUWorker 实例
        - 测试 GPU 环境设置
        - 测试 ZMQ socket 创建
        - 测试 Adapter 配置初始化
      - 预期结果：Worker 初始化成功，无异常
      - 运行：`python test/phase1_manual/test_worker_init.py`
    
    - [x] 8.2.3 测试 DataParallelRouterManager 初始化
      - 创建 `test/phase1_manual/test_manager_init.py`
      - 测试内容：
        - 创建 DataParallelRouterManager 实例
        - 测试 GPU 检测
        - 测试端口分配
        - 测试 ZMQ socket 创建
      - 预期结果：Manager 初始化成功
      - 运行：`python test/phase1_manual/test_manager_init.py`
  
  - [x] 8.3 进程级测试（启动但不发送请求）
    
    - [x] 8.3.1 测试单个 Worker 进程启动
      - 创建 `test/phase1_manual/test_single_worker_startup.py`
      - 测试内容：
        - 启动 1 个 Worker 进程（Dummy 模式）
        - 检查进程是否存活
        - 检查启动日志输出
        - 等待 5 秒后检查进程状态
        - 优雅关闭进程
      - 预期结果：Worker 进程启动成功，输出"Worker 0 ready"
      - 运行：`python test/phase1_manual/test_single_worker_startup.py`
    
    - [x] 8.3.2 测试多个 Worker 进程启动
      - 创建 `test/phase1_manual/test_multi_worker_startup.py`
      - 测试内容：
        - 启动 2 个 Worker 进程（Dummy 模式）
        - 检查所有进程是否存活
        - 检查启动日志（每个 Worker 的就绪消息）
        - 验证 GPU 隔离（每个 Worker 使用不同 GPU）
        - 优雅关闭所有进程
      - 预期结果：所有 Worker 启动成功，GPU 正确分配
      - 运行：`python test/phase1_manual/test_multi_worker_startup.py`
    
    - [x] 8.3.3 测试 Response Merger 进程启动
      - 创建 `test/phase1_manual/test_merger_startup.py`
      - 测试内容：
        - 启动 Response Merger 进程
        - 检查进程是否存活
        - 检查 ZMQ socket 绑定
        - 优雅关闭进程
      - 预期结果：Response Merger 启动成功
      - 运行：`python test/phase1_manual/test_merger_startup.py`
  
  - [x] 8.4 通信级测试（测试 ZMQ 消息传递）
    
    - [x] 8.4.1 测试 Manager → Worker 通信
      - 创建 `test/phase1_manual/test_manager_to_worker.py`
      - 测试内容：
        - 启动 Manager 和 1 个 Worker
        - Manager 发送测试消息到 Worker
        - Worker 接收并打印消息
        - 验证消息内容正确
      - 预期结果：消息成功传递，内容正确
      - 运行：`python test/phase1_manual/test_manager_to_worker.py`
    
    - [x] 8.4.2 测试 Worker → Merger 通信
      - 创建 `test/phase1_manual/test_worker_to_merger.py`
      - 测试内容：
        - 启动 1 个 Worker 和 Response Merger
        - Worker 发送测试响应到 Merger
        - Merger 接收并打印响应
        - 验证响应内容正确
      - 预期结果：响应成功传递，内容正确
      - 运行：`python test/phase1_manual/test_worker_to_merger.py`
    
    - [x] 8.4.3 测试完整通信链路
      - 创建 `test/phase1_manual/test_full_communication.py`
      - 测试内容：
        - 启动 Manager、2 个 Worker、Response Merger
        - Manager 发送 10 个测试请求
        - 验证请求轮询分配到不同 Worker
        - 验证所有响应都被 Merger 接收
        - 验证响应顺序和内容
      - 预期结果：所有消息正确传递，轮询分配正确
      - 运行：`python test/phase1_manual/test_full_communication.py`
  
  - [x] 8.5 功能级测试（Dummy 模式，简化推理）
    
    - [x] 8.5.1 测试单个请求处理（1 Worker）
      - 创建 `test/phase1_manual/test_single_request.py`
      - 测试内容：
        - 启动完整系统（1 Worker，Dummy 模式）
        - 发送 1 个测试请求
        - 验证响应返回
        - 验证响应格式正确
        - 检查日志输出
      - 预期结果：请求成功处理，响应正确返回
      - 运行：`python test/phase1_manual/test_single_request.py`
    
    - [x] 8.5.2 测试多个串行请求（1 Worker）
      - 创建 `test/phase1_manual/test_serial_requests.py`
      - 测试内容：
        - 启动完整系统（1 Worker，Dummy 模式）
        - 串行发送 5 个测试请求
        - 验证所有响应返回
        - 验证响应顺序正确
        - 检查请求统计日志
      - 预期结果：所有请求成功处理，统计正确
      - 运行：`python test/phase1_manual/test_serial_requests.py`
    
    - [x] 8.5.3 测试并发请求（2 Workers）
      - 创建 `test/phase1_manual/test_concurrent_requests.py`
      - 测试内容：
        - 启动完整系统（2 Workers，Dummy 模式）
        - 并发发送 10 个测试请求
        - 验证所有响应返回
        - 验证请求分配到不同 Worker
        - 检查每个 Worker 的请求计数
      - 预期结果：请求均匀分配，所有响应正确
      - 运行：`python test/phase1_manual/test_concurrent_requests.py`
    
    - [x] 8.5.4 测试高并发（4 Workers）
      - 创建 `test/phase1_manual/test_high_concurrency.py`
      - 测试内容：
        - 启动完整系统（4 Workers，Dummy 模式）
        - 并发发送 100 个测试请求
        - 验证所有响应返回
        - 计算吞吐量和平均延迟
        - 检查 GPU 利用率
      - 预期结果：系统稳定，吞吐量符合预期
      - 运行：`python test/phase1_manual/test_high_concurrency.py`
  
  - [x] 8.6 错误处理测试
    
    - [x] 8.6.1 测试 Worker 启动失败
      - 创建 `test/phase1_manual/test_worker_startup_failure.py`
      - 测试内容：
        - 尝试启动 Worker 但指定不存在的 GPU
        - 验证 Manager 检测到启动失败
        - 验证错误日志输出
        - 验证系统优雅退出
      - 预期结果：错误被正确捕获和记录
      - 运行：`python test/phase1_manual/test_worker_startup_failure.py`
    
    - [x] 8.6.2 测试 Worker 进程崩溃
      - 创建 `test/phase1_manual/test_worker_crash.py`
      - 测试内容：
        - 启动完整系统（2 Workers）
        - 手动杀死一个 Worker 进程
        - 验证 Manager 检测到进程退出
        - 验证错误日志输出
        - 验证系统继续使用剩余 Worker
      - 预期结果：系统检测到崩溃，继续运行
      - 运行：`python test/phase1_manual/test_worker_crash.py`
    
    - [x] 8.6.3 测试推理异常
      - 创建 `test/phase1_manual/test_inference_error.py`
      - 测试内容：
        - 启动完整系统
        - 发送会导致推理失败的请求
        - 验证返回错误响应
        - 验证错误响应格式正确
        - 验证系统继续处理后续请求
      - 预期结果：错误被正确处理，系统稳定
      - 运行：`python test/phase1_manual/test_inference_error.py`
  
  - [ ] 8.7 监控和日志测试
    
    - [x] 8.7.1 验证启动日志
      - 测试内容：
        - 启动完整系统（2 Workers）
        - 检查启动日志是否包含：
          - 并行模式信息
          - Worker 数量和 GPU 列表
          - 每个 Worker 的就绪消息
          - GPU 信息（型号、内存）
      - 预期结果：所有启动日志正确输出
    
    - [x] 8.7.2 验证请求统计
      - 测试内容：
        - 启动完整系统
        - 发送 20 个请求
        - 等待 10 秒
        - 检查统计日志是否输出：
          - 总请求数
          - 成功/失败数
          - 每个 Worker 的请求分布
      - 预期结果：统计信息正确输出
    
    - [x] 8.7.3 验证调试日志
      - 测试内容：
        - 启动完整系统（DEBUG 模式）
        - 发送测试请求
        - 检查 DEBUG 日志是否包含：
          - 路由决策日志
          - 批次处理日志
          - Adapter 加载日志
          - 响应返回日志
      - 预期结果：所有 DEBUG 日志正确输出
  
  - [ ] 8.8 总结和报告
    - [x] 8.8.1 汇总测试结果
      - 记录所有测试的通过/失败状态
      - 记录发现的问题和 Bug
      - 记录性能数据（吞吐量、延迟）
    
    - [ ] 8.8.2 更新文档
      - 更新 `Phase1-代码修改记录.md`
      - 记录 Task 8 的验证结果
      - 列出需要修复的问题
    
    - [ ] 8.8.3 决定下一步
      - 如果所有测试通过 → 继续 Task 9
      - 如果有问题 → 修复后重新测试
      - 询问用户/团队意见

- [ ] 9. 集成测试
  - [ ] 9.1 端到端测试（使用实际模型）
    - 创建 `test/test_e2e_data_parallel.py`
    - 测试内容：
      - 启动完整的数据并行系统（使用实际模型，如 Llama-7B）
      - 发送真实的推理请求
      - 验证响应正确性（与张量并行模式对比）
      - 测试不同的 prompt 长度和采样参数
      - 测试 Adapter 切换功能
    - 预期结果：
      - 所有请求成功处理
      - 响应内容正确
      - Adapter 正确加载和使用
    - _Requirements: 10.1, 10.3_
  
  - [ ] 9.2 多 Worker 并发测试
    - 创建 `test/test_e2e_multi_worker_concurrent.py`
    - 测试内容：
      - 启动完整的数据并行系统（3 个 Worker）
      - 并发发送 30 个请求
      - 验证所有响应正确
      - 验证请求均匀分配到所有 Worker
      - 验证并发性能提升
    - 预期结果：
      - 所有请求成功处理
      - 请求分配均匀（每个 Worker 约 10 个请求）
      - 吞吐量显著提升（接近 3x）
    - _Requirements: 10.2_
  
  - [ ] 9.3 ZMQ 通信可靠性测试
    - 创建 `test/test_zmq_reliability.py`
    - 测试内容：
      - 测试 Router → Worker 通信可靠性
      - 测试 Worker → Response Merger 通信可靠性
      - 发送大量消息验证无丢失
      - 使用 request_id 跟踪消息
      - 验证所有消息都被接收
    - 预期结果：
      - 所有消息成功传递
      - 无消息丢失
      - 消息顺序正确（如果需要）
    - _Requirements: 10.5_
  
  - [ ] 9.4 Adapter 切换测试
    - 创建 `test/test_e2e_adapter_switching.py`
    - 测试内容：
      - 准备多个测试 Adapter
      - 交替发送使用不同 Adapter 的请求
      - 验证 Adapter 正确加载和使用
      - 验证输出结果符合预期
      - 检查内存占用统计
    - 预期结果：
      - Adapter 正确切换
      - 输出结果正确
      - 内存管理正常
    - _Requirements: 3.3, 3.4_

- [ ]* 9.5 编写 Property-Based 测试
  - **Property 1: Worker 启动完整性**
  - **Validates: Requirements 1.1, 1.5**
  - **Property 2: 轮询路由公平性**
  - **Validates: Requirements 2.1, 2.2**
  - **Property 3: 请求响应一致性**
  - **Validates: Requirements 4.3**
  - **Property 4: GPU 隔离性**
  - **Validates: Requirements 1.3**
  - **Property 5: 消息完整性**
  - **Validates: Requirements 2.4**

- [ ] 10. 性能测试
  - [ ] 10.1 编写性能测试脚本
    - 创建 `benchmarks/benchmark_phase1_data_parallel.py`
    - 实现吞吐量测试（requests/second）
    - 实现延迟测试（P50, P90, P99）
    - 实现 GPU 利用率监控
    - 支持不同 Worker 数量的对比测试
    - 实现方式：
      - 使用 `time.time()` 测量延迟
      - 使用 `asyncio` 并发发送请求测量吞吐量
      - 使用 `nvidia-smi` 或 `pynvml` 监控 GPU 利用率
      - 生成性能报告（CSV 或 JSON 格式）
    - _Requirements: 9.4_
  
  - [ ] 10.2 运行基准测试
    - 测试 1, 2, 3, 4 Workers 的性能
    - 记录吞吐量和延迟数据
    - 验证 GPU 利用率
    - 生成性能对比图表
    - 测试配置：
      - 模型：Llama-7B
      - Prompt 长度：128 tokens
      - 输出长度：128 tokens
      - 并发请求数：10, 20, 50
      - 每个配置运行 3 次取平均值
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ] 10.3 性能分析和优化（可选）
    - 分析性能瓶颈（使用 profiler）
    - 优化 ZMQ 通信（如果需要）
    - 优化消息序列化（如果需要）
    - 优化批次管理（如果需要）
    - 验证优化效果
    - 分析工具：
      - Python cProfile 或 py-spy
      - ZMQ 消息大小和频率分析
      - 批次大小和利用率分析
    - _Requirements: 9.1, 9.2_
  
  - [ ] 10.4 性能验收
    - 验证 3 Workers 吞吐量 ≥ 2.5x 单 Worker
    - 验证平均延迟增加 < 10%
    - 验证 GPU 利用率 > 85%
    - 记录性能测试结果
    - 验收标准：
      - 吞吐量：3 Workers ≥ 2.5x baseline
      - 延迟：P50 增加 < 10%, P99 增加 < 20%
      - GPU 利用率：> 85%（使用 nvidia-smi）
      - 内存占用：无明显泄漏（运行 1 小时）
    - _Requirements: 9.1, 9.2, 9.3_

- [ ] 11. 文档和示例
  - [ ] 11.1 编写快速开始指南
    - 创建 `docs/data_parallel_quickstart.md`
    - 说明如何启动数据并行模式
    - 说明配置参数（--parallel-mode, --num-workers, --gpu-ids）
    - 提供完整的启动命令示例
    - 说明如何验证系统正常工作
    - 内容包括：
      - 前置条件（GPU 数量、模型路径、依赖安装）
      - 基本启动命令
      - 参数说明和推荐配置
      - 验证方法（发送测试请求）
      - 常见问题和解决方法
  
  - [ ] 11.2 创建示例脚本
    - 创建 `examples/run_data_parallel.sh`
    - 提供不同配置的示例（1/2/3/4 Workers）
    - 提供不同 GPU 配置的示例
    - 添加注释说明每个参数的作用
    - 示例包括：
      - 单 Worker 模式（测试）
      - 多 Worker 模式（生产）
      - 指定 GPU 模式
      - 不同模型大小的配置
  
  - [ ] 11.3 编写故障排查指南
    - 创建 `docs/data_parallel_troubleshooting.md`
    - 常见错误和解决方法
    - 调试技巧（如何查看日志、如何检查进程状态）
    - 性能调优建议
    - 内容包括：
      - Worker 启动失败（GPU 不可用、内存不足）
      - ZMQ 通信错误（端口占用、超时）
      - 推理错误（CUDA OOM、模型加载失败）
      - 性能问题（吞吐量低、延迟高）
      - 日志查看方法
      - 进程监控方法
  
  - [ ] 11.4 更新主 README
    - 在主 README 中添加数据并行模式的说明
    - 添加快速开始链接
    - 添加性能对比数据
    - 内容包括：
      - 数据并行模式简介
      - 与张量并行模式的对比
      - 使用场景和优势
      - 快速开始链接
      - 性能基准数据

- [ ] 12. Final Checkpoint - Phase 1 完成验收
  - 所有功能测试通过
  - 所有集成测试通过
  - 性能目标达成（3 Workers ≥ 2.5x）
  - 文档完整且准确
  - 代码修改记录完整
  - 验收清单：
    - [ ] 核心功能：单 Worker 和多 Worker 模式都能正常工作
    - [ ] 错误处理：各种异常场景都能正确处理
    - [ ] 监控：日志和统计信息完整准确
    - [ ] 性能：达到或超过性能目标
    - [ ] 测试：单元测试和集成测试覆盖率 > 80%
    - [ ] 文档：快速开始指南、故障排查指南、示例脚本完整
  - 询问用户是否满意，是否需要调整

## Notes

- 标记 `*` 的任务为可选任务（测试相关），可以根据时间安排决定是否实施
- 每个任务都标注了对应的 Requirements，便于追溯
- Checkpoint 任务用于阶段性验证，确保及时发现问题
- Property-Based 测试使用 Hypothesis 框架
- 性能测试目标：3 Workers 吞吐量 ≥ 单 Worker 的 2.5 倍

## 实施优先级建议

**高优先级**（必须完成）:
1. Task 8: Checkpoint - 基础功能验证（确保系统能够正常工作）
2. Task 9.1-9.2: 端到端集成测试（验证系统正确性）
3. Task 11.1-11.2: 快速开始指南和示例（便于用户使用）

**中优先级**（建议完成）:
1. Task 9.3-9.4: 通信和 Adapter 测试（验证关键功能）
2. Task 10.1-10.2: 性能测试（验证性能目标）
3. Task 11.3: 故障排查指南（便于问题诊断）

**低优先级**（可选）:
1. Task 9.5: Property-Based 测试（提高测试质量）
2. Task 10.3-10.4: 性能优化和验收（进一步提升性能）
3. Task 11.4: README 更新（完善文档）

## 下一步行动

根据当前进度（Tasks 1-7 已完成），建议按以下顺序执行：

1. **Task 8**: Checkpoint - 基础功能验证（预计 0.5-1 天）
   - 手动测试验证系统能够正常工作
   - 发现并修复基础问题
   - 确认所有组件正确集成

2. **Task 9.1-9.2**: 端到端集成测试（预计 2-3 天）
   - 验证系统正确性
   - 为后续优化提供基准
   - 确保核心功能稳定

3. **Task 11.1-11.2**: 文档和示例（预计 1-2 天）
   - 便于用户使用
   - 降低学习成本
   - 提供最佳实践

4. **Task 10.1-10.2**: 性能测试（预计 2-3 天）
   - 验证性能目标
   - 识别性能瓶颈
   - 为优化提供数据支持

5. **Task 12**: Final Checkpoint（预计 0.5 天）
   - 全面验收
   - 确认所有目标达成
   - 准备交付

**总预计时间**: 6-10 天（不包括可选任务）

## 当前状态详细说明

### 已完成的功能（Tasks 1-7）

**核心组件**:
- ✅ Round Robin Router: 轮询路由策略，支持任意数量 Worker
- ✅ GPU Worker: 完整的推理实例，包括：
  - ZMQ 通信（PULL/PUSH sockets）
  - ReqQueue 集成（批处理、显存管理）
  - Adapter 管理（rank 配置、内存跟踪、加载/卸载）
  - 模型 RPC 初始化（单 GPU 模式）
  - 推理逻辑（prefill/decode）
  - 批次管理（生成、合并、过滤）
- ✅ DataParallelRouterManager: Worker 管理和请求路由，包括：
  - GPU 检测和分配
  - Worker 进程启动和监控
  - ZMQ 通信设置
  - 请求路由（Round Robin）
  - 进程健康检查
- ✅ Response Merger: 响应收集和转发
- ✅ API Server 集成: 命令行参数、模式选择、启动日志

**错误处理**:
- ✅ Worker 启动失败检测和进程清理
- ✅ 推理异常处理（CUDA OOM、运行时错误、RPC 超时）
- ✅ ZMQ 通信超时和重试机制
- ✅ Worker 进程监控和退出检测

**监控和日志**:
- ✅ 详细的启动日志（GPU 信息、模型加载进度）
- ✅ 请求统计（总数、成功数、失败数、Worker 分布、吞吐量）
- ✅ 调试日志（批次处理、Adapter 加载、响应转发）

**测试覆盖**:
- ✅ 20+ 个单元测试文件
- ✅ 覆盖所有核心组件
- ✅ 测试错误处理和异常场景

### 待完成的任务（Tasks 8-12）

**Task 8: 基础功能验证**
- 目标：确认系统能够正常工作
- 方法：手动测试 + 简单测试脚本
- 重点：端到端流程、错误处理、日志输出

**Task 9: 集成测试**
- 目标：验证系统正确性和稳定性
- 方法：自动化集成测试
- 重点：单/多 Worker、并发、通信、Adapter 切换

**Task 10: 性能测试**
- 目标：验证性能目标（3 Workers ≥ 2.5x）
- 方法：基准测试脚本
- 重点：吞吐量、延迟、GPU 利用率

**Task 11: 文档和示例**
- 目标：便于用户使用和问题排查
- 方法：编写文档和示例脚本
- 重点：快速开始、故障排查、最佳实践

**Task 12: 最终验收**
- 目标：确认所有目标达成
- 方法：全面检查和验收
- 重点：功能、性能、测试、文档
