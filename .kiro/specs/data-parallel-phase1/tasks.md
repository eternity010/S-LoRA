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

**已完成的核心组件**：
- ✅ Round Robin Router（轮询路由器）
- ✅ GPU Worker 基础框架（ZMQ 通信、ReqQueue 集成、Adapter 管理）
- ✅ GPU Worker 推理逻辑（模型 RPC 初始化、批次推理、批次管理）
- ✅ DataParallelRouterManager（Worker 管理、请求路由）
- ✅ 单元测试覆盖（21 个测试文件）

**待实现的组件**：
- ⏳ Response Merger（响应合并器）
- ⏳ API Server 集成（命令行参数、模式选择）
- ⏳ 错误处理完善（Worker 启动失败、ZMQ 超时、进程监控）
- ⏳ 监控完善（请求统计、调试日志）
- ⏳ 集成测试（端到端测试）
- ⏳ 性能测试（吞吐量、延迟基准）
- ⏳ 文档和示例

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

- [ ] 4. 实现 Response Merger
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

- [ ] 5. 修改 API Server 入口
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
  
  - [ ] 5.3 添加启动日志
    - 输出当前并行模式
    - 输出 Worker 数量和 GPU 列表
    - _Requirements: 6.5, 8.1_
  
  - [ ] 5.4 更新 router/manager.py 或创建新的入口
    - 确保数据并行模式和张量并行模式可以共存
    - 保持向后兼容性
    - _Requirements: 6.1, 6.2, 6.4_

- [ ]* 5.5 编写 API Server 集成测试
  - 测试命令行参数解析
  - 测试模式选择逻辑
  - 测试启动日志输出

- [ ] 6. 完善错误处理
  - [ ] 6.1 Worker 启动失败处理
    - 在 `run_gpu_worker_process()` 中添加更完善的异常捕获
    - 记录详细错误日志和堆栈跟踪
    - 通过退出码通知 Router Manager
    - 在 Router Manager 中检测 Worker 启动失败
    - _Requirements: 7.1, 7.2_
  
  - [ ] 6.2 推理异常处理（验证和完善）
    - 验证 Worker 中的推理异常捕获是否完整
    - 确保返回错误响应格式正确
    - 添加更详细的错误日志
    - 测试各种异常场景（CUDA OOM、模型错误等）
    - _Requirements: 7.3_
  
  - [ ] 6.3 ZMQ 通信超时处理
    - 在 Router Manager 中设置 socket 超时
    - 在 Worker 中设置 socket 超时
    - 实现重试逻辑（最多 3 次）
    - 记录超时错误日志
    - _Requirements: 7.4_
  
  - [ ] 6.4 Worker 进程监控
    - 在 Router Manager 中实现进程状态检查
    - 定期检测 Worker 进程是否存活
    - 检测 Worker 进程意外退出
    - 记录进程退出日志
    - _Requirements: 7.5_

- [ ]* 6.5 编写错误处理测试
  - 测试 Worker 启动失败场景
  - 测试推理异常处理
  - 测试 ZMQ 通信超时
  - 测试 Worker 进程崩溃恢复

- [ ] 7. 完善基础监控
  - [ ] 7.1 验证和完善启动日志
    - 确认 Worker 就绪日志已输出（已实现）
    - 确认 Router Manager 启动日志已输出（已实现）
    - 添加更详细的启动信息（GPU 型号、内存等）
    - 添加模型加载进度日志
    - _Requirements: 8.1, 8.3_
  
  - [ ] 7.2 实现请求统计
    - 在 Router Manager 中添加统计计数器
    - 统计总请求数、成功数、失败数
    - 每 10 秒输出统计信息
    - 添加每个 Worker 的请求分布统计
    - 添加平均延迟和吞吐量统计
    - _Requirements: 8.2_
  
  - [ ] 7.3 验证和完善调试日志
    - 确认路由决策日志已输出（已实现）
    - 添加响应返回日志（DEBUG 级别）
    - 添加批次处理日志（DEBUG 级别）
    - 添加 Adapter 加载/卸载日志
    - _Requirements: 8.4, 8.5_

- [ ]* 7.4 编写监控测试
  - 测试启动日志输出
  - 测试请求统计功能
  - 测试调试日志输出

- [ ] 8. Checkpoint - 基础功能验证
  - 手动测试所有核心组件能够正常工作
  - 验证单 Worker 能够处理请求（使用测试脚本）
  - 验证多 Worker 能够并发处理（使用测试脚本）
  - 验证 Response Merger 正确转发响应
  - 验证 API Server 集成正常工作
  - 询问用户是否有问题

- [ ] 9. 集成测试
  - [ ] 9.1 单 Worker 端到端测试
    - 创建 `test/test_e2e_single_worker.py`
    - 启动完整的数据并行系统（1 个 Worker）
    - 发送测试请求到 API Server
    - 验证响应正确
    - 验证延迟在合理范围内
    - _Requirements: 10.1_
  
  - [ ] 9.2 多 Worker 并发测试
    - 创建 `test/test_e2e_multi_worker.py`
    - 启动完整的数据并行系统（3 个 Worker）
    - 并发发送 10 个请求
    - 验证所有响应正确
    - 验证请求均匀分配到所有 Worker
    - 验证并发性能提升
    - _Requirements: 10.2_
  
  - [ ] 9.3 ZMQ 通信测试
    - 创建 `test/test_zmq_communication.py`
    - 测试 Router → Worker 通信
    - 测试 Worker → Response Merger 通信
    - 测试消息不丢失
    - 测试消息顺序（如果需要）
    - _Requirements: 10.5_
  
  - [ ] 9.4 Adapter 切换测试
    - 创建 `test/test_e2e_adapter_switching.py`
    - 测试在不同请求中切换 Adapter
    - 验证 Adapter 正确加载和使用
    - 验证内存管理正确
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
    - 创建 `benchmarks/benchmark_phase1.py`
    - 实现吞吐量测试（requests/second）
    - 实现延迟测试（P50, P90, P99）
    - 实现 GPU 利用率监控
    - 支持不同 Worker 数量的对比测试
    - _Requirements: 9.4_
  
  - [ ] 10.2 运行基准测试
    - 测试 1, 2, 3, 4 Workers 的性能
    - 记录吞吐量和延迟数据
    - 验证 GPU 利用率
    - 生成性能对比图表
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ] 10.3 性能分析和优化
    - 分析性能瓶颈（使用 profiler）
    - 优化 ZMQ 通信（如果需要）
    - 优化消息序列化（如果需要）
    - 优化批次管理（如果需要）
    - 验证优化效果
    - _Requirements: 9.1, 9.2_
  
  - [ ] 10.4 性能验收
    - 验证 3 Workers 吞吐量 ≥ 2.5x 单 Worker
    - 验证平均延迟增加 < 10%
    - 验证 GPU 利用率 > 85%
    - 记录性能测试结果
    - _Requirements: 9.1, 9.2, 9.3_

- [ ] 11. 文档和示例
  - [ ] 11.1 编写快速开始指南
    - 创建 `docs/data_parallel_quickstart.md`
    - 说明如何启动数据并行模式
    - 说明配置参数（--parallel-mode, --num-workers, --gpu-ids）
    - 提供完整的启动命令示例
    - 说明如何验证系统正常工作
  
  - [ ] 11.2 创建示例脚本
    - 创建 `examples/run_data_parallel.sh`
    - 提供不同配置的示例（1/2/3/4 Workers）
    - 提供不同 GPU 配置的示例
    - 添加注释说明每个参数的作用
  
  - [ ] 11.3 编写故障排查指南
    - 创建 `docs/data_parallel_troubleshooting.md`
    - 常见错误和解决方法
    - 调试技巧（如何查看日志、如何检查进程状态）
    - 性能调优建议
  
  - [ ] 11.4 更新主 README
    - 在主 README 中添加数据并行模式的说明
    - 添加快速开始链接
    - 添加性能对比数据

- [ ] 12. Final Checkpoint - Phase 1 完成验收
  - 所有功能测试通过
  - 所有集成测试通过
  - 性能目标达成（3 Workers ≥ 2.5x）
  - 文档完整且准确
  - 代码修改记录完整
  - 询问用户是否满意，是否需要调整

## Notes

- 标记 `*` 的任务为可选任务（测试相关），可以根据时间安排决定是否实施
- 每个任务都标注了对应的 Requirements，便于追溯
- Checkpoint 任务用于阶段性验证，确保及时发现问题
- Property-Based 测试使用 Hypothesis 框架
- 性能测试目标：3 Workers 吞吐量 ≥ 单 Worker 的 2.5 倍
