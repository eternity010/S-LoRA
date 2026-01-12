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
- 测试命令：`pytest tests/` 或 `pytest <specific_test_file>`

## Tasks

- [x] 1. 创建 Round Robin Router
  - 实现轮询路由逻辑
  - 支持任意数量的 Worker
  - _Requirements: 2.1, 2.2_

- [ ]* 1.1 编写 Round Robin Router 单元测试
  - **Property 2: 轮询路由公平性**
  - **Validates: Requirements 2.1, 2.2**
  - 测试轮询顺序正确性
  - 测试不同 Worker 数量下的行为

- [ ] 2. 实现 GPU Worker 基础框架
  - [x] 2.1 创建 GPUWorker 类框架
    - 实现 `__init__` 方法
    - 实现 GPU 环境设置
    - _Requirements: 1.2, 1.3_
  
  - [x] 2.2 实现 ZMQ 通信设置
    - 创建 PULL socket 接收请求
    - 创建 PUSH socket 发送响应
    - _Requirements: 2.3, 4.1_
  
  - [ ] 2.3 集成模型加载逻辑
    - 复用 `slora/common/basemodel/` 代码
    - 在指定 GPU 上加载模型
    - _Requirements: 1.4_
  
  - [ ] 2.4 实现请求处理循环
    - 实现 `_receive_request()` 方法
    - 实现 `_process_request()` 方法
    - 实现 `_send_response()` 方法
    - 实现 `run()` 主循环
    - _Requirements: 3.1, 3.2, 3.5_

- [ ]* 2.5 编写 GPU Worker 单元测试
  - 测试 GPU 环境设置
  - 测试 ZMQ socket 创建
  - 测试消息接收和发送

- [ ] 3. 实现 Data Parallel Router Manager
  - [ ] 3.1 创建 DataParallelRouterManager 类
    - 实现 `__init__` 方法
    - 实现 GPU 检测逻辑
    - 实现 GPU ID 解析
    - _Requirements: 1.1, 5.2, 5.3_
  
  - [ ] 3.2 实现 Worker 进程管理
    - 实现 `_start_worker()` 方法
    - 实现 `start_workers()` 方法
    - 为每个 Worker 分配端口
    - _Requirements: 1.1, 1.2_
  
  - [ ] 3.3 实现请求路由逻辑
    - 集成 Round Robin Router
    - 实现 `route_request()` 方法
    - 设置 ZMQ PUSH sockets
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [ ] 3.4 实现主循环
    - 接收来自 API Server 的请求
    - 调用路由器选择 Worker
    - 发送请求到选定的 Worker
    - _Requirements: 2.1, 2.3_

- [ ]* 3.5 编写 Router Manager 单元测试
  - 测试 GPU 检测
  - 测试 Worker 启动
  - 测试请求路由

- [ ] 4. 实现 Response Merger
  - [ ] 4.1 创建 ResponseMerger 类
    - 实现 `__init__` 方法
    - 设置 ZMQ PULL socket（接收 Worker 响应）
    - 设置 ZMQ PUSH socket（发送到 Detokenization）
    - _Requirements: 4.1, 4.2_
  
  - [ ] 4.2 实现响应转发逻辑
    - 实现 `_convert_to_detoken_format()` 方法
    - 实现 `_forward_to_detokenization()` 方法
    - 实现 `run()` 主循环
    - _Requirements: 4.3, 4.4, 4.5_

- [ ]* 4.3 编写 Response Merger 单元测试
  - 测试消息格式转换
  - 测试响应转发

- [ ] 5. 修改 API Server 入口
  - [ ] 5.1 添加命令行参数
    - 添加 `--parallel-mode` 参数
    - 添加 `--num-workers` 参数
    - 添加 `--gpu-ids` 参数
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [ ] 5.2 实现模式选择逻辑
    - 根据 `parallel-mode` 选择启动逻辑
    - 数据并行模式：启动 DataParallelRouterManager
    - 张量并行模式：使用原有逻辑
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [ ] 5.3 添加启动日志
    - 输出当前并行模式
    - 输出 Worker 数量和 GPU 列表
    - _Requirements: 6.5, 8.1_

- [ ] 6. 实现错误处理
  - [ ] 6.1 Worker 启动失败处理
    - 捕获模型加载异常
    - 记录详细错误日志
    - 通知 Router Manager
    - _Requirements: 7.1, 7.2_
  
  - [ ] 6.2 推理异常处理
    - 在 Worker 中捕获推理异常
    - 返回错误响应
    - _Requirements: 7.3_
  
  - [ ] 6.3 ZMQ 通信超时处理
    - 设置 socket 超时
    - 实现重试逻辑
    - _Requirements: 7.4_

- [ ] 7. 添加基础监控
  - [ ] 7.1 实现启动日志
    - Worker 就绪日志
    - Router Manager 启动日志
    - _Requirements: 8.1, 8.3_
  
  - [ ] 7.2 实现请求统计
    - 统计总请求数
    - 统计成功/失败数
    - 每 10 秒输出统计信息
    - _Requirements: 8.2_
  
  - [ ] 7.3 添加调试日志
    - 路由决策日志（DEBUG 级别）
    - 响应返回日志（DEBUG 级别）
    - _Requirements: 8.4, 8.5_

- [ ] 8. Checkpoint - 基础功能验证
  - 确保所有核心组件能够正常工作
  - 验证单 Worker 能够处理请求
  - 验证多 Worker 能够并发处理
  - 询问用户是否有问题

- [ ] 9. 集成测试
  - [ ] 9.1 单 Worker 端到端测试
    - 启动 1 个 Worker
    - 发送测试请求
    - 验证响应正确
    - _Requirements: 10.1_
  
  - [ ] 9.2 多 Worker 并发测试
    - 启动 3 个 Worker
    - 并发发送 10 个请求
    - 验证所有响应正确
    - 验证请求均匀分配
    - _Requirements: 10.2_
  
  - [ ] 9.3 ZMQ 通信测试
    - 测试 Router → Worker 通信
    - 测试 Worker → Response Merger 通信
    - 测试消息不丢失
    - _Requirements: 10.5_

- [ ]* 9.4 编写 Property-Based 测试
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
    - 实现吞吐量测试
    - 实现延迟测试
    - _Requirements: 9.4_
  
  - [ ] 10.2 运行基准测试
    - 测试 1, 2, 3, 4 Workers 的性能
    - 记录吞吐量和延迟数据
    - 验证 GPU 利用率
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ] 10.3 性能分析和优化
    - 分析性能瓶颈
    - 优化 ZMQ 通信
    - 优化消息序列化
    - _Requirements: 9.1, 9.2_

- [ ] 11. 文档和示例
  - [ ] 11.1 编写快速开始指南
    - 创建 `docs/data_parallel_quickstart.md`
    - 说明如何启动数据并行模式
    - 说明配置参数
  
  - [ ] 11.2 创建示例脚本
    - 创建 `examples/run_data_parallel.sh`
    - 提供不同配置的示例
  
  - [ ] 11.3 编写故障排查指南
    - 常见错误和解决方法
    - 调试技巧

- [ ] 12. Final Checkpoint - Phase 1 完成验收
  - 所有功能测试通过
  - 性能目标达成（3 Workers ≥ 2.5x）
  - 文档完整
  - 询问用户是否满意

## Notes

- 标记 `*` 的任务为可选任务（测试相关），可以根据时间安排决定是否实施
- 每个任务都标注了对应的 Requirements，便于追溯
- Checkpoint 任务用于阶段性验证，确保及时发现问题
- Property-Based 测试使用 Hypothesis 框架
- 性能测试目标：3 Workers 吞吐量 ≥ 单 Worker 的 2.5 倍
