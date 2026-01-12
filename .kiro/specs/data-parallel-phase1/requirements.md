# Requirements Document - Phase 1: 数据并行基础框架

## Introduction

本文档定义了 S-LoRA 数据并行模式 Phase 1（基础框架搭建）的需求。Phase 1 的目标是实现最基本的数据并行功能，使多个 GPU 能够独立处理请求，为后续的优化奠定基础。

## Glossary

- **Data_Parallel_Mode**: 数据并行模式，每张 GPU 拥有完整的基座模型，独立处理不同的请求
- **GPU_Worker**: GPU 工作进程，在单张 GPU 上运行的独立推理实例
- **Router_Manager**: 路由管理器，负责启动 Worker 并分发请求
- **Round_Robin_Router**: 轮询路由器，按顺序将请求分配到不同的 Worker
- **Response_Merger**: 响应合并器，收集来自多个 Worker 的响应并转发
- **ZMQ**: ZeroMQ，用于进程间通信的消息队列库
- **Base_Model**: 基座模型，如 Llama-7B，每个 Worker 都会加载完整的基座模型

## Requirements

### Requirement 1: GPU Worker 进程管理

**User Story:** 作为系统开发者，我希望能够在每张 GPU 上启动独立的 Worker 进程，以便实现数据并行处理。

#### Acceptance Criteria

1. WHEN 系统启动时，THE Router_Manager SHALL 根据配置创建指定数量的 GPU_Worker 进程
2. WHEN 创建 GPU_Worker 时，THE System SHALL 为每个 Worker 分配唯一的 worker_id 和 gpu_id
3. WHEN GPU_Worker 启动时，THE System SHALL 设置 CUDA_VISIBLE_DEVICES 环境变量为对应的 gpu_id
4. WHEN GPU_Worker 初始化时，THE System SHALL 在指定的 GPU 上加载完整的 Base_Model
5. WHEN 所有 GPU_Worker 启动完成时，THE Router_Manager SHALL 确认所有 Worker 处于就绪状态

### Requirement 2: 请求路由与分发

**User Story:** 作为系统开发者，我希望实现基本的请求路由功能，以便将请求分发到不同的 Worker。

#### Acceptance Criteria

1. WHEN 新请求到达 Router_Manager 时，THE Round_Robin_Router SHALL 选择下一个 Worker 来处理该请求
2. WHEN 使用轮询策略时，THE Round_Robin_Router SHALL 按照 Worker ID 的顺序循环分配请求
3. WHEN 请求被分配到 Worker 时，THE Router_Manager SHALL 通过 ZMQ PUSH socket 发送请求消息
4. THE System SHALL 在请求消息中包含 request_id、adapter_dir、prompt_ids 和 sampling_params
5. WHEN 发送请求失败时，THE System SHALL 记录错误日志并返回错误响应

### Requirement 3: Worker 请求处理

**User Story:** 作为 GPU Worker，我希望能够接收和处理分配给我的请求，以便完成推理任务。

#### Acceptance Criteria

1. WHEN GPU_Worker 运行时，THE Worker SHALL 持续监听 ZMQ PULL socket 接收请求
2. WHEN 收到请求时，THE Worker SHALL 解析请求消息并提取必要的参数
3. WHEN 请求需要的 adapter 未加载时，THE Worker SHALL 从磁盘加载该 adapter
4. WHEN 执行推理时，THE Worker SHALL 使用现有的模型推理逻辑处理请求
5. WHEN 推理完成时，THE Worker SHALL 生成包含 output_ids 和 metadata 的响应消息

### Requirement 4: 响应收集与转发

**User Story:** 作为系统开发者，我希望能够收集来自多个 Worker 的响应并正确转发，以便完成端到端的请求处理。

#### Acceptance Criteria

1. WHEN GPU_Worker 完成推理时，THE Worker SHALL 通过 ZMQ PUSH socket 发送响应消息
2. WHEN Response_Merger 运行时，THE Merger SHALL 通过 ZMQ PULL socket 接收来自所有 Worker 的响应
3. WHEN 收到响应时，THE Response_Merger SHALL 根据 request_id 匹配原始请求
4. WHEN 响应匹配成功时，THE Response_Merger SHALL 将响应转发到 Detokenization 进程
5. THE System SHALL 确保响应消息包含 request_id、worker_id、output_ids 和 success 状态

### Requirement 5: 配置与启动

**User Story:** 作为用户，我希望能够通过命令行参数配置数据并行模式，以便灵活使用系统。

#### Acceptance Criteria

1. THE System SHALL 支持通过 `--parallel-mode data` 参数启用数据并行模式
2. THE System SHALL 支持通过 `--num-workers N` 参数指定 Worker 数量
3. THE System SHALL 支持通过 `--gpu-ids 0,1,2` 参数指定使用的 GPU 列表
4. WHEN 未指定 num-workers 时，THE System SHALL 自动检测可用 GPU 数量并创建对应数量的 Worker
5. WHEN 未指定 gpu-ids 时，THE System SHALL 使用所有可用的 GPU

### Requirement 6: 向后兼容

**User Story:** 作为现有用户，我希望新功能不破坏现有的张量并行模式，以便根据需要选择合适的模式。

#### Acceptance Criteria

1. WHEN 未指定 parallel-mode 参数时，THE System SHALL 默认使用张量并行模式
2. WHEN 使用 `--parallel-mode tensor` 时，THE System SHALL 使用原有的张量并行逻辑
3. WHEN 使用 `--parallel-mode data` 时，THE System SHALL 使用新的数据并行逻辑
4. THE System SHALL 保持现有的 API 接口不变
5. THE System SHALL 在启动日志中明确输出当前使用的并行模式

### Requirement 7: 错误处理

**User Story:** 作为系统开发者，我希望系统能够处理基本的错误情况，以便提高系统稳定性。

#### Acceptance Criteria

1. WHEN GPU_Worker 启动失败时，THE System SHALL 记录详细的错误信息并退出
2. WHEN 模型加载失败时，THE Worker SHALL 记录错误并通知 Router_Manager
3. WHEN 推理过程中发生异常时，THE Worker SHALL 捕获异常并返回错误响应
4. WHEN ZMQ 通信超时时，THE System SHALL 记录超时错误并重试
5. WHEN Worker 进程意外退出时，THE Router_Manager SHALL 检测到并记录日志

### Requirement 8: 基础监控

**User Story:** 作为运维人员，我希望能够查看基本的运行状态，以便了解系统是否正常工作。

#### Acceptance Criteria

1. THE System SHALL 在启动时输出所有 Worker 的 worker_id 和 gpu_id
2. THE System SHALL 每 10 秒输出一次整体的请求处理统计（总请求数、成功数、失败数）
3. WHEN Worker 启动完成时，THE System SHALL 输出 "Worker {id} ready on GPU {gpu_id}"
4. WHEN 请求被路由时，THE System SHALL 在 DEBUG 级别记录路由决策
5. WHEN 响应返回时，THE System SHALL 在 DEBUG 级别记录响应信息

### Requirement 9: 性能基准

**User Story:** 作为性能工程师，我希望验证数据并行模式的性能提升，以便确认实现目标。

#### Acceptance Criteria

1. WHEN 使用 3 个 Worker 时，THE System SHALL 达到单 Worker 吞吐量的 2.5 倍以上
2. WHEN 使用数据并行模式时，THE System SHALL 保持平均延迟增加小于 10%
3. WHEN 系统运行时，THE System SHALL 保持 GPU 利用率高于 85%
4. THE System SHALL 支持通过性能测试脚本验证吞吐量和延迟指标
5. THE System SHALL 在性能测试中无明显的内存泄漏

### Requirement 10: 测试覆盖

**User Story:** 作为质量工程师，我希望有完整的测试覆盖，以便确保代码质量。

#### Acceptance Criteria

1. THE System SHALL 包含单 Worker 启动和推理的单元测试
2. THE System SHALL 包含多 Worker 并发处理的集成测试
3. THE System SHALL 包含端到端的功能测试
4. THE System SHALL 包含轮询路由正确性的测试
5. THE System SHALL 包含 ZMQ 通信的测试
