# Implementation Plan: Rank-Aware Routing
# 实现计划：Rank 感知路由

## Overview / 概述

This implementation plan breaks down the Rank-Aware Routing feature into discrete coding tasks. The implementation follows a phased approach: data structures first, then data collection, then routing logic, and finally configuration and testing.

本实现计划将 Rank 感知路由功能分解为离散的编码任务。实现遵循分阶段方法：首先是数据结构，然后是数据收集，接着是路由逻辑，最后是配置和测试。

## Tasks / 任务列表

- [x] 1. Extend WorkerState and RoutingConfig data structures / 扩展 WorkerState 和 RoutingConfig 数据结构
  - [x] 1.1 Add rank distribution fields to WorkerState dataclass / 向 WorkerState 添加 rank 分布字段
    - Add `avg_rank: float = 0.0` field / 添加平均 rank 字段
    - Add `min_rank: int = 0` field / 添加最小 rank 字段
    - Add `max_rank: int = 0` field / 添加最大 rank 字段
    - Update `to_dict()` to include rank fields / 更新序列化方法
    - Update `from_dict()` to deserialize rank fields with backward compatibility / 更新反序列化方法（向后兼容）
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 8.2_
    - **文件**: `S-LoRA/slora/server/router/worker_state.py`

  - [x]* 1.2 Write property test for WorkerState serialization round-trip / 编写 WorkerState 序列化往返测试
    - **Property 2: WorkerState Serialization Round-Trip**
    - **Validates: Requirements 2.4, 2.5**
    - **说明**: 验证序列化后再反序列化能得到相同的对象

  - [x] 1.3 Add rank-aware parameters to RoutingConfig dataclass / 向 RoutingConfig 添加 rank 感知参数
    - Add `w3: float = 0.0` field for rank mismatch penalty weight / 添加 rank 不匹配惩罚权重
    - Add `default_lora_rank: int = 16` field for unknown adapters / 添加未知 adapter 的默认 rank
    - Add `max_rank_diff: int = 64` field for normalization / 添加归一化用的最大 rank 差异
    - Add validation in `__post_init__` for new fields / 添加参数验证
    - Update `to_dict()` and `from_dict()` methods / 更新序列化方法
    - _Requirements: 6.5_
    - **文件**: `S-LoRA/slora/server/router/worker_state.py`

  - [x]* 1.4 Write property test for RoutingConfig validation / 编写 RoutingConfig 验证测试
    - **Property 7: Configuration Validation**
    - **Validates: Requirements 6.5**
    - **说明**: 验证无效配置（如 w3<0）会抛出 ValueError

  - [x] 1.5 Add rank-aware statistics to RoutingStats dataclass / 向 RoutingStats 添加 rank 感知统计
    - Add `rank_matched_count: int = 0` field / 添加 rank 匹配计数
    - Add `total_rank_mismatch: float = 0.0` field / 添加总 rank 不匹配度
    - Add `avg_rank_mismatch` property / 添加平均 rank 不匹配度属性
    - Add `record_rank_mismatch()` method / 添加记录方法
    - Update `to_dict()` and `reset()` methods / 更新相关方法
    - _Requirements: 7.1, 7.2_
    - **文件**: `S-LoRA/slora/server/router/worker_state.py`

  - [x]* 1.6 Write property test for RoutingStats tracking / 编写 RoutingStats 追踪测试
    - **Property 8: Routing Statistics Tracking**
    - **Validates: Requirements 7.1, 7.2**
    - **说明**: 验证统计数据正确累积和计算

- [x] 2. Checkpoint - Verify data structure changes / 检查点 - 验证数据结构变更
  - Ensure all tests pass, ask the user if questions arise.
  - 确保所有测试通过，如有问题询问用户。

- [ ] 3. Implement Worker rank information reporting / 实现 Worker rank 信息上报
  - [x] 3.1 Extend `_get_state_for_reporter()` in GPUWorker / 扩展 GPUWorker 的状态上报方法
    - Calculate avg_rank from current batch's adapter ranks / 计算当前批次的平均 rank
    - Calculate min_rank from current batch's adapter ranks / 计算最小 rank
    - Calculate max_rank from current batch's adapter ranks / 计算最大 rank
    - Handle empty batch case (return 0, 0, 0) / 处理空批次情况
    - Use `self.lora_ranks` mapping to get adapter ranks / 使用 lora_ranks 映射获取 rank
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
    - **文件**: `S-LoRA/slora/server/router/gpu_worker.py`

  - [x]* 3.2 Write property test for batch rank distribution calculation / 编写批次 rank 分布计算测试
    - **Property 1: Batch Rank Distribution Calculation**
    - **Validates: Requirements 1.1, 1.2, 1.3**
    - **说明**: 验证 avg/min/max 计算正确

  - [x] 3.3 Update WorkerStateReporter `_create_state_message()` method / 更新状态上报器的消息创建方法
    - Include avg_rank, min_rank, max_rank in state message / 在状态消息中包含 rank 信息
    - _Requirements: 1.1, 1.2, 1.3_
    - **文件**: `S-LoRA/slora/server/router/worker_state_reporter.py`

  - [x] 3.4 Update WorkerStateCache to handle rank fields / 更新 WorkerStateCache 处理 rank 字段
    - Update `update_state()` to process rank fields from state messages / 更新状态处理方法
    - Ensure backward compatibility with messages without rank fields / 确保向后兼容
    - _Requirements: 8.2, 8.3_
    - **文件**: `S-LoRA/slora/server/router/worker_state_cache.py`

  - [x]* 3.5 Write property test for backward compatible state deserialization / 编写向后兼容反序列化测试
    - **Property 9: Backward Compatible State Deserialization**
    - **Validates: Requirements 8.2, 8.3**
    - **说明**: 验证旧格式消息（无 rank 字段）能正确处理

- [x] 4. Checkpoint - Verify Worker reporting changes / 检查点 - 验证 Worker 上报变更
  - Ensure all tests pass, ask the user if questions arise.
  - 确保所有测试通过，如有问题询问用户。

- [ ] 5. Implement Router rank management and scoring / 实现 Router rank 管理和评分
  - [x] 5.1 Add adapter rank management to AdapterAwareRouter / 向 AdapterAwareRouter 添加 adapter rank 管理
    - Add `adapter_ranks: Dict[str, int]` attribute / 添加 adapter rank 映射
    - Implement `set_adapter_ranks()` method / 实现设置方法
    - Implement `get_adapter_rank()` method with default fallback / 实现获取方法（带默认值回退）
    - _Requirements: 3.2, 3.3, 3.4_
    - **文件**: `S-LoRA/slora/server/router/adapter_aware_router.py`

  - [ ]* 5.2 Write property test for adapter rank lookup / 编写 adapter rank 查询测试
    - **Property 3: Adapter Rank Lookup**
    - **Validates: Requirements 3.2, 3.3, 3.4**
    - **说明**: 验证已知 adapter 返回配置的 rank，未知 adapter 返回默认值

  - [x] 5.3 Implement `calculate_rank_mismatch()` method / 实现 rank 不匹配度计算方法
    - Calculate normalized mismatch between request rank and Worker's avg_rank / 计算归一化的不匹配度
    - Return 0.0 for empty batch (avg_rank=0) / 空批次返回 0
    - Return 0.0 when ranks are equal / rank 相等时返回 0
    - Normalize to [0, 1] range using max_rank_diff / 归一化到 [0, 1] 范围
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
    - **文件**: `S-LoRA/slora/server/router/adapter_aware_router.py`
    - **公式**: `mismatch = |request_rank - avg_rank| / max_rank_diff`

  - [ ]* 5.4 Write property test for rank mismatch normalization / 编写 rank 不匹配度归一化测试
    - **Property 4: Rank Mismatch Normalization**
    - **Validates: Requirements 4.1, 4.3, 4.4, 4.5**
    - **说明**: 验证返回值在 [0, 1] 范围内

  - [x] 5.5 Extend `calculate_score()` method with rank mismatch penalty / 扩展评分函数添加 rank 不匹配惩罚
    - Add rank mismatch term: `- w3 * RankMismatch` / 添加 rank 不匹配项
    - Only apply when w3 > 0 / 仅当 w3 > 0 时应用
    - Add debug logging for rank mismatch component / 添加调试日志
    - _Requirements: 5.1, 5.2, 5.4, 5.5_
    - **文件**: `S-LoRA/slora/server/router/adapter_aware_router.py`
    - **新公式**: `Score = w1·I(cache) - w2·QueueLen - w3·RankMismatch`

  - [ ]* 5.6 Write property test for extended scoring function / 编写扩展评分函数测试
    - **Property 5: Extended Scoring Function**
    - **Validates: Requirements 5.1, 5.2**
    - **说明**: 验证评分公式正确计算

  - [ ]* 5.7 Write property test for backward compatibility (w3=0) / 编写向后兼容测试 (w3=0)
    - **Property 6: Backward Compatibility (w3=0)**
    - **Validates: Requirements 5.3, 8.1**
    - **说明**: 验证 w3=0 时行为与原实现一致

  - [x] 5.8 Update `select_worker()` to record rank statistics / 更新 select_worker() 记录 rank 统计
    - Call `stats.record_rank_mismatch()` after routing decision / 路由决策后记录统计
    - _Requirements: 7.1, 7.2_
    - **文件**: `S-LoRA/slora/server/router/adapter_aware_router.py`

  - [x] 5.9 Update `log_stats_summary()` to include rank statistics / 更新日志输出包含 rank 统计
    - Add rank_matched_count and avg_rank_mismatch to log output / 添加 rank 相关统计到日志
    - _Requirements: 7.3_
    - **文件**: `S-LoRA/slora/server/router/adapter_aware_router.py`

  - [x] 5.10 Update `get_stats()` to include rank information / 更新 get_stats() 包含 rank 信息
    - Add rank-related statistics to returned dictionary / 添加 rank 相关统计到返回字典
    - _Requirements: 7.4_
    - **文件**: `S-LoRA/slora/server/router/adapter_aware_router.py`

- [x] 6. Checkpoint - Verify Router changes / 检查点 - 验证 Router 变更
  - Ensure all tests pass, ask the user if questions arise.
  - 确保所有测试通过，如有问题询问用户。

- [ ] 7. Implement DataParallelRouterManager integration / 实现 DataParallelRouterManager 集成
  - [x] 7.1 Add `_load_adapter_ranks()` method to DataParallelRouterManager / 添加加载 adapter rank 的方法
    - Load adapter ranks from lora_dirs configurations / 从 lora_dirs 配置加载 rank
    - Use get_lora_config to read rank from adapter_config.json / 使用 get_lora_config 读取 rank
    - Handle missing/invalid configurations with default rank / 处理缺失/无效配置
    - _Requirements: 3.1_
    - **文件**: `S-LoRA/slora/server/router/dp_manager.py`

  - [x] 7.2 Update `__init__` to load and pass adapter ranks to router / 更新 __init__ 加载并传递 adapter rank
    - Call `_load_adapter_ranks()` during initialization / 初始化时调用加载方法
    - Call `router.set_adapter_ranks()` for adapter-aware strategy / 调用 router 的设置方法
    - _Requirements: 3.1_
    - **文件**: `S-LoRA/slora/server/router/dp_manager.py`

  - [x] 7.3 Update `_get_routing_config()` to include new parameters / 更新 _get_routing_config() 包含新参数
    - Add w3 from args.routing_w3 / 添加 w3 参数
    - Add default_lora_rank from args.default_lora_rank / 添加默认 rank 参数
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
    - **文件**: `S-LoRA/slora/server/router/dp_manager.py`

- [ ] 8. Add command-line parameters / 添加命令行参数
  - [x] 8.1 Add --routing-w3 parameter to argument parser / 添加 --routing-w3 参数
    - Type: float, default: 0.0 / 类型: float, 默认值: 0.0
    - Help text explaining rank mismatch penalty weight / 帮助文本说明 rank 不匹配惩罚权重
    - _Requirements: 6.1, 6.2_
    - **文件**: `S-LoRA/slora/server/api_server.py` 或参数解析文件

  - [x] 8.2 Add --default-lora-rank parameter to argument parser / 添加 --default-lora-rank 参数
    - Type: int, default: 16 / 类型: int, 默认值: 16
    - Help text explaining default rank for unknown adapters / 帮助文本说明未知 adapter 的默认 rank
    - _Requirements: 6.3, 6.4_
    - **文件**: `S-LoRA/slora/server/api_server.py` 或参数解析文件

- [x] 9. Checkpoint - Verify integration and CLI changes / 检查点 - 验证集成和 CLI 变更
  - Ensure all tests pass, ask the user if questions arise.
  - 确保所有测试通过，如有问题询问用户。

- [ ] 10. Write integration tests / 编写集成测试
  - [x]* 10.1 Write integration test for end-to-end rank-aware routing / 编写端到端 rank 感知路由测试
    - **文件**: `S-LoRA/test/test_e2e_rank_aware_routing.py`
    - Test that requests are routed to Workers with similar batch ranks / 测试请求被路由到相似 rank 的 Worker
    - Test with multiple Workers having different batch compositions / 测试多个 Worker 有不同批次组成
    - _Requirements: 5.1, 5.2_
    - **说明**: 验证整体功能正确

  - [ ]* 10.2 Write integration test for backward compatibility / 编写向后兼容集成测试
    - Test that w3=0 produces same routing as original implementation / 测试 w3=0 时路由与原实现一致
    - Test mixed Workers (some with rank info, some without) / 测试混合 Worker（部分有 rank 信息）
    - _Requirements: 8.1, 8.3_
    - **说明**: 验证不破坏现有功能

- [ ] 11. Final checkpoint - Ensure all tests pass / 最终检查点 - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.
  - 确保所有测试通过，如有问题询问用户。

## Notes / 备注

- Tasks marked with `*` are optional and can be skipped for faster MVP / 标记 `*` 的任务是可选的，可跳过以加快 MVP
- Each task references specific requirements for traceability / 每个任务引用特定需求以便追溯
- Checkpoints ensure incremental validation / 检查点确保增量验证
- Property tests validate universal correctness properties / 属性测试验证通用正确性属性
- Unit tests validate specific examples and edge cases / 单元测试验证特定示例和边界情况
- The implementation follows the existing code patterns in the S-LoRA codebase / 实现遵循 S-LoRA 代码库的现有模式
- Python `hypothesis` library should be used for property-based testing / 应使用 Python `hypothesis` 库进行属性测试

## 关键文件索引 / Key Files Index

| 文件 | 修改内容 |
|------|---------|
| `worker_state.py` | WorkerState、RoutingConfig、RoutingStats 数据结构扩展 |
| `gpu_worker.py` | `_get_state_for_reporter()` 添加 rank 信息 |
| `worker_state_reporter.py` | 状态消息包含 rank 字段 |
| `worker_state_cache.py` | 处理 rank 字段的状态更新 |
| `adapter_aware_router.py` | rank 管理、mismatch 计算、评分函数扩展 |
| `dp_manager.py` | 加载 adapter rank、传递给 router |
| `api_server.py` | 命令行参数 --routing-w3, --default-lora-rank |

## 评分函数演进 / Scoring Function Evolution

**原公式 (Current)**:
```
Score = w1·I(cache) - w2·QueueLen
```

**新公式 (New)**:
```
Score = w1·I(cache) - w2·QueueLen - w3·RankMismatch
```

其中:
- `w1` = 缓存亲和性权重 (默认 10.0)
- `w2` = 队列长度惩罚权重 (默认 1.0)
- `w3` = Rank 不匹配惩罚权重 (默认 0.0，禁用)
- `RankMismatch` = `|request_rank - worker_avg_rank| / max_rank_diff`
