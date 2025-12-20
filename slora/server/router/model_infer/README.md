# model_infer 目录文件说明

本目录包含 S-LoRA 模型推理相关的核心模块，负责批次管理、适配器管理和推理执行。

## 文件概览

### 1. `infer_adapter.py` - LoRA 适配器管理器

**核心类**: `InferAdapter`

**主要功能**:
- **适配器加载/卸载**: 管理 LoRA 适配器在显存中的生命周期
- **统一分页**: 使用统一内存池管理适配器权重，减少内存碎片
- **预取机制**: 支持异步预取适配器，提升性能
- **统计追踪**: 维护适配器使用次数、访问时间、活跃请求数等统计信息
- **评分系统**: 计算适配器综合分数，支持智能淘汰策略

**关键方法**:
- `load_adapters()`: 批量加载适配器到显存
- `offload_adapters()`: 卸载不需要的适配器，释放显存
- `update_adapter_stats_batch()`: 按批次更新适配器使用统计
- `calculate_adapter_score()`: 计算适配器综合分数

---

### 2. `infer_batch.py` - 推理批次数据结构

**核心类**: `InferBatch`, `InferSamplingParams`

**主要功能**:
- **批次初始化**: 将请求列表转换为批次数据结构
- **无填充批处理**: 支持变长序列的无填充批处理（no-padding）
- **批次操作**: 支持批次过滤、合并、释放等操作
- **采样参数**: 管理每个请求的采样参数（temperature, top_p, top_k 等）

**关键方法**:
- `init_batch()`: 初始化批次，处理输入 token 和位置信息
- `filter()`: 过滤批次中的请求（完成或取消的请求）
- `merge()`: 合并两个批次
- `free_self()`: 释放批次占用的显存
- `get_post_sample_tensors()`: 获取后处理所需的张量

---

### 3. `model_rpc.py` - 模型 RPC 服务

**核心类**: `ModelRpcServer`

**主要功能**:
- **RPC 服务**: 提供远程过程调用接口，支持多 GPU 分布式推理
- **模型初始化**: 初始化基础模型和 LoRA 适配器
- **批次推理**: 执行 prefill 和 decode 阶段的前向传播
- **批次管理**: 管理批次的生命周期（添加、过滤、合并、删除）

**关键方法**:
- `exposed_init_model()`: 初始化模型和适配器
- `exposed_prefill_batch()`: 执行 prefill 阶段推理
- `exposed_decode_batch()`: 执行 decode 阶段推理
- `exposed_load_adapters()`: 加载适配器到显存
- `exposed_offload_adapters()`: 卸载适配器

**工作流程**:
```
请求 → init_batch → load_adapters → prefill_batch → decode_batch → offload_adapters
```

---

### 4. `naive_infer_adapter.py` - 简单适配器管理器

**核心类**: `NaiveInferAdapter`

**主要功能**:
- **简单实现**: 不使用统一内存池的适配器管理实现
- **直接存储**: 将适配器权重直接存储在独立的 buffer 中
- **适用场景**: 当 `no_mem_pool=True` 时使用，用于对比测试

**与 `InferAdapter` 的区别**:
- 不使用 `MemoryAllocator`，直接管理 `key_buffer` 和 `value_buffer`
- 不支持预取机制
- 内存管理更简单，但灵活性较低

---

### 5. `post_process.py` - 后处理模块

**核心函数**: `sample()`, `_top_p_top_k()`

**主要功能**:
- **采样**: 根据 logits 进行 token 采样
- **惩罚机制**: 应用 presence penalty 和 frequency penalty
- **Top-p/Top-k**: 实现 nucleus sampling 和 top-k sampling
- **温度控制**: 应用 temperature 缩放

**处理流程**:
```
logits → apply_penalty → temperature → softmax → top_p/top_k → multinomial → token_id
```

---

## 模块间关系

```
RouterManager
    ↓
ModelRpcServer (model_rpc.py)
    ↓
InferBatch (infer_batch.py) ←→ InferAdapter (infer_adapter.py)
    ↓                                    ↓
forward()                          load/offload adapters
    ↓
post_process.sample() (post_process.py)
```

## 关键概念

### 统一分页 (Unified Paging)
- 使用 `MemoryAllocator` 统一管理 KV cache 和 LoRA 权重
- 以 "cell" 为单位分配内存（1 cell = head_num × head_dim）
- 一个 rank=r 的 LoRA 占用 r×4 个 cells

### 无填充批处理 (No-Padding Batching)
- 使用 `nopad_b_loc`, `nopad_b_start_loc`, `nopad_b_seq_len` 管理变长序列
- 避免填充带来的计算浪费
- 支持高效的批次合并和过滤

### 适配器预取 (Adapter Prefetching)
- 在 decode 阶段异步预取下一批可能需要的适配器
- 使用 `prefetch_stream` 和 `prefetch_tag` 保护预取中的适配器
- 减少适配器加载延迟

---

## 使用示例

### 更新适配器统计信息

```python
# 在批次处理时更新统计
batch_adapter_dirs = batch.adapter_dirs
infer_adapter.update_adapter_stats_batch(batch_adapter_dirs)
```

### 获取适配器分数

```python
# 计算并获取适配器分数
scored_adapters = infer_adapter.get_adapters_by_score(top_k=10, ascending=False)
for adapter_dir, score in scored_adapters:
    print(f"{adapter_dir}: {score:.4f}")
```

---

## 注意事项

1. **内存管理**: `InferAdapter` 使用统一分页，需要与 `MemoryAllocator` 配合使用
2. **批次生命周期**: 批次创建后需要及时释放，避免内存泄漏
3. **适配器卸载**: 卸载时会清理相关统计数据，确保数据一致性
4. **多 GPU**: RPC 服务支持多 GPU 分布式推理，需要正确设置 `world_size`

