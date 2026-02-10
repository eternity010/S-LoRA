# Bug 修复：数据并行模式下的批次合并问题

## 问题描述

在数据并行模式下，当新请求加入现有批次时，会出现 `KeyError` 错误：

```
File "/home/hzheng/S-LoRA/slora/server/router/model_infer/infer_batch.py", line 162, in filter
    idx = self.requests_idx_mapping[request_id]
KeyError: 'af81a58fedd143a5b2a861bff722c5b8'
```

## 根本原因

**Worker 端和 RPC 端的批次状态不同步**

在 `gpu_worker.py` 的 `_process_requests` 方法中，当合并新批次时：

```python
# 旧代码（有问题）
if new_batch is not None:
    await self._load_adapters(new_batch.adapter_dirs)
    
    if self.current_batch is None:
        self.current_batch = new_batch
    else:
        self.current_batch.merge(new_batch)  # 只在 Worker 端合并！
```

**问题**：
1. Worker 端的 `Batch` 对象合并了新请求
2. 但 RPC 端的 `InferBatch` 对象没有这些新请求
3. 当调用 `filter_batch(req_id_list)` 时，传递了 RPC 端不存在的请求 ID
4. 导致 `KeyError`

## 解决方案

参考张量并行模式的实现，在合并批次时需要：

1. **先对新批次执行 prefill**（初始化并处理 prompt）
2. **在 RPC 端合并批次**（调用 `merge_batch` RPC 方法）
3. **在 Worker 端合并批次**（调用 `Batch.merge` 方法）

### 修复后的代码

```python
if new_batch is not None:
    # 1. 加载批次所需的 adapters
    if not getattr(self.args, 'no_lora', False) and new_batch.adapter_dirs:
        await self._load_adapters(new_batch.adapter_dirs)
    
    # 2. 对新批次执行 prefill（初始化并处理 prompt）
    reqs_rpc = [req.to_rpc_obj() for req in new_batch.reqs]
    await self.model_rpc.init_batch(new_batch.batch_id, reqs_rpc)
    req_to_out_token_id = await self.model_rpc.prefill_batch(new_batch.batch_id)
    
    # 3. 将第一个 token 添加到新批次的请求中
    for req_id, (new_token_id, new_gen_metadata) in req_to_out_token_id.items():
        req = new_batch.id_to_reqs[req_id]
        req.output_ids.append(new_token_id)
        req.output_metadata_list.append(new_gen_metadata)
    
    # 4. 标记并处理新批次中已完成的请求
    eos_id = getattr(self.args, 'eos_id', 2)
    has_new_finished = new_batch.mark_finished_req(eos_id)
    if has_new_finished:
        await self._handle_finish_req(new_batch, has_new_finished)
    
    # 5. 合并到当前批次
    if self.current_batch is None:
        self.current_batch = new_batch
    else:
        if not new_batch.is_clear():
            # 先在 RPC 端合并
            await self.model_rpc.merge_batch(self.current_batch.batch_id, new_batch.batch_id)
            # 再在 Worker 端合并
            self.current_batch.merge(new_batch)
```

## 关键改进

### 1. 新批次的 Prefill
- 新请求必须先执行 prefill 处理 prompt
- 生成第一个 token
- 在 RPC 端创建 `InferBatch` 对象

### 2. RPC 端的批次合并
- 调用 `model_rpc.merge_batch()` 在 RPC 端合并 `InferBatch` 对象
- 确保 RPC 端的 `requests_idx_mapping` 包含所有请求
- 避免后续 `filter_batch` 调用时的 `KeyError`

### 3. 完成请求的处理
- 新批次 prefill 后可能有请求立即完成（如遇到 EOS）
- 需要在合并前处理这些完成的请求
- 只合并未完成的请求

## 与张量并行模式的对比

### 张量并行模式（manager.py）

```python
# 对新批次执行 prefill
await self._prefill_batch(new_mini_batch, minibatch=True)

# 如果新批次没有清空，合并批次
if not new_mini_batch.is_clear():
    await self._merge_batch(self.running_batch, new_mini_batch)  # RPC 端
    self.running_batch.merge(new_mini_batch)  # Worker 端
```

### 数据并行模式（gpu_worker.py）

```python
# 对新批次执行 prefill
reqs_rpc = [req.to_rpc_obj() for req in new_batch.reqs]
await self.model_rpc.init_batch(new_batch.batch_id, reqs_rpc)
await self.model_rpc.prefill_batch(new_batch.batch_id)

# 处理完成的请求
has_new_finished = new_batch.mark_finished_req(eos_id)
if has_new_finished:
    await self._handle_finish_req(new_batch, has_new_finished)

# 如果新批次没有清空，合并批次
if not new_batch.is_clear():
    await self.model_rpc.merge_batch(self.current_batch.batch_id, new_batch.batch_id)  # RPC 端
    self.current_batch.merge(new_batch)  # Worker 端
```

**核心逻辑一致**：都是先 prefill，再在 RPC 端和 Worker 端同步合并。

## 测试验证

修复后，系统应该能够：

1. ✅ 正确处理新请求加入现有批次的场景
2. ✅ 避免 `KeyError` 错误
3. ✅ 正确同步 Worker 端和 RPC 端的批次状态
4. ✅ 正确处理批次中完成的请求

## 相关文件

- `S-LoRA/slora/server/router/gpu_worker.py` - 主要修复文件
- `S-LoRA/slora/server/router/model_infer/infer_batch.py` - 添加了容错处理（临时修复）
- `S-LoRA/slora/server/router/manager.py` - 参考的张量并行实现

## 注意事项

1. **临时修复已移除**：之前在 `infer_batch.py` 中添加的容错处理（跳过不存在的请求）应该不再需要，因为根本问题已解决
2. **性能影响**：新批次的 prefill 会增加一些延迟，但这是正确性所必需的
3. **内存管理**：确保在合并前处理完成的请求，避免不必要的内存占用

## 修复日期

2026-01-31
