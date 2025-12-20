# LoRA 适配器评分机制集成说明

本文档说明 LoRA 适配器评分机制的集成情况。

## 已完成的集成工作

### 1. 数据结构（已实现）

在 `infer_adapter.py` 的 `InferAdapter` 类中添加了以下数据结构：

```python
adapter_scores: Dict[str, float]          # 适配器综合分数
score_update_counter: Dict[str, int]      # 使用次数统计
last_access_time: Dict[str, float]        # 最后访问时间
total_use_duration: Dict[str, float]      # 累计使用时长
current_request_count: Dict[str, int]     # 当前活跃请求数
load_time: Dict[str, float]               # 适配器加载时间
```

### 2. 统计方法（已实现）

- `update_adapter_stats_batch()`: 按批次更新适配器使用统计
- `calculate_adapter_score()`: 计算适配器综合分数（支持多维度权重）
- `get_adapters_by_score()`: 获取按分数排序的适配器列表
- `decrease_request_count()`: 减少适配器的当前请求计数
- `print_adapter_stats()`: 打印统计信息（调试用）

### 3. RPC 方法（新增）

在 `model_rpc.py` 中添加了两个新的 RPC 方法：

#### `exposed_update_adapter_stats(adapter_dirs)`
- 作用：更新适配器使用统计信息
- 调用时机：每次 decode 批次时
- 参数：当前批次中使用的适配器目录列表

#### `exposed_decrease_request_counts(adapter_dirs)`
- 作用：减少适配器的当前请求计数
- 调用时机：请求完成时
- 参数：完成请求使用的适配器目录列表

### 4. 客户端方法（新增）

在 `ModelRpcClient` 中添加了对应的异步方法：

```python
async def update_adapter_stats(self, adapter_dirs)
async def decrease_request_counts(self, adapter_dirs)
```

### 5. 管理器集成（已完成）

#### 在 `_decode_batch()` 中更新统计

```python
async def _decode_batch(self, batch:Batch):
    self.req_queue.update_counter(batch)
    
    # 更新适配器使用统计信息（在推理前）
    if not self.input_params.no_lora:
        adapter_dirs_list = list(batch.adapter_dirs)
        ret = []
        for tp_rank in range(self.world_size):
            ret.append(self.model_rpcs[tp_rank].update_adapter_stats(adapter_dirs_list))
        await asyncio.gather(*ret)
    
    # ... 执行推理 ...
```

**作用**：
- 在每次 decode 推理前更新适配器统计
- 记录使用次数、最后访问时间、当前请求数

#### 在 `_handle_finish_req()` 中减少计数

```python
async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False):
    if has_new_finished_req:
        # 记录完成的请求使用的适配器（在 filter_finished 之前）
        finished_adapter_dirs = []
        if not self.input_params.no_lora:
            for req in batch.reqs:
                if req.has_generate_finished:
                    finished_adapter_dirs.append(req.adapter_dir)
        
        batch.filter_finished()
        
        # 减少完成请求的适配器的当前请求计数
        if finished_adapter_dirs and not self.input_params.no_lora:
            ret = []
            for tp_rank in range(self.world_size):
                ret.append(self.model_rpcs[tp_rank].decrease_request_counts(finished_adapter_dirs))
            await asyncio.gather(*ret)
        
        # ... 淘汰适配器等后续操作 ...
```

**作用**：
- 在请求完成时减少对应适配器的当前请求计数
- 保持统计数据的准确性

## 工作流程

```
请求到达
  ↓
_decode_batch() 调用
  ↓
update_adapter_stats() - 更新统计（使用次数+1，更新访问时间）
  ↓
执行推理
  ↓
请求完成
  ↓
_handle_finish_req() 调用
  ↓
decrease_request_counts() - 减少请求计数
  ↓
淘汰适配器（基于使用状态，暂不使用分数）
```

## 统计信息说明

### 使用次数（score_update_counter）
- 每次 decode 时增加
- 反映适配器的总使用频率

### 最后访问时间（last_access_time）
- 每次 decode 时更新为当前时间
- 用于计算时间衰减分数

### 当前请求数（current_request_count）
- decode 时增加，请求完成时减少
- 反映适配器的当前活跃度

### 累计使用时长（total_use_duration）
- 目前未自动更新（需要在适当位置手动调用 `update_adapter_duration()`）
- 可以在请求完成时计算请求总时长并更新

## 评分计算

`calculate_adapter_score()` 方法计算综合分数，考虑四个维度：

1. **使用次数分数**（默认权重 0.3）：归一化的使用次数
2. **最近访问分数**（默认权重 0.3）：基于时间衰减的分数
3. **使用时长分数**（默认权重 0.2）：归一化的累计使用时长
4. **活跃请求分数**（默认权重 0.2）：归一化的当前请求数

公式：
```
total_score = w1 * usage_score + w2 * recency_score + w3 * duration_score + w4 * active_score
```

## 调试方法

可以在代码中调用 `print_adapter_stats()` 查看统计信息：

```python
# 在 manager.py 或 model_rpc.py 中
self.infer_adapter.print_adapter_stats(top_k=10)
```

输出示例：
```
================================================================================
适配器统计信息 (显存中共 5 个适配器)
================================================================================
排名    适配器                                  分数        使用次数    活跃请求  
--------------------------------------------------------------------------------
1     lora-64-1                             0.8523    150       3         
2     lora-32-1                             0.7214    120       2         
3     lora-16-1                             0.6108    90        1         
...
================================================================================
```

## 未实现的部分

### 基于分数的淘汰策略

当前淘汰策略仍然是"使用即保留，不使用即淘汰"。

如需实现基于分数的智能淘汰，可以修改 `offload_adapters()` 方法：

```python
def offload_adapters_with_score(self, max_adapters=10):
    """
    基于分数的智能淘汰策略（示例代码，未实现）
    
    保留分数最高的 max_adapters 个适配器
    """
    # 获取所有适配器的分数
    scored_adapters = self.get_adapters_by_score(ascending=False)
    
    # 保留前 N 个高分适配器
    if len(scored_adapters) > max_adapters:
        reserve_adapters = [a[0] for a in scored_adapters[:max_adapters]]
        self.offload_adapters(reserve_adapters)
```

### 使用时长统计

`total_use_duration` 目前未自动更新，需要在适当位置添加：

```python
# 在请求开始时记录时间
req.start_time = time.time()

# 在请求完成时更新时长
duration = time.time() - req.start_time
infer_adapter.update_adapter_duration(req.adapter_dir, duration)
```

## 注意事项

1. **性能影响**：统计更新操作非常轻量，对性能影响极小
2. **多 GPU 支持**：所有统计方法都支持多 GPU 环境
3. **内存开销**：统计数据结构占用内存极小（每个适配器约几十字节）
4. **数据一致性**：适配器卸载时会自动清理对应的统计数据

## 总结

✅ 数据结构已添加  
✅ 统计方法已实现  
✅ RPC 方法已添加  
✅ 管理器已集成  
✅ 统计信息自动更新  
⏸️ 基于分数的淘汰（暂不实现）  
⏸️ 使用时长自动统计（可选）  

评分机制的基础设施已完全就绪，可以随时启用基于分数的智能淘汰策略。

