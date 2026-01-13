# Adapter 管理更新说明

## 更新日期
2025-01-13

## 更新内容

本次更新将原 `RouterManager` (manager.py) 中成熟的 Adapter 管理策略集成到数据并行的 GPU Worker 设计中。

## 1. Tasks.md 更新

### 新增 Task 2.7: 实现 Adapter 管理

添加了 4 个子任务，全部标记为 Phase 1 必需：

#### 2.7.1 实现 Adapter Rank 配置
- 初始化 `self.lora_ranks` 字典
- 读取所有 adapter 的 rank 配置
- 用于 ReqQueue 计算显存占用

#### 2.7.2 实现实际内存占用跟踪
- 初始化 `self.actual_adapter_memory_usage`
- 实现 `_update_actual_adapter_usage()` 方法
- 通过 RPC 查询实际占用

#### 2.7.3 实现基本的 Adapter 加载/卸载
- 实现 `_load_adapters()` 方法
- 在批次生成后加载 adapters
- 加载后更新内存占用

#### 2.7.4 更新 ReqQueue 调用
- 传递 `lora_ranks` 参数
- 传递 `actual_adapter_size` 参数

### 新增 Task 2.8: 编写 Adapter 管理单元测试
- 测试 lora_ranks 初始化
- 测试内存占用查询
- 测试 adapter 加载流程

## 2. Design.md 更新

### 2.1 更新 GPUWorker.__init__()

添加了 Adapter 管理相关的实例变量：

```python
# Adapter 管理（借鉴 manager.py）
self.lora_ranks = {}  # adapter_dir -> rank 映射
self.actual_adapter_memory_usage = 0  # 实际 adapter 显存占用（cells）

# 模型 RPC 客户端
self.model_rpc = None  # 用于与模型进程通信

# EOS token ID
self.eos_id = args.eos_id
```

### 2.2 新增方法

#### `_setup_adapter_config()`
```python
def _setup_adapter_config(self) -> None:
    """初始化 Adapter 配置（借鉴 manager.py）"""
    from slora.models.peft.lora_adapter import get_lora_config
    
    self.lora_ranks = {}
    for lora_dir in self.args.lora_dirs:
        config, _ = get_lora_config(lora_dir, self.args.dummy)
        self.lora_ranks[lora_dir] = config["r"]
    self.lora_ranks[None] = 0
```

#### `_update_actual_adapter_usage()`
```python
async def _update_actual_adapter_usage(self) -> None:
    """查询并更新实际的 adapter 内存占用（借鉴 manager.py）"""
    if self.args.no_lora:
        self.actual_adapter_memory_usage = 0
        return
    
    try:
        memory_info = await self.model_rpc.check_lora_memory()
        if memory_info:
            adapter_cells_list = memory_info.get('adapter_cells', [])
            self.actual_adapter_memory_usage = sum(adapter_cells_list)
    except Exception as e:
        print(f"[Worker {self.worker_id}] 警告：无法查询 adapter 占用: {e}")
```

#### `_load_adapters()`
```python
async def _load_adapters(self, adapter_dirs: set) -> None:
    """加载 adapters（借鉴 manager.py）"""
    if self.args.no_lora or not adapter_dirs:
        return
    
    await self.model_rpc.load_adapters(list(adapter_dirs))
    await self._update_actual_adapter_usage()
```

#### `_handle_finish_req()`
```python
async def _handle_finish_req(self, batch: Batch, has_new_finished_req: bool) -> None:
    """处理完成的请求（借鉴 manager.py）"""
    if not has_new_finished_req:
        return
    
    batch.filter_finished()
    
    if batch.is_clear():
        self.current_batch = None
    
    # Phase 2 将添加智能淘汰策略
```

### 2.3 更新 `_process_requests()` 方法

添加了 Adapter 管理逻辑：

```python
async def _process_requests(self) -> List[dict]:
    # 生成新批次（传递 lora_ranks 和实际占用）
    new_batch = self.req_queue.generate_new_batch(
        self.current_batch,
        self.lora_ranks,  # Adapter rank 配置
        actual_adapter_size=self.actual_adapter_memory_usage  # 实际显存占用
    )
    
    if new_batch is not None:
        # 加载批次需要的 adapters
        await self._load_adapters(new_batch.adapter_dirs)
        
        # 合并到当前批次
        ...
    
    # 执行推理
    if self.current_batch is not None:
        outputs = await self._infer_batch(self.current_batch)
        
        # 标记完成的请求
        has_new_finished_req = self.current_batch.mark_finished_req(self.eos_id)
        
        # 处理完成的请求
        if has_new_finished_req:
            await self._handle_finish_req(self.current_batch, has_new_finished_req)
        
        return responses
```

### 2.4 更新 Key Methods 说明

添加了 Adapter 管理相关的方法说明：

- `_setup_adapter_config()`: 初始化 Adapter rank 配置（Phase 1）
- `_update_actual_adapter_usage()`: 查询实际 adapter 显存占用（Phase 1）
- `_load_adapters()`: 加载 adapters（Phase 1）
- `_handle_finish_req()`: 处理完成的请求（Phase 1 基础版，Phase 2 添加淘汰）

添加了"借鉴 manager.py 的 Adapter 管理"部分：
- Adapter rank 配置管理（`lora_ranks` 字典）
- 实际内存占用跟踪（`actual_adapter_memory_usage`）
- Adapter 加载时机（批次生成后）
- 加载后更新内存占用统计

## 3. 设计理念

### 3.1 为什么借鉴 manager.py？

1. **成熟的实现**：manager.py 中的 Adapter 管理已经在张量并行中验证
2. **代码一致性**：数据并行和张量并行使用相同的策略
3. **减少开发工作**：复用已有代码，降低风险
4. **易于维护**：相似的代码结构便于理解和调试

### 3.2 关键设计决策

#### Adapter Rank 配置
- **作用**：ReqQueue 需要 lora_ranks 来计算 adapter 显存占用
- **实现**：读取所有 adapter 的配置文件，提取 rank 值
- **必要性**：🔴 必需（ReqQueue 无法正常工作）

#### 实际内存占用跟踪
- **作用**：精确的并发控制，避免 OOM
- **实现**：通过 RPC 查询 check_lora_memory()
- **必要性**：🔴 必需（准确的显存管理）

#### Adapter 加载时机
- **作用**：确保推理前 adapters 已就绪
- **实现**：批次生成后立即加载
- **必要性**：🔴 必需（推理依赖 adapters）

## 4. Phase 划分

### Phase 1（当前）- 基础功能
- ✅ Adapter Rank 配置
- ✅ 实际内存占用跟踪
- ✅ 基本的 Adapter 加载
- ✅ 简单的请求完成处理

### Phase 2（优化）- 智能淘汰
- ⏳ 智能淘汰策略（trigger_threshold_eviction）
- ⏳ 请求计数管理（decrease_request_counts）
- ⏳ 保护机制（preserve_dirs）
- ⏳ 监控和调试工具

### Phase 3（完善）- 性能调优
- ⏳ 性能分析和优化
- ⏳ 错误处理增强
- ⏳ 日志和统计工具

## 5. 实施顺序

1. **Task 2.7.1**: 实现 Adapter Rank 配置
   - 最基础，其他功能依赖它

2. **Task 2.7.2**: 实现实际内存占用跟踪
   - 为 ReqQueue 提供准确的显存信息

3. **Task 2.7.3**: 实现 Adapter 加载
   - 集成到批次处理流程

4. **Task 2.7.4**: 更新 ReqQueue 调用
   - 确保所有参数正确传递

5. **Task 2.8**: 编写单元测试
   - 验证所有功能正常工作

## 6. 与原 manager.py 的对比

| 特性 | manager.py (张量并行) | GPUWorker (数据并行) |
|------|---------------------|---------------------|
| **lora_ranks** | ✅ 全局一份 | ✅ 每个 Worker 一份 |
| **actual_adapter_memory_usage** | ✅ 全局跟踪 | ✅ 每个 Worker 独立跟踪 |
| **_load_adapters()** | ✅ 所有 GPU 同步加载 | ✅ 单个 GPU 加载 |
| **_update_actual_adapter_usage()** | ✅ 查询第一个 RPC | ✅ 查询自己的 RPC |
| **智能淘汰** | ✅ Phase 1 已实现 | ⏳ Phase 2 实现 |
| **RPC 通信** | ✅ 多个 RPC 客户端 | ✅ 单个 RPC 客户端 |

## 7. 注意事项

### 7.1 必需的依赖
- `get_lora_config`: 从 `slora.models.peft.lora_adapter` 导入
- `Batch`: 从 `slora.server.io_struct` 导入
- RPC 客户端：需要实现与模型进程的通信

### 7.2 参数传递
- `generate_new_batch()` 必须传递 `lora_ranks` 和 `actual_adapter_size`
- 否则 ReqQueue 无法正确计算显存占用

### 7.3 错误处理
- `_update_actual_adapter_usage()` 应该捕获异常
- 查询失败时使用保守估计（当前值不变）

## 8. 测试策略

### 8.1 单元测试
- 测试 lora_ranks 初始化是否正确
- 测试内存占用查询是否准确
- 测试 adapter 加载流程是否完整

### 8.2 集成测试
- 测试 ReqQueue 与 Adapter 管理的集成
- 测试批次生成时的 adapter 加载
- 测试请求完成时的状态更新

### 8.3 性能测试
- 测试 adapter 加载的延迟
- 测试内存占用查询的开销
- 测试整体吞吐量影响

## 9. 后续工作

### Phase 2 需要添加的功能
1. **智能淘汰策略**
   - 实现 `trigger_threshold_eviction()`
   - 配置淘汰阈值和比例
   - 保护当前批次的 adapters

2. **请求计数管理**
   - 实现 `decrease_request_counts()`
   - 跟踪每个 adapter 的活跃请求数

3. **监控工具**
   - 实现 `_print_lora_status()`
   - 实现 `_print_eviction_summary()`
   - 添加统计信息输出

## 10. 总结

本次更新成功将原 RouterManager 中成熟的 Adapter 管理策略集成到数据并行的 GPU Worker 设计中。通过借鉴已验证的实现，我们能够：

1. ✅ 复用成熟的代码逻辑
2. ✅ 保持代码一致性
3. ✅ 减少开发风险
4. ✅ 提高可维护性

Phase 1 专注于基础功能，确保 ReqQueue 能够正常工作。Phase 2 将添加智能淘汰等优化功能，进一步提高 adapter 利用率和系统性能。

---

**文档版本**: v1.0  
**创建日期**: 2025-01-13  
**状态**: 设计更新完成
