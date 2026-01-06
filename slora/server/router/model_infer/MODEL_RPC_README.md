# model_rpc.py 文件说明

## 📖 文件概述

`model_rpc.py` 是 S-LoRA 系统中负责**模型推理**和**LoRA 适配器管理**的核心文件。它就像一个"服务员"，接收来自路由器的请求，然后指挥 GPU 上的模型进行推理计算。

### 简单理解

想象一下餐厅的场景：
- **Router（路由器）** = 前台接待员，接收客户点单
- **ModelRpcServer（本文件）** = 后厨服务员，负责做菜（推理）
- **GPU** = 厨房，实际做菜的地方

这个文件就是连接"前台"和"厨房"的桥梁。

---

## 🏗️ 文件结构

这个文件主要包含两个大"类"（可以理解为两种不同的服务员）：

### 1. ModelRpcServer（服务端）
- **作用**：真正在 GPU 上干活的服务员
- **位置**：在 GPU 进程/线程中运行
- **职责**：执行模型推理、管理适配器、查询内存

### 2. ModelRpcClient（客户端）
- **作用**：前台的服务员，负责接收订单并转发给后厨
- **位置**：在路由器进程中运行
- **职责**：封装调用，让路由器可以方便地使用

---

## 🔧 主要功能详解

### 一、模型初始化（exposed_init_model）

**作用**：启动时加载基础模型和所有 LoRA 适配器

**通俗解释**：
- 就像餐厅开业前，要把所有食材（模型）和调料（适配器）都准备好
- 把模型加载到 GPU 显存中
- 把所有适配器的信息记录下来，但先不加载到显存

**关键步骤**：
1. 加载基础模型（比如 LLaMA）
2. 记录所有适配器的位置
3. 初始化内存管理器

---

### 二、适配器加载（exposed_load_adapters）

**作用**：把需要的 LoRA 适配器加载到 GPU 显存

**通俗解释**：
- 就像客人点菜时，从仓库取出对应的调料（适配器）
- 把调料放到厨房（GPU 显存）里，准备使用

**参数**：
- `adapter_dirs`：要加载的适配器列表（比如 `['adapter_001', 'adapter_002']`）
- `prefetch`：是否为预取模式（提前准备，不等点单）

**工作流程**：
```
路由器说："我需要 adapter_001 和 adapter_002"
    ↓
服务端检查：这两个适配器在显存里吗？
    ↓
如果不在：从磁盘/CPU 内存加载到 GPU 显存
    ↓
加载完成，可以开始推理了
```

---

### 三、适配器卸载（exposed_offload_adapters）

**作用**：把不需要的适配器从 GPU 显存中移除，释放空间

**通俗解释**：
- 就像用完的调料放回仓库，腾出厨房空间
- 只保留当前正在使用的适配器

**参数**：
- `reserve_dirs`：要保留的适配器列表（不卸载这些）
- 如果传入空列表，则卸载所有适配器

**工作流程**：
```
路由器说："保留 adapter_001，其他都卸载"
    ↓
服务端检查：显存中有哪些适配器？
    ↓
保留 adapter_001，卸载其他所有适配器
    ↓
释放显存空间
```

---

### 四、批次推理（exposed_prefill_batch / exposed_decode_batch）

**作用**：对一批请求进行推理计算

**通俗解释**：
- **Prefill（预填充）**：第一次处理请求，就像第一次做菜，需要准备所有食材
- **Decode（解码）**：继续生成后续内容，就像继续做下一道菜，可以复用一些东西

**两种模式的区别**：

| 模式 | 特点 | 比喻 |
|------|------|------|
| **Prefill** | 处理输入文本，生成第一个 token | 第一次做菜，准备时间长 |
| **Decode** | 继续生成后续 token | 继续做菜，速度快 |

**工作流程**：
```
1. 接收一批请求（batch）
2. 准备输入数据（input_ids）
3. 调用模型进行推理
4. 采样生成下一个 token
5. 返回结果给路由器
```

---

### 五、适配器统计信息（exposed_update_adapter_stats）

**作用**：记录哪些适配器被使用了，更新使用统计

**通俗解释**：
- 就像记录"哪些调料被用过了"
- 用于后续的智能淘汰（优先保留常用的适配器）

**更新内容**：
- 使用次数
- 最后访问时间
- 当前活跃请求数

---

### 六、内存查询（exposed_check_lora_memory）

**作用**：查询当前 LoRA 适配器占用的显存情况

**通俗解释**：
- 就像查看"厨房还有多少空间"
- 返回详细的内存使用信息

**返回信息**：
```python
{
    'total_cells': 60000,      # 总空间
    'used_cells': 45000,       # 已使用
    'available_cells': 15000,  # 可用空间
    'usage_ratio': 0.75,       # 使用率 75%
    'num_adapters': 20         # 已加载的适配器数量
}
```

---

### 七、阈值淘汰（exposed_trigger_threshold_eviction）

**作用**：当内存使用率超过阈值时，自动淘汰低分适配器

**通俗解释**：
- 就像厨房空间不够时，把不常用的调料放回仓库
- 优先保留正在使用和常用的适配器

**参数**：
- `preserve_dirs`：必须保留的适配器（当前正在使用的）
- `threshold`：触发淘汰的阈值（比如 0.9 = 90%）
- `evict_ratio`：淘汰比例（比如 0.2 = 淘汰 20%）

**工作流程**：
```
1. 检查内存使用率是否超过阈值（比如 90%）
2. 如果超过：
   - 计算所有适配器的"分数"（使用频率、最近访问等）
   - 选择分数最低的适配器
   - 卸载这些低分适配器
   - 释放显存空间
3. 返回淘汰结果
```

---

## 🔄 RPC 通信机制

### 什么是 RPC？

**RPC（Remote Procedure Call）** = 远程过程调用

**通俗解释**：
- 就像打电话点外卖
- 你（路由器）打电话（RPC 调用）给餐厅（GPU 进程）
- 餐厅接电话，做菜，然后告诉你结果

### 单卡 vs 多卡

**单卡模式（world_size = 1）**：
- 不需要 RPC，直接调用
- 就像在餐厅里直接跟厨师说话

**多卡模式（world_size > 1）**：
- 需要 RPC 通信
- 就像打电话给多个餐厅，协调工作

---

## 📊 数据流向

### 请求处理流程

```
路由器（Router）
    ↓
    | 调用 RPC 方法
    ↓
ModelRpcClient（客户端）
    ↓
    | 如果是多卡：通过网络发送
    | 如果是单卡：直接调用
    ↓
ModelRpcServer（服务端）
    ↓
    | 在 GPU 上执行
    ↓
模型推理引擎
    ↓
返回结果
    ↓
ModelRpcClient
    ↓
路由器
```

### 适配器加载流程

```
1. 路由器：需要加载 adapter_001
    ↓
2. Client：调用 load_adapters(['adapter_001'])
    ↓
3. Server：从适配器列表中找到 adapter_001
    ↓
4. Server：检查显存空间
    ↓
5. Server：如果空间不够，触发淘汰
    ↓
6. Server：加载适配器到 GPU 显存
    ↓
7. Server：更新索引和统计信息
    ↓
8. 返回：加载完成
```

---

## 🎯 关键概念解释

### 1. Batch（批次）

**含义**：一批请求打包在一起处理

**为什么需要批次？**
- 提高 GPU 利用率
- 就像餐厅一次做多道菜，比一道一道做更高效

### 2. Prefill vs Decode

| 阶段 | 输入 | 输出 | 特点 |
|------|------|------|------|
| **Prefill** | 用户输入的完整文本 | 第一个生成的 token | 需要处理所有输入，较慢 |
| **Decode** | 上一个生成的 token | 下一个 token | 只需处理一个 token，较快 |

**比喻**：
- Prefill = 第一次做菜，需要准备所有食材
- Decode = 继续做菜，只需要加新食材

### 3. 显存管理

**问题**：GPU 显存有限，不能加载所有适配器

**解决方案**：
- 只加载当前需要的适配器
- 使用完及时卸载
- 当空间不够时，淘汰不常用的适配器

---

## 🔍 常用方法速查

### 初始化相关
- `exposed_init_model()` - 初始化模型和适配器

### 适配器管理
- `exposed_load_adapters()` - 加载适配器到显存
- `exposed_offload_adapters()` - 卸载适配器，释放显存
- `exposed_update_adapter_stats()` - 更新适配器使用统计
- `exposed_decrease_request_counts()` - 减少适配器的请求计数

### 推理相关
- `exposed_prefill_batch()` - 预填充批次推理
- `exposed_decode_batch()` - 解码批次推理
- `exposed_add_batch()` - 添加新批次
- `exposed_filter_batch()` - 过滤批次中的请求
- `exposed_merge_batch()` - 合并两个批次
- `exposed_remove_batch()` - 移除批次

### 内存管理
- `exposed_check_lora_memory()` - 查询内存使用情况
- `exposed_trigger_threshold_eviction()` - 触发阈值淘汰

---

## 💡 实际使用示例

### 示例 1：加载适配器并推理

```python
# 1. 加载适配器
await model_rpc.load_adapters(['adapter_001', 'adapter_002'])

# 2. 创建批次
await model_rpc.init_batch(batch_id, requests)

# 3. 预填充推理
result = await model_rpc.prefill_batch(batch_id)

# 4. 继续解码（生成后续内容）
for i in range(10):
    result = await model_rpc.decode_batch(batch_id)
```

### 示例 2：查询内存并触发淘汰

```python
# 1. 查询内存使用情况
usage = await model_rpc.check_lora_memory()
print(f"内存使用率: {usage['usage_ratio']:.1%}")

# 2. 如果使用率超过 90%，触发淘汰
if usage['usage_ratio'] > 0.9:
    preserve = ['adapter_001']  # 保留当前使用的
    result = await model_rpc.trigger_threshold_eviction(
        preserve_dirs=preserve,
        threshold=0.9,
        evict_ratio=0.2
    )
    print(f"淘汰了 {result['evicted_count']} 个适配器")
```

---

## ⚠️ 注意事项

### 1. 线程安全
- RPC 调用是异步的，需要正确使用 `await`
- 多卡模式下，需要等待所有卡完成

### 2. 内存管理
- 显存有限，不要同时加载太多适配器
- 及时卸载不需要的适配器
- 使用阈值淘汰机制自动管理

### 3. 错误处理
- 如果显存不足，加载会失败
- 需要检查返回值，处理异常情况

---

## 🎓 总结

`model_rpc.py` 文件是 S-LoRA 系统的"执行层"：

1. **接收任务**：从路由器接收推理请求
2. **管理资源**：管理 GPU 显存和适配器
3. **执行推理**：在 GPU 上执行模型计算
4. **返回结果**：把推理结果返回给路由器

**核心思想**：
- 把复杂的 GPU 操作封装成简单的函数调用
- 让路由器可以方便地使用模型和适配器
- 自动管理显存，提高资源利用率

---

## 📚 相关文件

- `infer_adapter.py` - 适配器管理的具体实现
- `infer_batch.py` - 批次数据的结构定义
- `manager.py` - 路由器管理器，调用本文件的方法

---

**最后更新**：2024年

