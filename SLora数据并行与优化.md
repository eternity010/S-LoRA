# S-LoRA 数据并行与优化方案

## 文档概述

本文档深入分析 S-LoRA 项目的多卡并行优化方案，对比张量并行与数据并行两种模式，并提出针对 S-LoRA 特点的优化策略。

---

## 一、当前架构分析

### 1.1 现有实现

S-LoRA 当前采用**张量并行（Tensor Parallelism）**架构：

```python
# 当前架构特点
- 单一 RouterManager 统一管理所有请求
- 多个 GPU 通过 world_size 和 tp_rank 协同工作
- 所有 GPU 必须同步处理同一批次请求
- 使用 NCCL 进行 GPU 间通信（all_reduce）
```

**代码位置**：
- `slora/server/router/manager.py` - 路由管理器
- `slora/server/router/model_infer/model_rpc.py` - 模型 RPC 通信
- `slora/common/basemodel/basemodel.py` - 基础模型（支持 tp_rank）

### 1.2 张量并行的执行流程

```python
# 以 decode_batch 为例
async def _decode_batch(self, batch: Batch):
    # 1. 所有 GPU 同时执行 decode
    rets = [
        self.model_rpcs[tp_rank].decode_batch(batch.batch_id) 
        for tp_rank in range(self.world_size)
    ]
    
    # 2. 等待所有 GPU 完成（同步点）
    ans = await asyncio.gather(*rets)
    
    # 3. 只使用第一个 GPU 的结果（其他 GPU 结果相同）
    req_to_out_token_id = obtain(ans[0])
```

**关键问题**：
- 所有 GPU 必须等待最慢的那个完成
- GPU 利用率可能不均衡
- 吞吐量受限于单批次处理能力

---

## 二、张量并行 vs 数据并行

### 2.1 对比表格

| 维度 | 张量并行 | 数据并行 |
|------|---------|---------|
| **模型分布** | 模型权重切分到多卡 | 每卡完整模型 |
| **请求处理** | 多卡协同处理同一请求 | 每卡独立处理不同请求 |
| **GPU 通信** | 频繁（all_reduce） | 几乎无 |
| **吞吐量** | 受限于通信开销 | 近似线性扩展（N 卡 ≈ N 倍） |
| **单请求延迟** | 低（多卡并行） | 不变（单卡处理） |
| **显存需求** | 低（权重分摊） | 高（每卡完整模型） |
| **扩展性** | 受通信带宽限制 | 优秀（加卡即加速） |
| **故障容错** | 任一卡故障全部停止 | 单卡故障不影响其他卡 |
| **适用场景** | 超大模型、低延迟 | 高并发、高吞吐 |

### 2.2 性能对比（理论分析）

假设场景：4 张 A100 GPU，Llama-7B 模型

**张量并行模式**：
```
单批次大小: 32 requests
单批次处理时间: 100ms（包含通信开销）
吞吐量: 32 / 0.1 = 320 req/s
GPU 利用率: 70%（通信等待导致）
```

**数据并行模式**：
```
每卡批次大小: 32 requests
每卡处理时间: 80ms（无通信开销）
总吞吐量: 4 × (32 / 0.08) = 1600 req/s
GPU 利用率: 95%（几乎无等待）
```

**结论**：数据并行吞吐量可达张量并行的 **4-5 倍**

---

## 三、S-LoRA 特点分析

### 3.1 S-LoRA 的独特需求

1. **服务大量 LoRA adapters**
   - 数百到数千个不同的 adapter
   - 每个 adapter 只有几 MB（相比基座模型很小）
   - 不同用户使用不同 adapter

2. **高并发请求场景**
   - 多个用户同时发送请求
   - 请求使用的 adapter 各不相同
   - 需要频繁切换 adapter

3. **基座模型大小适中**
   - 主要支持 7B-30B 模型
   - 单张 A100（80GB）可以容纳完整模型
   - 不需要张量并行来突破显存限制

4. **Adapter 加载开销**
   - 从 CPU 加载 adapter 到 GPU 需要时间
   - Adapter 缓存命中率影响性能
   - 频繁的 adapter 切换是性能瓶颈

### 3.2 为什么数据并行更适合 S-LoRA

**匹配度分析**：

✅ **高并发场景** → 数据并行天然支持并发
- 不同的请求可以分配到不同的 GPU
- 无需等待其他 GPU 完成

✅ **Adapter 多样性** → 减少 adapter 冲突
- 每张卡独立管理自己的 adapter 缓存
- 可以根据 adapter 亲和性路由请求
- 减少 adapter 加载/卸载频率

✅ **模型大小适中** → 显存足够
- 7B 模型约 14GB（FP16）
- A100 80GB 可以容纳模型 + 大量 adapter + KV cache
- 不需要张量并行来节省显存

✅ **吞吐量优先** → 数据并行优势明显
- S-LoRA 的目标是服务大量用户
- 吞吐量比单请求延迟更重要
- 数据并行可以实现近线性扩展

---

## 四、数据并行优化方案

### 4.1 架构设计

```
                    ┌─────────────────┐
                    │  Request Router │
                    │  (负载均衡器)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
         │ GPU 0   │    │ GPU 1   │    │ GPU 2   │
         │ Worker  │    │ Worker  │    │ Worker  │
         ├─────────┤    ├─────────┤    ├─────────┤
         │ 基座模型 │    │ 基座模型 │    │ 基座模型 │
         │ Adapter │    │ Adapter │    │ Adapter │
         │ Cache   │    │ Cache   │    │ Cache   │
         └─────────┘    └─────────┘    └─────────┘
              │              │              │
              └──────────────┴──────────────┘
                             │
                      ┌──────▼──────┐
                      │   Response  │
                      │   Collector │
                      └─────────────┘
```

### 4.2 核心组件

#### 4.2.1 Request Router（请求路由器）

**职责**：
- 接收传入的请求
- 根据策略选择目标 GPU Worker
- 维护各 Worker 的状态信息

**路由策略**：

1. **轮询（Round Robin）**
   ```python
   def route_round_robin(request):
       gpu_id = self.counter % self.num_gpus
       self.counter += 1
       return gpu_id
   ```
   - 优点：简单、公平
   - 缺点：不考虑负载差异

2. **最少连接（Least Connections）**
   ```python
   def route_least_connections(request):
       queue_lengths = [worker.queue_length for worker in self.workers]
       return queue_lengths.index(min(queue_lengths))
   ```
   - 优点：动态负载均衡
   - 缺点：不考虑 adapter 缓存

3. **Adapter 亲和性（Affinity-based）** ⭐ 推荐
   ```python
   def route_affinity(request):
       adapter = request.adapter_dir
       
       # 1. 优先选择已缓存该 adapter 的 GPU
       cached_gpus = [i for i, w in enumerate(self.workers) 
                      if adapter in w.adapter_cache]
       
       if cached_gpus:
           # 在已缓存的 GPU 中选择负载最低的
           return min(cached_gpus, 
                     key=lambda i: self.workers[i].queue_length)
       
       # 2. 否则选择负载最低的 GPU
       return min(range(self.num_gpus), 
                 key=lambda i: self.workers[i].queue_length)
   ```
   - 优点：最大化缓存命中率，减少加载开销
   - 缺点：实现稍复杂

#### 4.2.2 GPU Worker（GPU 工作进程）

**职责**：
- 在单张 GPU 上运行独立的推理实例
- 管理本地的 adapter 缓存
- 处理分配给它的请求队列

**实现要点**：
```python
class GPUWorker:
    def __init__(self, gpu_id, model_dir, adapter_dirs):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        
        # 加载完整的基座模型
        self.model = load_model(model_dir)
        
        # 初始化 adapter 缓存
        self.adapter_cache = AdapterCache(max_size=100)
        
        # 请求队列
        self.request_queue = Queue()
        
    async def process_requests(self):
        while True:
            request = await self.request_queue.get()
            
            # 确保 adapter 已加载
            if request.adapter not in self.adapter_cache:
                await self.load_adapter(request.adapter)
            
            # 执行推理
            result = await self.infer(request)
            
            # 返回结果
            await self.send_response(result)
```

#### 4.2.3 Adapter Cache Manager（适配器缓存管理器）

**缓存策略**：

1. **LRU（Least Recently Used）** - 基础策略
   ```python
   class LRUAdapterCache:
       def __init__(self, max_size):
           self.cache = OrderedDict()
           self.max_size = max_size
       
       def get(self, adapter_dir):
           if adapter_dir in self.cache:
               # 移到最后（最近使用）
               self.cache.move_to_end(adapter_dir)
               return self.cache[adapter_dir]
           return None
       
       def put(self, adapter_dir, adapter):
           if len(self.cache) >= self.max_size:
               # 淘汰最久未使用的
               self.cache.popitem(last=False)
           self.cache[adapter_dir] = adapter
   ```

2. **热门 Adapter 固定** - 优化策略
   ```python
   class SmartAdapterCache(LRUAdapterCache):
       def __init__(self, max_size, hot_threshold=100):
           super().__init__(max_size)
           self.access_count = defaultdict(int)
           self.hot_adapters = set()
           self.hot_threshold = hot_threshold
       
       def get(self, adapter_dir):
           self.access_count[adapter_dir] += 1
           
           # 标记热门 adapter
           if self.access_count[adapter_dir] >= self.hot_threshold:
               self.hot_adapters.add(adapter_dir)
           
           return super().get(adapter_dir)
       
       def can_evict(self, adapter_dir):
           # 热门 adapter 不淘汰
           return adapter_dir not in self.hot_adapters
   ```

### 4.3 实现阶段

#### Phase 1: 基础数据并行（核心功能）

**目标**：实现基本的数据并行，每张卡独立处理请求

**工作内容**：
1. 创建 `DataParallelRouter` 类
2. 实现 `GPUWorker` 进程管理
3. 实现轮询或最少连接路由策略
4. 修改启动脚本支持 `--parallel-mode data`

**预期效果**：
- 4 卡吞吐量达到单卡的 3-3.5 倍
- 基本的负载均衡

#### Phase 2: Adapter 亲和性路由（性能优化）

**目标**：根据 adapter 缓存情况优化请求分配

**工作内容**：
1. 实现 adapter 缓存状态同步机制
2. 实现亲和性路由算法
3. 添加缓存命中率统计

**预期效果**：
- Adapter 缓存命中率提升 30-50%
- 减少 adapter 加载延迟
- 吞吐量进一步提升 10-20%

#### Phase 3: 跨 GPU Adapter 共享（高级优化）

**目标**：利用 NVLink 实现 GPU 间 adapter 快速共享

**工作内容**：
1. 检测 NVLink 拓扑
2. 实现 GPU Direct P2P 复制
3. 维护全局 adapter 位置索引

**预期效果**：
- Adapter 加载时间减少 50-70%（相比从 CPU 加载）
- 进一步提升吞吐量

#### Phase 4: 混合并行模式（可选）

**目标**：支持张量并行 + 数据并行的混合模式

**工作内容**：
1. 实现 GPU 分组逻辑
2. 组内使用张量并行，组间使用数据并行
3. 支持配置参数 `--parallel-mode hybrid --tp-size 2`

**适用场景**：
- 大模型（30B+）需要张量并行
- 同时希望利用数据并行提升吞吐

---

## 五、性能预估

### 5.1 理论分析

**假设条件**：
- 硬件：4 × A100 80GB
- 模型：Llama-7B
- Adapter：平均 rank=16，约 8MB
- 请求：平均 input=512 tokens，output=128 tokens

**张量并行（当前）**：
```
批次大小: 32
处理时间: 100ms（含通信）
吞吐量: 320 req/s
GPU 利用率: 70%
```

**数据并行（优化后）**：

Phase 1（基础）：
```
每卡批次: 32
每卡时间: 80ms
总吞吐量: 4 × 400 = 1600 req/s
提升: 5x
```

Phase 2（亲和性路由）：
```
缓存命中率: 70% → 85%
Adapter 加载减少: 50%
总吞吐量: 1760 req/s
提升: 5.5x
```

Phase 3（跨 GPU 共享）：
```
缓存未命中时加载时间: 20ms → 5ms
总吞吐量: 1840 req/s
提升: 5.75x
```

### 5.2 实际测试建议

**测试场景**：
1. **低并发**（10 req/s）- 验证正确性
2. **中并发**（100 req/s）- 验证负载均衡
3. **高并发**（500 req/s）- 验证极限吞吐
4. **Adapter 多样性**（100 个不同 adapter）- 验证缓存策略

**关键指标**：
- 吞吐量（requests/second）
- 平均延迟（ms）
- P99 延迟（ms）
- GPU 利用率（%）
- Adapter 缓存命中率（%）

---

## 六、实现挑战与解决方案

### 6.1 挑战 1：进程间通信

**问题**：多个 GPU Worker 进程如何高效通信？

**解决方案**：
- 使用 ZMQ 进行轻量级消息传递
- Router 使用 PUSH/PULL 模式分发请求
- Worker 使用 PUSH 模式返回结果

### 6.2 挑战 2：状态同步

**问题**：Router 如何获知各 Worker 的实时状态？

**解决方案**：
- Worker 定期（每秒）上报状态（队列长度、缓存列表）
- Router 维护状态缓存，路由时使用最新状态
- 使用心跳机制检测 Worker 健康状态

### 6.3 挑战 3：故障恢复

**问题**：某个 GPU Worker 崩溃怎么办？

**解决方案**：
- 实现健康检查机制（心跳超时检测）
- 自动重启失败的 Worker
- 请求重试机制（失败请求重新路由）

### 6.4 挑战 4：显存管理

**问题**：每张卡都加载完整模型，显存够用吗？

**解决方案**：
- 7B 模型（FP16）约 14GB
- Adapter 缓存预留 10GB（可缓存 1000+ adapters）
- KV cache 预留 50GB
- 总计：14 + 10 + 50 = 74GB < 80GB ✅

---

## 七、代码改动范围

### 7.1 新增文件

```
slora/server/router/
├── data_parallel_router.py      # 数据并行路由器
├── gpu_worker.py                # GPU Worker 进程
├── load_balancer.py             # 负载均衡器
└── adapter_cache.py             # Adapter 缓存管理
```

### 7.2 修改文件

```
slora/server/
├── api_server.py                # 添加 --parallel-mode 参数
└── router/
    └── manager.py               # 支持选择不同的并行模式
```

### 7.3 配置文件

```python
# 新增配置参数
--parallel-mode [tensor|data|hybrid]  # 并行模式
--num-workers N                       # Worker 数量（默认=GPU数量）
--gpu-ids 0,1,2,3                    # 指定使用的 GPU
--routing-strategy [round-robin|least-conn|affinity]  # 路由策略
--adapter-cache-size 100             # 每个 Worker 的 adapter 缓存大小
```

---

## 八、总结与建议

### 8.1 核心结论

1. **数据并行更适合 S-LoRA**
   - 高并发场景下吞吐量提升 4-6 倍
   - 充分利用多 GPU 资源
   - 实现相对简单

2. **优化空间巨大**
   - Adapter 亲和性路由可进一步提升 10-20%
   - 跨 GPU 共享可减少 50% 的加载时间
   - 混合模式可支持更大模型

3. **实现可行性高**
   - 代码改动范围可控
   - 向后兼容现有张量并行模式
   - 可以分阶段实施

### 8.2 实施建议

**优先级排序**：
1. ⭐⭐⭐ Phase 1：基础数据并行（必须）
2. ⭐⭐⭐ Phase 2：Adapter 亲和性路由（强烈推荐）
3. ⭐⭐ Phase 3：跨 GPU Adapter 共享（推荐）
4. ⭐ Phase 4：混合并行模式（可选）

**开发时间估算**：
- Phase 1：2-3 周
- Phase 2：1-2 周
- Phase 3：2-3 周
- Phase 4：3-4 周

**风险评估**：
- 低风险：Phase 1、Phase 2
- 中风险：Phase 3（依赖 NVLink）
- 高风险：Phase 4（复杂度高）

### 8.3 下一步行动

1. **验证假设**：在单卡上测试基准性能
2. **原型开发**：实现 Phase 1 的最小可行版本
3. **性能测试**：对比张量并行和数据并行的实际性能
4. **迭代优化**：根据测试结果逐步实施 Phase 2-4

---

## 附录

### A. 参考资料

- [S-LoRA 论文](https://arxiv.org/abs/2311.03285)
- [vLLM 数据并行实现](https://github.com/vllm-project/vllm)
- [Ray Serve 分布式推理](https://docs.ray.io/en/latest/serve/)
- [NVIDIA NVLink 文档](https://www.nvidia.com/en-us/data-center/nvlink/)

### B. 性能测试脚本

```python
# benchmark_parallel.py
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def benchmark(mode, num_requests, concurrency):
    """
    性能测试脚本
    
    Args:
        mode: 'tensor' or 'data'
        num_requests: 总请求数
        concurrency: 并发数
    """
    start_time = time.time()
    
    # 发送请求
    tasks = []
    for i in range(num_requests):
        task = send_request(
            adapter=f"adapter_{i % 100}",
            prompt="测试提示词" * 100
        )
        tasks.append(task)
    
    # 等待完成
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 统计
    throughput = num_requests / duration
    avg_latency = sum(r.latency for r in results) / len(results)
    
    print(f"模式: {mode}")
    print(f"吞吐量: {throughput:.2f} req/s")
    print(f"平均延迟: {avg_latency:.2f} ms")
    print(f"P99 延迟: {np.percentile([r.latency for r in results], 99):.2f} ms")
```

### C. 监控指标

```python
# 关键监控指标
metrics = {
    "throughput": "requests/second",
    "latency_avg": "milliseconds",
    "latency_p99": "milliseconds",
    "gpu_utilization": "percentage",
    "adapter_cache_hit_rate": "percentage",
    "queue_length_per_gpu": "count",
    "adapter_load_time": "milliseconds",
}
```

---

**文档版本**: v1.0  
**创建日期**: 2025-01-12  
**作者**: Kiro AI Assistant  
**状态**: 讨论稿
