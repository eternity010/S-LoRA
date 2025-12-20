# Router 目录说明

本目录包含 S-LoRA 的请求调度和管理核心代码。

## 📁 文件结构

```
router/
├── manager.py              # 主调度管理器（核心入口）
├── req_queue.py            # 基础请求队列（默认调度器）
├── vtc_req_queue.py        # VTC 公平调度器
├── pets_req_queue.py       # PETS 预测调度器
├── peft_req_queue.py       # PEFT 基线调度器
├── cluster_req_queue.py    # 集群调度器
├── abort_req_queue.py      # 支持请求中止的调度器
├── stats.py                # 性能统计工具
├── profiler.py             # 性能分析器（用于 PETS）
└── model_infer/            # 模型推理相关
    └── model_rpc.py        # 模型 RPC 客户端
```

## 🔧 核心文件详解

### 1. `manager.py` - 主调度管理器

**功能**：S-LoRA 请求调度的核心控制器

**核心类**：
- `RouterManager`: 管理整个请求调度流程

**主要职责**：
- 接收来自 HTTP 服务器的请求
- 管理请求队列和批次生成
- 协调模型推理进程
- 处理 Prefill 和 Decode 阶段
- 管理 LoRA 适配器的加载/卸载

**关键方法**：
- `add_req()`: 添加新请求到队列
- `loop_for_fwd()`: 主事件循环
- `_step()`: 执行单步调度逻辑
- `abort()`: 中止指定请求

**调度器选择**：
```python
get_scheduler()  # 根据配置选择不同的调度器
```

---

### 2. `req_queue.py` - 基础请求队列（默认调度器）

**功能**：S-LoRA 的默认 FIFO 调度器

**核心类**：
- `ReqQueue`: 基础请求队列类

**调度策略**：
- **FIFO（先进先出）**：按请求到达顺序处理
- **内存感知**：检查 token 和适配器内存限制
- **批处理优化**：尽可能将请求打包成批次

**关键方法**：
- `generate_new_batch()`: 生成新的批处理批次
- `_can_add_new_req()`: 判断是否可以添加新请求
- `next_batch()`: 获取下一个预取批次

**使用场景**：
- 默认调度器（`--scheduler slora`）
- 其他调度器的基类

---

### 3. `vtc_req_queue.py` - VTC 公平调度器

**功能**：实现 VTC（Virtual Token Counter）公平调度算法

**核心类**：
- `VTCReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **公平性保证**：确保不同适配器/用户获得公平的服务
- **虚拟令牌计数**：使用虚拟令牌计数器跟踪服务量
- **权重支持**：支持为不同适配器设置权重

**关键特性**：
- 维护每个适配器的服务计数器
- 优先服务服务量较少的适配器
- 支持公平权重配置

**使用场景**：
- 需要公平性的多租户场景
- 不同适配器有不同优先级需求

**启用方式**：
```bash
--scheduler vtc_fair --fair-weights <权重列表>
```

---

### 4. `pets_req_queue.py` - PETS 预测调度器

**功能**：实现 PETS（Predictive Early Termination Scheduling）调度算法

**核心类**：
- `PETSReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **预测性调度**：基于性能模型预测请求延迟
- **任务内批处理**：将相同适配器的请求聚类批处理
- **动态规划优化**：使用 DP 算法优化批次组合

**关键特性**：
- 使用 `AlphaModel` 和 `BetaModel` 预测延迟
- 任务内批处理（intra-task batching）
- 任务间批处理（inter-task batching）

**使用场景**：
- 需要优化延迟的场景
- 有性能分析数据的场景

**启用方式**：
```bash
--scheduler pets --profile
```

---

### 5. `peft_req_queue.py` - PEFT 基线调度器

**功能**：用于 HuggingFace PEFT 基线对比

**核心类**：
- `PEFTReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **单适配器批处理**：每个批次只包含同一适配器的请求
- **适配器切换**：批次之间切换适配器权重

**使用场景**：
- 与 HuggingFace PEFT 进行性能对比
- 测试单适配器批处理性能

**启用方式**：
```bash
--scheduler peft
```

---

### 6. `cluster_req_queue.py` - 集群调度器

**功能**：限制批次中的适配器数量

**核心类**：
- `ClusterReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **适配器数量限制**：限制单个批次中的适配器数量
- **优先级策略**：优先使用当前批次已有的适配器

**关键特性**：
- 当批次适配器数量达到上限时，优先添加使用相同适配器的请求
- 减少适配器切换开销

**使用场景**：
- 需要控制批次复杂度的场景
- 减少适配器加载/卸载频率

**启用方式**：
```bash
--batch-num-adapters <数量>
```

---

### 7. `abort_req_queue.py` - 支持中止的调度器

**功能**：支持请求中止功能的调度器

**核心类**：
- `AbortReqQueue`: 继承自 `ReqQueue`

**调度策略**：
- **请求中止**：自动中止超时请求
- **动态批处理大小**：根据请求速率调整批处理大小
- **LIFO/FIFO 切换**：根据负载情况切换调度顺序

**关键特性**：
- 跟踪请求时间戳
- 使用 attainment function 判断请求是否超时
- 自适应批处理大小

**使用场景**：
- 需要支持请求取消的场景
- 高负载下的动态调度

**启用方式**：
```bash
--enable-abort
```

---

### 8. `stats.py` - 性能统计工具

**功能**：收集和输出性能统计信息

**核心类**：
- `Stats`: 性能统计类

**统计指标**：
- 总 token 吞吐量（prompt + generate）
- Prompt token 吞吐量
- Generate token 吞吐量

**使用方式**：
- 自动集成到 `RouterManager` 中
- 定期输出统计信息到控制台

---

### 9. `profiler.py` - 性能分析器

**功能**：为 PETS 调度器提供性能预测模型

**核心类**：
- `AlphaModel`: 基础模型延迟预测
- `BetaModel`: LoRA 适配器延迟预测

**功能**：
- 加载性能分析结果（`.pkl` 文件）
- 根据批次大小和序列长度预测延迟
- 支持插值和外推

**使用场景**：
- PETS 调度器需要性能预测时
- 性能分析和优化

---

## 🔄 调度流程

```
1. HTTP 服务器接收请求
   ↓
2. RouterManager.add_req() 添加请求
   ↓
3. RouterManager.loop_for_fwd() 主循环
   ↓
4. RouterManager._step() 执行调度
   ↓
5. ReqQueue.generate_new_batch() 生成批次
   ↓
6. 检查内存/适配器限制
   ↓
7. 加载适配器 → Prefill → Decode
   ↓
8. 返回结果并更新统计
```

## 📊 调度器对比

| 调度器 | 策略 | 特点 | 适用场景 |
|--------|------|------|----------|
| **ReqQueue** | FIFO | 简单高效，默认选择 | 通用场景 |
| **VTCReqQueue** | 公平调度 | 保证公平性，支持权重 | 多租户场景 |
| **PETSReqQueue** | 预测调度 | 优化延迟，需要性能分析 | 延迟敏感场景 |
| **PEFTReqQueue** | 单适配器批处理 | 与 PEFT 对比 | 基线测试 |
| **ClusterReqQueue** | 适配器限制 | 控制批次复杂度 | 减少切换开销 |
| **AbortReqQueue** | 支持中止 | 动态调整，自动超时 | 需要请求取消 |

## 🎯 如何选择调度器

### 默认场景
```bash
# 使用默认 FIFO 调度器
python launch_server.py --num-adapter 100 --num-token 10000
```

### 公平性需求
```bash
# 使用 VTC 公平调度器
python launch_server.py --scheduler vtc_fair --fair-weights 1 2 3
```

### 延迟优化
```bash
# 使用 PETS 预测调度器（需要先进行性能分析）
python launch_server.py --scheduler pets --profile
```

### 控制适配器数量
```bash
# 限制批次中最多 10 个适配器
python launch_server.py --batch-num-adapters 10
```

### 支持请求中止
```bash
# 启用请求中止功能
python launch_server.py --enable-abort
```

## 📝 关键数据结构

位置：`slora/server/io_struct.py`

- `Req`: 单个请求对象
- `Batch`: 请求批次对象
- `BatchAbortReq`: 中止请求批次

## 🔗 相关文件

- `../api_server.py`: API 服务器入口
- `../io_struct.py`: 数据结构定义
- `../sampling_params.py`: 采样参数
- `model_infer/model_rpc.py`: 模型推理 RPC 客户端

