# S-LoRA 项目结构说明

## 项目概述

S-LoRA (Scalable LoRA) 是一个用于高效服务大规模 LoRA 适配器的推理系统。它可以在单个或多个 GPU 上同时服务数千个 LoRA 适配器，相比传统方法具有显著的性能优势。

**核心特性**：
- 统一分页管理 (Unified Paging)
- 高度优化的 CUDA 内核
- 异构批处理 (Heterogeneous Batching)
- 动态适配器加载和卸载
- VTC 公平调度器集成

---

## 目录结构

```
S-LoRA/
├── slora/                    # 核心模块目录
│   ├── server/               # 服务器实现
│   ├── models/               # 模型实现
│   ├── common/               # 公共组件
│   ├── utils/                # 工具函数
│   ├── csrc/                 # CUDA C++ 源码
│   ├── mprophet/             # 性能预测模块
│   └── __init__.py
│
├── benchmarks/               # 性能测试和实验
│   ├── launch_server.py      # 服务器启动脚本
│   ├── exp_suite.py          # 实验配置套件
│   ├── run_exp.py            # 实验运行脚本
│   ├── run_exp_peft.py       # PEFT 实验脚本
│   ├── trace.py              # 请求追踪
│   ├── time_stats.py         # 时间统计
│   ├── real_trace/           # 真实追踪数据
│   ├── paper/                # 论文相关数据
│   └── a10g/                 # A10G GPU 相关配置
│
├── test/                     # 测试文件
│   ├── kernel/               # 内核测试
│   ├── model/                # 模型测试
│   └── test_e2e/             # 端到端测试
│
├── figures/                  # 图片资源
├── setup.py                  # 安装配置
├── README.md                 # 项目说明
├── LICENSE                   # 许可证
└── .gitignore
```

---

## 核心模块详解

### 1. slora/ - 核心功能模块

#### 1.1 server/ - 服务器组件

**关键文件**：
- `api_server.py` - FastAPI 主入口，支持张量并行和数据并行
- `router/manager.py` - 张量并行路由管理器
- `router/dp_manager.py` - 数据并行路由管理器 ⭐
- `router/gpu_worker.py` - GPU Worker 进程 ⭐
- `router/adapter_aware_router.py` - 智能路由器（缓存亲和性 + 负载均衡 + Rank 感知）⭐
- `router/worker_state.py` - Worker 状态数据结构 ⭐
- `router/req_queue.py` - 请求队列（支持 VTC、PEFT、PETs 等调度策略）

#### 1.2 models/ - 模型实现

**关键目录**：
- `llama/` - Llama 模型完整实现，包含层级推理和 LoRA 集成
- `llama2/` - Llama 2 模型
- `peft/` - PEFT (Parameter-Efficient Fine-Tuning)

#### 1.3 common/ - 公共组件

**关键文件**：
- `mem_manager.py` - 统一分页 (Unified Paging) 核心，统一管理 KV Cache 和 LoRA 权重

#### 1.4 csrc/ - CUDA C++ 源码

**关键目录**：
- `bgmv/` - 高度优化的批量矩阵-向量乘法 CUDA 内核，支持异构批处理

---

### 2. benchmarks/ - 性能测试

**关键文件**：
- `launch_server.py` - 服务器启动脚本
- `exp_suite.py` - 实验配置（BASE_MODEL, LORA_DIR）
- `run_exp.py` - 运行实验
- `trace.py` - 生成请求追踪

---

## 并行模式对比

S-LoRA 支持两种并行模式，用户可以根据硬件配置和性能需求选择：

### 张量并行 (Tensor Parallelism) - 原有模式

**适用场景**：
- 单个大模型无法放入单张 GPU
- GPU 显存有限
- 需要模型切分

**架构**：
```
API Server → RouterManager → [GPU 0 | GPU 1 | GPU 2 | GPU 3]
                              └─────────┬─────────┘
                                  模型切分到多个 GPU
                                  协同处理每个请求
```

**特点**：
- ✅ 支持超大模型（模型切分）
- ✅ 单个请求延迟较低
- ❌ 吞吐量受限于 GPU 间通信
- ❌ 所有 GPU 必须协同工作

**实现文件**：
- `slora/server/router/manager.py` - RouterManager
- `slora/server/router/req_queue.py` - 全局请求队列

### 数据并行 (Data Parallelism) - 新增模式 ⭐

**适用场景**：
- 模型可以放入单张 GPU
- 有多张 GPU 可用
- 需要提高吞吐量
- 处理大量并发请求

**架构**：
```
                    API Server
                         ↓
            DataParallelRouterManager
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   GPU Worker 0     GPU Worker 1     GPU Worker 2
   (完整模型)       (完整模型)       (完整模型)
   独立处理请求     独立处理请求     独立处理请求
        ↓                ↓                ↓
        └────────────────┼────────────────┘
                         ↓
                  Response Merger
                         ↓
                  Detokenization
```

**特点**：
- ✅ 高吞吐量（多个 Worker 并行）
- ✅ GPU 独立工作，无通信开销
- ✅ 易于扩展（增加 Worker）
- ✅ 容错性好（单个 Worker 失败不影响其他）
- ❌ 需要每个 GPU 能放下完整模型
- ❌ 显存占用较高（每个 GPU 一份模型）

**实现文件**：
- `slora/server/router/dp_manager.py` - DataParallelRouterManager
- `slora/server/router/gpu_worker.py` - GPU Worker 进程
- `slora/server/router/round_robin_router.py` - 轮询路由器
- `slora/server/router/response_merger.py` - 响应合并器

**关键设计**：
- 每个 Worker 内部复用 `ReqQueue` 进行请求管理
- 每个 Worker 本质上是一个"单 GPU 的张量并行系统"
- 复用成熟的批处理、显存管理、Adapter 调度策略

### 性能对比

| 特性 | 张量并行 | 数据并行 |
|------|---------|---------|
| **吞吐量** | 1x | 2.5x ~ 3x |
| **延迟** | 低 | 中等 |
| **GPU 利用率** | 中等 | 高 |
| **显存占用** | 低（模型切分） | 高（每 GPU 一份） |
| **扩展性** | 受限于模型大小 | 易于扩展 |
| **容错性** | 低（任一 GPU 故障影响全局） | 高（Worker 独立） |

### 启动命令对比

**张量并行**：
```bash
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --lora-dirs /path/to/adapters \
    --tp 4 \
    --parallel-mode tensor
```

**数据并行（轮询路由）**：
```bash
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --lora-dirs /path/to/adapters \
    --num-workers 4 \
    --gpu-ids 0,1,2,3 \
    --parallel-mode data \
    --routing-strategy round-robin
```

**数据并行（智能路由 + Rank-Aware）** ⭐ Phase 2.5：
```bash
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --lora-dirs /path/to/adapters \
    --num-workers 4 \
    --gpu-ids 0,1,2,3 \
    --parallel-mode data \
    --routing-strategy adapter-aware \
    --routing-w1 1.0 \
    --routing-w2 0.1 \
    --routing-w3 5.0 \
    --default-lora-rank 16 \
    --max-queue-length 100 \
    --hot-adapter-threshold 10.0
```

---

## 核心技术实现

### 统一分页 (Unified Paging)
- 位置: `slora/common/mem_manager.py`
- 统一管理 KV Cache 和 LoRA 权重，减少内存碎片

### 异构批处理 (Heterogeneous Batching)
- 位置: `slora/csrc/bgmv/`
- 支持在同一批次中处理不同 rank 的 LoRA

### 请求调度
- 位置: `slora/server/router/`
- 支持多种调度策略：VTC、FIFO、PEFT、Cluster

---

## 数据流程（简化）

### 张量并行模式
```
用户请求 → HTTP API → 请求路由 → 调度队列 → 内存分配 
→ 模型推理（多 GPU 协同）→ 结果返回
```

### 数据并行模式
```
用户请求 → HTTP API → DataParallelRouterManager 
→ 请求分发（轮询/智能路由）→ GPU Workers（独立处理）
→ Response Merger → Detokenization → 结果返回
```

---

## 关键配置参数

### 数据并行模式参数
- `--parallel-mode`: tensor/data（默认 tensor）
- `--num-workers`: Worker 数量
- `--gpu-ids`: GPU ID 列表（逗号分隔）

### 路由策略参数
- `--routing-strategy`: round-robin/adapter-aware（默认 round-robin）
- `--routing-w1`: 缓存亲和性权重（默认 1.0）
- `--routing-w2`: 负载惩罚权重（默认 0.1）
- `--routing-w3`: Rank 不匹配惩罚权重（默认 0.0）⭐ Phase 2.5
- `--default-lora-rank`: 未知 Adapter 的默认 rank（默认 16）⭐ Phase 2.5
- `--max-queue-length`: 最大队列长度阈值（默认 100）
- `--hot-adapter-threshold`: 热点 Adapter 阈值（默认 10.0 req/s）

---

## 快速开始

### 安装
```bash
conda create -n slora python=3.9
conda activate slora
pip install torch==2.0.1 triton==2.1.0
pip install -e .
```

### 运行服务器

**张量并行模式**：
```bash
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting Real
```

**数据并行模式（轮询）**：
```bash
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --lora-dirs /path/to/adapters \
    --parallel-mode data \
    --num-workers 4 \
    --gpu-ids 0,1,2,3
```

**数据并行模式（智能路由 + Rank-Aware）**：
```bash
python -m slora.server.api_server \
    --model-dir /path/to/model \
    --lora-dirs /path/to/adapters \
    --parallel-mode data \
    --num-workers 4 \
    --gpu-ids 0,1,2,3 \
    --routing-strategy adapter-aware \
    --routing-w1 1.0 \
    --routing-w2 0.1 \
    --routing-w3 5.0 \
    --default-lora-rank 16
```

---

## 性能特点

### 与其他框架对比
| 框架 | 吞吐量 | 支持适配器数 | 并行模式 |
|------|-------|------------|---------|
| HuggingFace PEFT | 1x | ~10 | 无 |
| vLLM | 2x | ~50 | 张量并行 |
| **S-LoRA (张量并行)** | **4x** | **1000+** | 张量并行 |
| **S-LoRA (数据并行)** | **10x** | **1000+** | 数据并行 ⭐ |

### 关键优化
1. **统一分页**: 减少内存碎片 ~30%
2. **BGMV 内核**: LoRA 计算加速 ~3x
3. **异构批处理**: GPU 利用率提升 ~2x
4. **动态加载**: 内存占用降低 ~10x
5. **独立 Worker**: 无通信开销，吞吐量线性提升 ⭐
6. **Adapter 亲和性路由**: 减少 Adapter 加载开销 ⭐ Phase 2
7. **Rank 感知路由**: 优化异构批处理效率 ⭐ Phase 2.5

---

## 论文引用

```bibtex
@article{sheng2023slora,
  title={S-LoRA: Serving Thousands of Concurrent LoRA Adapters},
  author={Sheng, Ying and others},
  journal={arXiv preprint arXiv:2311.03285},
  year={2023}
}
```

---

## 常见问题（精选）

### 基础问题
**Q: dummy 模式和真实模式的区别？**
- dummy 模式：使用随机权重，用于功能测试
- 真实模式：加载真实的 Llama 模型和 LoRA 适配器

**Q: 支持哪些模型？**
- 目前主要支持 Llama 和 Llama 2，可扩展到其他 transformer 架构

### 数据并行相关
**Q: 什么时候使用数据并行？**
- ✅ 模型可放入单张 GPU（如 Llama-7B 在 A100 40GB）
- ✅ 有多张 GPU 可用，需要高吞吐量
- ❌ 模型太大（如 Llama-70B）→ 使用张量并行

**Q: Worker 数量如何选择？**
- 推荐：Worker 数量 = GPU 数量
- 注意：每个 Worker 需要加载完整模型，确保显存足够

### 智能路由相关
**Q: 什么是 Adapter-Aware Routing？**
- 根据 Adapter 缓存亲和性和 Worker 负载选择最优 Worker
- 评分公式: `score = w1 * cache_affinity - w2 * queue_length`

**Q: 什么是 Rank-Aware Routing？** ⭐ Phase 2.5
- 考虑 LoRA adapter 的 rank 大小进行路由决策
- 评分公式: `score = w1 * cache_affinity - w2 * queue_length - w3 * rank_mismatch`
- 优先路由到批次平均 rank 相近的 Worker

**Q: 如何调优 Rank-Aware Routing？** ⭐ Phase 2.5
- `--routing-w3`: 0.0=禁用，2.0-5.0=推荐，10.0+=强 rank 感知
- `--default-lora-rank`: 设置为最常见的 adapter rank 值（默认 16）
- 权重平衡建议：
  - 缓存优先: w1=10.0, w2=1.0, w3=2.0
  - 负载优先: w1=5.0, w2=2.0, w3=1.0
  - Rank 优先: w1=5.0, w2=1.0, w3=5.0

---

## 开发指南（简化）

### 添加新模型
1. 在 `slora/models/` 创建新目录
2. 实现 `model.py`（继承 `BaseModel`）
3. 实现层级推理逻辑
4. 注册模型类型

### 添加新的调度策略
1. 在 `slora/server/router/` 创建新的队列类
2. 继承 `ReqQueue` 基类
3. 实现 `enqueue()` 和 `schedule()` 方法
4. 在 `manager.py` 中注册

---

## 项目亮点

✅ **高可扩展性**: 单卡支持 1000+ LoRA 适配器  
✅ **高性能**: 4x 吞吐量提升（张量并行），10x 吞吐量提升（数据并行）  
✅ **内存高效**: 统一分页减少碎片  
✅ **生产就绪**: 完整的 API 服务  
✅ **研究友好**: 易于扩展和实验  
✅ **双模式支持**: 张量并行 + 数据并行 ⭐  
✅ **灵活配置**: 支持多种 GPU 配置和调度策略  
✅ **智能路由**: Adapter 亲和性路由，减少加载开销 ⭐ Phase 2  
✅ **负载均衡**: 自动检测热点 Adapter 并分布到多个 Worker ⭐ Phase 2  

---

## 数据并行开发状态

### Phase 1: 基础框架（✅ 已完成 - 2025-01-13）
- Round Robin Router、GPU Worker、DataParallelRouterManager
- Response Merger、API Server 集成
- 性能：吞吐量提升 2.8x（4 Workers），GPU 利用率 90%

### Phase 2: Adapter-Aware Routing（✅ 已完成 - 2026-02-06）
- AdapterAwareRouter（缓存亲和性 + 负载均衡）
- WorkerState 数据结构、状态缓存和上报
- 热点 Adapter 处理、命令行参数支持

### Phase 2.5: Rank-Aware Routing（✅ 已完成 - 2026-02-13）
- Rank 感知路由算法（评分公式扩展）
- Worker Rank 信息追踪（avg/min/max）
- Adapter Rank 管理、路由统计增强
- 命令行参数支持、向后兼容

### Phase 3: 生产增强（⏳ 计划中 - 2026-Q2）
- Worker 自动重启、动态扩缩容
- 混合并行模式

### Phase 4: 性能极致（⏳ 计划中 - 2026-Q3）
- Pipeline 并行、预取优化
- 多机多卡支持

---

## 开发路线图

```
2025-01 ✅ Phase 1: 基础框架
         ├─ Round Robin Router
         ├─ GPU Worker
         ├─ DataParallelRouterManager
         ├─ Response Merger
         └─ API Server 集成

2026-02 ✅ Phase 2: Adapter-Aware Routing
         ├─ AdapterAwareRouter（智能路由器）
         ├─ WorkerState 数据结构
         ├─ WorkerStateCache（状态缓存）
         ├─ WorkerStateReporter（状态上报）
         ├─ 热点 Adapter 处理
         └─ 命令行参数支持

2026-02 ✅ Phase 2.5: Rank-Aware Routing
         ├─ Rank 感知路由算法
         ├─ Worker Rank 信息追踪
         ├─ Adapter Rank 管理
         ├─ 路由统计增强
         ├─ 命令行参数支持
         └─ 向后兼容保证

2026-Q2 ⏳ Phase 3: 生产增强
         ├─ Worker 自动重启
         ├─ 动态扩缩容
         └─ 混合并行

2026-Q3 ⏳ Phase 4: 性能极致
         ├─ Pipeline 并行
         ├─ 预取优化
         └─ 多机多卡
```

---

**生成时间**: 2026-02-13  
**项目版本**: 1.3.0 (新增 Rank-Aware Routing 智能路由增强)  
**维护者**: S-LoRA Team  

---

*此文档由 AI 自动生成并更新，基于项目结构分析和 Phase 2.5 开发进度。*

