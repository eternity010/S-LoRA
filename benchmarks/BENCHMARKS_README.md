{"config": {"num_adapters": 100, "alpha": 1, "req_rate": 10, "cv": 1, "duration": 60, "input_range": [8, 512], "output_range": [8, 64]}, "result": {"total_time": 62.401503562927246, "gpu_peak_mem": 0, "num_abort": 0, "throughput": 9.615152932892792, "strip_throughput": 9.43362773460068, "avg_latency": 2.2998752029736838, "avg_per_token_latency": 0.011686934039283961, "avg_per_output_token_latency": 0.07032705316033748, "avg_first_token_latency": 0.7361556688944498, "avg_satisfaction": 0.9801219151881159, "avg_attainment": 1.0}}


# Benchmarks 目录说明

本目录包含 S-LoRA 的基准测试工具和实验脚本。

## 📁 目录结构

```
benchmarks/
├── launch_server.py      # 启动 S-LoRA 服务器
├── run_exp.py            # 运行基准测试实验（S-LoRA）
├── run_exp_peft.py       # 运行 PEFT 基线对比实验
├── trace.py              # 生成/处理请求跟踪数据
├── exp_suite.py          # 实验配置套件
├── time_stats.py         # 性能时间统计工具
├── real_trace/           # 真实跟踪数据目录
├── paper/                # 论文实验数据
└── a10g/                 # A10G GPU 实验数据
```

## 🔧 核心文件功能

### 1. `launch_server.py` - 服务器启动脚本

**功能**：启动 S-LoRA 或对比基线（vLLM、LightLLM）服务器

**主要参数**：
- `--device`: 设备预设（`debug`/`a10g`/`h100`）
- `--backend`: 后端选择（`slora`/`vllm`/`lightllm`/`vllm-packed`）
- `--model-setting`: 模型配置（`S1`/`S2`/`S3`/`S4`/`Real`）
- `--num-adapter`: LoRA 适配器数量
- `--num-token`: 最大 token 容量
- `--dummy`: 使用虚拟权重模式
- `--prefetch`: 启用适配器预取
- `--no-mem-pool`: 禁用内存池（共享内存）
- `--bmm`: 使用 BMM 模式
- `--enable-abort`: 启用请求中止功能

**使用示例**：
# 使用 GPU 0 和 1
export CUDA_VISIBLE_DEVICES=2

```bash
# 启动 S-LoRA 服务器
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting Real
python run_exp.py --debug --model-setting Real

# 使用虚拟权重测试
python launch_server.py --num-adapter 100 --num-token 10000 --dummy
python run_exp.py --debug
```

### 2. `run_exp.py` - 基准测试主脚本

**功能**：运行 S-LoRA 基准测试，发送请求并收集性能指标

**主要参数**：
- `--backend`: 后端选择（`slora`/`vllm`/`lightllm`）
- `--suite`: 测试套件名称（如 `a10g-num-adapter`）
- `--mode`: 运行模式（`synthetic`/`real`）
- `--debug`: 调试模式
- `--breakdown`: 显示详细性能分解

**使用示例**：
```bash
# 合成模式测试
python run_exp.py --backend slora --suite a10g-num-adapter --mode synthetic

# 真实跟踪数据测试
python run_exp.py --backend slora --suite a10g --mode real --debug
```

### 3. `run_exp_peft.py` - PEFT 基线对比

**功能**：运行 HuggingFace PEFT 基线对比实验

**参数**：与 `run_exp.py` 类似

**使用示例**：
```bash
python run_exp_peft.py --backend peft --suite a10g --mode synthetic
```

### 4. `trace.py` - 请求跟踪生成

**功能**：生成合成请求或处理真实跟踪数据

**核心函数**：
- `generate_requests()`: 生成合成请求（基于幂律分布）
- `get_real_requests()`: 从真实跟踪文件加载请求
- `Request`: 请求数据结构

**请求参数**：
- `num_adapters`: 适配器数量
- `alpha`: 幂律分布参数（控制适配器使用频率）
- `req_rate`: 请求速率（req/s）
- `cv`: 变异系数（控制到达过程）
- `duration`: 测试持续时间（秒）
- `input_range`: 输入长度范围 `[min, max]`
- `output_range`: 输出长度范围 `[min, max]`

### 5. `exp_suite.py` - 实验配置

**功能**：定义模型配置和测试套件

**配置项**：
- `BASE_MODEL`: 基础模型路径映射
- `LORA_DIR`: LoRA 适配器目录映射
- `paper_suite`: 论文实验配置套件

**模型设置**：
- `S1`/`S2`: Llama-7B（不同 rank 配置）
- `S3`/`S4`: Llama-13B（不同 rank 配置）
- `Real`: 本地真实模型路径

**测试套件示例**：
- `a10g-num-adapter`: 不同适配器数量测试
- `a10g-alpha`: 不同幂律分布参数测试
- `a10g-cv`: 不同变异系数测试
- `a10g-req-rate`: 不同请求速率测试

### 6. `time_stats.py` - 性能统计工具

**功能**：从日志文件中提取和统计性能指标

**统计项**：
- `load`: 适配器加载时间
- `prefetch`: 预取时间
- `offload`: 卸载时间
- `prefill`: Prefill 阶段时间
- `decode`: Decode 阶段时间
- `filter`: 过滤时间

## 🚀 快速开始

### 1. 启动服务器

```bash
cd benchmarks

# 真实模型
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting Real

# 虚拟权重（无需模型文件）
python launch_server.py --num-adapter 100 --num-token 10000 --dummy
```

### 2. 运行测试

```bash
# 在另一个终端运行测试
python run_exp.py --debug --model-setting Real
```

### 3. 查看结果

结果保存在 JSONL 格式文件中，包含：
- 吞吐量（throughput）
- 延迟（latency）
- 内存使用（memory usage）
- 其他性能指标

## 📊 实验配置说明

### 合成模式（Synthetic Mode）

使用 `generate_requests()` 生成合成请求：
- 适配器选择：基于幂律分布（Zipf-like）
- 请求到达：泊松过程（可配置变异系数）
- 输入/输出长度：均匀随机分布

### 真实模式（Real Mode）

使用 `get_real_requests()` 从跟踪文件加载：
- 支持 JSON Lines 格式（`.jsonl`）
- 包含真实的时间戳和请求模式
- 需要预先处理跟踪数据

## 🔍 关键概念

### 请求结构

```python
Request(
    req_id: str,           # 请求 ID
    model_dir: str,        # 模型目录
    adapter_dir: str,      # 适配器目录
    prompt: str,           # 输入提示
    prompt_len: int,       # 输入长度
    output_len: int,      # 输出长度
    req_time: float        # 请求时间戳
)
```

### 性能指标

- **吞吐量**：每秒处理的 token 数（tokens/s）
- **延迟**：端到端请求处理时间
- **内存使用**：峰值 GPU 内存占用
- **适配器加载/卸载时间**：内存管理开销

## 📝 注意事项

1. **服务器必须先启动**：运行 `run_exp.py` 前确保服务器已启动
2. **模型路径**：使用 `Real` 模式需要确保模型文件存在
3. **端口冲突**：默认使用 8000 端口，确保未被占用
4. **内存限制**：根据 GPU 显存调整 `--num-token` 和适配器数量
5. **跟踪数据格式**：真实模式需要符合 S-LoRA 的 JSONL 格式

## 🔗 相关文件

- `../slora/server/api_server.py`: S-LoRA 服务器实现
- `../slora/server/router/`: 请求路由和调度
- `../README_CN.md`: 项目主文档

