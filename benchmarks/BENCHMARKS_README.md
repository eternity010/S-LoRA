# Benchmarks

## 目录结构

```
benchmarks/
├── launch_server.py      # 启动服务器
├── run_exp.py            # 运行基准测试
├── run_exp_peft.py       # PEFT 基线对比
├── trace.py              # 请求生成（Power Law 分布）
├── exp_suite.py          # 实验配置套件
└── time_stats.py         # 性能统计
```

## 快速开始

```bash
cd benchmarks

# 1. 启动服务器
# 张量并行（默认）
python launch_server.py --num-adapter 100 --num-token 10000 --dummy

# 数据并行（3 GPU）
python launch_server.py --dummy --num-adapter 100 --num-token 10000 \
    --parallel-mode data --num-workers 3 --gpu-ids 1,2,3

# 数据并行 + 智能路由
python launch_server.py --dummy --num-adapter 100 --num-token 10000 \
    --parallel-mode data --num-workers 3 --gpu-ids 1,2,3 \
    --routing-strategy adapter-aware

# 数据并行 + Rank-Aware 路由
python launch_server.py --dummy --num-adapter 100 --num-token 10000 \
    --parallel-mode data --num-workers 3 --gpu-ids 1,2,3 \
    --routing-w3 5.0

# 2. 运行测试（另一个终端）
python run_exp.py --debug                          # 默认配置（均匀分布）
python run_exp.py --suite routing-test --debug     # 智能路由测试（热点分布）
```

> 使用真实模型时，将 `--dummy` 替换为 `--model-setting Real`，两者不要同时使用。

## launch_server.py 参数

| 参数 | 说明 |
|------|------|
| `--model-setting` | 模型配置：`S1`/`S2`(7B) `S3`/`S4`(13B) `Real`(本地) |
| `--dummy` | 虚拟权重，快速测试用 |
| `--num-adapter` | LoRA 适配器数量 |
| `--num-token` | 最大 token 容量 |
| `--parallel-mode` | `tensor`(默认) / `data` |
| `--num-workers` | 数据并行 Worker 数量 |
| `--gpu-ids` | GPU ID，逗号分隔 |
| `--routing-strategy` | 路由策略：`round-robin`(默认) / `adapter-aware` |
| `--routing-w3` | Rank-Aware 权重（0=禁用，5.0=中度，10.0=强） |

## 测试套件（exp_suite.py）

### debug_suite（开发测试用）

| 套件名 | alpha | num_adapters | 说明 |
|--------|-------|-------------|------|
| `default` | 1.0 | 100 | 均匀分布，baseline |
| `routing-test` | 0.6 | 100 | 热点分布，测试智能路由 |
| `debug` | 1.0 | 20 | 小规模快速测试 |

### paper_suite（论文实验用）

`a10g-num-adapter`, `a10g-alpha`, `a10g-cv`, `a10g-req-rate` 等，详见代码。

## Alpha 参数（trace.py）

`alpha` 控制 Power Law 分布，决定 adapter 访问的热点程度：

```python
probs = np.random.power(alpha, tot_req)
ind = (probs * num_adapters).astype(int)
```

| alpha | 分布 | 说明 |
|-------|------|------|
| 0.1-0.3 | 极端热点 | 少数 adapter 占 80%+ 流量 |
| **0.6** | **明显热点** | **前 20% adapter 占 60%+ 流量（推荐测试智能路由）** |
| 1.0 | 均匀分布 | 所有 adapter 等概率访问 |

alpha < 1 时热点越集中，智能路由的缓存亲和性优势越明显。alpha = 1 是均匀分布，难以体现路由优化效果。

## 并行模式对比

| 特性 | 张量并行 | 数据并行 |
|------|---------|---------|
| 场景 | 模型放不下单卡 | 模型放得下，要高吞吐 |
| 吞吐量 | 1x | 2.5x ~ 3x |
| 显存占用 | 低（模型切分） | 高（每 GPU 一份） |

## 注意事项

- 服务器必须先启动，再运行 `run_exp.py`
- 数据并行模式下每个 GPU 需放下完整模型（7B 需 24GB+，13B 需 40GB+）
- Worker 数量推荐等于 GPU 数量
- 监控 GPU：`watch -n 1 nvidia-smi`
