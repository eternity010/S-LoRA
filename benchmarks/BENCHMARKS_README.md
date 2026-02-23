# Benchmarks 结构说明

## 目录结构

```
benchmarks/
├── launch_server.py          # 服务器启动脚本（原始 + 扩展了数据并行/路由参数）
├── exp_suite.py              # 模型路径配置 & 实验套件定义（BASE_MODEL, LORA_DIR）
├── trace.py                  # 请求生成（Power Law 分布 + Gamma 到达间隔）
├── run_exp.py                # 原始单机 benchmark 运行脚本
├── run_exp_peft.py           # PEFT 基线对比脚本
├── run_routing_comparison.py # 路由策略对比实验入口（新增）
├── time_stats.py             # 性能统计工具
│
├── routing_experiment/       # 路由对比实验框架（新增）
│   ├── config.py             # 单次实验参数定义（ExperimentConfig）
│   ├── suite.py              # 实验套件，笛卡尔积展开参数组合（ExperimentSuite）
│   ├── runner.py             # 实验执行引擎（启动服务器、发请求、收指标）
│   ├── result.py             # 结果数据结构与 JSONL 持久化（ExperimentResult）
│   ├── analyzer.py           # 结果分析（改进比率、统计量、负载均衡 CV）
│   └── charts.py             # 可视化（吞吐量/延迟/缓存命中率/w2 调优图）
│
└── routing_comparison_results/  # 实验输出目录
    ├── results.jsonl            # 所有实验结果（追加写入）
    ├── checkpoint.json          # 断点续跑记录
    ├── charts/                  # 生成的图表（PNG + PDF）
    └── logs/                    # 各次实验的服务器日志
```

## 文件复用关系

`routing_experiment/` 框架复用了以下原始文件，未做修改：

- **`exp_suite.py`** — 提供 `BASE_MODEL`、`LORA_DIR` 路径配置
- **`trace.py`** — 提供 `generate_requests()`，按 alpha/cv/req_rate 生成合成请求

**`launch_server.py`** 在原始基础上扩展，新增了 `--parallel-mode`、`--routing-strategy`、`--routing-w1/w2/w3`、`--evict-*`、`--max-lora-ratio` 等参数，由 `runner.py` 通过 `subprocess` 调用。

## 实验框架数据流

```
ExperimentSuite.get_configs(suite_name)
        │  笛卡尔积展开所有参数组合
        ▼
ExperimentRunner.run_suite()
        │  对每个 ExperimentConfig：
        │  1. 判断是否需要重启服务器（alpha/w2 变化不需要重启）
        │  2. 启动 launch_server.py（subprocess）
        │  3. 动态更新路由权重（POST /update_routing_config）
        │  4. 调用 trace.generate_requests() 生成请求
        │  5. aiohttp 异步发送，按时间戳控制速率
        │  6. 收集 /routing_stats 前后 delta
        ▼
ExperimentResult  →  results.jsonl
        │
        ▼
ResultAnalyzer    →  改进比率、统计量、Markdown 汇总表
ChartGenerator    →  charts/*.png / *.pdf
```

## 预定义实验套件

| 套件名 | 变量 | 用途 |
|--------|------|------|
| `routing-alpha-comparison` | alpha × 策略 | 热点程度对路由效果的影响 |
| `routing-adapter-scaling` | num_adapters × 策略 | adapter 规模扩展性 |
| `routing-full-comparison` | alpha × adapters × 策略 | 完整对比 |
| `routing-weight-comparison` | w2 ∈ [0.08, 0.2] | 负载惩罚权重调优 |

## 快速使用

```bash
cd benchmarks

# 运行路由对比实验
python run_routing_comparison.py --suite routing-alpha-comparison

# 断点续跑
python run_routing_comparison.py --suite routing-full-comparison --resume

# 仅分析已有结果
python run_routing_comparison.py --analyze-only --output-dir routing_comparison_results

# 仅生成图表
python run_routing_comparison.py --generate-charts --output-dir routing_comparison_results

# 列出所有套件
python run_routing_comparison.py --list-suites
```
