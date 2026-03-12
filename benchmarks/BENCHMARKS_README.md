# Benchmarks 结构说明

## 目录结构

```
benchmarks/
├── launch_server.py              # 服务器启动脚本（数据并行 / 路由参数）
├── exp_suite.py                  # 模型路径配置 (BASE_MODEL, LORA_DIR)
├── trace.py                      # 请求生成（Power Law + Gamma 到达间隔）
├── run_exp.py                    # 原始单机 benchmark
├── run_routing_comparison.py     # 路由策略对比实验入口
│
├── routing_experiment/           # 实验框架
│   ├── config.py                 # ExperimentConfig 参数定义
│   ├── suite.py                  # 实验套件（笛卡尔积展开）
│   ├── runner.py                 # 执行引擎（服务器管理 / 请求发送 / 指标收集）
│   ├── result.py                 # ExperimentResult + JSONL 持久化
│   ├── analyzer.py               # 结果分析（改进比率 / 统计量）
│   └── charts.py                 # 可视化
│
└── routing_comparison_results/   # 实验输出（per-suite 子目录）
    ├── checkpoint.json           # 全局断点续跑记录
    ├── results.jsonl             # 旧数据（legacy，不再追加）
    ├── extract_w2_search_data.py # 汇总 CSV 提取脚本
    ├── rwpt-w2-search/           # RWPT w2 搜索结果
    │   ├── results.jsonl
    │   ├── logs/
    │   └── charts/
    ├── tc-w2-search/             # Token Count w2 搜索结果
    │   └── ...
    └── ql-w2-search/             # Queue Length w2 搜索结果
        └── ...
```

## 实验框架数据流

```
ExperimentSuite.get_configs(suite_name)
        │  笛卡尔积展开参数组合
        ▼
ExperimentRunner.run_suite(suite_name)
        │  1. 创建 per-suite 输出目录
        │  2. 按需启动/复用服务器（alpha/w2 变化不重启）
        │  3. POST /update_routing_config 动态更新权重
        │  4. trace.generate_requests() → aiohttp 异步发送
        │  5. 收集 /routing_stats delta
        ▼
results.jsonl (per-suite)  →  ResultAnalyzer / ChartGenerator
```

## 当前实验套件

| 套件名 | 变量 | 实验数 | 用途 |
|--------|------|--------|------|
| `rwpt-w2-search` | w2 ∈ [0.5, 5.0] | 10 | RWPT 负载度量 w2 甜点搜索 |
| `tc-w2-search` | w2 ∈ [0.3, 4.0] | 10 | Token Count w2 甜点搜索 |
| `ql-w2-search` | w2 ∈ [0.05, 0.8] | 10 | Queue Length w2 甜点搜索 |
| `routing-alpha-comparison` | alpha × 策略 | 8 | 热点程度对路由效果的影响 |
| `routing-adapter-scaling` | adapters × 策略 | 8 | adapter 规模扩展性 |
| `routing-full-comparison` | alpha × adapters × 策略 | 12 | 完整对比 |

## 环境参数

- 3× RTX 3090 (GPU 1,2,3)，数据并行模式
- `max_lora_ratio = 0.2`（每 Worker 约 18-19 个 adapter）
- `max_total_token_num = 15000`
- 100 个 adapter（alpaca-lora-7b rank=16 + bactrian-x-llama-7b-lora rank=64 交替）
- 淘汰阈值 85%

## 快速使用

```bash
cd benchmarks

# 运行实验
python run_routing_comparison.py --suite ql-w2-search

# 断点续跑
python run_routing_comparison.py --suite ql-w2-search --resume

# 分析指定 suite 的结果
python run_routing_comparison.py --analyze-only --suite rwpt-w2-search

# 生成图表
python run_routing_comparison.py --generate-charts --suite tc-w2-search

# 列出所有套件
python run_routing_comparison.py --list-suites

# 提取 w2 搜索汇总 CSV
python routing_comparison_results/extract_w2_search_data.py
```
