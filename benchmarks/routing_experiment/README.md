# Routing Strategy Comparison Experiment

自动化路由策略对比实验框架，用于系统性地比较 Round-Robin 和 Adapter-Aware 路由策略的性能。

## 快速开始

```bash
cd benchmarks

# 列出可用套件
python run_routing_comparison.py --list-suites

# 运行实验（自动启动服务器、运行测试、分析结果、生成图表）
python run_routing_comparison.py --suite dp-roundrobin-baseline

# 从中断处恢复
python run_routing_comparison.py --suite dp-rwpt-rate-scaling --resume

# 仅分析已有结果
python run_routing_comparison.py --analyze-only --output-dir routing_comparison_results

# 仅生成图表
python run_routing_comparison.py --generate-charts --output-dir routing_comparison_results
```

## 预定义实验套件

| 套件名 | 说明 | 配置数 |
|--------|------|--------|
| `dp-roundrobin-baseline` | Round-Robin 基线（alpha=0.1/0.3/0.8） | 3 |
| `dp-rwpt-baseline` | RWPT 基线（alpha=0.1/0.3/0.8） | 3 |
| `dp-roundrobin-rate-scaling` | Round-Robin 速率扩展（2/4/6/8 req/s） | 4 |
| `dp-rwpt-rate-scaling` | RWPT 速率扩展（2/4/6/8 req/s） | 4 |

## 动态路由配置更新

实验框架支持动态更新路由配置参数（w1/w2/w3），无需重启服务器。这大幅减少了实验时间（每次重启需要约 20 分钟）。

### API 端点

**获取当前配置和统计**
```bash
curl http://localhost:8000/routing_stats | jq '.w1, .w2, .w3'
```

**动态更新路由配置**
```bash
curl -X POST http://localhost:8000/update_routing_config \
  -H "Content-Type: application/json" \
  -d '{"w1": 2.0, "w2": 0.5, "reset_stats": true}'
```

参数说明：
- `w1`: 缓存亲和性权重（可选，默认 1.0）
- `w2`: 负载惩罚权重（可选，默认 0.1）
- `w3`: Rank 不匹配惩罚权重（可选，默认 0.0）
- `reset_stats`: 是否重置统计信息（可选，默认 false）

### routing-weight-comparison 套件

专门用于测试不同 w1/w2 组合的实验套件：

```bash
python run_routing_comparison.py --suite routing-weight-comparison
```

该套件会：
1. 启动一次服务器（约 20 分钟）
2. 动态更新 w1/w2 配置运行 18 个实验
3. 无需重启服务器，大幅节省时间

测试的参数组合：
- w1 (缓存亲和性): 0.5, 1.0, 2.0
- w2 (负载惩罚): 0.0, 0.1, 0.5
- alpha: 0.3, 0.6

## 输出文件

```
routing_comparison_results/
├── results.jsonl              # 实验结果（JSONL 格式）
├── checkpoint.json            # 检查点（用于恢复）
├── summary.md                 # 汇总表格
└── charts/                    # 图表（PNG + PDF）
    ├── throughput_comparison.*
    ├── latency_comparison.*
    ├── cache_hit_rate.*
    ├── scaling_trend.*
    └── worker_load_distribution.*
```

## 主要参数

```bash
python run_routing_comparison.py [OPTIONS]
```

- `--suite <name>` - 实验套件（默认：dp-roundrobin-baseline）
- `--output-dir <path>` - 输出目录（默认：routing_comparison_results）
- `--model-setting <Real|Dummy>` - 模型设置（默认：Real）
- `--gpu-ids <ids>` - GPU ID 列表（默认：0,1,2）
- `--num-workers <n>` - Worker 数量（默认：3）
- `--resume` - 从检查点恢复

## 编程接口

```python
from routing_experiment import ExperimentRunner, ResultAnalyzer, ChartGenerator

# 运行实验
runner = ExperimentRunner(output_dir="results")
runner.run_suite("dp-roundrobin-baseline")

# 动态更新路由配置（无需重启服务器）
runner._update_routing_config(w1=2.0, w2=0.5, reset_stats=True)

# 分析结果
from routing_experiment import ExperimentRecord
records = ExperimentRecord.load_from_jsonl("results/results.jsonl")
analyzer = ResultAnalyzer([r.result for r in records])
print(analyzer.generate_summary_table())

# 生成图表
generator = ChartGenerator(output_dir="charts")
generator.plot_all_comparisons([r.result for r in records])
```

## 模块说明

- `config.py` - 实验配置（ExperimentConfig）
- `suite.py` - 实验套件（ExperimentSuite）
- `result.py` - 结果数据结构（ExperimentResult, ExperimentRecord）
- `analyzer.py` - 统计分析（ResultAnalyzer）
- `charts.py` - 图表生成（ChartGenerator）
- `runner.py` - 实验运行器（ExperimentRunner）

## 注意事项

- 实验会自动管理服务器生命周期，无需手动启动/停止
- 使用 `--resume` 可从中断处恢复，避免重复运行
- 真实模式需确保模型和 adapter 路径正确
- 每个配置运行约 60 秒，完整套件需较长时间
- 建议先用小规模正式套件测试（如 dp-roundrobin-baseline）
- **w1/w2/w3 变化不会触发服务器重启**，通过 API 动态更新
