# 真实 Trace 生成与接入记录

本文记录将 Azure 真实 trace 转换为 6 req/s 和 8 req/s 请求序列，并接入 S-LoRA 路由实验框架的完整流程。

## 1. 数据来源

### 1.1 Azure Functions d01
- 原始文件：`/home/hzheng/datasets/azure_public/functions2019/invocations_per_function_md.anon.d01.csv`
- 用途：提取 function 调用次数，构造 adapter popularity 分布。
- 说明：这里只用于估计 adapter 热度分布，不改变请求本身的时间与 token 长度。

### 1.2 Azure LLM
- 原始文件：`/home/hzheng/datasets/azure_public/llm2024/AzureLLMInferenceTrace_conv_1week.csv`
- 用途：提供真实请求到达时间、context tokens、generated tokens。

## 2. 真实 trace 的生成逻辑

### 2.1 提取 Azure LLM 请求窗口
使用 `benchmarks/real_workload/preprocess_azure_llm_trace.py` 从 CSV 中截取固定时间窗口，并限制 token 范围。

核心步骤：
1. 读取 Azure LLM CSV。
2. 按 `start-offset-sec` 和 `duration` 截取窗口。
3. 可选过滤过长输入/输出。
4. 按目标到达率采样成固定的 `6 req/s` 或 `8 req/s`。
5. 生成 compact JSONL 请求序列。

对应脚本：
- `benchmarks/real_workload/preprocess_azure_llm_trace.py`
- `benchmarks/real_workload/azure_trace.py`

### 2.2 绑定 adapter popularity
使用 Azure Functions d01 的 HTTP top-100 调用分布，为每个请求分配 `adapter_id`。

当前采用的映射策略：
- `HTTP-only + top100 truncated`
- 原因：更贴近在线请求热点，同时避免 `top99 + tail` 把冷 function 聚成异常热门 adapter。

对应脚本：
- `benchmarks/real_workload/build_azure_llm_functions_workload.py`
- `benchmarks/real_workload/azure_functions_popularity_notes_zh.md`

### 2.3 最终输出
最终生成的真实 trace 文件是：

- `benchmarks/real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl`
- `benchmarks/real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl`

其中：
- `6rps / 8rps` 表示目标请求到达率。
- `180s` 表示实验窗口长度。
- `capped2048_512` 表示输入/输出长度使用了上限约束，避免超出模型限制。

## 3. 接入实验框架

### 3.1 Benchmark 侧支持 trace workload
实验框架增加了 `trace` workload：
- `benchmarks/routing_experiment/config.py`
- `benchmarks/routing_experiment/runner.py`
- `benchmarks/trace.py`

支持字段：
- `workload_type="trace"`
- `trace_file`
- `workload_name`

### 3.2 Suite 接入
新增 suite：
- `dp-realtrace-comparison`

配置为 4 组：
- round-robin + 6rps trace
- adapter-aware + 6rps trace
- round-robin + 8rps trace
- adapter-aware + 8rps trace

当前真实 trace 主线中，`adapter-aware + RWPT` 使用：

- `routing_w1 = 1.0`
- `routing_w2 = 3.0`
- `routing_w3 = 0.0`
- `load_metric = "rwpt"`

`routing_w2 = 3.0` 来自 6 rps 真实 trace 小范围搜索。相较 `w2=1.0`，
`w2=3.0` 在平均指标损失很小的情况下提供更稳定的 P90 latency / P90 TTFT，
因此作为当前论文主线设置。

对应位置：
- `benchmarks/routing_experiment/suite.py`

### 3.3 运行方式
示例：

```bash
cd /home/hzheng/S-LoRA/benchmarks
python run_routing_comparison.py \
  --suite dp-realtrace-comparison \
  --output-dir routing_comparison_results/realtrace_comparison_capped2048_512
```

## 4. 结果与状态

当前已确认：
- `6rps + round-robin` 已成功跑通。
- 其余 3 组仍在补跑中或已中断重跑。

## 5. 相关文件索引

- `benchmarks/real_workload/preprocess_azure_llm_trace.py`
- `benchmarks/real_workload/build_azure_llm_functions_workload.py`
- `benchmarks/real_workload/azure_trace.py`
- `benchmarks/real_workload/azure_functions_popularity_notes_zh.md`
- `benchmarks/routing_experiment/suite.py`
- `benchmarks/routing_experiment/runner.py`
- `benchmarks/routing_experiment/config.py`
- `benchmarks/trace.py`
