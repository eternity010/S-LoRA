# Azure Functions d01 Adapter Popularity Notes

数据源：

- `/home/hzheng/datasets/azure_public/functions2019/invocations_per_function_md.anon.d01.csv`
- 每行按 `HashFunction` 统计一天内 `1..1440` 分钟列的总调用次数。
- adapter 映射暂只用于估计 popularity，不改变 Azure LLM 的 `req_time/input_len/output_len`。

## All Triggers

包含全部 Trigger 类型：

| Trigger | Function Count |
| --- | ---: |
| timer | 17405 |
| http | 15933 |
| queue | 6924 |
| orchestration | 3287 |
| event | 1245 |
| storage | 949 |
| others | 669 |

基础统计：

| Metric | Value |
| --- | ---: |
| functions | 46412 |
| total invocations | 909,783,379 |
| top1 share | 14.0% |
| top5 share | 33.7% |
| top10 share | 43.8% |
| top100 share | 76.6% |

映射成 100 adapters 的两种视图：

| Mapping | Total Invocations Covered | Top1 | Top5 | Top10 | Top20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| top100 truncated | 697,189,536 | 18.3% | 44.0% | 57.2% | 72.6% |
| top99 + tail | 909,783,379 | 23.5% | 54.7% | 65.5% | 78.3% |

观察：

- 全 Trigger 分布已经有明显长尾，但没有 HTTP-only 那么极端。
- `top100 truncated` 会丢弃尾部调用，归一化后热点会变强。
- `top99 + tail` 会把大量冷 function 聚合成一个超大 tail adapter，不太符合 LoRA adapter 的语义。

## HTTP Only

只保留 `Trigger == http`。

基础统计：

| Metric | Value |
| --- | ---: |
| functions | 15933 |
| total invocations | 195,994,220 |
| top1 share | 42.6% |
| top5 share | 56.9% |
| top10 share | 61.5% |
| top100 share | 84.4% |

映射成 100 adapters 的两种视图：

| Mapping | Total Invocations Covered | Top1 | Top5 | Top10 | Top20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| top100 truncated | 165,474,481 | 50.5% | 67.4% | 72.9% | 80.4% |
| top99 + tail | 195,994,220 | 42.6% | 71.4% | 76.4% | 83.0% |

观察：

- HTTP-only 更接近在线请求语义，但热点极强。
- 如果直接作为主实验 workload，RWPT 的 cache-affinity 优势可能会被放大。
- 更适合作为“真实高热点负载”补充实验。

## Current Recommendation

后续如果要从 Azure Functions 映射 adapter popularity，优先使用：

1. `HTTP-only + top100 truncated`：用于真实在线请求高热点场景。
2. `All-triggers + top100 truncated`：用于更宽泛的服务函数长尾场景。

暂不建议使用 `top99 + tail` 作为 adapter popularity，因为 tail 聚合会把很多冷 function 合成一个异常热门 adapter。
