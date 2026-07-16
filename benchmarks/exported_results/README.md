# Exported Experiment Results

本目录只整理状态上报修复后的真实 trace 实验。Synthetic workload 暂不进入当前主线汇总。

## 快速入口

| 想看什么 | 文件 |
|---|---|
| RR 与最终 RWPT Active 在 4/6/8 RPS 下的主结果 | `main_realtrace_rate_summary.csv` |
| 6 RPS 下 RR、原 RWPT、TC Active 与 RWPT Active 对比 | `active_metric_comparison_6rps.csv` |
| 8 RPS 下 RR、QL、TC、原 RWPT 和 RWPT Active 的消融 | `load_metric_ablation_8rps.csv` |
| 每一轮实验的完整指标和原始结果路径 | `manual_experiment_registry.csv` |
| 台账字段、状态和使用规则 | `manual_experiment_registry_notes.md` |

## 当前主线配置

- Workload: Azure 融合真实 trace，180 秒，100 adapters。
- Router: `adapter-aware`。
- Load metric: `rwpt_active`。
- Weight: `w1=1.0`, `w2=0.4`, `w3=0.0`。
- 每个 RPS 使用 fresh server run 和 cache-reset/reuse run，共两轮并取算术平均。

## 当前结论

| RPS | RR 平均延迟 | RWPT Active 平均延迟 | 延迟改善 | P90 改善 | Cache hit 变化 |
|---:|---:|---:|---:|---:|---:|
| 4 | 2.324 s | 2.385 s | -2.65% | -7.19% | +9.72 pp |
| 6 | 3.641 s | 3.314 s | +8.96% | +18.00% | +8.98 pp |
| 8 | 10.423 s | 8.763 s | +15.92% | +26.12% | unavailable |

4 RPS 下 RWPT Active 的缓存命中率更高，但延迟略差；6/8 RPS 下负载惩罚开始产生明确收益。8 RPS 的 RR 实验早于 RR cache observation 修复，因此不能计算 cache hit 差值。

修复 active-request 盲区后，TC Active 在 6/8 RPS 均明显优于 waiting-only TC。RWPT Active 相对 TC Active 在 6 RPS 的平均延迟改善为 1.79%、P90 改善为 4.30%；在 8 RPS 的平均延迟改善为 3.75%、P90 改善为 2.97%。

## 数据规则

- 汇总 CSV 只包含两轮均值，便于画图和快速比较。
- `manual_experiment_registry.csv` 保留逐轮数据，是汇总表的依据。
- 原始 `results.jsonl` 不移动；通过台账的 `source_file` 回溯。
- 原 `rwpt` 标记为 `rwpt_prefill`，作为 prefill-only 消融，不再代表最终算法。
- `cache_hit_rate` 使用 `0..1` 比例；`*_pct` 是百分比；`*_pp` 是百分点。
