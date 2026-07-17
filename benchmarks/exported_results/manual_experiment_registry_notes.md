# Manual Experiment Registry Notes

`manual_experiment_registry.csv` 是人工复核后的逐轮实验台账。它保存原始运行级指标和 `results.jsonl` 路径；面向论文阅读的均值结果位于同目录下的两个 summary CSV。

## 当前范围

- 只登记状态上报修复后的 real trace 结果。
- 当前主线算法为 `rwpt_active, w2=0.4`。
- `rwpt, w2=3.0` 保留为 prefill-only 历史消融。
- Synthetic workload 暂不纳入当前整理结果。
- 4/6/8 RPS 的 RR 和 RWPT Active 主线均各有两轮有效记录。
- 6/8 RPS 的 TC Active 均各有两轮有效记录。
- 8 RPS 的 QL、waiting-only TC 和原 RWPT 各有两轮有效消融记录。
- 8 RPS rank-swapped 下的 TC Active 和 RWPT Active 均各有两轮有效记录。

## 汇总关系

- `main_realtrace_rate_summary.csv`
  - 使用 RR 与 RWPT Active 在 4/6/8 RPS 下的两轮算术平均。
  - 正向的 `*_improvement_vs_rr_pct` 表示 RWPT Active 优于 RR。
- `load_metric_ablation_8rps.csv`
  - 使用每种 load metric 的两轮算术平均。
  - 用于比较 QL、waiting-only TC、TC Active、prefill-only RWPT 和最终 RWPT Active。
- `active_metric_comparison_6rps.csv`
  - 比较 RR、prefill-only RWPT、TC Active 和最终 RWPT Active。
  - 6 RPS 尚无修复后的 QL 重跑，因此不称为完整 load-metric 消融。
- `figure_data/`
  - 保存小论文四张核心实验图的逐轮输入数据。
  - 由 `build_figure_data.py` 从本台账按明确 suite 名生成。

## 台账规则

- 原始 `results.jsonl` 不移动、不删除，CSV 通过 `source_file` 建立索引。
- 同一配置重复运行时新增记录，用不同 `record_id` 和 `run_label` 区分。
- 发现结果不可靠时不删除，将 `status` 改为 `excluded` 或 `suspect` 并在 `notes` 说明。
- `include_in_paper` 建议值：
  - `candidate`: 论文候选数据；
  - `yes`: 已决定进入论文；
  - `no`: 不进入论文；
  - `supporting`: 校准或辅助结果；
  - `debug`: 仅用于诊断。
- `status` 建议值：
  - `usable`: 已复核可用；
  - `excluded`: 明确排除；
  - `suspect`: 有疑点；
  - `failed`: 实验失败。

## 指标注意事项

- `cache_hit_rate` 使用 `0..1` 比例，而不是百分数。
- 8 RPS 的两轮 RR 运行早于 round-robin cache observation 修复，台账中的零值代表未收集。汇总表将其留空，不能解释为实际命中率为 0%。
- 当前不使用 `p95_first_token_latency`，近期原始结果中该字段固定为 `0.0`，不可信。
- `source_file` 和 `timestamp` 共同定位原始运行，复核时以原始 JSONL 为准。
