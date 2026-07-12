# Manual Experiment Registry Notes

`manual_experiment_registry.csv` 是人工确认后的实验结果台账。它不是自动扫描产物，原则是每次实验结束后只追加“确认可用”的记录。

## 当前已恢复的数据

当前只登记人工复核过的 realtrace 4/6/8rps 数据：

- `dp-realtrace-rwpt-4rps-debug`
  - 记录两轮 debug 中延迟与尾延迟更优的一轮。

- `dp-realtrace-roundrobin-6rps`
  - round-robin 基线 1 条。
- `dp-realtrace-w2-search`
  - adapter-aware + rwpt，已完成两轮搜索：
    - 粗搜：w2 = 1.0, 1.2, 1.5, 2.0, 3.0；
    - 细搜：w2 = 2.5, 3.0, 3.5。
  - 当前真实 trace 主线选择 `w2 = 3.0`。
- `dp-realtrace-comparison`
  - 记录已人工确认的 8rps round-robin 对照结果。

没有恢复旧的 baseline 导出、rate scaling 导出图、旧 JSON/CSV 汇总文件。

## 使用原则

- 原始 `results.jsonl` 不移动、不删除，CSV 只保存人工确认后的索引和关键指标。
- 如果同一配置重复跑多次，可以新增多行，用不同 `record_id` / `run_label` 区分。
- 如果某条结果后来发现不可靠，不删除该行，把 `status` 改为 `excluded`，并在 `notes` 写原因。
- `include_in_paper` 可用值建议：
  - `candidate`: 候选可用结果；
  - `yes`: 已决定进入论文图表；
  - `no`: 不用于论文；
  - `debug`: 仅调试参考。
- `status` 可用值建议：
  - `usable`: 已检查且可用于观察；
  - `excluded`: 明确排除；
  - `suspect`: 有疑点，暂不下结论；
  - `failed`: 失败实验记录。

## 注意事项

- round-robin 路径当前没有真实 cache hit 统计，`cache_hit_rate=0` 不能解释为真实缓存命中率为 0。
- 当前表里没有使用 `p95_first_token_latency`，因为近期结果中该字段为 `0.0`，明显不可信。
- `source_file` 指向原始 `results.jsonl`，后续复核时以原始文件为证据。
