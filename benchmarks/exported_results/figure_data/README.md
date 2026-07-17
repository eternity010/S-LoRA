# Paper Figure Data

这些 CSV 由 `../build_figure_data.py` 从人工复核台账生成。每行对应一次实际运行，
`run=1` 是 fresh server，`run=2` 是确认 cache reset 后复用服务器。

| 文件 | 用途 |
|---|---|
| `fig1_realtrace_rate_scaling.csv` | 4/6/8 RPS 下 RR 与 RWPT Active 压力曲线 |
| `fig2_load_metric_comparison_8rps.csv` | 8 RPS 下 RR、QL、TC Active、RWPT Active 对比 |
| `fig3_active_request_ablation_8rps.csv` | waiting-only 与 active-aware 负载定义消融 |
| `fig4_rank_mapping_ablation_8rps.csv` | 低-rank热点与高-rank热点映射消融 |

绘图时按 `scenario + method` 对两轮取算术平均。当前只有两轮数据，建议显示两个
原始点或 min-max 范围，不应绘制置信区间。8 RPS Round-Robin 的 cache hit 未采集，
对应单元格为空。

重新生成：

```bash
cd /home/hzheng/S-LoRA/benchmarks/exported_results
python build_figure_data.py
```
