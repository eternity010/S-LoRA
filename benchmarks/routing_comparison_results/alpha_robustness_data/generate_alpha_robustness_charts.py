"""
生成 Alpha 鲁棒性实验对比图（SLA 约束模型 w2）。
每个 alpha 下三种 metric 的表现对比（grouped bar chart）。

w2 选取：SLA 约束最大化模型
  max Cache_HR  s.t. P90 ≤ P90_min*(1+0.05), Tput ≥ Tput_max*(1-0.02)
  rwpt: w2=1.0, token_count: w2=3.5, queue_length: w2=0.15

输出：
  - alpha_robustness_comparison.png  (主对比图)
  - alpha_robustness_comparison.pdf  (论文用矢量图)
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 读取数据 ──
data = {}  # (alpha, metric) -> row dict
with open("alpha_robustness_summary.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        alpha = float(row["alpha"])
        metric = row["load_metric"]
        data[(alpha, metric)] = {
            "throughput": float(row["throughput"]),
            "avg_latency": float(row["avg_latency"]),
            "avg_first_token_latency": float(row["avg_first_token_latency"]),
            "p90_latency": float(row["p90_latency"]),
            "cache_hit_rate": float(row["cache_hit_rate"]),
        }

alphas = [0.1, 0.3, 0.8]
metrics_order = ["queue_length", "token_count", "rwpt"]
metric_labels = ["Queue Length", "Token Count", "RWPT"]
colors = ["#FF9800", "#4CAF50", "#2196F3"]  # 橙 绿 蓝

alpha_labels = [r"$\alpha$=0.1" + "\n(dispersed)",
                r"$\alpha$=0.3" + "\n(moderate)",
                r"$\alpha$=0.8" + "\n(concentrated)"]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(r"Load Metric Robustness Across $\alpha$ ($w_2$: RWPT=1.0, TC=1.0, QL=0.15)",
             fontsize=14, fontweight="bold", y=1.02)

x = np.arange(len(alphas))
bar_width = 0.22
offsets = [-bar_width, 0, bar_width]

chart_configs = [
    ("throughput", "Throughput (req/s)", "(a) Throughput", False),
    ("p90_latency", "P90 Latency (s)", "(b) P90 Tail Latency", True),
    ("avg_first_token_latency", "Avg First Token Latency (s)", "(c) First Token Latency", True),
    ("cache_hit_rate", "Cache Hit Rate (%)", "(d) Cache Hit Rate", False),
]

for ax, (key, ylabel, title, lower_better) in zip(axes, chart_configs):
    # 先收集所有值，确定 y 轴范围
    all_vals = []
    for alpha in alphas:
        for metric in metrics_order:
            v = data[(alpha, metric)][key]
            if key == "cache_hit_rate":
                v *= 100
            all_vals.append(v)

    if key == "cache_hit_rate":
        y_bottom, y_top = 0, 105
    elif key == "throughput":
        y_bottom, y_top = min(all_vals) * 0.97, max(all_vals) * 1.04
    else:
        y_bottom, y_top = 0, max(all_vals) * 1.2

    # 标注偏移量 = y 轴范围的 1.5%
    label_offset = (y_top - y_bottom) * 0.015

    for j, (metric, label, color) in enumerate(zip(metrics_order, metric_labels, colors)):
        vals = []
        for alpha in alphas:
            v = data[(alpha, metric)][key]
            if key == "cache_hit_rate":
                v *= 100
            vals.append(v)
        bars = ax.bar(x + offsets[j], vals, bar_width, label=label,
                      color=color, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.1f}%" if key == "cache_hit_rate" else (f"{v:.2f}" if key == "throughput" else f"{v:.1f}s")
            ax.text(bar.get_x() + bar.get_width() / 2, v + label_offset,
                    fmt, ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels, fontsize=9)
    ax.set_ylim(bottom=y_bottom, top=y_top)

axes[0].legend(fontsize=9, loc="lower left")

plt.tight_layout()
fig.savefig("alpha_robustness_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig("alpha_robustness_comparison.pdf", bbox_inches="tight")
print("Saved: alpha_robustness_comparison.png / .pdf")
plt.close()

# 打印 RWPT 优势分析
print("\n=== RWPT Advantage Summary ===")
for alpha in alphas:
    rwpt = data[(alpha, "rwpt")]
    tc = data[(alpha, "token_count")]
    ql = data[(alpha, "queue_length")]
    print(f"alpha={alpha}: RWPT P90={rwpt['p90_latency']:.2f}s  "
          f"vs TC {tc['p90_latency']:.2f}s ({(tc['p90_latency']-rwpt['p90_latency'])/tc['p90_latency']*100:+.1f}%)  "
          f"vs QL {ql['p90_latency']:.2f}s ({(ql['p90_latency']-rwpt['p90_latency'])/ql['p90_latency']*100:+.1f}%)")
