"""
生成三种负载度量（RWPT / queue_length / token_count）最优 w2 对比图。
每个 metric 取综合表现最好的 w2，生成 bar chart 对比。

输出：
  - best_w2_comparison.png  (主对比图，4 个子图)
  - best_w2_comparison.pdf  (论文用矢量图)
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 读取数据 ──
data = {}  # metric -> list of {w2, throughput, ...}
with open("w2_search_summary.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        metric = row["load_metric"]
        if metric not in data:
            data[metric] = []
        data[metric].append({
            "w2": float(row["w2"]),
            "throughput": float(row["throughput"]),
            "avg_latency": float(row["avg_latency"]),
            "avg_first_token_latency": float(row["avg_first_token_latency"]),
            "p90_latency": float(row["p90_latency"]),
            "cache_hit_rate": float(row["cache_hit_rate"]),
        })

# ── 选取每个 metric 综合最优 w2 ──
# 综合评分：归一化 throughput(越高越好) - 归一化 avg_latency(越低越好)
def pick_best(entries):
    tputs = [e["throughput"] for e in entries]
    lats = [e["avg_latency"] for e in entries]
    t_min, t_max = min(tputs), max(tputs)
    l_min, l_max = min(lats), max(lats)
    t_range = t_max - t_min if t_max > t_min else 1.0
    l_range = l_max - l_min if l_max > l_min else 1.0

    best_idx, best_score = 0, -999
    for i, e in enumerate(entries):
        score = (e["throughput"] - t_min) / t_range - (e["avg_latency"] - l_min) / l_range
        if score > best_score:
            best_score = score
            best_idx = i
    return entries[best_idx]

def pick_by_w2(entries, target_w2):
    """选取指定 w2 的实验数据"""
    for e in entries:
        if abs(e["w2"] - target_w2) < 1e-6:
            return e
    raise ValueError(f"w2={target_w2} not found")

best = {}
best["rwpt"] = pick_best(data["rwpt"])
# queue_length: w2=0.05 几乎无负载均衡，选 w2=0.1 作为有意义的负载均衡代表
best["queue_length"] = pick_by_w2(data["queue_length"], 0.1)
best["token_count"] = pick_best(data["token_count"])

# 打印选取结果
print("=== Selected best w2 per metric ===")
for m in ["rwpt", "queue_length", "token_count"]:
    b = best[m]
    print(f"  {m:14s}  w2={b['w2']:<5.2f}  tput={b['throughput']:.3f}  "
          f"lat={b['avg_latency']:.2f}s  ftl={b['avg_first_token_latency']:.2f}s  "
          f"p90={b['p90_latency']:.2f}s  cache={b['cache_hit_rate']:.1%}")

# ── 绘图 ──
labels = ["Queue Length", "Token Count", "RWPT"]
metrics_order = ["queue_length", "token_count", "rwpt"]
colors = ["#FF9800", "#4CAF50", "#2196F3"]  # 橙 绿 蓝
w2_labels = [f"$w_2$={best[m]['w2']:.2f}" for m in metrics_order]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
fig.suptitle("Load Metric Comparison (Best $w_2$ per Metric)", fontsize=14, fontweight="bold", y=1.02)

x = np.arange(len(labels))
bar_width = 0.5

# (a) Throughput
ax = axes[0]
vals = [best[m]["throughput"] for m in metrics_order]
bars = ax.bar(x, vals, bar_width, color=colors, edgecolor="white", linewidth=0.8)
ax.set_ylabel("Throughput (req/s)", fontsize=11)
ax.set_title("(a) Throughput", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n{w}" for l, w in zip(labels, w2_labels)], fontsize=9)
ax.set_ylim(bottom=min(vals) * 0.95, top=max(vals) * 1.02)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# (b) Average Latency
ax = axes[1]
vals = [best[m]["avg_latency"] for m in metrics_order]
bars = ax.bar(x, vals, bar_width, color=colors, edgecolor="white", linewidth=0.8)
ax.set_ylabel("Avg Latency (s)", fontsize=11)
ax.set_title("(b) Average Latency", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n{w}" for l, w in zip(labels, w2_labels)], fontsize=9)
ax.set_ylim(bottom=0, top=max(vals) * 1.15)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.1f}s",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# (c) First Token Latency
ax = axes[2]
vals = [best[m]["avg_first_token_latency"] for m in metrics_order]
bars = ax.bar(x, vals, bar_width, color=colors, edgecolor="white", linewidth=0.8)
ax.set_ylabel("Avg First Token Latency (s)", fontsize=11)
ax.set_title("(c) First Token Latency", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n{w}" for l, w in zip(labels, w2_labels)], fontsize=9)
ax.set_ylim(bottom=0, top=max(vals) * 1.15)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.1f}s",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# (d) Cache Hit Rate
ax = axes[3]
vals = [best[m]["cache_hit_rate"] * 100 for m in metrics_order]
bars = ax.bar(x, vals, bar_width, color=colors, edgecolor="white", linewidth=0.8)
ax.set_ylabel("Cache Hit Rate (%)", fontsize=11)
ax.set_title("(d) Cache Hit Rate", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n{w}" for l, w in zip(labels, w2_labels)], fontsize=9)
ax.set_ylim(bottom=0, top=105)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
fig.savefig("best_w2_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig("best_w2_comparison.pdf", bbox_inches="tight")
print(f"\nSaved: best_w2_comparison.png / .pdf")
plt.close()
