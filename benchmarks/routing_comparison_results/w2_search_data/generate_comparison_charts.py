"""
生成三种负载度量（RWPT / queue_length / token_count）最优 w2 对比图。
两种选取模型各生成一组图：

模型 A — 三目标加权评分 (Weighted Multi-Objective):
  S(w2) = 0.3·T_norm + 0.4·(1 - P90_norm) + 0.3·C_norm

模型 B — SLA 约束最大化 (SLA-Constrained Optimization):
  max Cache_Hit_Rate
  s.t. P90 <= P90_min * (1 + ε),  ε=0.05
       Tput >= Tput_max * (1 - δ),  δ=0.02

输出：
  - best_w2_weighted.png / .pdf     (模型 A)
  - best_w2_sla.png / .pdf          (模型 B)
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 读取数据 ──
data = {}
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


# ── 模型 A: 三目标加权评分 ──
L1, L2, L3 = 0.3, 0.4, 0.3

def pick_weighted(entries):
    """S(w2) = 0.3·T_norm + 0.4·(1-P90_norm) + 0.3·C_norm"""
    tputs = [e["throughput"] for e in entries]
    p90s = [e["p90_latency"] for e in entries]
    caches = [e["cache_hit_rate"] for e in entries]
    def minmax(vals):
        vmin, vmax = min(vals), max(vals)
        r = vmax - vmin if vmax > vmin else 1.0
        return [(v - vmin) / r for v in vals]
    n_t = minmax(tputs)
    n_p = minmax(p90s)
    n_c = minmax(caches)
    best_idx, best_score = 0, -999
    for i in range(len(entries)):
        score = L1 * n_t[i] + L2 * (1 - n_p[i]) + L3 * n_c[i]
        if score > best_score:
            best_score = score
            best_idx = i
    return entries[best_idx]


# ── 模型 B: SLA 约束最大化 ──
EPSILON = 0.05  # P90 容忍度
DELTA = 0.02    # 吞吐下跌容忍度

def pick_sla(entries):
    """max cache_hit_rate s.t. P90 <= min*(1+ε), Tput >= max*(1-δ)"""
    p90_min = min(e["p90_latency"] for e in entries)
    tput_max = max(e["throughput"] for e in entries)
    p90_th = p90_min * (1 + EPSILON)
    tput_th = tput_max * (1 - DELTA)
    feasible = [e for e in entries
                if e["p90_latency"] <= p90_th and e["throughput"] >= tput_th]
    if not feasible:
        # fallback: 放松约束，取 P90 最低的
        return min(entries, key=lambda e: e["p90_latency"])
    return max(feasible, key=lambda e: e["cache_hit_rate"])


# ── 绘图函数 ──
labels = ["Queue Length", "Token Count", "RWPT"]
metrics_order = ["queue_length", "token_count", "rwpt"]
colors = ["#FF9800", "#4CAF50", "#2196F3"]  # 橙 绿 蓝


def draw_comparison(best, title_suffix, filename):
    """绘制 4 子图对比 bar chart"""
    w2_labels = [f"$w_2$={best[m]['w2']:.2f}" for m in metrics_order]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(f"Load Metric Comparison — {title_suffix}",
                 fontsize=14, fontweight="bold", y=1.02)

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

    # (b) P90 Latency
    ax = axes[1]
    vals = [best[m]["p90_latency"] for m in metrics_order]
    bars = ax.bar(x, vals, bar_width, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("P90 Latency (s)", fontsize=11)
    ax.set_title("(b) P90 Tail Latency", fontsize=12)
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
    fig.savefig(f"{filename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    print(f"  Saved: {filename}.png / .pdf")
    plt.close()


# ── 生成两组图 ──

# 模型 A: 加权评分
best_weighted = {m: pick_weighted(data[m]) for m in metrics_order}
print("=== Model A: Weighted Multi-Objective ===")
print(f"  S(w2) = {L1}·T + {L2}·(1-P90) + {L3}·C")
for m in metrics_order:
    b = best_weighted[m]
    print(f"  {m:15s} w2={b['w2']:<5.2f}  tput={b['throughput']:.3f}  "
          f"P90={b['p90_latency']:.2f}s  cache={b['cache_hit_rate']:.1%}")
draw_comparison(best_weighted,
                f"Weighted Score ($\\lambda$={L1},{L2},{L3})",
                "best_w2_weighted")

# 模型 B: SLA 约束
best_sla = {m: pick_sla(data[m]) for m in metrics_order}
print(f"\n=== Model B: SLA-Constrained (ε={EPSILON}, δ={DELTA}) ===")
print(f"  max Cache HR  s.t. P90 ≤ min*(1+{EPSILON}), Tput ≥ max*(1-{DELTA})")
for m in metrics_order:
    b = best_sla[m]
    print(f"  {m:15s} w2={b['w2']:<5.2f}  tput={b['throughput']:.3f}  "
          f"P90={b['p90_latency']:.2f}s  cache={b['cache_hit_rate']:.1%}")
draw_comparison(best_sla,
                f"SLA-Constrained ($\\epsilon$={EPSILON}, $\\delta$={DELTA})",
                "best_w2_sla")
