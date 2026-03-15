"""
Academic-quality figures for alpha-robustness-sla experiment.
Generates 3 plots:
  Fig 1 - Grouped Bar: P90 TTFT vs alpha
  Fig 2 - Grouped Bar: Cache Hit Rate vs alpha
  Fig 3 - Pareto Scatter: Hit Rate vs P90 TTFT
"""

import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).parent
records = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]

# Strategy display names & style
STRATEGY_MAP = {
    "rwpt":         ("RWPT",         "#2166ac", "o"),
    "token_count":  ("Token Count",  "#d6604d", "s"),
    "queue_length": ("Queue Length", "#4dac26", "^"),
}
ALPHAS = [0.1, 0.3, 0.8]
ALPHA_LABELS = [r"$\alpha=0.1$", r"$\alpha=0.3$", r"$\alpha=0.8$"]

# ── Parse into lookup table ────────────────────────────────────────────────────
# data[load_metric][alpha] = {p90_ttft, cache_hit_rate}
data = {k: {} for k in STRATEGY_MAP}
for rec in records:
    cfg, res = rec["config"], rec["result"]
    metric = cfg["load_metric"]
    alpha  = cfg["alpha"]
    if metric in data:
        data[metric][alpha] = {
            "p90_ttft":      res["p90_first_token_latency"],
            "cache_hit_rate": res["cache_hit_rate"] * 100,  # → %
        }

# ── Shared style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       150,
})

strategies = ["queue_length", "token_count", "rwpt"]   # order: QL, TC, RWPT
n_groups   = len(ALPHAS)
n_bars     = len(strategies)
bar_width  = 0.22
x          = np.arange(n_groups)
offsets    = np.array([-1, 0, 1]) * bar_width

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – P90 TTFT Grouped Bar
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7, 4.5))

for i, (metric, offset) in enumerate(zip(strategies, offsets)):
    label, color, _ = STRATEGY_MAP[metric]
    values = [data[metric][a]["p90_ttft"] for a in ALPHAS]
    bars = ax1.bar(x + offset, values, bar_width, label=label,
                   color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
    # value labels on top
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

ax1.set_xticks(x)
ax1.set_xticklabels(ALPHA_LABELS)
ax1.set_xlabel("Traffic Skewness (Zipf α)")
ax1.set_ylabel("P90 TTFT (s)")
ax1.set_title("Fig 1 · P90 First-Token Latency vs. Traffic Skewness")
ax1.legend(loc="upper left", framealpha=0.85)
ax1.set_ylim(0, max(data[m][a]["p90_ttft"] for m in strategies for a in ALPHAS) * 1.18)
ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
ax1.set_axisbelow(True)

fig1.tight_layout()
out1 = HERE / "fig1_p90_ttft.pdf"
fig1.savefig(out1, bbox_inches="tight")
fig1.savefig(str(out1).replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Cache Hit Rate Grouped Bar
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 4.5))

for i, (metric, offset) in enumerate(zip(strategies, offsets)):
    label, color, _ = STRATEGY_MAP[metric]
    values = [data[metric][a]["cache_hit_rate"] for a in ALPHAS]
    bars = ax2.bar(x + offset, values, bar_width, label=label,
                   color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)

ax2.set_xticks(x)
ax2.set_xticklabels(ALPHA_LABELS)
ax2.set_xlabel("Traffic Skewness (Zipf α)")
ax2.set_ylabel("Cache Hit Rate (%)")
ax2.set_title("Fig 2 · KV-Cache Hit Rate vs. Traffic Skewness")
ax2.legend(loc="upper right", framealpha=0.85)
ax2.set_ylim(0, 105)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
ax2.set_axisbelow(True)

fig2.tight_layout()
out2 = HERE / "fig2_cache_hit_rate.pdf"
fig2.savefig(out2, bbox_inches="tight")
fig2.savefig(str(out2).replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Pareto Scatter: Hit Rate vs P90 TTFT
# ══════════════════════════════════════════════════════════════════════════════
ALPHA_MARKERS = ["o", "D", "^"]   # shape encodes alpha
ALPHA_SIZES   = [90, 80, 90]

fig3, ax3 = plt.subplots(figsize=(6.5, 5))

for metric in strategies:
    label, color, _ = STRATEGY_MAP[metric]
    for j, (alpha, mk, sz) in enumerate(zip(ALPHAS, ALPHA_MARKERS, ALPHA_SIZES)):
        hit  = data[metric][alpha]["cache_hit_rate"]
        ttft = data[metric][alpha]["p90_ttft"]
        ax3.scatter(hit, ttft, color=color, marker=mk, s=sz,
                    zorder=5, edgecolors="white", linewidths=0.7,
                    label=f"{label} (α={alpha})" if j == 0 else "_nolegend_")
        # annotate alpha value next to each point
        ax3.annotate(f"α={alpha}", (hit, ttft),
                     textcoords="offset points", xytext=(5, 3),
                     fontsize=7.5, color=color)

# Pareto frontier annotation arrow
ax3.annotate("Pareto\nFrontier →", xy=(0, 0), xytext=(0.05, 0.08),
             textcoords="axes fraction", fontsize=9, color="#555555",
             arrowprops=None)

# Ideal region shading (low TTFT, high hit rate)
ax3.axhline(y=min(data["rwpt"][a]["p90_ttft"] for a in ALPHAS) * 1.05,
            color="#2166ac", linestyle=":", linewidth=1, alpha=0.5)

ax3.set_xlabel("Cache Hit Rate (%)")
ax3.set_ylabel("P90 TTFT (s)  ↓ better")
ax3.set_title("Fig 3 · Pareto Trade-off: Cache Hit Rate vs. P90 TTFT")
ax3.invert_yaxis()   # lower TTFT = better → visually "up"

# Legend: strategy colors
strategy_patches = [
    mpatches.Patch(color=STRATEGY_MAP[m][1], label=STRATEGY_MAP[m][0])
    for m in strategies
]
# Legend: alpha shapes
shape_handles = [
    plt.scatter([], [], marker=mk, color="gray", s=sz, label=f"α={a}")
    for a, mk, sz in zip(ALPHAS, ALPHA_MARKERS, ALPHA_SIZES)
]
leg1 = ax3.legend(handles=strategy_patches, loc="lower left",
                  title="Strategy", framealpha=0.85, fontsize=9)
ax3.add_artist(leg1)
ax3.legend(handles=shape_handles, loc="upper right",
           title="Skewness", framealpha=0.85, fontsize=9)

ax3.yaxis.grid(True, linestyle="--", alpha=0.4)
ax3.xaxis.grid(True, linestyle="--", alpha=0.4)
ax3.set_axisbelow(True)

fig3.tight_layout()
out3 = HERE / "fig3_pareto_scatter.pdf"
fig3.savefig(out3, bbox_inches="tight")
fig3.savefig(str(out3).replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved {out3}")

plt.close("all")
print("Done. All figures saved to", HERE)
