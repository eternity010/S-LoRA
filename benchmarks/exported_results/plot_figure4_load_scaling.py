#!/usr/bin/env python3
"""Plot the paper's three-panel load-scaling result."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "modelctx2048_external_baseline_summary.csv"
OUTPUT_DIR = ROOT / "figures"
PNG = OUTPUT_DIR / "figure4_load_scaling_modelctx2048.png"
PDF = OUTPUT_DIR / "figure4_load_scaling_modelctx2048.pdf"

METHODS = {
    "rlora_rr": {
        "label": "Round-Robin",
        "color": "#2f6690",
        "marker": "o",
        "linestyle": "--",
        "markerfacecolor": "white",
    },
    "rlora_rwpt_active": {
        "label": "RankFlow",
        "color": "#d97941",
        "marker": "s",
        "linestyle": "-",
        "markerfacecolor": "#d97941",
    },
}
RATES = (6.0, 8.0, 9.0, 10.0)


def load_rows():
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        (row["method"], float(row["req_rate"])): row
        for row in rows
        if row["backend"] == "rlora"
        and row["method"] in METHODS
        and float(row["req_rate"]) in RATES
    }
    missing = [
        (method, rate)
        for method in METHODS
        for rate in RATES
        if (method, rate) not in selected
    ]
    if missing:
        raise ValueError(f"Missing load-scaling results: {missing}")
    return selected


def plot_method(ax, rows, method, metric, scale=1.0):
    style = METHODS[method]
    values = [float(rows[(method, rate)][metric]) * scale for rate in RATES]
    ax.plot(
        RATES,
        values,
        label=style["label"],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        markerfacecolor=style["markerfacecolor"],
        markeredgecolor=style["color"],
        markeredgewidth=1.2,
        markersize=5.5,
        linewidth=1.8,
        zorder=3,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), dpi=180)
    panels = [
        (axes[0], "avg_ttft_s", "Average TTFT (s)", "(a) Average TTFT", 1.0, (0.2, 5.6)),
        (axes[1], "p90_ttft_s", "P90 TTFT (s)", "(b) P90 TTFT", 1.0, (0.5, 13.8)),
        (axes[2], "cache_hit_rate", "Cache hit rate (%)", "(c) Adapter cache hit rate", 100.0, (65, 83)),
    ]

    for ax, metric, ylabel, title, scale, ylim in panels:
        for method in METHODS:
            plot_method(ax, rows, method, metric, scale)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Offered load (req/s)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(RATES)
        ax.set_xlim(5.8, 10.2)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=2,
        columnspacing=1.8,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.35)
    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PNG}")
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
