#!/usr/bin/env python3
"""Plot Figure 5: external-system comparison at 8 req/s."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "modelctx2048_external_baseline_summary.csv"
OUTPUT_DIR = ROOT / "figures"
PNG = OUTPUT_DIR / "figure5_external_system_comparison_8rps.png"
PDF = OUTPUT_DIR / "figure5_external_system_comparison_8rps.pdf"

SYSTEMS = [
    ("vllm_replicated_rr", "vLLM", "#6c7a89"),
    ("slora_original_3replica_rr", "S-LoRA", "#5b8e7d"),
    ("rlora_rwpt_active", "RankFlow", "#d97941"),
]


def load_rows():
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for row in rows:
        if float(row["req_rate"]) != 8.0:
            continue
        if row["method"] == "vllm_replicated_rr" and row["backend"] != "vllm-0.6.3":
            continue
        selected[row["method"]] = row
    missing = [method for method, _, _ in SYSTEMS if method not in selected]
    if missing:
        raise ValueError(f"Missing 8 RPS external baselines: {missing}")
    return [selected[method] for method, _, _ in SYSTEMS]


def add_value_labels(ax, bars, values):
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    labels = [label for _, label, _ in SYSTEMS]
    colors = [color for _, _, color in SYSTEMS]
    x = np.arange(len(SYSTEMS))

    panels = [
        ("throughput", "Throughput (req/s)", False, (4.5, 8.0), None),
        (
            "avg_ttft_s",
            "Average TTFT (s)",
            True,
            (0.4, 130),
            [0.5, 1, 2, 5, 10, 20, 50, 100],
        ),
        (
            "p90_ttft_s",
            "P90 TTFT (s)",
            True,
            (0.8, 160),
            [1, 2, 5, 10, 20, 50, 100],
        ),
    ]

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), dpi=180)
    for ax, (metric, ylabel, use_log, limits, ticks) in zip(axes, panels):
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(
            x,
            values,
            width=0.64,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        if use_log:
            ax.set_yscale("log")
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{tick:g}" for tick in ticks])
        ax.set_ylim(*limits)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", which="both", alpha=0.22, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_value_labels(ax, bars, values)

    fig.tight_layout(w_pad=1.1)
    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PNG}")
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
