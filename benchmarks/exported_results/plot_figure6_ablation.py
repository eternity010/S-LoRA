#!/usr/bin/env python3
"""Plot the paper's single-column ablation figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
LOAD_DATA = ROOT / "figure_data" / "fig4_load_metric_ablation_9rps_modelctx2048.csv"
LIFECYCLE_DATA = ROOT / "figure_data" / "fig7_active_request_ablation_9rps_modelctx2048.csv"
OUTPUT_DIR = ROOT / "figures"
LOAD_PNG = OUTPUT_DIR / "figure6_load_metric_ablation_9rps_modelctx2048.png"
LOAD_PDF = OUTPUT_DIR / "figure6_load_metric_ablation_9rps_modelctx2048.pdf"
LIFECYCLE_PNG = OUTPUT_DIR / "figure7_lifecycle_ablation_9rps_modelctx2048.png"
LIFECYCLE_PDF = OUTPUT_DIR / "figure7_lifecycle_ablation_9rps_modelctx2048.pdf"

LOAD_METHODS = [
    ("Queue Length", "Queue\nLength", "#5b8e7d"),
    ("Token Count Active", "Token\nCount", "#e0a458"),
    ("RWPT Active", "RWPT", "#d97941"),
]
LIFECYCLE_METHODS = [
    ("TC", "TC-\nPrefill", "#a8c5bc"),
    ("TC Active", "TC", "#5b8e7d"),
    ("RWPT", "RWPT-\nPrefill", "#efb184"),
    ("RWPT Active", "RWPT", "#d97941"),
]


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows, methods):
    by_method = {row["method"]: row for row in rows}
    missing = [source_name for source_name, _, _ in methods if source_name not in by_method]
    if missing:
        raise ValueError(f"Missing ablation rows: {missing}")
    selected = [by_method[source_name] for source_name, _, _ in methods]
    traces = {row["trace_file"] for row in selected}
    rates = {float(row["req_rate"]) for row in selected}
    if len(traces) != 1 or rates != {9.0}:
        raise ValueError("Ablation rows must use one common 9 RPS trace")
    return selected


def plot_panel(ax, rows, methods, upper):
    x = np.arange(len(methods))
    width = 0.34
    for offset, metric, label, color in [
        (-width / 2, "avg_ttft_s", "Average TTFT", "#5b8e7d"),
        (width / 2, "p90_ttft_s", "P90 TTFT", "#d97941"),
    ]:
        values = np.array([float(row[metric]) for row in rows])
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=label,
            zorder=2,
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )
    ax.set_ylabel("TTFT (s)")
    ax.set_xticks(x, [label for _, label, _ in methods])
    ax.set_ylim(0, upper)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render(rows, methods, upper, png, pdf):
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.05), dpi=180)
    plot_panel(ax, rows, methods, upper)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        frameon=False,
        ncol=2,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_rows = select_rows(read_rows(LOAD_DATA), LOAD_METHODS)
    lifecycle_rows = select_rows(read_rows(LIFECYCLE_DATA), LIFECYCLE_METHODS)
    render(load_rows, LOAD_METHODS, 9.2, LOAD_PNG, LOAD_PDF)
    render(
        lifecycle_rows,
        LIFECYCLE_METHODS,
        4.8,
        LIFECYCLE_PNG,
        LIFECYCLE_PDF,
    )


if __name__ == "__main__":
    main()
