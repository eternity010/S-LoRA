#!/usr/bin/env python3
"""Plot the lifecycle blind spot from deduplicated worker reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TRACE = (
    ROOT.parent
    / "routing_comparison_results/post_state_fix_v1/09_figure1_motivation/"
    "rwpt_active_8rps_debug/"
    "dp-realtrace-rwpt-active-8rps-modelctx2048/diagnostics/"
    "server_20260722_220131_977528/route_decisions.jsonl"
)
OUTPUT_DIR = ROOT / "figures"
PNG = OUTPUT_DIR / "figure1b_lifecycle_blindspot.png"
PDF = OUTPUT_DIR / "figure1b_lifecycle_blindspot.pdf"


def load_unique_reports() -> list[tuple[int, int]]:
    reports: dict[tuple[str, int], tuple[int, int]] = {}
    with TRACE.open() as handle:
        for line in handle:
            decision = json.loads(line)
            for worker_id, state in decision["workers"].items():
                reports[(worker_id, state["report_seq"])] = (
                    state["waiting_request_count"],
                    state["current_batch_size"],
                )
    return list(reports.values())


def main() -> None:
    samples = load_unique_reports()
    active_samples = [sample for sample in samples if sample[1] > 0]
    blind_samples = [sample for sample in active_samples if sample[0] == 0]

    mean_waiting = sum(waiting for waiting, _ in samples) / len(samples)
    mean_active = sum(active for _, active in samples) / len(samples)
    blind_ratio = len(blind_samples) / len(active_samples)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.55, 2.65), dpi=180)

    bars = ax.bar(
        [0, 1],
        [mean_waiting, mean_active],
        width=0.58,
        color=["#6C8E8B", "#D97941"],
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    labels = [f"{mean_waiting:.2f}", f"{mean_active:.2f}"]
    for bar, label in zip(bars, labels):
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.text(
        0.04,
        0.95,
        f"{blind_ratio:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#B55233",
    )
    ax.text(
        0.04,
        0.79,
        "of active states report\nzero waiting requests",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        linespacing=1.15,
        color="#333333",
    )

    ax.set_ylabel("Mean requests per worker", fontsize=9)
    ax.set_xticks([0, 1], ["Waiting", "Active"])
    ax.set_ylim(0, 10.4)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.7)

    fig.savefig(PNG, dpi=300)
    fig.savefig(PDF)
    plt.close(fig)

    print(f"Unique worker reports: {len(samples)}")
    print(f"Reports with active requests: {len(active_samples)}")
    print(f"Active reports with zero waiting: {len(blind_samples)}")
    print(f"Blind-spot ratio: {blind_ratio:.4%}")
    print(f"Mean waiting requests: {mean_waiting:.4f}")
    print(f"Mean active requests: {mean_active:.4f}")
    print(f"Wrote {PNG}")
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
