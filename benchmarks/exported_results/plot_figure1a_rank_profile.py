#!/usr/bin/env python3
"""Plot normalized per-token prefill cost across LoRA ranks."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MEASUREMENTS = (
    ROOT.parent
    / "routing_comparison_results/rank_prefill_profile/"
    "rank_prefill_measurements.jsonl"
)
OUTPUT_DIR = ROOT / "figures"
PNG = OUTPUT_DIR / "figure1a_rank_prefill_profile.png"
PDF = OUTPUT_DIR / "figure1a_rank_prefill_profile.pdf"
CSV = ROOT / "figure_data" / "fig1a_rank_prefill_profile.csv"
FIT = ROOT / "figure_data" / "fig1a_rank_prefill_fit.json"


def load_measurements() -> dict[int, dict[int, list[float]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    with MEASUREMENTS.open() as handle:
        for line in handle:
            row = json.loads(line)
            grouped[int(row["rank"])][int(row["prompt_tokens"])].append(
                float(row["ttft_s"])
            )
    return grouped


def main() -> None:
    grouped = load_measurements()
    ranks = np.array(sorted(grouped), dtype=float)
    slopes = []
    rows = []
    for rank in ranks.astype(int):
        lengths = np.array(sorted(grouped[rank]), dtype=float)
        medians = np.array(
            [np.median(grouped[rank][int(length)]) for length in lengths]
        )
        slope, intercept = np.polyfit(lengths, medians, 1)
        if slope <= 0:
            raise ValueError(f"Non-positive prefill slope for rank {rank}: {slope}")
        slopes.append(slope)
        for length, median in zip(lengths, medians):
            rows.append(
                {
                    "rank": rank,
                    "prompt_tokens": int(length),
                    "median_ttft_s": median,
                    "samples": len(grouped[rank][int(length)]),
                    "fitted_intercept_s": intercept,
                    "fitted_time_per_token_s": slope,
                }
            )

    slopes = np.array(slopes)
    normalized = slopes / slopes[ranks == 0][0]

    rng = np.random.default_rng(2027)
    bootstrap_normalized = []
    for _ in range(2000):
        sampled_slopes = []
        for rank in ranks.astype(int):
            lengths = np.array(sorted(grouped[rank]), dtype=float)
            sampled_medians = []
            for length in lengths.astype(int):
                values = np.array(grouped[rank][length])
                sampled = rng.choice(values, size=len(values), replace=True)
                sampled_medians.append(np.median(sampled))
            sampled_slope, _ = np.polyfit(lengths, sampled_medians, 1)
            sampled_slopes.append(sampled_slope)
        sampled_slopes = np.array(sampled_slopes)
        if np.all(sampled_slopes > 0):
            bootstrap_normalized.append(sampled_slopes / sampled_slopes[0])
    bootstrap_normalized = np.array(bootstrap_normalized)
    ci_lower, ci_upper = np.percentile(bootstrap_normalized, [2.5, 97.5], axis=0)

    delta = normalized - 1.0
    beta = max(0.0, float(np.dot(ranks, delta) / np.dot(ranks, ranks)))
    fitted = 1.0 + beta * ranks
    residual = float(np.sum((normalized - fitted) ** 2))
    total = float(np.sum((normalized - np.mean(normalized)) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 1.0

    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    FIT.write_text(
        json.dumps(
            {
                "beta_per_rank": beta,
                "r_squared": r_squared,
                "ranks": ranks.tolist(),
                "normalized_time_per_token": normalized.tolist(),
                "normalized_ci95_lower": ci_lower.tolist(),
                "normalized_ci95_upper": ci_upper.tolist(),
                "time_per_token_s": slopes.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.55, 2.65), dpi=180)
    dense_ranks = np.linspace(0, ranks.max(), 200)
    ax.plot(
        dense_ranks,
        1.0 + beta * dense_ranks,
        color="#6C8E8B",
        linewidth=2.0,
        label="Profiled linear fit",
        zorder=1,
    )
    ax.errorbar(
        ranks,
        normalized,
        yerr=np.vstack([normalized - ci_lower, ci_upper - normalized]),
        fmt="o",
        markersize=7.5,
        color="#D97941",
        markeredgecolor="white",
        markeredgewidth=0.8,
        ecolor="#B86139",
        elinewidth=1.0,
        capsize=2.5,
        label="Measured",
        zorder=2,
    )
    ax.text(
        0.04,
        0.95,
        rf"$\omega(r)=1+{beta:.4f}r$" + "\n" + rf"$R^2={r_squared:.4f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    ax.set_xlabel("LoRA rank", fontsize=9)
    ax.set_ylabel("Normalized prefill time / token", fontsize=9)
    ax.set_xticks(ranks.astype(int))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(pad=0.7)
    fig.savefig(PNG, dpi=300)
    fig.savefig(PDF)
    plt.close(fig)

    print(f"Profiled beta: {beta:.8f} per rank")
    print(f"R^2: {r_squared:.6f}")
    for rank, value in zip(ranks.astype(int), normalized):
        print(f"rank={rank:>2}: normalized prefill time/token={value:.6f}")
    print(f"Wrote {PNG}")
    print(f"Wrote {PDF}")
    print(f"Wrote {CSV}")
    print(f"Wrote {FIT}")


if __name__ == "__main__":
    main()
