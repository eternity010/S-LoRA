#!/usr/bin/env python3
"""Build paper figure inputs from the reviewed experiment registry."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "manual_experiment_registry.csv"
OUTPUT_DIR = ROOT / "figure_data"

OUTPUT_FIELDS = [
    "scenario",
    "req_rate",
    "method",
    "routing_strategy",
    "load_metric",
    "w2",
    "run",
    "server_state",
    "throughput",
    "strip_throughput",
    "avg_latency_s",
    "avg_ttft_s",
    "p50_latency_s",
    "p90_latency_s",
    "p50_ttft_s",
    "p90_ttft_s",
    "cache_hit_rate",
    "rank64_request_share",
    "rank64_input_token_share",
    "source_file",
    "timestamp",
]


def load_registry():
    with REGISTRY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["status"] == "usable"]


def run_number(row):
    match = re.search(r"run([12])(?:_|$)", row["run_label"])
    if not match:
        raise ValueError(f"Cannot determine run number from {row['run_label']}")
    return int(match.group(1))


def figure_row(row, scenario, method, rank64_request_share="", rank64_token_share=""):
    run = run_number(row)
    cache_hit_rate = row["cache_hit_rate"]
    if row["routing_strategy"] == "round-robin" and "unavailable" in row["notes"]:
        cache_hit_rate = ""
    return {
        "scenario": scenario,
        "req_rate": row["req_rate"],
        "method": method,
        "routing_strategy": row["routing_strategy"],
        "load_metric": row["load_metric"],
        "w2": row["routing_w2"] if row["routing_strategy"] != "round-robin" else "",
        "run": run,
        "server_state": "fresh" if run == 1 else "cache_reset_reuse",
        "throughput": row["throughput"],
        "strip_throughput": row["strip_throughput"],
        "avg_latency_s": row["avg_latency"],
        "avg_ttft_s": row["avg_first_token_latency"],
        "p50_latency_s": row["p50_latency"],
        "p90_latency_s": row["p90_latency"],
        "p50_ttft_s": row["p50_first_token_latency"],
        "p90_ttft_s": row["p90_first_token_latency"],
        "cache_hit_rate": cache_hit_rate,
        "rank64_request_share": rank64_request_share,
        "rank64_input_token_share": rank64_token_share,
        "source_file": row["source_file"],
        "timestamp": row["timestamp"],
    }


def select_suite(rows, suite_name):
    selected = [row for row in rows if row["suite_or_group"] == suite_name]
    if len(selected) != 2:
        raise ValueError(f"Expected two rows for {suite_name}, got {len(selected)}")
    return sorted(selected, key=run_number)


def write_figure(name, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_rate_scaling(rows):
    output = []
    for rate in (4, 6, 8):
        suites = [
            (f"dp-realtrace-roundrobin-{rate}rps-repeat2", "Round-Robin"),
            (f"dp-realtrace-rwpt-active-{rate}rps-repeat2", "RWPT Active"),
        ]
        for suite, method in suites:
            output.extend(
                figure_row(row, "realtrace_rate_scaling", method)
                for row in select_suite(rows, suite)
            )
    write_figure("fig1_realtrace_rate_scaling.csv", output)


def build_load_metric_comparison(rows):
    suites = [
        ("dp-realtrace-roundrobin-8rps-repeat2", "Round-Robin"),
        ("dp-realtrace-queue-length-8rps-repeat2", "Queue Length"),
        ("dp-realtrace-token-count-active-8rps-repeat2", "Token Count Active"),
        ("dp-realtrace-rwpt-active-8rps-repeat2", "RWPT Active"),
    ]
    output = []
    for suite, method in suites:
        output.extend(
            figure_row(row, "realtrace_8rps", method)
            for row in select_suite(rows, suite)
        )
    write_figure("fig2_load_metric_comparison_8rps.csv", output)


def build_active_request_ablation(rows):
    suites = [
        ("dp-realtrace-token-count-8rps-repeat2", "Token Count Waiting"),
        ("dp-realtrace-token-count-active-8rps-repeat2", "Token Count Active"),
        ("dp-realtrace-rwpt-8rps-repeat2", "RWPT Prefill"),
        ("dp-realtrace-rwpt-active-8rps-repeat2", "RWPT Active"),
    ]
    output = []
    for suite, method in suites:
        output.extend(
            figure_row(row, "active_request_ablation_8rps", method)
            for row in select_suite(rows, suite)
        )
    write_figure("fig3_active_request_ablation_8rps.csv", output)


def build_rank_mapping_ablation(rows):
    scenarios = [
        (
            "low_rank_hot",
            "0.2743",
            "0.2639",
            [
                ("dp-realtrace-token-count-active-8rps-repeat2", "Token Count Active"),
                ("dp-realtrace-rwpt-active-8rps-repeat2", "RWPT Active"),
            ],
        ),
        (
            "high_rank_hot",
            "0.7257",
            "0.7361",
            [
                (
                    "dp-realtrace-token-count-active-8rps-rank-swapped-repeat2",
                    "Token Count Active",
                ),
                (
                    "dp-realtrace-rwpt-active-8rps-rank-swapped-repeat2",
                    "RWPT Active",
                ),
            ],
        ),
    ]
    output = []
    for scenario, request_share, token_share, suites in scenarios:
        for suite, method in suites:
            output.extend(
                figure_row(row, scenario, method, request_share, token_share)
                for row in select_suite(rows, suite)
            )
    write_figure("fig4_rank_mapping_ablation_8rps.csv", output)


def main():
    rows = load_registry()
    build_rate_scaling(rows)
    build_load_metric_comparison(rows)
    build_active_request_ablation(rows)
    build_rank_mapping_ablation(rows)
    print(f"Figure data written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
