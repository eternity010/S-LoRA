#!/usr/bin/env python3
"""Build reviewed cross-system results for the modelctx2048 workload."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent

RUN_FIELDS = [
    "backend",
    "method",
    "req_rate",
    "run_label",
    "server_state",
    "routing_strategy",
    "load_metric",
    "w2",
    "num_replicas",
    "gpu_ids",
    "total_requests",
    "throughput",
    "strip_throughput",
    "avg_latency_s",
    "avg_ttft_s",
    "p50_latency_s",
    "p90_latency_s",
    "p50_ttft_s",
    "p90_ttft_s",
    "cache_hit_rate",
    "worker_request_counts",
    "trace_file",
    "source_file",
    "timestamp",
    "notes",
]

SUMMARY_METRICS = [
    "throughput",
    "strip_throughput",
    "avg_latency_s",
    "avg_ttft_s",
    "p50_latency_s",
    "p90_latency_s",
    "p50_ttft_s",
    "p90_ttft_s",
    "cache_hit_rate",
]

SUMMARY_FIELDS = [
    "backend",
    "method",
    "req_rate",
    "runs",
    *SUMMARY_METRICS,
    "throughput_vs_rlora_rr_pct",
    "avg_latency_improvement_vs_rlora_rr_pct",
    "avg_ttft_improvement_vs_rlora_rr_pct",
    "p90_latency_improvement_vs_rlora_rr_pct",
    "p90_ttft_improvement_vs_rlora_rr_pct",
]


RUN_SPECS = [
    {
        "backend": "vllm-0.4.0",
        "method": "vllm_replicated_rr",
        "source": "routing_comparison_results/post_state_fix_v1/08_external_baselines/vllm_6rps/results.jsonl",
        "notes": "three independent single-GPU vLLM replicas; client-side strict round robin",
    },
    {
        "backend": "vllm-0.6.3",
        "method": "vllm_replicated_rr",
        "source": "routing_comparison_results/post_state_fix_v1/08_external_baselines/vllm063_8rps_singlefrontend/results.jsonl",
        "run_label": "run1",
        "notes": "vLLM 0.6.3; three independent replicas; frontend multiprocessing disabled; client-side strict round robin; single run",
    },
    {
        "backend": "slora-original",
        "method": "slora_original_3replica_rr",
        "source": "routing_comparison_results/post_state_fix_v1/08_external_baselines/slora_original_3replica_6rps/results.jsonl",
        "notes": "three independent original S-LoRA instances; client-side strict round robin",
    },
    {
        "backend": "slora-original",
        "method": "slora_original_3replica_rr",
        "source": "routing_comparison_results/post_state_fix_v1/08_external_baselines/slora_original_3replica_8rps/results.jsonl",
        "run_label": "run1",
        "notes": "three independent original S-LoRA instances; client-side strict round robin; single run",
    },
    *[
        {
            "backend": "rlora",
            "method": method,
            "source": source,
            "run_label": run_label,
            "notes": notes,
        }
        for method, source, run_label, notes in [
            (
                "rlora_rr",
                "routing_comparison_results/post_state_fix_v1/08_external_baselines/rlora_rr_6rps_run1/dp-realtrace-roundrobin-6rps-modelctx2048/results.jsonl",
                "run1",
                "fresh server",
            ),
            (
                "rlora_rr",
                "routing_comparison_results/post_state_fix_v1/08_external_baselines/rlora_rr_6rps_run2/dp-realtrace-roundrobin-6rps-modelctx2048/results.jsonl",
                "run2",
                "fresh server",
            ),
            (
                "rlora_rwpt_active",
                "routing_comparison_results/post_state_fix_v1/08_external_baselines/rlora_rwpt_active_6rps_run1/dp-realtrace-rwpt-active-6rps-modelctx2048/results.jsonl",
                "run1",
                "fresh server; w2=0.4",
            ),
            (
                "rlora_rwpt_active",
                "routing_comparison_results/post_state_fix_v1/08_external_baselines/rlora_rwpt_active_6rps_run2/dp-realtrace-rwpt-active-6rps-modelctx2048/results.jsonl",
                "run2",
                "fresh server; w2=0.4",
            ),
            *[
                (
                    method,
                    f"routing_comparison_results/post_state_fix_v1/08_external_baselines/{directory}/dp-realtrace-{suite}/results.jsonl",
                    "run1",
                    notes,
                )
                for method, directory, suite, notes in [
                    (
                        "rlora_rr",
                        "rlora_rr_8rps_run1",
                        "roundrobin-8rps-modelctx2048",
                        "fresh server; single run",
                    ),
                    (
                        "rlora_rwpt_active",
                        "rlora_rwpt_active_8rps_run1",
                        "rwpt-active-8rps-modelctx2048",
                        "fresh server; single run; w2=0.4",
                    ),
                    (
                        "rlora_rr",
                        "rlora_rr_9rps_run1",
                        "roundrobin-9rps-modelctx2048",
                        "fresh server; single run",
                    ),
                    (
                        "rlora_rwpt_active",
                        "rlora_rwpt_active_9rps_run1",
                        "rwpt-active-9rps-modelctx2048",
                        "fresh server; single run; w2=0.4",
                    ),
                    (
                        "rlora_rr",
                        "rlora_rr_10rps_run1",
                        "roundrobin-10rps-modelctx2048",
                        "fresh server; single run; overload point",
                    ),
                    (
                        "rlora_rwpt_active",
                        "rlora_rwpt_active_10rps_run1",
                        "rwpt-active-10rps-modelctx2048",
                        "fresh server; single run; overload point; w2=0.4",
                    ),
                ]
            ],
        ]
    ],
]


def _load_records(spec):
    source_path = BENCHMARKS_DIR / spec["source"]
    with source_path.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    run_label = spec.get("run_label")
    if run_label is not None:
        if len(records) != 1:
            raise ValueError(f"Expected one record in {source_path}, got {len(records)}")
        records[0]["config"]["run_label"] = run_label
    return records


def _normalize(spec, record):
    config = record["config"]
    result = record["result"]
    total_requests = result.get(
        "total_requests",
        config.get("total_requests", result.get("completed_requests")),
    )
    if total_requests is None:
        raise ValueError(f"Result has no request count: {spec['source']}")
    run_label = config.get("run_label", spec.get("run_label", "run1"))
    gpu_ids = config.get("gpu_ids", "")
    if isinstance(gpu_ids, list):
        gpu_ids = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    worker_counts = result.get("worker_request_counts", "")
    if worker_counts:
        worker_counts = json.dumps(worker_counts, sort_keys=True, separators=(",", ":"))
    return {
        "backend": spec["backend"],
        "method": spec["method"],
        "req_rate": config.get("req_rate", total_requests / config["duration"]),
        "run_label": run_label,
        "server_state": "fresh",
        "routing_strategy": config.get("routing_strategy", "round-robin"),
        "load_metric": config.get("load_metric", "none"),
        "w2": config.get("routing_w2", ""),
        "num_replicas": config.get("num_replicas", config.get("num_workers", "")),
        "gpu_ids": gpu_ids,
        "total_requests": total_requests,
        "throughput": result["throughput"],
        "strip_throughput": result["strip_throughput"],
        "avg_latency_s": result["avg_latency"],
        "avg_ttft_s": result["avg_first_token_latency"],
        "p50_latency_s": result["p50_latency"],
        "p90_latency_s": result["p90_latency"],
        "p50_ttft_s": result["p50_first_token_latency"],
        "p90_ttft_s": result["p90_first_token_latency"],
        "cache_hit_rate": result.get("cache_hit_rate", ""),
        "worker_request_counts": worker_counts,
        "trace_file": config["trace_file"],
        "source_file": spec["source"],
        "timestamp": record["timestamp"],
        "notes": spec["notes"],
    }


def build_run_rows():
    rows = []
    for spec in RUN_SPECS:
        rows.extend(_normalize(spec, record) for record in _load_records(spec))
    return sorted(rows, key=lambda row: (float(row["req_rate"]), row["backend"], row["method"], row["run_label"]))


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row[field] != ""]
    return statistics.fmean(values) if values else ""


def build_summary_rows(run_rows):
    groups = defaultdict(list)
    for row in run_rows:
        groups[(row["backend"], row["method"], float(row["req_rate"]))].append(row)

    summaries = []
    for (backend, method, req_rate), rows in sorted(groups.items(), key=lambda item: item[0]):
        summary = {
            "backend": backend,
            "method": method,
            "req_rate": req_rate,
            "runs": len(rows),
        }
        summary.update({field: _mean(rows, field) for field in SUMMARY_METRICS})
        summaries.append(summary)

    rr_by_rate = {
        row["req_rate"]: row
        for row in summaries
        if row["backend"] == "rlora" and row["method"] == "rlora_rr"
    }
    for row in summaries:
        rr = rr_by_rate.get(row["req_rate"])
        for field in SUMMARY_FIELDS[-5:]:
            row[field] = ""
        if rr is None or row is rr:
            continue
        row["throughput_vs_rlora_rr_pct"] = (
            float(row["throughput"]) / float(rr["throughput"]) - 1.0
        ) * 100.0
        for metric, output_field in [
            ("avg_latency_s", "avg_latency_improvement_vs_rlora_rr_pct"),
            ("avg_ttft_s", "avg_ttft_improvement_vs_rlora_rr_pct"),
            ("p90_latency_s", "p90_latency_improvement_vs_rlora_rr_pct"),
            ("p90_ttft_s", "p90_ttft_improvement_vs_rlora_rr_pct"),
        ]:
            row[output_field] = (
                1.0 - float(row[metric]) / float(rr[metric])
            ) * 100.0
    return summaries


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    run_rows = build_run_rows()
    summary_rows = build_summary_rows(run_rows)
    runs_path = OUTPUT_DIR / "modelctx2048_external_baseline_runs.csv"
    summary_path = OUTPUT_DIR / "modelctx2048_external_baseline_summary.csv"
    _write_csv(runs_path, RUN_FIELDS, run_rows)
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    print(f"Wrote {len(run_rows)} runs to {runs_path}")
    print(f"Wrote {len(summary_rows)} summaries to {summary_path}")


if __name__ == "__main__":
    main()
