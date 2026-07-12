"""Tests for request-level latency and routing diagnostics."""

import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "benchmarks"))

from routing_experiment.request_diagnostics import join_request_diagnostics


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _worker_state(selected=False, load=0.0, queue=0, decode=0, prefill=0, cached=False):
    return {
        "selected": selected,
        "score": 1.0 if selected else 0.0,
        "has_adapter": cached,
        "cache_bonus": 1.0 if cached else 0.0,
        "load_pressure": load,
        "queue_length": queue,
        "pending_prefill_tokens": prefill,
        "pending_raw_tokens": prefill,
        "active_decode_seqs": decode,
        "cached_adapters": 1 if cached else 0,
        "is_healthy": True,
    }


def test_join_request_diagnostics_matches_ids_and_summarizes(tmp_path):
    latency_file = tmp_path / "request_latencies_run1.jsonl"
    routing_file = tmp_path / "route_decisions.jsonl"
    _write_jsonl(latency_file, [
        {
            "request_id": "req-1", "success": True, "ttft": 1.0,
            "total_latency": 2.0, "adapter_name": "adapter-a",
        },
        {
            "request_id": "req-2", "success": True, "ttft": 9.0,
            "total_latency": 12.0, "adapter_name": "adapter-b",
        },
    ])
    _write_jsonl(routing_file, [
        {
            "request_id": "req-1", "timestamp": 10.0, "selected_worker": 0,
            "workers": {
                "0": _worker_state(selected=True, cached=True),
                "1": _worker_state(),
            },
        },
        {
            "request_id": "req-2", "timestamp": 11.0, "selected_worker": 1,
            "workers": {
                "0": _worker_state(load=0.0, queue=0),
                "1": _worker_state(
                    selected=True, load=0.0, queue=3, decode=3, prefill=0, cached=True,
                ),
            },
        },
    ])

    summary = join_request_diagnostics(latency_file, routing_file)

    assert summary["matched_requests"] == 2
    assert summary["missing_route_count"] == 0
    assert summary["routing"]["cache_hits"] == 2
    assert summary["routing"]["selected_active_decode_with_zero_prefill"] == 1
    assert summary["routing"]["selected_higher_queue_at_same_min_pressure"] == 1
    assert summary["slow_ttft"]["worker_counts"] == {"1": 1}
    assert Path(summary["joined_file"]).exists()
    assert Path(summary["summary_file"]).exists()


def test_join_request_diagnostics_reports_missing_routes(tmp_path):
    latency_file = tmp_path / "request_latencies_run1.jsonl"
    routing_file = tmp_path / "route_decisions.jsonl"
    _write_jsonl(latency_file, [
        {
            "request_id": "req-1", "success": True, "ttft": 1.0,
            "total_latency": 2.0, "adapter_name": "adapter-a",
        },
    ])
    _write_jsonl(routing_file, [])

    summary = join_request_diagnostics(latency_file, routing_file)

    assert summary["matched_requests"] == 0
    assert summary["missing_route_count"] == 1
    assert summary["missing_route_request_ids"] == ["req-1"]


def test_join_request_diagnostics_rejects_duplicate_ids(tmp_path):
    latency_file = tmp_path / "request_latencies_run1.jsonl"
    routing_file = tmp_path / "route_decisions.jsonl"
    duplicate = {
        "request_id": "req-1", "success": True, "ttft": 1.0,
        "total_latency": 2.0, "adapter_name": "adapter-a",
    }
    _write_jsonl(latency_file, [duplicate, duplicate])
    _write_jsonl(routing_file, [])

    with pytest.raises(ValueError, match="duplicate request_id req-1"):
        join_request_diagnostics(latency_file, routing_file)


def test_join_request_diagnostics_filters_reused_server_runs(tmp_path):
    latency_file = tmp_path / "request_latencies_run2.jsonl"
    routing_file = tmp_path / "route_decisions.jsonl"
    _write_jsonl(latency_file, [
        {
            "request_id": "bench-run2-req-1", "success": True, "ttft": 1.0,
            "total_latency": 2.0, "adapter_name": "adapter-a",
        },
    ])
    _write_jsonl(routing_file, [
        {
            "request_id": "bench-run1-req-1", "timestamp": 9.0,
            "selected_worker": 0,
            "workers": {"0": _worker_state(selected=True)},
        },
        {
            "request_id": "bench-run2-req-1", "timestamp": 10.0,
            "selected_worker": 0,
            "workers": {"0": _worker_state(selected=True)},
        },
    ])

    summary = join_request_diagnostics(latency_file, routing_file)

    assert summary["routing_requests"] == 1
    assert summary["matched_requests"] == 1
    assert summary["missing_latency_count"] == 0
