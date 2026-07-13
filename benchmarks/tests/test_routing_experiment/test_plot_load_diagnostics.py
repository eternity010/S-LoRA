import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "benchmarks"))

from routing_experiment.plot_load_diagnostics import (
    analyze_load_diagnostics,
    plot_load_diagnostics,
)


def _state(raw):
    return {"pending_raw_tokens": raw}


def _routing_records():
    return [
        {
            "timestamp": 100.0,
            "request_id": "req-0",
            "selected_worker": 0,
            "prompt_len": 100,
            "workers": {"0": _state(100), "1": _state(0)},
        },
        {
            "timestamp": 105.0,
            "request_id": "req-1",
            "selected_worker": 1,
            "prompt_len": 200,
            "workers": {"0": _state(50), "1": _state(200)},
        },
        {
            "timestamp": 112.0,
            "request_id": "req-2",
            "selected_worker": 1,
            "prompt_len": 300,
            "workers": {"0": _state(0), "1": _state(300)},
        },
    ]


def _joined_records():
    return [
        {"route_timestamp": 100.0, "selected_worker": 0, "success": True, "ttft": 1.0},
        {"route_timestamp": 105.0, "selected_worker": 1, "success": True, "ttft": 3.0},
        {"route_timestamp": 112.0, "selected_worker": 1, "success": True, "ttft": 5.0},
    ]


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_analyze_load_diagnostics_builds_worker_and_window_metrics():
    summary = analyze_load_diagnostics(_routing_records(), _joined_records(), window_sec=10)

    assert summary["workers"]["0"]["request_count"] == 1
    assert summary["workers"]["1"]["request_count"] == 2
    assert summary["workers"]["1"]["prompt_tokens"] == 500
    assert summary["workers"]["1"]["avg_ttft"] == pytest.approx(4.0)
    assert summary["windows"][0]["request_counts"] == {"0": 1, "1": 1}
    assert summary["windows"][0]["avg_ttft"] == {"0": 1.0, "1": 3.0}
    assert summary["windows"][1]["request_counts"] == {"0": 0, "1": 1}


def test_plot_load_diagnostics_writes_plot_and_summary(tmp_path):
    _write_jsonl(tmp_path / "route_decisions.jsonl", _routing_records())
    _write_jsonl(tmp_path / "joined_requests_run.jsonl", _joined_records())

    outputs = plot_load_diagnostics(tmp_path, window_sec=10)

    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())
    summary = json.loads(outputs["summary"].read_text())
    assert summary["routing_requests"] == 3


def test_plot_load_diagnostics_resolves_single_nested_run(tmp_path):
    run_dir = tmp_path / "server_run"
    run_dir.mkdir()
    _write_jsonl(run_dir / "route_decisions.jsonl", _routing_records())
    _write_jsonl(run_dir / "joined_requests_run.jsonl", _joined_records())

    outputs = plot_load_diagnostics(tmp_path, output_dir=tmp_path / "charts")

    assert outputs["png"].exists()


def test_analyze_load_diagnostics_rejects_invalid_input():
    with pytest.raises(ValueError, match="empty"):
        analyze_load_diagnostics([], [], window_sec=10)
    with pytest.raises(ValueError, match="positive"):
        analyze_load_diagnostics(_routing_records(), [], window_sec=0)


def test_analyze_load_diagnostics_reports_state_freshness():
    records = _routing_records()
    records[0]["workers"]["0"].update({
        "report_age_ms": 350.0,
        "report_seq": 2,
        "optimistic_request_count": 1,
    })
    records[1]["workers"]["0"].update({
        "report_age_ms": 50.0,
        "report_seq": 1,
        "optimistic_request_count": 0,
    })

    summary = analyze_load_diagnostics(records, _joined_records(), window_sec=10)
    worker = summary["workers"]["0"]

    assert worker["stale_report_samples_over_300ms"] == 1
    assert worker["optimistic_state_samples"] == 1
    assert worker["report_seq_regressions"] == 1
