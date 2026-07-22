import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from external_baselines.slora.run_replicated_realtrace import (
    RUNTIME_SHIM_DIR,
    adapter_dirs,
    build_server_command,
    load_trace,
    summarize,
)


def test_load_trace_requires_actual_total_within_model_limit(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({
            "req_id": 0,
            "req_time": 0.0,
            "adapter_id": 1,
            "input_len": 1848,
            "output_len": 200,
        }) + "\n"
    )

    with pytest.raises(ValueError, match="total tokens exceed"):
        load_trace(trace, num_adapters=2, max_model_len=2048)


def test_adapter_dirs_match_rlora_logical_mapping():
    assert adapter_dirs(4, "/rank16", "/rank64") == [
        "/rank16-0",
        "/rank64-0",
        "/rank16-1",
        "/rank64-1",
    ]


def test_server_command_matches_common_capacity_and_registers_adapters():
    args = SimpleNamespace(
        slora_python="/env/bin/python",
        model="/model",
        max_total_tokens=15000,
        batch_max_tokens=3072,
        max_model_len=2048,
        num_adapters=100,
        rank16_adapter="/rank16",
        rank64_adapter="/rank64",
    )

    command = build_server_command(args, port=38200, nccl_port=39200)

    assert command[command.index("--port") + 1] == "38200"
    assert command[command.index("--nccl_port") + 1] == "39200"
    assert command[command.index("--max_total_token_num") + 1] == "15000"
    assert command[command.index("--batch_max_tokens") + 1] == "3072"
    assert command[command.index("--max_req_total_len") + 1] == "2048"
    assert "--swap" in command
    assert command.count("--lora-dirs") == 100


def test_runtime_shim_is_available_for_spawn_children():
    shim = RUNTIME_SHIM_DIR / "sitecustomize.py"

    assert shim.is_file()
    source = shim.read_text()
    assert "SLORA_BASELINE_NCCL_PORT" in source
    assert "SLORA_BASELINE_MAX_REQ_TOTAL_LEN" in source


def test_summarize_uses_successful_requests_only():
    results = [
        {"success": True, "total_latency": 2.0, "ttft": 0.5},
        {"success": True, "total_latency": 4.0, "ttft": 1.5},
        {"success": False, "total_latency": None, "ttft": None},
    ]

    summary = summarize(results, benchmark_time=5.0, req_rate=1.0)

    assert summary["completed_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["throughput"] == pytest.approx(0.4)
    assert summary["avg_latency"] == pytest.approx(3.0)
    assert summary["avg_first_token_latency"] == pytest.approx(1.0)
