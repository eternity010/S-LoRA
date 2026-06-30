"""
Tests for benchmark trace request loading.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trace import load_jsonl_trace_requests


def test_load_jsonl_trace_requests_maps_fields(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "req_id": 7,
                "req_time": 1.25,
                "adapter_id": 1,
                "input_len": 3,
                "output_len": 9,
            }
        )
        + "\n"
    )

    requests = load_jsonl_trace_requests(
        trace_file=trace_path,
        base_model="/models/base",
        adapter_dirs=[
            ("/models/base", "/adapters/a"),
            ("/models/base", "/adapters/b"),
        ],
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.req_id == 7
    assert request.model_dir == "/models/base"
    assert request.adapter_dir == "/adapters/b"
    assert request.prompt_len == 3
    assert request.output_len == 9
    assert request.req_time == pytest.approx(1.25)
    assert request.prompt == "Hello " * 3


def test_load_jsonl_trace_requests_rejects_adapter_id_out_of_range(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "req_id": 0,
                "req_time": 0.0,
                "adapter_id": 2,
                "input_len": 4,
                "output_len": 8,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="adapter_id"):
        load_jsonl_trace_requests(
            trace_file=trace_path,
            base_model="/models/base",
            adapter_dirs=[("/models/base", "/adapters/a")],
        )
