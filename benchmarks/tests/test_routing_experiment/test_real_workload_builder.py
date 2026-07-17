"""
Tests for fused real-workload trace builder helpers.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from real_workload.azure_trace import AzureTraceRequest
from real_workload.build_azure_llm_functions_workload import (
    load_selected_adapter_counts,
    swap_adjacent_adapter_ids,
)


def test_load_selected_adapter_counts_reads_top_truncated_mapping(tmp_path):
    summary_path = tmp_path / "functions_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "adapter_popularity": {
                    "selected_mapping": "top_n_truncated",
                    "selected_counts": [
                        {"adapter_id": 1, "invocations": 30},
                        {"adapter_id": 0, "invocations": 70},
                    ],
                }
            }
        )
    )

    assert load_selected_adapter_counts(summary_path) == [70, 30]


def test_swap_adjacent_adapter_ids_preserves_request_shape():
    requests = [
        AzureTraceRequest(
            req_id=index,
            req_time=float(index),
            adapter_id=index,
            input_len=100 + index,
            output_len=20 + index,
            source_timestamp=f"ts-{index}",
        )
        for index in range(4)
    ]

    remapped = swap_adjacent_adapter_ids(requests, adapter_count=4)

    assert [request.adapter_id for request in remapped] == [1, 0, 3, 2]
    assert [request.req_id for request in remapped] == [0, 1, 2, 3]
    assert [request.input_len for request in remapped] == [100, 101, 102, 103]
    assert [request.output_len for request in remapped] == [20, 21, 22, 23]


def test_swap_adjacent_adapter_ids_requires_even_adapter_count():
    with pytest.raises(ValueError, match="positive even"):
        swap_adjacent_adapter_ids([], adapter_count=3)
