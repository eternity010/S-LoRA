"""
Tests for fused real-workload trace builder helpers.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from real_workload.build_azure_llm_functions_workload import load_selected_adapter_counts


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
