"""
Tests for Azure LLM trace preprocessing helpers.
"""

import csv
import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from real_workload.azure_trace import (
    assign_zipf_adapters,
    downsample_requests,
    load_azure_llm_window,
    summarize_requests,
    write_jsonl,
)


def write_azure_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
        )
        writer.writeheader()
        writer.writerows(rows)


class TestAzureTracePreprocessing:
    def test_load_window_uses_relative_time_and_token_filters(self, tmp_path):
        csv_path = tmp_path / "trace.csv"
        write_azure_csv(
            csv_path,
            [
                {
                    "TIMESTAMP": "2024-05-12 00:00:00.000000+00:00",
                    "ContextTokens": "10",
                    "GeneratedTokens": "5",
                },
                {
                    "TIMESTAMP": "2024-05-12 00:00:01.000000+00:00",
                    "ContextTokens": "20",
                    "GeneratedTokens": "1",
                },
                {
                    "TIMESTAMP": "2024-05-12 00:00:02.500000+00:00",
                    "ContextTokens": "30",
                    "GeneratedTokens": "7",
                },
                {
                    "TIMESTAMP": "2024-05-12 00:00:05.000000+00:00",
                    "ContextTokens": "40",
                    "GeneratedTokens": "8",
                },
            ],
        )

        rows = load_azure_llm_window(
            csv_path,
            start_offset_sec=1.0,
            duration_sec=3.0,
            min_generated_tokens=2,
        )

        assert rows == [
            {
                "req_time": 1.5,
                "input_len": 30,
                "output_len": 7,
                "source_timestamp": "2024-05-12 00:00:02.500000+00:00",
            }
        ]

    def test_downsample_preserves_time_order(self):
        rows = [{"req_time": float(i), "input_len": i + 1, "output_len": 1} for i in range(10)]

        sampled = downsample_requests(rows, target_count=4, seed=1)

        assert len(sampled) == 4
        assert [row["req_time"] for row in sampled] == sorted(row["req_time"] for row in sampled)

    def test_assign_zipf_adapters_is_stable_and_bounded(self):
        rows = [
            {
                "req_time": 0.1 * i,
                "input_len": 128 + i,
                "output_len": 64 + i,
                "source_timestamp": f"ts-{i}",
            }
            for i in range(20)
        ]

        first = assign_zipf_adapters(rows, num_adapters=5, adapter_skew=0.8, seed=42)
        second = assign_zipf_adapters(rows, num_adapters=5, adapter_skew=0.8, seed=42)

        assert [request.adapter_id for request in first] == [
            request.adapter_id for request in second
        ]
        assert all(0 <= request.adapter_id < 5 for request in first)
        assert first[0].req_id == 0
        assert first[-1].req_id == 19

    def test_write_jsonl_and_summary(self, tmp_path):
        rows = [
            {
                "req_time": 0.0,
                "input_len": 100,
                "output_len": 20,
                "source_timestamp": "ts-0",
            },
            {
                "req_time": 0.5,
                "input_len": 200,
                "output_len": 40,
                "source_timestamp": "ts-1",
            },
        ]
        requests = assign_zipf_adapters(rows, num_adapters=2, adapter_skew=1.0, seed=7)
        output_path = tmp_path / "trace.jsonl"

        write_jsonl(requests, output_path)
        loaded = [json.loads(line) for line in output_path.read_text().splitlines()]
        summary = summarize_requests(requests, duration_sec=1.0)

        assert len(loaded) == 2
        assert loaded[0]["input_len"] == 100
        assert summary["count"] == 2
        assert summary["achieved_rate"] == pytest.approx(2.0)
        assert summary["input_len"]["mean"] == pytest.approx(150.0)
