#!/usr/bin/env python3
"""
Build a fused real-workload trace.

Azure LLM provides request time and token lengths. Azure Functions provides the
HTTP top-100 adapter popularity distribution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

try:
    from azure_trace import (
        assign_weighted_adapters,
        load_azure_llm_window,
        sample_requests_per_second,
        summarize_requests,
        write_jsonl,
    )
except ImportError:
    from real_workload.azure_trace import (
        assign_weighted_adapters,
        load_azure_llm_window,
        sample_requests_per_second,
        summarize_requests,
        write_jsonl,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse Azure LLM request shapes with Azure Functions adapter popularity."
    )
    parser.add_argument("--llm-trace", required=True, type=Path, help="Azure LLM CSV path")
    parser.add_argument(
        "--functions-summary",
        required=True,
        type=Path,
        help="Summary JSON from inspect_azure_functions_trace.py",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument(
        "--start-offset-sec",
        type=float,
        default=0.0,
        help="Seconds after the first LLM CSV timestamp where the window starts",
    )
    parser.add_argument("--duration", required=True, type=float, help="Window duration in seconds")
    parser.add_argument("--target-rate", required=True, type=float, help="Target request rate")
    parser.add_argument("--min-context-tokens", type=int, default=1)
    parser.add_argument("--min-generated-tokens", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--max-generated-tokens", type=int, default=None)
    parser.add_argument(
        "--swap-adjacent-adapter-ids",
        action="store_true",
        help="Swap adapter ids in adjacent pairs (0<->1, 2<->3, ...).",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path for writing the summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_azure_llm_window(
        csv_path=args.llm_trace,
        start_offset_sec=args.start_offset_sec,
        duration_sec=args.duration,
        min_context_tokens=args.min_context_tokens,
        min_generated_tokens=args.min_generated_tokens,
        max_context_tokens=args.max_context_tokens,
        max_generated_tokens=args.max_generated_tokens,
    )
    sampled_rows = sample_requests_per_second(
        rows,
        target_rate=args.target_rate,
        duration_sec=args.duration,
    )

    adapter_counts = load_selected_adapter_counts(args.functions_summary)
    requests = assign_weighted_adapters(sampled_rows, adapter_counts=adapter_counts)
    if args.swap_adjacent_adapter_ids:
        requests = swap_adjacent_adapter_ids(requests, adapter_count=len(adapter_counts))
    write_jsonl(requests, args.output)

    summary = summarize_requests(requests, duration_sec=args.duration)
    summary.update(
        {
            "llm_trace": str(args.llm_trace),
            "functions_summary": str(args.functions_summary),
            "output": str(args.output),
            "start_offset_sec": args.start_offset_sec,
            "target_rate": args.target_rate,
            "source_rows_in_window": len(rows),
            "adapter_source": "azure_functions_http_top100_truncated",
            "adapter_count": len(adapter_counts),
            "adapter_id_mapping": (
                "adjacent_pair_swap" if args.swap_adjacent_adapter_ids else "identity"
            ),
        }
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def load_selected_adapter_counts(summary_path: Path) -> list[int]:
    with Path(summary_path).open("r") as handle:
        summary = json.load(handle)

    adapter_popularity = summary.get("adapter_popularity", {})
    selected_mapping = adapter_popularity.get("selected_mapping")
    if selected_mapping != "top_n_truncated":
        raise ValueError(f"Expected selected_mapping=top_n_truncated, got {selected_mapping}")

    selected_counts = adapter_popularity.get("selected_counts")
    if not selected_counts:
        raise ValueError("functions summary has no adapter_popularity.selected_counts")

    sorted_counts = sorted(selected_counts, key=lambda item: int(item["adapter_id"]))
    return [int(item["invocations"]) for item in sorted_counts]


def swap_adjacent_adapter_ids(requests, adapter_count: int):
    if adapter_count <= 0 or adapter_count % 2 != 0:
        raise ValueError(
            f"adapter_count must be a positive even number, got {adapter_count}"
        )

    remapped = []
    for request in requests:
        if request.adapter_id < 0 or request.adapter_id >= adapter_count:
            raise ValueError(
                f"adapter_id {request.adapter_id} is out of range for {adapter_count} adapters"
            )
        remapped.append(replace(request, adapter_id=request.adapter_id ^ 1))
    return remapped


if __name__ == "__main__":
    main()
