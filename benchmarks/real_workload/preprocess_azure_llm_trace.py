#!/usr/bin/env python3
"""
Preprocess Azure LLM inference traces into compact S-LoRA benchmark traces.

Example:
    python benchmarks/real_workload/preprocess_azure_llm_trace.py \
      --input /home/hzheng/datasets/azure_public/llm2024/AzureLLMInferenceTrace_conv_1week.csv \
      --output benchmarks/real_workload/outputs/conv_8rps_180s.jsonl \
      --start-offset-sec 0 \
      --duration 180 \
      --target-rate 8 \
      --num-adapters 100 \
      --adapter-skew 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from azure_trace import (
    assign_zipf_adapters,
    load_azure_llm_window,
    sample_requests_per_second,
    summarize_requests,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Azure LLM trace window into compact JSONL requests."
    )
    parser.add_argument("--input", required=True, type=Path, help="Azure LLM CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument(
        "--start-offset-sec",
        type=float,
        default=0.0,
        help="Seconds after the first CSV timestamp where the window starts",
    )
    parser.add_argument("--duration", required=True, type=float, help="Window duration in seconds")
    parser.add_argument(
        "--target-rate",
        type=float,
        default=None,
        help="Optional target request rate. If set, downsample to rate * duration requests.",
    )
    parser.add_argument("--num-adapters", type=int, default=100)
    parser.add_argument(
        "--adapter-skew",
        type=float,
        default=0.8,
        help="Zipf skew for synthetic adapter assignment. Larger means more hotspot-heavy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-context-tokens", type=int, default=1)
    parser.add_argument("--min-generated-tokens", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--max-generated-tokens", type=int, default=None)
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
        csv_path=args.input,
        start_offset_sec=args.start_offset_sec,
        duration_sec=args.duration,
        min_context_tokens=args.min_context_tokens,
        min_generated_tokens=args.min_generated_tokens,
        max_context_tokens=args.max_context_tokens,
        max_generated_tokens=args.max_generated_tokens,
    )

    sampled_rows = rows
    if args.target_rate is not None:
        sampled_rows = sample_requests_per_second(
            rows,
            target_rate=args.target_rate,
            duration_sec=args.duration,
        )

    requests = assign_zipf_adapters(
        sampled_rows,
        num_adapters=args.num_adapters,
        adapter_skew=args.adapter_skew,
        seed=args.seed,
    )
    write_jsonl(requests, args.output)

    summary = summarize_requests(requests, duration_sec=args.duration)
    summary.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "start_offset_sec": args.start_offset_sec,
            "target_rate": args.target_rate,
            "num_adapters": args.num_adapters,
            "adapter_skew": args.adapter_skew,
            "seed": args.seed,
            "source_rows_in_window": len(rows),
        }
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
