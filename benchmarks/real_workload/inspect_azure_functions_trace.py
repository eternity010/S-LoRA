#!/usr/bin/env python3
"""
Inspect Azure Functions 2019 invocation traces for adapter popularity.

By default this extracts the chosen workload distribution:
HTTP-only functions mapped to 100 adapters by top-100 truncation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract adapter popularity from an Azure Functions invocation CSV."
    )
    parser.add_argument("input", type=Path, help="Path to invocations_per_function CSV")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top functions to print")
    parser.add_argument(
        "--num-adapters",
        type=int,
        default=100,
        help="Number of adapter popularity entries to derive from function totals",
    )
    parser.add_argument(
        "--trigger",
        default="http",
        help="Trigger value to keep. Defaults to http for online request-like workloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError(f"top-k must be positive, got {args.top_k}")
    if args.num_adapters <= 0:
        raise ValueError(f"num-adapters must be positive, got {args.num_adapters}")

    summary = inspect_invocation_csv(
        args.input,
        top_k=args.top_k,
        num_adapters=args.num_adapters,
        trigger_filter=args.trigger,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def inspect_invocation_csv(
    path: Path,
    top_k: int,
    num_adapters: int,
    trigger_filter: str | None = None,
) -> dict:
    path = Path(path)
    trigger_counts: Counter[str] = Counter()
    kept_trigger_counts: Counter[str] = Counter()
    owner_ids = set()
    app_ids = set()
    function_ids = set()
    function_totals = []
    top_functions = []
    minute_totals = None
    row_count = 0
    kept_row_count = 0

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        metadata_columns = header[:4]
        minute_columns = header[4:]

        if metadata_columns != ["HashOwner", "HashApp", "HashFunction", "Trigger"]:
            raise ValueError(f"Unexpected metadata columns: {metadata_columns}")
        if len(minute_columns) != 1440:
            raise ValueError(f"Expected 1440 minute columns, got {len(minute_columns)}")

        minute_totals = np.zeros(len(minute_columns), dtype=np.int64)

        for row in reader:
            row_count += 1
            owner, app, function, trigger = row[:4]
            trigger_counts[trigger] += 1
            if trigger_filter is not None and trigger != trigger_filter:
                continue

            kept_row_count += 1
            kept_trigger_counts[trigger] += 1
            counts = np.fromiter((int(value) for value in row[4:]), dtype=np.int64)
            total = int(counts.sum())

            owner_ids.add(owner)
            app_ids.add(app)
            function_ids.add(function)
            function_totals.append(total)
            minute_totals += counts

            if total > 0:
                top_functions.append(
                    {
                        "total_invocations": total,
                        "trigger": trigger,
                        "hash_owner": owner,
                        "hash_app": app,
                        "hash_function": function,
                    }
                )

    totals = np.array(function_totals, dtype=np.int64)
    positive_totals = totals[totals > 0]
    top_functions.sort(key=lambda item: item["total_invocations"], reverse=True)

    return {
        "path": str(path),
        "rows": row_count,
        "kept_rows": kept_row_count,
        "trigger_filter": trigger_filter,
        "metadata_columns": metadata_columns,
        "minute_columns": {
            "count": len(minute_columns),
            "first": minute_columns[0],
            "last": minute_columns[-1],
        },
        "unique": {
            "owners": len(owner_ids),
            "apps": len(app_ids),
            "functions": len(function_ids),
            "triggers": len(trigger_counts),
        },
        "trigger_counts": dict(trigger_counts.most_common()),
        "kept_trigger_counts": dict(kept_trigger_counts.most_common()),
        "function_invocations": summarize_values(positive_totals),
        "zero_invocation_functions": int((totals == 0).sum()),
        "total_invocations": int(totals.sum()),
        "top_share": {
            "top1": top_share(top_functions, 1),
            "top5": top_share(top_functions, 5),
            "top10": top_share(top_functions, 10),
            "top100": top_share(top_functions, 100),
        },
        "adapter_popularity": summarize_adapter_popularity(top_functions, num_adapters),
        "minute_invocations": summarize_values(minute_totals),
        "top_functions": top_functions[:top_k],
    }


def summarize_values(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "min": int(values.min()),
        "max": int(values.max()),
    }


def top_share(top_functions: list[dict], top_k: int) -> float:
    total = sum(item["total_invocations"] for item in top_functions)
    if total == 0:
        return 0.0
    return sum(item["total_invocations"] for item in top_functions[:top_k]) / total


def summarize_adapter_popularity(top_functions: list[dict], num_adapters: int) -> dict:
    totals = [item["total_invocations"] for item in top_functions]
    if not totals:
        return {}

    truncated = totals[:num_adapters]
    tail_total = sum(totals[num_adapters - 1 :]) if num_adapters <= len(totals) else 0
    top_plus_tail = totals[: max(num_adapters - 1, 0)] + ([tail_total] if tail_total else [])

    top_n_truncated = summarize_popularity_vector(truncated)
    top_n_minus_1_plus_tail = summarize_popularity_vector(top_plus_tail)

    return {
        "num_adapters": num_adapters,
        "selected_mapping": "top_n_truncated",
        "selected_counts": [
            {
                "adapter_id": index,
                "function_rank": index + 1,
                "hash_function": item["hash_function"],
                "trigger": item["trigger"],
                "invocations": item["total_invocations"],
            }
            for index, item in enumerate(top_functions[:num_adapters])
        ],
        "top_n_truncated": top_n_truncated,
        "top_n_minus_1_plus_tail": top_n_minus_1_plus_tail,
    }


def summarize_popularity_vector(counts: list[int]) -> dict:
    total = sum(counts)
    if total == 0:
        return {}

    shares = [count / total for count in counts]
    top_entries = [
        {"adapter_id": index, "invocations": count, "share": share}
        for index, (count, share) in enumerate(zip(counts, shares))
    ]
    sorted_shares = sorted(shares, reverse=True)

    return {
        "total_invocations": total,
        "top1_share": sum(sorted_shares[:1]),
        "top5_share": sum(sorted_shares[:5]),
        "top10_share": sum(sorted_shares[:10]),
        "top20_share": sum(sorted_shares[:20]),
        "entries": top_entries[:10],
    }


if __name__ == "__main__":
    main()
