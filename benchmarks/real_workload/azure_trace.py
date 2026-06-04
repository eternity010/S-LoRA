"""
Utilities for converting Azure LLM inference traces into compact benchmark traces.

The Azure CSV files are large, so this module reads only the requested time
window and avoids loading the full trace into memory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class AzureTraceRequest:
    """One request in the compact trace consumed by benchmark preprocessing."""

    req_id: int
    req_time: float
    adapter_id: int
    input_len: int
    output_len: int
    source_timestamp: str


def parse_azure_timestamp(value: str) -> datetime:
    """Parse Azure trace timestamps such as '2024-05-12 00:00:00.001163+00:00'."""

    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def load_azure_llm_window(
    csv_path: Path,
    start_offset_sec: float,
    duration_sec: float,
    min_context_tokens: int = 1,
    min_generated_tokens: int = 1,
    max_context_tokens: Optional[int] = None,
    max_generated_tokens: Optional[int] = None,
) -> List[dict]:
    """
    Stream a sorted Azure LLM CSV and return rows in the requested time window.

    Returned rows contain relative request time, context/generated token counts,
    and the original timestamp. Rows outside optional token bounds are skipped.
    """

    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")
    if start_offset_sec < 0:
        raise ValueError(f"start_offset_sec must be non-negative, got {start_offset_sec}")

    csv_path = Path(csv_path)
    rows: List[dict] = []

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} must contain columns: {', '.join(sorted(required))}"
            )

        first_row = next(reader, None)
        if first_row is None:
            return rows

        trace_start = parse_azure_timestamp(first_row["TIMESTAMP"])
        window_start = trace_start + timedelta(seconds=start_offset_sec)
        window_end = window_start + timedelta(seconds=duration_sec)

        for row in _iter_with_first(first_row, reader):
            timestamp = parse_azure_timestamp(row["TIMESTAMP"])
            if timestamp < window_start:
                continue
            if timestamp >= window_end:
                break

            context_tokens = int(row["ContextTokens"])
            generated_tokens = int(row["GeneratedTokens"])
            if context_tokens < min_context_tokens or generated_tokens < min_generated_tokens:
                continue
            if max_context_tokens is not None and context_tokens > max_context_tokens:
                continue
            if max_generated_tokens is not None and generated_tokens > max_generated_tokens:
                continue

            rows.append(
                {
                    "req_time": (timestamp - window_start).total_seconds(),
                    "input_len": context_tokens,
                    "output_len": generated_tokens,
                    "source_timestamp": row["TIMESTAMP"],
                }
            )

    return rows


def downsample_requests(
    rows: List[dict],
    target_count: Optional[int],
    seed: int,
) -> List[dict]:
    """Downsample rows without replacement while preserving timestamp order."""

    if target_count is None or target_count >= len(rows):
        return list(rows)
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(rows), size=target_count, replace=False))
    return [rows[int(index)] for index in indices]


def assign_zipf_adapters(
    rows: List[dict],
    num_adapters: int,
    adapter_skew: float,
    seed: int,
) -> List[AzureTraceRequest]:
    """Assign synthetic adapter ids using a Zipf popularity distribution."""

    if num_adapters <= 0:
        raise ValueError(f"num_adapters must be positive, got {num_adapters}")
    if adapter_skew < 0:
        raise ValueError(f"adapter_skew must be non-negative, got {adapter_skew}")

    ranks = np.arange(1, num_adapters + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, adapter_skew)
    probabilities = weights / weights.sum()

    rng = np.random.default_rng(seed)
    adapter_ids = rng.choice(num_adapters, size=len(rows), p=probabilities)

    requests = []
    for req_id, (row, adapter_id) in enumerate(zip(rows, adapter_ids)):
        requests.append(
            AzureTraceRequest(
                req_id=req_id,
                req_time=float(row["req_time"]),
                adapter_id=int(adapter_id),
                input_len=int(row["input_len"]),
                output_len=int(row["output_len"]),
                source_timestamp=str(row["source_timestamp"]),
            )
        )
    return requests


def write_jsonl(requests: Iterable[AzureTraceRequest], output_path: Path) -> None:
    """Write compact trace requests as JSONL."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for request in requests:
            handle.write(json.dumps(asdict(request), sort_keys=True) + "\n")


def summarize_requests(requests: List[AzureTraceRequest], duration_sec: float) -> dict:
    """Return a small summary dictionary for sanity-checking generated traces."""

    if not requests:
        return {
            "count": 0,
            "duration_sec": duration_sec,
            "achieved_rate": 0.0,
        }

    input_lens = np.array([request.input_len for request in requests])
    output_lens = np.array([request.output_len for request in requests])
    adapter_ids = np.array([request.adapter_id for request in requests])
    counts = np.bincount(adapter_ids)
    top_counts = sorted(
        ((adapter_id, int(count)) for adapter_id, count in enumerate(counts) if count),
        key=lambda item: item[1],
        reverse=True,
    )

    def percentile(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values, q))

    return {
        "count": len(requests),
        "duration_sec": duration_sec,
        "achieved_rate": len(requests) / duration_sec,
        "input_len": {
            "mean": float(input_lens.mean()),
            "p50": percentile(input_lens, 50),
            "p90": percentile(input_lens, 90),
            "min": int(input_lens.min()),
            "max": int(input_lens.max()),
        },
        "output_len": {
            "mean": float(output_lens.mean()),
            "p50": percentile(output_lens, 50),
            "p90": percentile(output_lens, 90),
            "min": int(output_lens.min()),
            "max": int(output_lens.max()),
        },
        "adapter": {
            "unique": int(np.count_nonzero(counts)),
            "top10": top_counts[:10],
            "top1_share": top_share(top_counts, len(requests), 1),
            "top5_share": top_share(top_counts, len(requests), 5),
            "top10_share": top_share(top_counts, len(requests), 10),
        },
    }


def top_share(top_counts: List[tuple], total: int, top_k: int) -> float:
    """Return request share covered by the top-k adapters."""

    if total == 0:
        return 0.0
    return sum(count for _, count in top_counts[:top_k]) / total


def _iter_with_first(first_row: dict, reader: Iterable[dict]) -> Iterable[dict]:
    yield first_row
    yield from reader
