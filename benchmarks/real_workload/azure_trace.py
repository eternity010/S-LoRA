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


def sample_requests_per_second(
    rows: List[dict],
    target_rate: float,
    duration_sec: float,
) -> List[dict]:
    """
    Deterministically sample a fixed number of requests from each one-second bucket.

    This keeps the benchmark request rate stable while preserving the trace order and
    representative token lengths inside each second.
    """

    if target_rate <= 0:
        raise ValueError(f"target_rate must be positive, got {target_rate}")
    target_per_second = int(target_rate)
    if not np.isclose(target_rate, target_per_second):
        raise ValueError(f"target_rate must be an integer for per-second sampling, got {target_rate}")

    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")
    duration_seconds = int(duration_sec)
    if not np.isclose(duration_sec, duration_seconds):
        raise ValueError(
            f"duration_sec must be an integer for per-second sampling, got {duration_sec}"
        )

    target_count = target_per_second * duration_seconds
    if target_count >= len(rows):
        return list(rows)

    buckets: List[List[int]] = [[] for _ in range(duration_seconds)]
    for index, row in enumerate(rows):
        second = int(float(row["req_time"]))
        if 0 <= second < duration_seconds:
            buckets[second].append(index)

    selected_indices: List[int] = []
    for bucket in buckets:
        count = min(target_per_second, len(bucket))
        selected_indices.extend(_evenly_spaced_indices(bucket, count))

    if len(selected_indices) < target_count:
        selected_set = set(selected_indices)
        remaining = [index for index in range(len(rows)) if index not in selected_set]
        needed = target_count - len(selected_indices)
        selected_indices.extend(_evenly_spaced_indices(remaining, needed))

    return [rows[index] for index in sorted(selected_indices)]


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


def assign_weighted_adapters(
    rows: List[dict],
    adapter_counts: List[int],
) -> List[AzureTraceRequest]:
    """Assign adapter ids deterministically according to invocation counts."""

    if not adapter_counts:
        raise ValueError("adapter_counts must not be empty")
    if any(count < 0 for count in adapter_counts):
        raise ValueError("adapter_counts must be non-negative")
    if sum(adapter_counts) <= 0:
        raise ValueError("adapter_counts must contain at least one positive count")

    adapter_ids = build_weighted_adapter_sequence(
        request_count=len(rows),
        adapter_counts=adapter_counts,
    )

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


def build_weighted_adapter_sequence(
    request_count: int,
    adapter_counts: List[int],
) -> List[int]:
    """Build a deterministic time-spread adapter sequence from popularity counts."""

    if request_count < 0:
        raise ValueError(f"request_count must be non-negative, got {request_count}")
    if request_count == 0:
        return []

    target_counts = allocate_counts_by_weight(request_count, adapter_counts)
    sequence: List[int | None] = [None] * request_count

    for adapter_id, count in enumerate(target_counts):
        if count == 0:
            continue
        for position in _evenly_spaced_indices(list(range(request_count)), count):
            while sequence[position] is not None:
                position = (position + 1) % request_count
            sequence[position] = adapter_id

    return [adapter_id if adapter_id is not None else 0 for adapter_id in sequence]


def allocate_counts_by_weight(total_count: int, weights: List[int]) -> List[int]:
    """Allocate integer counts using largest remainders."""

    if total_count < 0:
        raise ValueError(f"total_count must be non-negative, got {total_count}")
    if not weights:
        raise ValueError("weights must not be empty")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("weights must contain at least one positive value")

    raw_counts = np.array(weights, dtype=np.float64) * total_count / weight_sum
    floor_counts = np.floor(raw_counts).astype(np.int64)
    remainder = total_count - int(floor_counts.sum())

    if remainder > 0:
        fractional = raw_counts - floor_counts
        order = np.argsort(-fractional, kind="stable")
        for index in order[:remainder]:
            floor_counts[index] += 1

    return [int(count) for count in floor_counts]


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


def _evenly_spaced_indices(indices: List[int], count: int) -> List[int]:
    if count <= 0:
        return []
    if count >= len(indices):
        return list(indices)

    size = len(indices)
    return [indices[min(int((offset + 0.5) * size / count), size - 1)] for offset in range(count)]
