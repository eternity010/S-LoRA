"""Join benchmark request timings with router decision records."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _index_unique(records: Iterable[Dict[str, Any]], source: Path) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for record in records:
        request_id = record.get("request_id")
        if not request_id:
            raise ValueError(f"{source}: record is missing request_id")
        if request_id in indexed:
            raise ValueError(f"{source}: duplicate request_id {request_id}")
        indexed[request_id] = record
    return indexed


def _benchmark_run_prefix(request_id: str):
    if "-req-" not in request_id:
        return None
    return request_id.rsplit("-req-", 1)[0]


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    ttfts = [float(record["ttft"]) for record in successful]
    totals = [float(record["total_latency"]) for record in successful]
    return {
        "requests": len(records),
        "successful_requests": len(successful),
        "avg_ttft": fmean(ttfts) if ttfts else 0.0,
        "p90_ttft": _percentile(ttfts, 90),
        "avg_total_latency": fmean(totals) if totals else 0.0,
        "p90_total_latency": _percentile(totals, 90),
    }


def _build_summary(
    latency_records: List[Dict[str, Any]],
    routing_records: List[Dict[str, Any]],
    joined_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    matched_ids = {record["request_id"] for record in joined_records}
    latency_ids = {record["request_id"] for record in latency_records}
    routing_ids = {record["request_id"] for record in routing_records}
    successful_joined = [record for record in joined_records if record.get("success")]

    worker_records = defaultdict(list)
    adapter_counts = Counter()
    selected_min_load = 0
    selected_above_min_load = 0
    selected_blind_active = 0
    selected_higher_queue_same_min_pressure = 0
    cache_hits = 0

    for record in joined_records:
        worker_id = str(record["selected_worker"])
        selected = record["selected_state"]
        workers = record["workers"]
        worker_records[worker_id].append(record)
        adapter_counts[record["adapter_name"]] += 1
        cache_hits += bool(selected["has_adapter"])

        healthy_states = [
            state for state in workers.values()
            if state.get("is_healthy", True)
        ]
        load_pressures = [state["load_pressure"] for state in healthy_states]
        min_load = min(load_pressures)
        if abs(selected["load_pressure"] - min_load) < 1e-12:
            selected_min_load += 1
        else:
            selected_above_min_load += 1

        if selected["active_decode_seqs"] > 0 and selected["pending_prefill_tokens"] == 0:
            selected_blind_active += 1

        min_queue = min(state["queue_length"] for state in healthy_states)
        if (
            selected["queue_length"] > min_queue
            and abs(selected["load_pressure"] - min_load) < 1e-12
        ):
            selected_higher_queue_same_min_pressure += 1

    ttft_values = [float(record["ttft"]) for record in successful_joined]
    slow_threshold = _percentile(ttft_values, 90)
    slow_records = [
        record for record in successful_joined
        if float(record["ttft"]) >= slow_threshold
    ]

    return {
        "latency_requests": len(latency_records),
        "routing_requests": len(routing_records),
        "matched_requests": len(joined_records),
        "missing_route_count": len(latency_ids - routing_ids),
        "missing_latency_count": len(routing_ids - latency_ids),
        "missing_route_request_ids": sorted(latency_ids - routing_ids)[:20],
        "missing_latency_request_ids": sorted(routing_ids - latency_ids)[:20],
        "latency": _latency_metrics(latency_records),
        "worker_metrics": {
            worker_id: _latency_metrics(records)
            for worker_id, records in sorted(worker_records.items())
        },
        "routing": {
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / len(joined_records) if joined_records else 0.0,
            "selected_min_load": selected_min_load,
            "selected_above_min_load": selected_above_min_load,
            "selected_active_decode_with_zero_prefill": selected_blind_active,
            "selected_higher_queue_at_same_min_pressure": (
                selected_higher_queue_same_min_pressure
            ),
        },
        "slow_ttft": {
            "threshold": slow_threshold,
            "requests": len(slow_records),
            "worker_counts": dict(Counter(str(r["selected_worker"]) for r in slow_records)),
            "top_adapters": Counter(r["adapter_name"] for r in slow_records).most_common(10),
            "active_decode_with_zero_prefill": sum(
                r["selected_state"]["active_decode_seqs"] > 0
                and r["selected_state"]["pending_prefill_tokens"] == 0
                for r in slow_records
            ),
        },
        "top_adapters": adapter_counts.most_common(10),
        "matched_request_ids": len(matched_ids),
    }


def join_request_diagnostics(
    latency_file: Path,
    routing_file: Path,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """Join latency and routing JSONL files and write joined data plus a summary."""
    latency_file = Path(latency_file)
    routing_file = Path(routing_file)
    output_dir = Path(output_dir) if output_dir else latency_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    latency_records = _load_jsonl(latency_file)
    routing_records = _load_jsonl(routing_file)
    latency_by_id = _index_unique(latency_records, latency_file)
    run_prefixes = {
        prefix for prefix in (
            _benchmark_run_prefix(request_id) for request_id in latency_by_id
        )
        if prefix is not None
    }
    if run_prefixes:
        routing_records = [
            record for record in routing_records
            if _benchmark_run_prefix(record.get("request_id", "")) in run_prefixes
        ]
    routing_by_id = _index_unique(routing_records, routing_file)

    joined_records = []
    for request_id, latency in latency_by_id.items():
        route = routing_by_id.get(request_id)
        if route is None:
            continue
        selected_worker = route["selected_worker"]
        selected_state = route["workers"][str(selected_worker)]
        joined_records.append({
            **latency,
            "route_timestamp": route["timestamp"],
            "selected_worker": selected_worker,
            "selected_state": selected_state,
            "workers": route["workers"],
        })

    run_suffix = latency_file.stem.removeprefix("request_latencies_")
    joined_file = output_dir / f"joined_requests_{run_suffix}.jsonl"
    summary_file = output_dir / f"summary_{run_suffix}.json"

    with open(joined_file, "w") as file:
        for record in joined_records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = _build_summary(latency_records, routing_records, joined_records)
    summary["latency_file"] = str(latency_file)
    summary["routing_file"] = str(routing_file)
    summary["joined_file"] = str(joined_file)
    summary["summary_file"] = str(summary_file)
    with open(summary_file, "w") as file:
        json.dump(summary, file, ensure_ascii=True, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-file", type=Path, required=True)
    parser.add_argument("--routing-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    summary = join_request_diagnostics(
        latency_file=args.latency_file,
        routing_file=args.routing_file,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
