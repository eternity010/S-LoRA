"""Plot worker load, routing choices, and TTFT from request diagnostics."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .request_diagnostics import _load_jsonl


WORKER_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _find_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise ValueError(f"No file matching {pattern!r} under {directory}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple files match {pattern!r} under {directory}; "
            "pass a single server diagnostics directory"
        )
    return matches[0]


def _resolve_run_directory(directory: Path) -> Path:
    if (directory / "route_decisions.jsonl").is_file():
        return directory
    routing_files = sorted(directory.glob("**/route_decisions.jsonl"))
    if not routing_files:
        raise ValueError(f"No route_decisions.jsonl found under {directory}")
    if len(routing_files) > 1:
        raise ValueError(
            f"Multiple diagnostics runs found under {directory}; "
            "pass one timestamped server directory"
        )
    return routing_files[0].parent


def _worker_ids(routing_records: Iterable[Dict[str, Any]]) -> List[str]:
    worker_ids = set()
    for record in routing_records:
        worker_ids.update(str(worker_id) for worker_id in record.get("workers", {}))
    return sorted(worker_ids, key=int)


def _window_index(elapsed: float, window_sec: float) -> int:
    return max(0, int(math.floor(elapsed / window_sec)))


def analyze_load_diagnostics(
    routing_records: List[Dict[str, Any]],
    joined_records: List[Dict[str, Any]],
    window_sec: float = 10.0,
) -> Dict[str, Any]:
    """Aggregate diagnostics into worker and fixed-window statistics."""
    if not routing_records:
        raise ValueError("Routing diagnostics are empty")
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")

    routing_records = sorted(routing_records, key=lambda record: record["timestamp"])
    start_time = float(routing_records[0]["timestamp"])
    worker_ids = _worker_ids(routing_records)
    if not worker_ids:
        raise ValueError("Routing diagnostics contain no worker states")

    request_counts = Counter()
    prompt_tokens = Counter()
    raw_samples = defaultdict(list)
    report_age_samples = defaultdict(list)
    stale_report_samples = Counter()
    optimistic_samples = Counter()
    report_seq_regressions = Counter()
    last_report_seq = {}
    highest_raw_counts = Counter()
    route_windows = defaultdict(Counter)

    for record in routing_records:
        elapsed = float(record["timestamp"]) - start_time
        selected = str(record["selected_worker"])
        request_counts[selected] += 1
        prompt_tokens[selected] += int(record.get("prompt_len", 0))
        route_windows[_window_index(elapsed, window_sec)][selected] += 1

        states = record["workers"]
        raw_by_worker = {
            worker_id: int(states[worker_id].get("pending_raw_tokens", 0))
            for worker_id in worker_ids
        }
        max_raw = max(raw_by_worker.values())
        for worker_id, raw in raw_by_worker.items():
            raw_samples[worker_id].append(raw)
            if raw == max_raw:
                highest_raw_counts[worker_id] += 1
            state = states[worker_id]
            report_age = state.get("report_age_ms")
            if report_age is not None:
                report_age = float(report_age)
                report_age_samples[worker_id].append(report_age)
                stale_report_samples[worker_id] += report_age > 300.0
            optimistic_samples[worker_id] += (
                int(state.get("optimistic_request_count", 0)) > 0
            )
            report_seq = int(state.get("report_seq", 0))
            if report_seq > 0:
                previous_seq = last_report_seq.get(worker_id)
                if previous_seq is not None and report_seq < previous_seq:
                    report_seq_regressions[worker_id] += 1
                last_report_seq[worker_id] = report_seq

    ttft_windows = defaultdict(lambda: defaultdict(list))
    worker_ttfts = defaultdict(list)
    for record in joined_records:
        if not record.get("success") or record.get("ttft") is None:
            continue
        route_timestamp = record.get("route_timestamp")
        if route_timestamp is None:
            continue
        elapsed = float(route_timestamp) - start_time
        if elapsed < 0:
            continue
        worker_id = str(record["selected_worker"])
        ttft = float(record["ttft"])
        ttft_windows[_window_index(elapsed, window_sec)][worker_id].append(ttft)
        worker_ttfts[worker_id].append(ttft)

    max_elapsed = float(routing_records[-1]["timestamp"]) - start_time
    window_count = _window_index(max_elapsed, window_sec) + 1
    windows = []
    for index in range(window_count):
        windows.append({
            "start_sec": index * window_sec,
            "end_sec": (index + 1) * window_sec,
            "request_counts": {
                worker_id: route_windows[index][worker_id]
                for worker_id in worker_ids
            },
            "avg_ttft": {
                worker_id: (
                    fmean(ttft_windows[index][worker_id])
                    if ttft_windows[index][worker_id]
                    else None
                )
                for worker_id in worker_ids
            },
        })

    return {
        "start_timestamp": start_time,
        "duration_sec": max_elapsed,
        "window_sec": window_sec,
        "routing_requests": len(routing_records),
        "joined_requests": len(joined_records),
        "workers": {
            worker_id: {
                "request_count": request_counts[worker_id],
                "prompt_tokens": prompt_tokens[worker_id],
                "avg_pending_raw_tokens": fmean(raw_samples[worker_id]),
                "max_pending_raw_tokens": max(raw_samples[worker_id]),
                "highest_raw_samples": highest_raw_counts[worker_id],
                "avg_report_age_ms": (
                    fmean(report_age_samples[worker_id])
                    if report_age_samples[worker_id]
                    else None
                ),
                "max_report_age_ms": (
                    max(report_age_samples[worker_id])
                    if report_age_samples[worker_id]
                    else None
                ),
                "stale_report_samples_over_300ms": stale_report_samples[worker_id],
                "optimistic_state_samples": optimistic_samples[worker_id],
                "report_seq_regressions": report_seq_regressions[worker_id],
                "avg_ttft": (
                    fmean(worker_ttfts[worker_id])
                    if worker_ttfts[worker_id]
                    else None
                ),
            }
            for worker_id in worker_ids
        },
        "windows": windows,
    }


def _plot(
    routing_records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_base: Path,
) -> Tuple[Path, Path]:
    routing_records = sorted(routing_records, key=lambda record: record["timestamp"])
    start_time = summary["start_timestamp"]
    worker_ids = list(summary["workers"])
    elapsed = [float(record["timestamp"]) - start_time for record in routing_records]

    has_report_age = any(
        any(record["workers"][worker_id].get("report_age_ms") is not None
            for record in routing_records)
        for worker_id in worker_ids
    )
    panel_count = 5 if has_report_age else 4
    fig, axes = plt.subplots(panel_count, 1, figsize=(14, 18 if has_report_age else 15),
                             sharex=True)
    for index, worker_id in enumerate(worker_ids):
        color = WORKER_COLORS[index % len(WORKER_COLORS)]
        raw_tokens = [
            record["workers"][worker_id].get("pending_raw_tokens", 0)
            for record in routing_records
        ]
        axes[0].plot(elapsed, raw_tokens, color=color, linewidth=1.2,
                     label=f"Worker {worker_id}")
    axes[0].set_ylabel("Pending raw tokens")
    axes[0].set_title("Worker load reported at each routing decision")
    axes[0].legend(ncol=len(worker_ids))

    next_axis = 1
    if has_report_age:
        for index, worker_id in enumerate(worker_ids):
            color = WORKER_COLORS[index % len(WORKER_COLORS)]
            ages = [
                record["workers"][worker_id].get("report_age_ms", math.nan)
                for record in routing_records
            ]
            axes[next_axis].plot(elapsed, ages, color=color, linewidth=1.0,
                                 label=f"Worker {worker_id}")
        axes[next_axis].axhline(300.0, color="#444444", linestyle="--",
                                linewidth=1.0, label="300 ms stale threshold")
        axes[next_axis].set_ylabel("Report age (ms)")
        axes[next_axis].set_title("Worker-state freshness at routing time")
        axes[next_axis].legend(ncol=len(worker_ids) + 1)
        next_axis += 1

    selected = [int(record["selected_worker"]) for record in routing_records]
    axes[next_axis].scatter(
        elapsed,
        selected,
        c=[WORKER_COLORS[value % len(WORKER_COLORS)] for value in selected],
        s=10,
        alpha=0.7,
    )
    axes[next_axis].set_yticks([int(worker_id) for worker_id in worker_ids])
    axes[next_axis].set_ylabel("Selected worker")
    axes[next_axis].set_title("Routing decisions")
    route_axis = next_axis + 1

    window_centers = [
        (window["start_sec"] + window["end_sec"]) / 2.0
        for window in summary["windows"]
    ]
    for index, worker_id in enumerate(worker_ids):
        color = WORKER_COLORS[index % len(WORKER_COLORS)]
        counts = [window["request_counts"][worker_id] for window in summary["windows"]]
        axes[route_axis].plot(window_centers, counts, marker="o", markersize=3,
                              color=color, linewidth=1.2, label=f"Worker {worker_id}")
    axes[route_axis].set_ylabel(f"Requests / {summary['window_sec']:g}s")
    axes[route_axis].set_title("Per-window routing volume")

    for index, worker_id in enumerate(worker_ids):
        color = WORKER_COLORS[index % len(WORKER_COLORS)]
        ttfts = [
            value if value is not None else math.nan
            for value in (window["avg_ttft"][worker_id] for window in summary["windows"])
        ]
        axes[route_axis + 1].plot(window_centers, ttfts, marker="o", markersize=3,
                                  color=color, linewidth=1.2,
                                  label=f"Worker {worker_id}")
    axes[route_axis + 1].set_ylabel("Average TTFT (s)")
    axes[route_axis + 1].set_xlabel("Elapsed time (s)")
    axes[route_axis + 1].set_title("Per-window first-token latency")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_load_diagnostics(
    diagnostics_dir: Path,
    output_dir: Path = None,
    window_sec: float = 10.0,
) -> Dict[str, Path]:
    """Load one diagnostics run and write plots plus an aggregate summary."""
    diagnostics_dir = _resolve_run_directory(Path(diagnostics_dir))
    routing_file = _find_single(diagnostics_dir, "route_decisions.jsonl")
    joined_file = _find_single(diagnostics_dir, "joined_requests_*.jsonl")
    output_dir = Path(output_dir) if output_dir else diagnostics_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    routing_records = _load_jsonl(routing_file)
    joined_records = _load_jsonl(joined_file)
    summary = analyze_load_diagnostics(routing_records, joined_records, window_sec)
    summary["routing_file"] = str(routing_file)
    summary["joined_file"] = str(joined_file)

    output_base = output_dir / "worker_load_timeline"
    png_path, pdf_path = _plot(routing_records, summary, output_base)
    summary_path = output_dir / "worker_load_timeline_summary.json"
    with open(summary_path, "w") as file:
        json.dump(summary, file, ensure_ascii=True, indent=2)

    return {"png": png_path, "pdf": pdf_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-sec", type=float, default=10.0)
    args = parser.parse_args()

    outputs = plot_load_diagnostics(
        diagnostics_dir=args.diagnostics_dir,
        output_dir=args.output_dir,
        window_sec=args.window_sec,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
