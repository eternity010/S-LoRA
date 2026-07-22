#!/usr/bin/env python3
"""Run three original S-LoRA instances with client-side round robin."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
import requests


DEFAULT_SLORA_PYTHON = "/home/hzheng/.conda/envs/slora-baseline/bin/python"
DEFAULT_SLORA_REPO = "/home/hzheng/S-LoRA-baseline"
DEFAULT_MODEL = "/home/hzheng/models/llama-7b"
DEFAULT_RANK16_ADAPTER = "/home/hzheng/models/alpaca-lora-7b"
DEFAULT_RANK64_ADAPTER = "/home/hzheng/models/bactrian-x-llama-7b-lora"
PROMPT_TOKEN_OVERHEAD = 2
RUNTIME_SHIM_DIR = Path(__file__).resolve().parent / "runtime_shim"


@dataclass(frozen=True)
class TraceRequest:
    req_id: int
    req_time: float
    adapter_id: int
    input_len: int
    output_len: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run three original S-LoRA replicas with round-robin routing."
    )
    parser.add_argument("--trace-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--slora-python", default=DEFAULT_SLORA_PYTHON)
    parser.add_argument("--slora-repo", type=Path, default=Path(DEFAULT_SLORA_REPO))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rank16-adapter", default=DEFAULT_RANK16_ADAPTER)
    parser.add_argument("--rank64-adapter", default=DEFAULT_RANK64_ADAPTER)
    parser.add_argument("--gpu-ids", default="1,2,3")
    parser.add_argument("--ports", default="38200,38201,38202")
    parser.add_argument("--nccl-ports", default="39200,39201,39202")
    parser.add_argument("--num-adapters", type=int, default=100)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=15000)
    parser.add_argument("--batch-max-tokens", type=int, default=3072)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    return parser.parse_args()


def load_trace(path: Path, num_adapters: int, max_model_len: int):
    requests_out = []
    required = {"req_id", "req_time", "adapter_id", "input_len", "output_len"}
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            request = TraceRequest(
                req_id=int(row["req_id"]),
                req_time=float(row["req_time"]),
                adapter_id=int(row["adapter_id"]),
                input_len=int(row["input_len"]),
                output_len=int(row["output_len"]),
            )
            if not 0 <= request.adapter_id < num_adapters:
                raise ValueError(
                    f"{path}:{line_no} adapter_id {request.adapter_id} out of range"
                )
            actual_total = (
                request.input_len + PROMPT_TOKEN_OVERHEAD + request.output_len
            )
            if actual_total > max_model_len:
                raise ValueError(
                    f"{path}:{line_no} total tokens exceed max_model_len: "
                    f"{request.input_len}+{PROMPT_TOKEN_OVERHEAD}+"
                    f"{request.output_len}>{max_model_len}"
                )
            requests_out.append(request)
    return requests_out


def adapter_dirs(num_adapters: int, rank16_path: str, rank64_path: str):
    if num_adapters <= 0:
        raise ValueError("num_adapters must be positive")
    return [
        f"{rank16_path if adapter_id % 2 == 0 else rank64_path}-{adapter_id // 2}"
        for adapter_id in range(num_adapters)
    ]


def build_server_command(args, port: int, nccl_port: int):
    command = [
        args.slora_python,
        "-m",
        "slora.server.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model_dir",
        args.model,
        "--tokenizer_mode",
        "auto",
        "--max_total_token_num",
        str(args.max_total_tokens),
        "--batch_max_tokens",
        str(args.batch_max_tokens),
        "--max_req_input_len",
        str(args.max_model_len - 1),
        "--max_req_total_len",
        str(args.max_model_len),
        "--nccl_port",
        str(nccl_port),
        "--disable_log_stats",
        "--swap",
    ]
    for adapter_dir in adapter_dirs(
        args.num_adapters, args.rank16_adapter, args.rank64_adapter
    ):
        command.extend(["--lora-dirs", adapter_dir])
    return command


def wait_for_server(process, port: int, timeout: float, log_path: Path):
    deadline = time.time() + timeout
    session = requests.Session()
    session.trust_env = False
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"S-LoRA server on port {port} exited with {process.returncode}"
            )
        if log_path.is_file() and "Traceback (most recent call last)" in log_path.read_text(
            errors="replace"
        ):
            raise RuntimeError(
                f"S-LoRA server on port {port} failed during startup; see {log_path}"
            )
        try:
            response = session.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"S-LoRA server on port {port} did not become ready")


def start_servers(args, gpu_ids, ports, nccl_ports):
    processes = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for replica_id, (gpu_id, port, nccl_port) in enumerate(
            zip(gpu_ids, ports, nccl_ports)
        ):
            log_path = (
                args.output_dir
                / f"server_{args.run_label}_replica{replica_id}_gpu{gpu_id}.log"
            )
            log_handle = log_path.open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["SLORA_BASELINE_NCCL_PORT"] = str(nccl_port)
            env["SLORA_BASELINE_MAX_REQ_TOTAL_LEN"] = str(args.max_model_len)
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                [str(RUNTIME_SHIM_DIR), str(args.slora_repo)]
            ) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
            command = build_server_command(args, port, nccl_port)
            process = subprocess.Popen(
                command,
                cwd=args.slora_repo,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            processes.append((process, log_handle, log_path))
            print(
                f"Starting replica {replica_id} on GPU {gpu_id}, port {port}",
                flush=True,
            )
            print(f"  log: {log_path}", flush=True)
            wait_for_server(process, port, args.startup_timeout, log_path)
            print(f"  replica {replica_id} ready", flush=True)
    except Exception:
        stop_servers(processes)
        raise
    return processes


def stop_servers(processes):
    for process, _, _ in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.time() + 20
    for process, log_handle, _ in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        log_handle.close()


def dummy_prompt(prompt_len: int):
    return "Hello " * prompt_len


async def send_request(session, request, port, adapter_paths, start_time):
    scheduled_at = start_time + request.req_time
    sent_at = time.time()
    result = {
        **asdict(request),
        "replica_port": port,
        "scheduled_at": scheduled_at,
        "sent_at": sent_at,
        "schedule_delay": sent_at - scheduled_at,
        "first_token_at": None,
        "finished_at": None,
        "ttft": None,
        "total_latency": None,
        "success": False,
        "error": None,
    }
    payload = {
        "req_id": f"slora-baseline-{request.req_id}",
        "lora_dir": adapter_paths[request.adapter_id],
        "inputs": dummy_prompt(request.input_len),
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": request.output_len,
        },
    }
    try:
        async with session.post(
            f"http://127.0.0.1:{port}/generate_stream", json=payload
        ) as response:
            if response.status != 200:
                result["error"] = await response.text()
                result["finished_at"] = time.time()
                return result
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                if result["first_token_at"] is None:
                    result["first_token_at"] = time.time()
                    result["ttft"] = result["first_token_at"] - sent_at
    except Exception as exc:
        result["error"] = str(exc)
        result["finished_at"] = time.time()
        return result

    result["finished_at"] = time.time()
    result["total_latency"] = result["finished_at"] - sent_at
    result["success"] = result["ttft"] is not None
    return result


async def run_workload(requests_in, ports, adapter_paths):
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, trust_env=False
    ) as session:
        start = time.time()
        tasks = []
        for request in requests_in:
            delay = start + request.req_time - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            port = ports[request.req_id % len(ports)]
            tasks.append(
                asyncio.create_task(
                    send_request(session, request, port, adapter_paths, start)
                )
            )
        results = await asyncio.gather(*tasks)
    return results, time.time() - start


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(results, benchmark_time, req_rate):
    valid = [row for row in results if row["success"]]
    latencies = [row["total_latency"] for row in valid]
    ttfts = [row["ttft"] for row in valid]
    warmup_requests = int(req_rate * 10)
    strip_throughput = len(valid) / benchmark_time
    if len(valid) > warmup_requests * 2 and benchmark_time > 20:
        strip_throughput = (
            len(valid) - warmup_requests * 2
        ) / (benchmark_time - 20)
    return {
        "completed_requests": len(valid),
        "failed_requests": len(results) - len(valid),
        "throughput": len(valid) / benchmark_time,
        "strip_throughput": strip_throughput,
        "avg_latency": statistics.fmean(latencies) if latencies else 0.0,
        "avg_first_token_latency": statistics.fmean(ttfts) if ttfts else 0.0,
        "p50_latency": percentile(latencies, 50),
        "p90_latency": percentile(latencies, 90),
        "p50_first_token_latency": percentile(ttfts, 50),
        "p90_first_token_latency": percentile(ttfts, 90),
        "benchmark_time": benchmark_time,
    }


def write_results(args, requests_in, results, summary, gpu_ids, ports):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latency_path = args.output_dir / f"request_latencies_{args.run_label}.jsonl"
    with latency_path.open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    record = {
        "config": {
            "backend": "slora-original-replicated",
            "model": args.model,
            "trace_file": str(args.trace_file),
            "run_label": args.run_label,
            "gpu_ids": gpu_ids,
            "ports": ports,
            "num_replicas": len(ports),
            "num_adapters": args.num_adapters,
            "routing_strategy": "round-robin",
            "max_model_len": args.max_model_len,
            "max_total_tokens": args.max_total_tokens,
            "batch_max_tokens": args.batch_max_tokens,
            "total_requests": len(requests_in),
            "duration": args.duration,
        },
        "result": summary,
        "timestamp": time.time(),
    }
    result_path = args.output_dir / "results.jsonl"
    with result_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"Request latencies: {latency_path}")
    print(f"Result: {result_path}")


def parse_int_list(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    args = parse_args()
    gpu_ids = parse_int_list(args.gpu_ids)
    ports = parse_int_list(args.ports)
    nccl_ports = parse_int_list(args.nccl_ports)
    if not gpu_ids or len(gpu_ids) != len(ports) or len(ports) != len(nccl_ports):
        raise ValueError(
            "gpu-ids, ports, and nccl-ports must contain the same nonzero count"
        )
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    if args.batch_max_tokens < args.max_model_len:
        raise ValueError("batch-max-tokens must be at least max-model-len")
    requests_in = load_trace(args.trace_file, args.num_adapters, args.max_model_len)
    req_rate = len(requests_in) / args.duration
    print(f"Loaded {len(requests_in)} requests ({req_rate:.2f} req/s)", flush=True)
    adapter_paths = adapter_dirs(
        args.num_adapters, args.rank16_adapter, args.rank64_adapter
    )
    processes = []
    try:
        processes = start_servers(args, gpu_ids, ports, nccl_ports)
        results, benchmark_time = asyncio.run(
            run_workload(requests_in, ports, adapter_paths)
        )
        summary = summarize(results, benchmark_time, req_rate)
        write_results(args, requests_in, results, summary, gpu_ids, ports)
    finally:
        stop_servers(processes)


if __name__ == "__main__":
    main()
