#!/usr/bin/env python3
"""Measure rank-dependent prefill latency with a warmed single-GPU server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/hzheng/models/llama-7b")
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=Path("/home/hzheng/models/rankflow-rank-profile"),
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", default=[256, 512, 1024]
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--port", type=int, default=38300)
    parser.add_argument("--max-total-tokens", type=int, default=15000)
    parser.add_argument("--batch-max-tokens", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/routing_comparison_results/rank_prefill_profile"),
    )
    return parser.parse_args()


def build_prompt(tokenizer, prompt_tokens: int) -> str:
    if prompt_tokens < 2:
        raise ValueError("prompt length must be at least two tokens")
    prompt = " a" * (prompt_tokens - 2)
    actual = len(tokenizer.encode(prompt))
    if actual != prompt_tokens:
        raise ValueError(f"Expected {prompt_tokens} tokens, tokenizer produced {actual}")
    return prompt


def wait_until_ready(process: subprocess.Popen, port: int, timeout: float = 600) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"S-LoRA server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"S-LoRA server was not ready within {timeout:.0f}s")


async def measure_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    adapter_dir: str | None,
    prompt: str,
) -> float:
    payload = {
        "req_id": f"rank-profile-{uuid.uuid4().hex}",
        "model_dir": model,
        "lora_dir": adapter_dir,
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": 1,
        },
    }
    started = time.perf_counter()
    first_token = None
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        async for chunk, _ in response.content.iter_chunks():
            if chunk and first_token is None:
                first_token = time.perf_counter() - started
    if first_token is None:
        raise RuntimeError("Server returned no token")
    return first_token


async def run_measurements(args: argparse.Namespace, adapter_dirs: dict[int, str]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompts = {
        length: build_prompt(tokenizer, length) for length in args.prompt_lengths
    }
    rank_order = [0] + sorted(adapter_dirs)
    rng = random.Random(args.seed)
    records = []
    timeout = aiohttp.ClientTimeout(total=3600)
    url = f"http://127.0.0.1:{args.port}/generate_stream"

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for prompt_length in sorted(prompts):
            prompt = prompts[prompt_length]
            for rank in rank_order:
                adapter = adapter_dirs.get(rank)
                for _ in range(args.warmups):
                    await measure_request(session, url, args.model, adapter, prompt)

            for repetition in range(args.repetitions):
                shuffled = list(rank_order)
                rng.shuffle(shuffled)
                for rank in shuffled:
                    ttft = await measure_request(
                        session,
                        url,
                        args.model,
                        adapter_dirs.get(rank),
                        prompt,
                    )
                    record = {
                        "rank": rank,
                        "prompt_tokens": prompt_length,
                        "repetition": repetition,
                        "ttft_s": ttft,
                        "adapter_dir": adapter_dirs.get(rank),
                    }
                    records.append(record)
                    print(
                        f"rank={rank:>2} prompt={prompt_length:>4} "
                        f"rep={repetition + 1:>2}/{args.repetitions} "
                        f"ttft={ttft * 1000:.2f} ms"
                    )
    return records


def terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dirs = {
        rank: str((args.adapter_root / f"rank{rank}").resolve()) + "-0"
        for rank in sorted(set(args.ranks))
    }
    for rank, adapter in adapter_dirs.items():
        if not Path(adapter[:-2]).is_dir():
            raise FileNotFoundError(
                f"Missing rank-{rank} adapter. Run prepare_rank_adapters.py first."
            )

    server_log = args.output_dir / "server.log"
    results_file = args.output_dir / "rank_prefill_measurements.jsonl"
    metadata_file = args.output_dir / "metadata.json"
    command = [
        sys.executable,
        "-u",
        "-m",
        "slora.server.api_server",
        "--max_total_token_num",
        str(args.max_total_tokens),
        "--batch_max_tokens",
        str(args.batch_max_tokens),
        "--model",
        args.model,
        "--tokenizer_mode",
        "auto",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp",
        "1",
        "--parallel-mode",
        "tensor",
        "--swap",
        "--disable_log_stats",
    ]
    for adapter in adapter_dirs.values():
        command.extend(["--lora", adapter])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    print("Starting server:", " ".join(command))
    with server_log.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            wait_until_ready(process, args.port)
            print("Server ready; starting warmed sequential measurements")
            records = asyncio.run(run_measurements(args, adapter_dirs))
        finally:
            terminate_process_group(process)

    with results_file.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "model": args.model,
        "source_adapter_root": str(args.adapter_root),
        "ranks": [0] + sorted(adapter_dirs),
        "prompt_lengths": sorted(args.prompt_lengths),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "gpu_id": args.gpu_id,
        "output_tokens": 1,
        "measurement": "client-observed TTFT with sequential requests",
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {results_file}")
    print(f"Wrote {metadata_file}")


if __name__ == "__main__":
    main()
