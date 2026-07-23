#!/usr/bin/env python3
"""Create same-source LoRA checkpoints for rank profiling."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-adapter",
        type=Path,
        default=Path("/home/hzheng/models/bactrian-x-llama-7b-lora"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/hzheng/models/rankflow-rank-profile"),
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    return parser.parse_args()


def slice_state_dict(state_dict: dict, rank: int, source_rank: int) -> dict:
    sliced = {}
    for name, tensor in state_dict.items():
        if name.endswith("lora_A.weight"):
            if tensor.shape[0] != source_rank:
                raise ValueError(f"Unexpected LoRA-A shape for {name}: {tensor.shape}")
            sliced[name] = tensor[:rank, :].contiguous()
        elif name.endswith("lora_B.weight"):
            if tensor.shape[1] != source_rank:
                raise ValueError(f"Unexpected LoRA-B shape for {name}: {tensor.shape}")
            sliced[name] = tensor[:, :rank].contiguous()
        else:
            sliced[name] = tensor
    return sliced


def main() -> None:
    args = parse_args()
    source = args.source_adapter.resolve()
    config_path = source / "adapter_config.json"
    weights_path = source / "adapter_model.bin"
    with config_path.open() as handle:
        source_config = json.load(handle)

    source_rank = int(source_config["r"])
    ranks = sorted(set(args.ranks))
    if any(rank <= 0 or rank > source_rank for rank in ranks):
        raise ValueError(f"Ranks must be in [1, {source_rank}], got {ranks}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    state_dict = None
    for rank in ranks:
        target = args.output_root / f"rank{rank}"
        if target.is_symlink() or target.exists():
            if target.is_symlink() and target.resolve() == source and rank == source_rank:
                print(f"Reusing {target} -> {source}")
                continue
            if target.is_dir() and (target / "adapter_model.bin").exists():
                with (target / "adapter_config.json").open() as handle:
                    existing = json.load(handle)
                if int(existing["r"]) == rank:
                    print(f"Reusing {target}")
                    continue
            raise FileExistsError(f"Refusing to replace existing path: {target}")

        if rank == source_rank:
            os.symlink(source, target, target_is_directory=True)
            print(f"Linked {target} -> {source}")
            continue

        if state_dict is None:
            state_dict = torch.load(weights_path, map_location="cpu")
        target.mkdir()
        config = dict(source_config)
        config["r"] = rank
        with (target / "adapter_config.json").open("w") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        torch.save(
            slice_state_dict(state_dict, rank=rank, source_rank=source_rank),
            target / "adapter_model.bin",
        )
        readme = source / "README.md"
        if readme.exists():
            shutil.copy2(readme, target / "README.md")
        print(f"Created rank-{rank} adapter at {target}")


if __name__ == "__main__":
    main()
