"""Propagate per-replica settings into original S-LoRA spawn children."""

import os

from slora.common.configs.config import setting


if "SLORA_BASELINE_NCCL_PORT" in os.environ:
    setting["nccl_port"] = int(os.environ["SLORA_BASELINE_NCCL_PORT"])

if "SLORA_BASELINE_MAX_REQ_TOTAL_LEN" in os.environ:
    setting["max_req_total_len"] = int(
        os.environ["SLORA_BASELINE_MAX_REQ_TOTAL_LEN"]
    )
