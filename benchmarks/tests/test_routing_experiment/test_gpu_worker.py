"""
Tests for data-parallel GPU worker scheduling edge cases.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from slora.server.io_struct import Batch, Req
from slora.server.router.gpu_worker import GPUWorker
from slora.server.sampling_params import SamplingParams


class DummyReqQueue:
    def __init__(self, batch):
        self.batch = batch
        self.waiting_req_list = []

    def generate_new_batch(self, *_args, **_kwargs):
        batch = self.batch
        self.batch = None
        return batch


class EmptyReqQueue:
    waiting_req_list = []

    def generate_new_batch(self, *_args, **_kwargs):
        return None


class DummyModelRpc:
    def __init__(self, fail_prefill=False):
        self.fail_prefill = fail_prefill
        self.removed_batches = []
        self.eviction_calls = []

    async def init_batch(self, _batch_id, _reqs_rpc):
        return None

    async def prefill_batch(self, _batch_id):
        if self.fail_prefill:
            raise RuntimeError("prefill failed")
        return {"req-1": (42, {"id": 42})}

    async def remove_batch(self, batch_id):
        self.removed_batches.append(batch_id)

    async def decrease_request_counts(self, _adapter_dirs):
        return None

    async def check_lora_memory(self):
        return {"adapter_cells": [100], "total_cells": 100}

    async def trigger_threshold_eviction(self, **kwargs):
        self.eviction_calls.append(kwargs)
        return {"evicted": False}


def test_process_requests_returns_prefill_finished_response():
    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.stop_sequences = []
    req = Req(
        adapter_dir="/adapters/a",
        request_id="req-1",
        prompt_ids=[1, 2, 3],
        sample_params=sampling_params,
    )
    batch = Batch("batch-1", [req])

    worker = GPUWorker.__new__(GPUWorker)
    worker.worker_id = 0
    worker.req_queue = DummyReqQueue(batch)
    worker.current_batch = None
    worker.lora_ranks = {}
    worker.actual_adapter_memory_usage = {}
    worker.args = SimpleNamespace(eos_id=2, no_lora=True)
    worker.model_rpc = DummyModelRpc()
    worker.state_reporter = None

    responses = asyncio.run(worker._process_requests())

    assert len(responses) == 1
    response = responses[0]
    assert response["request_id"] == "req-1"
    assert response["output_ids"] == [1, 2, 3, 42]
    assert response["metadata"]["completion_tokens"] == 1
    assert response["finished"] is True
    assert worker.current_batch is None
    assert worker.model_rpc.removed_batches == ["batch-1"]


def test_process_requests_returns_error_response_for_failed_new_batch():
    sampling_params = SamplingParams(max_new_tokens=8)
    sampling_params.stop_sequences = []
    req = Req(
        adapter_dir="/adapters/a",
        request_id="req-1",
        prompt_ids=[1, 2, 3],
        sample_params=sampling_params,
    )
    batch = Batch("batch-1", [req])

    worker = GPUWorker.__new__(GPUWorker)
    worker.worker_id = 0
    worker.req_queue = DummyReqQueue(batch)
    worker.current_batch = None
    worker.lora_ranks = {}
    worker.actual_adapter_memory_usage = {}
    worker.args = SimpleNamespace(eos_id=2, no_lora=True)
    worker.model_rpc = DummyModelRpc(fail_prefill=True)
    worker.state_reporter = None

    responses = asyncio.run(worker._process_requests())

    assert len(responses) == 1
    assert responses[0]["request_id"] == "req-1"
    assert responses[0]["success"] is False
    assert responses[0]["finished"] is True
    assert worker.model_rpc.removed_batches == ["batch-1"]


def test_finish_eviction_preserves_existing_current_batch_adapters():
    finishing_params = SamplingParams(max_new_tokens=1)
    finishing_params.stop_sequences = []
    finishing_req = Req(
        adapter_dir="/adapters/new",
        request_id="new-req",
        prompt_ids=[1],
        sample_params=finishing_params,
    )
    finishing_req.output_ids.append(42)
    finishing_req.has_generate_finished = True
    finishing_batch = Batch("new-batch", [finishing_req])

    active_params = SamplingParams(max_new_tokens=8)
    active_params.stop_sequences = []
    active_req = Req(
        adapter_dir="/adapters/active",
        request_id="active-req",
        prompt_ids=[2],
        sample_params=active_params,
    )
    active_batch = Batch("active-batch", [active_req])

    worker = GPUWorker.__new__(GPUWorker)
    worker.worker_id = 0
    worker.req_queue = EmptyReqQueue()
    worker.current_batch = active_batch
    worker.args = SimpleNamespace(
        no_lora=False,
        max_lora_ratio=0.2,
        evict_interval_threshold=0.5,
    )
    worker.model_rpc = DummyModelRpc()
    worker.adapter_cache = {}
    worker.state_reporter = None

    async def noop_update():
        return None

    worker._update_actual_adapter_usage = noop_update

    asyncio.run(worker._handle_finish_req(finishing_batch, True))

    assert worker.model_rpc.eviction_calls
    preserve_dirs = worker.model_rpc.eviction_calls[0]["preserve_dirs"]
    assert preserve_dirs == {"/adapters/new", "/adapters/active"}


def test_reporter_state_uses_profiled_rank_cost_for_waiting_active_and_top_k():
    waiting = SimpleNamespace(
        adapter_dir="/adapters/rank16",
        prompt_ids=list(range(100)),
    )
    active = SimpleNamespace(
        adapter_dir="/adapters/rank64",
        prompt_ids=list(range(100)),
    )

    worker = GPUWorker.__new__(GPUWorker)
    worker.worker_id = 0
    worker.req_queue = SimpleNamespace(waiting_req_list=[waiting])
    worker.current_batch = SimpleNamespace(reqs=[active])
    worker.lora_ranks = {
        "/adapters/rank16": 16,
        "/adapters/rank64": 64,
    }
    worker.adapter_cache = {}
    worker.model_rpc = None
    worker._profiled_alpha = 0.0
    worker._profiled_rank_beta = 0.00845
    worker._hidden_dim = 4096

    state = worker._get_state_for_reporter()

    assert state["pending_prefill_tokens"] == 113
    assert state["active_rwpt_tokens"] == 154
    assert state["top_k_rwpt_adapters"] == [("/adapters/rank16", 113.0)]
    assert state["profiled_rank_beta"] == 0.00845
