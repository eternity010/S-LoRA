"""CPU regressions for load visibility between admission and batch merge."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from slora.server.io_struct import Batch, Req
from slora.server.router.adapter_aware_router import AdapterAwareRouter
from slora.server.router.dp_manager import DataParallelRouterManager
from slora.server.router.gpu_worker import GPUWorker
from slora.server.router.req_queue import ReqQueue
from slora.server.router.worker_state import RoutingConfig, WorkerState
from slora.server.router.worker_state_cache import WorkerStateCache
from slora.server.router.worker_state_reporter import WorkerStateReporter
from slora.server.sampling_params import SamplingParams


def make_req(request_id, rank=16, max_new_tokens=2):
    params = SamplingParams(max_new_tokens=max_new_tokens)
    params.stop_sequences = []
    return Req(f"/adapters/rank{rank}", request_id, list(range(100)), params)


def visible_load(state):
    return (
        state["queue_length"],
        state["pending_raw_tokens"] + state["prefilling_raw_tokens"]
        + state["current_batch_prompt_tokens"],
        state["pending_prefill_tokens"] + state["prefilling_rwpt_tokens"]
        + state["active_rwpt_tokens"],
    )


@pytest.fixture
def worker_factory(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    def build(requests, current=None, fail_at=None):
        worker = GPUWorker.__new__(GPUWorker)
        worker.worker_id = 0
        worker.req_queue = ReqQueue(15000, 3072, 64)
        for req in requests:
            worker.req_queue.append(req)
        worker.current_batch = current
        worker.prefilling_batch = None
        worker.lora_ranks = {"/adapters/rank16": 16, "/adapters/rank64": 64}
        worker.actual_adapter_memory_usage = 0
        worker.adapter_cache = {}
        worker.args = SimpleNamespace(eos_id=2, no_lora=False)
        worker.state_reporter = None
        worker._profiled_alpha = 0.0
        worker._profiled_rank_beta = 0.00845
        worker._hidden_dim = 4096
        snapshots = {}

        async def capture(stage):
            await asyncio.sleep(0)
            snapshots.setdefault(stage, []).append(worker._get_state_for_reporter())
            if fail_at == stage:
                raise RuntimeError(f"{stage} failed")
            if fail_at == f"cancel_{stage}":
                raise asyncio.CancelledError()

        async def load(_adapters):
            await capture("load")

        async def init(_batch_id, _reqs):
            await capture("init")

        async def prefill(_batch_id):
            await capture("prefill")
            return {req.request_id: (42, {}) for req in worker.prefilling_batch.reqs}

        async def merge(*_args):
            await capture("merge")

        async def decode(batch):
            await capture("decode")
            return {req.request_id: (42, {}) for req in batch.reqs}

        async def cleanup(*_args):
            await capture("cleanup")

        worker._load_adapters = AsyncMock(side_effect=load)
        worker._infer_batch = AsyncMock(side_effect=decode)
        worker.model_rpc = SimpleNamespace(
            init_batch=AsyncMock(side_effect=init),
            prefill_batch=AsyncMock(side_effect=prefill),
            merge_batch=AsyncMock(side_effect=merge),
            remove_batch=AsyncMock(side_effect=cleanup),
            filter_batch=AsyncMock(side_effect=cleanup),
            decrease_request_counts=AsyncMock(),
            check_lora_memory=AsyncMock(return_value={}),
            trigger_threshold_eviction=AsyncMock(return_value={"evicted": False}),
        )
        return worker, snapshots

    return build


def test_load_is_conserved_from_waiting_through_execution(worker_factory):
    worker, snapshots = worker_factory([make_req("new", max_new_tokens=3)])
    assert visible_load(worker._get_state_for_reporter()) == (1, 100, 113)

    asyncio.run(worker._process_requests())

    for stage in ("load", "init", "prefill"):
        state = snapshots[stage][0]
        assert visible_load(state) == (1, 100, 113)
        assert state["waiting_request_count"] == state["current_batch_size"] == 0
        assert state["prefilling_request_count"] == 1
        assert state["active_decode_seqs"] == 0
    decoding = snapshots["decode"][0]
    assert visible_load(decoding) == (1, 100, 113)
    assert decoding["prefilling_request_count"] == 0
    assert decoding["current_batch_size"] == decoding["active_decode_seqs"] == 1
    assert worker.prefilling_batch is None
    assert visible_load(worker._get_state_for_reporter()) == (1, 100, 113)

    asyncio.run(worker._process_requests())
    assert visible_load(worker._get_state_for_reporter()) == (0, 0, 0)


def test_existing_and_new_batches_are_counted_once_across_merge(worker_factory):
    current = Batch("old-batch", [make_req("old", rank=64)])
    worker, snapshots = worker_factory([make_req("new")], current)

    asyncio.run(worker._process_requests())

    for stage in ("load", "init", "prefill", "merge", "decode"):
        state = snapshots[stage][0]
        assert visible_load(state) == (2, 200, 267)
        assert state["prefilling_request_count"] == (0 if stage == "decode" else 1)
    assert worker.prefilling_batch is None
    assert visible_load(worker._get_state_for_reporter()) == (1, 100, 154)


def test_partial_prefill_completion_excludes_finished_requests(worker_factory):
    worker, snapshots = worker_factory([
        make_req("one-token", max_new_tokens=1), make_req("longer", max_new_tokens=3),
    ])

    responses = asyncio.run(worker._process_requests())

    assert visible_load(snapshots["prefill"][0]) == (2, 200, 226)
    assert visible_load(snapshots["cleanup"][0]) == (1, 100, 113)
    assert snapshots["cleanup"][0]["prefilling_request_count"] == 1
    assert visible_load(snapshots["decode"][0]) == (1, 100, 113)
    assert any(r["request_id"] == "one-token" and r["finished"] for r in responses)


def test_single_token_completion_clears_prefill_load(worker_factory):
    worker, snapshots = worker_factory([make_req("one-token", max_new_tokens=1)])
    asyncio.run(worker._process_requests())
    assert "decode" not in snapshots
    assert visible_load(snapshots["cleanup"][0]) == (0, 0, 0)
    assert worker.prefilling_batch is None
    assert visible_load(worker._get_state_for_reporter()) == (0, 0, 0)


def test_waiting_requests_remain_separate_during_prefill(worker_factory):
    worker, snapshots = worker_factory([make_req("admitted"), make_req("waiting", rank=64)])
    worker.req_queue.batch_max_tokens = 100
    asyncio.run(worker._process_requests())
    state = snapshots["prefill"][0]
    assert visible_load(state) == (2, 200, 267)
    assert state["pending_prefill_tokens"] == 154
    assert state["prefilling_rwpt_tokens"] == 113
    assert state["waiting_request_count"] == state["prefilling_request_count"] == 1


@pytest.mark.parametrize("stage", ["load", "init", "prefill", "merge", "decode"])
def test_failed_batches_leave_no_prefill_load(worker_factory, stage):
    current = Batch("old-batch", [make_req("old", rank=64)]) if stage == "merge" else None
    worker, _ = worker_factory([make_req("new")], current, fail_at=stage)
    responses = asyncio.run(worker._process_requests())
    assert any(not response["success"] for response in responses)
    assert worker.prefilling_batch is None
    assert visible_load(worker._get_state_for_reporter()) == (0, 0, 0)


@pytest.mark.parametrize("stage", ["load", "init", "prefill"])
def test_failed_new_batch_preserves_existing_batch_load(worker_factory, stage):
    current = Batch("old-batch", [make_req("old", rank=64)])
    worker, snapshots = worker_factory([make_req("new")], current, fail_at=stage)
    responses = asyncio.run(worker._process_requests())
    assert any(r["request_id"] == "new" and not r["success"] for r in responses)
    assert visible_load(snapshots[stage][0]) == (2, 200, 267)
    assert visible_load(snapshots["decode"][0]) == (1, 100, 154)
    assert worker.current_batch is current
    assert worker.prefilling_batch is None
    assert visible_load(worker._get_state_for_reporter()) == (1, 100, 154)


@pytest.mark.parametrize("stage", ["load", "init", "prefill", "merge"])
def test_cancelled_processing_clears_prefill_reference(worker_factory, stage):
    current = Batch("old-batch", [make_req("old", rank=64)]) if stage == "merge" else None
    worker, _ = worker_factory([make_req("new")], current, fail_at=f"cancel_{stage}")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker._process_requests())
    assert worker.prefilling_batch is None
    state = worker._get_state_for_reporter()
    assert state["prefilling_request_count"] == state["prefilling_rwpt_tokens"] == 0


@pytest.mark.parametrize("metric, pressure", [
    ("rwpt_active", 113 / 3072), ("token_count_active", 100 / 3072),
    ("queue_length", 1), ("rwpt", 0), ("token_count", 0),
])
def test_prefill_load_reaches_router_and_diagnostics(worker_factory, metric, pressure, monkeypatch):
    monkeypatch.setenv("DEBUG", "0")
    worker, _ = worker_factory([])
    worker.prefilling_batch = Batch("prefilling", [make_req("new")])
    reporter = WorkerStateReporter(worker_id=0, state_getter=worker._get_state_for_reporter)
    state = reporter.get_current_state()
    restored = WorkerState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.prefilling_request_count == 1
    assert restored.prefilling_raw_tokens == 100
    assert restored.prefilling_rwpt_tokens == 113
    message = json.loads(json.dumps(reporter._create_state_message(restored)))
    cache = WorkerStateCache(2)
    cache.update_from_message(message)
    assert cache.get(0).prefilling_rwpt_tokens == 113
    assert cache.get(0).prefilling_raw_tokens == 100
    assert cache.get(0).prefilling_request_count == 1

    manager = DataParallelRouterManager.__new__(DataParallelRouterManager)
    manager.num_workers = 2
    manager.routing_strategy = "adapter-aware"
    manager.router = AdapterAwareRouter(2, RoutingConfig(
        w1=0, w2=0.4, w3=0, load_metric=metric, batch_max_tokens=3072,
    ))
    manager.worker_state_cache = WorkerStateCache(2)
    manager.replica_manager = None
    manager.state_receiver = SimpleNamespace(
        recv_json=AsyncMock(side_effect=[message, asyncio.CancelledError()]),
    )
    asyncio.run(manager._process_worker_states())
    received = manager.worker_state_cache.get(0)
    assert received.prefilling_rwpt_tokens == 113
    assert received.prefilling_raw_tokens == 100
    assert received.prefilling_request_count == 1
    assert manager.router.calculate_score(0, "/adapters/rank16") == pytest.approx(-0.4 * pressure)
    assert manager.router.calculate_score(1, "/adapters/rank16") == 0
    debug = manager._build_routing_debug_record(1, {"adapter_dir": "/adapters/rank16"})
    assert debug["workers"]["0"]["load_pressure"] == pytest.approx(pressure)
    assert debug["workers"]["0"]["prefilling_rwpt_tokens"] == 113


def test_legacy_state_messages_default_prefill_load_to_zero():
    message = {"type": "worker_state", "worker_id": 0, "queue_length": 2}
    cache = WorkerStateCache(1)
    cache.update_from_message(message)
    for state in (WorkerState.from_dict(message), cache.get(0)):
        assert state.prefilling_request_count == 0
        assert state.prefilling_raw_tokens == 0
        assert state.prefilling_rwpt_tokens == 0
