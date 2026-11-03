"""CPU regressions for resident-adapter accounting and admission snapshots."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from slora.server.io_struct import Batch, Req
from slora.server.router.gpu_worker import GPUWorker
from slora.server.router.model_infer.infer_adapter import InferAdapter
from slora.server.router.req_queue import ReqQueue
from slora.server.sampling_params import SamplingParams


RANKS = {"a": 16, "b": 64, "c": 32, None: 0}


def request(adapter="a", name="new", length=100, output=100):
    params = SamplingParams(max_new_tokens=output)
    params.stop_sequences = []
    return Req(adapter, name, [1] * length, params)


@pytest.mark.parametrize("resident,current_adapters,new_adapters,extra", [
    ({"b": 256}, ["b"], ["b"], 0),
    ({"b": 256}, [], ["b"], 0),
    ({"a": 64, "b": 256}, ["a"], ["b"], 0),
    ({"a": 64, "b": 256}, [], ["c", "c"], 128),
    ({"a": 64}, ["a"], ["b", "b", "c"], 384),
    ({}, [], ["b", "b"], 256),
    ({}, ["a"], ["a", "b"], 320),
    ({}, [], [None], 0),
])
def test_resident_and_new_adapters_are_each_counted_once(
    resident, current_adapters, new_adapters, extra,
):
    queue = ReqQueue(15000, 3072, 64)
    current_reqs = [request(a, f"old-{i}") for i, a in enumerate(current_adapters)]
    current = Batch("old", current_reqs) if current_reqs else None
    for i, adapter in enumerate(new_adapters):
        queue.append(request(adapter, f"new-{i}"))
    batch = queue.generate_new_batch(current, RANKS, resident_adapter_cells=resident)
    assert len(batch.reqs) == len(new_adapters)
    assert queue.adapter_size == extra
    assert queue.actual_total_adapter_size == sum(resident.values())
    assert not queue.waiting_req_list


@pytest.mark.parametrize("capacity,admitted", [(653, False), (654, False), (655, True), (800, True)])
def test_398_cell_kv_peak_preserves_strict_capacity_boundary(capacity, admitted):
    old = request("b", "old")
    old.output_ids = [42]
    queue = ReqQueue(capacity, 3072, 64)
    queue.append(request("b"))
    batch = queue.generate_new_batch(
        Batch("old", [old]), RANKS, actual_adapter_size=256,
        resident_adapter_cells={"b": 256},
    )
    assert (batch is not None) is admitted
    assert queue.adapter_size == 0
    assert len(queue.waiting_req_list) == (0 if admitted else 1)


def test_idle_cached_adapters_still_consume_capacity():
    queue = ReqQueue(800, 3072, 64)
    queue.append(request("b", length=400, output=200))
    assert queue.generate_new_batch(
        None, RANKS, resident_adapter_cells={"a": 64, "b": 256},
    ) is None
    assert queue.actual_total_adapter_size == 320
    assert queue.adapter_size == 0


def test_legacy_numeric_only_call_keeps_existing_behavior():
    queue = ReqQueue(15000, 3072, 64)
    queue.append(request("b"))
    assert queue.generate_new_batch(None, RANKS, 256) is not None
    assert queue.actual_total_adapter_size == 256
    assert queue.adapter_size == 256


@pytest.mark.parametrize("batch_tokens,max_requests", [(99, 64), (3072, 0)])
def test_snapshot_does_not_bypass_other_admission_limits(batch_tokens, max_requests):
    queue = ReqQueue(15000, batch_tokens, max_requests)
    queue.append(request("b"))
    assert queue.generate_new_batch(None, RANKS, resident_adapter_cells={"b": 256}) is None
    assert len(queue.waiting_req_list) == 1


@pytest.mark.parametrize("resident", [{}, {"a": 64}, {"a": 64, "b": 256}])
def test_infer_adapter_returns_names_and_cells_from_one_snapshot(resident):
    adapter = InferAdapter.__new__(InferAdapter)
    adapter.adapter_dirs = list(resident)
    adapter.a_len = torch.tensor(list(resident.values()), dtype=torch.long, device="cpu")
    adapter.mem_manager = SimpleNamespace(tot_size=800, cache_size=800, can_use_mem_size=400)
    info = adapter.get_lora_memory_usage()
    assert info["adapter_cells_by_dir"] == resident
    assert info["adapter_cells"] == list(resident.values())
    assert info["lora_used_cells"] == sum(resident.values())


@pytest.fixture
def worker_factory(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    def build(resident=None, waiting=None, current=None):
        worker = GPUWorker.__new__(GPUWorker)
        worker.worker_id = 0
        worker.args = SimpleNamespace(no_lora=False, eos_id=2)
        worker.lora_ranks = dict(RANKS)
        worker.adapter_cache = {a: True for a in resident or {}}
        worker.resident_adapter_cells = None if resident is None else dict(resident)
        worker.actual_adapter_memory_usage = sum((resident or {}).values())
        worker.actual_adapter_size = worker.actual_adapter_memory_usage
        worker._adapter_memory_lock = None
        worker._adapter_memory_retry_at = 0.0
        worker.current_batch = current
        worker.prefilling_batch = None
        worker.req_queue = ReqQueue(15000, 3072, 64)
        for req in waiting or []:
            worker.req_queue.append(req)
        worker.state_reporter = None
        storage = dict(resident or {})

        async def memory():
            return {
                "total_cells": 15000,
                "adapter_cells": list(storage.values()),
                "adapter_cells_by_dir": dict(storage),
            }

        async def load(adapters):
            for adapter in adapters:
                if adapter is not None:
                    storage[adapter] = RANKS[adapter] * 4

        async def preload(adapter, _protection):
            await load({adapter})
            return {"success": True, "error": None}

        async def clear():
            storage.clear()

        async def prefill(_batch_id):
            assert not worker._get_adapter_memory_lock().locked()
            return {r.request_id: (42, {}) for r in worker.prefilling_batch.reqs}

        async def decode(batch):
            assert not worker._get_adapter_memory_lock().locked()
            return {r.request_id: (42, {}) for r in batch.reqs}

        worker._infer_batch = AsyncMock(side_effect=decode)
        worker.model_rpc = SimpleNamespace(
            check_lora_memory=AsyncMock(side_effect=memory),
            load_adapters=AsyncMock(side_effect=load),
            init_batch=AsyncMock(),
            prefill_batch=AsyncMock(side_effect=prefill),
            merge_batch=AsyncMock(),
            remove_batch=AsyncMock(),
            filter_batch=AsyncMock(),
            preload_adapter=AsyncMock(side_effect=preload),
            clear_all_adapters=AsyncMock(side_effect=clear),
            trigger_threshold_eviction=AsyncMock(return_value={"evicted": False}),
        )
        return worker, storage

    return build


@pytest.mark.asyncio
@pytest.mark.parametrize("info", [
    None, {"adapter_cells": [64]},
    {"adapter_cells": [64], "adapter_cells_by_dir": {"a": 256}},
    {"adapter_cells": [-1], "adapter_cells_by_dir": {"a": -1}},
])
async def test_missing_or_inconsistent_snapshots_cannot_enable_admission(worker_factory, info):
    worker, _ = worker_factory({"a": 64}, [request()])
    worker.model_rpc.check_lora_memory = AsyncMock(return_value=info)
    assert await worker._update_actual_adapter_usage() is False
    assert worker.resident_adapter_cells is None
    batch, error = await worker._prepare_new_batch()
    assert batch is error is None
    assert len(worker.req_queue.waiting_req_list) == 1
    worker.model_rpc.load_adapters.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_refresh_retains_active_execution_and_recovers(worker_factory):
    old = request("a", "old")
    worker, storage = worker_factory(None, [request("b")], Batch("old", [old]))
    storage["a"] = 64
    original_query = worker.model_rpc.check_lora_memory
    worker.model_rpc.check_lora_memory = AsyncMock(return_value=None)
    responses = await worker._process_requests()
    assert [r["request_id"] for r in responses] == ["old"]
    assert len(old.output_ids) == 1
    assert len(worker.req_queue.waiting_req_list) == 1
    worker.model_rpc.init_batch.assert_not_awaited()
    await worker._process_requests()
    worker.model_rpc.check_lora_memory.assert_awaited_once()

    worker.model_rpc.check_lora_memory = original_query
    worker._adapter_memory_retry_at = 0
    await worker._process_requests()
    assert not worker.req_queue.waiting_req_list
    assert worker.resident_adapter_cells == {"a": 64, "b": 256}
    assert worker.actual_adapter_memory_usage == 320


@pytest.mark.asyncio
async def test_worker_uses_gpu_snapshot_instead_of_stale_cache_directory(worker_factory):
    worker, _ = worker_factory({"a": 64}, [request("b")])
    worker.adapter_cache["b"] = True
    await worker._process_requests()
    assert worker.req_queue.adapter_size == 256
    assert worker.resident_adapter_cells == {"a": 64, "b": 256}
    assert worker.actual_adapter_size == worker.actual_adapter_memory_usage == 320


@pytest.mark.asyncio
async def test_load_triggered_eviction_replaces_resident_snapshot(worker_factory):
    worker, storage = worker_factory({"a": 64}, [request("b")])

    async def load_with_eviction(_adapters):
        storage.clear()
        storage["b"] = 256

    worker.model_rpc.load_adapters.side_effect = load_with_eviction
    await worker._process_requests()
    assert worker.resident_adapter_cells == {"b": 256}
    assert worker.actual_adapter_memory_usage == 256
    worker.req_queue._init_cache_list(None, RANKS, resident_adapter_cells=worker.resident_adapter_cells)
    assert worker.req_queue._can_add_new_req(request("a"), RANKS)
    assert worker.req_queue.adapter_size == 64


@pytest.mark.asyncio
async def test_failed_post_load_refresh_blocks_only_later_admission(worker_factory):
    worker, _ = worker_factory({}, [request()])
    worker.model_rpc.check_lora_memory = AsyncMock(side_effect=RuntimeError("query failed"))
    await worker._process_requests()
    assert worker.current_batch is not None
    assert worker.resident_adapter_cells is None
    worker.req_queue.append(request("b", "later"))
    await worker._process_requests()
    assert len(worker.req_queue.waiting_req_list) == 1
    assert len(worker.current_batch.reqs[0].output_ids) == 3
    worker.model_rpc.load_adapters.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["no_lora", "bmm", "no_mem_pool"])
async def test_modes_without_pooled_lora_keep_legacy_admission(worker_factory, mode):
    worker, _ = worker_factory(None, [request()])
    setattr(worker.args, mode, True)
    batch, error = await worker._prepare_new_batch()
    assert error is None and batch is not None


@pytest.mark.asyncio
async def test_partial_load_failure_refreshes_residency_before_error_cleanup(worker_factory):
    worker, storage = worker_factory({}, [request("b")])

    async def fail_after_allocation(_adapters):
        storage["b"] = 256
        raise RuntimeError("partial load failure")

    worker.model_rpc.load_adapters.side_effect = fail_after_allocation
    responses = await worker._process_requests()
    assert len(responses) == 1 and not responses[0]["success"]
    assert worker.resident_adapter_cells == {"b": 256}
    assert worker.actual_adapter_memory_usage == 256
    worker.model_rpc.init_batch.assert_not_awaited()
    assert worker.prefilling_batch is None


@pytest.mark.asyncio
async def test_preload_refreshes_snapshot_even_when_backend_reports_failure(worker_factory):
    worker, storage = worker_factory({"a": 64})

    async def partial_preload(*_args):
        storage.clear()
        storage["b"] = 256
        return {"success": False, "error": "partial failure"}

    worker.model_rpc.preload_adapter.side_effect = partial_preload
    result = await worker.handle_preload_adapter("b")
    assert not result["success"]
    assert worker.resident_adapter_cells == {"b": 256}
    assert worker.actual_adapter_memory_usage == 256


@pytest.mark.asyncio
async def test_eviction_failure_after_mutation_refreshes_snapshot(worker_factory):
    worker, storage = worker_factory({"a": 64, "b": 256})

    async def partial_eviction(**_kwargs):
        storage.pop("b")
        raise RuntimeError("partial eviction")

    worker.model_rpc.trigger_threshold_eviction.side_effect = partial_eviction
    with pytest.raises(RuntimeError, match="partial eviction"):
        await worker._evict_adapters()
    assert worker.resident_adapter_cells == {"a": 64}
    assert worker.actual_adapter_memory_usage == 64


@pytest.mark.asyncio
async def test_clear_resets_both_usage_fields_and_residency(worker_factory):
    worker, storage = worker_factory({"a": 64})
    result = await worker.reset_adapter_cache()
    assert result["success"]
    assert storage == worker.resident_adapter_cells == {}
    assert worker.actual_adapter_memory_usage == worker.actual_adapter_size == 0


@pytest.mark.asyncio
async def test_clear_does_not_publish_empty_snapshot_if_refresh_fails(worker_factory):
    worker, _ = worker_factory({"a": 64})
    worker.model_rpc.check_lora_memory = AsyncMock(side_effect=RuntimeError("query failed"))
    await worker.reset_adapter_cache()
    assert worker.resident_adapter_cells is None
    assert worker.actual_adapter_memory_usage == 64


@pytest.mark.asyncio
async def test_clear_is_rejected_while_batch_is_in_flight(worker_factory):
    worker, _ = worker_factory({"a": 64}, current=Batch("old", [request()]))
    result = await worker.reset_adapter_cache()
    assert not result["success"]
    worker.model_rpc.clear_all_adapters.assert_not_awaited()
    assert worker.resident_adapter_cells == {"a": 64}


@pytest.mark.asyncio
async def test_admission_waits_for_in_progress_preload(worker_factory):
    worker, storage = worker_factory({}, [request("b")])
    entered, release = asyncio.Event(), asyncio.Event()

    async def preload(*_args):
        storage["b"] = 256
        entered.set()
        await release.wait()
        return {"success": True}

    worker.model_rpc.preload_adapter.side_effect = preload
    task = asyncio.create_task(worker.handle_preload_adapter("b"))
    await asyncio.wait_for(entered.wait(), 1)
    assert worker.resident_adapter_cells is None
    admission = asyncio.create_task(worker._prepare_new_batch())
    await asyncio.sleep(0)
    assert not admission.done()
    assert len(worker.req_queue.waiting_req_list) == 1
    release.set()
    await asyncio.wait_for(task, 1)
    batch, error = await asyncio.wait_for(admission, 1)
    assert error is None and batch is not None
    assert worker.req_queue.adapter_size == 0


@pytest.mark.asyncio
async def test_preload_waits_until_admitted_batch_is_registered(worker_factory):
    worker, storage = worker_factory({}, [request()])
    entered, release = asyncio.Event(), asyncio.Event()

    async def load(_adapters):
        storage["a"] = 64
        entered.set()
        await release.wait()

    worker.model_rpc.load_adapters.side_effect = load
    admission = asyncio.create_task(worker._prepare_new_batch())
    await asyncio.wait_for(entered.wait(), 1)
    preload = asyncio.create_task(worker.handle_preload_adapter("b"))
    await asyncio.sleep(0)
    worker.model_rpc.preload_adapter.assert_not_awaited()
    assert worker.prefilling_batch is not None
    release.set()
    await asyncio.wait_for(admission, 1)
    await asyncio.wait_for(preload, 1)
    worker.model_rpc.init_batch.assert_awaited_once()
    assert worker.resident_adapter_cells == {"a": 64, "b": 256}


@pytest.mark.asyncio
async def test_cancelled_load_refreshes_snapshot_and_releases_lock(worker_factory):
    worker, storage = worker_factory({}, [request()])
    entered = asyncio.Event()

    async def load(_adapters):
        storage["a"] = 64
        entered.set()
        await asyncio.Event().wait()

    worker.model_rpc.load_adapters.side_effect = load
    task = asyncio.create_task(worker._process_requests())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not worker._get_adapter_memory_lock().locked()
    assert worker.prefilling_batch is None
    assert worker.resident_adapter_cells == {"a": 64}
