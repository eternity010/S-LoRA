import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slora.server.router.gpu_worker import GPUWorker


def test_worker_state_reports_rank_weighted_active_prompt_tokens():
    worker = GPUWorker.__new__(GPUWorker)
    worker.adapter_cache = {}
    worker.req_queue = SimpleNamespace(waiting_req_list=[])
    worker.current_batch = SimpleNamespace(reqs=[
        SimpleNamespace(prompt_ids=list(range(100)), adapter_dir="adapter-a"),
        SimpleNamespace(prompt_ids=list(range(200)), adapter_dir="adapter-b"),
    ])
    worker.lora_ranks = {"adapter-a": 16, "adapter-b": 64}
    worker._hidden_dim = 4096
    worker.model_rpc = None
    worker._profiled_alpha = 0.1
    worker._compute_top_k_rwpt_adapters = lambda: []

    state = worker._get_state_for_reporter()

    gamma = 2.0 / (3.0 * 4096)
    expected = int(100 * (1.0 + gamma * 16) + 200 * (1.0 + gamma * 64))
    assert state["active_rwpt_tokens"] == expected
    assert state["current_batch_prompt_tokens"] == 300
    assert state["active_decode_seqs"] == 2


@pytest.mark.asyncio
async def test_cooperative_checkpoint_reports_due_state_and_yields():
    worker = GPUWorker.__new__(GPUWorker)
    worker.state_reporter = SimpleNamespace(
        report_if_due=AsyncMock(return_value=True)
    )
    peer_ran = False

    async def peer_task():
        nonlocal peer_ran
        peer_ran = True

    peer = asyncio.create_task(peer_task())
    await worker._cooperative_checkpoint()
    await peer

    worker.state_reporter.report_if_due.assert_awaited_once_with()
    assert peer_ran is True


@pytest.mark.asyncio
async def test_busy_process_loop_does_not_starve_peer_tasks():
    worker = GPUWorker.__new__(GPUWorker)
    worker.worker_id = 0
    worker.req_queue = SimpleNamespace(waiting_req_list=[object()])
    worker.current_batch = object()
    worker._received_count = 0
    worker.state_reporter = None
    worker._send_response = AsyncMock()
    iterations = 0
    observed_iteration = None

    async def process_requests():
        nonlocal iterations
        iterations += 1
        if iterations >= 1000:
            raise asyncio.CancelledError
        return []

    async def peer_task():
        nonlocal observed_iteration
        await asyncio.sleep(0)
        observed_iteration = iterations

    worker._process_requests = process_requests
    await asyncio.gather(worker._process_loop(), peer_task())

    assert observed_iteration is not None
    assert observed_iteration < 100
