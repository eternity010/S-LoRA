import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slora.server.router.gpu_worker import GPUWorker


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
