"""Deterministic dispatch/report interleavings, without GPU execution."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slora.server.router.dp_manager import DataParallelRouterManager
from slora.server.router.dispatch_tracker import DispatchTracker
from slora.server.router.gpu_worker import GPUWorker
from slora.server.router.req_queue import ReqQueue
from slora.server.router.worker_state import WorkerState
from slora.server.router.worker_state_reporter import WorkerStateReporter
from slora.server.router.worker_state_cache import WorkerStateCache
from slora.server.router.validation_log import ValidationLog
from routing_experiment.gpu_validation import analyze_dispatch_evidence, wait_for_dispatch_drain


@pytest.fixture
def manager():
    args = argparse.Namespace(num_workers=2, gpu_ids='0,1', routing_strategy='adapter-aware',
                              load_metric='rwpt_active', lora_dirs=[])
    result = DataParallelRouterManager(args, 60000, 60001)
    result.router.set_adapter_ranks({'adapter': 64})
    result.request_senders = [SimpleNamespace(send_json=AsyncMock()) for _ in range(2)]
    return result


def request(name='a'):
    return dict(request_id=name, adapter_dir='adapter', prompt_ids=list(range(100)),
                sampling_params={})


async def report(manager, sequence, ack, worker_id=0, **fields):
    message = dict(type='worker_state', worker_id=worker_id, report_seq=sequence,
                   routing_epoch=manager._routing_epoch, ack_dispatch_seq=ack, **fields)
    manager.state_receiver = SimpleNamespace(
        recv_json=AsyncMock(side_effect=[message, asyncio.CancelledError()]))
    await manager._process_worker_states()


@pytest.fixture
def worker(manager, monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    result = GPUWorker.__new__(GPUWorker)
    result.worker_id = 0
    result.args = manager.args
    result._routing_epoch = manager._routing_epoch
    result._ack_dispatch_seq = 0
    result.req_queue = ReqQueue(4096, 3072, 64)
    result.current_batch = result.prefilling_batch = None
    result.adapter_cache = {}
    result.lora_ranks = {'adapter': 64}
    result._profiled_rank_beta = .00845
    result._profiled_alpha = 0
    result._hidden_dim = 4096
    result.model_rpc = None
    return result


def test_report_sampled_before_dispatch_does_not_erase_new_load(manager):
    async def run():
        # The snapshot predates the dispatch even though it arrives afterwards.
        old = dict(type='worker_state', worker_id=0, report_seq=1, queue_length=0,
                   routing_epoch=getattr(manager, '_routing_epoch', None), ack_dispatch_seq=0)
        manager.router.select_worker = lambda _adapter: 0
        await manager.route_request(request())
        assert manager.router.worker_states[0].queue_length == 1
        manager.state_receiver = SimpleNamespace(
            recv_json=AsyncMock(side_effect=[old, asyncio.CancelledError()]))
        await manager._process_worker_states()
        state = manager.router.worker_states[0]
        assert state.queue_length == 1
        assert state.pending_prefill_tokens == 154
        assert state.optimistic_request_count == 1

    asyncio.run(run())


def test_partial_confirmation_and_fast_completion(manager):
    async def run():
        manager.router.select_worker = lambda _adapter: 0
        await manager.route_request(request('a'))
        await manager.route_request(request('b'))
        await report(manager, 1, 1, queue_length=1, active_rwpt_tokens=154)
        state = manager.router.worker_states[0]
        assert (state.queue_length, state.active_rwpt_tokens, state.pending_prefill_tokens) == (2, 154, 154)
        assert list(manager._dispatch_tracker.pending[0]) == [2]
        assert manager.worker_state_cache.get(0) is state
        assert manager._dispatch_tracker.reported[0].pending_prefill_tokens == 0
        # Both requests can finish before the next report is sent.
        await report(manager, 2, 2, queue_length=0)
        assert manager.router.worker_states[0].queue_length == 0
        assert not manager._dispatch_tracker.pending[0]

    asyncio.run(run())


@pytest.mark.parametrize('corruption', [
    {'report_seq': 1}, {'report_seq': 0}, {'routing_epoch': 'old-run'},
    {'routing_epoch': None}, {'ack_dispatch_seq': None}, {'ack_dispatch_seq': 2},
    {'ack_dispatch_seq': -1}, {'ack_dispatch_seq': True},
])
def test_stale_or_invalid_report_does_not_replace_state(manager, corruption):
    async def run():
        manager.router.select_worker = lambda _adapter: 0
        await manager.route_request(request())
        await report(manager, 1, 0, queue_length=3)
        old = manager.router.worker_states[0]
        message = dict(type='worker_state', worker_id=0, report_seq=2,
                       routing_epoch=manager._routing_epoch, ack_dispatch_seq=0, queue_length=0)
        message.update(corruption)
        manager.state_receiver = SimpleNamespace(
            recv_json=AsyncMock(side_effect=[message, asyncio.CancelledError()]))
        await manager._process_worker_states()
        assert manager.router.worker_states[0] is old
        assert manager.worker_state_cache.get(0) is old
        assert list(manager._dispatch_tracker.pending[0]) == [1]

    asyncio.run(run())


def test_ack_watermark_cannot_go_backwards(manager):
    async def run():
        manager.router.select_worker = lambda _adapter: 0
        await manager.route_request(request())
        await report(manager, 1, 1)
        old = manager.router.worker_states[0]
        await report(manager, 2, 0, queue_length=20)
        assert manager.router.worker_states[0] is old

    asyncio.run(run())


def test_ack_before_send_returns_does_not_readd_reservation(manager):
    async def send(message):
        assert manager.router.worker_states[0].optimistic_request_count == 1
        await report(manager, 1, message['dispatch_seq'], queue_length=1, pending_prefill_tokens=154)

    manager.router.select_worker = lambda _adapter: 0
    manager.request_senders[0].send_json.side_effect = send
    asyncio.run(manager.route_request(request()))
    state = manager.router.worker_states[0]
    assert state.queue_length == 1 and state.pending_prefill_tokens == 154
    assert state.optimistic_request_count == 0


def test_definite_send_failure_rolls_back_and_keeps_sequence_gap(manager):
    manager.router.select_worker = lambda _adapter: 0
    manager.request_senders[0].send_json.side_effect = [zmq.Again(), None]
    asyncio.run(manager.route_request(request()))
    assert list(manager._dispatch_tracker.pending[0]) == [2]
    assert manager.router.worker_states[0].pending_prefill_tokens == 154
    asyncio.run(report(manager, 1, 2, queue_length=1, pending_prefill_tokens=154))
    assert not manager._dispatch_tracker.pending[0]


def test_all_definite_send_failures_leave_no_reservations(manager):
    manager.router.select_worker = lambda _adapter: 0
    manager.request_senders[0].send_json.side_effect = zmq.Again()
    with pytest.raises(RuntimeError, match='ZMQ timeout'):
        asyncio.run(manager.route_request(request()))
    assert not manager._dispatch_tracker.pending[0]
    assert manager.router.worker_states[0].queue_length == 0


@pytest.mark.parametrize('error', [RuntimeError('unknown delivery'), asyncio.CancelledError()])
def test_uncertain_delivery_is_not_retried_or_forgotten(manager, error):
    manager.router.select_worker = lambda _adapter: 0
    manager.request_senders[0].send_json.side_effect = error
    with pytest.raises(type(error)):
        asyncio.run(manager.route_request(request()))
    assert manager.request_senders[0].send_json.await_count == 1
    assert manager.router.worker_states[0].optimistic_request_count == 1
    assert not manager.router.worker_states[0].is_healthy
    with pytest.raises(RuntimeError, match='unhealthy Worker'):
        asyncio.run(manager.route_request(request('b')))
    asyncio.run(report(manager, 1, 1))
    assert manager.router.worker_states[0].is_healthy
    assert not manager._dispatch_tracker.pending[0]


def test_diagnostic_failures_never_replay_a_sent_request(manager):
    manager._validation_log = Mock(side_effect=OSError('disk full'))
    manager._write_routing_debug_record = Mock(side_effect=OSError('disk full'))
    manager.router.select_worker = lambda _adapter: 0
    asyncio.run(manager.route_request(request()))
    assert manager.request_senders[0].send_json.await_count == 1
    assert manager.stats['successful_requests'] == 1


def test_metric_change_recomposes_pending_without_mutating_raw_snapshot(manager, tmp_path):
    manager.router.select_worker = lambda _adapter: 0
    asyncio.run(manager.route_request(request()))
    config = tmp_path / 'routing.json'
    config.write_text(json.dumps({'load_metric': 'token_count_active'}))
    manager._write_stats_file = Mock()
    manager._check_config_update(str(config))
    assert manager.router.worker_states[0].pending_raw_tokens == 100
    assert manager._dispatch_tracker.reported[0].pending_raw_tokens == 0
    assert manager.router.worker_states[0].optimistic_request_count == 1


def test_worker_snapshot_carries_enqueue_progress_through_serialization(manager, worker):
    manager.router.select_worker = lambda _adapter: 0
    asyncio.run(manager.route_request(request()))
    payload = manager.request_senders[0].send_json.call_args.args[0]
    worker._enqueue_routed_request(payload)
    reporter = WorkerStateReporter(0, state_getter=worker._get_state_for_reporter)
    message = reporter._create_state_message(reporter.get_current_state())
    cache = WorkerStateCache(2)
    cache.update_from_message(json.loads(json.dumps(message)))
    restored = WorkerState.from_dict(cache.get(0).to_dict())
    assert restored.routing_epoch == manager._routing_epoch and restored.ack_dispatch_seq == 1
    assert restored.queue_length == 1 and restored.pending_prefill_tokens == 154
    # A previously sampled snapshot must keep its old watermark.
    frozen = reporter.get_current_state()
    worker._enqueue_routed_request(dict(payload, request_id='b', dispatch_seq=2))
    assert reporter._create_state_message(frozen)['ack_dispatch_seq'] == 1


@pytest.mark.parametrize('fields', [dict(routing_epoch='old'), dict(dispatch_seq=0),
                                  dict(dispatch_seq=None), dict(dispatch_seq=True)])
def test_worker_rejects_invalid_protocol_without_advancing(manager, worker, fields):
    payload = dict(request(), routing_epoch=manager._routing_epoch, dispatch_seq=1)
    payload.update(fields)
    with pytest.raises(ValueError):
        worker._enqueue_routed_request(payload)
    assert worker._ack_dispatch_seq == 0 and not worker.req_queue.waiting_req_list


def test_worker_rejects_duplicate_dispatch_without_enqueuing_twice(manager, worker):
    payload = dict(request(), routing_epoch=manager._routing_epoch, dispatch_seq=1)
    worker._enqueue_routed_request(payload)
    with pytest.raises(ValueError):
        worker._enqueue_routed_request(payload)
    assert len(worker.req_queue.waiting_req_list) == 1 and worker._ack_dispatch_seq == 1


def test_explicit_worker_rejection_can_retire_reservation(manager, worker):
    manager.router.select_worker = lambda _adapter: 0
    bad = request()
    bad.pop('prompt_ids')
    asyncio.run(manager.route_request(bad))
    payload = manager.request_senders[0].send_json.call_args.args[0]
    with pytest.raises(KeyError):
        worker._enqueue_routed_request(payload)
    assert worker._ack_dispatch_seq == 1 and not worker.req_queue.waiting_req_list
    asyncio.run(report(manager, 1, worker._ack_dispatch_seq))
    assert not manager._dispatch_tracker.pending[0]


def test_receive_loop_returns_terminal_error_for_rejected_request(manager, worker):
    payload = dict(request(), routing_epoch=manager._routing_epoch, dispatch_seq=1)
    payload.pop('prompt_ids')
    worker._received_count = 0
    worker._send_response = AsyncMock()
    worker.request_receiver = SimpleNamespace(
        recv_json=AsyncMock(side_effect=[payload, asyncio.CancelledError()]))
    asyncio.run(worker._receive_requests_loop())
    response = worker._send_response.call_args.args[0]
    assert response['request_id'] == 'a'
    assert response['finished'] and not response['success']
    assert worker._ack_dispatch_seq == 1 and not worker.req_queue.waiting_req_list


def test_control_commands_do_not_advance_dispatch_acknowledgement(worker):
    worker.reset_adapter_cache = AsyncMock(return_value={'success': True})
    worker._send_response = AsyncMock()
    worker.request_receiver = SimpleNamespace(recv_json=AsyncMock(side_effect=[
        dict(type='reset_cache', request_id='reset'), asyncio.CancelledError()]))
    asyncio.run(worker._receive_requests_loop())
    worker.reset_adapter_cache.assert_awaited_once()
    assert worker._ack_dispatch_seq == 0


def test_failed_sampling_cannot_acknowledge_pending_dispatch(manager):
    manager.router.select_worker = lambda _adapter: 0
    asyncio.run(manager.route_request(request()))
    reporter = WorkerStateReporter(0, state_getter=Mock(side_effect=RuntimeError('bad snapshot')))
    message = reporter._create_state_message(reporter.get_current_state())
    assert message['is_healthy'] is False
    manager.state_receiver = SimpleNamespace(
        recv_json=AsyncMock(side_effect=[message, asyncio.CancelledError()]))
    asyncio.run(manager._process_worker_states())
    assert manager.router.worker_states[0].optimistic_request_count == 1


def test_concurrent_callers_keep_dispatch_and_send_order(manager):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        sequences = []

        async def send(message):
            sequences.append(message['dispatch_seq'])
            if message['dispatch_seq'] == 1:
                started.set()
                await release.wait()

        manager.router.select_worker = lambda _adapter: 0
        manager.request_senders[0].send_json.side_effect = send
        first = asyncio.create_task(manager.route_request(request('a')))
        await started.wait()
        second = asyncio.create_task(manager.route_request(request('b')))
        await asyncio.sleep(0)
        assert sequences == [1]
        release.set()
        await asyncio.gather(first, second)
        assert sequences == [1, 2]

    asyncio.run(run())


def test_stats_and_cache_reset_do_not_reset_dispatch_progress(manager):
    manager.router.select_worker = lambda _adapter: 0
    asyncio.run(manager.route_request(request()))
    manager._reset_experiment_stats()
    manager._clear_cached_adapter_observations()
    assert manager._dispatch_tracker.issued[0] == 1
    assert list(manager._dispatch_tracker.pending[0]) == [1]


def test_new_server_epoch_rejects_previous_server_reports():
    tracker = DispatchTracker(1, 'new')
    assert not tracker.accept_report(WorkerState(0, routing_epoch='old', ack_dispatch_seq=0, report_seq=100))
    assert tracker.accept_report(WorkerState(0, routing_epoch='new', ack_dispatch_seq=0, report_seq=1))


def test_worker_acknowledgements_are_independent(manager):
    async def run():
        selected = iter([0, 1])
        manager.router.select_worker = lambda _adapter: next(selected)
        await manager.route_request(request('a'))
        await manager.route_request(request('b'))
        await report(manager, 1, 1, worker_id=0)
        assert not manager._dispatch_tracker.pending[0]
        assert list(manager._dispatch_tracker.pending[1]) == [1]
        assert manager.router.worker_states[1].optimistic_request_count == 1

    asyncio.run(run())


def test_legacy_worker_cannot_silently_disable_accounting(manager):
    async def run():
        manager.state_receiver = SimpleNamespace(recv_json=AsyncMock(side_effect=[
            dict(type='worker_state', worker_id=0, report_seq=1), asyncio.CancelledError()]))
        await manager._process_worker_states()
        assert not manager.router.worker_states[0].is_healthy
        manager.router.select_worker = lambda _adapter: 0
        with pytest.raises(RuntimeError, match='unhealthy Worker'):
            await manager.route_request(request())
        await report(manager, 1, 0)
        assert manager.router.worker_states[0].is_healthy

    asyncio.run(run())


@pytest.fixture
def dispatch_evidence(manager, worker, tmp_path):
    async def run():
        manager._validation_log = ValidationLog(tmp_path, 'router')
        worker._validation_log = ValidationLog(tmp_path, 0)
        manager.router.select_worker = lambda _adapter: 0
        await manager.route_request(request('a'))
        await report(manager, 1, 0)  # Does not yet cover the dispatched request.
        await manager.route_request(request('b'))
        payloads = [call.args[0] for call in manager.request_senders[0].send_json.call_args_list]
        worker._enqueue_routed_request(payloads[0])
        worker._write_validation_state('queued')
        await report(manager, 2, 1, queue_length=1, pending_prefill_tokens=154, pending_raw_tokens=100)
        worker._enqueue_routed_request(payloads[1])
        worker._write_validation_state('queued')
        await report(manager, 3, 2, queue_length=2, pending_prefill_tokens=308, pending_raw_tokens=200)
        worker.req_queue.waiting_req_list.clear()
        worker._write_validation_state('drained')
        await report(manager, 4, 2)
        await report(manager, 1, 0, worker_id=1)

    asyncio.run(run())
    return tmp_path


def test_dispatch_evidence_checks_actual_router_and_worker_logs(dispatch_evidence):
    summary = analyze_dispatch_evidence(dispatch_evidence, 2, 2)
    assert not summary['errors'] and not summary['missing_coverage'], summary
    assert summary['sent_requests'] == summary['received_requests'] == 2
    assert summary['accepted_reports'] == 5
    assert summary['noncovering_reports_preserved'] == 2
    wait_for_dispatch_drain(dispatch_evidence, 2, 2, timeout=0)


@pytest.mark.parametrize('corruption', ['overlay', 'ack', 'epoch', 'receive_sequence', 'snapshot'])
def test_dispatch_evidence_detects_causal_corruption(dispatch_evidence, corruption):
    path = dispatch_evidence / ('worker_0.jsonl' if corruption in ('receive_sequence', 'snapshot')
                                else 'worker_router.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == 'receive_sequence':
        received = [row for row in rows if row['event'] == 'received']
        received[1]['dispatch_seq'] = received[0]['dispatch_seq']
    elif corruption == 'snapshot':
        next(row for row in rows if row['event'] == 'lifecycle')['ack_dispatch_seq'] = 0
    else:
        row = next(row for row in rows if row['event'] == 'worker_report')
        if corruption == 'overlay':
            row['effective_queue_length'] = 0
        elif corruption == 'ack':
            row['ack_dispatch_seq'] = 100
        else:
            row['routing_epoch'] = 'old'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert analyze_dispatch_evidence(dispatch_evidence, 2, 2)['errors']


def test_dispatch_drain_timeout_does_not_claim_success(dispatch_evidence):
    path = dispatch_evidence / 'worker_router.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row['event'] != 'worker_report' or row['report_seq'] < 3]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(TimeoutError, match='did not confirm'):
        wait_for_dispatch_drain(dispatch_evidence, 2, 2, timeout=0)
    assert analyze_dispatch_evidence(dispatch_evidence, 2, 2)['missing_coverage']
