"""CPU checks for the GPU validation protocol and process ownership."""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import run_routing_comparison as cli
from routing_experiment.gpu_validation import (
    analyze_validation, build_validation_phases, run_validation_phases,
)
from routing_experiment.runner import ExperimentRunner
from slora.server.io_struct import Batch, Req
from slora.server.router.req_queue import ReqQueue
from slora.server.router.validation_log import ValidationLog
from slora.server.sampling_params import SamplingParams


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(ExperimentRunner, 'RUNTIME_LOCK_FILE', str(tmp_path / 'runtime.lock'))
    return ExperimentRunner(output_dir=str(tmp_path), benchmarks_dir='.')


def test_gpu_overrides_preserve_old_defaults_and_override_explicit_suites():
    old = ExperimentRunner.resolve_suite_configs('dp-realtrace-rwpt-active-8rps-modelctx2048')[0]
    assert (old.gpu_ids, old.num_workers) == ('1,2,3', 3)
    override = ExperimentRunner.resolve_suite_configs('dp-gpu-fix-validation', ' 0, 2 ')[0]
    assert (override.gpu_ids, override.num_workers) == ('0,2', 2)
    assert override.to_server_args()[override.to_server_args().index('--gpu-ids') + 1] == '0,2'


@pytest.mark.parametrize('gpu_ids,workers', [('0,1', 3), ('0,0', 2), ('0,x', 2), ('', None)])
def test_invalid_gpu_overrides_fail_before_startup(gpu_ids, workers):
    with pytest.raises(ValueError, match='gpu_ids'):
        ExperimentRunner.resolve_suite_configs('dp-gpu-fix-validation', gpu_ids, workers)


def test_cli_dry_run_does_not_construct_runner_or_create_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, 'argv', ['run_routing_comparison.py', '--suite', 'dp-gpu-fix-validation',
                                    '--gpu-ids', '0', '--dry-run', '--output-dir', str(tmp_path / 'absent')])
    monkeypatch.setattr(ExperimentRunner, '__init__', Mock(side_effect=AssertionError('constructed runner')))
    cli.main()
    config = json.loads(capsys.readouterr().out)[0]
    assert (config['gpu_ids'], config['num_workers']) == ('0', 1)
    assert not (tmp_path / 'absent').exists()


def test_cli_forwards_explicit_environment_overrides(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(cli, 'ExperimentRunner', factory)
    cli.run_experiments(SimpleNamespace(
        output_dir='unused', model_setting='Real', host='localhost', port=38000,
        benchmarks_dir='.', model_dir=None, adapter_dir=None, debug=False,
        gpu_ids='0,1', num_workers=2, validation_log=True, suite='dp-gpu-fix-validation', resume=False,
    ))
    assert factory.call_args.kwargs['gpu_ids'] == '0,1'
    assert factory.call_args.kwargs['num_workers'] == 2
    assert factory.call_args.kwargs['validation_log'] is True


def test_cleanup_never_scans_or_signals_without_owned_group(runner, monkeypatch):
    run = Mock(side_effect=AssertionError('global process scan'))
    kill = Mock(side_effect=AssertionError('unowned process signal'))
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(os, 'kill', kill)
    runner._cleanup_slora_processes()
    run.assert_not_called()
    kill.assert_not_called()


def test_cleanup_only_targets_owned_non_zombie_group(runner, monkeypatch):
    runner._server_pgid = 5000
    uid = os.getuid()
    process_lists = iter([
        f'{uid} 5010 5000 S\n{uid} 6000 6000 S\n{uid + 1} 5011 5000 S\n{uid} 5012 5000 Z\n',
        '',
    ])
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: SimpleNamespace(stdout=next(process_lists)))
    killed = []
    monkeypatch.setattr(os, 'kill', lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr('routing_experiment.runner.time.sleep', lambda _seconds: None)
    runner._cleanup_slora_processes()
    assert killed == [(5010, signal.SIGKILL)]


def test_stale_parent_still_cleans_its_owned_group(runner, monkeypatch):
    runner._server_pgid = 5000
    runner.server_process = Mock(pid=5000)
    runner.server_process.wait.side_effect = ProcessLookupError()
    kill_group = Mock()
    monkeypatch.setattr(os, 'killpg', kill_group)
    monkeypatch.setattr(runner, '_get_slora_pids', lambda: set())
    runner._stop_server()
    kill_group.assert_called_once_with(5000, signal.SIGTERM)
    assert runner._server_pgid is None


@pytest.mark.parametrize('selected,blocked', [('0,1', False), ('1,2', True)])
def test_gpu_preflight_refuses_occupied_selected_devices(monkeypatch, selected, blocked):
    outputs = iter(['0, GPU-a\n1, GPU-b\n2, GPU-c\n', '99, GPU-c\n'])
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: SimpleNamespace(stdout=next(outputs)))
    if blocked:
        with pytest.raises(RuntimeError, match='GPU 2: PID 99'):
            ExperimentRunner._check_gpu_availability(selected)
    else:
        ExperimentRunner._check_gpu_availability(selected)


def test_runtime_guard_refuses_foreign_slora_without_signaling(runner, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: SimpleNamespace(
        stdout='200 200 python -m slora.server.api_server --port 39000\n'))
    monkeypatch.setattr(os, 'kill', Mock(side_effect=AssertionError('foreign kill')))
    with pytest.raises(RuntimeError, match='already running'):
        runner._acquire_runtime_lock()
    assert runner._runtime_lock is None


def test_runtime_lock_blocks_concurrent_runner(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *_args, **_kwargs: SimpleNamespace(stdout=''))
    other = ExperimentRunner(output_dir=str(tmp_path / 'other'))
    runner._acquire_runtime_lock()
    try:
        with pytest.raises(BlockingIOError):
            other._acquire_runtime_lock()
    finally:
        runner._runtime_lock.close()
        runner._runtime_lock = None


def test_occupied_gpu_does_not_clean_runtime_or_launch(runner, monkeypatch):
    config = ExperimentRunner.resolve_suite_configs('dp-realtrace-rwpt-active-8rps-modelctx2048')[0]
    monkeypatch.setattr(runner, '_acquire_runtime_lock', lambda: None)
    monkeypatch.setattr(runner, '_check_gpu_availability', Mock(side_effect=RuntimeError('occupied')))
    cleanup = Mock()
    launch = Mock()
    monkeypatch.setattr(runner, '_cleanup_runtime_state_files', cleanup)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError, match='occupied'):
        runner._start_server(config)
    cleanup.assert_not_called()
    launch.assert_not_called()


def test_suite_failure_is_not_reported_as_success(runner, monkeypatch):
    monkeypatch.setattr(runner, 'run_single_experiment', Mock(side_effect=RuntimeError('incomplete validation')))
    stop = Mock()
    monkeypatch.setattr(runner, '_stop_server', stop)
    with pytest.raises(RuntimeError, match='Suite failed'):
        runner.run_suite('dp-gpu-fix-validation')
    checkpoint = json.loads(runner.checkpoint_file.read_text())
    assert not checkpoint['completed']
    assert stop.called


def adapters(tmp_path):
    result = []
    for rank in (16, 64, 16, 64):
        directory = tmp_path / f'adapter{len(result)}'
        directory.mkdir()
        (directory / 'adapter_config.json').write_text(json.dumps({'r': rank}))
        result.append(('/base', str(directory) + '-0'))
    return result


def test_validation_plan_is_small_deterministic_and_uses_resident_adapter(tmp_path):
    config = ExperimentRunner.resolve_suite_configs('dp-gpu-fix-validation')[0]
    phases = build_validation_phases(config, '/base', adapters(tmp_path))
    assert [name for name, _ in phases] == ['cold_load', 'resident_reuse', 'capacity_boundary']
    assert [len(requests) for _, requests in phases] == [4, 8, 24]
    assert {req.adapter_dir for req in phases[0][1]} == {req.adapter_dir for req in phases[1][1]}
    assert len({req.adapter_dir for req in phases[2][1]}) == 4
    requests = [req for _, group in phases for req in group]
    assert len({req.req_id for req in requests}) == len(requests)
    assert all(req.req_time == 0 and req.prompt_len + req.output_len <= 2048 for req in requests)


def request(name):
    params = SamplingParams(max_new_tokens=100)
    params.stop_sequences = []
    return Req('b', name, [1] * 100, params)


def test_admission_evidence_distinguishes_reuse_and_old_false_rejection(tmp_path):
    queue = ReqQueue(800, 3072, 64)
    queue.validation_callback = ValidationLog(tmp_path, 0)
    old = request('old')
    old.output_ids = [42]
    queue.append(request('new'))
    assert queue.generate_new_batch(Batch('old', [old]), {'b': 64},
                                    resident_adapter_cells={'b': 256}) is not None
    row = json.loads((tmp_path / 'worker_0.jsonl').read_text())
    assert row['resident_cells'] == 256 and row['reserved_cells'] == 0
    assert row['kv_peak'] == 398 and row['kv_budget'] == 544
    assert row['legacy_kv_budget'] == 288 and row['memory_allowed'] is True


def write_complete_evidence(tmp_path):
    log = ValidationLog(tmp_path, 0)
    req = dict(request_id='new', adapter_dir='b', input_len=100, rank=64, beta=.00845, rwpt=154)
    log('received', **req)
    for stage in ('admitted', 'adapters_loaded', 'before_prefill', 'after_prefill', 'active', 'step_end'):
        group = 'active' if stage == 'active' else 'prefilling'
        groups = {name: [] for name in ('waiting', 'prefilling', 'active')}
        if stage == 'step_end':
            log('terminal', request_id='new', success=True)
        else:
            groups[group] = [req]
        values = dict(queue_length=sum(map(len, groups.values())))
        for name, raw, weighted in (
            ('waiting', 'pending_raw_tokens', 'pending_prefill_tokens'),
            ('prefilling', 'prefilling_raw_tokens', 'prefilling_rwpt_tokens'),
            ('active', 'current_batch_prompt_tokens', 'active_rwpt_tokens'),
        ):
            values[raw] = 100 * len(groups[name])
            values[weighted] = 154 * len(groups[name])
        log('lifecycle', stage=stage, requests=groups, reported=values)
    for resident, reserved, peak in (({'b': 256}, {}, 398), ({}, {'b': 256}, 600)):
        budget = 800 - sum(resident.values()) - sum(reserved.values())
        log('admission', request_id='new', adapter_dir='b', required_adapter_ranks={'b': 64},
            resident_adapter_cells=resident, reserved_adapter_cells=reserved,
            resident_cells=sum(resident.values()), reserved_cells=sum(reserved.values()),
            total_cells=800, kv_peak=peak, kv_budget=budget, memory_allowed=peak < budget,
            concurrency_allowed=True, batch_tokens_allowed=True,
            legacy_kv_budget=800 - sum(resident.values()) - 256)
    state = dict(pending_prefill_tokens=100, prefilling_rwpt_tokens=154, active_rwpt_tokens=200,
                 load_pressure=454 / 800)
    (tmp_path / 'route_decisions.jsonl').write_text(json.dumps(
        dict(workers={'0': state}, capacity=800, load_metric='rwpt_active')) + '\n')
    ValidationLog(tmp_path, 'router')(
        'worker_report', report_seq=1, applied_seq=1, source_worker_id=0,
        prefilling_request_count=1, prefilling_rwpt_tokens=154,
        applied_prefilling_request_count=1, applied_prefilling_rwpt_tokens=154,
    )


def test_validation_summary_requires_actual_witnesses(tmp_path):
    write_complete_evidence(tmp_path)
    summary = analyze_validation(tmp_path, 1, 1)
    assert summary['status'] == 'passed', summary
    assert summary['legacy_false_rejection_witnesses'] == 1


@pytest.mark.parametrize('corruption', ['lost', 'double_count', 'budget', 'router', 'prompt_buffer'])
def test_validation_detects_accounting_corruption(tmp_path, corruption):
    write_complete_evidence(tmp_path)
    path = tmp_path / 'worker_0.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == 'lost':
        rows[1]['requests']['prefilling'] = []
    elif corruption == 'double_count':
        rows[1]['requests']['active'] = list(rows[1]['requests']['prefilling'])
    elif corruption == 'budget':
        next(row for row in rows if row['event'] == 'admission')['kv_budget'] += 256
    elif corruption == 'prompt_buffer':
        rows[1]['requests']['prefilling'][0]['prompt_buffer_len'] = 101
    else:
        router = tmp_path / 'worker_router.jsonl'
        row = json.loads(router.read_text())
        row['applied_prefilling_rwpt_tokens'] = 0
        router.write_text(json.dumps(row) + '\n')
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert analyze_validation(tmp_path, 1, 1)['status'] == 'failed'


def test_missing_boundary_witness_is_incomplete_not_passed(tmp_path):
    write_complete_evidence(tmp_path)
    path = tmp_path / 'worker_0.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row['event'] != 'admission']
    for i, row in enumerate(rows, 1):
        row['sequence'] = i
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summary = analyze_validation(tmp_path, 1, 1)
    assert summary['status'] == 'incomplete', summary
    assert 'memory_boundary_rejection' in summary['missing_coverage']


def test_phases_drain_sequentially_and_timeout_preserves_failure_summary(tmp_path, monkeypatch):
    import routing_experiment.gpu_validation as validation
    config = ExperimentRunner.resolve_suite_configs('dp-gpu-fix-validation')[0]
    calls = []

    async def benchmark(requests, request_id_prefix):
        calls.append(request_id_prefix)
        if len(calls) == 2:
            raise asyncio.TimeoutError()
        return [dict(success=True)]

    fake = SimpleNamespace(_current_diagnostics_dir=tmp_path, _async_benchmark=benchmark,
                           _save_request_latency_trace=Mock())
    phases = [(name, [SimpleNamespace(req_id=i, adapter_dir='b', prompt_len=10, output_len=20)])
              for i, name in enumerate(('cold', 'reuse', 'boundary'))]
    monkeypatch.setattr(validation, 'build_validation_phases', lambda *_args: phases)
    with pytest.raises(asyncio.TimeoutError):
        run_validation_phases(fake, config, '/base', [])
    assert len(calls) == 2
    assert fake._save_request_latency_trace.call_count == 1
    assert json.loads((tmp_path / 'validation_summary.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('enabled,suite,expected', [
    (False, 'dp-realtrace-rwpt-active-8rps-modelctx2048', False),
    (True, 'dp-realtrace-rwpt-active-8rps-modelctx2048', True),
    (False, 'dp-gpu-fix-validation', True),
])
def test_validation_logging_is_opt_in_and_scoped_to_server(runner, monkeypatch, enabled, suite, expected, tmp_path):
    runner.validation_log = enabled
    config = ExperimentRunner.resolve_suite_configs(suite)[0]
    if config.workload_type == 'validation':
        (tmp_path / 'config.json').write_text('{}')
        monkeypatch.setattr(runner, '_benchmark_assets', lambda _config: (str(tmp_path), []))
        monkeypatch.setattr('routing_experiment.gpu_validation.build_validation_phases', lambda *_args: [])
    monkeypatch.setenv('SLORA_VALIDATION_DIR', '/foreign/validation')
    monkeypatch.setattr(runner, '_acquire_runtime_lock', lambda: None)
    monkeypatch.setattr(runner, '_check_gpu_availability', lambda _ids: None)
    monkeypatch.setattr(runner, '_cleanup_slora_processes', lambda **_kwargs: None)
    monkeypatch.setattr(runner, '_cleanup_runtime_state_files', lambda: None)
    launch = Mock(return_value=SimpleNamespace(pid=5555))
    monkeypatch.setattr(subprocess, 'Popen', launch)
    try:
        runner._start_server(config)
        env = launch.call_args.kwargs['env']
        if expected:
            assert Path(env['SLORA_VALIDATION_DIR']).is_relative_to(runner.output_dir)
            assert env['SLORA_ROUTING_DEBUG_FILE'].endswith('route_decisions.jsonl')
        else:
            assert 'SLORA_VALIDATION_DIR' not in env
        assert runner._server_pgid == 5555
    finally:
        runner.server_log.close()
