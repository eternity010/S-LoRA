"""Drained GPU correctness phases using the existing request/benchmark path."""

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import asdict
from pathlib import Path


def build_validation_phases(config, base_model, adapter_dirs):
    from trace import Request, dummy_prompt

    ranks = {}
    for _, adapter in adapter_dirs:
        path = Path(re.sub(r'-\d+$', '', adapter)) / 'adapter_config.json'
        with path.open() as handle:
            ranks[adapter] = int(json.load(handle)['r'])
    if len(set(ranks.values())) < 2:
        raise ValueError('GPU fix validation requires adapters with at least two ranks')
    high = max(ranks, key=ranks.get)
    mixed = sorted(ranks, key=ranks.get)
    mixed = [mixed[0], high] + [adapter for adapter in mixed if adapter not in (mixed[0], high)][:2]
    phases = []
    next_id = 0
    for name, adapters, count, prompt_len, output_len in (
        ('cold_load', [high], 2 * config.num_workers, 256, 64),
        ('resident_reuse', [high], 4 * config.num_workers, 256, 128),
        ('capacity_boundary', mixed, 12 * config.num_workers, 512, 768),
    ):
        requests = []
        for i in range(count):
            adapter = adapters[i % len(adapters)]
            requests.append(Request(next_id, base_model, adapter, dummy_prompt(prompt_len),
                                    prompt_len, output_len, 0.0))
            next_id += 1
        phases.append((name, requests))
    return phases


def analyze_validation(directory, expected_requests, num_workers):
    directory = Path(directory)
    errors, missing = [], []
    received_ids, terminal_ids = set(), set()
    stages, workers, snapshots = set(), set(), 0
    drained_workers = set()
    reuse, reservations, rejections, legacy_false_rejections = 0, 0, 0, 0
    for path in sorted(directory.glob('worker_[0-9]*.jsonl')):
        live = {}
        last_sequence = 0
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                worker = row['worker_id']
                workers.add(worker)
                prefix = f'worker {worker} event {row["sequence"]}'
                if row['sequence'] != last_sequence + 1:
                    errors.append(f'{prefix}: missing/reordered log record')
                last_sequence = row['sequence']
                event = row['event']
                if event == 'received':
                    request_id = row['request_id']
                    if request_id in received_ids:
                        errors.append(f'{prefix}: duplicate received request {request_id}')
                    received_ids.add(request_id)
                    live[request_id] = row
                elif event == 'terminal':
                    request_id = row['request_id']
                    if request_id not in live or request_id in terminal_ids:
                        errors.append(f'{prefix}: terminal without unique live request {request_id}')
                    live.pop(request_id, None)
                    terminal_ids.add(request_id)
                    if not row['success']:
                        errors.append(f'{prefix}: failed request {request_id}')
                elif event == 'residency':
                    if sum(row['resident_adapter_cells'].values()) != row['resident_cells']:
                        errors.append(f'{prefix}: inconsistent residency snapshot')
                elif event == 'lifecycle':
                    snapshots += 1
                    stages.add(row['stage'])
                    groups = row['requests']
                    requests = [req for group in groups.values() for req in group]
                    ids = [req['request_id'] for req in requests]
                    if len(ids) != len(set(ids)) or set(ids) != set(live):
                        errors.append(f'{prefix}: lifecycle lost or duplicated in-flight requests')
                    if not ids and not live:
                        drained_workers.add(worker)
                    for req in requests:
                        if req.get('prompt_buffer_len', req['input_len']) != req['input_len']:
                            errors.append(f'{prefix}: inference mutated original prompt buffer')
                        if req['rwpt'] != int(req['input_len'] * (1 + req['beta'] * req['rank'])):
                            errors.append(f'{prefix}: incorrect request RWPT')
                        original = live.get(req['request_id'])
                        if original and any(req[key] != original[key]
                                            for key in ('input_len', 'rank', 'beta', 'rwpt')):
                            errors.append(f'{prefix}: request contribution changed across stages')
                    reported = row['reported']
                    expected = {'queue_length': len(ids)}
                    for group, raw_key, rwpt_key in (
                        ('waiting', 'pending_raw_tokens', 'pending_prefill_tokens'),
                        ('prefilling', 'prefilling_raw_tokens', 'prefilling_rwpt_tokens'),
                        ('active', 'current_batch_prompt_tokens', 'active_rwpt_tokens'),
                    ):
                        expected[raw_key] = sum(req['input_len'] for req in groups[group])
                        expected[rwpt_key] = sum(req['rwpt'] for req in groups[group])
                    if any(reported.get(key) != value for key, value in expected.items()):
                        errors.append(f'{prefix}: reported load differs from request accounting')
                elif event == 'admission':
                    resident = row['resident_adapter_cells']
                    reserved = row['reserved_adapter_cells']
                    if resident is None:
                        errors.append(f'{prefix}: admission without verified residency')
                        continue
                    expected_reserved = {adapter: rank * 4
                                         for adapter, rank in row['required_adapter_ranks'].items()
                                         if adapter not in resident}
                    if (set(resident) & set(reserved)
                            or reserved != expected_reserved
                            or sum(resident.values()) != row['resident_cells']
                            or sum(reserved.values()) != row['reserved_cells']
                            or row['kv_budget'] != row['total_cells'] - row['resident_cells'] - row['reserved_cells']
                            or row['memory_allowed'] != (row['kv_peak'] < row['kv_budget'])):
                        errors.append(f'{prefix}: invalid adapter/KV budget')
                    reuse += row['adapter_dir'] in resident and row['adapter_dir'] not in reserved
                    reservations += bool(reserved)
                    rejections += not row['memory_allowed']
                    legacy_false_rejections += (row['memory_allowed'] and row['concurrency_allowed']
                                                and row['batch_tokens_allowed']
                                                and row['kv_peak'] >= row['legacy_kv_budget'])
        if live:
            errors.append(f'{path.name}: {len(live)} requests did not terminate')
    if len(received_ids) != expected_requests or terminal_ids != received_ids:
        errors.append('Worker receive/terminal counts do not match the benchmark request count')
    for name, covered in (
        ('all_workers', workers == set(range(num_workers))),
        ('all_workers_drained', drained_workers == set(range(num_workers))),
        ('load_and_prefill', {'admitted', 'adapters_loaded', 'before_prefill', 'after_prefill'} <= stages),
        ('active_execution', 'active' in stages),
        ('lifecycle_snapshots', snapshots > 0),
        ('resident_reuse', reuse > 0), ('new_reservation', reservations > 0),
        ('memory_boundary_rejection', rejections > 0),
        ('legacy_false_rejection_witness', legacy_false_rejections > 0),
    ):
        if not covered:
            missing.append(name)
    routing_samples = 0
    routing_path = directory / 'route_decisions.jsonl'
    if routing_path.exists():
        with routing_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                for state in row['workers'].values():
                    routing_samples += 1
                    expected = (state['pending_prefill_tokens'] + state['prefilling_rwpt_tokens']
                                + state['active_rwpt_tokens']) / row['capacity']
                    if row['load_metric'] != 'rwpt_active' or not math.isclose(state['load_pressure'], expected):
                        errors.append('Router pressure does not include all three RWPT stages')
    if not routing_samples:
        missing.append('router_load_samples')
    prefilling_reports = 0
    receiver_path = directory / 'worker_router.jsonl'
    if receiver_path.exists():
        with receiver_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row['event'] == 'worker_report' and row['report_seq'] == row['applied_seq']:
                    if (row['prefilling_request_count'] != row['applied_prefilling_request_count']
                            or row['prefilling_rwpt_tokens'] != row['applied_prefilling_rwpt_tokens']):
                        errors.append('Router dropped prefilling contribution from a worker report')
                    prefilling_reports += row['prefilling_request_count'] > 0 and row['prefilling_rwpt_tokens'] > 0
    if not prefilling_reports:
        missing.append('prefilling_report_received_by_router')
    return dict(status='failed' if errors else ('incomplete' if missing else 'passed'),
                errors=errors, missing_coverage=missing, received_requests=len(received_ids),
                terminal_requests=len(terminal_ids), lifecycle_snapshots=snapshots,
                resident_reuse_checks=reuse, new_reservation_checks=reservations,
                memory_boundary_rejections=rejections,
                legacy_false_rejection_witnesses=legacy_false_rejections,
                router_load_samples=routing_samples, prefilling_reports_received=prefilling_reports)


def run_validation_phases(runner, config, base_model, adapter_dirs):
    directory = runner._current_diagnostics_dir
    phases = build_validation_phases(config, base_model, adapter_dirs)
    source_root = Path(__file__).resolve().parents[2]
    sources = [source_root / name for name in (
        'slora/server/router/gpu_worker.py', 'slora/server/router/req_queue.py',
        'slora/server/router/model_infer/infer_adapter.py',
        'slora/server/router/model_infer/infer_batch.py',
        'slora/server/router/model_infer/model_rpc.py', 'slora/server/io_struct.py',
        'slora/server/router/validation_log.py', 'benchmarks/routing_experiment/runner.py',
        'benchmarks/routing_experiment/gpu_validation.py',
        'slora/server/router/dp_manager.py', 'slora/server/router/rwpt.py',
        'benchmarks/routing_experiment/suite.py', 'benchmarks/routing_experiment/config.py',
    )]
    manifest = dict(config=asdict(config), performance_result=False,
                    protocol='one startup; drain between phases; retain adapter residency',
                    phase_timeout_seconds=config.duration,
                    source_sha256={str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sources})
    (directory / 'validation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    plan = [dict(phase=name, req_id=req.req_id, adapter_dir=req.adapter_dir,
                 input_len=req.prompt_len, output_len=req.output_len)
            for name, requests in phases for req in requests]
    (directory / 'validation_request_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    all_records = []
    start = time.time()
    try:
        for name, requests in phases:
            print(f'     Validation phase: {name}, {len(requests)} requests (drain before next phase)')
            prefix = f'validation-{directory.name}-{name}'
            records = asyncio.run(asyncio.wait_for(
                runner._async_benchmark(requests, request_id_prefix=prefix),
                timeout=config.duration,
            ))
            for record in records:
                record['validation_phase'] = name
            runner._save_request_latency_trace(records, config, prefix)
            all_records.extend(records)
            if not all(record['success'] for record in records):
                raise RuntimeError(f'GPU validation phase {name} has failed requests')
        summary = analyze_validation(directory, len(plan), config.num_workers)
    except Exception as exc:
        summary = dict(status='failed', errors=[f'{type(exc).__name__}: {exc}'])
        (directory / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        raise
    (directory / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'     GPU correctness validation: {summary["status"]}; {directory / "validation_summary.json"}')
    if summary['status'] != 'passed':
        raise RuntimeError('GPU validation did not pass; inspect validation_summary.json')
    return runner._calculate_benchmark_stats(all_records, time.time() - start, config.req_rate)
