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


def analyze_dispatch_evidence(directory, expected_requests, num_workers):
    directory = Path(directory)
    errors, missing = [], []
    pending = {wid: {} for wid in range(num_workers)}
    issued = dict.fromkeys(pending, 0)
    confirmed = dict.fromkeys(pending, 0)
    reports = dict.fromkeys(pending, 0)
    reserved, sent, received, latest = {}, set(), set(), {}
    epochs = set()
    noncovering_reports, accepted_reports = 0, 0
    path = directory / 'worker_router.jsonl'
    if path.exists():
        for line in path.open():
            row = json.loads(line)
            event = row['event']
            if event not in ('dispatch_reserved', 'dispatch_sent', 'dispatch_rollback',
                             'dispatch_uncertain', 'worker_report'):
                continue
            worker = row['source_worker_id']
            epochs.add(row.get('routing_epoch'))
            if event == 'worker_report':
                ack = row.get('ack_dispatch_seq')
                if (type(ack) is not int or not confirmed[worker] <= ack <= issued[worker]
                        or row['report_seq'] <= reports[worker]):
                    errors.append(f'Worker {worker}: invalid dispatch acknowledgement')
                    continue
                confirmed[worker], reports[worker] = ack, row['report_seq']
                accepted_reports += 1
                pending[worker] = {seq: req for seq, req in pending[worker].items() if seq > ack}
                entries = list(pending[worker].values())
                raw = sum(req['raw_tokens'] for req in entries)
                rwpt = sum(req['rwpt_tokens'] for req in entries)
                if row.get('load_metric') not in ('token_count', 'token_count_active'):
                    raw = 0
                if row.get('load_metric') not in ('rwpt', 'rwpt_active'):
                    rwpt = 0
                if (set(row.get('pending_sequences', [])) != set(pending[worker])
                        or row.get('optimistic_request_count') != len(entries)
                        or row.get('optimistic_raw_tokens') != raw
                        or row.get('optimistic_prefill_tokens') != rwpt
                        or row.get('effective_queue_length') != row.get('reported_queue_length', 0) + len(entries)
                        or row.get('effective_raw_tokens') != row.get('reported_raw_tokens', 0) + raw
                        or row.get('effective_rwpt_tokens') != row.get('reported_rwpt_tokens', 0) + rwpt):
                    errors.append(f'Worker {worker}: dispatch overlay lost or double counted requests')
                noncovering_reports += bool(entries)
                latest[worker] = row
                continue
            seq = row['dispatch_seq']
            key = (worker, seq)
            if event == 'dispatch_reserved':
                if seq <= issued[worker]:
                    errors.append(f'Worker {worker}: repeated or reordered dispatch sequence')
                issued[worker] = seq
                reserved[key] = row
                pending[worker][seq] = row
            elif event == 'dispatch_sent':
                if key not in reserved or key in sent:
                    errors.append(f'Worker {worker}: send without unique dispatch reservation')
                sent.add(key)
            elif event == 'dispatch_rollback':
                pending[worker].pop(seq, None)
            else:
                errors.append(f'Worker {worker}: uncertain delivery in correctness validation')
    for path in directory.glob('worker_[0-9]*.jsonl'):
        previous = 0
        for line in path.open():
            row = json.loads(line)
            if row['event'] == 'received':
                seq, worker = row.get('dispatch_seq'), row['worker_id']
                key = (worker, seq)
                epochs.add(row.get('routing_epoch'))
                if type(seq) is not int or seq <= previous or key in received:
                    errors.append(f'{path.name}: invalid receive sequence')
                    continue
                previous = seq
                received.add(key)
                original = reserved.get(key)
                if (original is None or original['request_id'] != row['request_id']
                        or original['raw_tokens'] != row['input_len']
                        or original['rwpt_tokens'] != row['rwpt']):
                    errors.append(f'{path.name}: received request differs from dispatch reservation')
            elif row['event'] == 'lifecycle' and row.get('ack_dispatch_seq') != previous:
                errors.append(f'{path.name}: snapshot watermark differs from received requests')
    if not sent:
        missing.append('dispatch_records')
    elif sent != received or len(sent) != expected_requests:
        errors.append('Dispatched and received requests do not match')
    if len(epochs) != 1 or None in epochs:
        errors.append('Missing or mixed routing epochs in dispatch evidence')
    for worker in pending:
        last = latest.get(worker, {})
        if pending[worker] or last.get('effective_queue_length') != 0:
            missing.append(f'dispatch_accounting_drained_worker_{worker}')
    return dict(errors=errors, missing_coverage=missing, sent_requests=len(sent),
                received_requests=len(received), accepted_reports=accepted_reports,
                noncovering_reports_preserved=noncovering_reports)


def wait_for_dispatch_drain(directory, expected_requests, num_workers, timeout=5.0):
    path = Path(directory) / 'worker_router.jsonl'
    deadline = time.monotonic() + timeout
    while True:
        latest, sent = {}, set()
        try:
            with path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    if row['event'] == 'dispatch_sent':
                        sent.add((row['source_worker_id'], row['dispatch_seq']))
                    elif row['event'] == 'worker_report':
                        latest[row['source_worker_id']] = row
            if len(sent) == expected_requests and all(
                wid in latest and latest[wid].get('pending_sequences') == []
                and latest[wid].get('effective_queue_length') == 0
                and latest[wid].get('ack_dispatch_seq', -1) >= max(
                    (seq for worker, seq in sent if worker == wid), default=0)
                for wid in range(num_workers)
            ):
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # A diagnostic line may still be being written.
        if time.monotonic() >= deadline:
            raise TimeoutError('Router did not confirm drained dispatch accounting')
        time.sleep(.05)


def analyze_validation(directory, expected_requests, num_workers, require_dispatch_accounting=False):
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
    dispatch = None
    if require_dispatch_accounting:
        dispatch = analyze_dispatch_evidence(directory, expected_requests, num_workers)
        errors.extend(dispatch['errors'])
        missing.extend(dispatch['missing_coverage'])
    summary = dict(status='failed' if errors else ('incomplete' if missing else 'passed'),
                errors=errors, missing_coverage=missing, received_requests=len(received_ids),
                terminal_requests=len(terminal_ids), lifecycle_snapshots=snapshots,
                resident_reuse_checks=reuse, new_reservation_checks=reservations,
                memory_boundary_rejections=rejections,
                legacy_false_rejection_witnesses=legacy_false_rejections,
                router_load_samples=routing_samples, prefilling_reports_received=prefilling_reports)
    if dispatch is not None:
        summary['dispatch_accounting'] = dispatch
    return summary


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
        'slora/server/router/dispatch_tracker.py', 'slora/server/router/worker_state.py',
        'slora/server/router/worker_state_reporter.py', 'slora/server/router/worker_state_cache.py',
        'benchmarks/routing_experiment/suite.py', 'benchmarks/routing_experiment/config.py',
    )]
    manifest = dict(config=asdict(config), performance_result=False,
                    dispatch_accounting=True,
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
        wait_for_dispatch_drain(directory, len(plan), config.num_workers)
        summary = analyze_validation(directory, len(plan), config.num_workers, require_dispatch_accounting=True)
    except Exception as exc:
        summary = dict(status='failed', errors=[f'{type(exc).__name__}: {exc}'])
        (directory / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        raise
    (directory / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'     GPU correctness validation: {summary["status"]}; {directory / "validation_summary.json"}')
    if summary['status'] != 'passed':
        raise RuntimeError('GPU validation did not pass; inspect validation_summary.json')
    return runner._calculate_benchmark_stats(all_records, time.time() - start, config.req_rate)
