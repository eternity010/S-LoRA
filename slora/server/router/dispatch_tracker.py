"""Reconcile ordered dispatches with atomic Worker load snapshots."""

from dataclasses import dataclass, field, replace
import time

from .worker_state import WorkerState


@dataclass
class PendingDispatch:
    request_id: str
    raw_tokens: int
    rwpt_tokens: int
    uncertain: bool = False
    created_at: float = field(default_factory=time.monotonic)


class DispatchTracker:
    def __init__(self, num_workers, epoch):
        self.epoch = epoch
        self.issued = [0] * num_workers
        self.pending = {wid: {} for wid in range(num_workers)}
        self.reported = {
            wid: WorkerState(worker_id=wid, routing_epoch=epoch, ack_dispatch_seq=0)
            for wid in range(num_workers)
        }

    def reserve(self, worker_id, request_id, raw_tokens, rwpt_tokens):
        self.issued[worker_id] += 1
        sequence = self.issued[worker_id]
        self.pending[worker_id][sequence] = PendingDispatch(request_id, raw_tokens, rwpt_tokens)
        return sequence

    def rollback(self, worker_id, sequence):
        self.pending[worker_id].pop(sequence, None)

    def mark_uncertain(self, worker_id, sequence):
        entry = self.pending[worker_id].get(sequence)
        if entry is not None:
            entry.uncertain = True

    def accept_report(self, state):
        old = self.reported[state.worker_id]
        ack = state.ack_dispatch_seq
        if (state.routing_epoch != self.epoch or not state.is_healthy
                or type(state.report_seq) is not int or state.report_seq <= old.report_seq
                or type(ack) is not int or not old.ack_dispatch_seq <= ack <= self.issued[state.worker_id]):
            return False
        self.reported[state.worker_id] = replace(
            state, cached_adapters=set(state.cached_adapters),
            top_k_rwpt_adapters=list(state.top_k_rwpt_adapters),
            optimistic_request_count=0, optimistic_raw_tokens=0, optimistic_prefill_tokens=0,
        )
        pending = self.pending[state.worker_id]
        for sequence in list(pending):
            if sequence <= ack:
                del pending[sequence]
        return True

    def effective_state(self, worker_id, metric):
        reported = self.reported[worker_id]
        entries = self.pending[worker_id].values()
        raw = sum(entry.raw_tokens for entry in entries)
        rwpt = sum(entry.rwpt_tokens for entry in entries)
        count = len(self.pending[worker_id])
        raw = raw if metric in ('token_count', 'token_count_active') else 0
        rwpt = rwpt if metric in ('rwpt', 'rwpt_active') else 0
        return replace(
            reported, cached_adapters=set(reported.cached_adapters),
            top_k_rwpt_adapters=list(reported.top_k_rwpt_adapters),
            queue_length=reported.queue_length + count,
            pending_raw_tokens=reported.pending_raw_tokens + raw,
            pending_prefill_tokens=reported.pending_prefill_tokens + rwpt,
            optimistic_request_count=count, optimistic_raw_tokens=raw,
            optimistic_prefill_tokens=rwpt,
            is_healthy=reported.is_healthy and not any(entry.uncertain for entry in entries),
        )
