"""
Tests for ExperimentRunner config identity behavior.
"""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "benchmarks"))

from routing_experiment.config import ExperimentConfig
from routing_experiment.runner import ExperimentRunner
from slora.server.router.adapter_aware_router import AdapterAwareRouter
from slora.server.router.dp_manager import DataParallelRouterManager
from slora.server.router.worker_state import RoutingConfig, WorkerState


class TestExperimentRunnerConfigId:
    def test_config_id_distinguishes_req_rate(self):
        config_a = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.3,
            req_rate=2.0,
            duration=180,
        )
        config_b = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.3,
            req_rate=4.0,
            duration=180,
        )

        assert ExperimentRunner._make_config_id(config_a) != ExperimentRunner._make_config_id(config_b)

    def test_config_id_distinguishes_synthetic_and_trace_workloads(self):
        synthetic = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.1,
            req_rate=6.0,
            duration=180,
        )
        trace = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.1,
            req_rate=6.0,
            duration=180,
            workload_type="trace",
            trace_file="real_workload/outputs/azure_llm_http_top100_6rps_180s_v1.jsonl",
            workload_name="azure-http-top100-6rps",
        )

        assert ExperimentRunner._make_config_id(synthetic) != ExperimentRunner._make_config_id(trace)

    def test_config_id_distinguishes_trace_workloads(self):
        trace_a = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.1,
            req_rate=6.0,
            duration=180,
            workload_type="trace",
            trace_file="real_workload/outputs/azure_llm_http_top100_6rps_180s_v1.jsonl",
            workload_name="azure-http-top100-6rps",
        )
        trace_b = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.1,
            req_rate=8.0,
            duration=180,
            workload_type="trace",
            trace_file="real_workload/outputs/azure_llm_http_top100_8rps_180s_v1.jsonl",
            workload_name="azure-http-top100-8rps",
        )

        assert ExperimentRunner._make_config_id(trace_a) != ExperimentRunner._make_config_id(trace_b)

    def test_resolve_trace_file_uses_benchmarks_dir_for_relative_paths(self):
        runner = ExperimentRunner(
            output_dir=tempfile.mkdtemp(),
            benchmarks_dir="/repo/benchmarks",
        )

        assert runner._resolve_trace_file("real_workload/outputs/trace.jsonl") == Path(
            "/repo/benchmarks/real_workload/outputs/trace.jsonl"
        )


class TestExperimentRunnerRuntimeCleanup:
    def test_cleanup_runtime_state_files_removes_stale_files(self):
        runner = ExperimentRunner(output_dir=tempfile.mkdtemp(), benchmarks_dir=".")

        runtime_files = [
            "/tmp/slora_routing_stats.json",
            "/tmp/slora_routing_stats.json.tmp",
            "/tmp/slora_routing_config_update.json",
            "/tmp/slora_reset_adapter_cache.trigger",
            "/tmp/slora_reset_adapter_cache.result",
        ]

        created = []
        try:
            for path in runtime_files:
                with open(path, "w") as f:
                    f.write("stale")
                created.append(path)

            runner._cleanup_runtime_state_files()

            for path in runtime_files:
                assert not os.path.exists(path)
        finally:
            for path in created:
                if os.path.exists(path):
                    os.remove(path)

    def test_config_id_distinguishes_duration(self):
        config_a = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.3,
            req_rate=6.0,
            duration=180,
            routing_w1=1.0,
            routing_w2=1.0,
            load_metric="rwpt",
        )
        config_b = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.3,
            req_rate=6.0,
            duration=240,
            routing_w1=1.0,
            routing_w2=1.0,
            load_metric="rwpt",
        )

        assert ExperimentRunner._make_config_id(config_a) != ExperimentRunner._make_config_id(config_b)


class DummyRouter:
    def __init__(self):
        self.reset_called = False

    def reset_stats(self):
        self.reset_called = True


class TestDataParallelRouterManagerStatsReset:
    def test_reset_experiment_stats_clears_manager_counters(self):
        manager = DataParallelRouterManager.__new__(DataParallelRouterManager)
        manager.num_workers = 3
        manager.router = DummyRouter()
        manager.stats = {
            "total_requests": 12,
            "successful_requests": 11,
            "failed_requests": 1,
            "worker_request_counts": [4, 4, 4],
            "start_time": 123.0,
        }

        manager._reset_experiment_stats()

        assert manager.stats["total_requests"] == 0
        assert manager.stats["successful_requests"] == 0
        assert manager.stats["failed_requests"] == 0
        assert manager.stats["worker_request_counts"] == [0, 0, 0]
        assert manager.stats["start_time"] is not None
        assert manager.router.reset_called is True


class TestDataParallelRouterManagerOptimisticLoadUpdate:
    def _make_adapter_aware_manager(self):
        manager = DataParallelRouterManager.__new__(DataParallelRouterManager)
        manager.routing_strategy = "adapter-aware"
        manager.router = AdapterAwareRouter(
            num_workers=1,
            config=RoutingConfig(
                strategy="adapter-aware",
                load_metric="rwpt",
                hidden_dim=4096,
                default_lora_rank=16,
            ),
        )
        manager.router.adapter_ranks = {"/adapters/a": 64}
        manager.router.worker_states[0] = WorkerState(worker_id=0)
        return manager

    def test_optimistic_update_increments_queue_and_prefill_for_rwpt(self):
        manager = self._make_adapter_aware_manager()

        manager._optimistically_update_worker_load(
            0,
            {
                "adapter_dir": "/adapters/a",
                "prompt_ids": list(range(120)),
            },
        )

        state = manager.router.worker_states[0]
        expected = int(120 * (1.0 + (2.0 / (3.0 * 4096)) * 64))
        assert state.queue_length == 1
        assert state.pending_prefill_tokens == expected

    def test_optimistic_update_uses_default_rank_for_unknown_adapter(self):
        manager = self._make_adapter_aware_manager()

        manager._optimistically_update_worker_load(
            0,
            {
                "adapter_dir": "/adapters/unknown",
                "prompt_ids": list(range(80)),
            },
        )

        state = manager.router.worker_states[0]
        expected = int(80 * (1.0 + (2.0 / (3.0 * 4096)) * 16))
        assert state.queue_length == 1
        assert state.pending_prefill_tokens == expected

    def test_optimistic_update_increments_raw_tokens_for_token_count(self):
        manager = self._make_adapter_aware_manager()
        manager.router.config.load_metric = "token_count"

        manager._optimistically_update_worker_load(
            0,
            {
                "adapter_dir": "/adapters/a",
                "prompt_ids": list(range(120)),
            },
        )

        state = manager.router.worker_states[0]
        assert state.queue_length == 1
        assert state.pending_raw_tokens == 120
        assert state.pending_prefill_tokens == 0

    def test_optimistic_update_does_not_increment_raw_tokens_for_queue_length(self):
        manager = self._make_adapter_aware_manager()
        manager.router.config.load_metric = "queue_length"

        manager._optimistically_update_worker_load(
            0,
            {
                "adapter_dir": "/adapters/a",
                "prompt_ids": list(range(120)),
            },
        )

        state = manager.router.worker_states[0]
        assert state.queue_length == 1
        assert state.pending_raw_tokens == 0
        assert state.pending_prefill_tokens == 0

    def test_optimistic_update_only_increments_queue_when_prompt_missing(self):
        manager = self._make_adapter_aware_manager()

        manager._optimistically_update_worker_load(
            0,
            {
                "adapter_dir": "/adapters/a",
            },
        )

        state = manager.router.worker_states[0]
        assert state.queue_length == 1
        assert state.pending_prefill_tokens == 0

    def test_optimistic_update_is_noop_when_worker_state_missing(self):
        manager = self._make_adapter_aware_manager()
        manager.router.worker_states.pop(0)

        manager._optimistically_update_worker_load(
            0,
            {
                "adapter_dir": "/adapters/a",
                "prompt_ids": list(range(32)),
            },
        )

        assert 0 not in manager.router.worker_states

    def test_route_request_round_robin_does_not_trigger_optimistic_rwpt_update(self):
        manager = DataParallelRouterManager.__new__(DataParallelRouterManager)
        manager.routing_strategy = "round-robin"
        manager.router = SimpleNamespace(select_worker=lambda: 0)
        manager.request_senders = [SimpleNamespace(send_json=self._async_noop)]
        manager.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "worker_request_counts": [0],
        }

        manager._optimistically_update_worker_load = self._boom

        import asyncio

        asyncio.run(
            manager.route_request(
                {
                    "request_id": "req-1",
                    "adapter_dir": "/adapters/a",
                    "prompt_ids": [1, 2, 3],
                    "sampling_params": {},
                }
            )
        )

        assert manager.stats["total_requests"] == 1
        assert manager.stats["successful_requests"] == 1
        assert manager.stats["worker_request_counts"] == [1]

    async def _async_noop(self, _request):
        return None

    def _boom(self, *_args, **_kwargs):
        raise AssertionError("optimistic update should not be called in round-robin mode")
