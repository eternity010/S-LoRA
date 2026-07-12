"""
Tests for ExperimentRunner config identity behavior.
"""

import os
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "benchmarks"))

from routing_experiment.config import ExperimentConfig
from routing_experiment.runner import ExperimentRunner
from trace import Request
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
            "/tmp/slora_reset_adapter_cache.trigger.tmp",
            "/tmp/slora_reset_adapter_cache.result",
            "/tmp/slora_reset_adapter_cache.result.tmp",
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

    def test_wait_for_routing_config_requires_matching_update_id(self, monkeypatch):
        runner = ExperimentRunner(output_dir=tempfile.mkdtemp(), benchmarks_dir=".")
        responses = iter([
            SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "config_update_id": "old-update",
                    "routing_config": {"w2": 2.0, "load_metric": "token_count"},
                },
            ),
            SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "config_update_id": "target-update",
                    "routing_config": {"w2": 2.0, "load_metric": "token_count"},
                },
            ),
        ])
        monkeypatch.setattr(
            "routing_experiment.runner.requests.get",
            lambda *_args, **_kwargs: next(responses),
        )
        monkeypatch.setattr("routing_experiment.runner.time.sleep", lambda _seconds: None)

        assert runner._wait_for_routing_config(
            "target-update",
            {"w2": 2.0, "load_metric": "token_count"},
            timeout=1.0,
        )

    def test_wait_for_routing_config_times_out_without_confirmation(self):
        runner = ExperimentRunner(output_dir=tempfile.mkdtemp(), benchmarks_dir=".")

        assert not runner._wait_for_routing_config(
            "missing-update",
            {"w2": 2.0},
            timeout=0.0,
        )

    def test_collect_routing_stats_waits_for_complete_snapshot(self, monkeypatch):
        runner = ExperimentRunner(output_dir=tempfile.mkdtemp(), benchmarks_dir=".")
        responses = iter([
            SimpleNamespace(
                status_code=200,
                json=lambda: {"total_requests": 1070},
            ),
            SimpleNamespace(
                status_code=200,
                json=lambda: {"total_requests": 1080},
            ),
        ])
        monkeypatch.setattr(
            "routing_experiment.runner.requests.get",
            lambda *_args, **_kwargs: next(responses),
        )
        monkeypatch.setattr("routing_experiment.runner.time.sleep", lambda _seconds: None)

        stats = runner._collect_routing_stats(
            min_total_requests=1080,
            timeout=1.0,
        )

        assert stats["total_requests"] == 1080

    def test_reset_adapter_cache_requires_matching_completion_id(self, monkeypatch):
        runner = ExperimentRunner(output_dir=tempfile.mkdtemp(), benchmarks_dir=".")

        def post(_url, json, timeout):
            assert timeout == 30
            assert json["wait_seconds"] == 20.0
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "status": "success",
                    "reset_id": json["reset_id"],
                    "message": "completed",
                },
            )

        monkeypatch.setattr("routing_experiment.runner.requests.post", post)

        assert runner._reset_adapter_cache()

    def test_reset_adapter_cache_rejects_stale_completion_id(self, monkeypatch):
        runner = ExperimentRunner(output_dir=tempfile.mkdtemp(), benchmarks_dir=".")
        monkeypatch.setattr(
            "routing_experiment.runner.requests.post",
            lambda *_args, **_kwargs: SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "status": "success",
                    "reset_id": "stale-reset",
                    "message": "completed",
                },
            ),
        )

        assert not runner._reset_adapter_cache()


class TestExperimentRunnerRequestTracing:
    def test_generation_request_preserves_request_id(self):
        req = Request(
            req_id=17,
            model_dir="/models/base",
            adapter_dir="/adapters/a",
            prompt="Hello",
            prompt_len=8,
            output_len=12,
            req_time=1.5,
        )

        payload = ExperimentRunner._build_generation_request(req, "run-1-req-17")

        assert payload["req_id"] == "run-1-req-17"
        assert payload["lora_dir"] == "/adapters/a"
        assert payload["parameters"]["max_new_tokens"] == 12

    def test_calculate_benchmark_stats_accepts_request_records(self):
        runner = ExperimentRunner(output_dir=tempfile.mkdtemp(), benchmarks_dir=".")
        records = [
            {"success": True, "total_latency": 2.0, "ttft": 1.0},
            {"success": True, "total_latency": 4.0, "ttft": 2.0},
            {"success": False, "total_latency": None, "ttft": None},
        ]

        stats = runner._calculate_benchmark_stats(records, benchmark_time=10.0, req_rate=1.0)

        assert stats["total_requests"] == 3
        assert stats["num_abort"] == 1
        assert stats["throughput"] == 0.2
        assert stats["avg_latency"] == 3.0
        assert stats["avg_first_token_latency"] == 1.5

    def test_save_request_latency_trace_adds_config_metadata(self, tmp_path):
        runner = ExperimentRunner(output_dir=str(tmp_path), benchmarks_dir=".", debug=False)
        runner._current_diagnostics_dir = tmp_path / "diagnostics"
        runner._current_diagnostics_dir.mkdir()
        config = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.1,
            req_rate=4.0,
            duration=180,
            workload_type="trace",
            trace_file="trace.jsonl",
            workload_name="trace-4rps",
            routing_w2=3.0,
        )
        records = [{"request_id": "run-req-1", "success": True}]

        output = runner._save_request_latency_trace(records, config, "run1")
        saved = json.loads(output.read_text().strip())

        assert saved["request_id"] == "run-req-1"
        assert saved["routing_strategy"] == "adapter-aware"
        assert saved["routing_w2"] == 3.0
        assert saved["workload_name"] == "trace-4rps"


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
