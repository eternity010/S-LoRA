"""
Tests for ExperimentRunner config identity behavior.
"""

import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "benchmarks"))

from routing_experiment.config import ExperimentConfig
from routing_experiment.runner import ExperimentRunner
from slora.server.router.dp_manager import DataParallelRouterManager


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
