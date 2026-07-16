"""
Tests for ExperimentSuite

Property 8: Suite Configuration Generation
Validates: Requirements 6.1, 6.2, 6.3, 6.4
"""

import pytest
from hypothesis import given, strategies as st
from itertools import product

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routing_experiment.suite import ExperimentSuite
from routing_experiment.config import ExperimentConfig
from routing_experiment.runner import ExperimentRunner


class TestExperimentSuite:
    """Unit tests for experiment suite"""
    
    def test_list_suites(self):
        """Test listing all available suites"""
        suites = ExperimentSuite.list_suites()
        
        assert isinstance(suites, list)
        assert len(suites) >= 3
        assert "dp-roundrobin-baseline" in suites
        assert "dp-rwpt-baseline" in suites
        assert "dp-roundrobin-rate-scaling" in suites
    
    def test_dp_roundrobin_baseline_suite(self):
        """Test dp-roundrobin-baseline suite generates correct configs"""
        configs = list(ExperimentSuite.get_configs("dp-roundrobin-baseline"))
        
        # Should have 1 strategy × 3 alphas × 1 adapter × 1 rate × 1 duration = 3 configs
        assert len(configs) == 3
        
        # Check all configs are valid
        for config in configs:
            assert isinstance(config, ExperimentConfig)
            config.validate()  # Should not raise
        
        # Check parameter coverage
        strategies = {c.routing_strategy for c in configs}
        alphas = {c.alpha for c in configs}
        
        assert strategies == {"round-robin"}
        assert alphas == {0.1, 0.3, 0.8}
        assert all(c.num_adapters == 100 for c in configs)
    
    def test_dp_roundrobin_rate_scaling_suite(self):
        """Test dp-roundrobin-rate-scaling suite generates correct configs"""
        configs = list(ExperimentSuite.get_configs("dp-roundrobin-rate-scaling"))
        
        # Should have 1 strategy × 1 alpha × 1 adapter × 4 rates × 1 duration = 4 configs
        assert len(configs) == 4
        
        # Check all configs are valid
        for config in configs:
            assert isinstance(config, ExperimentConfig)
            config.validate()  # Should not raise
        
        # Check parameter coverage
        strategies = {c.routing_strategy for c in configs}
        rates = {c.req_rate for c in configs}
        
        assert strategies == {"round-robin"}
        assert rates == {2.0, 4.0, 6.0, 8.0}
        assert all(c.alpha == 0.3 for c in configs)

    def test_dp_roundrobin_rate8_validation_suite(self):
        """Test dp-roundrobin-rate8-validation suite generates one skewed rate-8 config"""
        configs = list(ExperimentSuite.get_configs("dp-roundrobin-rate8-validation"))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "round-robin"
        assert config.alpha == 0.1
        assert config.num_adapters == 100
        assert config.req_rate == 8.0
        assert config.duration == 180

    def test_dp_roundrobin_rate8_alpha02_validation_suite(self):
        """Test dp-roundrobin-rate8-alpha02-validation suite generates one rate-8 config"""
        configs = list(ExperimentSuite.get_configs("dp-roundrobin-rate8-alpha02-validation"))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "round-robin"
        assert config.alpha == 0.2
        assert config.num_adapters == 100
        assert config.req_rate == 8.0
        assert config.duration == 180

    def test_dp_roundrobin_rate8_alpha03_validation_suite(self):
        """Test dp-roundrobin-rate8-alpha03-validation suite generates one rate-8 config"""
        configs = list(ExperimentSuite.get_configs("dp-roundrobin-rate8-alpha03-validation"))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "round-robin"
        assert config.alpha == 0.3
        assert config.num_adapters == 100
        assert config.req_rate == 8.0
        assert config.duration == 180

    def test_dp_rwpt_rate8_w2_search_suite(self):
        """Test dp-rwpt-rate8-w2-search suite generates one skewed rate-8 config"""
        configs = list(ExperimentSuite.get_configs("dp-rwpt-rate8-w2-search"))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "adapter-aware"
        assert config.alpha == 0.1
        assert config.num_adapters == 100
        assert config.req_rate == 8.0
        assert config.duration == 180
        assert config.routing_w2 == 1.0
        assert config.load_metric == "rwpt"

    def test_dp_rwpt_rate8_alpha02_validation_suite(self):
        """Test dp-rwpt-rate8-alpha02-validation suite generates one rate-8 config"""
        configs = list(ExperimentSuite.get_configs("dp-rwpt-rate8-alpha02-validation"))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "adapter-aware"
        assert config.alpha == 0.2
        assert config.num_adapters == 100
        assert config.req_rate == 8.0
        assert config.duration == 180
        assert config.routing_w2 == 1.0
        assert config.load_metric == "rwpt"

    def test_dp_rwpt_rate8_alpha03_validation_suite(self):
        """Test dp-rwpt-rate8-alpha03-validation suite generates one rate-8 config"""
        configs = list(ExperimentSuite.get_configs("dp-rwpt-rate8-alpha03-validation"))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "adapter-aware"
        assert config.alpha == 0.3
        assert config.num_adapters == 100
        assert config.req_rate == 8.0
        assert config.duration == 180
        assert config.routing_w2 == 1.0
        assert config.load_metric == "rwpt"

    def test_dp_realtrace_comparison_suite(self):
        """Test dp-realtrace-comparison suite generates paired trace configs"""
        configs = list(ExperimentSuite.get_configs("dp-realtrace-comparison"))

        assert len(configs) == 2
        for config in configs:
            config.validate()
            assert config.workload_type == "trace"
            assert config.num_adapters == 100
            assert config.duration == 180
            assert config.trace_file is not None

        strategies = {config.routing_strategy for config in configs}
        workloads = {config.workload_name for config in configs}
        trace_files = {config.trace_file for config in configs}
        rates = {config.req_rate for config in configs}

        assert strategies == {"round-robin", "adapter-aware"}
        assert workloads == {"azure-http-top100-4rps"}
        assert rates == {4.0}
        for config in configs:
            if config.routing_strategy == "adapter-aware":
                assert config.routing_w2 == 3.0
        assert trace_files == {
            "real_workload/outputs/azure_llm_http_top100_4rps_180s_capped2048_512_v1.jsonl",
        }

    def test_dp_realtrace_comparison_suite_info(self):
        """Test dp-realtrace-comparison suite info"""
        info = ExperimentSuite.get_suite_info("dp-realtrace-comparison")

        assert info["name"] == "dp-realtrace-comparison"
        assert info["config_count"] == 2
        assert info["parameters"]["workload_name"] == ["azure-http-top100-4rps"]
        assert info["parameters"]["routing_w2"] == [3.0]

    def test_dp_realtrace_2rps_comparison_suite(self):
        configs = list(ExperimentSuite.get_configs("dp-realtrace-comparison-2rps"))

        assert len(configs) == 2
        assert {config.routing_strategy for config in configs} == {
            "round-robin", "adapter-aware",
        }
        for config in configs:
            config.validate()
            assert config.workload_type == "trace"
            assert config.workload_name == "azure-http-top100-2rps"
            assert config.req_rate == 2.0
            assert config.duration == 180
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file == (
                "real_workload/outputs/"
                "azure_llm_http_top100_2rps_180s_capped2048_512_v1.jsonl"
            )
            expected_w2 = 3.0 if config.routing_strategy == "adapter-aware" else 1.0
            assert config.routing_w2 == expected_w2

        info = ExperimentSuite.get_suite_info("dp-realtrace-comparison-2rps")
        assert info["config_count"] == 2
        assert info["parameters"]["workload_name"] == ["azure-http-top100-2rps"]

    def test_dp_realtrace_rwpt_4rps_debug_suite(self):
        """Test the single-point real-trace RWPT diagnostic suite."""
        configs = list(ExperimentSuite.get_configs("dp-realtrace-rwpt-4rps-debug"))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "adapter-aware"
        assert config.workload_type == "trace"
        assert config.workload_name == "azure-http-top100-4rps"
        assert config.req_rate == 4.0
        assert config.duration == 180
        assert config.routing_w2 == 3.0
        assert config.load_metric == "rwpt"
        assert config.trace_file == (
            "real_workload/outputs/"
            "azure_llm_http_top100_4rps_180s_capped2048_512_v1.jsonl"
        )

    def test_dp_realtrace_queue_length_w2_search_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-queue-length-w2-search"
        ))

        assert len(configs) == 4
        assert {config.routing_w2 for config in configs} == {
            0.05, 0.10, 0.20, 0.30,
        }
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.workload_type == "trace"
            assert config.workload_name == "azure-http-top100-6rps"
            assert config.req_rate == 6.0
            assert config.duration == 180
            assert config.gpu_ids == "1,2,3"
            assert config.load_metric == "queue_length"
            assert config.trace_file == (
                "real_workload/outputs/"
                "azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl"
            )

        info = ExperimentSuite.get_suite_info(
            "dp-realtrace-queue-length-w2-search"
        )
        assert info["config_count"] == 4

    def test_dp_realtrace_token_count_w2_search_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-token-count-w2-search"
        ))

        assert len(configs) == 4
        assert {config.routing_w2 for config in configs} == {
            2.0, 3.0, 4.0, 5.0,
        }
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.workload_type == "trace"
            assert config.workload_name == "azure-http-top100-6rps"
            assert config.req_rate == 6.0
            assert config.duration == 180
            assert config.gpu_ids == "1,2,3"
            assert config.load_metric == "token_count"
            assert config.trace_file == (
                "real_workload/outputs/"
                "azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl"
            )

        info = ExperimentSuite.get_suite_info(
            "dp-realtrace-token-count-w2-search"
        )
        assert info["config_count"] == 4

    def test_dp_realtrace_load_metric_comparison_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-load-metric-comparison"
        ))

        assert len(configs) == 3
        assert {(config.load_metric, config.routing_w2) for config in configs} == {
            ("queue_length", 0.20),
            ("token_count", 3.0),
            ("rwpt", 3.0),
        }
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.workload_type == "trace"
            assert config.workload_name == "azure-http-top100-6rps"
            assert config.req_rate == 6.0
            assert config.duration == 180
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file == (
                "real_workload/outputs/"
                "azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl"
            )

        info = ExperimentSuite.get_suite_info(
            "dp-realtrace-load-metric-comparison"
        )
        assert info["config_count"] == 3
        assert set(info["parameters"]["metric_w2_pairs"]) == {
            ("queue_length", 0.20),
            ("token_count", 3.0),
            ("rwpt", 3.0),
        }

    def test_dp_realtrace_8rps_load_metric_comparison_suite(self):
        suite_name = "dp-realtrace-8rps-load-metric-comparison"
        configs = list(ExperimentSuite.get_configs(suite_name))

        assert len(configs) == 4
        assert [
            (config.routing_strategy, config.load_metric, config.routing_w2)
            for config in configs
        ] == [
            ("round-robin", "rwpt", 1.0),
            ("adapter-aware", "rwpt", 3.0),
            ("adapter-aware", "token_count", 3.0),
            ("adapter-aware", "queue_length", 0.20),
        ]
        for config in configs:
            config.validate()
            assert config.workload_type == "trace"
            assert config.workload_name == "azure-http-top100-8rps"
            assert config.req_rate == 8.0
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl"
            )

        info = ExperimentSuite.get_suite_info(suite_name)
        assert info["config_count"] == 4

    def test_dp_realtrace_token_count_state_debug_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-token-count-state-debug"
        ))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "adapter-aware"
        assert config.load_metric == "token_count"
        assert config.routing_w2 == 3.0
        assert config.req_rate == 6.0
        assert config.duration == 60
        assert config.gpu_ids == "1,2,3"
        assert config.trace_file.endswith(
            "azure_llm_http_top100_6rps_60s_capped2048_512_debug.jsonl"
        )

    def test_dp_realtrace_token_count_8rps_debug_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-token-count-8rps-debug"
        ))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "adapter-aware"
        assert config.load_metric == "token_count"
        assert config.routing_w2 == 3.0
        assert config.req_rate == 8.0
        assert config.duration == 180
        assert config.gpu_ids == "1,2,3"
        assert config.trace_file.endswith(
            "azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl"
        )

    def test_dp_realtrace_token_count_8rps_repeat2_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-token-count-8rps-repeat2"
        ))

        assert len(configs) == 2
        assert [config.workload_name for config in configs] == [
            "azure-http-top100-8rps-token-count-run1",
            "azure-http-top100-8rps-token-count-run2",
        ]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "token_count"
            assert config.routing_w1 == 1.0
            assert config.routing_w2 == 3.0
            assert config.workload_type == "trace"
            assert config.req_rate == 8.0
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl"
            )

        assert ExperimentRunner._make_config_id(configs[0]) != (
            ExperimentRunner._make_config_id(configs[1])
        )
        assert ExperimentRunner._get_server_config_key(None, configs[0]) == (
            ExperimentRunner._get_server_config_key(None, configs[1])
        )

    def test_dp_realtrace_queue_length_8rps_standalone_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-queue-length-8rps-standalone"
        ))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "adapter-aware"
        assert config.load_metric == "queue_length"
        assert config.routing_w2 == 0.20
        assert config.req_rate == 8.0
        assert config.duration == 180
        assert config.num_adapters == 100
        assert config.gpu_ids == "1,2,3"
        assert config.trace_file.endswith(
            "azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl"
        )

    def test_dp_realtrace_queue_length_8rps_repeat2_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-queue-length-8rps-repeat2"
        ))

        assert len(configs) == 2
        assert [config.workload_name for config in configs] == [
            "azure-http-top100-8rps-queue-length-run1",
            "azure-http-top100-8rps-queue-length-run2",
        ]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "queue_length"
            assert config.routing_w1 == 1.0
            assert config.routing_w2 == 0.20
            assert config.workload_type == "trace"
            assert config.req_rate == 8.0
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl"
            )

        assert ExperimentRunner._make_config_id(configs[0]) != (
            ExperimentRunner._make_config_id(configs[1])
        )
        assert ExperimentRunner._get_server_config_key(None, configs[0]) == (
            ExperimentRunner._get_server_config_key(None, configs[1])
        )

    def test_dp_realtrace_rwpt_8rps_no_decay_standalone_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-rwpt-8rps-no-decay-standalone"
        ))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "adapter-aware"
        assert config.load_metric == "rwpt"
        assert config.routing_w2 == 3.0
        assert config.req_rate == 8.0
        assert config.duration == 180
        assert config.gpu_ids == "1,2,3"
        assert config.trace_file.endswith(
            "azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl"
        )

    def test_dp_realtrace_rwpt_8rps_repeat2_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-rwpt-8rps-repeat2"
        ))

        assert len(configs) == 2
        assert [config.workload_name for config in configs] == [
            "azure-http-top100-8rps-rwpt-run1",
            "azure-http-top100-8rps-rwpt-run2",
        ]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "rwpt"
            assert config.routing_w1 == 1.0
            assert config.routing_w2 == 3.0
            assert config.workload_type == "trace"
            assert config.req_rate == 8.0
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl"
            )

        assert ExperimentRunner._make_config_id(configs[0]) != (
            ExperimentRunner._make_config_id(configs[1])
        )
        assert ExperimentRunner._get_server_config_key(None, configs[0]) == (
            ExperimentRunner._get_server_config_key(None, configs[1])
        )

    def test_dp_realtrace_roundrobin_8rps_repeat2_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-roundrobin-8rps-repeat2"
        ))

        assert len(configs) == 2
        assert [config.workload_name for config in configs] == [
            "azure-http-top100-8rps-roundrobin-run1",
            "azure-http-top100-8rps-roundrobin-run2",
        ]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "round-robin"
            assert config.workload_type == "trace"
            assert config.req_rate == 8.0
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl"
            )

        assert ExperimentRunner._make_config_id(configs[0]) != (
            ExperimentRunner._make_config_id(configs[1])
        )
        runner = ExperimentRunner(output_dir="unused", benchmarks_dir=".")
        assert runner._get_server_config_key(configs[0]) == (
            runner._get_server_config_key(configs[1])
        )

    def test_dp_realtrace_roundrobin_4rps_repeat2_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-roundrobin-4rps-repeat2"
        ))

        assert len(configs) == 2
        assert [config.workload_name for config in configs] == [
            "azure-http-top100-4rps-roundrobin-run1",
            "azure-http-top100-4rps-roundrobin-run2",
        ]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "round-robin"
            assert config.workload_type == "trace"
            assert config.req_rate == 4.0
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_4rps_180s_capped2048_512_v1.jsonl"
            )

        assert ExperimentRunner._make_config_id(configs[0]) != (
            ExperimentRunner._make_config_id(configs[1])
        )
        runner = ExperimentRunner(output_dir="unused", benchmarks_dir=".")
        assert runner._get_server_config_key(configs[0]) == (
            runner._get_server_config_key(configs[1])
        )

    def test_dp_realtrace_rwpt_4rps_repeat2_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-rwpt-4rps-repeat2"
        ))

        assert len(configs) == 2
        assert [config.workload_name for config in configs] == [
            "azure-http-top100-4rps-rwpt-run1",
            "azure-http-top100-4rps-rwpt-run2",
        ]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "rwpt"
            assert config.routing_w1 == 1.0
            assert config.routing_w2 == 3.0
            assert config.workload_type == "trace"
            assert config.req_rate == 4.0
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_4rps_180s_capped2048_512_v1.jsonl"
            )

        assert ExperimentRunner._make_config_id(configs[0]) != (
            ExperimentRunner._make_config_id(configs[1])
        )
        runner = ExperimentRunner(output_dir="unused", benchmarks_dir=".")
        assert runner._get_server_config_key(configs[0]) == (
            runner._get_server_config_key(configs[1])
        )

    @pytest.mark.parametrize(
        ("suite_name", "strategy", "workload_prefix"),
        [
            (
                "dp-realtrace-roundrobin-6rps-repeat2",
                "round-robin",
                "azure-http-top100-6rps-roundrobin-run",
            ),
            (
                "dp-realtrace-rwpt-6rps-repeat2",
                "adapter-aware",
                "azure-http-top100-6rps-rwpt-run",
            ),
        ],
    )
    def test_dp_realtrace_6rps_repeat2_suites(
        self, suite_name, strategy, workload_prefix
    ):
        configs = list(ExperimentSuite.get_configs(suite_name))

        assert len(configs) == 2
        assert [config.workload_name for config in configs] == [
            f"{workload_prefix}1",
            f"{workload_prefix}2",
        ]
        for config in configs:
            config.validate()
            assert config.routing_strategy == strategy
            assert config.workload_type == "trace"
            assert config.req_rate == 6.0
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl"
            )
            if strategy == "adapter-aware":
                assert config.load_metric == "rwpt"
                assert config.routing_w1 == 1.0
                assert config.routing_w2 == 3.0

        assert ExperimentRunner._make_config_id(configs[0]) != (
            ExperimentRunner._make_config_id(configs[1])
        )
        runner = ExperimentRunner(output_dir="unused", benchmarks_dir=".")
        assert runner._get_server_config_key(configs[0]) == (
            runner._get_server_config_key(configs[1])
        )

    @pytest.mark.parametrize(
        ("suite_name", "rate"),
        [
            ("dp-realtrace-rwpt-active-4rps-repeat2", 4.0),
            ("dp-realtrace-rwpt-active-6rps-repeat2", 6.0),
            ("dp-realtrace-rwpt-active-8rps-repeat2", 8.0),
        ],
    )
    def test_dp_realtrace_rwpt_active_repeat2_suites(self, suite_name, rate):
        configs = list(ExperimentSuite.get_configs(suite_name))

        assert len(configs) == 2
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "rwpt_active"
            assert config.routing_w1 == 1.0
            assert config.routing_w2 == 0.4
            assert config.workload_type == "trace"
            assert config.req_rate == rate
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert "w2-0p4" in config.workload_name
            assert f"_{int(rate)}rps_" in config.trace_file

        assert ExperimentRunner._make_config_id(configs[0]) != (
            ExperimentRunner._make_config_id(configs[1])
        )
        runner = ExperimentRunner(output_dir="unused", benchmarks_dir=".")
        assert runner._get_server_config_key(configs[0]) == (
            runner._get_server_config_key(configs[1])
        )

    @pytest.mark.parametrize("rate", [6.0, 8.0, 10.0])
    def test_dp_realtrace_token_count_active_repeat2_suites(self, rate):
        suite_name = f"dp-realtrace-token-count-active-{int(rate)}rps-repeat2"
        configs = list(ExperimentSuite.get_configs(suite_name))

        assert len(configs) == 2
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "token_count_active"
            assert config.routing_w1 == 1.0
            assert config.routing_w2 == 0.4
            assert config.workload_type == "trace"
            assert config.req_rate == rate
            assert config.duration == 180
            assert config.num_adapters == 100
            assert config.gpu_ids == "1,2,3"
            assert "token-count-active-w2-0p4" in config.workload_name
            assert f"_{int(rate)}rps_" in config.trace_file

        assert ExperimentRunner._make_config_id(configs[0]) != (
            ExperimentRunner._make_config_id(configs[1])
        )
        runner = ExperimentRunner(output_dir="unused", benchmarks_dir=".")
        assert runner._get_server_config_key(configs[0]) == (
            runner._get_server_config_key(configs[1])
        )

    def test_dp_realtrace_w2_search_suite(self):
        configs = list(ExperimentSuite.get_configs("dp-realtrace-w2-search"))

        assert len(configs) == 4
        assert [config.routing_w2 for config in configs] == [1.0, 2.0, 3.0, 4.0]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "rwpt"
            assert config.req_rate == 6.0
            assert config.workload_name == "azure-http-top100-6rps"

        info = ExperimentSuite.get_suite_info("dp-realtrace-w2-search")
        assert info["config_count"] == 4
        assert info["parameters"]["routing_w2"] == [1.0, 2.0, 3.0, 4.0]

    def test_dp_realtrace_rwpt_w2_fine_search_suite(self):
        suite_name = "dp-realtrace-rwpt-w2-fine-search"
        configs = list(ExperimentSuite.get_configs(suite_name))

        assert len(configs) == 6
        assert [config.routing_w2 for config in configs] == [
            3.0, 3.5, 4.0, 4.5, 5.0, 6.0,
        ]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "rwpt"
            assert config.req_rate == 6.0
            assert config.duration == 180
            assert config.workload_name == "azure-http-top100-6rps"
            assert config.gpu_ids == "1,2,3"

        info = ExperimentSuite.get_suite_info(suite_name)
        assert info["config_count"] == 6

    def test_dp_realtrace_rwpt_active_w2_search_6rps_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-rwpt-active-w2-search-6rps"
        ))

        assert len(configs) == 4
        assert [config.routing_w2 for config in configs] == [0.5, 1.0, 1.5, 2.0]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "rwpt_active"
            assert config.routing_w1 == 1.0
            assert config.req_rate == 6.0
            assert config.duration == 180
            assert config.workload_name == (
                "azure-http-top100-6rps-rwpt-active-w2-search"
            )
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl"
            )

        assert len({ExperimentRunner._make_config_id(c) for c in configs}) == 4
        runner = ExperimentRunner(output_dir="unused", benchmarks_dir=".")
        assert len({runner._get_server_config_key(c) for c in configs}) == 1

    def test_dp_realtrace_rwpt_active_w2_low_search_6rps_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-rwpt-active-w2-low-search-6rps"
        ))

        assert len(configs) == 4
        assert [config.routing_w2 for config in configs] == [0.4, 0.3, 0.2, 0.1]
        for config in configs:
            config.validate()
            assert config.routing_strategy == "adapter-aware"
            assert config.load_metric == "rwpt_active"
            assert config.routing_w1 == 1.0
            assert config.req_rate == 6.0
            assert config.duration == 180
            assert config.workload_name == (
                "azure-http-top100-6rps-rwpt-active-w2-low-search"
            )
            assert config.gpu_ids == "1,2,3"
            assert config.trace_file.endswith(
                "azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl"
            )

        assert len({ExperimentRunner._make_config_id(c) for c in configs}) == 4
        runner = ExperimentRunner(output_dir="unused", benchmarks_dir=".")
        assert len({runner._get_server_config_key(c) for c in configs}) == 1

    def test_dp_realtrace_rwpt_active_w2_1_fresh_6rps_suite(self):
        configs = list(ExperimentSuite.get_configs(
            "dp-realtrace-rwpt-active-w2-1-fresh-6rps"
        ))

        assert len(configs) == 1
        config = configs[0]
        config.validate()
        assert config.routing_strategy == "adapter-aware"
        assert config.load_metric == "rwpt_active"
        assert config.routing_w1 == 1.0
        assert config.routing_w2 == 1.0
        assert config.req_rate == 6.0
        assert config.duration == 180
        assert config.workload_name == (
            "azure-http-top100-6rps-rwpt-active-w2-1-fresh"
        )
        assert config.gpu_ids == "1,2,3"
        assert config.trace_file.endswith(
            "azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl"
        )

    def test_dp_rwpt_baseline_suite(self):
        """Test dp-rwpt-baseline suite generates correct configs"""
        configs = list(ExperimentSuite.get_configs("dp-rwpt-baseline"))
        
        # Should have 1 strategy × 3 alphas × 1 adapter × 1 rate × 1 duration = 3 configs
        assert len(configs) == 3
        
        # Check all configs are valid
        for config in configs:
            assert isinstance(config, ExperimentConfig)
            config.validate()  # Should not raise
        
        # Check parameter coverage
        strategies = {c.routing_strategy for c in configs}
        alphas = {c.alpha for c in configs}
        load_metrics = {c.load_metric for c in configs}
        
        assert strategies == {"adapter-aware"}
        assert alphas == {0.1, 0.3, 0.8}
        assert load_metrics == {"rwpt"}

    @pytest.mark.parametrize(
        ("suite_name", "load_metric", "expected_w2"),
        [
            ("dp-tokencount-baseline", "token_count", 3.0),
            ("dp-queuelength-baseline", "queue_length", 0.20),
        ],
    )
    def test_metric_baselines_use_calibrated_w2(
        self, suite_name, load_metric, expected_w2
    ):
        configs = list(ExperimentSuite.get_configs(suite_name))

        assert len(configs) == 3
        assert {config.load_metric for config in configs} == {load_metric}
        assert {config.routing_w2 for config in configs} == {expected_w2}
        for config in configs:
            config.validate()

    def test_alpha_robustness_sla_uses_current_metric_w2(self):
        configs = list(ExperimentSuite.get_configs("alpha-robustness-sla"))
        metric_w2 = {
            (config.load_metric, config.routing_w2) for config in configs
        }

        assert metric_w2 == {
            ("rwpt", 1.0),
            ("token_count", 3.0),
            ("queue_length", 0.20),
        }

        info = ExperimentSuite.get_suite_info("alpha-robustness-sla")
        assert set(info["parameters"]["metric_w2_pairs"]) == metric_w2
    
    def test_unknown_suite_raises_error(self):
        """Test requesting unknown suite raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            list(ExperimentSuite.get_configs("nonexistent-suite"))
        
        error_msg = str(exc_info.value)
        assert "unknown suite" in error_msg.lower()
        assert "nonexistent-suite" in error_msg
    
    def test_get_suite_info(self):
        """Test getting suite information"""
        info = ExperimentSuite.get_suite_info("dp-roundrobin-baseline")
        
        assert info["name"] == "dp-roundrobin-baseline"
        assert "parameters" in info
        assert "config_count" in info
        assert info["config_count"] == 3
    
    def test_get_suite_info_unknown_suite(self):
        """Test getting info for unknown suite raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            ExperimentSuite.get_suite_info("nonexistent-suite")
        
        assert "unknown suite" in str(exc_info.value).lower()
    
    def test_configs_with_defaults(self):
        """Test generating configs with default parameters"""
        configs = list(ExperimentSuite.get_configs(
            "dp-roundrobin-baseline",
            cv=2.0,
            num_token=15000
        ))
        
        # Check defaults are applied
        assert all(c.cv == 2.0 for c in configs)
        assert all(c.num_token == 15000 for c in configs)
    
    def test_add_custom_suite(self):
        """Test adding a custom suite"""
        custom_suite = {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.5],
            "num_adapters": [75],
            "req_rate": [3.0],
            "duration": [60],
        }
        
        ExperimentSuite.add_custom_suite("test-custom", custom_suite)
        
        # Verify suite was added
        assert "test-custom" in ExperimentSuite.list_suites()
        
        # Verify configs can be generated
        configs = list(ExperimentSuite.get_configs("test-custom"))
        assert len(configs) == 1
        assert configs[0].routing_strategy == "adapter-aware"
        assert configs[0].alpha == 0.5
        
        # Cleanup
        del ExperimentSuite.SUITES["test-custom"]
    
    def test_add_custom_suite_duplicate_name(self):
        """Test adding suite with existing name raises error"""
        custom_suite = {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.5],
            "num_adapters": [75],
            "req_rate": [3.0],
            "duration": [60],
        }
        
        with pytest.raises(ValueError) as exc_info:
            ExperimentSuite.add_custom_suite("dp-roundrobin-baseline", custom_suite)
        
        assert "already exists" in str(exc_info.value).lower()
    
    def test_add_custom_suite_missing_parameters(self):
        """Test adding suite with missing required parameters raises error"""
        incomplete_suite = {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.5],
            # Missing num_adapters, req_rate, duration
        }
        
        with pytest.raises(ValueError) as exc_info:
            ExperimentSuite.add_custom_suite("test-incomplete", incomplete_suite)
        
        error_msg = str(exc_info.value).lower()
        assert "missing" in error_msg


class TestSuitePropertyBased:
    """Property-based tests for suite generation"""
    
    @given(
        suite_name=st.sampled_from([
            "dp-roundrobin-baseline",
            "dp-roundrobin-rate-scaling",
            "dp-rwpt-baseline"
        ])
    )
    def test_suite_generation_property(self, suite_name):
        """
        Feature: routing-comparison, Property 8: Suite Configuration Generation
        
        For any predefined suite name, get_configs() SHALL generate exactly
        the Cartesian product of all parameter lists defined in that suite,
        with each generated config having valid parameter values.
        """
        suite_def = ExperimentSuite.SUITES[suite_name]
        configs = list(ExperimentSuite.get_configs(suite_name))
        
        # Calculate expected count (Cartesian product size)
        expected_count = 1
        for values in suite_def.values():
            expected_count *= len(values)
        
        # Verify count matches
        assert len(configs) == expected_count
        
        # Verify all configs have valid values from the suite definition
        for config in configs:
            assert config.routing_strategy in suite_def["routing_strategy"]
            assert config.alpha in suite_def["alpha"]
            assert config.num_adapters in suite_def["num_adapters"]
            assert config.req_rate in suite_def["req_rate"]
            assert config.duration in suite_def["duration"]
            
            # Verify config is valid
            config.validate()  # Should not raise
        
        # Verify all combinations are present (no duplicates, no missing)
        param_combinations = set()
        for config in configs:
            combo = (
                config.routing_strategy,
                config.alpha,
                config.num_adapters,
                config.req_rate,
                config.duration
            )
            param_combinations.add(combo)
        
        # Generate expected combinations
        expected_combinations = set(product(
            suite_def["routing_strategy"],
            suite_def["alpha"],
            suite_def["num_adapters"],
            suite_def["req_rate"],
            suite_def["duration"]
        ))
        
        assert param_combinations == expected_combinations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
