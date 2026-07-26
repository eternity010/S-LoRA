"""
Tests for ExperimentConfig

Property 1: Configuration Validation
Validates: Requirements 1.1, 1.2, 1.3, 1.6, 1.7
"""

import pytest
from hypothesis import given, strategies as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routing_experiment.config import ExperimentConfig


class TestExperimentConfigValidation:
    """Unit tests for configuration validation"""
    
    def test_valid_config_round_robin(self):
        """Test valid round-robin configuration"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.6,
            req_rate=4.0,
            duration=120
        )
        config.validate()  # Should not raise
    
    def test_valid_config_adapter_aware(self):
        """Test valid adapter-aware configuration"""
        config = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=50,
            alpha=0.3,
            req_rate=5.0,
            duration=180
        )
        config.validate()  # Should not raise

    def test_rwpt_active_is_forwarded_to_server(self):
        config = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.1,
            req_rate=4.0,
            duration=180,
            load_metric="rwpt_active",
        )

        config.validate()
        args = config.to_server_args()

        index = args.index("--load-metric")
        assert args[index + 1] == "rwpt_active"

    def test_token_count_active_is_forwarded_to_server(self):
        config = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.1,
            req_rate=6.0,
            duration=180,
            load_metric="token_count_active",
        )

        config.validate()
        args = config.to_server_args()

        index = args.index("--load-metric")
        assert args[index + 1] == "token_count_active"

    def test_profiled_rank_beta_is_forwarded_to_server(self):
        config = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.1,
            req_rate=6.0,
            duration=180,
            profiled_rank_beta=0.00845,
        )

        args = config.to_server_args()

        index = args.index("--profiled-rank-beta")
        assert args[index + 1] == "0.00845"

    def test_negative_profiled_rank_beta_is_invalid(self):
        config = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.1,
            req_rate=6.0,
            duration=180,
            profiled_rank_beta=-0.1,
        )

        with pytest.raises(ValueError, match="profiled_rank_beta"):
            config.validate()

    def test_valid_trace_workload_config(self):
        """Test valid trace workload configuration"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.1,
            req_rate=6.0,
            duration=180,
            workload_type="trace",
            trace_file="real_workload/outputs/trace.jsonl",
            workload_name="realtrace-6rps",
        )
        config.validate()

        params = config.to_benchmark_args()
        assert params["workload_type"] == "trace"
        assert params["trace_file"] == "real_workload/outputs/trace.jsonl"
        assert params["workload_name"] == "realtrace-6rps"

    def test_trace_workload_requires_trace_file(self):
        """Test trace workload without trace_file raises ValueError"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.1,
            req_rate=6.0,
            duration=180,
            workload_type="trace",
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "trace_file" in str(exc_info.value)

    def test_invalid_workload_type(self):
        """Test invalid workload type raises ValueError"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.1,
            req_rate=6.0,
            duration=180,
            workload_type="invalid",
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "workload_type" in str(exc_info.value)
    
    def test_invalid_routing_strategy(self):
        """Test invalid routing strategy raises ValueError"""
        config = ExperimentConfig(
            routing_strategy="invalid-strategy",
            num_adapters=100,
            alpha=0.6,
            req_rate=4.0,
            duration=120
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "routing_strategy" in str(exc_info.value).lower()
    
    def test_alpha_too_small(self):
        """Test alpha < 0.1 raises ValueError"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.05,
            req_rate=4.0,
            duration=120
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "alpha" in str(exc_info.value).lower()
    
    def test_alpha_too_large(self):
        """Test alpha > 1.0 raises ValueError"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=1.5,
            req_rate=4.0,
            duration=120
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "alpha" in str(exc_info.value).lower()
    
    def test_alpha_boundary_values(self):
        """Test alpha boundary values (0.1 and 1.0) are valid"""
        config1 = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.1,
            req_rate=4.0,
            duration=120
        )
        config1.validate()  # Should not raise
        
        config2 = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=1.0,
            req_rate=4.0,
            duration=120
        )
        config2.validate()  # Should not raise
    
    def test_num_adapters_too_small(self):
        """Test num_adapters < 10 raises ValueError"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=5,
            alpha=0.6,
            req_rate=4.0,
            duration=120
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "adapter" in str(exc_info.value).lower()
    
    def test_num_adapters_too_large(self):
        """Test num_adapters > 200 raises ValueError"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=250,
            alpha=0.6,
            req_rate=4.0,
            duration=120
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "adapter" in str(exc_info.value).lower()
    
    def test_num_adapters_boundary_values(self):
        """Test num_adapters boundary values (10 and 200) are valid"""
        config1 = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=10,
            alpha=0.6,
            req_rate=4.0,
            duration=120
        )
        config1.validate()  # Should not raise
        
        config2 = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=200,
            alpha=0.6,
            req_rate=4.0,
            duration=120
        )
        config2.validate()  # Should not raise
    
    def test_to_server_args_round_robin(self):
        """Test conversion to server arguments for round-robin"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.6,
            req_rate=4.0,
            duration=120,
            gpu_ids="0,1,2",
            num_workers=3,
            num_token=12000
        )
        args = config.to_server_args()
        
        assert "--routing-strategy" in args
        assert "round-robin" in args
        assert "--num-adapter" in args
        assert "100" in args
        assert "--gpu-ids" in args
        assert "0,1,2" in args
        # Should not include routing weights for round-robin
        assert "--routing-w1" not in args
    
    def test_to_server_args_adapter_aware(self):
        """Test conversion to server arguments for adapter-aware"""
        config = ExperimentConfig(
            routing_strategy="adapter-aware",
            num_adapters=100,
            alpha=0.6,
            req_rate=4.0,
            duration=120,
            routing_w1=1.0,
            routing_w2=0.1,
            routing_w3=0.0
        )
        args = config.to_server_args()
        
        assert "--routing-strategy" in args
        assert "adapter-aware" in args
        # Should include routing weights for adapter-aware
        assert "--routing-w1" in args
        assert "1.0" in args
        assert "--routing-w2" in args
        assert "0.1" in args
    
    def test_to_benchmark_args(self):
        """Test conversion to benchmark parameters"""
        config = ExperimentConfig(
            routing_strategy="round-robin",
            num_adapters=100,
            alpha=0.6,
            req_rate=4.0,
            duration=120,
            cv=1.0,
            input_range=(128, 512),
            output_range=(64, 256)
        )
        params = config.to_benchmark_args()
        
        assert params["num_adapters"] == 100
        assert params["alpha"] == 0.6
        assert params["req_rate"] == 4.0
        assert params["duration"] == 120
        assert params["cv"] == 1.0
        assert params["input_range"] == (128, 512)
        assert params["output_range"] == (64, 256)


class TestConfigPropertyBased:
    """Property-based tests for configuration validation"""
    
    @given(
        routing_strategy=st.text(),
        alpha=st.floats(allow_nan=False, allow_infinity=False),
        num_adapters=st.integers()
    )
    def test_config_validation_property(self, routing_strategy, alpha, num_adapters):
        """
        Feature: routing-comparison, Property 1: Configuration Validation
        
        For any ExperimentConfig with routing_strategy not in ["round-robin", "adapter-aware"],
        or alpha not in [0.1, 1.0], or num_adapters not in [10, 200],
        calling validate() SHALL raise a ValueError with a message containing
        the invalid parameter name.
        """
        config = ExperimentConfig(
            routing_strategy=routing_strategy,
            alpha=alpha,
            num_adapters=num_adapters,
            req_rate=4.0,
            duration=60
        )
        
        is_valid_strategy = routing_strategy in ["round-robin", "adapter-aware"]
        is_valid_alpha = 0.1 <= alpha <= 1.0
        is_valid_adapters = 10 <= num_adapters <= 200
        
        if is_valid_strategy and is_valid_alpha and is_valid_adapters:
            config.validate()  # Should not raise
        else:
            with pytest.raises(ValueError) as exc_info:
                config.validate()
            # Error message should mention the invalid parameter
            error_msg = str(exc_info.value).lower()
            if not is_valid_strategy:
                assert "routing" in error_msg or "strategy" in error_msg
            elif not is_valid_alpha:
                assert "alpha" in error_msg
            else:
                assert "adapter" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
