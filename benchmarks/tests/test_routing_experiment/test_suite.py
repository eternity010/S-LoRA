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


class TestExperimentSuite:
    """Unit tests for experiment suite"""
    
    def test_list_suites(self):
        """Test listing all available suites"""
        suites = ExperimentSuite.list_suites()
        
        assert isinstance(suites, list)
        assert len(suites) >= 3
        assert "routing-alpha-comparison" in suites
        assert "routing-adapter-scaling" in suites
        assert "routing-full-comparison" in suites
    
    def test_routing_alpha_comparison_suite(self):
        """Test routing-alpha-comparison suite generates correct configs"""
        configs = list(ExperimentSuite.get_configs("routing-alpha-comparison"))
        
        # Should have 2 strategies × 4 alphas × 1 adapter × 1 rate × 1 duration = 8 configs
        assert len(configs) == 8
        
        # Check all configs are valid
        for config in configs:
            assert isinstance(config, ExperimentConfig)
            config.validate()  # Should not raise
        
        # Check parameter coverage
        strategies = {c.routing_strategy for c in configs}
        alphas = {c.alpha for c in configs}
        
        assert strategies == {"round-robin", "adapter-aware"}
        assert alphas == {0.3, 0.6, 0.8, 1.0}
        assert all(c.num_adapters == 100 for c in configs)
    
    def test_routing_adapter_scaling_suite(self):
        """Test routing-adapter-scaling suite generates correct configs"""
        configs = list(ExperimentSuite.get_configs("routing-adapter-scaling"))
        
        # Should have 2 strategies × 1 alpha × 4 adapters × 1 rate × 1 duration = 8 configs
        assert len(configs) == 8
        
        # Check all configs are valid
        for config in configs:
            assert isinstance(config, ExperimentConfig)
            config.validate()  # Should not raise
        
        # Check parameter coverage
        strategies = {c.routing_strategy for c in configs}
        adapters = {c.num_adapters for c in configs}
        
        assert strategies == {"round-robin", "adapter-aware"}
        assert adapters == {20, 50, 100, 150}
        assert all(c.alpha == 0.6 for c in configs)
    
    def test_routing_full_comparison_suite(self):
        """Test routing-full-comparison suite generates correct configs"""
        configs = list(ExperimentSuite.get_configs("routing-full-comparison"))
        
        # Should have 2 strategies × 3 alphas × 2 adapters × 1 rate × 1 duration = 12 configs
        assert len(configs) == 12
        
        # Check all configs are valid
        for config in configs:
            assert isinstance(config, ExperimentConfig)
            config.validate()  # Should not raise
        
        # Check parameter coverage
        strategies = {c.routing_strategy for c in configs}
        alphas = {c.alpha for c in configs}
        adapters = {c.num_adapters for c in configs}
        
        assert strategies == {"round-robin", "adapter-aware"}
        assert alphas == {0.3, 0.6, 1.0}
        assert adapters == {50, 100}
    
    def test_unknown_suite_raises_error(self):
        """Test requesting unknown suite raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            list(ExperimentSuite.get_configs("nonexistent-suite"))
        
        error_msg = str(exc_info.value)
        assert "unknown suite" in error_msg.lower()
        assert "nonexistent-suite" in error_msg
    
    def test_get_suite_info(self):
        """Test getting suite information"""
        info = ExperimentSuite.get_suite_info("routing-alpha-comparison")
        
        assert info["name"] == "routing-alpha-comparison"
        assert "parameters" in info
        assert "config_count" in info
        assert info["config_count"] == 8
    
    def test_get_suite_info_unknown_suite(self):
        """Test getting info for unknown suite raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            ExperimentSuite.get_suite_info("nonexistent-suite")
        
        assert "unknown suite" in str(exc_info.value).lower()
    
    def test_configs_with_defaults(self):
        """Test generating configs with default parameters"""
        configs = list(ExperimentSuite.get_configs(
            "routing-alpha-comparison",
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
            ExperimentSuite.add_custom_suite("routing-alpha-comparison", custom_suite)
        
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
            "routing-alpha-comparison",
            "routing-adapter-scaling",
            "routing-full-comparison"
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
