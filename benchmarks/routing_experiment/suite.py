"""
Experiment suite module for predefined experiment configurations.
"""

from typing import List, Iterator, Dict, Any
from itertools import product

from .config import ExperimentConfig


class ExperimentSuite:
    """Experiment suite definition and configuration generator"""
    
    # Predefined experiment suites
    SUITES: Dict[str, Dict[str, List[Any]]] = {
        "routing-alpha-comparison": {
            "routing_strategy": ["round-robin", "adapter-aware"],
            "alpha": [0.3, 0.6, 0.8, 1.0],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [60],
        },
        "routing-adapter-scaling": {
            "routing_strategy": ["round-robin", "adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [50, 100, 150, 200],
            "req_rate": [8.0],
            "duration": [120],
        },
        "routing-full-comparison": {
            "routing_strategy": ["round-robin", "adapter-aware"],
            "alpha": [0.3, 0.6, 1.0],
            "num_adapters": [50, 100],
            "req_rate": [8.0],
            "duration": [180],
        },
        "routing-weight-comparison": {
            # 仅测试 adapter-aware 策略下不同 w2 (负载惩罚权重) 组合
            # w1 保持默认值 1.0，只变化 w2
            # 服务器只需启动一次，w2 通过 API 动态更新
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [60],
            "routing_w1": [1.0],  # 缓存亲和性权重（固定）
            "routing_w2": [0.1, 0.2, 0.3 ,0.4, 0.5, 0.7, 0.8, 0.9],  # 负载惩罚权重（从0.1开始增大）
        },
    }
    
    @classmethod
    def get_configs(cls, suite_name: str, **defaults) -> Iterator[ExperimentConfig]:
        """
        Generate all experiment configurations for a given suite.
        
        Args:
            suite_name: Name of the predefined suite
            **defaults: Default values for parameters not specified in suite
        
        Yields:
            ExperimentConfig objects for each parameter combination
        
        Raises:
            ValueError: If suite_name is not found
        """
        suite = cls.SUITES.get(suite_name)
        if suite is None:
            available = ", ".join(cls.SUITES.keys())
            raise ValueError(
                f"Unknown suite: {suite_name}. Available suites: {available}"
            )
        
        # Get parameter names and their value lists
        keys = list(suite.keys())
        values = [suite[k] for k in keys]
        
        # Generate Cartesian product of all parameter combinations
        for combo in product(*values):
            params = dict(zip(keys, combo))
            
            # Merge with defaults
            config_params = {**defaults, **params}
            
            yield ExperimentConfig(**config_params)
    
    @classmethod
    def list_suites(cls) -> List[str]:
        """
        List all available predefined suites.
        
        Returns:
            List of suite names
        """
        return list(cls.SUITES.keys())
    
    @classmethod
    def get_suite_info(cls, suite_name: str) -> Dict[str, Any]:
        """
        Get information about a specific suite.
        
        Args:
            suite_name: Name of the suite
        
        Returns:
            Dictionary containing suite parameters and expected config count
        
        Raises:
            ValueError: If suite_name is not found
        """
        suite = cls.SUITES.get(suite_name)
        if suite is None:
            available = ", ".join(cls.SUITES.keys())
            raise ValueError(
                f"Unknown suite: {suite_name}. Available suites: {available}"
            )
        
        # Calculate expected number of configurations
        config_count = 1
        for values in suite.values():
            config_count *= len(values)
        
        return {
            "name": suite_name,
            "parameters": suite,
            "config_count": config_count,
        }
    
    @classmethod
    def add_custom_suite(cls, suite_name: str, suite_def: Dict[str, List[Any]]) -> None:
        """
        Add a custom experiment suite.
        
        Args:
            suite_name: Name for the new suite
            suite_def: Dictionary mapping parameter names to value lists
        
        Raises:
            ValueError: If suite_name already exists
        """
        if suite_name in cls.SUITES:
            raise ValueError(f"Suite '{suite_name}' already exists")
        
        # Validate that required parameters are present
        required_params = {"routing_strategy", "alpha", "num_adapters", "req_rate", "duration"}
        provided_params = set(suite_def.keys())
        
        if not required_params.issubset(provided_params):
            missing = required_params - provided_params
            raise ValueError(f"Missing required parameters: {missing}")
        
        cls.SUITES[suite_name] = suite_def
