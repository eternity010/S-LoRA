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
            "duration": [120],
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
            # [已过期] alpha=0.1 回退值下的 w2 搜索结果，保留供参考
            # 甜点 w2≈4.0（alpha=0.1 时 RWPT/Cap 偏小，需要较大 w2 补偿）
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "routing_w1": [1.0],
            "routing_w2": [4.5, 4.7, 5.0, 5.2, 5.5],
        },
        "routing-weight-v2": {
            # [历史] alpha 修正后的 w2 甜点重搜
            # Score = w1·Cache - w2·(RWPT/Capacity)
            # RWPT = pending_prefill_tokens（仅 prefill token，无 decode 折算）
            # Capacity = batch_max_tokens ≈ 2500
            #
            # 量纲分析（avg≈320 tokens/req）：
            #   3 req → RWPT/Cap ≈ 0.38
            #   8 req → RWPT/Cap ≈ 1.02
            # 预判甜点 w2 ∈ [1.5, 4.0]
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "routing_w1": [1.0],
            "routing_w2": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        },
        "load-metric-ablation": {
            # 负载度量消融实验：对比三种负载度量粒度
            # queue_length: V2 基线（仅请求数）
            # token_count:  token 级但无 rank 加权（消融 γ·r 项）
            # rwpt:         完整 V3 RWPT（默认）
            #
            # w2=3.0: alpha 修正后（decode_cost_alpha≈90）的甜点值
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [120],
            "routing_w1": [1.0],
            "routing_w2": [0.8],
            "load_metric": ["rwpt", "queue_length", "token_count"],
        },
        "rwpt-w2-search": {
            # RWPT 专用 w2 甜点搜索（移除 decode 项后）
            # load-metric-w2-sweep 结果显示 rwpt 在 w2=0.8 时最优（5.21/20.5s）
            # 但仍差于 queue_length/token_count，需要在更小 w2 范围细搜
            # RWPT/Capacity 值域 ~[0, 0.5]（纯 prefill token，无 decode 折算）
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [120],
            "routing_w1": [1.0],
            "routing_w2": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            "load_metric": ["rwpt"],
        },
        "ql-w2-search": {
            # queue_length 专用 w2 甜点搜索
            # queue_length 模式无 Capacity 归一化：Score -= w2 * QueueLen
            # QueueLen 通常 0~5，cache 命中得分 = w1 = 1.0
            # 要让负载惩罚与缓存亲和性竞争：w2 * QueueLen ≈ 1.0
            # → QueueLen=3 时 w2≈0.3 为平衡点
            # 搜索范围 w2 ∈ [0.05, 0.8]，覆盖"几乎纯缓存"到"强负载均衡"
            # 10 experiments
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [120],
            "routing_w1": [1.0],
            "routing_w2": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "load_metric": ["queue_length"],
        },
        "tc-w2-search": {
            # token_count 专用 w2 甜点搜索
            # token_count 模式：pending_raw_tokens / Capacity（无 rank 加权，无 decode 折算）
            # 与 rwpt 归一化方式相同（除以 batch_max_tokens≈2500），但不乘 (1+γ·r)
            # 值域与 rwpt 接近，预期甜点区间也接近
            # rwpt 甜点 w2∈[0.5, 2.5]，token_count 搜索范围覆盖 [0.3, 4.0]
            # 10 experiments
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [120],
            "routing_w1": [1.0],
            "routing_w2": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "load_metric": ["token_count"],
        },
        "best-w2-comparison": {
            # 三种负载度量各取最优 w2 的公平对比
            # 在同一次服务器启动下连续跑，消除环境噪声
            # 最优 w2 来源：
            #   rwpt:         w2=2.0  (tput=5.65, lat=10.86s)  ← rwpt-w2-search 第二轮
            #   queue_length: w2=0.05 (tput=5.54, lat=11.56s)  ← ql-w2-search
            #   token_count:  w2=0.8  (tput=5.50, lat=13.03s)  ← tc-w2-search
            #
            # 注意：这里用 Cartesian product 会产生 3×3=9 组合，
            # 但我们只需要 3 个有效组合（每个 metric 对应自己的最优 w2）。
            # 所以不能用 suite 的笛卡尔积机制，需要手动定义。
            # 折中方案：分别定义三个单点 suite，或者用 load-metric-ablation 风格
            # 但 suite 机制不支持 (metric, w2) 配对，所以这里用三个固定 w2 值
            # 配合三个 metric，产生 9 个实验，其中 3 个是有效对比点。
            # 多出的 6 个实验也有参考价值（交叉验证）。
            # 9 experiments
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [120],
            "routing_w1": [1.0],
            "routing_w2": [0.05, 0.8, 2.0],
            "load_metric": ["queue_length", "token_count", "rwpt"],
        },
        "load-metric-w2-sweep": {
            # 三种负载度量的综合 w2 甜点搜索
            # 目标：每种 metric 在各自最优 w2 下对比
            #
            # queue_length: 无 Capacity 归一化，w2 直接乘 QueueLen
            #   → 甜点区间偏小（QueueLen 通常 0-5）
            # token_count: pending_raw_tokens / Capacity（无 rank 加权，无 decode 折算）
            # rwpt: pending_prefill_tokens / Capacity（rank 加权，无 decode 折算）
            #   → token_count/rwpt 归一化到 ~[0,2]
            #
            # 覆盖范围：w2 ∈ [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
            # 3 metrics × 7 w2 = 21 experiments
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [120],
            "routing_w1": [1.0],
            "routing_w2": [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0],
            "load_metric": ["queue_length", "token_count", "rwpt"],
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
