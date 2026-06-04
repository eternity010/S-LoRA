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

        # R-LoRA round-robin 基线（与 S-LoRA 基线参数对齐）
        # DP=3 round-robin，用于对比 DP 架构本身的收益（无智能路由）
        # 3 alphas = 3 experiments
        "dp-roundrobin-baseline": {
            "routing_strategy": ["round-robin"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
        },

        # 端到端性能基线对比：固定 alpha=0.3，改变请求到达率
        # 用于绘制 round-robin 的吞吐-延迟曲线
        # 5 request rates = 5 experiments
        "dp-roundrobin-rate-scaling": {
            "routing_strategy": ["round-robin"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [2.0, 4.0, 6.0, 8.0],
            "duration": [180],
        },

        # 单点复核 round-robin 在强热点 8 req/s 下的高压表现
        "dp-roundrobin-rate8-validation": {
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
        },

        # 端到端性能基线对比：固定 alpha=0.3，改变请求到达率
        # 与 dp-roundrobin-rate-scaling 完全对齐，仅切换为 adapter-aware + rwpt
        # 3 request rates = 3 experiments
        "dp-rwpt-rate-scaling": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [2.0, 4.0, 6.0, 8.0],
            "duration": [180],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
        },

        # 验证强热点 8 req/s 下 RWPT 尾延迟表现
        # 固定 alpha=0.1、req_rate=8.0，仅保留当前主线 w2
        # 1 experiment
        "dp-rwpt-rate8-w2-search": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
        },

        # 图 6 / 系统基线对比用的 RWPT 对照组
        # 与 dp-roundrobin-baseline 保持相同 alpha、req_rate、duration，仅切换为 adapter-aware + rwpt
        # 3 alphas = 3 experiments
        "dp-rwpt-baseline": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
        },

        # 图 6 / 系统基线对比用的 token_count 对照组
        # 与 dp-rwpt-baseline 保持相同 alpha、req_rate、duration，仅切换为 token_count + w2=1.5
        # 3 alphas = 3 experiments
        "dp-tokencount-baseline": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
            "routing_w1": [1.0],
            "routing_w2": [1.5],
            "load_metric": ["token_count"],
        },

        # 图 6 / 系统基线对比用的 queue_length 对照组
        # 与 dp-rwpt-baseline 保持相同 alpha、req_rate、duration，仅切换为 queue_length + w2=0.10
        # 3 alphas = 3 experiments
        "dp-queuelength-baseline": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
            "routing_w1": [1.0],
            "routing_w2": [0.10],
            "load_metric": ["queue_length"],
        },

        # 热门 Adapter 主动复制对比实验
        # 对照组 (enable_replication=False) vs 实验组 (enable_replication=True)
        # 在三种 Zipf 分布下对比 P90 TTFT、缓存命中率、吞吐量
        # 2 strategies × 3 alphas = 6 experiments
        "replication-comparison": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
            "enable_replication": [False, True],
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
        # Special suites with non-Cartesian configs
        if suite_name == "alpha-robustness":
            yield from cls.get_alpha_robustness_configs(**defaults)
            return
        if suite_name == "alpha-robustness-sla":
            yield from cls.get_alpha_robustness_sla_configs(**defaults)
            return
        
        suite = cls.SUITES.get(suite_name)
        if suite is None:
            available = ", ".join(list(cls.SUITES.keys()) + ["alpha-robustness", "alpha-robustness-sla"])
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
        return list(cls.SUITES.keys()) + ["alpha-robustness", "alpha-robustness-sla"]
    
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
        # Special suites
        if suite_name == "alpha-robustness":
            return {
                "name": suite_name,
                "parameters": {
                    "alpha": [0.1, 0.3, 0.8],
                    "metric_w2_pairs": [
                        ("rwpt", 0.5), ("token_count", 0.3), ("queue_length", 0.15),
                    ],
                },
                "config_count": 9,
            }
        if suite_name == "alpha-robustness-sla":
            return {
                "name": suite_name,
                "parameters": {
                    "alpha": [0.1, 0.3, 0.8],
                    "metric_w2_pairs": [
                        ("rwpt", 1.0), ("token_count", 1.5), ("queue_length", 0.10),
                    ],
                },
                "config_count": 9,
            }
        
        suite = cls.SUITES.get(suite_name)
        if suite is None:
            available = ", ".join(list(cls.SUITES.keys()) + ["alpha-robustness", "alpha-robustness-sla"])
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

    @classmethod
    def get_alpha_robustness_configs(cls, **defaults):
        """
        Alpha 鲁棒性实验（旧版，加权评分 w2）：固定每个 metric 的最优 w2，变化 alpha。
        3 metrics × 3 alphas = 9 experiments.

        w2 来源：早期加权评分选取（已被 SLA 版本替代，保留供对比）
          rwpt:         w2=0.5
          token_count:  w2=0.3
          queue_length: w2=0.15
        """
        metric_w2_pairs = [
            ("rwpt", 0.5),
            ("token_count", 0.3),
            ("queue_length", 0.15),
        ]
        alphas = [0.1, 0.3, 0.8]

        for alpha in alphas:
            for load_metric, w2 in metric_w2_pairs:
                yield ExperimentConfig(
                    routing_strategy="adapter-aware",
                    alpha=alpha,
                    num_adapters=100,
                    req_rate=6.0,
                    duration=120,
                    routing_w1=1.0,
                    routing_w2=w2,
                    routing_w3=0.0,
                    load_metric=load_metric,
                    **defaults,
                )

    @classmethod
    def get_alpha_robustness_sla_configs(cls, **defaults):
        """
        Alpha 鲁棒性实验（SLA 约束模型 w2）：固定每个 metric 的最优 w2，变化 alpha。
        3 metrics × 3 alphas = 9 experiments.

        w2 来源：SLA 约束最大化模型（ε=5% P90, δ=2% Tput），基于 max_lora_ratio=0.2 实验数据
          max Cache_Hit_Rate
          s.t. P90 ≤ P90_min*(1+0.05), Tput ≥ Tput_max*(1-0.02)

        选取结果：
          rwpt:         w2=1.0   (P90=14.14s, tput=5.553, cache=72.1%)
          token_count:  w2=1.5   (P90=14.68s, tput=5.508, cache=75.4%)
          queue_length: w2=0.10  (P90=15.83s, tput=5.504, cache=66.7%)
        """
        metric_w2_pairs = [
            ("rwpt", 1.0),
            ("token_count", 1.5),
            ("queue_length", 0.10),
        ]
        alphas = [0.1, 0.3, 0.8]

        for alpha in alphas:
            for load_metric, w2 in metric_w2_pairs:
                yield ExperimentConfig(
                    routing_strategy="adapter-aware",
                    alpha=alpha,
                    num_adapters=100,
                    req_rate=6.0,
                    duration=240,
                    routing_w1=1.0,
                    routing_w2=w2,
                    routing_w3=0.0,
                    load_metric=load_metric,
                    **defaults,
                )
