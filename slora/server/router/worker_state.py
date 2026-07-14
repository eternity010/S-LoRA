"""
Worker State Data Structures for Adapter-Aware Routing

This module defines the core data structures used for tracking Worker states,
routing statistics, and configuration for the intelligent routing strategy.

Requirements: 1.1, 1.2, 6.1-6.5
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Optional, List, Tuple
import time


@dataclass
class WorkerState:
    """
    Worker 状态信息
    
    Tracks the current state of a GPU Worker including its cached adapters,
    queue length, health status, and batch rank distribution.
    
    Attributes:
        worker_id: Worker 唯一标识
        cached_adapters: 已缓存的 Adapter 目录集合
        queue_length: 当前队列长度
        gpu_memory_free: 可用 GPU 显存（bytes）
        last_heartbeat: 最后心跳时间戳
        is_healthy: 是否健康
        avg_rank: 当前批次的平均 LoRA rank
        min_rank: 当前批次的最小 LoRA rank
        max_rank: 当前批次的最大 LoRA rank
    
    Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 8.2
    """
    worker_id: int
    cached_adapters: Set[str] = field(default_factory=set)
    queue_length: int = 0
    gpu_memory_free: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    is_healthy: bool = True
    # Rank distribution fields for rank-aware routing
    avg_rank: float = 0.0      # Average rank of current batch
    min_rank: int = 0          # Minimum rank in current batch
    max_rank: int = 0          # Maximum rank in current batch
    # RWPT (Rank-Calibrated Workload) fields
    pending_prefill_tokens: int = 0    # 等待队列中 prompt token 总数（rank 加权）
    pending_raw_tokens: int = 0        # 等待队列中 prompt token 总数（不含 rank 加权）
    active_decode_seqs: int = 0        # 当前 batch 中 decode 序列数
    pool_used_ratio: float = 0.0       # 内存池使用率 (0.0-1.0)
    # State-report diagnostics. These fields do not participate in scoring.
    report_seq: int = 0
    worker_report_time: float = 0.0
    router_received_time: float = 0.0
    waiting_request_count: int = 0
    current_batch_size: int = 0
    current_batch_prompt_tokens: int = 0
    optimistic_request_count: int = 0
    optimistic_raw_tokens: int = 0
    optimistic_prefill_tokens: int = 0
    # Hot adapter replication: top-K RWPT contributors
    top_k_rwpt_adapters: List[Tuple[str, float]] = field(default_factory=list)  # [(adapter_dir, rwpt_contribution), ...]
    
    def has_adapter(self, adapter_dir: str) -> bool:
        """检查是否缓存了指定的 Adapter"""
        return adapter_dir in self.cached_adapters
    
    def update_heartbeat(self) -> None:
        """更新心跳时间戳"""
        self.last_heartbeat = time.time()
    
    def to_dict(self) -> dict:
        """转换为字典格式（用于序列化）"""
        return {
            'worker_id': self.worker_id,
            'cached_adapters': list(self.cached_adapters),
            'queue_length': self.queue_length,
            'gpu_memory_free': self.gpu_memory_free,
            'last_heartbeat': self.last_heartbeat,
            'is_healthy': self.is_healthy,
            'avg_rank': self.avg_rank,
            'min_rank': self.min_rank,
            'max_rank': self.max_rank,
            'pending_prefill_tokens': self.pending_prefill_tokens,
            'pending_raw_tokens': self.pending_raw_tokens,
            'active_decode_seqs': self.active_decode_seqs,
            'pool_used_ratio': self.pool_used_ratio,
            'report_seq': self.report_seq,
            'worker_report_time': self.worker_report_time,
            'router_received_time': self.router_received_time,
            'waiting_request_count': self.waiting_request_count,
            'current_batch_size': self.current_batch_size,
            'current_batch_prompt_tokens': self.current_batch_prompt_tokens,
            'optimistic_request_count': self.optimistic_request_count,
            'optimistic_raw_tokens': self.optimistic_raw_tokens,
            'optimistic_prefill_tokens': self.optimistic_prefill_tokens,
            'top_k_rwpt_adapters': [list(t) for t in self.top_k_rwpt_adapters],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WorkerState':
        """从字典创建 WorkerState 实例（向后兼容，支持无新字段的旧格式）"""
        return cls(
            worker_id=data['worker_id'],
            cached_adapters=set(data.get('cached_adapters', [])),
            queue_length=data.get('queue_length', 0),
            gpu_memory_free=data.get('gpu_memory_free', 0),
            last_heartbeat=data.get('last_heartbeat', time.time()),
            is_healthy=data.get('is_healthy', True),
            avg_rank=data.get('avg_rank', 0.0),
            min_rank=data.get('min_rank', 0),
            max_rank=data.get('max_rank', 0),
            pending_prefill_tokens=data.get('pending_prefill_tokens', 0),
            pending_raw_tokens=data.get('pending_raw_tokens', 0),
            active_decode_seqs=data.get('active_decode_seqs', 0),
            pool_used_ratio=data.get('pool_used_ratio', 0.0),
            report_seq=data.get('report_seq', 0),
            worker_report_time=data.get('worker_report_time', 0.0),
            router_received_time=data.get('router_received_time', 0.0),
            waiting_request_count=data.get('waiting_request_count', 0),
            current_batch_size=data.get('current_batch_size', 0),
            current_batch_prompt_tokens=data.get('current_batch_prompt_tokens', 0),
            optimistic_request_count=data.get('optimistic_request_count', 0),
            optimistic_raw_tokens=data.get('optimistic_raw_tokens', 0),
            optimistic_prefill_tokens=data.get('optimistic_prefill_tokens', 0),
            top_k_rwpt_adapters=[tuple(t) for t in data.get('top_k_rwpt_adapters', [])],
        )


@dataclass
class RoutingStats:
    """
    路由统计信息
    
    Tracks routing decisions and cache performance metrics.
    
    Attributes:
        total_requests: 总请求数
        cache_hits: 缓存命中次数
        cache_misses: 缓存未命中次数
        cold_starts: 冷启动次数
        hot_adapter_redistributions: 热点 Adapter 重分配次数
        worker_request_counts: 每个 Worker 的请求计数
        rank_matched_count: Rank 匹配的路由次数（不匹配度低于阈值）
        total_rank_mismatch: 总 rank 不匹配度
    
    Requirements: 7.1, 7.2, 7.3
    """
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cold_starts: int = 0
    hot_adapter_redistributions: int = 0
    worker_request_counts: Dict[int, int] = field(default_factory=dict)
    # Rank-aware statistics
    rank_matched_count: int = 0           # Requests routed to similar-rank Workers
    total_rank_mismatch: float = 0.0      # Sum of all rank mismatches
    
    @property
    def cache_hit_rate(self) -> float:
        """
        计算缓存命中率
        
        Returns:
            缓存命中率（0.0 - 1.0）
        
        Requirements: 7.1
        """
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def cache_miss_rate(self) -> float:
        """计算缓存未命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_misses / self.total_requests
    
    @property
    def avg_rank_mismatch(self) -> float:
        """
        计算平均 rank 不匹配度
        
        Returns:
            平均 rank 不匹配度（0.0 - 1.0）
        
        Requirements: 7.2
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_rank_mismatch / self.total_requests
    
    def record_request(self, worker_id: int, cache_hit: bool, is_cold_start: bool = False) -> None:
        """
        记录一次路由请求
        
        Args:
            worker_id: 选中的 Worker ID
            cache_hit: 是否缓存命中
            is_cold_start: 是否为冷启动
        """
        self.total_requests += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        if is_cold_start:
            self.cold_starts += 1
        
        # 更新 Worker 请求计数
        if worker_id not in self.worker_request_counts:
            self.worker_request_counts[worker_id] = 0
        self.worker_request_counts[worker_id] += 1
    
    def record_rank_mismatch(self, mismatch: float, threshold: float = 0.1) -> None:
        """
        记录一次 rank 不匹配度
        
        Args:
            mismatch: rank 不匹配度（0.0 - 1.0）
            threshold: 判定为"匹配"的阈值，默认 0.1
        
        Requirements: 7.1, 7.2
        """
        self.total_rank_mismatch += mismatch
        if mismatch <= threshold:
            self.rank_matched_count += 1
    
    def record_hot_adapter_redistribution(self) -> None:
        """记录一次热点 Adapter 重分配"""
        self.hot_adapter_redistributions += 1
    
    def reset(self) -> None:
        """重置所有统计数据"""
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cold_starts = 0
        self.hot_adapter_redistributions = 0
        self.worker_request_counts.clear()
        self.rank_matched_count = 0
        self.total_rank_mismatch = 0.0
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cold_starts': self.cold_starts,
            'hot_adapter_redistributions': self.hot_adapter_redistributions,
            'cache_hit_rate': self.cache_hit_rate,
            'worker_request_counts': dict(self.worker_request_counts),
            'rank_matched_count': self.rank_matched_count,
            'total_rank_mismatch': self.total_rank_mismatch,
            'avg_rank_mismatch': self.avg_rank_mismatch
        }
    
    def __str__(self) -> str:
        """格式化输出统计信息"""
        return (
            f"RoutingStats(total={self.total_requests}, "
            f"hits={self.cache_hits}, misses={self.cache_misses}, "
            f"hit_rate={self.cache_hit_rate:.2%}, "
            f"cold_starts={self.cold_starts})"
        )


@dataclass
class RoutingConfig:
    """
    路由配置
    
    Configuration parameters for the Adapter-Aware Router.
    
    Attributes:
        strategy: 路由策略 ('adapter-aware' or 'round-robin')
        w1: 缓存亲和性权重
        w2: 负载惩罚权重
        w3: Rank 不匹配惩罚权重（用于 rank 感知路由）
        default_lora_rank: 未知 adapter 的默认 LoRA rank
        max_rank_diff: 用于归一化 rank 差异的最大值
        heartbeat_interval_ms: 心跳间隔（毫秒）
        heartbeat_timeout_ms: 心跳超时（毫秒）
        max_queue_length: 最大队列长度阈值
        hot_adapter_threshold: 热点 Adapter 请求率阈值（req/s）
    
    Requirements: 6.1-6.5
    """
    strategy: str = 'adapter-aware'
    w1: float = 1.0
    w2: float = 1.0
    w3: float = 0.0                    # Rank mismatch penalty weight (NEW)
    default_lora_rank: int = 16        # Default rank for unknown adapters (NEW)
    max_rank_diff: int = 64            # Max rank difference for normalization (NEW)
    heartbeat_interval_ms: int = 100
    heartbeat_timeout_ms: int = 300
    max_queue_length: int = 100
    hot_adapter_threshold: float = 1e9
    # RWPT (Rank-Calibrated Workload) parameters
    hidden_dim: int = 4096             # 模型隐藏层维度，用于计算 γ
    decode_cost_alpha: float = None     # Decode 序列负载折算系数（None 时由 Worker profiling 自动测量）
    max_total_token_num: int = 6000    # KV Cache capacity in tokens (from --max_total_token_num)
    batch_max_tokens: int = 1000       # 单次 prefill 批次最大 token 数，RWPT 归一化分母
    # Load metric ablation
    load_metric: str = 'rwpt'          # 负载度量类型: 'queue_length' | 'token_count' | 'rwpt'
    
    # 有效的 load_metric 取值
    VALID_LOAD_METRICS = ('queue_length', 'token_count', 'rwpt')
    
    def __post_init__(self):
        """验证配置参数"""
        if self.strategy not in ('adapter-aware', 'round-robin'):
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be 'adapter-aware' or 'round-robin'")
        if self.w1 < 0:
            raise ValueError(f"w1 must be non-negative, got {self.w1}")
        if self.w2 < 0:
            raise ValueError(f"w2 must be non-negative, got {self.w2}")
        if self.w3 < 0:
            raise ValueError(f"w3 must be non-negative, got {self.w3}")
        if self.default_lora_rank <= 0:
            raise ValueError(f"default_lora_rank must be positive, got {self.default_lora_rank}")
        if self.max_rank_diff <= 0:
            raise ValueError(f"max_rank_diff must be positive, got {self.max_rank_diff}")
        if self.heartbeat_interval_ms <= 0:
            raise ValueError(f"heartbeat_interval_ms must be positive, got {self.heartbeat_interval_ms}")
        if self.heartbeat_timeout_ms <= 0:
            raise ValueError(f"heartbeat_timeout_ms must be positive, got {self.heartbeat_timeout_ms}")
        if self.max_queue_length <= 0:
            raise ValueError(f"max_queue_length must be positive, got {self.max_queue_length}")
        if self.hot_adapter_threshold <= 0:
            raise ValueError(f"hot_adapter_threshold must be positive, got {self.hot_adapter_threshold}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.decode_cost_alpha is not None and self.decode_cost_alpha < 0:
            raise ValueError(f"decode_cost_alpha must be non-negative or None, got {self.decode_cost_alpha}")
        if self.max_total_token_num <= 0:
            raise ValueError(f"max_total_token_num must be positive, got {self.max_total_token_num}")
        if self.batch_max_tokens <= 0:
            raise ValueError(f"batch_max_tokens must be positive, got {self.batch_max_tokens}")
        if self.load_metric not in self.VALID_LOAD_METRICS:
            raise ValueError(f"Invalid load_metric: {self.load_metric}. Must be one of {self.VALID_LOAD_METRICS}")
    
    @property
    def heartbeat_interval_sec(self) -> float:
        """心跳间隔（秒）"""
        return self.heartbeat_interval_ms / 1000.0
    
    @property
    def heartbeat_timeout_sec(self) -> float:
        """心跳超时（秒）"""
        return self.heartbeat_timeout_ms / 1000.0
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'strategy': self.strategy,
            'w1': self.w1,
            'w2': self.w2,
            'w3': self.w3,
            'default_lora_rank': self.default_lora_rank,
            'max_rank_diff': self.max_rank_diff,
            'heartbeat_interval_ms': self.heartbeat_interval_ms,
            'heartbeat_timeout_ms': self.heartbeat_timeout_ms,
            'max_queue_length': self.max_queue_length,
            'hot_adapter_threshold': self.hot_adapter_threshold,
            'hidden_dim': self.hidden_dim,
            'decode_cost_alpha': self.decode_cost_alpha,
            'max_total_token_num': self.max_total_token_num,
            'batch_max_tokens': self.batch_max_tokens,
            'load_metric': self.load_metric,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RoutingConfig':
        """从字典创建 RoutingConfig 实例（向后兼容）"""
        return cls(
            strategy=data.get('strategy', 'adapter-aware'),
            w1=data.get('w1', 1.0),
            w2=data.get('w2', 1.0),
            w3=data.get('w3', 0.0),
            default_lora_rank=data.get('default_lora_rank', 16),
            max_rank_diff=data.get('max_rank_diff', 64),
            heartbeat_interval_ms=data.get('heartbeat_interval_ms', 100),
            heartbeat_timeout_ms=data.get('heartbeat_timeout_ms', 300),
            max_queue_length=data.get('max_queue_length', 100),
            hot_adapter_threshold=data.get('hot_adapter_threshold', 1e9),
            hidden_dim=data.get('hidden_dim', 4096),
            decode_cost_alpha=data.get('decode_cost_alpha', None),
            max_total_token_num=data.get('max_total_token_num', 6000),
            batch_max_tokens=data.get('batch_max_tokens', 1000),
            load_metric=data.get('load_metric', 'rwpt'),
        )
