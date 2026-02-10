"""
Worker State Data Structures for Adapter-Aware Routing

This module defines the core data structures used for tracking Worker states,
routing statistics, and configuration for the intelligent routing strategy.

Requirements: 1.1, 1.2, 6.1-6.5
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Optional
import time


@dataclass
class WorkerState:
    """
    Worker 状态信息
    
    Tracks the current state of a GPU Worker including its cached adapters,
    queue length, and health status.
    
    Attributes:
        worker_id: Worker 唯一标识
        cached_adapters: 已缓存的 Adapter 目录集合
        queue_length: 当前队列长度
        gpu_memory_free: 可用 GPU 显存（bytes）
        last_heartbeat: 最后心跳时间戳
        is_healthy: 是否健康
    
    Requirements: 1.1, 1.2
    """
    worker_id: int
    cached_adapters: Set[str] = field(default_factory=set)
    queue_length: int = 0
    gpu_memory_free: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    is_healthy: bool = True
    
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
            'is_healthy': self.is_healthy
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WorkerState':
        """从字典创建 WorkerState 实例"""
        return cls(
            worker_id=data['worker_id'],
            cached_adapters=set(data.get('cached_adapters', [])),
            queue_length=data.get('queue_length', 0),
            gpu_memory_free=data.get('gpu_memory_free', 0),
            last_heartbeat=data.get('last_heartbeat', time.time()),
            is_healthy=data.get('is_healthy', True)
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
    
    Requirements: 7.1, 7.2, 7.3
    """
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cold_starts: int = 0
    hot_adapter_redistributions: int = 0
    worker_request_counts: Dict[int, int] = field(default_factory=dict)
    
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
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cold_starts': self.cold_starts,
            'hot_adapter_redistributions': self.hot_adapter_redistributions,
            'cache_hit_rate': self.cache_hit_rate,
            'worker_request_counts': dict(self.worker_request_counts)
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
        heartbeat_interval_ms: 心跳间隔（毫秒）
        heartbeat_timeout_ms: 心跳超时（毫秒）
        max_queue_length: 最大队列长度阈值
        hot_adapter_threshold: 热点 Adapter 请求率阈值（req/s）
    
    Requirements: 6.1-6.5
    """
    strategy: str = 'adapter-aware'
    w1: float = 1.0
    w2: float = 0.1
    heartbeat_interval_ms: int = 100
    heartbeat_timeout_ms: int = 300
    max_queue_length: int = 100
    hot_adapter_threshold: float = 10.0
    
    def __post_init__(self):
        """验证配置参数"""
        if self.strategy not in ('adapter-aware', 'round-robin'):
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be 'adapter-aware' or 'round-robin'")
        if self.w1 < 0:
            raise ValueError(f"w1 must be non-negative, got {self.w1}")
        if self.w2 < 0:
            raise ValueError(f"w2 must be non-negative, got {self.w2}")
        if self.heartbeat_interval_ms <= 0:
            raise ValueError(f"heartbeat_interval_ms must be positive, got {self.heartbeat_interval_ms}")
        if self.heartbeat_timeout_ms <= 0:
            raise ValueError(f"heartbeat_timeout_ms must be positive, got {self.heartbeat_timeout_ms}")
        if self.max_queue_length <= 0:
            raise ValueError(f"max_queue_length must be positive, got {self.max_queue_length}")
        if self.hot_adapter_threshold <= 0:
            raise ValueError(f"hot_adapter_threshold must be positive, got {self.hot_adapter_threshold}")
    
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
            'heartbeat_interval_ms': self.heartbeat_interval_ms,
            'heartbeat_timeout_ms': self.heartbeat_timeout_ms,
            'max_queue_length': self.max_queue_length,
            'hot_adapter_threshold': self.hot_adapter_threshold
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RoutingConfig':
        """从字典创建 RoutingConfig 实例"""
        return cls(
            strategy=data.get('strategy', 'adapter-aware'),
            w1=data.get('w1', 1.0),
            w2=data.get('w2', 0.1),
            heartbeat_interval_ms=data.get('heartbeat_interval_ms', 100),
            heartbeat_timeout_ms=data.get('heartbeat_timeout_ms', 300),
            max_queue_length=data.get('max_queue_length', 100),
            hot_adapter_threshold=data.get('hot_adapter_threshold', 10.0)
        )
