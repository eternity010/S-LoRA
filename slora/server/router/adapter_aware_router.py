"""
Adapter-Aware Router for Intelligent Request Routing

This module implements the core routing logic that considers cache affinity
and load balancing to optimize request distribution across GPU Workers.

Requirements: 2.1-2.5, 3.1-3.4, 5.1-5.3
"""

import time
import logging
from typing import Dict, List, Optional, Set, Deque
from collections import defaultdict, deque

from .worker_state import WorkerState, RoutingStats, RoutingConfig


logger = logging.getLogger(__name__)


class RequestRateTracker:
    """
    请求率追踪器 - 使用滑动窗口计算请求率
    
    Requirements: 5.2
    """
    
    def __init__(self, window_size_sec: float = 1.0):
        """
        初始化请求率追踪器
        
        Args:
            window_size_sec: 滑动窗口大小（秒）
        """
        self.window_size_sec = window_size_sec
        # adapter_dir -> deque of timestamps
        self._request_times: Dict[str, Deque[float]] = defaultdict(deque)
    
    def record_request(self, adapter_dir: str) -> None:
        """记录一次请求"""
        current_time = time.time()
        self._request_times[adapter_dir].append(current_time)
        self._cleanup(adapter_dir, current_time)
    
    def get_request_rate(self, adapter_dir: str) -> float:
        """
        获取指定 Adapter 的请求率 (req/s)
        
        Args:
            adapter_dir: Adapter 目录
            
        Returns:
            请求率（每秒请求数）
        """
        current_time = time.time()
        self._cleanup(adapter_dir, current_time)
        
        timestamps = self._request_times.get(adapter_dir)
        if not timestamps:
            return 0.0
        
        return len(timestamps) / self.window_size_sec
    
    def _cleanup(self, adapter_dir: str, current_time: float) -> None:
        """清理过期的时间戳"""
        timestamps = self._request_times.get(adapter_dir)
        if not timestamps:
            return
        
        cutoff = current_time - self.window_size_sec
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        
        # 如果队列为空，删除该条目
        if not timestamps:
            del self._request_times[adapter_dir]
    
    def is_hot(self, adapter_dir: str, threshold: float) -> bool:
        """
        检查 Adapter 是否为热点
        
        Args:
            adapter_dir: Adapter 目录
            threshold: 热点阈值 (req/s)
            
        Returns:
            是否为热点
        """
        return self.get_request_rate(adapter_dir) >= threshold
    
    def reset(self) -> None:
        """重置追踪器"""
        self._request_times.clear()


class AdapterAwareRouter:
    """
    基于亲和性的智能路由器
    
    Implements the scoring function:
    Score_i(r) = w1 · I(adapter ∈ Cache_i) - w2 · QueueLen_i
    
    Attributes:
        num_workers: Worker 数量
        config: 路由配置
        worker_states: Worker 状态字典
        adapter_to_workers: Adapter 到 Worker 的倒排索引
        stats: 路由统计信息
    
    Requirements: 2.1, 2.2, 3.1, 3.2
    """
    
    def __init__(self, 
                 num_workers: int,
                 config: Optional[RoutingConfig] = None):
        """
        初始化路由器
        
        Args:
            num_workers: Worker 数量
            config: 路由配置，如果为 None 则使用默认配置
        """
        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers}")
        
        self.num_workers = num_workers
        self.config = config or RoutingConfig()
        
        # Worker 状态缓存
        self.worker_states: Dict[int, WorkerState] = {}
        for i in range(num_workers):
            self.worker_states[i] = WorkerState(worker_id=i)
        
        # Adapter -> Worker 倒排索引
        self.adapter_to_workers: Dict[str, Set[int]] = defaultdict(set)
        
        # 路由统计
        self.stats = RoutingStats()
        
        # 请求率追踪器（用于热点检测）
        self._rate_tracker = RequestRateTracker(window_size_sec=1.0)
        
        # 热点 Adapter 的 Round-Robin 计数器
        self._hot_adapter_rr: Dict[str, int] = defaultdict(int)
        
        # Round-Robin 计数器（用于回退模式）
        self._rr_counter = 0
        
        # 定期统计输出时间戳
        self._last_stats_log_time = time.time()
        self._stats_log_interval = 10.0  # 每 10 秒输出一次统计
        
        logger.info(f"AdapterAwareRouter initialized with {num_workers} workers, "
                   f"strategy={self.config.strategy}, w1={self.config.w1}, w2={self.config.w2}")
    
    def calculate_score(self, worker_id: int, adapter_dir: str) -> float:
        """
        计算 Worker 对请求的评分
        
        Score_i(r) = w1 · I(adapter ∈ Cache_i) - w2 · QueueLen_i
        
        Args:
            worker_id: Worker ID
            adapter_dir: Adapter 目录
            
        Returns:
            评分值
        
        Requirements: 2.2
        """
        state = self.worker_states.get(worker_id)
        if state is None:
            return float('-inf')
        
        # I(adapter ∈ Cache_i): 指示函数
        cache_indicator = 1.0 if state.has_adapter(adapter_dir) else 0.0
        
        # 评分公式
        score = self.config.w1 * cache_indicator - self.config.w2 * state.queue_length
        
        return score
    
    def select_worker(self, adapter_dir: str) -> int:
        """
        选择最佳 Worker
        
        Args:
            adapter_dir: 请求的 Adapter 目录
            
        Returns:
            选中的 Worker ID
        
        Requirements: 2.3, 2.4, 5.1, 5.3
        """
        # 如果使用 Round-Robin 策略，直接轮询
        if self.config.strategy == 'round-robin':
            return self._round_robin_select()
        
        # 记录请求用于热点检测
        self._rate_tracker.record_request(adapter_dir)
        
        # 获取健康的 Worker 列表
        healthy_workers = self._get_healthy_workers()
        
        if not healthy_workers:
            logger.warning("No healthy workers available, falling back to round-robin")
            return self._round_robin_select()
        
        # 检查是否为热点 Adapter
        is_hot_adapter = self._rate_tracker.is_hot(
            adapter_dir, self.config.hot_adapter_threshold
        )
        
        # 热点 Adapter 处理：分散到多个 Worker
        if is_hot_adapter:
            selected = self._select_for_hot_adapter(adapter_dir, healthy_workers)
            if selected is not None:
                return selected
        
        # 计算每个 Worker 的评分
        scores: List[tuple] = []
        for worker_id in healthy_workers:
            state = self.worker_states[worker_id]
            
            # 检查队列是否超限
            if state.queue_length >= self.config.max_queue_length:
                continue
            
            score = self.calculate_score(worker_id, adapter_dir)
            scores.append((worker_id, score, state.queue_length))
        
        # 如果所有 Worker 都超限，选择队列最短的
        if not scores:
            logger.warning("All workers exceed queue threshold, selecting shortest queue")
            scores = [
                (wid, self.calculate_score(wid, adapter_dir), self.worker_states[wid].queue_length)
                for wid in healthy_workers
            ]
        
        # 选择最高分的 Worker，平分时选择队列最短的
        # 排序：先按分数降序，再按队列长度升序
        scores.sort(key=lambda x: (-x[1], x[2]))
        
        selected_worker_id = scores[0][0]
        selected_score = scores[0][1]
        
        # 记录统计
        cache_hit = self.worker_states[selected_worker_id].has_adapter(adapter_dir)
        is_cold_start = not any(
            self.worker_states[wid].has_adapter(adapter_dir) 
            for wid in healthy_workers
        )
        
        self.stats.record_request(selected_worker_id, cache_hit, is_cold_start)
        
        # 冷启动日志 (Requirements 4.2)
        if is_cold_start:
            logger.info(f"Cold start: adapter={adapter_dir} -> worker={selected_worker_id}, "
                       f"queue_length={self.worker_states[selected_worker_id].queue_length}")
        
        # DEBUG 日志
        logger.debug(f"Routing request for adapter={adapter_dir} to worker={selected_worker_id}, "
                    f"score={selected_score:.3f}, cache_hit={cache_hit}, cold_start={is_cold_start}")
        
        # 检查是否需要输出统计日志
        self.maybe_log_stats()
        
        return selected_worker_id
    
    def _select_for_hot_adapter(self, adapter_dir: str, healthy_workers: List[int]) -> Optional[int]:
        """
        热点 Adapter 的 Worker 选择
        
        将请求分散到多个 Worker，避免单个 Worker 过载。
        
        Args:
            adapter_dir: Adapter 目录
            healthy_workers: 健康的 Worker 列表
            
        Returns:
            选中的 Worker ID，如果无法处理则返回 None
        
        Requirements: 5.1, 5.3
        """
        # 过滤掉队列超限的 Worker
        available_workers = [
            wid for wid in healthy_workers
            if self.worker_states[wid].queue_length < self.config.max_queue_length
        ]
        
        if not available_workers:
            return None
        
        # 使用 Round-Robin 在可用 Worker 间分配
        rr_index = self._hot_adapter_rr[adapter_dir]
        selected_worker_id = available_workers[rr_index % len(available_workers)]
        self._hot_adapter_rr[adapter_dir] = rr_index + 1
        
        # 记录统计
        cache_hit = self.worker_states[selected_worker_id].has_adapter(adapter_dir)
        self.stats.record_request(selected_worker_id, cache_hit, is_cold_start=False)
        self.stats.record_hot_adapter_redistribution()
        
        logger.debug(f"Hot adapter routing: adapter={adapter_dir} -> worker={selected_worker_id}, "
                    f"rate={self._rate_tracker.get_request_rate(adapter_dir):.1f} req/s")
        
        return selected_worker_id
    
    def _round_robin_select(self) -> int:
        """
        Round-Robin 选择（回退模式）
        
        Requirements: 8.2
        """
        worker_id = self._rr_counter % self.num_workers
        self._rr_counter += 1
        
        # 记录统计（Round-Robin 模式下无法判断缓存命中）
        self.stats.record_request(worker_id, cache_hit=False, is_cold_start=False)
        
        return worker_id
    
    def _get_healthy_workers(self) -> List[int]:
        """获取所有健康的 Worker ID 列表"""
        return [
            wid for wid, state in self.worker_states.items()
            if state.is_healthy
        ]
    
    def update_worker_state(self, worker_id: int, state: WorkerState) -> None:
        """
        更新 Worker 状态
        
        Args:
            worker_id: Worker ID
            state: 新的 Worker 状态
        
        Requirements: 1.4
        """
        if worker_id not in self.worker_states:
            logger.warning(f"Unknown worker_id: {worker_id}")
            return
        
        old_state = self.worker_states[worker_id]
        
        # 更新倒排索引
        self._update_adapter_index(worker_id, old_state.cached_adapters, state.cached_adapters)
        
        # 更新状态
        self.worker_states[worker_id] = state
        
        logger.debug(f"Updated worker {worker_id} state: queue={state.queue_length}, "
                    f"adapters={len(state.cached_adapters)}")
    
    def _update_adapter_index(self, worker_id: int, 
                              old_adapters: Set[str], 
                              new_adapters: Set[str]) -> None:
        """
        更新 Adapter-to-Worker 倒排索引
        
        Requirements: 2.5
        """
        # 移除不再缓存的 Adapter
        removed = old_adapters - new_adapters
        for adapter in removed:
            if adapter in self.adapter_to_workers:
                self.adapter_to_workers[adapter].discard(worker_id)
                if not self.adapter_to_workers[adapter]:
                    del self.adapter_to_workers[adapter]
        
        # 添加新缓存的 Adapter
        added = new_adapters - old_adapters
        for adapter in added:
            self.adapter_to_workers[adapter].add(worker_id)
    
    def get_workers_with_adapter(self, adapter_dir: str) -> Set[int]:
        """
        获取缓存了指定 Adapter 的 Worker 列表
        
        Args:
            adapter_dir: Adapter 目录
            
        Returns:
            Worker ID 集合
        """
        return self.adapter_to_workers.get(adapter_dir, set()).copy()
    
    def get_stats(self) -> dict:
        """
        获取路由统计信息
        
        Requirements: 7.1, 7.2, 7.3
        """
        stats = self.stats.to_dict()
        
        # 添加额外的统计信息
        healthy_workers = self._get_healthy_workers()
        stats['healthy_workers'] = len(healthy_workers)
        stats['unhealthy_workers'] = self.num_workers - len(healthy_workers)
        stats['total_cached_adapters'] = len(self.adapter_to_workers)
        
        return stats
    
    def log_stats_summary(self) -> None:
        """
        输出统计摘要日志
        
        Requirements: 7.4, 7.5
        """
        stats = self.get_stats()
        
        logger.info(
            f"Routing Stats: "
            f"total={stats['total_requests']}, "
            f"hits={stats['cache_hits']}, "
            f"misses={stats['cache_misses']}, "
            f"hit_rate={stats['cache_hit_rate']:.2%}, "
            f"cold_starts={stats['cold_starts']}, "
            f"hot_redistributions={stats['hot_adapter_redistributions']}, "
            f"healthy_workers={stats['healthy_workers']}/{self.num_workers}"
        )
        
        # 输出 Worker 请求分布
        if stats['worker_request_counts']:
            distribution = ", ".join(
                f"W{wid}:{count}" 
                for wid, count in sorted(stats['worker_request_counts'].items())
            )
            logger.info(f"Worker distribution: {distribution}")
    
    def maybe_log_stats(self) -> None:
        """
        检查是否需要输出统计日志（每 10 秒一次）
        
        Requirements: 7.5
        """
        current_time = time.time()
        if current_time - self._last_stats_log_time >= self._stats_log_interval:
            self.log_stats_summary()
            self._last_stats_log_time = current_time
    
    def get_adapter_request_rate(self, adapter_dir: str) -> float:
        """
        获取指定 Adapter 的请求率
        
        Args:
            adapter_dir: Adapter 目录
            
        Returns:
            请求率 (req/s)
        
        Requirements: 5.2
        """
        return self._rate_tracker.get_request_rate(adapter_dir)
    
    def is_hot_adapter(self, adapter_dir: str) -> bool:
        """
        检查 Adapter 是否为热点
        
        Args:
            adapter_dir: Adapter 目录
            
        Returns:
            是否为热点
        
        Requirements: 5.1
        """
        return self._rate_tracker.is_hot(adapter_dir, self.config.hot_adapter_threshold)
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats.reset()
        logger.info("Routing statistics reset")
    
    def reset(self) -> None:
        """重置路由器状态"""
        # 重置 Worker 状态
        for i in range(self.num_workers):
            self.worker_states[i] = WorkerState(worker_id=i)
        
        # 清空倒排索引
        self.adapter_to_workers.clear()
        
        # 重置统计
        self.stats.reset()
        
        # 重置请求率追踪器
        self._rate_tracker.reset()
        
        # 重置热点 Adapter Round-Robin 计数器
        self._hot_adapter_rr.clear()
        
        # 重置 Round-Robin 计数器
        self._rr_counter = 0
        
        logger.info("AdapterAwareRouter reset")
