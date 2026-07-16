"""
Adapter-Aware Router for Intelligent Request Routing

This module implements the core routing logic that considers cache affinity
and load balancing to optimize request distribution across GPU Workers.

Requirements: 2.1-2.5, 3.1-3.4, 5.1-5.3
"""

import os
import time
import logging
from typing import Dict, List, Optional, Set, Deque
from collections import defaultdict, deque

from .worker_state import WorkerState, RoutingStats, RoutingConfig


logger = logging.getLogger(__name__)


def _get_debug_file_logger():
    """获取写入文件的调试 logger（单例）"""
    debug_logger = logging.getLogger('routing_debug')
    if not debug_logger.handlers:
        debug_logger.setLevel(logging.DEBUG)
        log_path = os.environ.get('ROUTING_DEBUG_LOG', 'routing_debug.log')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
        debug_logger.addHandler(fh)
        debug_logger.propagate = False  # 不输出到终端
    return debug_logger


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
    Score_i(r) = w1 · I(adapter ∈ Cache_i) - w2 · RWPT_i/Capacity_i - w3 · RankMismatch_i
    
    RWPT_i = pending_prefill_tokens
    RWPTActive_i = pending_prefill_tokens + active_rwpt_tokens
    
    当 Worker 未上报 RWPT 字段时，回退到 QueueLen 逻辑保证向后兼容。
    
    Attributes:
        num_workers: Worker 数量
        config: 路由配置
        worker_states: Worker 状态字典
        adapter_to_workers: Adapter 到 Worker 的倒排索引
        adapter_ranks: Adapter 到 rank 的映射（用于 Rank-Aware Routing）
        stats: 路由统计信息
        _gamma: γ = 2/(3·d)，LoRA 对 Q/K/V/O 四投影的 FLOPs 增比推导
        _capacity: 归一化容量（KV Cache token 容量）
        _decode_cost_alpha: Decode 序列负载折算系数 α
    
    Requirements: 2.1, 2.2, 3.1, 3.2, 3.3, 3.4, 4.2, 4.3, 4.4, 4.5
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
        
        # Adapter -> Rank 映射（用于 Rank-Aware Routing）
        # Requirements: 3.2, 3.3, 3.4
        self.adapter_ranks: Dict[str, int] = {}
        
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
        
        # RWPT parameters (Requirements: 4.2)
        # γ = 2/(3d): LoRA 对 Q/K/V/O 四投影各加 4dr FLOPs，共 16dr / 24d² = 2r/(3d)
        self._gamma = 2.0 / (3.0 * self.config.hidden_dim)
        # RWPT 归一化分母：用 batch_max_tokens（单次 prefill 批次容量）而非 max_total_token_num（KV Cache 总槽位）
        # 物理意义：RWPT/batch_max_tokens ≈ "排队批次数"（Expected Batches to Process）
        self._capacity = self.config.batch_max_tokens
        # decode_cost_alpha: None means waiting for Worker profiling, fallback to 0.1
        self._decode_cost_alpha = self.config.decode_cost_alpha if self.config.decode_cost_alpha is not None else 0.1
        
        alpha_status = "auto-profiling (pending)" if self.config.decode_cost_alpha is None else f"{self._decode_cost_alpha}"
        logger.info(f"AdapterAwareRouter initialized with {num_workers} workers, "
                   f"strategy={self.config.strategy}, w1={self.config.w1}, w2={self.config.w2}, "
                   f"w3={self.config.w3}, load_metric={self.config.load_metric}, "
                   f"gamma={self._gamma:.6f}, "
                   f"capacity(batch_max_tokens)={self._capacity}, decode_cost_alpha={alpha_status}")
    
    def calculate_score(self, worker_id: int, adapter_dir: str) -> float:
        """
        计算 Worker 对请求的评分

        Score_i(r) = w1 · I(adapter ∈ Cache_i) - w2 · Load_i/Capacity_i - w3 · RankMismatch_i

        Load_i 根据 load_metric 选择：
        - queue_length: queue_length（无归一化）
        - token_count: pending_raw_tokens（等待队列中的原始 prefill token 数）
        - rwpt: pending_prefill_tokens（等待队列中 rank 加权的 input token 数）
        - rwpt_active: pending_prefill_tokens + active_rwpt_tokens

        当 Worker 未上报 token 字段时（pending_*_tokens == 0 且 queue_length > 0），
        回退到 queue_length 逻辑，保证向后兼容。

        Args:
            worker_id: Worker ID
            adapter_dir: Adapter 目录

        Returns:
            评分值

        Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 5.1, 5.2, 5.3, 5.4, 6.1
        """
        state = self.worker_states.get(worker_id)
        if state is None:
            return float('-inf')

        # Load penalty — 根据 load_metric 分支
        metric = self.config.load_metric
        load_value = 0.0  # 用于评分分解日志
        load_pressure = 0.0

        if metric == 'queue_length':
            # Variant A: 仅队列长度（基线，无 Capacity 归一化）
            load_value = self.config.w2 * state.queue_length
            logger.debug(f"Worker {worker_id} queue_length mode: queue={state.queue_length}")

        elif metric == 'token_count':
            # Variant B: token 级负载，仅 prefill token（无 rank 加权，无 decode 折算）
            raw = state.pending_raw_tokens
            load_pressure = raw / self._capacity if self._capacity > 0 else 0.0
            load_value = self.config.w2 * load_pressure
            logger.debug(f"Worker {worker_id} token_count: raw={raw}, "
                        f"load_pressure={load_pressure:.4f}")

        elif metric == 'rwpt_active':
            rwpt = state.pending_prefill_tokens + state.active_rwpt_tokens
            load_pressure = rwpt / self._capacity if self._capacity > 0 else 0.0
            load_value = self.config.w2 * load_pressure
            logger.debug(f"Worker {worker_id} active-aware RWPT: rwpt={rwpt}, "
                        f"waiting={state.pending_prefill_tokens}, "
                        f"active={state.active_rwpt_tokens}, "
                        f"load_pressure={load_pressure:.4f}")

        else:  # 'rwpt' (默认)
            # Variant C: RWPT — rank 加权的 prefill token 数（无 decode 折算）
            rwpt = state.pending_prefill_tokens
            load_pressure = rwpt / self._capacity if self._capacity > 0 else 0.0
            load_value = self.config.w2 * load_pressure
            logger.debug(f"Worker {worker_id} RWPT: rwpt={rwpt}, "
                        f"prefill_tokens={state.pending_prefill_tokens}, "
                        f"load_pressure={load_pressure:.4f}")

        # Cache affinity: I(adapter in Cache_i), with a fixed weight for all metrics.
        cache_indicator = 1.0 if state.has_adapter(adapter_dir) else 0.0
        score = self.config.w1 * cache_indicator - load_value

        # Rank 不匹配惩罚（仅当 w3 > 0 时应用）
        # Requirements: 5.1, 5.2, 5.4, 5.5
        if self.config.w3 > 0:
            request_rank = self.get_adapter_rank(adapter_dir)
            rank_mismatch = self.calculate_rank_mismatch(request_rank, worker_id)
            rank_penalty = self.config.w3 * rank_mismatch
            score -= rank_penalty

            logger.debug(f"Worker {worker_id} score breakdown: "
                        f"cache={self.config.w1 * cache_indicator:.2f}, "
                        f"load={-load_value:.2f}, "
                        f"rank_penalty={-rank_penalty:.2f}, "
                        f"total={score:.2f}")

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
        import os
        debug_mode = os.environ.get('DEBUG', '0') == '1'
        
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
        
        # DEBUG: 输出 Worker 状态信息到文件
        if debug_mode:
            dlog = _get_debug_file_logger()
            adapter_name = adapter_dir.split('/')[-1] if adapter_dir else 'None'
            dlog.debug(f"[Router] Routing adapter: {adapter_name}")
            for wid in healthy_workers:
                state = self.worker_states[wid]
                dlog.debug(f"[Router]   W{wid}: queue={state.queue_length}, "
                           f"cached={len(state.cached_adapters)}, "
                           f"has_target={state.has_adapter(adapter_dir)}")
        
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
            
            # DEBUG: 输出评分详情到文件
            if debug_mode:
                cache_hit = 1.0 if state.has_adapter(adapter_dir) else 0.0
                dlog.debug(f"[Router]   W{worker_id} score={score:.3f} "
                           f"(cache={self.config.w1 * cache_hit:.2f}, "
                           f"qpen={-self.config.w2 * state.queue_length:.2f})")
        
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
        
        # 记录 Rank 不匹配统计 (Requirements: 7.1, 7.2)
        if self.config.w3 > 0:
            request_rank = self.get_adapter_rank(adapter_dir)
            rank_mismatch = self.calculate_rank_mismatch(request_rank, selected_worker_id)
            self.stats.record_rank_mismatch(rank_mismatch)
        
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
    
    def set_adapter_ranks(self, adapter_ranks: Dict[str, int]) -> None:
        """
        设置 Adapter rank 映射
        
        从 DataParallelRouterManager 调用，传入从 adapter 配置文件中读取的 rank 信息。
        
        Args:
            adapter_ranks: Adapter 目录到 rank 的映射字典
        
        Requirements: 3.2
        """
        self.adapter_ranks = adapter_ranks.copy()
        logger.info(f"Loaded {len(self.adapter_ranks)} adapter ranks for rank-aware routing")
        
        # DEBUG: 输出部分 adapter rank 信息
        if self.adapter_ranks:
            sample_adapters = list(self.adapter_ranks.items())[:5]
            for adapter_dir, rank in sample_adapters:
                adapter_name = adapter_dir.split('/')[-1] if '/' in adapter_dir else adapter_dir
                logger.debug(f"  Adapter {adapter_name}: rank={rank}")
    
    def get_adapter_rank(self, adapter_dir: str) -> int:
        """
        获取指定 Adapter 的 rank
        
        如果 Adapter 未知（不在 adapter_ranks 映射中），返回配置的默认 rank 值。
        
        Args:
            adapter_dir: Adapter 目录
            
        Returns:
            Adapter 的 rank 值
        
        Requirements: 3.3, 3.4
        """
        if adapter_dir in self.adapter_ranks:
            return self.adapter_ranks[adapter_dir]
        
        # 未知 Adapter，使用默认 rank
        default_rank = self.config.default_lora_rank
        logger.debug(f"Unknown adapter {adapter_dir}, using default rank={default_rank}")
        return default_rank
    
    def calculate_rank_mismatch(self, request_rank: int, worker_id: int) -> float:
        """
        计算请求 rank 与 Worker 平均 rank 的归一化不匹配度
        
        公式: mismatch = |request_rank - avg_rank| / max_rank_diff
        
        Args:
            request_rank: 请求的 Adapter rank
            worker_id: Worker ID
            
        Returns:
            归一化的不匹配度，范围 [0, 1]
            - 0.0 表示完全匹配或空批次
            - 1.0 表示最大不匹配
        
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        state = self.worker_states.get(worker_id)
        if state is None:
            return 0.0
        
        # 空批次情况：avg_rank=0，返回 0.0（不惩罚）
        if state.avg_rank == 0.0:
            return 0.0
        
        # 计算绝对差值
        diff = abs(request_rank - state.avg_rank)
        
        # rank 相等时返回 0.0
        if diff == 0.0:
            return 0.0
        
        # 归一化到 [0, 1] 范围
        max_rank_diff = self.config.max_rank_diff
        if max_rank_diff <= 0:
            return 0.0
        
        # 限制在 [0, 1] 范围内
        mismatch = min(diff / max_rank_diff, 1.0)
        
        return mismatch
    
    def get_stats(self) -> dict:
        """
        获取路由统计信息
        
        Requirements: 7.1, 7.2, 7.3, 7.4
        """
        stats = self.stats.to_dict()
        
        # 添加额外的统计信息
        healthy_workers = self._get_healthy_workers()
        stats['healthy_workers'] = len(healthy_workers)
        stats['unhealthy_workers'] = self.num_workers - len(healthy_workers)
        stats['total_cached_adapters'] = len(self.adapter_to_workers)
        
        # 添加当前路由配置参数 (w1, w2, w3)
        stats['w1'] = self.config.w1
        stats['w2'] = self.config.w2
        stats['w3'] = self.config.w3
        
        # 添加 Rank-Aware 路由配置信息 (Requirements: 7.4)
        stats['rank_aware_enabled'] = self.config.w3 > 0
        stats['loaded_adapter_ranks'] = len(self.adapter_ranks)
        
        # RWPT 可观测性 (Requirements: 7.1, 7.2)
        # 检查是否有任何 Worker 在上报 RWPT 字段
        rwpt_active = any(
            s.pending_prefill_tokens > 0 or s.active_decode_seqs > 0
            for s in self.worker_states.values()
        )
        stats['rwpt_enabled'] = rwpt_active
        stats['decode_cost_alpha'] = self._decode_cost_alpha
        stats['load_metric'] = self.config.load_metric
        
        return stats
    
    def get_cache_hit_rate(self) -> float:
        """
        获取缓存命中率
        
        Returns:
            缓存命中率（0.0 - 1.0）
        """
        return self.stats.cache_hit_rate
    
    def log_stats_summary(self) -> None:
        """
        输出统计摘要日志
        
        Requirements: 7.3, 7.4, 7.5
        """
        stats = self.get_stats()
        
        # 基础统计日志
        log_msg = (
            f"Routing Stats: "
            f"total={stats['total_requests']}, "
            f"hits={stats['cache_hits']}, "
            f"misses={stats['cache_misses']}, "
            f"hit_rate={stats['cache_hit_rate']:.2%}, "
            f"cold_starts={stats['cold_starts']}, "
            f"hot_redistributions={stats['hot_adapter_redistributions']}, "
            f"healthy_workers={stats['healthy_workers']}/{self.num_workers}"
        )
        
        # 添加 Rank 统计信息 (Requirements: 7.3)
        if stats.get('rank_aware_enabled', False):
            log_msg += (
                f", rank_matched={stats.get('rank_matched_count', 0)}, "
                f"avg_rank_mismatch={stats.get('avg_rank_mismatch', 0.0):.3f}"
            )
        
        logger.info(log_msg)
        
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
    
    def update_config(self,
                     w1: float = None,
                     w2: float = None,
                     w3: float = None,
                     load_metric: str = None,
                     reset_stats: bool = True) -> dict:
        """
        动态更新路由配置
        
        允许在运行时更新路由权重参数，无需重启服务器。
        
        Args:
            w1: 缓存亲和性权重（None 表示不更新）
            w2: 负载惩罚权重（None 表示不更新）
            w3: Rank 不匹配惩罚权重（None 表示不更新）
            load_metric: 负载度量类型（None 表示不更新）
            reset_stats: 是否重置统计信息，默认 True
        
        Returns:
            更新后的配置字典，包含 w1, w2, w3, load_metric 的当前值
        
        Note:
            - 参数值必须为非负数，否则忽略该更新
            - 更新后建议重置统计以便观察新配置的效果
        """
        updated = []
        
        if w1 is not None and w1 >= 0:
            self.config.w1 = w1
            updated.append(f"w1={w1}")
        
        if w2 is not None and w2 >= 0:
            self.config.w2 = w2
            updated.append(f"w2={w2}")
        
        if w3 is not None and w3 >= 0:
            self.config.w3 = w3
            updated.append(f"w3={w3}")
        
        if load_metric is not None and load_metric in RoutingConfig.VALID_LOAD_METRICS:
            old_metric = self.config.load_metric
            self.config.load_metric = load_metric
            updated.append(f"load_metric={old_metric}->{load_metric}")
        
        if reset_stats:
            self.stats.reset()
            updated.append("stats_reset=True")
        
        if updated:
            logger.info(f"Routing config updated: {', '.join(updated)}")
        
        return {
            'w1': self.config.w1,
            'w2': self.config.w2,
            'w3': self.config.w3,
            'load_metric': self.config.load_metric,
        }
    
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
        
        # 清空 adapter rank 映射
        self.adapter_ranks.clear()
        
        # 重置统计
        self.stats.reset()
        
        # 重置请求率追踪器
        self._rate_tracker.reset()
        
        # 重置热点 Adapter Round-Robin 计数器
        self._hot_adapter_rr.clear()
        
        # 重置 Round-Robin 计数器
        self._rr_counter = 0
        
        logger.info("AdapterAwareRouter reset")
