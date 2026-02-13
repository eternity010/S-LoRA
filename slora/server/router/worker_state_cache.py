"""
Worker State Cache for Adapter-Aware Routing

This module implements the Router-side cache for Worker states,
supporting efficient state lookup and health monitoring.

Requirements: 1.4, 1.5
"""

import time
import logging
from typing import Dict, List, Optional, Set
from threading import Lock

from .worker_state import WorkerState, RoutingConfig


logger = logging.getLogger(__name__)


class WorkerStateCache:
    """
    Worker 状态缓存
    
    维护所有 Worker 的最新状态，支持快速查询和健康检测。
    
    Attributes:
        num_workers: Worker 数量
        heartbeat_timeout_ms: 心跳超时时间（毫秒）
        heartbeat_interval_ms: 心跳间隔（毫秒）
        _states: Worker 状态字典
        _lock: 线程安全锁
    
    Requirements: 1.4, 1.5
    """
    
    def __init__(self, 
                 num_workers: int,
                 heartbeat_interval_ms: int = 100,
                 heartbeat_timeout_ms: int = 300):
        """
        初始化状态缓存
        
        Args:
            num_workers: Worker 数量
            heartbeat_interval_ms: 心跳间隔（毫秒）
            heartbeat_timeout_ms: 心跳超时（毫秒），默认为 3 倍心跳间隔
        """
        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers}")
        if heartbeat_interval_ms <= 0:
            raise ValueError(f"heartbeat_interval_ms must be positive, got {heartbeat_interval_ms}")
        if heartbeat_timeout_ms <= 0:
            raise ValueError(f"heartbeat_timeout_ms must be positive, got {heartbeat_timeout_ms}")
        
        self.num_workers = num_workers
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.heartbeat_timeout_ms = heartbeat_timeout_ms
        
        # 初始化所有 Worker 状态
        self._states: Dict[int, WorkerState] = {}
        for i in range(num_workers):
            self._states[i] = WorkerState(worker_id=i)
        
        # 线程安全锁
        self._lock = Lock()
        
        logger.info(f"WorkerStateCache initialized with {num_workers} workers, "
                   f"heartbeat_interval={heartbeat_interval_ms}ms, "
                   f"heartbeat_timeout={heartbeat_timeout_ms}ms")
    
    def update(self, worker_id: int, state: WorkerState) -> None:
        """
        更新 Worker 状态
        
        Args:
            worker_id: Worker ID
            state: 新的 Worker 状态
        
        Requirements: 1.4
        """
        if worker_id < 0 or worker_id >= self.num_workers:
            logger.warning(f"Invalid worker_id: {worker_id}, ignoring update")
            return
        
        with self._lock:
            self._states[worker_id] = state
            logger.debug(f"Updated worker {worker_id} state: "
                        f"queue={state.queue_length}, "
                        f"adapters={len(state.cached_adapters)}, "
                        f"healthy={state.is_healthy}")
    
    def update_from_message(self, message: dict) -> None:
        """
        从消息更新 Worker 状态
        
        Args:
            message: 状态上报消息，格式为:
                {
                    'type': 'worker_state',
                    'worker_id': int,
                    'cached_adapters': List[str],
                    'queue_length': int,
                    'gpu_memory_free': int,
                    'timestamp': float,
                    'avg_rank': float (optional),
                    'min_rank': int (optional),
                    'max_rank': int (optional)
                }
        
        Requirements: 1.4, 8.2, 8.3
        """
        try:
            if message.get('type') != 'worker_state':
                logger.warning(f"Invalid message type: {message.get('type')}")
                return
            
            worker_id = message.get('worker_id')
            if worker_id is None or worker_id < 0 or worker_id >= self.num_workers:
                logger.warning(f"Invalid worker_id in message: {worker_id}")
                return
            
            state = WorkerState(
                worker_id=worker_id,
                cached_adapters=set(message.get('cached_adapters', [])),
                queue_length=message.get('queue_length', 0),
                gpu_memory_free=message.get('gpu_memory_free', 0),
                last_heartbeat=message.get('timestamp', time.time()),
                is_healthy=True,  # 收到消息说明 Worker 健康
                # Rank distribution fields for Rank-Aware Routing
                # Use default values (0.0, 0, 0) for backward compatibility with old Workers
                avg_rank=message.get('avg_rank', 0.0),
                min_rank=message.get('min_rank', 0),
                max_rank=message.get('max_rank', 0)
            )
            
            self.update(worker_id, state)
            
        except Exception as e:
            logger.error(f"Error processing state message: {e}")
    
    def get(self, worker_id: int) -> Optional[WorkerState]:
        """
        获取指定 Worker 状态
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Worker 状态，如果不存在则返回 None
        """
        with self._lock:
            return self._states.get(worker_id)
    
    def get_all(self) -> Dict[int, WorkerState]:
        """
        获取所有 Worker 状态
        
        Returns:
            Worker ID 到状态的字典副本
        """
        with self._lock:
            return dict(self._states)
    
    def get_healthy_workers(self) -> List[int]:
        """
        获取所有健康的 Worker ID 列表
        
        Returns:
            健康 Worker 的 ID 列表
        
        Requirements: 1.5
        """
        with self._lock:
            return [
                wid for wid, state in self._states.items()
                if state.is_healthy
            ]
    
    def check_health(self) -> List[int]:
        """
        检查并更新 Worker 健康状态
        
        如果 Worker 在 heartbeat_timeout_ms 内没有发送心跳，
        则标记为不健康。
        
        Returns:
            新标记为不健康的 Worker ID 列表
        
        Requirements: 1.5
        """
        current_time = time.time()
        timeout_sec = self.heartbeat_timeout_ms / 1000.0
        newly_unhealthy = []
        
        with self._lock:
            for worker_id, state in self._states.items():
                if state.is_healthy:
                    time_since_heartbeat = current_time - state.last_heartbeat
                    if time_since_heartbeat > timeout_sec:
                        # 标记为不健康
                        state.is_healthy = False
                        newly_unhealthy.append(worker_id)
                        logger.warning(
                            f"Worker {worker_id} marked unhealthy: "
                            f"no heartbeat for {time_since_heartbeat:.2f}s "
                            f"(timeout={timeout_sec:.2f}s)"
                        )
        
        return newly_unhealthy
    
    def mark_healthy(self, worker_id: int) -> None:
        """
        标记 Worker 为健康
        
        Args:
            worker_id: Worker ID
        """
        with self._lock:
            if worker_id in self._states:
                self._states[worker_id].is_healthy = True
                self._states[worker_id].update_heartbeat()
                logger.info(f"Worker {worker_id} marked healthy")
    
    def mark_unhealthy(self, worker_id: int) -> None:
        """
        标记 Worker 为不健康
        
        Args:
            worker_id: Worker ID
        """
        with self._lock:
            if worker_id in self._states:
                self._states[worker_id].is_healthy = False
                logger.warning(f"Worker {worker_id} marked unhealthy")
    
    def get_workers_with_adapter(self, adapter_dir: str) -> Set[int]:
        """
        获取缓存了指定 Adapter 的 Worker 列表
        
        Args:
            adapter_dir: Adapter 目录
            
        Returns:
            Worker ID 集合
        """
        with self._lock:
            return {
                wid for wid, state in self._states.items()
                if adapter_dir in state.cached_adapters
            }
    
    def reset(self) -> None:
        """重置所有 Worker 状态"""
        with self._lock:
            for i in range(self.num_workers):
                self._states[i] = WorkerState(worker_id=i)
        logger.info("WorkerStateCache reset")
    
    def get_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            healthy_count = sum(1 for s in self._states.values() if s.is_healthy)
            total_adapters = sum(len(s.cached_adapters) for s in self._states.values())
            total_queue = sum(s.queue_length for s in self._states.values())
            
            return {
                'num_workers': self.num_workers,
                'healthy_workers': healthy_count,
                'unhealthy_workers': self.num_workers - healthy_count,
                'total_cached_adapters': total_adapters,
                'total_queue_length': total_queue,
                'avg_queue_length': total_queue / self.num_workers if self.num_workers > 0 else 0
            }
