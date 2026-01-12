"""
Round Robin Router for Data Parallel Mode

This module implements a simple round-robin routing strategy that distributes
requests evenly across multiple GPU workers in a cyclic manner.
"""

import threading
from typing import Optional


class RoundRobinRouter:
    """
    轮询路由器 - 按顺序循环分配请求到不同的 Worker
    
    该路由器维护一个计数器，每次选择 Worker 时递增，
    通过取模运算确保请求均匀分配到所有 Worker。
    
    线程安全：使用锁保护计数器，支持多线程并发调用。
    
    Attributes:
        num_workers: Worker 总数
        counter: 当前计数器值，用于轮询选择
        _lock: 线程锁，保护计数器的并发访问
    """
    
    def __init__(self, num_workers: int):
        """
        初始化轮询路由器
        
        Args:
            num_workers: Worker 数量，必须大于 0
            
        Raises:
            ValueError: 如果 num_workers <= 0
        """
        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers}")
        
        self.num_workers = num_workers
        self.counter = 0
        self._lock = threading.Lock()
    
    def select_worker(self) -> int:
        """
        选择下一个 Worker
        
        使用轮询策略：按照 0, 1, 2, ..., N-1, 0, 1, ... 的顺序循环选择。
        该方法是线程安全的，可以在多线程环境中并发调用。
        
        Returns:
            Worker ID (0 到 num_workers-1)
            
        Example:
            >>> router = RoundRobinRouter(num_workers=3)
            >>> [router.select_worker() for _ in range(6)]
            [0, 1, 2, 0, 1, 2]
        """
        with self._lock:
            worker_id = self.counter % self.num_workers
            self.counter += 1
            return worker_id
    
    def reset(self) -> None:
        """
        重置计数器到 0
        
        该方法主要用于测试或需要重新开始轮询的场景。
        """
        with self._lock:
            self.counter = 0
    
    def get_stats(self) -> dict:
        """
        获取路由器统计信息
        
        Returns:
            包含统计信息的字典：
            - num_workers: Worker 数量
            - total_requests: 已路由的总请求数
            - next_worker: 下一个将被选择的 Worker ID
        """
        with self._lock:
            return {
                'num_workers': self.num_workers,
                'total_requests': self.counter,
                'next_worker': self.counter % self.num_workers
            }
