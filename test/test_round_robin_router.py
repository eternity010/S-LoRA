"""
Unit tests for Round Robin Router

Tests the correctness of the round-robin routing strategy including:
- Basic selection order
- Fairness across workers
- Thread safety
- Edge cases
"""

import pytest
import threading
from slora.server.router.round_robin_router import RoundRobinRouter


class TestRoundRobinRouter:
    """Test suite for RoundRobinRouter"""
    
    def test_initialization(self):
        """测试路由器初始化"""
        router = RoundRobinRouter(num_workers=3)
        assert router.num_workers == 3
        assert router.counter == 0
    
    def test_initialization_invalid_workers(self):
        """测试无效的 Worker 数量"""
        with pytest.raises(ValueError):
            RoundRobinRouter(num_workers=0)
        
        with pytest.raises(ValueError):
            RoundRobinRouter(num_workers=-1)
    
    def test_basic_selection_order(self):
        """测试基本的轮询顺序"""
        router = RoundRobinRouter(num_workers=3)
        
        # 测试两轮完整的轮询
        selections = [router.select_worker() for _ in range(6)]
        
        assert selections == [0, 1, 2, 0, 1, 2]
    
    def test_single_worker(self):
        """测试单个 Worker 的情况"""
        router = RoundRobinRouter(num_workers=1)
        
        selections = [router.select_worker() for _ in range(5)]
        
        # 单个 Worker 时，应该总是返回 0
        assert selections == [0, 0, 0, 0, 0]
    
    def test_fairness_property(self):
        """
        测试公平性属性：
        对于 N 个 Worker，连续 N 个请求应该均匀分配
        """
        for num_workers in [2, 3, 4, 5, 8]:
            router = RoundRobinRouter(num_workers=num_workers)
            
            # 选择 num_workers 个请求
            selections = [router.select_worker() for _ in range(num_workers)]
            
            # 每个 Worker 应该被选中恰好一次
            assert sorted(selections) == list(range(num_workers))
    
    def test_large_number_of_requests(self):
        """测试大量请求的分配"""
        router = RoundRobinRouter(num_workers=4)
        
        # 发送 1000 个请求
        selections = [router.select_worker() for _ in range(1000)]
        
        # 统计每个 Worker 收到的请求数
        counts = [selections.count(i) for i in range(4)]
        
        # 每个 Worker 应该收到 250 个请求
        assert counts == [250, 250, 250, 250]
    
    def test_reset(self):
        """测试重置功能"""
        router = RoundRobinRouter(num_workers=3)
        
        # 选择几个 Worker
        router.select_worker()
        router.select_worker()
        
        # 重置
        router.reset()
        
        # 应该从 0 开始
        assert router.select_worker() == 0
        assert router.select_worker() == 1
    
    def test_get_stats(self):
        """测试统计信息"""
        router = RoundRobinRouter(num_workers=3)
        
        # 初始状态
        stats = router.get_stats()
        assert stats['num_workers'] == 3
        assert stats['total_requests'] == 0
        assert stats['next_worker'] == 0
        
        # 选择几个 Worker
        router.select_worker()
        router.select_worker()
        
        stats = router.get_stats()
        assert stats['total_requests'] == 2
        assert stats['next_worker'] == 2
    
    def test_thread_safety(self):
        """测试线程安全性"""
        router = RoundRobinRouter(num_workers=4)
        selections = []
        lock = threading.Lock()
        
        def worker_thread():
            """工作线程：选择 100 个 Worker"""
            for _ in range(100):
                worker_id = router.select_worker()
                with lock:
                    selections.append(worker_id)
        
        # 创建 10 个线程，每个选择 100 次
        threads = [threading.Thread(target=worker_thread) for _ in range(10)]
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 应该有 1000 个选择
        assert len(selections) == 1000
        
        # 统计每个 Worker 的选择次数
        counts = [selections.count(i) for i in range(4)]
        
        # 每个 Worker 应该被选中 250 次
        assert counts == [250, 250, 250, 250]
    
    def test_modulo_behavior(self):
        """测试取模行为的正确性"""
        router = RoundRobinRouter(num_workers=7)
        
        # 选择 21 个（3 轮完整循环）
        selections = [router.select_worker() for _ in range(21)]
        
        # 验证每 7 个为一个循环
        for i in range(3):
            cycle = selections[i*7:(i+1)*7]
            assert cycle == list(range(7))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
