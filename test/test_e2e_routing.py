"""
End-to-End Routing Tests for Adapter-Aware Router

Tests the complete routing flow in multi-worker scenarios including:
- Multi-worker routing decisions
- Cache hit and miss scenarios
- Integration between AdapterAwareRouter and WorkerStateCache

Requirements: 2.1-2.5
"""

import pytest
import time
from typing import List, Dict, Set

from slora.server.router.adapter_aware_router import AdapterAwareRouter
from slora.server.router.worker_state import WorkerState, RoutingConfig
from slora.server.router.worker_state_cache import WorkerStateCache


class TestMultiWorkerRoutingDecisions:
    """
    Test suite for multi-worker routing decisions
    
    Requirements: 2.1, 2.3, 2.4
    """
    
    def test_routing_with_three_workers_cache_distribution(self):
        """
        测试三个 Worker 场景下的路由决策
        
        场景：
        - Worker 0: 缓存 adapter_a, adapter_b
        - Worker 1: 缓存 adapter_b, adapter_c
        - Worker 2: 缓存 adapter_c, adapter_d
        
        验证请求被路由到正确的 Worker
        
        Requirements: 2.1, 2.3
        """
        router = AdapterAwareRouter(num_workers=3)
        
        # 设置 Worker 状态
        router.update_worker_state(0, WorkerState(
            worker_id=0,
            cached_adapters={"/adapters/adapter_a", "/adapters/adapter_b"},
            queue_length=0
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1,
            cached_adapters={"/adapters/adapter_b", "/adapters/adapter_c"},
            queue_length=0
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2,
            cached_adapters={"/adapters/adapter_c", "/adapters/adapter_d"},
            queue_length=0
        ))
        
        # adapter_a 只在 Worker 0 上
        assert router.select_worker("/adapters/adapter_a") == 0
        
        # adapter_d 只在 Worker 2 上
        assert router.select_worker("/adapters/adapter_d") == 2
    
    def test_routing_with_shared_adapter_selects_lowest_queue(self):
        """
        测试多个 Worker 共享 Adapter 时选择队列最短的
        
        场景：
        - Worker 0: 缓存 adapter_shared, 队列长度 5
        - Worker 1: 缓存 adapter_shared, 队列长度 2
        - Worker 2: 缓存 adapter_shared, 队列长度 8
        
        应该选择 Worker 1（队列最短）
        
        Requirements: 2.3, 2.4
        """
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/adapters/adapter_shared"
        
        router.update_worker_state(0, WorkerState(
            worker_id=0,
            cached_adapters={adapter_dir},
            queue_length=5
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1,
            cached_adapters={adapter_dir},
            queue_length=2
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2,
            cached_adapters={adapter_dir},
            queue_length=8
        ))
        
        # 所有 Worker 都有缓存，应该选择队列最短的 Worker 1
        selected = router.select_worker(adapter_dir)
        assert selected == 1
    
    def test_routing_with_five_workers_complex_scenario(self):
        """
        测试五个 Worker 的复杂场景
        
        场景：
        - Worker 0: 缓存 adapter_a, 队列长度 10
        - Worker 1: 缓存 adapter_a, 队列长度 5
        - Worker 2: 无缓存, 队列长度 0
        - Worker 3: 缓存 adapter_a, 队列长度 3
        - Worker 4: 无缓存, 队列长度 1
        
        对于 adapter_a 请求，应该选择 Worker 3（缓存命中且队列最短）
        
        Requirements: 2.1, 2.2, 2.3, 2.4
        """
        router = AdapterAwareRouter(num_workers=5)
        adapter_dir = "/adapters/adapter_a"
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=10
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_dir}, queue_length=5
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=0
        ))
        router.update_worker_state(3, WorkerState(
            worker_id=3, cached_adapters={adapter_dir}, queue_length=3
        ))
        router.update_worker_state(4, WorkerState(
            worker_id=4, cached_adapters=set(), queue_length=1
        ))
        
        # Worker 3 有缓存且队列最短（在有缓存的 Worker 中）
        # Score(0) = 1.0 - 0.1*10 = 0.0
        # Score(1) = 1.0 - 0.1*5 = 0.5
        # Score(2) = 0.0 - 0.1*0 = 0.0
        # Score(3) = 1.0 - 0.1*3 = 0.7
        # Score(4) = 0.0 - 0.1*1 = -0.1
        selected = router.select_worker(adapter_dir)
        assert selected == 3
    
    def test_routing_respects_worker_health(self):
        """
        测试路由决策尊重 Worker 健康状态
        
        场景：
        - Worker 0: 缓存 adapter_a, 队列长度 0, 不健康
        - Worker 1: 无缓存, 队列长度 5, 健康
        - Worker 2: 无缓存, 队列长度 3, 健康
        
        应该选择 Worker 2（健康且队列最短）
        
        Requirements: 2.1
        """
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/adapters/adapter_a"
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=0, is_healthy=False
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters=set(), queue_length=5, is_healthy=True
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=3, is_healthy=True
        ))
        
        # Worker 0 不健康，应该选择健康的 Worker 中队列最短的 Worker 2
        selected = router.select_worker(adapter_dir)
        assert selected == 2


class TestCacheHitAndMissScenarios:
    """
    Test suite for cache hit and miss scenarios
    
    Requirements: 2.1, 2.2, 2.5
    """
    
    def test_cache_hit_scenario(self):
        """
        测试缓存命中场景
        
        验证：
        1. 请求被路由到有缓存的 Worker
        2. 统计正确记录缓存命中
        
        Requirements: 2.1, 2.2
        """
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/adapters/adapter_a"
        
        # 只有 Worker 1 缓存了 adapter
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_dir}, queue_length=0
        ))
        
        # 发送请求
        selected = router.select_worker(adapter_dir)
        
        # 验证路由到有缓存的 Worker
        assert selected == 1
        
        # 验证统计
        stats = router.get_stats()
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 0
    
    def test_cache_miss_scenario(self):
        """
        测试缓存未命中场景
        
        验证：
        1. 请求被路由到队列最短的 Worker
        2. 统计正确记录缓存未命中
        
        Requirements: 2.1, 2.2
        """
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/adapters/new_adapter"
        
        # 没有 Worker 缓存该 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters=set(), queue_length=5
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters=set(), queue_length=2
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=8
        ))
        
        # 发送请求
        selected = router.select_worker(adapter_dir)
        
        # 验证路由到队列最短的 Worker
        assert selected == 1
        
        # 验证统计
        stats = router.get_stats()
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 1
        assert stats['cold_starts'] == 1
    
    def test_mixed_cache_hit_miss_sequence(self):
        """
        测试混合缓存命中和未命中的请求序列
        
        场景：
        - Worker 0: 缓存 adapter_a
        - Worker 1: 缓存 adapter_b
        - Worker 2: 无缓存
        
        发送请求序列：adapter_a, adapter_b, adapter_c, adapter_a
        
        Requirements: 2.1, 2.2, 2.5
        """
        router = AdapterAwareRouter(num_workers=3)
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={"/adapters/adapter_a"}, queue_length=0
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={"/adapters/adapter_b"}, queue_length=0
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=0
        ))
        
        # 请求 adapter_a -> 应该路由到 Worker 0（缓存命中）
        assert router.select_worker("/adapters/adapter_a") == 0
        
        # 请求 adapter_b -> 应该路由到 Worker 1（缓存命中）
        assert router.select_worker("/adapters/adapter_b") == 1
        
        # 请求 adapter_c -> 冷启动，路由到队列最短的 Worker
        router.select_worker("/adapters/adapter_c")
        
        # 请求 adapter_a -> 应该路由到 Worker 0（缓存命中）
        assert router.select_worker("/adapters/adapter_a") == 0
        
        # 验证统计
        stats = router.get_stats()
        assert stats['total_requests'] == 4
        assert stats['cache_hits'] == 3  # adapter_a x2, adapter_b x1
        assert stats['cache_misses'] == 1  # adapter_c
        assert stats['cold_starts'] == 1  # adapter_c
    
    def test_adapter_index_consistency_after_updates(self):
        """
        测试状态更新后 Adapter 索引的一致性
        
        验证 Adapter-to-Worker 索引在状态更新后保持正确
        
        Requirements: 2.5
        """
        router = AdapterAwareRouter(num_workers=3)
        adapter_a = "/adapters/adapter_a"
        adapter_b = "/adapters/adapter_b"
        
        # 初始状态：Worker 0 缓存 adapter_a
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_a}, queue_length=0
        ))
        
        # 验证索引
        assert router.get_workers_with_adapter(adapter_a) == {0}
        assert router.get_workers_with_adapter(adapter_b) == set()
        
        # Worker 1 也加载了 adapter_a
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_a}, queue_length=0
        ))
        
        # 验证索引更新
        assert router.get_workers_with_adapter(adapter_a) == {0, 1}
        
        # Worker 0 卸载 adapter_a，加载 adapter_b
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_b}, queue_length=0
        ))
        
        # 验证索引更新
        assert router.get_workers_with_adapter(adapter_a) == {1}
        assert router.get_workers_with_adapter(adapter_b) == {0}
    
    def test_cache_hit_rate_calculation(self):
        """
        测试缓存命中率计算
        
        发送 10 个请求，其中 7 个命中缓存，3 个未命中
        验证命中率为 70%
        
        Requirements: 2.1, 2.2
        """
        router = AdapterAwareRouter(num_workers=2)
        cached_adapter = "/adapters/cached"
        uncached_adapter = "/adapters/uncached"
        
        # Worker 0 缓存了 cached_adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={cached_adapter}, queue_length=0
        ))
        
        # 发送 7 个缓存命中请求
        for _ in range(7):
            router.select_worker(cached_adapter)
        
        # 发送 3 个缓存未命中请求
        for _ in range(3):
            router.select_worker(uncached_adapter)
        
        # 验证统计
        stats = router.get_stats()
        assert stats['total_requests'] == 10
        assert stats['cache_hits'] == 7
        assert stats['cache_misses'] == 3
        assert stats['cache_hit_rate'] == pytest.approx(0.7)


class TestRouterWithWorkerStateCache:
    """
    Test suite for integration between AdapterAwareRouter and WorkerStateCache
    
    Requirements: 2.1-2.5
    """
    
    def test_router_with_state_cache_integration(self):
        """
        测试 Router 与 WorkerStateCache 的集成
        
        模拟 Worker 状态上报流程，验证路由决策正确
        
        Requirements: 2.1, 2.5
        """
        num_workers = 3
        router = AdapterAwareRouter(num_workers=num_workers)
        cache = WorkerStateCache(num_workers=num_workers)
        
        adapter_dir = "/adapters/adapter_a"
        
        # 模拟 Worker 1 上报状态（缓存了 adapter）
        state_msg = {
            'type': 'worker_state',
            'worker_id': 1,
            'cached_adapters': [adapter_dir],
            'queue_length': 2,
            'gpu_memory_free': 1024 * 1024 * 1024,
            'timestamp': time.time()
        }
        cache.update_from_message(state_msg)
        
        # 从 cache 获取状态并更新 router
        state = cache.get(1)
        router.update_worker_state(1, state)
        
        # 验证路由决策
        selected = router.select_worker(adapter_dir)
        assert selected == 1
    
    def test_state_cache_health_check_affects_routing(self):
        """
        测试状态缓存健康检查影响路由决策
        
        模拟 Worker 心跳超时，验证不健康的 Worker 被排除
        
        Requirements: 2.1
        """
        num_workers = 2
        router = AdapterAwareRouter(num_workers=num_workers)
        cache = WorkerStateCache(
            num_workers=num_workers,
            heartbeat_interval_ms=100,
            heartbeat_timeout_ms=200
        )
        
        adapter_dir = "/adapters/adapter_a"
        
        # Worker 0 上报状态（缓存了 adapter）
        cache.update_from_message({
            'type': 'worker_state',
            'worker_id': 0,
            'cached_adapters': [adapter_dir],
            'queue_length': 0,
            'gpu_memory_free': 1024 * 1024 * 1024,
            'timestamp': time.time() - 1.0  # 1 秒前的心跳（已超时）
        })
        
        # Worker 1 上报状态（无缓存）
        cache.update_from_message({
            'type': 'worker_state',
            'worker_id': 1,
            'cached_adapters': [],
            'queue_length': 5,
            'gpu_memory_free': 1024 * 1024 * 1024,
            'timestamp': time.time()  # 当前时间
        })
        
        # 执行健康检查
        cache.check_health()
        
        # 更新 router 状态
        for wid in range(num_workers):
            state = cache.get(wid)
            router.update_worker_state(wid, state)
        
        # Worker 0 不健康，应该选择 Worker 1
        selected = router.select_worker(adapter_dir)
        assert selected == 1
    
    def test_multiple_state_updates_sequence(self):
        """
        测试多次状态更新序列
        
        模拟 Worker 状态变化，验证路由决策随之更新
        
        Requirements: 2.1, 2.5
        """
        num_workers = 3
        router = AdapterAwareRouter(num_workers=num_workers)
        
        adapter_a = "/adapters/adapter_a"
        adapter_b = "/adapters/adapter_b"
        
        # 初始状态：所有 Worker 无缓存
        for i in range(num_workers):
            router.update_worker_state(i, WorkerState(
                worker_id=i, cached_adapters=set(), queue_length=0
            ))
        
        # 请求 adapter_a -> 冷启动，选择任意 Worker
        first_selection = router.select_worker(adapter_a)
        assert first_selection in [0, 1, 2]
        
        # 模拟 Worker 0 加载了 adapter_a
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_a}, queue_length=1
        ))
        
        # 再次请求 adapter_a -> 应该选择 Worker 0
        assert router.select_worker(adapter_a) == 0
        
        # 模拟 Worker 1 也加载了 adapter_a，且队列更短
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_a}, queue_length=0
        ))
        
        # 再次请求 adapter_a -> 应该选择 Worker 1（队列更短）
        assert router.select_worker(adapter_a) == 1
        
        # 模拟 Worker 2 加载了 adapter_b
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters={adapter_b}, queue_length=0
        ))
        
        # 请求 adapter_b -> 应该选择 Worker 2
        assert router.select_worker(adapter_b) == 2


class TestWorkerRequestDistribution:
    """
    Test suite for worker request distribution
    
    Requirements: 2.1, 2.3, 2.4
    """
    
    def test_request_distribution_with_cache_affinity(self):
        """
        测试带缓存亲和性的请求分布
        
        场景：
        - Worker 0: 缓存 adapter_a
        - Worker 1: 缓存 adapter_b
        - Worker 2: 缓存 adapter_c
        
        发送多个请求，验证分布符合缓存亲和性
        
        Requirements: 2.1, 2.3
        """
        router = AdapterAwareRouter(num_workers=3)
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={"/adapters/adapter_a"}, queue_length=0
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={"/adapters/adapter_b"}, queue_length=0
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters={"/adapters/adapter_c"}, queue_length=0
        ))
        
        # 发送请求并记录分布
        request_counts = {0: 0, 1: 0, 2: 0}
        
        # 发送 adapter_a 请求
        for _ in range(5):
            selected = router.select_worker("/adapters/adapter_a")
            request_counts[selected] += 1
        
        # 发送 adapter_b 请求
        for _ in range(3):
            selected = router.select_worker("/adapters/adapter_b")
            request_counts[selected] += 1
        
        # 发送 adapter_c 请求
        for _ in range(2):
            selected = router.select_worker("/adapters/adapter_c")
            request_counts[selected] += 1
        
        # 验证分布
        stats = router.get_stats()
        assert stats['worker_request_counts'][0] == 5  # adapter_a
        assert stats['worker_request_counts'][1] == 3  # adapter_b
        assert stats['worker_request_counts'][2] == 2  # adapter_c
    
    def test_load_balancing_with_equal_cache(self):
        """
        测试相同缓存情况下的负载均衡
        
        场景：所有 Worker 都缓存了同一个 adapter
        验证请求根据队列长度分布
        
        Requirements: 2.3, 2.4
        """
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/adapters/shared_adapter"
        
        # 所有 Worker 都缓存了 adapter，但队列长度不同
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=10
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_dir}, queue_length=5
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters={adapter_dir}, queue_length=2
        ))
        
        # 发送请求，应该优先选择队列最短的 Worker
        selected = router.select_worker(adapter_dir)
        assert selected == 2  # 队列最短
        
        # 模拟 Worker 2 队列增长
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters={adapter_dir}, queue_length=8
        ))
        
        # 再次发送请求，应该选择 Worker 1
        selected = router.select_worker(adapter_dir)
        assert selected == 1


class TestEdgeCases:
    """
    Test suite for edge cases in routing
    
    Requirements: 2.1-2.5
    """
    
    def test_single_worker_scenario(self):
        """
        测试单 Worker 场景
        
        所有请求都应该路由到唯一的 Worker
        
        Requirements: 2.1
        """
        router = AdapterAwareRouter(num_workers=1)
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={"/adapters/adapter_a"}, queue_length=0
        ))
        
        # 所有请求都应该路由到 Worker 0
        for _ in range(10):
            assert router.select_worker("/adapters/adapter_a") == 0
            assert router.select_worker("/adapters/adapter_b") == 0
    
    def test_all_workers_unhealthy_fallback(self):
        """
        测试所有 Worker 不健康时的回退行为
        
        应该回退到 Round-Robin
        
        Requirements: 2.1
        """
        router = AdapterAwareRouter(num_workers=3)
        
        # 设置所有 Worker 为不健康
        for i in range(3):
            router.update_worker_state(i, WorkerState(
                worker_id=i, cached_adapters=set(), queue_length=0, is_healthy=False
            ))
        
        # 应该回退到 Round-Robin
        selections = [router.select_worker("/any/adapter") for _ in range(6)]
        assert selections == [0, 1, 2, 0, 1, 2]
    
    def test_empty_adapter_dir(self):
        """
        测试空 adapter 目录
        
        应该正常处理，按冷启动逻辑路由
        
        Requirements: 2.1
        """
        router = AdapterAwareRouter(num_workers=2)
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters=set(), queue_length=5
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters=set(), queue_length=2
        ))
        
        # 空 adapter 目录应该按冷启动处理
        selected = router.select_worker("")
        assert selected == 1  # 队列最短
    
    def test_very_long_queue_scenario(self):
        """
        测试超长队列场景
        
        验证队列阈值过滤正常工作
        
        Requirements: 2.1
        """
        config = RoutingConfig(max_queue_length=50)
        router = AdapterAwareRouter(num_workers=3, config=config)
        adapter_dir = "/adapters/adapter_a"
        
        # Worker 0: 缓存命中，队列超限
        # Worker 1: 缓存命中，队列正常但较长
        # Worker 2: 缓存命中，队列正常且较短
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=100
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_dir}, queue_length=30
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters={adapter_dir}, queue_length=10
        ))
        
        # Worker 0 超限被排除，Worker 2 队列最短
        # Score(1) = 1.0 - 0.1*30 = -2.0
        # Score(2) = 1.0 - 0.1*10 = 0.0
        # 应该选择 Worker 2（分数最高）
        selected = router.select_worker(adapter_dir)
        assert selected == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
