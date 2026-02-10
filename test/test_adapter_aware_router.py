"""
Unit tests for Adapter-Aware Router

Tests the correctness of the adapter-aware routing strategy including:
- Scoring formula correctness
- Worker selection logic
- Tie-breaking behavior
- Round-robin fallback
- Hot adapter handling

Requirements: 2.1-2.5, 3.1-3.4, 5.1-5.3
"""

import pytest
import time
from slora.server.router.adapter_aware_router import AdapterAwareRouter, RequestRateTracker
from slora.server.router.worker_state import WorkerState, RoutingStats, RoutingConfig


class TestAdapterAwareRouterInit:
    """Test suite for AdapterAwareRouter initialization"""
    
    def test_initialization_default_config(self):
        """测试使用默认配置初始化"""
        router = AdapterAwareRouter(num_workers=3)
        
        assert router.num_workers == 3
        assert router.config.w1 == 1.0
        assert router.config.w2 == 0.1
        assert router.config.strategy == 'adapter-aware'
        assert len(router.worker_states) == 3
    
    def test_initialization_custom_config(self):
        """测试使用自定义配置初始化"""
        config = RoutingConfig(w1=2.0, w2=0.5, max_queue_length=50)
        router = AdapterAwareRouter(num_workers=4, config=config)
        
        assert router.num_workers == 4
        assert router.config.w1 == 2.0
        assert router.config.w2 == 0.5
        assert router.config.max_queue_length == 50
    
    def test_initialization_invalid_workers(self):
        """测试无效的 Worker 数量"""
        with pytest.raises(ValueError):
            AdapterAwareRouter(num_workers=0)
        
        with pytest.raises(ValueError):
            AdapterAwareRouter(num_workers=-1)
    
    def test_worker_states_initialized(self):
        """测试 Worker 状态正确初始化"""
        router = AdapterAwareRouter(num_workers=3)
        
        for i in range(3):
            assert i in router.worker_states
            assert router.worker_states[i].worker_id == i
            assert router.worker_states[i].queue_length == 0
            assert len(router.worker_states[i].cached_adapters) == 0
            assert router.worker_states[i].is_healthy == True


class TestScoringFormula:
    """Test suite for scoring formula correctness - Requirements 2.2"""
    
    def test_score_cache_hit_empty_queue(self):
        """测试缓存命中且队列为空的评分"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter_a"
        
        # 设置 Worker 0 缓存了 adapter
        state = WorkerState(
            worker_id=0,
            cached_adapters={adapter_dir},
            queue_length=0
        )
        router.update_worker_state(0, state)
        
        # Score = w1 * 1 - w2 * 0 = 1.0 * 1 - 0.1 * 0 = 1.0
        score = router.calculate_score(0, adapter_dir)
        assert score == 1.0
    
    def test_score_cache_miss_empty_queue(self):
        """测试缓存未命中且队列为空的评分"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0 没有缓存 adapter
        # Score = w1 * 0 - w2 * 0 = 0.0
        score = router.calculate_score(0, adapter_dir)
        assert score == 0.0
    
    def test_score_cache_hit_with_queue(self):
        """测试缓存命中且有队列的评分"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter_a"
        
        # 设置 Worker 0 缓存了 adapter，队列长度为 5
        state = WorkerState(
            worker_id=0,
            cached_adapters={adapter_dir},
            queue_length=5
        )
        router.update_worker_state(0, state)
        
        # Score = w1 * 1 - w2 * 5 = 1.0 * 1 - 0.1 * 5 = 0.5
        score = router.calculate_score(0, adapter_dir)
        assert score == pytest.approx(0.5)
    
    def test_score_cache_miss_with_queue(self):
        """测试缓存未命中且有队列的评分"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter_a"
        
        # 设置 Worker 0 队列长度为 10，没有缓存
        state = WorkerState(
            worker_id=0,
            cached_adapters=set(),
            queue_length=10
        )
        router.update_worker_state(0, state)
        
        # Score = w1 * 0 - w2 * 10 = 0 - 0.1 * 10 = -1.0
        score = router.calculate_score(0, adapter_dir)
        assert score == pytest.approx(-1.0)
    
    def test_score_with_custom_weights(self):
        """测试自定义权重的评分"""
        config = RoutingConfig(w1=2.0, w2=0.5)
        router = AdapterAwareRouter(num_workers=2, config=config)
        adapter_dir = "/path/to/adapter_a"
        
        # 设置 Worker 0 缓存了 adapter，队列长度为 3
        state = WorkerState(
            worker_id=0,
            cached_adapters={adapter_dir},
            queue_length=3
        )
        router.update_worker_state(0, state)
        
        # Score = w1 * 1 - w2 * 3 = 2.0 * 1 - 0.5 * 3 = 0.5
        score = router.calculate_score(0, adapter_dir)
        assert score == pytest.approx(0.5)
    
    def test_score_unknown_worker(self):
        """测试未知 Worker 的评分"""
        router = AdapterAwareRouter(num_workers=2)
        
        # Worker 99 不存在
        score = router.calculate_score(99, "/path/to/adapter")
        assert score == float('-inf')


class TestWorkerSelection:
    """Test suite for worker selection logic - Requirements 2.3, 2.4"""
    
    def test_select_worker_with_cache_hit(self):
        """测试选择有缓存命中的 Worker"""
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 1 缓存了 adapter
        state = WorkerState(
            worker_id=1,
            cached_adapters={adapter_dir},
            queue_length=0
        )
        router.update_worker_state(1, state)
        
        # 应该选择 Worker 1
        selected = router.select_worker(adapter_dir)
        assert selected == 1
    
    def test_select_worker_highest_score(self):
        """测试选择最高分的 Worker"""
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0: 缓存命中，队列长度 5 -> Score = 1.0 - 0.5 = 0.5
        # Worker 1: 缓存命中，队列长度 2 -> Score = 1.0 - 0.2 = 0.8
        # Worker 2: 缓存未命中，队列长度 0 -> Score = 0.0 - 0.0 = 0.0
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=5
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_dir}, queue_length=2
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=0
        ))
        
        # 应该选择 Worker 1（最高分 0.8）
        selected = router.select_worker(adapter_dir)
        assert selected == 1
    
    def test_select_worker_tie_breaking(self):
        """测试平分时的 tie-breaking 逻辑"""
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0: 缓存命中，队列长度 5 -> Score = 1.0 - 0.5 = 0.5
        # Worker 1: 缓存命中，队列长度 5 -> Score = 1.0 - 0.5 = 0.5
        # Worker 2: 缓存命中，队列长度 3 -> Score = 1.0 - 0.3 = 0.7
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=5
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_dir}, queue_length=5
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters={adapter_dir}, queue_length=3
        ))
        
        # 应该选择 Worker 2（最高分 0.7）
        selected = router.select_worker(adapter_dir)
        assert selected == 2
    
    def test_select_worker_tie_breaking_same_score(self):
        """测试相同分数时选择队列最短的 Worker"""
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/path/to/adapter_a"
        
        # 所有 Worker 都没有缓存，分数都是 -w2 * queue_length
        # Worker 0: 队列长度 5 -> Score = -0.5
        # Worker 1: 队列长度 3 -> Score = -0.3
        # Worker 2: 队列长度 3 -> Score = -0.3
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters=set(), queue_length=5
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters=set(), queue_length=3
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=3
        ))
        
        # Worker 1 和 2 分数相同，应该选择队列最短的（都是 3）
        # 由于排序稳定性，应该选择 Worker 1（先出现）
        selected = router.select_worker(adapter_dir)
        assert selected in [1, 2]  # 两者都可接受
    
    def test_select_worker_cold_start(self):
        """测试冷启动场景（无缓存命中）"""
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/path/to/new_adapter"
        
        # 所有 Worker 都没有缓存该 adapter
        # Worker 0: 队列长度 5
        # Worker 1: 队列长度 2
        # Worker 2: 队列长度 8
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters=set(), queue_length=5
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters=set(), queue_length=2
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=8
        ))
        
        # 应该选择队列最短的 Worker 1
        selected = router.select_worker(adapter_dir)
        assert selected == 1
    
    def test_select_worker_unhealthy_excluded(self):
        """测试不健康的 Worker 被排除"""
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0: 缓存命中，健康
        # Worker 1: 缓存命中，不健康
        # Worker 2: 缓存未命中，健康
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=5, is_healthy=True
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_dir}, queue_length=0, is_healthy=False
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=0, is_healthy=True
        ))
        
        # Worker 1 虽然分数最高但不健康，应该选择 Worker 0
        selected = router.select_worker(adapter_dir)
        assert selected == 0


class TestRoundRobinFallback:
    """Test suite for round-robin fallback - Requirements 8.2"""
    
    def test_round_robin_strategy(self):
        """测试 Round-Robin 策略"""
        config = RoutingConfig(strategy='round-robin')
        router = AdapterAwareRouter(num_workers=3, config=config)
        
        # 应该按顺序轮询
        selections = [router.select_worker("/any/adapter") for _ in range(6)]
        assert selections == [0, 1, 2, 0, 1, 2]
    
    def test_round_robin_fallback_no_healthy_workers(self):
        """测试所有 Worker 不健康时回退到 Round-Robin"""
        router = AdapterAwareRouter(num_workers=3)
        
        # 设置所有 Worker 为不健康
        for i in range(3):
            router.update_worker_state(i, WorkerState(
                worker_id=i, cached_adapters=set(), queue_length=0, is_healthy=False
            ))
        
        # 应该回退到 Round-Robin
        selections = [router.select_worker("/any/adapter") for _ in range(3)]
        assert selections == [0, 1, 2]


class TestStatistics:
    """Test suite for routing statistics"""
    
    def test_stats_cache_hit(self):
        """测试缓存命中统计"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0 缓存了 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=0
        ))
        
        # 发送请求
        router.select_worker(adapter_dir)
        
        stats = router.get_stats()
        assert stats['total_requests'] == 1
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 0
    
    def test_stats_cache_miss(self):
        """测试缓存未命中统计"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter_a"
        
        # 没有 Worker 缓存该 adapter
        router.select_worker(adapter_dir)
        
        stats = router.get_stats()
        assert stats['total_requests'] == 1
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 1
    
    def test_stats_cold_start(self):
        """测试冷启动统计"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/new_adapter"
        
        # 没有 Worker 缓存该 adapter（冷启动）
        router.select_worker(adapter_dir)
        
        stats = router.get_stats()
        assert stats['cold_starts'] == 1
    
    def test_stats_reset(self):
        """测试统计重置"""
        router = AdapterAwareRouter(num_workers=2)
        
        # 发送一些请求
        router.select_worker("/adapter1")
        router.select_worker("/adapter2")
        
        # 重置统计
        router.reset_stats()
        
        stats = router.get_stats()
        assert stats['total_requests'] == 0
        assert stats['cache_hits'] == 0
    
    def test_stats_includes_worker_health(self):
        """测试统计包含 Worker 健康状态"""
        router = AdapterAwareRouter(num_workers=3)
        
        # 标记一个 Worker 为不健康
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters=set(), queue_length=0, is_healthy=False
        ))
        
        stats = router.get_stats()
        assert stats['healthy_workers'] == 2
        assert stats['unhealthy_workers'] == 1
    
    def test_stats_includes_cached_adapters_count(self):
        """测试统计包含缓存的 Adapter 数量"""
        router = AdapterAwareRouter(num_workers=2)
        
        # 添加一些缓存的 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={"/adapter_a", "/adapter_b"}, queue_length=0
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={"/adapter_c"}, queue_length=0
        ))
        
        stats = router.get_stats()
        assert stats['total_cached_adapters'] == 3


class TestQueueThreshold:
    """Test suite for queue threshold filtering - Requirements 3.4"""
    
    def test_queue_threshold_filtering(self):
        """测试队列超限的 Worker 被过滤"""
        config = RoutingConfig(max_queue_length=10)
        router = AdapterAwareRouter(num_workers=3, config=config)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0: 缓存命中，队列长度 15（超限）
        # Worker 1: 缓存命中，队列长度 5（未超限）
        # Worker 2: 缓存未命中，队列长度 3（未超限）
        
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=15
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_dir}, queue_length=5
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=3
        ))
        
        # Worker 0 虽然缓存命中但超限，应该选择 Worker 1
        selected = router.select_worker(adapter_dir)
        assert selected == 1
    
    def test_all_workers_exceed_threshold(self):
        """测试所有 Worker 都超限时选择队列最短的"""
        config = RoutingConfig(max_queue_length=10)
        router = AdapterAwareRouter(num_workers=3, config=config)
        adapter_dir = "/path/to/adapter_a"
        
        # 所有 Worker 都超限
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=20
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters={adapter_dir}, queue_length=15
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=12
        ))
        
        # 应该选择 Worker 1（缓存命中且队列最短）
        selected = router.select_worker(adapter_dir)
        assert selected == 1
    
    def test_queue_at_threshold_excluded(self):
        """测试队列长度等于阈值时被排除"""
        config = RoutingConfig(max_queue_length=10)
        router = AdapterAwareRouter(num_workers=2)
        router.config = config
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0: 队列长度 = 10（等于阈值，应被排除）
        # Worker 1: 队列长度 = 9（小于阈值）
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=10
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters=set(), queue_length=9
        ))
        
        # Worker 0 等于阈值被排除，应该选择 Worker 1
        selected = router.select_worker(adapter_dir)
        assert selected == 1


class TestColdStart:
    """Test suite for cold start handling - Requirements 4.1, 4.2"""
    
    def test_cold_start_selects_shortest_queue(self):
        """测试冷启动时选择队列最短的 Worker"""
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/path/to/new_adapter"
        
        # 没有 Worker 缓存该 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters=set(), queue_length=10
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters=set(), queue_length=3
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters=set(), queue_length=7
        ))
        
        # 应该选择队列最短的 Worker 1
        selected = router.select_worker(adapter_dir)
        assert selected == 1
    
    def test_cold_start_recorded_in_stats(self):
        """测试冷启动被记录到统计"""
        router = AdapterAwareRouter(num_workers=2)
        
        # 发送一个冷启动请求
        router.select_worker("/new/adapter")
        
        stats = router.get_stats()
        assert stats['cold_starts'] == 1
        assert stats['cache_misses'] == 1
    
    def test_not_cold_start_when_cached(self):
        """测试有缓存时不是冷启动"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0 缓存了 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=0
        ))
        
        router.select_worker(adapter_dir)
        
        stats = router.get_stats()
        assert stats['cold_starts'] == 0
        assert stats['cache_hits'] == 1


class TestAdapterIndex:
    """Test suite for adapter-to-worker index - Requirements 2.5"""
    
    def test_index_update_on_state_change(self):
        """测试状态更新时索引更新"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter_a"
        
        # 初始状态：没有 Worker 缓存该 adapter
        assert router.get_workers_with_adapter(adapter_dir) == set()
        
        # Worker 0 加载了 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=0
        ))
        
        # 索引应该更新
        assert router.get_workers_with_adapter(adapter_dir) == {0}
    
    def test_index_multiple_workers(self):
        """测试多个 Worker 缓存同一 adapter"""
        router = AdapterAwareRouter(num_workers=3)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0 和 2 都缓存了 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=0
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2, cached_adapters={adapter_dir}, queue_length=0
        ))
        
        assert router.get_workers_with_adapter(adapter_dir) == {0, 2}
    
    def test_index_removal_on_unload(self):
        """测试 adapter 卸载时索引更新"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0 加载了 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=0
        ))
        assert router.get_workers_with_adapter(adapter_dir) == {0}
        
        # Worker 0 卸载了 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters=set(), queue_length=0
        ))
        assert router.get_workers_with_adapter(adapter_dir) == set()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestRequestRateTracker:
    """Test suite for RequestRateTracker - Requirements 5.2"""
    
    def test_initial_rate_is_zero(self):
        """测试初始请求率为零"""
        tracker = RequestRateTracker(window_size_sec=1.0)
        
        assert tracker.get_request_rate("/adapter_a") == 0.0
    
    def test_record_and_get_rate(self):
        """测试记录请求并获取请求率"""
        tracker = RequestRateTracker(window_size_sec=1.0)
        
        # 记录 5 个请求
        for _ in range(5):
            tracker.record_request("/adapter_a")
        
        # 请求率应该是 5 req/s
        rate = tracker.get_request_rate("/adapter_a")
        assert rate == 5.0
    
    def test_is_hot(self):
        """测试热点检测"""
        tracker = RequestRateTracker(window_size_sec=1.0)
        
        # 记录 15 个请求
        for _ in range(15):
            tracker.record_request("/adapter_a")
        
        # 阈值 10，应该是热点
        assert tracker.is_hot("/adapter_a", threshold=10.0) == True
        
        # 阈值 20，不应该是热点
        assert tracker.is_hot("/adapter_a", threshold=20.0) == False
    
    def test_window_expiration(self):
        """测试滑动窗口过期"""
        tracker = RequestRateTracker(window_size_sec=0.1)  # 100ms 窗口
        
        # 记录请求
        for _ in range(10):
            tracker.record_request("/adapter_a")
        
        # 等待窗口过期
        time.sleep(0.15)
        
        # 请求率应该为 0
        assert tracker.get_request_rate("/adapter_a") == 0.0
    
    def test_reset(self):
        """测试重置"""
        tracker = RequestRateTracker(window_size_sec=1.0)
        
        tracker.record_request("/adapter_a")
        tracker.record_request("/adapter_b")
        
        tracker.reset()
        
        assert tracker.get_request_rate("/adapter_a") == 0.0
        assert tracker.get_request_rate("/adapter_b") == 0.0


class TestHotAdapterHandling:
    """Test suite for hot adapter handling - Requirements 5.1, 5.3"""
    
    def test_hot_adapter_distribution(self):
        """测试热点 Adapter 请求分布到多个 Worker"""
        # 设置较低的热点阈值便于测试
        config = RoutingConfig(hot_adapter_threshold=5.0)
        router = AdapterAwareRouter(num_workers=3, config=config)
        adapter_dir = "/path/to/hot_adapter"
        
        # Worker 0 缓存了 adapter
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=0
        ))
        
        # 发送足够多的请求使其成为热点
        selections = []
        for _ in range(20):
            selected = router.select_worker(adapter_dir)
            selections.append(selected)
        
        # 热点 Adapter 应该分布到多个 Worker
        unique_workers = set(selections)
        assert len(unique_workers) > 1, "Hot adapter should be distributed to multiple workers"
    
    def test_hot_adapter_redistribution_stats(self):
        """测试热点 Adapter 重分配统计"""
        config = RoutingConfig(hot_adapter_threshold=3.0)
        router = AdapterAwareRouter(num_workers=3, config=config)
        adapter_dir = "/path/to/hot_adapter"
        
        # 发送足够多的请求触发热点处理
        for _ in range(10):
            router.select_worker(adapter_dir)
        
        stats = router.get_stats()
        # 应该有热点重分配记录
        assert stats['hot_adapter_redistributions'] > 0
    
    def test_is_hot_adapter_method(self):
        """测试 is_hot_adapter 方法"""
        config = RoutingConfig(hot_adapter_threshold=5.0)
        router = AdapterAwareRouter(num_workers=2, config=config)
        adapter_dir = "/path/to/adapter"
        
        # 初始不是热点
        assert router.is_hot_adapter(adapter_dir) == False
        
        # 发送足够多的请求
        for _ in range(10):
            router.select_worker(adapter_dir)
        
        # 现在应该是热点
        assert router.is_hot_adapter(adapter_dir) == True
    
    def test_get_adapter_request_rate(self):
        """测试获取 Adapter 请求率"""
        router = AdapterAwareRouter(num_workers=2)
        adapter_dir = "/path/to/adapter"
        
        # 初始请求率为 0
        assert router.get_adapter_request_rate(adapter_dir) == 0.0
        
        # 发送 5 个请求
        for _ in range(5):
            router.select_worker(adapter_dir)
        
        # 请求率应该是 5
        rate = router.get_adapter_request_rate(adapter_dir)
        assert rate == 5.0
    
    def test_hot_adapter_respects_queue_threshold(self):
        """测试热点 Adapter 处理仍然遵守队列阈值"""
        config = RoutingConfig(hot_adapter_threshold=3.0, max_queue_length=10)
        router = AdapterAwareRouter(num_workers=2, config=config)
        adapter_dir = "/path/to/hot_adapter"
        
        # Worker 0 队列超限
        router.update_worker_state(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}, queue_length=15
        ))
        # Worker 1 队列正常
        router.update_worker_state(1, WorkerState(
            worker_id=1, cached_adapters=set(), queue_length=5
        ))
        
        # 发送足够多的请求触发热点处理
        selections = []
        for _ in range(10):
            selected = router.select_worker(adapter_dir)
            selections.append(selected)
        
        # 所有请求应该路由到 Worker 1（Worker 0 超限）
        assert all(s == 1 for s in selections)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
