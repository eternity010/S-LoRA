"""
Unit tests for Worker State Data Structures

Tests the correctness of WorkerState, RoutingStats, and RoutingConfig
data classes used in the adapter-aware routing system.

Requirements: 1.1, 1.2, 6.1-6.5, 7.1-7.3
"""

import pytest
import time
from slora.server.router.worker_state import WorkerState, RoutingStats, RoutingConfig


class TestWorkerState:
    """Test suite for WorkerState data class - Requirements 1.1, 1.2"""
    
    def test_initialization_defaults(self):
        """测试默认值初始化"""
        state = WorkerState(worker_id=0)
        
        assert state.worker_id == 0
        assert state.cached_adapters == set()
        assert state.queue_length == 0
        assert state.gpu_memory_free == 0
        assert state.is_healthy == True
        assert state.last_heartbeat > 0
    
    def test_initialization_with_values(self):
        """测试带参数初始化"""
        adapters = {"/path/to/adapter_a", "/path/to/adapter_b"}
        state = WorkerState(
            worker_id=1,
            cached_adapters=adapters,
            queue_length=5,
            gpu_memory_free=1024 * 1024 * 1024,
            is_healthy=True
        )
        
        assert state.worker_id == 1
        assert state.cached_adapters == adapters
        assert state.queue_length == 5
        assert state.gpu_memory_free == 1024 * 1024 * 1024
    
    def test_has_adapter(self):
        """测试 has_adapter 方法"""
        adapter_a = "/path/to/adapter_a"
        adapter_b = "/path/to/adapter_b"
        
        state = WorkerState(
            worker_id=0,
            cached_adapters={adapter_a}
        )
        
        assert state.has_adapter(adapter_a) == True
        assert state.has_adapter(adapter_b) == False
    
    def test_update_heartbeat(self):
        """测试心跳更新"""
        state = WorkerState(worker_id=0)
        old_heartbeat = state.last_heartbeat
        
        time.sleep(0.01)  # 等待一小段时间
        state.update_heartbeat()
        
        assert state.last_heartbeat > old_heartbeat
    
    def test_to_dict(self):
        """测试序列化为字典"""
        adapters = {"/path/to/adapter_a"}
        state = WorkerState(
            worker_id=1,
            cached_adapters=adapters,
            queue_length=3,
            gpu_memory_free=1000,
            is_healthy=True
        )
        
        d = state.to_dict()
        
        assert d['worker_id'] == 1
        assert d['cached_adapters'] == ["/path/to/adapter_a"]
        assert d['queue_length'] == 3
        assert d['gpu_memory_free'] == 1000
        assert d['is_healthy'] == True
    
    def test_from_dict(self):
        """测试从字典反序列化"""
        data = {
            'worker_id': 2,
            'cached_adapters': ["/path/to/adapter_a", "/path/to/adapter_b"],
            'queue_length': 7,
            'gpu_memory_free': 2000,
            'is_healthy': False
        }
        
        state = WorkerState.from_dict(data)
        
        assert state.worker_id == 2
        assert state.cached_adapters == {"/path/to/adapter_a", "/path/to/adapter_b"}
        assert state.queue_length == 7
        assert state.gpu_memory_free == 2000
        assert state.is_healthy == False


class TestRoutingStats:
    """Test suite for RoutingStats data class - Requirements 7.1-7.3"""
    
    def test_initialization_defaults(self):
        """测试默认值初始化"""
        stats = RoutingStats()
        
        assert stats.total_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.cold_starts == 0
        assert stats.hot_adapter_redistributions == 0
        assert stats.worker_request_counts == {}
    
    def test_cache_hit_rate_zero_requests(self):
        """测试零请求时的命中率"""
        stats = RoutingStats()
        
        assert stats.cache_hit_rate == 0.0
    
    def test_cache_hit_rate_calculation(self):
        """测试命中率计算"""
        stats = RoutingStats()
        stats.total_requests = 100
        stats.cache_hits = 75
        stats.cache_misses = 25
        
        assert stats.cache_hit_rate == 0.75
    
    def test_cache_miss_rate_calculation(self):
        """测试未命中率计算"""
        stats = RoutingStats()
        stats.total_requests = 100
        stats.cache_hits = 75
        stats.cache_misses = 25
        
        assert stats.cache_miss_rate == 0.25
    
    def test_record_request_cache_hit(self):
        """测试记录缓存命中请求"""
        stats = RoutingStats()
        
        stats.record_request(worker_id=0, cache_hit=True)
        
        assert stats.total_requests == 1
        assert stats.cache_hits == 1
        assert stats.cache_misses == 0
        assert stats.worker_request_counts[0] == 1
    
    def test_record_request_cache_miss(self):
        """测试记录缓存未命中请求"""
        stats = RoutingStats()
        
        stats.record_request(worker_id=1, cache_hit=False)
        
        assert stats.total_requests == 1
        assert stats.cache_hits == 0
        assert stats.cache_misses == 1
        assert stats.worker_request_counts[1] == 1
    
    def test_record_request_cold_start(self):
        """测试记录冷启动请求"""
        stats = RoutingStats()
        
        stats.record_request(worker_id=0, cache_hit=False, is_cold_start=True)
        
        assert stats.cold_starts == 1
    
    def test_record_hot_adapter_redistribution(self):
        """测试记录热点 Adapter 重分配"""
        stats = RoutingStats()
        
        stats.record_hot_adapter_redistribution()
        stats.record_hot_adapter_redistribution()
        
        assert stats.hot_adapter_redistributions == 2
    
    def test_reset(self):
        """测试重置统计"""
        stats = RoutingStats()
        stats.record_request(0, True)
        stats.record_request(1, False, True)
        stats.record_hot_adapter_redistribution()
        
        stats.reset()
        
        assert stats.total_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.cold_starts == 0
        assert stats.hot_adapter_redistributions == 0
        assert stats.worker_request_counts == {}
    
    def test_to_dict(self):
        """测试序列化为字典"""
        stats = RoutingStats()
        stats.record_request(0, True)
        stats.record_request(1, False)
        
        d = stats.to_dict()
        
        assert d['total_requests'] == 2
        assert d['cache_hits'] == 1
        assert d['cache_misses'] == 1
        assert d['cache_hit_rate'] == 0.5
        assert d['worker_request_counts'] == {0: 1, 1: 1}
    
    def test_str_representation(self):
        """测试字符串表示"""
        stats = RoutingStats()
        stats.total_requests = 100
        stats.cache_hits = 80
        stats.cache_misses = 20
        stats.cold_starts = 5
        
        s = str(stats)
        
        assert "total=100" in s
        assert "hits=80" in s
        assert "misses=20" in s
        assert "80.00%" in s


class TestRoutingConfig:
    """Test suite for RoutingConfig data class - Requirements 6.1-6.5"""
    
    def test_initialization_defaults(self):
        """测试默认值初始化"""
        config = RoutingConfig()
        
        assert config.strategy == 'adapter-aware'
        assert config.w1 == 1.0
        assert config.w2 == 0.1
        assert config.heartbeat_interval_ms == 100
        assert config.heartbeat_timeout_ms == 300
        assert config.max_queue_length == 100
        assert config.hot_adapter_threshold == 10.0
    
    def test_initialization_custom_values(self):
        """测试自定义值初始化"""
        config = RoutingConfig(
            strategy='round-robin',
            w1=2.0,
            w2=0.5,
            heartbeat_interval_ms=200,
            heartbeat_timeout_ms=600,
            max_queue_length=50,
            hot_adapter_threshold=20.0
        )
        
        assert config.strategy == 'round-robin'
        assert config.w1 == 2.0
        assert config.w2 == 0.5
        assert config.heartbeat_interval_ms == 200
        assert config.heartbeat_timeout_ms == 600
        assert config.max_queue_length == 50
        assert config.hot_adapter_threshold == 20.0
    
    def test_invalid_strategy(self):
        """测试无效的策略"""
        with pytest.raises(ValueError):
            RoutingConfig(strategy='invalid')
    
    def test_invalid_w1(self):
        """测试无效的 w1"""
        with pytest.raises(ValueError):
            RoutingConfig(w1=-1.0)
    
    def test_invalid_w2(self):
        """测试无效的 w2"""
        with pytest.raises(ValueError):
            RoutingConfig(w2=-0.5)
    
    def test_invalid_heartbeat_interval(self):
        """测试无效的心跳间隔"""
        with pytest.raises(ValueError):
            RoutingConfig(heartbeat_interval_ms=0)
        
        with pytest.raises(ValueError):
            RoutingConfig(heartbeat_interval_ms=-100)
    
    def test_invalid_heartbeat_timeout(self):
        """测试无效的心跳超时"""
        with pytest.raises(ValueError):
            RoutingConfig(heartbeat_timeout_ms=0)
    
    def test_invalid_max_queue_length(self):
        """测试无效的最大队列长度"""
        with pytest.raises(ValueError):
            RoutingConfig(max_queue_length=0)
    
    def test_invalid_hot_adapter_threshold(self):
        """测试无效的热点阈值"""
        with pytest.raises(ValueError):
            RoutingConfig(hot_adapter_threshold=0)
    
    def test_heartbeat_interval_sec(self):
        """测试心跳间隔秒数转换"""
        config = RoutingConfig(heartbeat_interval_ms=100)
        
        assert config.heartbeat_interval_sec == 0.1
    
    def test_heartbeat_timeout_sec(self):
        """测试心跳超时秒数转换"""
        config = RoutingConfig(heartbeat_timeout_ms=300)
        
        assert config.heartbeat_timeout_sec == 0.3
    
    def test_to_dict(self):
        """测试序列化为字典"""
        config = RoutingConfig()
        
        d = config.to_dict()
        
        assert d['strategy'] == 'adapter-aware'
        assert d['w1'] == 1.0
        assert d['w2'] == 0.1
        assert d['heartbeat_interval_ms'] == 100
        assert d['heartbeat_timeout_ms'] == 300
        assert d['max_queue_length'] == 100
        assert d['hot_adapter_threshold'] == 10.0
    
    def test_from_dict(self):
        """测试从字典反序列化"""
        data = {
            'strategy': 'round-robin',
            'w1': 1.5,
            'w2': 0.2,
            'heartbeat_interval_ms': 150,
            'heartbeat_timeout_ms': 450,
            'max_queue_length': 75,
            'hot_adapter_threshold': 15.0
        }
        
        config = RoutingConfig.from_dict(data)
        
        assert config.strategy == 'round-robin'
        assert config.w1 == 1.5
        assert config.w2 == 0.2
        assert config.heartbeat_interval_ms == 150
        assert config.heartbeat_timeout_ms == 450
        assert config.max_queue_length == 75
        assert config.hot_adapter_threshold == 15.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
