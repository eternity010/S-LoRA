"""
Unit tests for Worker State Cache

Tests the correctness of the WorkerStateCache including:
- State updates
- Health checking
- Adapter-to-worker lookup

Requirements: 1.4, 1.5
"""

import pytest
import time
from slora.server.router.worker_state_cache import WorkerStateCache
from slora.server.router.worker_state import WorkerState


class TestWorkerStateCacheInit:
    """Test suite for WorkerStateCache initialization"""
    
    def test_initialization(self):
        """测试初始化"""
        cache = WorkerStateCache(num_workers=3)
        
        assert cache.num_workers == 3
        assert len(cache.get_all()) == 3
    
    def test_initialization_invalid_workers(self):
        """测试无效的 Worker 数量"""
        with pytest.raises(ValueError):
            WorkerStateCache(num_workers=0)
        
        with pytest.raises(ValueError):
            WorkerStateCache(num_workers=-1)
    
    def test_initialization_invalid_heartbeat(self):
        """测试无效的心跳参数"""
        with pytest.raises(ValueError):
            WorkerStateCache(num_workers=3, heartbeat_interval_ms=0)
        
        with pytest.raises(ValueError):
            WorkerStateCache(num_workers=3, heartbeat_timeout_ms=-100)
    
    def test_all_workers_healthy_initially(self):
        """测试初始状态所有 Worker 健康"""
        cache = WorkerStateCache(num_workers=3)
        
        healthy = cache.get_healthy_workers()
        assert len(healthy) == 3
        assert set(healthy) == {0, 1, 2}


class TestWorkerStateCacheUpdate:
    """Test suite for state update functionality - Requirements 1.4"""
    
    def test_update_state(self):
        """测试状态更新"""
        cache = WorkerStateCache(num_workers=2)
        
        state = WorkerState(
            worker_id=0,
            cached_adapters={"/path/to/adapter_a"},
            queue_length=5
        )
        cache.update(0, state)
        
        retrieved = cache.get(0)
        assert retrieved.queue_length == 5
        assert "/path/to/adapter_a" in retrieved.cached_adapters
    
    def test_update_invalid_worker(self):
        """测试更新无效 Worker（应忽略）"""
        cache = WorkerStateCache(num_workers=2)
        
        state = WorkerState(worker_id=99, queue_length=10)
        cache.update(99, state)  # Should not raise
        
        # Worker 99 should not exist
        assert cache.get(99) is None
    
    def test_update_from_message(self):
        """测试从消息更新状态"""
        cache = WorkerStateCache(num_workers=2)
        
        message = {
            'type': 'worker_state',
            'worker_id': 1,
            'cached_adapters': ['/adapter_a', '/adapter_b'],
            'queue_length': 3,
            'gpu_memory_free': 1024,
            'timestamp': time.time()
        }
        cache.update_from_message(message)
        
        state = cache.get(1)
        assert state.queue_length == 3
        assert state.cached_adapters == {'/adapter_a', '/adapter_b'}
        assert state.gpu_memory_free == 1024
    
    def test_update_from_invalid_message(self):
        """测试从无效消息更新（应忽略）"""
        cache = WorkerStateCache(num_workers=2)
        
        # Invalid type
        cache.update_from_message({'type': 'invalid'})
        
        # Missing worker_id
        cache.update_from_message({'type': 'worker_state'})
        
        # Invalid worker_id
        cache.update_from_message({'type': 'worker_state', 'worker_id': 99})
        
        # Should not raise any errors


class TestWorkerStateCacheGet:
    """Test suite for state retrieval"""
    
    def test_get_existing_worker(self):
        """测试获取存在的 Worker 状态"""
        cache = WorkerStateCache(num_workers=2)
        
        state = cache.get(0)
        assert state is not None
        assert state.worker_id == 0
    
    def test_get_nonexistent_worker(self):
        """测试获取不存在的 Worker 状态"""
        cache = WorkerStateCache(num_workers=2)
        
        state = cache.get(99)
        assert state is None
    
    def test_get_all(self):
        """测试获取所有状态"""
        cache = WorkerStateCache(num_workers=3)
        
        all_states = cache.get_all()
        assert len(all_states) == 3
        assert set(all_states.keys()) == {0, 1, 2}


class TestWorkerStateCacheHealth:
    """Test suite for health checking - Requirements 1.5"""
    
    def test_check_health_timeout(self):
        """测试心跳超时检测"""
        # 使用较短的超时时间便于测试
        cache = WorkerStateCache(
            num_workers=2,
            heartbeat_interval_ms=50,
            heartbeat_timeout_ms=100
        )
        
        # 等待超过超时时间
        time.sleep(0.15)
        
        # 检查健康状态
        unhealthy = cache.check_health()
        
        # 所有 Worker 应该被标记为不健康
        assert len(unhealthy) == 2
        assert cache.get_healthy_workers() == []
    
    def test_check_health_with_recent_heartbeat(self):
        """测试有最近心跳的 Worker 保持健康"""
        cache = WorkerStateCache(
            num_workers=2,
            heartbeat_interval_ms=50,
            heartbeat_timeout_ms=200
        )
        
        # 更新 Worker 0 的心跳
        state = WorkerState(worker_id=0, last_heartbeat=time.time())
        cache.update(0, state)
        
        # 等待一小段时间（但不超过超时）
        time.sleep(0.05)
        
        # Worker 0 应该仍然健康
        cache.check_health()
        healthy = cache.get_healthy_workers()
        assert 0 in healthy
    
    def test_mark_healthy(self):
        """测试手动标记为健康"""
        cache = WorkerStateCache(num_workers=2, heartbeat_timeout_ms=50)
        
        # 等待超时
        time.sleep(0.1)
        cache.check_health()
        
        # Worker 0 应该不健康
        assert 0 not in cache.get_healthy_workers()
        
        # 手动标记为健康
        cache.mark_healthy(0)
        assert 0 in cache.get_healthy_workers()
    
    def test_mark_unhealthy(self):
        """测试手动标记为不健康"""
        cache = WorkerStateCache(num_workers=2)
        
        # 初始应该健康
        assert 0 in cache.get_healthy_workers()
        
        # 手动标记为不健康
        cache.mark_unhealthy(0)
        assert 0 not in cache.get_healthy_workers()


class TestWorkerStateCacheAdapterLookup:
    """Test suite for adapter-to-worker lookup"""
    
    def test_get_workers_with_adapter(self):
        """测试获取缓存了指定 Adapter 的 Worker"""
        cache = WorkerStateCache(num_workers=3)
        adapter_dir = "/path/to/adapter_a"
        
        # Worker 0 和 2 缓存了 adapter
        cache.update(0, WorkerState(
            worker_id=0, cached_adapters={adapter_dir}
        ))
        cache.update(2, WorkerState(
            worker_id=2, cached_adapters={adapter_dir}
        ))
        
        workers = cache.get_workers_with_adapter(adapter_dir)
        assert workers == {0, 2}
    
    def test_get_workers_with_adapter_none(self):
        """测试没有 Worker 缓存指定 Adapter"""
        cache = WorkerStateCache(num_workers=2)
        
        workers = cache.get_workers_with_adapter("/nonexistent/adapter")
        assert workers == set()


class TestWorkerStateCacheStats:
    """Test suite for cache statistics"""
    
    def test_get_stats(self):
        """测试获取统计信息"""
        cache = WorkerStateCache(num_workers=3)
        
        # 更新一些状态
        cache.update(0, WorkerState(
            worker_id=0,
            cached_adapters={"/adapter_a", "/adapter_b"},
            queue_length=5
        ))
        cache.update(1, WorkerState(
            worker_id=1,
            cached_adapters={"/adapter_c"},
            queue_length=3
        ))
        cache.mark_unhealthy(2)
        
        stats = cache.get_stats()
        
        assert stats['num_workers'] == 3
        assert stats['healthy_workers'] == 2
        assert stats['unhealthy_workers'] == 1
        assert stats['total_cached_adapters'] == 3
        assert stats['total_queue_length'] == 8
    
    def test_reset(self):
        """测试重置缓存"""
        cache = WorkerStateCache(num_workers=2)
        
        # 更新状态
        cache.update(0, WorkerState(
            worker_id=0,
            cached_adapters={"/adapter_a"},
            queue_length=10
        ))
        cache.mark_unhealthy(1)
        
        # 重置
        cache.reset()
        
        # 所有状态应该恢复默认
        state = cache.get(0)
        assert state.queue_length == 0
        assert state.cached_adapters == set()
        assert cache.get_healthy_workers() == [0, 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
