"""
Unit tests for Worker State Data Structures

Tests the correctness of WorkerState, RoutingStats, and RoutingConfig
data classes used in the adapter-aware routing system.

Requirements: 1.1, 1.2, 6.1-6.5, 7.1-7.3
"""

import pytest
import time
from hypothesis import given, strategies as st, settings
from slora.server.router.worker_state import WorkerState, RoutingStats, RoutingConfig


# Property-based test strategies
adapter_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/_-'),
    min_size=1, max_size=50
).filter(lambda x: len(x.strip()) > 0)

worker_state_strategy = st.builds(
    WorkerState,
    worker_id=st.integers(min_value=0, max_value=1000),
    cached_adapters=st.frozensets(adapter_path_strategy, min_size=0, max_size=10).map(set),
    queue_length=st.integers(min_value=0, max_value=1000),
    gpu_memory_free=st.integers(min_value=0, max_value=10**12),
    last_heartbeat=st.floats(min_value=0.0, max_value=10**10, allow_nan=False, allow_infinity=False),
    is_healthy=st.booleans(),
    avg_rank=st.floats(min_value=0.0, max_value=256.0, allow_nan=False, allow_infinity=False),
    min_rank=st.integers(min_value=0, max_value=256),
    max_rank=st.integers(min_value=0, max_value=256)
)


class TestWorkerStatePropertyBased:
    """
    Property-based tests for WorkerState
    Feature: rank-aware-routing, Property 2: WorkerState Serialization Round-Trip
    **Validates: Requirements 2.4, 2.5**
    """
    
    @given(state=worker_state_strategy)
    @settings(max_examples=100)
    def test_serialization_round_trip(self, state: WorkerState):
        """
        Property 2: WorkerState Serialization Round-Trip
        
        For any valid WorkerState instance with rank distribution fields,
        serializing to dictionary via to_dict() and then deserializing via
        from_dict() SHALL produce an equivalent WorkerState with identical rank values.
        
        **Validates: Requirements 2.4, 2.5**
        """
        # Serialize to dict
        serialized = state.to_dict()
        
        # Deserialize back
        deserialized = WorkerState.from_dict(serialized)
        
        # Verify all fields match
        assert deserialized.worker_id == state.worker_id
        assert deserialized.cached_adapters == state.cached_adapters
        assert deserialized.queue_length == state.queue_length
        assert deserialized.gpu_memory_free == state.gpu_memory_free
        assert deserialized.is_healthy == state.is_healthy
        # Rank fields must match exactly
        assert deserialized.avg_rank == state.avg_rank
        assert deserialized.min_rank == state.min_rank
        assert deserialized.max_rank == state.max_rank


class TestRoutingStatsPropertyBased:
    """
    Property-based tests for RoutingStats tracking
    Feature: rank-aware-routing, Property 8: Routing Statistics Tracking
    **Validates: Requirements 7.1, 7.2**
    """
    
    @given(
        mismatches=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=100
        ),
        threshold=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_rank_mismatch_tracking(self, mismatches, threshold):
        """
        Property 8: Routing Statistics Tracking
        
        For any sequence of routing decisions with recorded rank mismatches,
        the RoutingStats SHALL correctly track the rank_matched_count
        (mismatches below threshold) and compute the correct avg_rank_mismatch
        as the sum of mismatches divided by total requests.
        
        **Validates: Requirements 7.1, 7.2**
        """
        stats = RoutingStats()
        stats.total_requests = len(mismatches)
        
        # Record all mismatches
        for mismatch in mismatches:
            stats.record_rank_mismatch(mismatch, threshold=threshold)
        
        # Verify total_rank_mismatch is sum of all mismatches
        expected_total = sum(mismatches)
        assert abs(stats.total_rank_mismatch - expected_total) < 1e-6
        
        # Verify rank_matched_count is count of mismatches <= threshold
        expected_matched = sum(1 for m in mismatches if m <= threshold)
        assert stats.rank_matched_count == expected_matched
        
        # Verify avg_rank_mismatch is correct
        expected_avg = expected_total / len(mismatches)
        assert abs(stats.avg_rank_mismatch - expected_avg) < 1e-6
    
    @given(
        num_requests=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100)
    def test_avg_rank_mismatch_bounds(self, num_requests):
        """
        Property 8: Routing Statistics Tracking - Bounds check
        
        For any number of requests with rank mismatches in [0, 1],
        the avg_rank_mismatch SHALL be in the range [0, 1].
        
        **Validates: Requirements 7.2**
        """
        stats = RoutingStats()
        stats.total_requests = num_requests
        
        # Record random mismatches in valid range
        import random
        for _ in range(num_requests):
            mismatch = random.uniform(0.0, 1.0)
            stats.record_rank_mismatch(mismatch)
        
        # avg_rank_mismatch must be in [0, 1]
        assert 0.0 <= stats.avg_rank_mismatch <= 1.0


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
        # Rank distribution fields should have default values
        assert state.avg_rank == 0.0
        assert state.min_rank == 0
        assert state.max_rank == 0
        assert state.active_rwpt_tokens == 0
    
    def test_initialization_with_values(self):
        """测试带参数初始化"""
        adapters = {"/path/to/adapter_a", "/path/to/adapter_b"}
        state = WorkerState(
            worker_id=1,
            cached_adapters=adapters,
            queue_length=5,
            gpu_memory_free=1024 * 1024 * 1024,
            is_healthy=True,
            avg_rank=24.5,
            min_rank=16,
            max_rank=32
        )
        
        assert state.worker_id == 1
        assert state.cached_adapters == adapters
        assert state.queue_length == 5
        assert state.gpu_memory_free == 1024 * 1024 * 1024
        assert state.avg_rank == 24.5
        assert state.min_rank == 16
        assert state.max_rank == 32
    
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
            is_healthy=True,
            avg_rank=16.0,
            min_rank=8,
            max_rank=32
        )
        
        d = state.to_dict()
        
        assert d['worker_id'] == 1
        assert d['cached_adapters'] == ["/path/to/adapter_a"]
        assert d['queue_length'] == 3
        assert d['gpu_memory_free'] == 1000
        assert d['is_healthy'] == True
        # Verify rank fields are serialized
        assert d['avg_rank'] == 16.0
        assert d['min_rank'] == 8
        assert d['max_rank'] == 32
    
    def test_from_dict(self):
        """测试从字典反序列化"""
        data = {
            'worker_id': 2,
            'cached_adapters': ["/path/to/adapter_a", "/path/to/adapter_b"],
            'queue_length': 7,
            'gpu_memory_free': 2000,
            'is_healthy': False,
            'avg_rank': 24.0,
            'min_rank': 16,
            'max_rank': 64
        }
        
        state = WorkerState.from_dict(data)
        
        assert state.worker_id == 2
        assert state.cached_adapters == {"/path/to/adapter_a", "/path/to/adapter_b"}
        assert state.queue_length == 7
        assert state.gpu_memory_free == 2000
        assert state.is_healthy == False
        # Verify rank fields are deserialized
        assert state.avg_rank == 24.0
        assert state.min_rank == 16
        assert state.max_rank == 64
    
    def test_from_dict_backward_compatibility(self):
        """测试从旧格式字典反序列化（向后兼容，无 rank 字段）"""
        # Old format without rank fields
        data = {
            'worker_id': 3,
            'cached_adapters': ["/path/to/adapter_a"],
            'queue_length': 5,
            'gpu_memory_free': 1500,
            'is_healthy': True
        }
        
        state = WorkerState.from_dict(data)
        
        assert state.worker_id == 3
        assert state.cached_adapters == {"/path/to/adapter_a"}
        assert state.queue_length == 5
        # Rank fields should use default values
        assert state.avg_rank == 0.0
        assert state.min_rank == 0
        assert state.max_rank == 0
        assert state.active_rwpt_tokens == 0

    def test_active_rwpt_tokens_round_trip(self):
        state = WorkerState(worker_id=0, active_rwpt_tokens=1234)

        restored = WorkerState.from_dict(state.to_dict())

        assert restored.active_rwpt_tokens == 1234


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
        # Rank-aware statistics defaults
        assert stats.rank_matched_count == 0
        assert stats.total_rank_mismatch == 0.0
    
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
    
    def test_record_rank_mismatch_below_threshold(self):
        """测试记录低于阈值的 rank 不匹配度"""
        stats = RoutingStats()
        stats.total_requests = 1  # Need at least one request for avg calculation
        
        stats.record_rank_mismatch(0.05, threshold=0.1)
        
        assert stats.total_rank_mismatch == 0.05
        assert stats.rank_matched_count == 1
    
    def test_record_rank_mismatch_above_threshold(self):
        """测试记录高于阈值的 rank 不匹配度"""
        stats = RoutingStats()
        stats.total_requests = 1
        
        stats.record_rank_mismatch(0.5, threshold=0.1)
        
        assert stats.total_rank_mismatch == 0.5
        assert stats.rank_matched_count == 0
    
    def test_avg_rank_mismatch_zero_requests(self):
        """测试零请求时的平均 rank 不匹配度"""
        stats = RoutingStats()
        
        assert stats.avg_rank_mismatch == 0.0
    
    def test_avg_rank_mismatch_calculation(self):
        """测试平均 rank 不匹配度计算"""
        stats = RoutingStats()
        stats.total_requests = 4
        stats.record_rank_mismatch(0.1)
        stats.record_rank_mismatch(0.2)
        stats.record_rank_mismatch(0.3)
        stats.record_rank_mismatch(0.4)
        
        # avg = (0.1 + 0.2 + 0.3 + 0.4) / 4 = 0.25
        assert stats.avg_rank_mismatch == 0.25
    
    def test_reset(self):
        """测试重置统计"""
        stats = RoutingStats()
        stats.record_request(0, True)
        stats.record_request(1, False, True)
        stats.record_hot_adapter_redistribution()
        stats.record_rank_mismatch(0.05)
        stats.record_rank_mismatch(0.5)
        
        stats.reset()
        
        assert stats.total_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.cold_starts == 0
        assert stats.hot_adapter_redistributions == 0
        assert stats.worker_request_counts == {}
        assert stats.rank_matched_count == 0
        assert stats.total_rank_mismatch == 0.0
    
    def test_to_dict(self):
        """测试序列化为字典"""
        stats = RoutingStats()
        stats.record_request(0, True)
        stats.record_request(1, False)
        stats.record_rank_mismatch(0.2)
        
        d = stats.to_dict()
        
        assert d['total_requests'] == 2
        assert d['cache_hits'] == 1
        assert d['cache_misses'] == 1
        assert d['cache_hit_rate'] == 0.5
        assert d['worker_request_counts'] == {0: 1, 1: 1}
        # Rank-aware statistics
        assert d['rank_matched_count'] == 0  # 0.2 > 0.1 threshold
        assert d['total_rank_mismatch'] == 0.2
        assert d['avg_rank_mismatch'] == 0.1  # 0.2 / 2 requests
    
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


class TestRoutingConfigPropertyBased:
    """
    Property-based tests for RoutingConfig validation
    Feature: rank-aware-routing, Property 7: Configuration Validation
    **Validates: Requirements 6.5**
    """
    
    @given(w3=st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_invalid_w3_raises_error(self, w3: float):
        """
        Property 7: Configuration Validation - Invalid w3
        
        For any RoutingConfig with w3 < 0, the configuration SHALL raise
        a ValueError during initialization.
        
        **Validates: Requirements 6.5**
        """
        with pytest.raises(ValueError):
            RoutingConfig(w3=w3)
    
    @given(default_lora_rank=st.integers(max_value=0))
    @settings(max_examples=100)
    def test_invalid_default_lora_rank_raises_error(self, default_lora_rank: int):
        """
        Property 7: Configuration Validation - Invalid default_lora_rank
        
        For any RoutingConfig with default_lora_rank <= 0, the configuration
        SHALL raise a ValueError during initialization.
        
        **Validates: Requirements 6.5**
        """
        with pytest.raises(ValueError):
            RoutingConfig(default_lora_rank=default_lora_rank)
    
    @given(max_rank_diff=st.integers(max_value=0))
    @settings(max_examples=100)
    def test_invalid_max_rank_diff_raises_error(self, max_rank_diff: int):
        """
        Property 7: Configuration Validation - Invalid max_rank_diff
        
        For any RoutingConfig with max_rank_diff <= 0, the configuration
        SHALL raise a ValueError during initialization.
        
        **Validates: Requirements 6.5**
        """
        with pytest.raises(ValueError):
            RoutingConfig(max_rank_diff=max_rank_diff)
    
    @given(
        w3=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        default_lora_rank=st.integers(min_value=1, max_value=256),
        max_rank_diff=st.integers(min_value=1, max_value=512)
    )
    @settings(max_examples=100)
    def test_valid_rank_params_accepted(self, w3: float, default_lora_rank: int, max_rank_diff: int):
        """
        Property 7: Configuration Validation - Valid parameters accepted
        
        For any RoutingConfig with valid rank-aware parameters (w3 >= 0,
        default_lora_rank > 0, max_rank_diff > 0), the configuration SHALL
        be created successfully without raising errors.
        
        **Validates: Requirements 6.5**
        """
        config = RoutingConfig(
            w3=w3,
            default_lora_rank=default_lora_rank,
            max_rank_diff=max_rank_diff
        )
        assert config.w3 == w3
        assert config.default_lora_rank == default_lora_rank
        assert config.max_rank_diff == max_rank_diff


class TestRoutingConfig:
    """Test suite for RoutingConfig data class - Requirements 6.1-6.5"""
    
    def test_initialization_defaults(self):
        """测试默认值初始化"""
        config = RoutingConfig()
        
        assert config.strategy == 'adapter-aware'
        assert config.w1 == 1.0
        assert config.w2 == 1.0
        assert config.w3 == 0.0
        assert config.default_lora_rank == 16
        assert config.max_rank_diff == 64
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
            w3=0.3,
            default_lora_rank=32,
            max_rank_diff=128,
            heartbeat_interval_ms=200,
            heartbeat_timeout_ms=600,
            max_queue_length=50,
            hot_adapter_threshold=20.0
        )
        
        assert config.strategy == 'round-robin'
        assert config.w1 == 2.0
        assert config.w2 == 0.5
        assert config.w3 == 0.3
        assert config.default_lora_rank == 32
        assert config.max_rank_diff == 128
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
    
    def test_invalid_w3(self):
        """测试无效的 w3"""
        with pytest.raises(ValueError):
            RoutingConfig(w3=-0.1)
    
    def test_invalid_default_lora_rank(self):
        """测试无效的 default_lora_rank"""
        with pytest.raises(ValueError):
            RoutingConfig(default_lora_rank=0)
        with pytest.raises(ValueError):
            RoutingConfig(default_lora_rank=-1)
    
    def test_invalid_max_rank_diff(self):
        """测试无效的 max_rank_diff"""
        with pytest.raises(ValueError):
            RoutingConfig(max_rank_diff=0)
        with pytest.raises(ValueError):
            RoutingConfig(max_rank_diff=-64)
    
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
        assert d['w2'] == 1.0
        assert d['w3'] == 0.0
        assert d['default_lora_rank'] == 16
        assert d['max_rank_diff'] == 64
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
            'w3': 0.5,
            'default_lora_rank': 32,
            'max_rank_diff': 128,
            'heartbeat_interval_ms': 150,
            'heartbeat_timeout_ms': 450,
            'max_queue_length': 75,
            'hot_adapter_threshold': 15.0
        }
        
        config = RoutingConfig.from_dict(data)
        
        assert config.strategy == 'round-robin'
        assert config.w1 == 1.5
        assert config.w2 == 0.2
        assert config.w3 == 0.5
        assert config.default_lora_rank == 32
        assert config.max_rank_diff == 128
        assert config.heartbeat_interval_ms == 150
        assert config.heartbeat_timeout_ms == 450
        assert config.max_queue_length == 75
        assert config.hot_adapter_threshold == 15.0
    
    def test_from_dict_backward_compatibility(self):
        """测试从旧格式字典反序列化（向后兼容，无 rank 相关字段）"""
        data = {
            'strategy': 'adapter-aware',
            'w1': 1.0,
            'w2': 0.1,
            'heartbeat_interval_ms': 100,
            'heartbeat_timeout_ms': 300,
            'max_queue_length': 100,
            'hot_adapter_threshold': 10.0
        }
        
        config = RoutingConfig.from_dict(data)
        
        # New fields should use default values
        assert config.w3 == 0.0
        assert config.default_lora_rank == 16
        assert config.max_rank_diff == 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
