"""
Property-based tests for backward compatible state deserialization

Tests that old format state messages (without rank fields) are correctly
handled by the system, ensuring backward compatibility.

Feature: rank-aware-routing, Property 9: Backward Compatible State Deserialization
**Validates: Requirements 8.2, 8.3**
"""

import pytest
import sys
import os
import time
from hypothesis import given, strategies as st, settings

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from slora.server.router.worker_state import WorkerState
from slora.server.router.worker_state_cache import WorkerStateCache


# Strategy for generating adapter paths
adapter_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/_-'),
    min_size=1, max_size=50
).filter(lambda x: len(x.strip()) > 0)


class TestBackwardCompatibleStateDeserialization:
    """
    Property-based tests for backward compatible state deserialization
    Feature: rank-aware-routing, Property 9: Backward Compatible State Deserialization
    **Validates: Requirements 8.2, 8.3**
    """
    
    @given(
        worker_id=st.integers(min_value=0, max_value=100),
        cached_adapters=st.lists(adapter_path_strategy, min_size=0, max_size=10),
        queue_length=st.integers(min_value=0, max_value=1000),
        gpu_memory_free=st.integers(min_value=0, max_value=10**12)
    )
    @settings(max_examples=100)
    def test_workerstate_from_dict_without_rank_fields(
        self, worker_id, cached_adapters, queue_length, gpu_memory_free
    ):
        """
        Property 9: Backward Compatible State Deserialization - WorkerState
        
        For any state dictionary without rank fields (from older Workers),
        WorkerState.from_dict() SHALL successfully deserialize and use
        default values (avg_rank=0.0, min_rank=0, max_rank=0).
        
        **Validates: Requirements 8.2, 8.3**
        """
        # Create old format state dict (without rank fields)
        old_format_dict = {
            'worker_id': worker_id,
            'cached_adapters': cached_adapters,
            'queue_length': queue_length,
            'gpu_memory_free': gpu_memory_free,
            'last_heartbeat': time.time(),
            'is_healthy': True
            # No rank fields - old format
        }
        
        # Deserialize
        state = WorkerState.from_dict(old_format_dict)
        
        # Verify basic fields
        assert state.worker_id == worker_id
        assert state.cached_adapters == set(cached_adapters)
        assert state.queue_length == queue_length
        assert state.gpu_memory_free == gpu_memory_free
        assert state.is_healthy == True
        
        # Verify rank fields use default values
        assert state.avg_rank == 0.0, \
            f"Expected avg_rank=0.0 for old format, got {state.avg_rank}"
        assert state.min_rank == 0, \
            f"Expected min_rank=0 for old format, got {state.min_rank}"
        assert state.max_rank == 0, \
            f"Expected max_rank=0 for old format, got {state.max_rank}"
    
    @given(
        worker_id=st.integers(min_value=0, max_value=3),
        cached_adapters=st.lists(adapter_path_strategy, min_size=0, max_size=10),
        queue_length=st.integers(min_value=0, max_value=1000),
        gpu_memory_free=st.integers(min_value=0, max_value=10**12)
    )
    @settings(max_examples=100)
    def test_cache_update_from_message_without_rank_fields(
        self, worker_id, cached_adapters, queue_length, gpu_memory_free
    ):
        """
        Property 9: Backward Compatible State Deserialization - WorkerStateCache
        
        For any state message without rank fields (from older Workers),
        WorkerStateCache.update_from_message() SHALL successfully process
        the message and use default rank values (avg_rank=0.0, min_rank=0, max_rank=0).
        
        **Validates: Requirements 8.2, 8.3**
        """
        # Create cache
        cache = WorkerStateCache(num_workers=4)
        
        # Create old format message (without rank fields)
        old_format_message = {
            'type': 'worker_state',
            'worker_id': worker_id,
            'cached_adapters': cached_adapters,
            'queue_length': queue_length,
            'gpu_memory_free': gpu_memory_free,
            'timestamp': time.time()
            # No rank fields - old format
        }
        
        # Update cache
        cache.update_from_message(old_format_message)
        
        # Get state
        state = cache.get(worker_id)
        
        # Verify basic fields
        assert state is not None
        assert state.worker_id == worker_id
        assert state.cached_adapters == set(cached_adapters)
        assert state.queue_length == queue_length
        assert state.gpu_memory_free == gpu_memory_free
        
        # Verify rank fields use default values
        assert state.avg_rank == 0.0, \
            f"Expected avg_rank=0.0 for old format, got {state.avg_rank}"
        assert state.min_rank == 0, \
            f"Expected min_rank=0 for old format, got {state.min_rank}"
        assert state.max_rank == 0, \
            f"Expected max_rank=0 for old format, got {state.max_rank}"
    
    @given(
        num_old_workers=st.integers(min_value=1, max_value=3),
        num_new_workers=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=100)
    def test_mixed_old_and_new_workers(self, num_old_workers, num_new_workers):
        """
        Property 9: Backward Compatible State Deserialization - Mixed deployment
        
        For any deployment with both old Workers (no rank fields) and new Workers
        (with rank fields), the system SHALL correctly handle both message formats
        simultaneously.
        
        **Validates: Requirements 8.3**
        """
        total_workers = num_old_workers + num_new_workers
        cache = WorkerStateCache(num_workers=total_workers)
        
        # Send messages from old Workers (no rank fields)
        for i in range(num_old_workers):
            old_message = {
                'type': 'worker_state',
                'worker_id': i,
                'cached_adapters': [f'lora_{i}'],
                'queue_length': i + 1,
                'gpu_memory_free': 8000000000,
                'timestamp': time.time()
                # No rank fields
            }
            cache.update_from_message(old_message)
        
        # Send messages from new Workers (with rank fields)
        for i in range(num_old_workers, total_workers):
            new_message = {
                'type': 'worker_state',
                'worker_id': i,
                'cached_adapters': [f'lora_{i}'],
                'queue_length': i + 1,
                'gpu_memory_free': 8000000000,
                'timestamp': time.time(),
                'avg_rank': float(16 + i),
                'min_rank': 8 + i,
                'max_rank': 32 + i
            }
            cache.update_from_message(new_message)
        
        # Verify old Workers have default rank values
        for i in range(num_old_workers):
            state = cache.get(i)
            assert state.avg_rank == 0.0, \
                f"Old worker {i} should have avg_rank=0.0, got {state.avg_rank}"
            assert state.min_rank == 0, \
                f"Old worker {i} should have min_rank=0, got {state.min_rank}"
            assert state.max_rank == 0, \
                f"Old worker {i} should have max_rank=0, got {state.max_rank}"
        
        # Verify new Workers have correct rank values
        for i in range(num_old_workers, total_workers):
            state = cache.get(i)
            assert state.avg_rank == float(16 + i), \
                f"New worker {i} should have avg_rank={16+i}, got {state.avg_rank}"
            assert state.min_rank == 8 + i, \
                f"New worker {i} should have min_rank={8+i}, got {state.min_rank}"
            assert state.max_rank == 32 + i, \
                f"New worker {i} should have max_rank={32+i}, got {state.max_rank}"
    
    def test_workerstate_serialization_roundtrip_old_format(self):
        """
        Property 9: Backward Compatible State Deserialization - Round-trip
        
        For any WorkerState created from old format dict (no rank fields),
        serializing and deserializing should preserve the default rank values.
        
        **Validates: Requirements 8.2**
        """
        # Create old format dict
        old_dict = {
            'worker_id': 5,
            'cached_adapters': ['lora1', 'lora2'],
            'queue_length': 10,
            'gpu_memory_free': 5000000000,
            'last_heartbeat': time.time(),
            'is_healthy': True
            # No rank fields
        }
        
        # Deserialize
        state1 = WorkerState.from_dict(old_dict)
        
        # Serialize
        serialized = state1.to_dict()
        
        # Deserialize again
        state2 = WorkerState.from_dict(serialized)
        
        # Verify rank fields are preserved (as defaults)
        assert state2.avg_rank == 0.0
        assert state2.min_rank == 0
        assert state2.max_rank == 0
        
        # Verify other fields
        assert state2.worker_id == state1.worker_id
        assert state2.cached_adapters == state1.cached_adapters
        assert state2.queue_length == state1.queue_length
    
    @given(
        num_messages=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_repeated_old_format_messages(self, num_messages):
        """
        Property 9: Backward Compatible State Deserialization - Repeated updates
        
        For any sequence of old format messages (no rank fields) sent to the same
        Worker, the system SHALL consistently use default rank values.
        
        **Validates: Requirements 8.2, 8.3**
        """
        cache = WorkerStateCache(num_workers=1)
        
        # Send multiple old format messages
        for i in range(num_messages):
            message = {
                'type': 'worker_state',
                'worker_id': 0,
                'cached_adapters': [f'lora_{i % 5}'],
                'queue_length': i,
                'gpu_memory_free': 8000000000 - i * 1000000,
                'timestamp': time.time()
                # No rank fields
            }
            cache.update_from_message(message)
            
            # Verify rank fields remain at default values
            state = cache.get(0)
            assert state.avg_rank == 0.0, \
                f"After {i+1} messages, avg_rank should be 0.0, got {state.avg_rank}"
            assert state.min_rank == 0, \
                f"After {i+1} messages, min_rank should be 0, got {state.min_rank}"
            assert state.max_rank == 0, \
                f"After {i+1} messages, max_rank should be 0, got {state.max_rank}"
    
    def test_transition_from_old_to_new_format(self):
        """
        Property 9: Backward Compatible State Deserialization - Format transition
        
        When a Worker transitions from old format (no rank fields) to new format
        (with rank fields), the system SHALL correctly update from default values
        to actual rank values.
        
        **Validates: Requirements 8.3**
        """
        cache = WorkerStateCache(num_workers=1)
        
        # Send old format message
        old_message = {
            'type': 'worker_state',
            'worker_id': 0,
            'cached_adapters': ['lora1'],
            'queue_length': 5,
            'gpu_memory_free': 8000000000,
            'timestamp': time.time()
            # No rank fields
        }
        cache.update_from_message(old_message)
        
        # Verify default rank values
        state1 = cache.get(0)
        assert state1.avg_rank == 0.0
        assert state1.min_rank == 0
        assert state1.max_rank == 0
        
        # Send new format message (Worker upgraded)
        new_message = {
            'type': 'worker_state',
            'worker_id': 0,
            'cached_adapters': ['lora1', 'lora2'],
            'queue_length': 3,
            'gpu_memory_free': 7500000000,
            'timestamp': time.time(),
            'avg_rank': 20.0,
            'min_rank': 16,
            'max_rank': 24
        }
        cache.update_from_message(new_message)
        
        # Verify rank values updated
        state2 = cache.get(0)
        assert state2.avg_rank == 20.0
        assert state2.min_rank == 16
        assert state2.max_rank == 24


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
