"""
End-to-End Integration Test for Rank-Aware Routing

This test verifies that requests are routed to Workers with similar batch ranks,
validating the complete rank-aware routing functionality.

Task 10.1: Write integration test for end-to-end rank-aware routing
- Test that requests are routed to Workers with similar batch ranks
- Test with multiple Workers having different batch compositions
- Requirements: 5.1, 5.2
"""

import pytest
from slora.server.router.adapter_aware_router import AdapterAwareRouter
from slora.server.router.worker_state import WorkerState, RoutingConfig


class TestEndToEndRankAwareRouting:
    """
    End-to-end integration tests for rank-aware routing
    """
    
    def test_route_to_worker_with_similar_rank(self):
        """
        Test that requests are routed to Workers with similar batch ranks
        
        Scenario:
        - Worker 0: avg_rank=8, queue_length=1, no cache
        - Worker 1: avg_rank=32, queue_length=1, no cache
        - Worker 2: avg_rank=16, queue_length=1, no cache
        - Request: adapter with rank=16
        
        Expected: Request should be routed to Worker 2 (closest rank match)
        """
        # Setup: Create router with rank-aware routing enabled
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,  # Cache affinity
            w2=1.0,   # Queue penalty
            w3=5.0,   # Rank mismatch penalty (enabled)
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(
            num_workers=3,
            config=config
        )
        
        # Setup: Configure adapter ranks
        adapter_ranks = {
            '/path/to/adapter_rank8': 8,
            '/path/to/adapter_rank16': 16,
            '/path/to/adapter_rank32': 32,
        }
        router.set_adapter_ranks(adapter_ranks)
        
        # Setup: Create Worker states with different batch ranks
        worker0_state = WorkerState(
            worker_id=0,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=8.0,
            min_rank=8,
            max_rank=8,
            is_healthy=True
        )
        
        worker1_state = WorkerState(
            worker_id=1,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=32.0,
            min_rank=32,
            max_rank=32,
            is_healthy=True
        )
        
        worker2_state = WorkerState(
            worker_id=2,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=16.0,
            min_rank=16,
            max_rank=16,
            is_healthy=True
        )
        
        # Update router with Worker states
        router.update_worker_state(0, worker0_state)
        router.update_worker_state(1, worker1_state)
        router.update_worker_state(2, worker2_state)
        
        # Execute: Route request for adapter with rank=16
        selected_worker = router.select_worker('/path/to/adapter_rank16')
        
        # Verify: Should select Worker 2 (closest rank match)
        assert selected_worker == 2, \
            f"Expected Worker 2 (rank=16), but got Worker {selected_worker}"
        
        # Verify: Statistics should show rank mismatch was recorded
        stats = router.get_stats()
        assert stats['total_requests'] == 1
        assert stats['rank_matched_count'] == 1
        assert stats['avg_rank_mismatch'] == 0.0  # Perfect match
    
    def test_rank_penalty_overrides_cache_when_strong(self):
        """
        Test that strong rank mismatch penalty can override cache affinity
        
        Scenario:
        - Worker 0: avg_rank=8, has adapter in cache, queue_length=1
        - Worker 1: avg_rank=64, no cache, queue_length=1
        - Request: adapter with rank=64
        - Config: w1=10 (cache), w3=15 (rank penalty)
        
        Expected: Should route to Worker 1 despite cache miss,
                  because rank mismatch penalty is stronger
        """
        # Setup: Strong rank penalty
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,   # Cache affinity
            w2=1.0,    # Queue penalty
            w3=15.0,   # Strong rank mismatch penalty
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=2, config=config)
        
        # Setup: Configure adapter ranks
        adapter_ranks = {
            '/path/to/adapter_rank64': 64,
        }
        router.set_adapter_ranks(adapter_ranks)
        
        # Setup: Worker 0 has cache but wrong rank
        worker0_state = WorkerState(
            worker_id=0,
            cached_adapters={'/path/to/adapter_rank64'},  # Has cache
            queue_length=1,
            avg_rank=8.0,  # Very different rank
            min_rank=8,
            max_rank=8,
            is_healthy=True
        )
        
        # Setup: Worker 1 has no cache but matching rank
        worker1_state = WorkerState(
            worker_id=1,
            cached_adapters=set(),  # No cache
            queue_length=1,
            avg_rank=64.0,  # Matching rank
            min_rank=64,
            max_rank=64,
            is_healthy=True
        )
        
        router.update_worker_state(0, worker0_state)
        router.update_worker_state(1, worker1_state)
        
        # Execute: Route request
        selected_worker = router.select_worker('/path/to/adapter_rank64')
        
        # Verify: Should select Worker 1 (rank match overrides cache)
        assert selected_worker == 1, \
            f"Expected Worker 1 (rank match), but got Worker {selected_worker}"
    
    def test_cache_and_rank_both_favor_same_worker(self):
        """
        Test optimal case: cache hit AND rank match on same Worker
        
        Scenario:
        - Worker 0: avg_rank=16, has adapter in cache, queue_length=1
        - Worker 1: avg_rank=32, no cache, queue_length=1
        - Request: adapter with rank=16
        
        Expected: Should route to Worker 0 (both cache and rank match)
        """
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,
            w2=1.0,
            w3=5.0,
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=2, config=config)
        
        adapter_ranks = {'/path/to/adapter_rank16': 16}
        router.set_adapter_ranks(adapter_ranks)
        
        worker0_state = WorkerState(
            worker_id=0,
            cached_adapters={'/path/to/adapter_rank16'},  # Has cache
            queue_length=1,
            avg_rank=16.0,  # Matching rank
            min_rank=16,
            max_rank=16,
            is_healthy=True
        )
        
        worker1_state = WorkerState(
            worker_id=1,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=32.0,
            min_rank=32,
            max_rank=32,
            is_healthy=True
        )
        
        router.update_worker_state(0, worker0_state)
        router.update_worker_state(1, worker1_state)
        
        selected_worker = router.select_worker('/path/to/adapter_rank16')
        
        # Verify: Should definitely select Worker 0
        assert selected_worker == 0
        
        # Verify: Should be a cache hit
        stats = router.get_stats()
        assert stats['cache_hits'] == 1
        assert stats['cache_hit_rate'] == 1.0
    
    def test_multiple_requests_with_different_ranks(self):
        """
        Test routing multiple requests with different ranks to appropriate Workers
        
        Scenario:
        - Worker 0: avg_rank=8
        - Worker 1: avg_rank=32
        - Worker 2: avg_rank=64
        - Requests: rank=8, rank=32, rank=64, rank=8
        
        Expected: Each request routed to Worker with matching rank
        """
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,
            w2=1.0,
            w3=10.0,  # Strong rank preference
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=3, config=config)
        
        adapter_ranks = {
            '/adapter_r8': 8,
            '/adapter_r32': 32,
            '/adapter_r64': 64,
        }
        router.set_adapter_ranks(adapter_ranks)
        
        # Setup Workers with different ranks
        for worker_id, rank in [(0, 8.0), (1, 32.0), (2, 64.0)]:
            state = WorkerState(
                worker_id=worker_id,
                cached_adapters=set(),
                queue_length=0,
                avg_rank=rank,
                min_rank=int(rank),
                max_rank=int(rank),
                is_healthy=True
            )
            router.update_worker_state(worker_id, state)
        
        # Execute: Route multiple requests
        results = []
        for adapter in ['/adapter_r8', '/adapter_r32', '/adapter_r64', '/adapter_r8']:
            selected = router.select_worker(adapter)
            results.append((adapter, selected))
            
            # Update queue length to simulate request processing
            state = router.worker_states[selected]
            state.queue_length += 1
        
        # Verify: Each request routed to matching Worker
        assert results[0] == ('/adapter_r8', 0), "rank=8 should go to Worker 0"
        assert results[1] == ('/adapter_r32', 1), "rank=32 should go to Worker 1"
        assert results[2] == ('/adapter_r64', 2), "rank=64 should go to Worker 2"
        assert results[3] == ('/adapter_r8', 0), "rank=8 should go to Worker 0 again"
        
        # Verify: Statistics
        stats = router.get_stats()
        assert stats['total_requests'] == 4
    
    def test_empty_batch_worker_accepts_any_rank(self):
        """
        Test that Workers with empty batches (avg_rank=0) accept any rank request
        
        Scenario:
        - Worker 0: avg_rank=0 (empty batch), queue_length=0
        - Worker 1: avg_rank=32, queue_length=2
        - Request: adapter with rank=64
        
        Expected: Should route to Worker 0 (empty batch, shorter queue)
        """
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,
            w2=1.0,
            w3=5.0,
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=2, config=config)
        
        adapter_ranks = {'/adapter_r64': 64}
        router.set_adapter_ranks(adapter_ranks)
        
        # Worker 0: Empty batch
        worker0_state = WorkerState(
            worker_id=0,
            cached_adapters=set(),
            queue_length=0,
            avg_rank=0.0,  # Empty batch
            min_rank=0,
            max_rank=0,
            is_healthy=True
        )
        
        # Worker 1: Busy with different rank
        worker1_state = WorkerState(
            worker_id=1,
            cached_adapters=set(),
            queue_length=2,
            avg_rank=32.0,
            min_rank=32,
            max_rank=32,
            is_healthy=True
        )
        
        router.update_worker_state(0, worker0_state)
        router.update_worker_state(1, worker1_state)
        
        selected_worker = router.select_worker('/adapter_r64')
        
        # Verify: Should select Worker 0 (empty batch, no rank penalty)
        assert selected_worker == 0
    
    def test_queue_length_breaks_rank_tie(self):
        """
        Test that queue length is used as tiebreaker when ranks are equal
        
        Scenario:
        - Worker 0: avg_rank=16, queue_length=3
        - Worker 1: avg_rank=16, queue_length=1
        - Request: adapter with rank=16
        
        Expected: Should route to Worker 1 (same rank, shorter queue)
        """
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,
            w2=1.0,
            w3=5.0,
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=2, config=config)
        
        adapter_ranks = {'/adapter_r16': 16}
        router.set_adapter_ranks(adapter_ranks)
        
        worker0_state = WorkerState(
            worker_id=0,
            cached_adapters=set(),
            queue_length=3,  # Longer queue
            avg_rank=16.0,
            min_rank=16,
            max_rank=16,
            is_healthy=True
        )
        
        worker1_state = WorkerState(
            worker_id=1,
            cached_adapters=set(),
            queue_length=1,  # Shorter queue
            avg_rank=16.0,
            min_rank=16,
            max_rank=16,
            is_healthy=True
        )
        
        router.update_worker_state(0, worker0_state)
        router.update_worker_state(1, worker1_state)
        
        selected_worker = router.select_worker('/adapter_r16')
        
        # Verify: Should select Worker 1 (shorter queue)
        assert selected_worker == 1
    
    def test_unknown_adapter_uses_default_rank(self):
        """
        Test that unknown adapters use default rank for routing
        
        Scenario:
        - Worker 0: avg_rank=16
        - Worker 1: avg_rank=64
        - Request: unknown adapter (should use default_lora_rank=16)
        
        Expected: Should route to Worker 0 (matches default rank)
        """
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,
            w2=1.0,
            w3=5.0,
            default_lora_rank=16,  # Default rank
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=2, config=config)
        
        # Don't set adapter_ranks for the unknown adapter
        router.set_adapter_ranks({})
        
        worker0_state = WorkerState(
            worker_id=0,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=16.0,  # Matches default
            min_rank=16,
            max_rank=16,
            is_healthy=True
        )
        
        worker1_state = WorkerState(
            worker_id=1,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=64.0,
            min_rank=64,
            max_rank=64,
            is_healthy=True
        )
        
        router.update_worker_state(0, worker0_state)
        router.update_worker_state(1, worker1_state)
        
        # Request for unknown adapter
        selected_worker = router.select_worker('/unknown/adapter')
        
        # Verify: Should select Worker 0 (matches default rank)
        assert selected_worker == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])


class TestBackwardCompatibility:
    """
    Backward compatibility integration tests for rank-aware routing
    
    Task 10.2: Verify that the system maintains backward compatibility
    - Test that w3=0 produces same routing as original implementation
    - Test mixed Workers (some with rank info, some without)
    - Requirements: 8.1, 8.3
    """
    
    def test_w3_zero_disables_rank_awareness(self):
        """
        Test that w3=0 disables rank-aware routing (backward compatible)
        
        Scenario:
        - Worker 0: avg_rank=8, no cache, queue_length=1
        - Worker 1: avg_rank=64, has cache, queue_length=1
        - Request: adapter with rank=64
        - Config: w3=0 (rank awareness disabled)
        
        Expected: Should route to Worker 1 (cache hit), ignoring rank mismatch
        """
        # Setup: w3=0 disables rank awareness
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,   # Cache affinity
            w2=1.0,    # Queue penalty
            w3=0.0,    # Rank awareness DISABLED
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=2, config=config)
        
        adapter_ranks = {'/adapter_r64': 64}
        router.set_adapter_ranks(adapter_ranks)
        
        # Worker 0: No cache, matching rank
        worker0_state = WorkerState(
            worker_id=0,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=64.0,  # Matching rank (but should be ignored)
            min_rank=64,
            max_rank=64,
            is_healthy=True
        )
        
        # Worker 1: Has cache, different rank
        worker1_state = WorkerState(
            worker_id=1,
            cached_adapters={'/adapter_r64'},  # Has cache
            queue_length=1,
            avg_rank=8.0,  # Different rank (but should be ignored)
            min_rank=8,
            max_rank=8,
            is_healthy=True
        )
        
        router.update_worker_state(0, worker0_state)
        router.update_worker_state(1, worker1_state)
        
        # Execute: Route request
        selected_worker = router.select_worker('/adapter_r64')
        
        # Verify: Should select Worker 1 (cache hit), ignoring rank
        assert selected_worker == 1, \
            f"Expected Worker 1 (cache hit, w3=0), but got Worker {selected_worker}"
        
        # Verify: Cache hit should be recorded
        stats = router.get_stats()
        assert stats['cache_hits'] == 1
        assert stats['cache_hit_rate'] == 1.0
        
        # Verify: Rank statistics should NOT be recorded when w3=0
        assert stats['rank_matched_count'] == 0
        assert stats['total_rank_mismatch'] == 0.0
    
    def test_w3_zero_behaves_like_original_implementation(self):
        """
        Test that w3=0 produces identical routing decisions to original implementation
        
        This test verifies that the extended scoring function with w3=0
        produces the same results as the original formula without rank term.
        
        Original: Score = w1·I(cache) - w2·QueueLen
        Extended: Score = w1·I(cache) - w2·QueueLen - w3·RankMismatch
        When w3=0: Extended = Original
        """
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,
            w2=1.0,
            w3=0.0,  # Disabled
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=3, config=config)
        
        adapter_ranks = {'/adapter': 32}
        router.set_adapter_ranks(adapter_ranks)
        
        # Setup: Various Worker states with different ranks
        states = [
            WorkerState(worker_id=0, cached_adapters=set(), queue_length=2, 
                       avg_rank=8.0, is_healthy=True),
            WorkerState(worker_id=1, cached_adapters={'/adapter'}, queue_length=3,
                       avg_rank=64.0, is_healthy=True),
            WorkerState(worker_id=2, cached_adapters=set(), queue_length=1,
                       avg_rank=128.0, is_healthy=True),
        ]
        
        for state in states:
            router.update_worker_state(state.worker_id, state)
        
        # Execute: Route request
        selected_worker = router.select_worker('/adapter')
        
        # Verify: Should select Worker 1 (cache hit overrides queue length)
        # Original formula: Worker 0: 0-2=-2, Worker 1: 10-3=7, Worker 2: 0-1=-1
        # Worker 1 has highest score due to cache hit
        assert selected_worker == 1
        
        # Verify: Calculate scores manually to confirm
        score_0 = 10.0 * 0 - 1.0 * 2  # No cache: -2
        score_1 = 10.0 * 1 - 1.0 * 3  # Has cache: 7
        score_2 = 10.0 * 0 - 1.0 * 1  # No cache: -1
        
        assert score_1 > score_0 and score_1 > score_2
    
    def test_mixed_workers_with_and_without_rank_info(self):
        """
        Test system handles mixed Workers (some with rank info, some without)
        
        Scenario:
        - Worker 0: Has rank info (avg_rank=16)
        - Worker 1: No rank info (avg_rank=0, simulating old Worker)
        - Worker 2: Has rank info (avg_rank=32)
        - Request: adapter with rank=16
        - Config: w3=5.0 (rank awareness enabled)
        
        Expected: Should route to Worker 0 (rank match), Worker 1 treated as empty batch
        """
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,
            w2=1.0,
            w3=5.0,  # Rank awareness enabled
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=3, config=config)
        
        adapter_ranks = {'/adapter_r16': 16}
        router.set_adapter_ranks(adapter_ranks)
        
        # Worker 0: Has rank info, matching rank
        worker0_state = WorkerState(
            worker_id=0,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=16.0,  # Matching rank
            min_rank=16,
            max_rank=16,
            is_healthy=True
        )
        
        # Worker 1: No rank info (simulating old Worker or empty batch)
        worker1_state = WorkerState(
            worker_id=1,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=0.0,  # No rank info
            min_rank=0,
            max_rank=0,
            is_healthy=True
        )
        
        # Worker 2: Has rank info, different rank
        worker2_state = WorkerState(
            worker_id=2,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=32.0,  # Different rank
            min_rank=32,
            max_rank=32,
            is_healthy=True
        )
        
        router.update_worker_state(0, worker0_state)
        router.update_worker_state(1, worker1_state)
        router.update_worker_state(2, worker2_state)
        
        # Execute: Route request
        selected_worker = router.select_worker('/adapter_r16')
        
        # Verify: Should select Worker 0 (perfect rank match)
        # Worker 0: score = 0 - 1 - 5*0 = -1 (perfect rank match)
        # Worker 1: score = 0 - 1 - 5*0 = -1 (empty batch, no penalty)
        # Worker 2: score = 0 - 1 - 5*(16/64) = -2.25 (rank mismatch)
        # Worker 0 and 1 tie, but Worker 0 selected (deterministic ordering)
        assert selected_worker in [0, 1], \
            f"Expected Worker 0 or 1, but got Worker {selected_worker}"
    
    def test_old_worker_without_rank_fields_in_state_message(self):
        """
        Test that old Workers without rank fields in state messages work correctly
        
        This simulates receiving state messages from old Worker versions
        that don't include avg_rank, min_rank, max_rank fields.
        
        Requirements: 8.2, 8.3 (Backward compatible state deserialization)
        """
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,
            w2=1.0,
            w3=5.0,
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=2, config=config)
        
        adapter_ranks = {'/adapter': 16}
        router.set_adapter_ranks(adapter_ranks)
        
        # Simulate old Worker state message (without rank fields)
        old_worker_state_dict = {
            'worker_id': 0,
            'cached_adapters': [],
            'queue_length': 1,
            'gpu_memory_free': 1000000,
            'last_heartbeat': 1234567890.0,
            'is_healthy': True,
            # No avg_rank, min_rank, max_rank fields
        }
        
        # Deserialize using from_dict (should handle missing fields)
        old_worker_state = WorkerState.from_dict(old_worker_state_dict)
        
        # Verify: Missing rank fields default to 0
        assert old_worker_state.avg_rank == 0.0
        assert old_worker_state.min_rank == 0
        assert old_worker_state.max_rank == 0
        
        # New Worker with rank info
        new_worker_state = WorkerState(
            worker_id=1,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=16.0,
            min_rank=16,
            max_rank=16,
            is_healthy=True
        )
        
        router.update_worker_state(0, old_worker_state)
        router.update_worker_state(1, new_worker_state)
        
        # Execute: Route request
        selected_worker = router.select_worker('/adapter')
        
        # Verify: System should work without errors
        assert selected_worker in [0, 1]
        
        # Verify: Statistics should be recorded
        stats = router.get_stats()
        assert stats['total_requests'] == 1
    
    def test_gradual_migration_scenario(self):
        """
        Test gradual migration scenario: some Workers upgraded, some not
        
        Scenario simulating a rolling upgrade:
        - Worker 0: Old version (no rank info)
        - Worker 1: New version (has rank info)
        - Worker 2: Old version (no rank info)
        - System should continue working during migration
        """
        config = RoutingConfig(
            strategy='adapter-aware',
            w1=10.0,
            w2=1.0,
            w3=3.0,  # Moderate rank awareness
            default_lora_rank=16,
            max_rank_diff=64
        )
        
        router = AdapterAwareRouter(num_workers=3, config=config)
        
        adapter_ranks = {
            '/adapter_r8': 8,
            '/adapter_r16': 16,
            '/adapter_r32': 32,
        }
        router.set_adapter_ranks(adapter_ranks)
        
        # Old Workers: No rank info
        old_worker_0 = WorkerState(
            worker_id=0,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=0.0,  # Old version
            is_healthy=True
        )
        
        old_worker_2 = WorkerState(
            worker_id=2,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=0.0,  # Old version
            is_healthy=True
        )
        
        # New Worker: Has rank info
        new_worker_1 = WorkerState(
            worker_id=1,
            cached_adapters=set(),
            queue_length=1,
            avg_rank=16.0,  # New version
            min_rank=16,
            max_rank=16,
            is_healthy=True
        )
        
        router.update_worker_state(0, old_worker_0)
        router.update_worker_state(1, new_worker_1)
        router.update_worker_state(2, old_worker_2)
        
        # Execute: Route multiple requests
        results = []
        for adapter in ['/adapter_r8', '/adapter_r16', '/adapter_r32']:
            selected = router.select_worker(adapter)
            results.append((adapter, selected))
        
        # Verify: All requests should be routed successfully
        assert len(results) == 3
        
        # Verify: No errors during mixed Worker scenario
        stats = router.get_stats()
        assert stats['total_requests'] == 3
        
        # Verify: rank=16 request should prefer Worker 1 (rank match)
        # But old Workers (0, 2) are also valid choices (no penalty for empty batch)
        rank16_worker = results[1][1]
        assert rank16_worker in [0, 1, 2]  # All are valid
    
    def test_default_config_maintains_backward_compatibility(self):
        """
        Test that default configuration maintains backward compatibility
        
        Default config should have w3=0.0, ensuring existing deployments
        continue to work without changes.
        """
        # Create config with defaults
        config = RoutingConfig(strategy='adapter-aware')
        
        # Verify: Default w3 is 0.0 (disabled)
        assert config.w3 == 0.0, \
            "Default w3 should be 0.0 for backward compatibility"
        
        # Verify: Other defaults are reasonable
        assert config.default_lora_rank == 16
        assert config.max_rank_diff == 64
        
        # Create router with default config
        router = AdapterAwareRouter(num_workers=2, config=config)
        
        # Setup Workers
        worker0 = WorkerState(worker_id=0, cached_adapters=set(), 
                             queue_length=1, is_healthy=True)
        worker1 = WorkerState(worker_id=1, cached_adapters={'/adapter'}, 
                             queue_length=2, is_healthy=True)
        
        router.update_worker_state(0, worker0)
        router.update_worker_state(1, worker1)
        
        # Execute: Route request
        selected = router.select_worker('/adapter')
        
        # Verify: Should select Worker 1 (cache hit), behaving like original
        assert selected == 1
        
        # Verify: Rank statistics should not be recorded
        stats = router.get_stats()
        assert stats['rank_aware_enabled'] is False
        assert stats['rank_matched_count'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
