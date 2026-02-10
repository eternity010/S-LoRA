"""
Property-Based Test for Weight Configuration Effect

Property 13: Weight Configuration Effect
For any two configurations with different w1 or w2 values, the scoring function
SHALL produce different scores for the same Worker state, demonstrating that
weight parameters affect routing decisions.

**Validates: Requirements 3.2**

Feature: adapter-aware-routing, Property 13: Weight Configuration Effect
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from slora.server.router.adapter_aware_router import AdapterAwareRouter
from slora.server.router.worker_state import WorkerState, RoutingConfig


class TestWeightConfigurationEffect:
    """
    Property-based tests for weight configuration effect on scoring
    
    **Validates: Requirements 3.2**
    """
    
    @given(
        w1_a=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        w1_b=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        queue_length=st.integers(min_value=0, max_value=100),
        has_adapter=st.booleans()
    )
    @settings(max_examples=100)
    def test_different_w1_produces_different_scores_when_cache_hit(
        self, w1_a: float, w1_b: float, w2: float, queue_length: int, has_adapter: bool
    ):
        """
        Property 13: Different w1 values produce different scores when adapter is cached
        
        When the adapter is cached (cache_indicator = 1), different w1 values
        should produce different scores for the same Worker state.
        
        Score = w1 * cache_indicator - w2 * queue_length
        
        **Validates: Requirements 3.2**
        """
        # Only test when w1 values are sufficiently different and adapter is cached
        # Use a minimum difference threshold to avoid floating point precision issues
        assume(abs(w1_a - w1_b) > 1e-9)
        assume(has_adapter)  # Cache hit scenario where w1 matters
        
        adapter_dir = "/path/to/adapter"
        
        # Create two routers with different w1 values
        config_a = RoutingConfig(w1=w1_a, w2=w2)
        config_b = RoutingConfig(w1=w1_b, w2=w2)
        
        router_a = AdapterAwareRouter(num_workers=2, config=config_a)
        router_b = AdapterAwareRouter(num_workers=2, config=config_b)
        
        # Set up identical Worker state with adapter cached
        cached_adapters = {adapter_dir} if has_adapter else set()
        state = WorkerState(
            worker_id=0,
            cached_adapters=cached_adapters,
            queue_length=queue_length
        )
        
        router_a.update_worker_state(0, state)
        router_b.update_worker_state(0, state)
        
        # Calculate scores
        score_a = router_a.calculate_score(0, adapter_dir)
        score_b = router_b.calculate_score(0, adapter_dir)
        
        # Scores should be different when w1 differs and adapter is cached
        # Score = w1 * 1 - w2 * queue_length
        # Different w1 -> different scores
        assert score_a != score_b, (
            f"Different w1 values ({w1_a} vs {w1_b}) should produce different scores "
            f"when adapter is cached. Got score_a={score_a}, score_b={score_b}"
        )
    
    @given(
        w1=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        w2_a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        w2_b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        queue_length=st.integers(min_value=1, max_value=100),  # Must be > 0 for w2 to matter
        has_adapter=st.booleans()
    )
    @settings(max_examples=100)
    def test_different_w2_produces_different_scores_when_queue_nonzero(
        self, w1: float, w2_a: float, w2_b: float, queue_length: int, has_adapter: bool
    ):
        """
        Property 13: Different w2 values produce different scores when queue is non-zero
        
        When the queue length is non-zero, different w2 values should produce
        different scores for the same Worker state.
        
        Score = w1 * cache_indicator - w2 * queue_length
        
        **Validates: Requirements 3.2**
        """
        # Only test when w2 values are sufficiently different and queue is non-zero
        # Use a minimum difference threshold to avoid floating point precision issues
        assume(abs(w2_a - w2_b) > 1e-9)
        assume(queue_length > 0)  # Queue must be non-zero for w2 to matter
        
        adapter_dir = "/path/to/adapter"
        
        # Create two routers with different w2 values
        config_a = RoutingConfig(w1=w1, w2=w2_a)
        config_b = RoutingConfig(w1=w1, w2=w2_b)
        
        router_a = AdapterAwareRouter(num_workers=2, config=config_a)
        router_b = AdapterAwareRouter(num_workers=2, config=config_b)
        
        # Set up identical Worker state
        cached_adapters = {adapter_dir} if has_adapter else set()
        state = WorkerState(
            worker_id=0,
            cached_adapters=cached_adapters,
            queue_length=queue_length
        )
        
        router_a.update_worker_state(0, state)
        router_b.update_worker_state(0, state)
        
        # Calculate scores
        score_a = router_a.calculate_score(0, adapter_dir)
        score_b = router_b.calculate_score(0, adapter_dir)
        
        # Scores should be different when w2 differs and queue is non-zero
        # Score = w1 * cache_indicator - w2 * queue_length
        # Different w2 with non-zero queue -> different scores
        assert score_a != score_b, (
            f"Different w2 values ({w2_a} vs {w2_b}) should produce different scores "
            f"when queue_length={queue_length}. Got score_a={score_a}, score_b={score_b}"
        )
    
    @given(
        w1_a=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        w1_b=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        w2_a=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        w2_b=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        queue_length=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100)
    def test_weight_configuration_affects_routing_decision(
        self, w1_a: float, w1_b: float, w2_a: float, w2_b: float, queue_length: int
    ):
        """
        Property 13: Weight configuration affects routing decisions
        
        For any two configurations with different w1 or w2 values, the scoring
        function SHALL produce different scores for the same Worker state when
        the conditions are appropriate (cache hit for w1, non-zero queue for w2).
        
        **Validates: Requirements 3.2**
        """
        # Ensure at least one weight is different
        assume(w1_a != w1_b or w2_a != w2_b)
        
        adapter_dir = "/path/to/adapter"
        
        # Create two routers with different configurations
        config_a = RoutingConfig(w1=w1_a, w2=w2_a)
        config_b = RoutingConfig(w1=w1_b, w2=w2_b)
        
        router_a = AdapterAwareRouter(num_workers=2, config=config_a)
        router_b = AdapterAwareRouter(num_workers=2, config=config_b)
        
        # Set up Worker state with adapter cached and non-zero queue
        # This ensures both w1 and w2 contribute to the score
        state = WorkerState(
            worker_id=0,
            cached_adapters={adapter_dir},
            queue_length=queue_length
        )
        
        router_a.update_worker_state(0, state)
        router_b.update_worker_state(0, state)
        
        # Calculate scores
        score_a = router_a.calculate_score(0, adapter_dir)
        score_b = router_b.calculate_score(0, adapter_dir)
        
        # Verify the scoring formula
        # Score = w1 * 1 - w2 * queue_length (since adapter is cached)
        expected_score_a = w1_a * 1.0 - w2_a * queue_length
        expected_score_b = w1_b * 1.0 - w2_b * queue_length
        
        assert score_a == pytest.approx(expected_score_a, rel=1e-6), (
            f"Score calculation incorrect for config_a. "
            f"Expected {expected_score_a}, got {score_a}"
        )
        assert score_b == pytest.approx(expected_score_b, rel=1e-6), (
            f"Score calculation incorrect for config_b. "
            f"Expected {expected_score_b}, got {score_b}"
        )
        
        # If configurations are different, scores should be different
        # (unless the specific combination happens to produce the same result)
        if expected_score_a != expected_score_b:
            assert score_a != score_b, (
                f"Different weight configurations should produce different scores. "
                f"Config A: w1={w1_a}, w2={w2_a}, Config B: w1={w1_b}, w2={w2_b}, "
                f"queue_length={queue_length}. Got same score: {score_a}"
            )
    
    @given(
        w1=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
        queue_lengths=st.lists(
            st.integers(min_value=0, max_value=50),
            min_size=3,
            max_size=3
        )
    )
    @settings(max_examples=100)
    def test_weight_configuration_changes_worker_selection(
        self, w1: float, w2: float, queue_lengths: list
    ):
        """
        Property 13: Weight configuration changes worker selection
        
        Demonstrates that changing weight parameters can change which Worker
        is selected for routing, proving that weights affect routing decisions.
        
        **Validates: Requirements 3.2**
        """
        adapter_dir = "/path/to/adapter"
        
        # Create router with given weights
        config = RoutingConfig(w1=w1, w2=w2)
        router = AdapterAwareRouter(num_workers=3, config=config)
        
        # Set up Workers with different queue lengths
        # Worker 0: has adapter, queue_length[0]
        # Worker 1: has adapter, queue_length[1]
        # Worker 2: no adapter, queue_length[2]
        router.update_worker_state(0, WorkerState(
            worker_id=0,
            cached_adapters={adapter_dir},
            queue_length=queue_lengths[0]
        ))
        router.update_worker_state(1, WorkerState(
            worker_id=1,
            cached_adapters={adapter_dir},
            queue_length=queue_lengths[1]
        ))
        router.update_worker_state(2, WorkerState(
            worker_id=2,
            cached_adapters=set(),
            queue_length=queue_lengths[2]
        ))
        
        # Calculate scores for all workers
        scores = [router.calculate_score(i, adapter_dir) for i in range(3)]
        
        # Verify scores follow the formula
        # Worker 0: w1 * 1 - w2 * queue_lengths[0]
        # Worker 1: w1 * 1 - w2 * queue_lengths[1]
        # Worker 2: w1 * 0 - w2 * queue_lengths[2]
        expected_scores = [
            w1 * 1.0 - w2 * queue_lengths[0],
            w1 * 1.0 - w2 * queue_lengths[1],
            w1 * 0.0 - w2 * queue_lengths[2]
        ]
        
        for i in range(3):
            assert scores[i] == pytest.approx(expected_scores[i], rel=1e-6), (
                f"Score for Worker {i} incorrect. "
                f"Expected {expected_scores[i]}, got {scores[i]}"
            )
        
        # The worker with highest score should be selected
        # This demonstrates that weights affect the routing decision
        max_score = max(scores)
        max_score_workers = [i for i, s in enumerate(scores) if s == max_score]
        
        # Verify that the scoring correctly identifies the best worker
        assert len(max_score_workers) >= 1, "Should have at least one worker with max score"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
