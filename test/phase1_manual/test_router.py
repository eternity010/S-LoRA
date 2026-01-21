#!/usr/bin/env python3
"""
Test 8.2.1: Round Robin Router Test

Tests the Round Robin Router component in isolation.
This is the simplest test - no processes, no network communication.

Test Coverage:
- Router instance creation
- select_worker() method functionality
- Round robin order verification (0, 1, 2, 0, 1, 2...)
- Different worker counts (1, 2, 3, 4 workers)
- Statistics tracking
- Reset functionality
"""

import sys
from pathlib import Path

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))

from test_base import ManualTest, run_test_suite
from slora.server.router.round_robin_router import RoundRobinRouter


class TestRouterCreation(ManualTest):
    """Test 1: Router instance creation"""
    
    def execute(self):
        self.logger.info("Testing router creation with different worker counts")
        
        # Test with 1 worker
        router1 = RoundRobinRouter(num_workers=1)
        self.assert_equal(router1.num_workers, 1, "Router should have 1 worker")
        self.logger.info("✓ Created router with 1 worker")
        
        # Test with 3 workers
        router3 = RoundRobinRouter(num_workers=3)
        self.assert_equal(router3.num_workers, 3, "Router should have 3 workers")
        self.logger.info("✓ Created router with 3 workers")
        
        # Test with 10 workers
        router10 = RoundRobinRouter(num_workers=10)
        self.assert_equal(router10.num_workers, 10, "Router should have 10 workers")
        self.logger.info("✓ Created router with 10 workers")
        
        # Store results
        self.result.details['router_counts'] = [1, 3, 10]


class TestSelectWorker(ManualTest):
    """Test 2: select_worker() method functionality"""
    
    def setup(self):
        self.router = RoundRobinRouter(num_workers=3)
    
    def execute(self):
        self.logger.info("Testing select_worker() method")
        
        # Select first worker
        worker_id = self.router.select_worker()
        self.assert_true(worker_id >= 0, "Worker ID should be non-negative")
        self.assert_true(worker_id < 3, "Worker ID should be less than num_workers")
        self.logger.info(f"✓ First selection: Worker {worker_id}")
        
        # Select multiple times
        selections = [self.router.select_worker() for _ in range(10)]
        self.logger.info(f"✓ Selected 10 workers: {selections}")
        
        # All selections should be valid
        for i, wid in enumerate(selections):
            self.assert_true(0 <= wid < 3, f"Selection {i}: Worker ID {wid} out of range")
        
        self.result.details['selections'] = selections


class TestRoundRobinOrder(ManualTest):
    """Test 3: Round robin order verification"""
    
    def setup(self):
        self.router = RoundRobinRouter(num_workers=3)
    
    def execute(self):
        self.logger.info("Testing round robin order: 0, 1, 2, 0, 1, 2...")
        
        # Expected pattern for 3 workers
        expected_pattern = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        actual_pattern = [self.router.select_worker() for _ in range(12)]
        
        self.logger.info(f"Expected: {expected_pattern}")
        self.logger.info(f"Actual:   {actual_pattern}")
        
        # Verify pattern matches
        self.assert_equal(actual_pattern, expected_pattern, 
                         "Round robin order should be 0, 1, 2, 0, 1, 2...")
        
        self.logger.info("✓ Round robin order is correct")
        
        self.result.details['expected'] = expected_pattern
        self.result.details['actual'] = actual_pattern


class TestDifferentWorkerCounts(ManualTest):
    """Test 4: Different worker counts"""
    
    def execute(self):
        self.logger.info("Testing round robin with different worker counts")
        
        test_cases = [
            (1, [0, 0, 0, 0, 0]),  # 1 worker: always 0
            (2, [0, 1, 0, 1, 0]),  # 2 workers: 0, 1, 0, 1...
            (4, [0, 1, 2, 3, 0]),  # 4 workers: 0, 1, 2, 3, 0...
        ]
        
        results = {}
        
        for num_workers, expected in test_cases:
            router = RoundRobinRouter(num_workers=num_workers)
            actual = [router.select_worker() for _ in range(len(expected))]
            
            self.logger.info(f"Workers={num_workers}: Expected={expected}, Actual={actual}")
            
            self.assert_equal(actual, expected, 
                            f"Pattern mismatch for {num_workers} workers")
            
            results[num_workers] = {
                'expected': expected,
                'actual': actual,
                'match': actual == expected
            }
            
            self.logger.info(f"✓ {num_workers} workers: Pattern correct")
        
        self.result.details['test_cases'] = results


class TestStatistics(ManualTest):
    """Test 5: Statistics tracking"""
    
    def setup(self):
        self.router = RoundRobinRouter(num_workers=3)
    
    def execute(self):
        self.logger.info("Testing statistics tracking")
        
        # Initial stats
        stats = self.router.get_stats()
        self.assert_equal(stats['total_requests'], 0, "Initial total should be 0")
        self.assert_equal(stats['next_worker'], 0, "Initial next_worker should be 0")
        self.logger.info(f"✓ Initial stats: {stats}")
        
        # Select 10 workers
        for _ in range(10):
            self.router.select_worker()
        
        # Check stats after selections
        stats = self.router.get_stats()
        self.assert_equal(stats['total_requests'], 10, "Total should be 10")
        self.assert_equal(stats['next_worker'], 1, "Next worker should be 1 (10 % 3 = 1)")
        
        self.logger.info(f"✓ Statistics after 10 requests: {stats}")
        self.logger.info("✓ Statistics tracking is correct")
        
        self.result.details['final_stats'] = stats


class TestReset(ManualTest):
    """Test 6: Reset functionality"""
    
    def setup(self):
        self.router = RoundRobinRouter(num_workers=3)
    
    def execute(self):
        self.logger.info("Testing reset functionality")
        
        # Select some workers
        for _ in range(7):
            self.router.select_worker()
        
        # Check stats before reset
        stats_before = self.router.get_stats()
        self.logger.info(f"Before reset: {stats_before}")
        self.assert_equal(stats_before['total_requests'], 7, "Should have 7 requests")
        
        # Reset router
        self.router.reset()
        
        # Check stats after reset
        stats_after = self.router.get_stats()
        self.logger.info(f"After reset: {stats_after}")
        
        self.assert_equal(stats_after['total_requests'], 0, "Total should be 0 after reset")
        self.assert_equal(stats_after['next_worker'], 0, "Next worker should be 0 after reset")
        
        # Verify round robin starts from 0 again
        first_selection = self.router.select_worker()
        self.assert_equal(first_selection, 0, "First selection after reset should be 0")
        
        self.logger.info("✓ Reset functionality works correctly")
        
        self.result.details['before_reset'] = stats_before
        self.result.details['after_reset'] = stats_after


class TestThreadSafety(ManualTest):
    """Test 7: Thread safety (basic verification)"""
    
    def setup(self):
        self.router = RoundRobinRouter(num_workers=3)
    
    def execute(self):
        self.logger.info("Testing thread safety (basic verification)")
        
        # Note: This is a basic test. Full thread safety testing would require
        # concurrent access from multiple threads, which is beyond the scope
        # of this simple component test.
        
        # Verify lock exists (it's _lock with underscore)
        self.assert_true(hasattr(self.router, '_lock'), "Router should have a _lock")
        self.logger.info("✓ Router has _lock attribute")
        
        # Verify selections work (lock is used internally)
        selections = [self.router.select_worker() for _ in range(100)]
        
        # Verify all selections are valid
        for i, wid in enumerate(selections):
            self.assert_true(0 <= wid < 3, f"Selection {i}: Invalid worker ID {wid}")
        
        # Verify round robin pattern is maintained
        # With 100 selections and 3 workers: each should get 33 or 34 requests
        # Count manually
        worker_counts = {0: 0, 1: 0, 2: 0}
        for wid in selections:
            worker_counts[wid] += 1
        
        expected_count = 100 // 3  # 33
        
        for worker_id in range(3):
            count = worker_counts[worker_id]
            self.assert_true(count >= expected_count, 
                           f"Worker {worker_id} should have at least {expected_count} requests")
            self.assert_true(count <= expected_count + 1, 
                           f"Worker {worker_id} should have at most {expected_count + 1} requests")
        
        self.logger.info(f"✓ Worker distribution: {worker_counts}")
        self.logger.info("✓ Thread safety basic verification passed")
        self.logger.info("  (Note: Full concurrent testing requires multi-threaded test)")
        
        self.result.details['total_selections'] = 100
        self.result.details['worker_counts'] = worker_counts


def main():
    """Run all Round Robin Router tests"""
    print("\n" + "="*70)
    print("Test 8.2.1: Round Robin Router Component Test")
    print("="*70)
    print("\nThis test validates the Round Robin Router in isolation.")
    print("No processes or network communication involved.\n")
    
    tests = [
        TestRouterCreation(
            "test_router_creation",
            "Test router instance creation with different worker counts"
        ),
        TestSelectWorker(
            "test_select_worker",
            "Test select_worker() method functionality"
        ),
        TestRoundRobinOrder(
            "test_round_robin_order",
            "Verify round robin order: 0, 1, 2, 0, 1, 2..."
        ),
        TestDifferentWorkerCounts(
            "test_different_worker_counts",
            "Test round robin with 1, 2, 4 workers"
        ),
        TestStatistics(
            "test_statistics",
            "Test statistics tracking and per-worker distribution"
        ),
        TestReset(
            "test_reset",
            "Test reset functionality"
        ),
        TestThreadSafety(
            "test_thread_safety",
            "Basic thread safety verification"
        ),
    ]
    
    results = run_test_suite(tests)
    
    # Print detailed summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"{status:12} {test_name:40} ({result.duration:.2f}s)")
    
    print("="*70)
    
    # Exit with appropriate code
    all_passed = all(r.passed for r in results.values())
    
    if all_passed:
        print("\n🎉 All Round Robin Router tests PASSED!")
        print("\nThe Round Robin Router component is working correctly:")
        print("  ✓ Router creation with different worker counts")
        print("  ✓ Worker selection functionality")
        print("  ✓ Round robin order (0, 1, 2, 0, 1, 2...)")
        print("  ✓ Statistics tracking")
        print("  ✓ Reset functionality")
        print("  ✓ Thread safety (basic verification)")
        print("\nReady to proceed to next test: 8.2.2 (GPU Worker Initialization)")
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
