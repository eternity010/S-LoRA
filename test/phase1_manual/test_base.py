"""
Base Test Framework for Phase 1 Manual Tests

Provides common utilities and helpers for manual testing scripts.
"""

import sys
import time
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))


class TestResult:
    """Container for test results"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.error_message: Optional[str] = None
        self.duration: float = 0.0
        self.details: Dict[str, Any] = {}
    
    def mark_passed(self, duration: float, **details):
        """Mark test as passed"""
        self.passed = True
        self.duration = duration
        self.details = details
    
    def mark_failed(self, error_message: str, duration: float, **details):
        """Mark test as failed"""
        self.passed = False
        self.error_message = error_message
        self.duration = duration
        self.details = details
    
    def print_summary(self):
        """Print test result summary"""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        print(f"\n{'='*70}")
        print(f"Test: {self.test_name}")
        print(f"Status: {status}")
        print(f"Duration: {self.duration:.2f}s")
        
        if self.details:
            print("\nDetails:")
            for key, value in self.details.items():
                print(f"  {key}: {value}")
        
        if self.error_message:
            print(f"\nError: {self.error_message}")
        
        print(f"{'='*70}\n")


class ManualTest:
    """Base class for manual tests"""
    
    def __init__(self, test_name: str, description: str):
        self.test_name = test_name
        self.description = description
        self.logger = self._setup_logger()
        self.result = TestResult(test_name)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup test logger"""
        logger = logging.getLogger(self.test_name)
        logger.setLevel(logging.INFO)
        
        # Console handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run(self) -> TestResult:
        """Run the test"""
        print(f"\n{'='*70}")
        print(f"Running: {self.test_name}")
        print(f"Description: {self.description}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        try:
            self.setup()
            self.execute()
            duration = time.time() - start_time
            self.result.mark_passed(duration)
            self.logger.info(f"Test passed in {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.result.mark_failed(error_msg, duration)
            self.logger.error(f"Test failed: {error_msg}")
        finally:
            try:
                self.teardown()
            except Exception as e:
                self.logger.error(f"Teardown error: {e}")
        
        self.result.print_summary()
        return self.result
    
    def setup(self):
        """Setup test environment (override in subclass)"""
        pass
    
    def execute(self):
        """Execute test logic (override in subclass)"""
        raise NotImplementedError("Subclass must implement execute()")
    
    def teardown(self):
        """Cleanup test environment (override in subclass)"""
        pass
    
    def assert_true(self, condition: bool, message: str):
        """Assert condition is true"""
        if not condition:
            raise AssertionError(message)
    
    def assert_equal(self, actual, expected, message: str = ""):
        """Assert values are equal"""
        if actual != expected:
            msg = f"Expected {expected}, got {actual}"
            if message:
                msg = f"{message}: {msg}"
            raise AssertionError(msg)
    
    def assert_not_none(self, value, message: str = ""):
        """Assert value is not None"""
        if value is None:
            msg = "Value is None"
            if message:
                msg = f"{message}: {msg}"
            raise AssertionError(msg)


def run_test_suite(tests: List[ManualTest]) -> Dict[str, TestResult]:
    """Run a suite of manual tests"""
    print(f"\n{'#'*70}")
    print(f"# Running Test Suite: {len(tests)} tests")
    print(f"{'#'*70}\n")
    
    results = {}
    passed = 0
    failed = 0
    
    for test in tests:
        result = test.run()
        results[test.test_name] = result
        
        if result.passed:
            passed += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'#'*70}")
    print(f"# Test Suite Summary")
    print(f"{'#'*70}")
    print(f"Total: {len(tests)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"{'#'*70}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    class ExampleTest(ManualTest):
        def execute(self):
            self.logger.info("Running example test")
            self.assert_true(True, "This should pass")
            self.assert_equal(1 + 1, 2, "Math works")
    
    test = ExampleTest("example_test", "Example test to demonstrate framework")
    result = test.run()
    
    sys.exit(0 if result.passed else 1)
