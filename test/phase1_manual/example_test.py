#!/usr/bin/env python3
"""
Example Manual Test

Demonstrates how to use the manual testing framework.
This can be used as a template for creating new manual tests.
"""

import sys
from test_base import ManualTest, run_test_suite


class SimpleTest(ManualTest):
    """Simple test example"""
    
    def setup(self):
        """Setup test environment"""
        self.logger.info("Setting up test environment")
        self.test_data = {"value": 42}
    
    def execute(self):
        """Execute test logic"""
        self.logger.info("Executing test")
        
        # Test assertions
        self.assert_not_none(self.test_data, "Test data should exist")
        self.assert_equal(self.test_data["value"], 42, "Value should be 42")
        self.assert_true(len(self.test_data) > 0, "Test data should not be empty")
        
        # Store results
        self.result.details["test_data"] = self.test_data
        
        self.logger.info("Test completed successfully")
    
    def teardown(self):
        """Cleanup test environment"""
        self.logger.info("Cleaning up test environment")
        self.test_data = None


class EnvironmentTest(ManualTest):
    """Test environment validation"""
    
    def execute(self):
        """Check environment"""
        import torch
        import zmq
        
        self.logger.info("Checking Python environment")
        
        # Check Python version
        python_version = sys.version
        self.logger.info(f"Python version: {python_version}")
        
        # Check PyTorch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        
        self.logger.info(f"PyTorch version: {torch_version}")
        self.logger.info(f"CUDA available: {cuda_available}")
        self.logger.info(f"GPU count: {gpu_count}")
        
        # Check ZMQ
        zmq_version = zmq.zmq_version()
        self.logger.info(f"ZMQ version: {zmq_version}")
        
        # Assertions
        self.assert_true(cuda_available, "CUDA should be available")
        self.assert_true(gpu_count > 0, "At least one GPU should be available")
        
        # Store results
        self.result.details.update({
            "python_version": python_version.split()[0],
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "zmq_version": zmq_version
        })


def main():
    """Run example tests"""
    tests = [
        SimpleTest(
            "simple_test",
            "Simple test demonstrating basic assertions"
        ),
        EnvironmentTest(
            "environment_test",
            "Validate Python environment and dependencies"
        )
    ]
    
    results = run_test_suite(tests)
    
    # Exit with error code if any test failed
    all_passed = all(r.passed for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
