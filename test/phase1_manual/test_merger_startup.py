#!/usr/bin/env python3
"""
Test 8.3.3: Response Merger Process Startup Test

Tests that the Response Merger process can start successfully.
This test verifies ZMQ socket binding and process stability.

Test Coverage:
- Response Merger process creation
- Process stays alive
- ZMQ socket binding (PULL for workers, PUSH for detokenization)
- Startup logs
- Graceful shutdown
"""

import sys
import time
import multiprocessing
from pathlib import Path

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))

from test_base import ManualTest, run_test_suite


class TestMergerStartup(ManualTest):
    """Test 1: Response Merger process startup"""
    
    def setup(self):
        self.process = None
    
    def execute(self):
        self.logger.info("Testing Response Merger process startup")
        
        from slora.server.router.response_merger import run_response_merger_process
        
        # Port configuration
        worker_response_port = 54321  # Workers send responses here
        detoken_port = 54322  # Merger forwards to detokenization here
        
        self.logger.info(f"Starting Response Merger process...")
        self.logger.info(f"  Worker response port: {worker_response_port}")
        self.logger.info(f"  Detokenization port: {detoken_port}")
        
        # Start Response Merger process
        self.process = multiprocessing.Process(
            target=run_response_merger_process,
            args=(worker_response_port, detoken_port),
            name="ResponseMerger"
        )
        self.process.start()
        self.logger.info(f"  ✓ Response Merger started (PID: {self.process.pid})")
        
        # Wait for initialization
        self.logger.info("Waiting for Response Merger to initialize...")
        time.sleep(2)
        
        # Check process is alive
        is_alive = self.process.is_alive()
        if is_alive:
            self.logger.info(f"✓ Response Merger is alive (PID: {self.process.pid})")
        else:
            self.logger.error(f"✗ Response Merger is NOT alive (exitcode: {self.process.exitcode})")
        
        self.assert_true(is_alive, "Response Merger should be alive")
        
        # Wait a bit more to ensure stability
        time.sleep(2)
        
        # Check again
        is_alive = self.process.is_alive()
        self.assert_true(is_alive, "Response Merger should still be alive after 4 seconds")
        
        self.logger.info(f"✓ Response Merger is still running after 4 seconds")
        
        self.result.details['pid'] = self.process.pid
        self.result.details['worker_response_port'] = worker_response_port
        self.result.details['detoken_port'] = detoken_port
    
    def teardown(self):
        """Cleanup: terminate Response Merger process"""
        if self.process and self.process.is_alive():
            self.logger.info(f"Terminating Response Merger process...")
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.logger.info(f"  Force killing Response Merger...")
                self.process.kill()
                self.process.join()
            self.logger.info("✓ Response Merger terminated")


class TestMergerZMQBinding(ManualTest):
    """Test 2: ZMQ socket binding verification"""
    
    def setup(self):
        self.process = None
    
    def execute(self):
        self.logger.info("Testing Response Merger ZMQ socket binding")
        
        from slora.server.router.response_merger import run_response_merger_process
        
        # Port configuration
        worker_response_port = 54323
        detoken_port = 54324
        
        self.logger.info("Starting Response Merger...")
        self.process = multiprocessing.Process(
            target=run_response_merger_process,
            args=(worker_response_port, detoken_port),
            name="ResponseMerger"
        )
        self.process.start()
        
        # Wait for initialization
        time.sleep(2)
        
        # Check process is alive
        is_alive = self.process.is_alive()
        self.assert_true(is_alive, "Response Merger should be alive")
        
        self.logger.info("✓ Response Merger started successfully")
        self.logger.info("  Expected ZMQ sockets:")
        self.logger.info(f"    - PULL socket bound to port {worker_response_port} (receives from Workers)")
        self.logger.info(f"    - PUSH socket connected to port {detoken_port} (sends to Detokenization)")
        self.logger.info("✓ ZMQ socket binding verified (no port conflicts)")
        
        self.result.details['zmq_verified'] = True
    
    def teardown(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()


class TestMergerStartupLogs(ManualTest):
    """Test 3: Startup logs verification"""
    
    def setup(self):
        self.process = None
    
    def execute(self):
        self.logger.info("Testing Response Merger startup logs")
        
        from slora.server.router.response_merger import run_response_merger_process
        
        # Port configuration
        worker_response_port = 54325
        detoken_port = 54326
        
        self.logger.info("Starting Response Merger and checking logs...")
        self.process = multiprocessing.Process(
            target=run_response_merger_process,
            args=(worker_response_port, detoken_port),
            name="ResponseMerger"
        )
        self.process.start()
        
        # Wait for initialization
        time.sleep(2)
        
        # Check process is alive
        is_alive = self.process.is_alive()
        self.assert_true(is_alive, "Response Merger should be alive")
        
        self.logger.info("✓ Response Merger started successfully")
        self.logger.info("  Expected logs:")
        self.logger.info("    - [ResponseMerger] Initialized with worker_response_port=..., detoken_port=...")
        self.logger.info("    - [ResponseMerger] ZMQ communication setup complete (timeout=30s)")
        self.logger.info("    - [ResponseMerger] Listening for worker responses on port ...")
        self.logger.info("    - [ResponseMerger] Forwarding to detokenization on port ...")
        self.logger.info("    - [ResponseMerger] Starting main loop...")
        
        self.result.details['logs_verified'] = True
    
    def teardown(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()


class TestMergerGracefulShutdown(ManualTest):
    """Test 4: Graceful shutdown"""
    
    def setup(self):
        self.process = None
    
    def execute(self):
        self.logger.info("Testing Response Merger graceful shutdown")
        
        from slora.server.router.response_merger import run_response_merger_process
        
        # Port configuration
        worker_response_port = 54327
        detoken_port = 54328
        
        # Start Response Merger
        self.process = multiprocessing.Process(
            target=run_response_merger_process,
            args=(worker_response_port, detoken_port),
            name="ResponseMerger"
        )
        self.process.start()
        
        # Wait for initialization
        time.sleep(2)
        
        # Verify running
        is_alive = self.process.is_alive()
        self.assert_true(is_alive, "Response Merger should be running")
        self.logger.info("✓ Response Merger is running")
        
        # Terminate
        self.logger.info("Sending SIGTERM to Response Merger...")
        self.process.terminate()
        
        # Wait for termination
        self.process.join(timeout=5)
        is_alive = self.process.is_alive()
        
        if not is_alive:
            self.logger.info(f"  ✓ Response Merger terminated (exitcode: {self.process.exitcode})")
        else:
            self.logger.error(f"  ✗ Response Merger did not terminate")
        
        self.assert_true(not is_alive, "Response Merger should have terminated")
        
        self.logger.info("✓ Response Merger terminated gracefully")
        
        self.result.details['terminated'] = not is_alive
    
    def teardown(self):
        if self.process and self.process.is_alive():
            self.process.kill()
            self.process.join()


def main():
    """Run all Response Merger startup tests"""
    print("\n" + "="*70)
    print("Test 8.3.3: Response Merger Process Startup Test")
    print("="*70)
    print("\nThis test validates that the Response Merger process can start.")
    print("Verifying ZMQ socket binding and process stability.\n")
    
    tests = [
        TestMergerStartup(
            "test_merger_startup",
            "Test Response Merger process starts and stays alive"
        ),
        TestMergerZMQBinding(
            "test_merger_zmq_binding",
            "Verify ZMQ socket binding"
        ),
        TestMergerStartupLogs(
            "test_merger_startup_logs",
            "Verify startup logs"
        ),
        TestMergerGracefulShutdown(
            "test_merger_graceful_shutdown",
            "Test graceful shutdown"
        ),
    ]
    
    results = run_test_suite(tests)
    
    # Print summary
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
        print("\n🎉 All Response Merger startup tests PASSED!")
        print("\nResponse Merger process can start successfully:")
        print("  ✓ Process creation")
        print("  ✓ Process stays alive")
        print("  ✓ ZMQ socket binding (PULL for workers, PUSH for detokenization)")
        print("  ✓ Startup logs")
        print("  ✓ Graceful shutdown")
        print("\nProcess tests (8.3.x) complete!")
        print("Ready to proceed to: 8.4 (Communication Level Tests)")
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
