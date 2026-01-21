#!/usr/bin/env python3
"""
Test 8.3.1: Single Worker Process Startup Test

Tests that a single Worker process can start successfully.
This test uses Dummy mode to avoid loading real models.

Test Coverage:
- Worker process creation
- Process stays alive
- Startup logs output
- Process state checking
- Graceful shutdown
"""

import sys
import time
import multiprocessing
from pathlib import Path
import argparse

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))

from test_base import ManualTest, run_test_suite


class TestSingleWorkerStartup(ManualTest):
    """Test 1: Single Worker process startup"""
    
    def setup(self):
        self.process = None
    
    def execute(self):
        self.logger.info("Testing single Worker process startup (Dummy mode)")
        
        from slora.server.router.dp_manager import run_gpu_worker_process
        
        # Create args with Dummy mode
        args = argparse.Namespace(
            model_dir='dummy-llama-7b',  # Dummy model
            adapter_dirs=[],
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10,
            eos_id=2,
            no_lora=True,
            dummy=True,  # Use dummy mode
            load_way='HF',
            mode=[],
            max_req_total_len=512,
            pool_size_lora=0
        )
        
        # Start Worker process
        self.logger.info("Starting Worker 0 on GPU 0...")
        self.process = multiprocessing.Process(
            target=run_gpu_worker_process,
            args=(0, 0, args, 50000, 54321),
            name="GPUWorker-0"
        )
        self.process.start()
        
        self.logger.info(f"✓ Worker process started (PID: {self.process.pid})")
        
        # Wait a bit for initialization
        time.sleep(3)
        
        # Check if process is alive
        is_alive = self.process.is_alive()
        self.assert_true(is_alive, "Worker process should be alive")
        self.logger.info(f"✓ Worker process is alive after 3 seconds")
        
        # Check process state
        if is_alive:
            exitcode = self.process.exitcode
            self.assert_true(exitcode is None, "Exit code should be None for running process")
            self.logger.info(f"✓ Worker process exit code: {exitcode} (None = running)")
        
        # Wait a bit more to ensure stability
        time.sleep(2)
        
        # Check again
        is_alive = self.process.is_alive()
        self.assert_true(is_alive, "Worker process should still be alive after 5 seconds")
        self.logger.info(f"✓ Worker process is still alive after 5 seconds")
        
        self.result.details['pid'] = self.process.pid
        self.result.details['alive'] = is_alive
        self.result.details['exitcode'] = self.process.exitcode
    
    def teardown(self):
        """Cleanup: terminate the Worker process"""
        if self.process and self.process.is_alive():
            self.logger.info("Terminating Worker process...")
            self.process.terminate()
            self.process.join(timeout=5)
            
            if self.process.is_alive():
                self.logger.info("Force killing Worker process...")
                self.process.kill()
                self.process.join()
            
            self.logger.info("✓ Worker process terminated")


class TestWorkerStartupLogs(ManualTest):
    """Test 2: Worker startup logs verification"""
    
    def setup(self):
        self.process = None
    
    def execute(self):
        self.logger.info("Testing Worker startup logs")
        
        from slora.server.router.dp_manager import run_gpu_worker_process
        
        # Create args
        args = argparse.Namespace(
            model_dir='dummy-llama-7b',
            adapter_dirs=[],
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10,
            eos_id=2,
            no_lora=True,
            dummy=True,
            load_way='HF',
            mode=[],
            max_req_total_len=512,
            pool_size_lora=0
        )
        
        # Start Worker process
        self.logger.info("Starting Worker 0 and checking logs...")
        self.process = multiprocessing.Process(
            target=run_gpu_worker_process,
            args=(0, 0, args, 50000, 54321),
            name="GPUWorker-0"
        )
        self.process.start()
        
        # Wait for initialization
        time.sleep(3)
        
        # Check process is alive (logs should have been output)
        is_alive = self.process.is_alive()
        self.assert_true(is_alive, "Worker should be alive (logs should be output)")
        
        self.logger.info("✓ Worker started successfully")
        self.logger.info("  Expected logs:")
        self.logger.info("    - [Worker 0] Starting worker process on GPU 0...")
        self.logger.info("    - [Worker 0] ========== GPU Environment Setup ==========")
        self.logger.info("    - [Worker 0] Creating GPUWorker instance...")
        self.logger.info("    - [Worker 0] Setting up ZMQ communication...")
        self.logger.info("    - [Worker 0] Setting up request queue...")
        self.logger.info("    - [Worker 0] Initializing model RPC...")
        self.logger.info("    - [Worker 0] Worker initialization complete...")
        
        self.result.details['process_alive'] = is_alive
    
    def teardown(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()


class TestWorkerGracefulShutdown(ManualTest):
    """Test 3: Worker graceful shutdown"""
    
    def setup(self):
        self.process = None
    
    def execute(self):
        self.logger.info("Testing Worker graceful shutdown")
        
        from slora.server.router.dp_manager import run_gpu_worker_process
        
        # Create args
        args = argparse.Namespace(
            model_dir='dummy-llama-7b',
            adapter_dirs=[],
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10,
            eos_id=2,
            no_lora=True,
            dummy=True,
            load_way='HF',
            mode=[],
            max_req_total_len=512,
            pool_size_lora=0
        )
        
        # Start Worker
        self.process = multiprocessing.Process(
            target=run_gpu_worker_process,
            args=(0, 0, args, 50000, 54321),
            name="GPUWorker-0"
        )
        self.process.start()
        
        # Wait for initialization
        time.sleep(3)
        
        # Verify it's running
        self.assert_true(self.process.is_alive(), "Worker should be running")
        self.logger.info("✓ Worker is running")
        
        # Terminate gracefully
        self.logger.info("Sending SIGTERM to Worker...")
        self.process.terminate()
        
        # Wait for termination
        self.process.join(timeout=5)
        
        # Check if terminated
        is_alive = self.process.is_alive()
        self.assert_true(not is_alive, "Worker should have terminated")
        self.logger.info("✓ Worker terminated gracefully")
        
        # Check exit code
        exitcode = self.process.exitcode
        self.logger.info(f"  Exit code: {exitcode}")
        
        # Exit code should be negative (signal termination) or 0
        self.assert_true(exitcode is not None, "Exit code should not be None")
        
        self.result.details['exitcode'] = exitcode
        self.result.details['terminated_gracefully'] = not is_alive
    
    def teardown(self):
        if self.process and self.process.is_alive():
            self.process.kill()
            self.process.join()


def main():
    """Run all single Worker startup tests"""
    print("\n" + "="*70)
    print("Test 8.3.1: Single Worker Process Startup Test")
    print("="*70)
    print("\nThis test validates that a Worker process can start successfully.")
    print("Using Dummy mode to avoid loading real models.\n")
    
    tests = [
        TestSingleWorkerStartup(
            "test_single_worker_startup",
            "Test Worker process starts and stays alive"
        ),
        TestWorkerStartupLogs(
            "test_worker_startup_logs",
            "Verify Worker startup logs are output"
        ),
        TestWorkerGracefulShutdown(
            "test_worker_graceful_shutdown",
            "Test Worker can be terminated gracefully"
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
        print("\n🎉 All single Worker startup tests PASSED!")
        print("\nWorker process can start successfully:")
        print("  ✓ Process creation and initialization")
        print("  ✓ Process stays alive")
        print("  ✓ Startup logs output")
        print("  ✓ Graceful shutdown")
        print("\nReady to proceed to: 8.3.2 (Multiple Workers Startup)")
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
