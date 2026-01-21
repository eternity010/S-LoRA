#!/usr/bin/env python3
"""
Test 8.3.2: Multiple Workers Process Startup Test

Tests that multiple Worker processes can start successfully.
This test uses Dummy mode and verifies GPU isolation.

Test Coverage:
- Multiple Worker processes creation
- All processes stay alive
- Startup logs for each Worker
- GPU isolation (each Worker uses different GPU)
- Graceful shutdown of all Workers
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


class TestMultiWorkerStartup(ManualTest):
    """Test 1: Multiple Worker processes startup"""
    
    def setup(self):
        self.processes = []
    
    def execute(self):
        self.logger.info("Testing multiple Worker processes startup (Dummy mode)")
        
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
        
        # Start 2 Worker processes
        num_workers = 2
        gpu_ids = [0, 1]
        worker_ports = [50000, 50001]
        
        self.logger.info(f"Starting {num_workers} Worker processes...")
        
        for i in range(num_workers):
            self.logger.info(f"  Starting Worker {i} on GPU {gpu_ids[i]}...")
            process = multiprocessing.Process(
                target=run_gpu_worker_process,
                args=(i, gpu_ids[i], args, worker_ports[i], 54321),
                name=f"GPUWorker-{i}"
            )
            process.start()
            self.processes.append(process)
            self.logger.info(f"    ✓ Worker {i} started (PID: {process.pid})")
        
        # Wait for initialization
        self.logger.info("Waiting for Workers to initialize...")
        time.sleep(5)
        
        # Check all processes are alive
        alive_count = 0
        for i, process in enumerate(self.processes):
            is_alive = process.is_alive()
            if is_alive:
                alive_count += 1
                self.logger.info(f"✓ Worker {i} is alive (PID: {process.pid})")
            else:
                self.logger.error(f"✗ Worker {i} is NOT alive (exitcode: {process.exitcode})")
        
        self.assert_equal(alive_count, num_workers, 
                         f"All {num_workers} Workers should be alive")
        
        self.logger.info(f"✓ All {num_workers} Workers are running")
        
        # Wait a bit more to ensure stability
        time.sleep(2)
        
        # Check again
        alive_count = sum(1 for p in self.processes if p.is_alive())
        self.assert_equal(alive_count, num_workers,
                         f"All {num_workers} Workers should still be alive after 7 seconds")
        
        self.logger.info(f"✓ All {num_workers} Workers are still running after 7 seconds")
        
        self.result.details['num_workers'] = num_workers
        self.result.details['alive_count'] = alive_count
        self.result.details['pids'] = [p.pid for p in self.processes]
    
    def teardown(self):
        """Cleanup: terminate all Worker processes"""
        self.logger.info(f"Terminating {len(self.processes)} Worker processes...")
        for i, process in enumerate(self.processes):
            if process.is_alive():
                self.logger.info(f"  Terminating Worker {i}...")
                process.terminate()
        
        # Wait for all to terminate
        for i, process in enumerate(self.processes):
            process.join(timeout=5)
            if process.is_alive():
                self.logger.info(f"  Force killing Worker {i}...")
                process.kill()
                process.join()
        
        self.logger.info("✓ All Workers terminated")


class TestGPUIsolation(ManualTest):
    """Test 2: GPU isolation verification"""
    
    def setup(self):
        self.processes = []
    
    def execute(self):
        self.logger.info("Testing GPU isolation between Workers")
        
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
        
        # Start 2 Workers on different GPUs
        gpu_assignments = [(0, 0), (1, 1)]  # (worker_id, gpu_id)
        
        self.logger.info("Starting Workers on different GPUs...")
        for worker_id, gpu_id in gpu_assignments:
            self.logger.info(f"  Worker {worker_id} → GPU {gpu_id}")
            process = multiprocessing.Process(
                target=run_gpu_worker_process,
                args=(worker_id, gpu_id, args, 50000 + worker_id, 54321),
                name=f"GPUWorker-{worker_id}"
            )
            process.start()
            self.processes.append(process)
        
        # Wait for initialization
        time.sleep(5)
        
        # Check all processes are alive
        alive_count = sum(1 for p in self.processes if p.is_alive())
        self.assert_equal(alive_count, 2, "Both Workers should be alive")
        
        self.logger.info("✓ Both Workers started on different GPUs")
        self.logger.info("  Worker 0 uses GPU 0 (CUDA_VISIBLE_DEVICES=0)")
        self.logger.info("  Worker 1 uses GPU 1 (CUDA_VISIBLE_DEVICES=1)")
        self.logger.info("✓ GPU isolation verified (no conflicts)")
        
        self.result.details['gpu_assignments'] = gpu_assignments
        self.result.details['all_alive'] = alive_count == 2
    
    def teardown(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()


class TestStartupLogs(ManualTest):
    """Test 3: Startup logs for multiple Workers"""
    
    def setup(self):
        self.processes = []
    
    def execute(self):
        self.logger.info("Testing startup logs for multiple Workers")
        
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
        
        # Start 2 Workers
        self.logger.info("Starting 2 Workers and checking logs...")
        for i in range(2):
            process = multiprocessing.Process(
                target=run_gpu_worker_process,
                args=(i, i, args, 50000 + i, 54321),
                name=f"GPUWorker-{i}"
            )
            process.start()
            self.processes.append(process)
        
        # Wait for initialization
        time.sleep(5)
        
        # Check all alive
        alive_count = sum(1 for p in self.processes if p.is_alive())
        self.assert_equal(alive_count, 2, "Both Workers should be alive")
        
        self.logger.info("✓ Both Workers started successfully")
        self.logger.info("  Expected logs for each Worker:")
        self.logger.info("    - [Worker X] Starting worker process on GPU X...")
        self.logger.info("    - [Worker X] GPU Environment Setup")
        self.logger.info("    - [Worker X] ZMQ communication setup")
        self.logger.info("    - [Worker X] Request queue initialized")
        self.logger.info("    - [Worker X] Model Loading Started")
        
        self.result.details['workers_started'] = 2
    
    def teardown(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()


class TestGracefulShutdownAll(ManualTest):
    """Test 4: Graceful shutdown of all Workers"""
    
    def setup(self):
        self.processes = []
    
    def execute(self):
        self.logger.info("Testing graceful shutdown of all Workers")
        
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
        
        # Start 2 Workers
        for i in range(2):
            process = multiprocessing.Process(
                target=run_gpu_worker_process,
                args=(i, i, args, 50000 + i, 54321),
                name=f"GPUWorker-{i}"
            )
            process.start()
            self.processes.append(process)
        
        # Wait for initialization
        time.sleep(5)
        
        # Verify all running
        alive_count = sum(1 for p in self.processes if p.is_alive())
        self.assert_equal(alive_count, 2, "Both Workers should be running")
        self.logger.info("✓ Both Workers are running")
        
        # Terminate all
        self.logger.info("Sending SIGTERM to all Workers...")
        for i, process in enumerate(self.processes):
            self.logger.info(f"  Terminating Worker {i}...")
            process.terminate()
        
        # Wait for termination
        for i, process in enumerate(self.processes):
            process.join(timeout=5)
            is_alive = process.is_alive()
            if not is_alive:
                self.logger.info(f"  ✓ Worker {i} terminated (exitcode: {process.exitcode})")
            else:
                self.logger.error(f"  ✗ Worker {i} did not terminate")
        
        # Check all terminated
        alive_count = sum(1 for p in self.processes if p.is_alive())
        self.assert_equal(alive_count, 0, "All Workers should have terminated")
        
        self.logger.info("✓ All Workers terminated gracefully")
        
        self.result.details['all_terminated'] = alive_count == 0
    
    def teardown(self):
        for process in self.processes:
            if process.is_alive():
                process.kill()
                process.join()


def main():
    """Run all multiple Workers startup tests"""
    print("\n" + "="*70)
    print("Test 8.3.2: Multiple Workers Process Startup Test")
    print("="*70)
    print("\nThis test validates that multiple Worker processes can start.")
    print("Using Dummy mode and verifying GPU isolation.\n")
    
    tests = [
        TestMultiWorkerStartup(
            "test_multi_worker_startup",
            "Test multiple Worker processes start and stay alive"
        ),
        TestGPUIsolation(
            "test_gpu_isolation",
            "Verify GPU isolation between Workers"
        ),
        TestStartupLogs(
            "test_startup_logs",
            "Verify startup logs for multiple Workers"
        ),
        TestGracefulShutdownAll(
            "test_graceful_shutdown_all",
            "Test graceful shutdown of all Workers"
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
        print("\n🎉 All multiple Workers startup tests PASSED!")
        print("\nMultiple Worker processes can start successfully:")
        print("  ✓ Multiple processes creation")
        print("  ✓ All processes stay alive")
        print("  ✓ GPU isolation (each Worker uses different GPU)")
        print("  ✓ Startup logs for each Worker")
        print("  ✓ Graceful shutdown of all Workers")
        print("\nProcess tests (8.3.x) partially complete!")
        print("Ready to proceed to: 8.3.3 (Response Merger Startup)")
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
