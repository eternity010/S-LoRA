#!/usr/bin/env python3
"""
Test 8.2.2: GPU Worker Initialization Test

Tests GPU Worker initialization without starting inference.
This test validates the Worker can be created and configured properly.

Test Coverage:
- Worker instance creation
- GPU environment setup
- ZMQ socket creation
- Adapter configuration initialization
- Request queue setup (without model loading)
"""

import sys
import os
from pathlib import Path
import argparse

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))

from test_base import ManualTest, run_test_suite


class TestWorkerCreation(ManualTest):
    """Test 1: Worker instance creation"""
    
    def execute(self):
        self.logger.info("Testing GPU Worker instance creation")
        
        # Import here to avoid early initialization
        from slora.server.router.gpu_worker import GPUWorker
        
        # Create minimal args
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            adapter_dirs=[],
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10,
            eos_id=2,
            no_lora=True
        )
        
        # Create worker (will setup GPU environment)
        worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
        
        # Verify basic attributes
        self.assert_equal(worker.worker_id, 0, "Worker ID should be 0")
        self.assert_equal(worker.gpu_id, 0, "GPU ID should be 0")
        self.assert_not_none(worker.args, "Args should not be None")
        
        self.logger.info(f"✓ Worker created: worker_id={worker.worker_id}, gpu_id={worker.gpu_id}")
        
        # Verify initialization state
        self.assert_true(worker.model is None, "Model should be None (not loaded yet)")
        self.assert_true(worker.model_rpc is None, "Model RPC should be None")
        self.assert_not_none(worker.lora_ranks, "lora_ranks should be initialized")
        
        self.logger.info("✓ Worker initialization state is correct")
        
        self.result.details['worker_id'] = worker.worker_id
        self.result.details['gpu_id'] = worker.gpu_id


class TestGPUEnvironmentSetup(ManualTest):
    """Test 2: GPU environment setup"""
    
    def execute(self):
        self.logger.info("Testing GPU environment setup")
        
        from slora.server.router.gpu_worker import GPUWorker
        import torch
        
        # Create args
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            adapter_dirs=[],
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10,
            eos_id=2,
            no_lora=True
        )
        
        # Save original CUDA_VISIBLE_DEVICES
        original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        
        # Create worker with GPU 0
        worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
        
        # Verify CUDA_VISIBLE_DEVICES was set
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        self.logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        self.assert_equal(cuda_visible, '0', "CUDA_VISIBLE_DEVICES should be '0'")
        
        # Verify CUDA is available
        self.assert_true(torch.cuda.is_available(), "CUDA should be available")
        
        # Verify device count (should see only 1 GPU from worker's perspective)
        device_count = torch.cuda.device_count()
        self.logger.info(f"Visible GPU count: {device_count}")
        self.assert_equal(device_count, 1, "Worker should see only 1 GPU")
        
        # Get GPU properties
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.logger.info(f"✓ GPU 0: {props.name}")
            self.logger.info(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            self.logger.info(f"  Compute capability: {props.major}.{props.minor}")
        
        self.logger.info("✓ GPU environment setup is correct")
        
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
        
        self.result.details['cuda_visible_devices'] = cuda_visible
        self.result.details['device_count'] = device_count


class TestZMQSocketCreation(ManualTest):
    """Test 3: ZMQ socket creation"""
    
    def execute(self):
        self.logger.info("Testing ZMQ socket creation")
        
        from slora.server.router.gpu_worker import GPUWorker
        
        # Create args
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            adapter_dirs=[],
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10,
            eos_id=2,
            no_lora=True
        )
        
        # Create worker
        worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
        
        # Setup ZMQ (without binding to avoid port conflicts)
        request_port = 50000
        response_port = 54321
        
        worker._setup_zmq(request_port, response_port)
        
        # Verify ZMQ context and sockets were created
        self.assert_not_none(worker.context, "ZMQ context should be created")
        self.assert_not_none(worker.request_receiver, "Request receiver socket should be created")
        self.assert_not_none(worker.response_sender, "Response sender socket should be created")
        
        self.logger.info("✓ ZMQ context created")
        self.logger.info("✓ Request receiver socket created")
        self.logger.info("✓ Response sender socket created")
        
        # Cleanup
        worker.request_receiver.close()
        worker.response_sender.close()
        worker.context.term()
        
        self.logger.info("✓ ZMQ sockets cleaned up")
        
        self.result.details['request_port'] = request_port
        self.result.details['response_port'] = response_port


class TestAdapterConfiguration(ManualTest):
    """Test 4: Adapter configuration initialization"""
    
    def execute(self):
        self.logger.info("Testing Adapter configuration initialization")
        
        from slora.server.router.gpu_worker import GPUWorker
        
        # Create args without adapters
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            adapter_dirs=[],
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10,
            eos_id=2,
            no_lora=True
        )
        
        # Create worker
        worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
        
        # Verify lora_ranks was initialized
        self.assert_not_none(worker.lora_ranks, "lora_ranks should be initialized")
        self.assert_true(isinstance(worker.lora_ranks, dict), "lora_ranks should be a dict")
        
        # Verify None key exists (for base model)
        self.assert_true(None in worker.lora_ranks, "lora_ranks should have None key")
        self.assert_equal(worker.lora_ranks[None], 0, "Base model rank should be 0")
        
        self.logger.info(f"✓ lora_ranks initialized: {worker.lora_ranks}")
        
        # Verify adapter memory tracking
        self.assert_equal(worker.actual_adapter_memory_usage, 0, 
                         "Initial adapter memory usage should be 0")
        
        self.logger.info("✓ Adapter configuration initialized correctly")
        
        self.result.details['lora_ranks'] = worker.lora_ranks
        self.result.details['adapter_memory'] = worker.actual_adapter_memory_usage


class TestRequestQueueSetup(ManualTest):
    """Test 5: Request queue setup (without model)"""
    
    def execute(self):
        self.logger.info("Testing request queue setup")
        
        from slora.server.router.gpu_worker import GPUWorker
        
        # Create args
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            adapter_dirs=[],
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10,
            eos_id=2,
            no_lora=True
        )
        
        # Create worker
        worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
        
        # Setup request queue
        worker._setup_request_queue()
        
        # Verify request queue was created
        self.assert_not_none(worker.req_queue, "Request queue should be created")
        
        # Verify queue configuration
        self.assert_equal(worker.req_queue.max_total_tokens, 1000, 
                         "max_total_tokens should match args")
        
        self.logger.info("✓ Request queue created")
        self.logger.info(f"  max_total_tokens: {worker.req_queue.max_total_tokens}")
        self.logger.info(f"  batch_max_tokens: {worker.req_queue.batch_max_tokens}")
        self.logger.info(f"  running_max_req_size: {worker.req_queue.running_max_req_size}")
        
        # Verify current_batch is None
        self.assert_true(worker.current_batch is None, "current_batch should be None initially")
        
        self.logger.info("✓ Request queue setup correctly")
        
        self.result.details['max_total_tokens'] = worker.req_queue.max_total_tokens
        self.result.details['batch_max_tokens'] = worker.req_queue.batch_max_tokens


class TestMultipleWorkers(ManualTest):
    """Test 6: Multiple workers with different GPUs"""
    
    def execute(self):
        self.logger.info("Testing multiple workers with different GPU IDs")
        
        from slora.server.router.gpu_worker import GPUWorker
        import torch
        
        # Get available GPU count
        gpu_count = torch.cuda.device_count()
        self.logger.info(f"Available GPUs: {gpu_count}")
        
        if gpu_count < 2:
            self.logger.info("⚠ Only 1 GPU available, testing with GPU 0 only")
            test_gpus = [0]
        else:
            self.logger.info(f"Testing with GPUs 0 and 1")
            test_gpus = [0, 1]
        
        # Create args
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            adapter_dirs=[],
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10,
            eos_id=2,
            no_lora=True
        )
        
        workers = []
        for i, gpu_id in enumerate(test_gpus):
            worker = GPUWorker(worker_id=i, gpu_id=gpu_id, args=args)
            workers.append(worker)
            
            self.assert_equal(worker.worker_id, i, f"Worker {i} ID should be {i}")
            self.assert_equal(worker.gpu_id, gpu_id, f"Worker {i} GPU ID should be {gpu_id}")
            
            self.logger.info(f"✓ Worker {i} created on GPU {gpu_id}")
        
        self.logger.info(f"✓ Created {len(workers)} workers successfully")
        
        self.result.details['num_workers'] = len(workers)
        self.result.details['gpu_ids'] = test_gpus


def main():
    """Run all GPU Worker initialization tests"""
    print("\n" + "="*70)
    print("Test 8.2.2: GPU Worker Initialization Test")
    print("="*70)
    print("\nThis test validates GPU Worker initialization without inference.")
    print("Tests: creation, GPU setup, ZMQ, adapters, request queue.\n")
    
    tests = [
        TestWorkerCreation(
            "test_worker_creation",
            "Test Worker instance creation"
        ),
        TestGPUEnvironmentSetup(
            "test_gpu_environment_setup",
            "Test GPU environment setup (CUDA_VISIBLE_DEVICES)"
        ),
        TestZMQSocketCreation(
            "test_zmq_socket_creation",
            "Test ZMQ socket creation"
        ),
        TestAdapterConfiguration(
            "test_adapter_configuration",
            "Test Adapter configuration initialization"
        ),
        TestRequestQueueSetup(
            "test_request_queue_setup",
            "Test request queue setup"
        ),
        TestMultipleWorkers(
            "test_multiple_workers",
            "Test multiple workers with different GPUs"
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
        print("\n🎉 All GPU Worker initialization tests PASSED!")
        print("\nThe GPU Worker component can be initialized correctly:")
        print("  ✓ Worker instance creation")
        print("  ✓ GPU environment setup")
        print("  ✓ ZMQ socket creation")
        print("  ✓ Adapter configuration")
        print("  ✓ Request queue setup")
        print("  ✓ Multiple workers support")
        print("\nReady to proceed to next test: 8.2.3 (Router Manager Initialization)")
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
