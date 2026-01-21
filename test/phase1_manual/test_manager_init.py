#!/usr/bin/env python3
"""
Test 8.2.3: DataParallelRouterManager Initialization Test

Tests DataParallelRouterManager initialization without starting workers.
This test validates the Manager can be created and configured properly.

Test Coverage:
- Manager instance creation
- GPU detection
- GPU ID parsing
- Port allocation
- ZMQ socket creation
"""

import sys
from pathlib import Path
import argparse

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))

from test_base import ManualTest, run_test_suite


class TestManagerCreation(ManualTest):
    """Test 1: Manager instance creation"""
    
    def execute(self):
        self.logger.info("Testing DataParallelRouterManager instance creation")
        
        from slora.server.router.dp_manager import DataParallelRouterManager
        
        # Create minimal args
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            num_workers=None,  # Auto-detect
            gpu_ids=None,  # Auto-assign
            adapter_dirs=[],
            no_lora=True,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        # Create manager
        manager = DataParallelRouterManager(
            args=args,
            router_port=12345,
            response_port=54321,
            detoken_port=12346
        )
        
        # Verify basic attributes
        self.assert_not_none(manager.args, "Args should not be None")
        self.assert_equal(manager.router_port, 12345, "Router port should be 12345")
        self.assert_equal(manager.response_port, 54321, "Response port should be 54321")
        self.assert_equal(manager.detoken_port, 12346, "Detoken port should be 12346")
        
        self.logger.info(f"✓ Manager created")
        self.logger.info(f"  router_port: {manager.router_port}")
        self.logger.info(f"  response_port: {manager.response_port}")
        self.logger.info(f"  detoken_port: {manager.detoken_port}")
        
        # Verify initialization state
        self.assert_not_none(manager.router, "Router should be initialized")
        self.assert_equal(len(manager.workers), 0, "Workers list should be empty initially")
        self.assert_equal(len(manager.worker_ports), 0, "Worker ports should be empty initially")
        
        self.logger.info("✓ Manager initialization state is correct")
        
        self.result.details['num_workers'] = manager.num_workers
        self.result.details['gpu_ids'] = manager.gpu_ids


class TestGPUDetection(ManualTest):
    """Test 2: GPU detection"""
    
    def execute(self):
        self.logger.info("Testing GPU detection")
        
        from slora.server.router.dp_manager import DataParallelRouterManager
        import torch
        
        # Get expected GPU count
        expected_gpu_count = torch.cuda.device_count()
        self.logger.info(f"Expected GPU count: {expected_gpu_count}")
        
        # Create args with auto-detect
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            num_workers=None,  # Auto-detect
            gpu_ids=None,  # Auto-assign
            adapter_dirs=[],
            no_lora=True,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        # Create manager
        manager = DataParallelRouterManager(
            args=args,
            router_port=12345,
            response_port=54321,
            detoken_port=12346
        )
        
        # Verify GPU detection
        self.assert_equal(manager.num_workers, expected_gpu_count, 
                         f"Should detect {expected_gpu_count} GPUs")
        self.assert_equal(len(manager.gpu_ids), expected_gpu_count,
                         f"Should have {expected_gpu_count} GPU IDs")
        
        # Verify GPU IDs are sequential
        expected_ids = list(range(expected_gpu_count))
        self.assert_equal(manager.gpu_ids, expected_ids,
                         f"GPU IDs should be {expected_ids}")
        
        self.logger.info(f"✓ Detected {manager.num_workers} GPUs")
        self.logger.info(f"✓ GPU IDs: {manager.gpu_ids}")
        
        self.result.details['detected_gpus'] = manager.num_workers
        self.result.details['gpu_ids'] = manager.gpu_ids


class TestGPUIDParsing(ManualTest):
    """Test 3: GPU ID parsing"""
    
    def execute(self):
        self.logger.info("Testing GPU ID parsing")
        
        from slora.server.router.dp_manager import DataParallelRouterManager
        
        test_cases = [
            ("0", [0]),
            ("0,1", [0, 1]),
            ("0,1,2", [0, 1, 2]),
            ("1,2", [1, 2]),
        ]
        
        for gpu_ids_str, expected_ids in test_cases:
            args = argparse.Namespace(
                model_dir='/models/llama-7b',
                num_workers=len(expected_ids),
                gpu_ids=gpu_ids_str,
                adapter_dirs=[],
                no_lora=True,
                max_total_token_num=1000,
                batch_max_tokens=500,
                running_max_req_size=10
            )
            
            manager = DataParallelRouterManager(
                args=args,
                router_port=12345,
                response_port=54321,
                detoken_port=12346
            )
            
            self.assert_equal(manager.gpu_ids, expected_ids,
                             f"GPU IDs for '{gpu_ids_str}' should be {expected_ids}")
            self.assert_equal(manager.num_workers, len(expected_ids),
                             f"Worker count should be {len(expected_ids)}")
            
            self.logger.info(f"✓ Parsed '{gpu_ids_str}' -> {manager.gpu_ids}")
        
        self.logger.info("✓ GPU ID parsing is correct")
        
        self.result.details['test_cases'] = test_cases


class TestPortAllocation(ManualTest):
    """Test 4: Port allocation"""
    
    def execute(self):
        self.logger.info("Testing port allocation")
        
        from slora.server.router.dp_manager import DataParallelRouterManager
        
        # Create args with 3 workers
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            num_workers=3,
            gpu_ids="0,1,2",
            adapter_dirs=[],
            no_lora=True,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        # Create manager
        manager = DataParallelRouterManager(
            args=args,
            router_port=12345,
            response_port=54321,
            detoken_port=12346
        )
        
        # Allocate ports
        manager._allocate_ports()
        
        # Verify ports were allocated
        self.assert_equal(len(manager.worker_ports), 3, "Should have 3 worker ports")
        
        # Verify ports are sequential starting from 50000
        expected_ports = [50000, 50001, 50002]
        self.assert_equal(manager.worker_ports, expected_ports,
                         f"Worker ports should be {expected_ports}")
        
        # Verify ports are unique
        unique_ports = set(manager.worker_ports)
        self.assert_equal(len(unique_ports), 3, "All ports should be unique")
        
        self.logger.info(f"✓ Allocated {len(manager.worker_ports)} ports")
        self.logger.info(f"✓ Worker ports: {manager.worker_ports}")
        
        self.result.details['worker_ports'] = manager.worker_ports


class TestZMQSetup(ManualTest):
    """Test 5: ZMQ socket setup"""
    
    def execute(self):
        self.logger.info("Testing ZMQ socket setup")
        
        from slora.server.router.dp_manager import DataParallelRouterManager
        
        # Create args with 2 workers
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            num_workers=2,
            gpu_ids="0,1",
            adapter_dirs=[],
            no_lora=True,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        # Create manager
        manager = DataParallelRouterManager(
            args=args,
            router_port=12345,
            response_port=54321,
            detoken_port=12346
        )
        
        # Allocate ports first
        manager._allocate_ports()
        
        # Setup ZMQ
        manager._setup_zmq()
        
        # Verify ZMQ context and sockets
        self.assert_not_none(manager.context, "ZMQ context should be created")
        self.assert_not_none(manager.request_receiver, "Request receiver should be created")
        self.assert_equal(len(manager.request_senders), 2, "Should have 2 request senders")
        
        self.logger.info("✓ ZMQ context created")
        self.logger.info("✓ Request receiver socket created")
        self.logger.info(f"✓ {len(manager.request_senders)} request sender sockets created")
        
        # Cleanup
        manager.request_receiver.close()
        for sender in manager.request_senders:
            sender.close()
        manager.context.term()
        
        self.logger.info("✓ ZMQ sockets cleaned up")
        
        self.result.details['num_senders'] = len(manager.request_senders)


class TestRouterIntegration(ManualTest):
    """Test 6: Router integration"""
    
    def execute(self):
        self.logger.info("Testing Router integration")
        
        from slora.server.router.dp_manager import DataParallelRouterManager
        
        # Create args with 3 workers
        args = argparse.Namespace(
            model_dir='/models/llama-7b',
            num_workers=3,
            gpu_ids="0,1,2",
            adapter_dirs=[],
            no_lora=True,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        # Create manager
        manager = DataParallelRouterManager(
            args=args,
            router_port=12345,
            response_port=54321,
            detoken_port=12346
        )
        
        # Verify router was created
        self.assert_not_none(manager.router, "Router should be created")
        
        # Verify router has correct number of workers
        self.assert_equal(manager.router.num_workers, 3, "Router should have 3 workers")
        
        # Test router selection
        worker_id = manager.router.select_worker()
        self.assert_true(0 <= worker_id < 3, f"Worker ID {worker_id} should be in range [0, 3)")
        
        self.logger.info(f"✓ Router created with {manager.router.num_workers} workers")
        self.logger.info(f"✓ Router selection works: selected worker {worker_id}")
        
        self.result.details['router_workers'] = manager.router.num_workers


def main():
    """Run all DataParallelRouterManager initialization tests"""
    print("\n" + "="*70)
    print("Test 8.2.3: DataParallelRouterManager Initialization Test")
    print("="*70)
    print("\nThis test validates Router Manager initialization.")
    print("Tests: creation, GPU detection, parsing, ports, ZMQ, router.\n")
    
    tests = [
        TestManagerCreation(
            "test_manager_creation",
            "Test Manager instance creation"
        ),
        TestGPUDetection(
            "test_gpu_detection",
            "Test GPU auto-detection"
        ),
        TestGPUIDParsing(
            "test_gpu_id_parsing",
            "Test GPU ID parsing from string"
        ),
        TestPortAllocation(
            "test_port_allocation",
            "Test worker port allocation"
        ),
        TestZMQSetup(
            "test_zmq_setup",
            "Test ZMQ socket setup"
        ),
        TestRouterIntegration(
            "test_router_integration",
            "Test Router integration"
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
        print("\n🎉 All Router Manager initialization tests PASSED!")
        print("\nThe DataParallelRouterManager can be initialized correctly:")
        print("  ✓ Manager instance creation")
        print("  ✓ GPU detection")
        print("  ✓ GPU ID parsing")
        print("  ✓ Port allocation")
        print("  ✓ ZMQ socket setup")
        print("  ✓ Router integration")
        print("\nComponent tests (8.2.x) complete!")
        print("Ready to proceed to: 8.3 Process Tests (startup validation)")
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
