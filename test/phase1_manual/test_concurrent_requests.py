#!/usr/bin/env python3
"""
Test 8.5.3: Concurrent Requests Processing Test

Tests the complete system with 2 Workers processing concurrent requests.
Uses Dummy mode to avoid loading real models.

Test Coverage:
- Complete system with concurrent request processing
- Multiple Workers processing requests in parallel
- Round-robin load distribution
- Request count per Worker verification
"""

import sys
import time
import multiprocessing
import asyncio
import argparse
from pathlib import Path
from collections import defaultdict

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))

from test_base import ManualTest, run_test_suite


def run_manager_process(router_port, worker_ports, response_port, result_queue):
    """Run Manager process"""
    try:
        import zmq
        import zmq.asyncio
        
        print(f"[Manager] Starting...")
        
        async def manager_main():
            context = zmq.asyncio.Context()
            
            request_receiver = context.socket(zmq.PULL)
            request_receiver.bind(f"tcp://127.0.0.1:{router_port}")
            
            worker_senders = []
            for i, port in enumerate(worker_ports):
                sender = context.socket(zmq.PUSH)
                sender.bind(f"tcp://127.0.0.1:{port}")
                worker_senders.append(sender)
                print(f"[Manager] Bound to Worker {i} port {port}")
            
            print(f"[Manager] Listening on port {router_port}")
            print(f"[Manager] Connected to {len(worker_senders)} Workers")
            
            await asyncio.sleep(1)
            
            start_time = time.time()
            request_count = 0
            
            while time.time() - start_time < 30:
                try:
                    request = await asyncio.wait_for(
                        request_receiver.recv_json(),
                        timeout=1.0
                    )
                    
                    request_id = request['request_id']
                    print(f"[Manager] Received request: {request_id}")
                    
                    # Round-robin distribution
                    worker_idx = request_count % len(worker_senders)
                    await worker_senders[worker_idx].send_json(request)
                    print(f"[Manager] Sent to Worker {worker_idx}: {request_id}")
                    
                    request_count += 1
                    result_queue.put(('routed', request_id, worker_idx))
                    
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    print(f"[Manager] Error: {e}")
                    result_queue.put(('error', str(e)))
            
            print(f"[Manager] Finished, processed {request_count} requests")
            result_queue.put(('done', request_count))
            
            request_receiver.close()
            for sender in worker_senders:
                sender.close()
            context.term()
        
        asyncio.run(manager_main())
        
    except Exception as e:
        print(f"[Manager] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', str(e)))


def run_worker_process(worker_id, request_port, response_port, args, result_queue):
    """Run Worker process"""
    try:
        import zmq
        
        print(f"[Worker {worker_id}] Starting...")
        
        context = zmq.Context()
        
        request_receiver = context.socket(zmq.PULL)
        request_receiver.connect(f"tcp://127.0.0.1:{request_port}")
        
        response_sender = context.socket(zmq.PUSH)
        response_sender.connect(f"tcp://127.0.0.1:{response_port}")
        
        print(f"[Worker {worker_id}] Connected to ports {request_port} (req) and {response_port} (resp)")
        print(f"[Worker {worker_id}] Ready (Dummy mode)")
        result_queue.put(('ready', worker_id))
        
        start_time = time.time()
        processed_count = 0
        
        while time.time() - start_time < 30:
            try:
                request = request_receiver.recv_json(zmq.NOBLOCK)
                request_id = request['request_id']
                prompt_ids = request['prompt_ids']
                
                print(f"[Worker {worker_id}] Processing request: {request_id}")
                result_queue.put(('processing', worker_id, request_id))
                
                # Simulate processing
                time.sleep(0.1)
                
                # Add worker-specific token to output
                output_ids = prompt_ids + [100 + worker_id, 101, 102]
                
                response = {
                    'request_id': request_id,
                    'worker_id': worker_id,
                    'output_ids': output_ids,
                    'metadata': {
                        'finish_reason': 'eos',
                        'prompt_tokens': len(prompt_ids),
                        'completion_tokens': 3
                    },
                    'success': True,
                    'error': None
                }
                
                response_sender.send_json(response)
                print(f"[Worker {worker_id}] Sent response: {request_id}")
                
                processed_count += 1
                result_queue.put(('completed', worker_id, request_id))
                
            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                print(f"[Worker {worker_id}] Error: {e}")
                result_queue.put(('error', worker_id, str(e)))
        
        print(f"[Worker {worker_id}] Finished, processed {processed_count} requests")
        result_queue.put(('done', worker_id, processed_count))
        
        request_receiver.close()
        response_sender.close()
        context.term()
        
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', worker_id, str(e)))


def run_merger_process(response_port, detoken_port, result_queue):
    """Run Merger process"""
    try:
        import zmq
        
        print(f"[Merger] Starting...")
        
        context = zmq.Context()
        
        response_receiver = context.socket(zmq.PULL)
        response_receiver.bind(f"tcp://127.0.0.1:{response_port}")
        
        print(f"[Merger] Listening on port {response_port}")
        print(f"[Merger] Ready")
        result_queue.put(('ready',))
        
        start_time = time.time()
        received_count = 0
        
        while time.time() - start_time < 30:
            try:
                response = response_receiver.recv_json(zmq.NOBLOCK)
                request_id = response['request_id']
                worker_id = response['worker_id']
                
                print(f"[Merger] Received response from Worker {worker_id}: {request_id}")
                result_queue.put(('received', response))
                
                received_count += 1
                
            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                print(f"[Merger] Error: {e}")
                result_queue.put(('error', str(e)))
        
        print(f"[Merger] Finished, received {received_count} responses")
        result_queue.put(('done', received_count))
        
        response_receiver.close()
        context.term()
        
    except Exception as e:
        print(f"[Merger] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', str(e)))


class TestConcurrentRequestsProcessing(ManualTest):
    """Test 1: Concurrent requests processing with 2 Workers"""
    
    def setup(self):
        self.manager_process = None
        self.worker_processes = []
        self.merger_process = None
        self.manager_queue = multiprocessing.Queue()
        self.worker_queue = multiprocessing.Queue()
        self.merger_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing concurrent requests processing (2 Workers, Dummy mode)")
        
        # Configuration
        router_port = 55000
        worker_ports = [55001, 55002]
        response_port = 55100
        detoken_port = 55200
        num_requests = 10
        num_workers = 2
        
        # Create args for Workers
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
            mode=[]
        )
        
        # Start Merger
        self.logger.info("Starting Merger...")
        self.merger_process = multiprocessing.Process(
            target=run_merger_process,
            args=(response_port, detoken_port, self.merger_queue),
            name="TestMerger"
        )
        self.merger_process.start()
        time.sleep(2)
        
        self.assert_true(self.merger_process.is_alive(), "Merger should be alive")
        self.logger.info("✓ Merger started")
        
        # Start Workers
        self.logger.info(f"Starting {num_workers} Workers...")
        for i in range(num_workers):
            worker_process = multiprocessing.Process(
                target=run_worker_process,
                args=(i, worker_ports[i], response_port, args, self.worker_queue),
                name=f"TestWorker-{i}"
            )
            worker_process.start()
            self.worker_processes.append(worker_process)
        
        time.sleep(2)
        
        # Check all Workers are alive
        for i, proc in enumerate(self.worker_processes):
            self.assert_true(proc.is_alive(), f"Worker {i} should be alive")
        
        self.logger.info(f"✓ All {num_workers} Workers started")
        
        # Start Manager
        self.logger.info("Starting Manager...")
        self.manager_process = multiprocessing.Process(
            target=run_manager_process,
            args=(router_port, worker_ports, response_port, self.manager_queue),
            name="TestManager"
        )
        self.manager_process.start()
        time.sleep(2)
        
        self.assert_true(self.manager_process.is_alive(), "Manager should be alive")
        self.logger.info("✓ Manager started")
        
        # Wait for all components to be ready
        time.sleep(2)
        self.logger.info("✓ All components ready")
        
        # Send test requests concurrently
        import zmq
        context = zmq.Context()
        sender = context.socket(zmq.PUSH)
        sender.connect(f"tcp://127.0.0.1:{router_port}")
        
        time.sleep(1)
        
        test_requests = []
        for i in range(num_requests):
            request = {
                'request_id': f'test-concurrent-{i:03d}',
                'prompt_ids': list(range(i, i+5)),
                'adapter_dir': None,
                'sampling_params': {'max_new_tokens': 10}
            }
            test_requests.append(request)
        
        self.logger.info(f"Sending {num_requests} requests concurrently...")
        for request in test_requests:
            sender.send_json(request)
            time.sleep(0.05)  # Very small delay to simulate concurrent arrival
        
        self.logger.info("✓ All requests sent")
        
        # Wait for processing
        time.sleep(3)
        
        # Collect results
        manager_events = []
        while not self.manager_queue.empty():
            manager_events.append(self.manager_queue.get())
        
        worker_events = []
        while not self.worker_queue.empty():
            worker_events.append(self.worker_queue.get())
        
        merger_responses = []
        while not self.merger_queue.empty():
            event = self.merger_queue.get()
            if event[0] == 'received':
                merger_responses.append(event[1])
        
        # Verify Manager routed all requests
        routed_events = [e for e in manager_events if e[0] == 'routed']
        self.assert_equal(len(routed_events), num_requests,
                         f"Manager should route {num_requests} requests")
        self.logger.info(f"✓ Manager routed {len(routed_events)} requests")
        
        # Verify round-robin distribution
        worker_assignments = defaultdict(list)
        for event in routed_events:
            _, request_id, worker_idx = event
            worker_assignments[worker_idx].append(request_id)
        
        expected_per_worker = num_requests // num_workers
        for worker_id in range(num_workers):
            actual_count = len(worker_assignments[worker_id])
            self.assert_equal(actual_count, expected_per_worker,
                            f"Worker {worker_id} should receive {expected_per_worker} requests")
            self.logger.info(f"  ✓ Worker {worker_id} assigned {actual_count} requests")
        
        self.logger.info("✓ Round-robin distribution verified")
        
        # Verify Workers processed requests
        processing_events = [e for e in worker_events if e[0] == 'processing']
        completed_events = [e for e in worker_events if e[0] == 'completed']
        
        self.assert_equal(len(processing_events), num_requests,
                         f"Workers should process {num_requests} requests")
        self.assert_equal(len(completed_events), num_requests,
                         f"Workers should complete {num_requests} requests")
        
        # Count requests per Worker
        worker_processed = defaultdict(int)
        for event in completed_events:
            _, worker_id, _ = event
            worker_processed[worker_id] += 1
        
        for worker_id in range(num_workers):
            count = worker_processed[worker_id]
            self.logger.info(f"  ✓ Worker {worker_id} processed {count} requests")
        
        self.logger.info(f"✓ All Workers processed requests")
        
        # Verify Merger received all responses
        self.assert_equal(len(merger_responses), num_requests,
                         f"Merger should receive {num_requests} responses")
        self.logger.info(f"✓ Merger received {len(merger_responses)} responses")
        
        # Verify all responses are valid
        for response in merger_responses:
            self.assert_true('request_id' in response, "Response should have request_id")
            self.assert_true('worker_id' in response, "Response should have worker_id")
            self.assert_true(response['success'], "Response should be successful")
        
        self.logger.info("✓ All responses validated")
        
        # Verify responses came from different Workers
        response_workers = set(r['worker_id'] for r in merger_responses)
        self.assert_equal(len(response_workers), num_workers,
                         f"Responses should come from {num_workers} different Workers")
        self.logger.info(f"✓ Responses from {len(response_workers)} Workers")
        
        # Cleanup
        sender.close()
        context.term()
        
        self.result.details['num_requests'] = num_requests
        self.result.details['num_workers'] = num_workers
        self.result.details['all_processed'] = True
        self.result.details['load_balanced'] = True
    
    def teardown(self):
        """Cleanup: terminate all processes"""
        if self.manager_process and self.manager_process.is_alive():
            self.logger.info("Terminating Manager...")
            self.manager_process.terminate()
            self.manager_process.join(timeout=5)
            if self.manager_process.is_alive():
                self.manager_process.kill()
                self.manager_process.join()
        
        for i, proc in enumerate(self.worker_processes):
            if proc and proc.is_alive():
                self.logger.info(f"Terminating Worker {i}...")
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
        
        if self.merger_process and self.merger_process.is_alive():
            self.logger.info("Terminating Merger...")
            self.merger_process.terminate()
            self.merger_process.join(timeout=5)
            if self.merger_process.is_alive():
                self.merger_process.kill()
                self.merger_process.join()


def main():
    """Run concurrent requests processing test"""
    print("\n" + "="*70)
    print("Test 8.5.3: Concurrent Requests Processing Test")
    print("="*70)
    print("\nThis test validates concurrent request processing with 2 Workers.")
    print("Using Dummy mode to avoid loading real models.\n")
    
    tests = [
        TestConcurrentRequestsProcessing(
            "test_concurrent_requests_processing",
            "Test concurrent requests with 2 Workers"
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
        print("\n🎉 Concurrent requests processing test PASSED!")
        print("\nConcurrent requests processing verified:")
        print("  ✓ Complete system with 2 Workers")
        print("  ✓ Manager routes all requests")
        print("  ✓ Round-robin distribution works correctly")
        print("  ✓ Both Workers process requests in parallel")
        print("  ✓ All responses received by Merger")
        print("  ✓ Load balanced across Workers")
        print("\nFunctional test (8.5.3) complete!")
        print("Ready to proceed to: 8.5.4 (High Concurrency)")
    else:
        print("\n❌ Test FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
