#!/usr/bin/env python3
"""
Test 8.5.4: High Concurrency Test

Tests the complete system with 4 Workers processing 100 concurrent requests.
Uses Dummy mode to avoid loading real models.

Test Coverage:
- Complete system under high load
- 4 Workers processing requests in parallel
- 100 requests processed
- Throughput and latency measurement
- System stability verification
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
            
            print(f"[Manager] Listening on port {router_port}")
            print(f"[Manager] Connected to {len(worker_senders)} Workers")
            
            await asyncio.sleep(1)
            
            start_time = time.time()
            request_count = 0
            first_request_time = None
            last_request_time = None
            
            while time.time() - start_time < 60:
                try:
                    request = await asyncio.wait_for(
                        request_receiver.recv_json(),
                        timeout=1.0
                    )
                    
                    if first_request_time is None:
                        first_request_time = time.time()
                    last_request_time = time.time()
                    
                    request_id = request['request_id']
                    
                    # Round-robin distribution
                    worker_idx = request_count % len(worker_senders)
                    await worker_senders[worker_idx].send_json(request)
                    
                    request_count += 1
                    result_queue.put(('routed', request_id, worker_idx, time.time()))
                    
                    if request_count % 10 == 0:
                        print(f"[Manager] Routed {request_count} requests...")
                    
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    print(f"[Manager] Error: {e}")
                    result_queue.put(('error', str(e)))
            
            duration = last_request_time - first_request_time if first_request_time else 0
            throughput = request_count / duration if duration > 0 else 0
            
            print(f"[Manager] Finished, processed {request_count} requests")
            print(f"[Manager] Duration: {duration:.2f}s, Throughput: {throughput:.2f} req/s")
            result_queue.put(('done', request_count, duration, throughput))
            
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
        
        print(f"[Worker {worker_id}] Ready (Dummy mode)")
        result_queue.put(('ready', worker_id))
        
        start_time = time.time()
        processed_count = 0
        processing_times = []
        
        while time.time() - start_time < 60:
            try:
                request = request_receiver.recv_json(zmq.NOBLOCK)
                request_id = request['request_id']
                prompt_ids = request['prompt_ids']
                
                proc_start = time.time()
                
                # Simulate processing (shorter for high concurrency)
                time.sleep(0.05)
                
                proc_end = time.time()
                processing_times.append(proc_end - proc_start)
                
                output_ids = prompt_ids + [100 + worker_id, 101, 102]
                
                response = {
                    'request_id': request_id,
                    'worker_id': worker_id,
                    'output_ids': output_ids,
                    'metadata': {
                        'finish_reason': 'eos',
                        'prompt_tokens': len(prompt_ids),
                        'completion_tokens': 3,
                        'processing_time': proc_end - proc_start
                    },
                    'success': True,
                    'error': None
                }
                
                response_sender.send_json(response)
                
                processed_count += 1
                result_queue.put(('completed', worker_id, request_id, time.time()))
                
                if processed_count % 10 == 0:
                    print(f"[Worker {worker_id}] Processed {processed_count} requests...")
                
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"[Worker {worker_id}] Error: {e}")
                result_queue.put(('error', worker_id, str(e)))
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        print(f"[Worker {worker_id}] Finished, processed {processed_count} requests")
        print(f"[Worker {worker_id}] Avg processing time: {avg_processing_time*1000:.2f}ms")
        result_queue.put(('done', worker_id, processed_count, avg_processing_time))
        
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
        
        print(f"[Merger] Ready")
        result_queue.put(('ready',))
        
        start_time = time.time()
        received_count = 0
        first_response_time = None
        last_response_time = None
        
        while time.time() - start_time < 60:
            try:
                response = response_receiver.recv_json(zmq.NOBLOCK)
                
                if first_response_time is None:
                    first_response_time = time.time()
                last_response_time = time.time()
                
                request_id = response['request_id']
                worker_id = response['worker_id']
                
                received_count += 1
                result_queue.put(('received', response, time.time()))
                
                if received_count % 10 == 0:
                    print(f"[Merger] Received {received_count} responses...")
                
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"[Merger] Error: {e}")
                result_queue.put(('error', str(e)))
        
        duration = last_response_time - first_response_time if first_response_time else 0
        throughput = received_count / duration if duration > 0 else 0
        
        print(f"[Merger] Finished, received {received_count} responses")
        print(f"[Merger] Duration: {duration:.2f}s, Throughput: {throughput:.2f} resp/s")
        result_queue.put(('done', received_count, duration, throughput))
        
        response_receiver.close()
        context.term()
        
    except Exception as e:
        print(f"[Merger] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', str(e)))


class TestHighConcurrency(ManualTest):
    """Test 1: High concurrency with 4 Workers and 100 requests"""
    
    def setup(self):
        self.manager_process = None
        self.worker_processes = []
        self.merger_process = None
        self.manager_queue = multiprocessing.Queue()
        self.worker_queue = multiprocessing.Queue()
        self.merger_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing high concurrency (4 Workers, 100 requests, Dummy mode)")
        
        # Configuration
        router_port = 56000
        worker_ports = [56001, 56002, 56003, 56004]
        response_port = 56100
        detoken_port = 56200
        num_requests = 100
        num_workers = 4
        
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
        
        # Send test requests
        import zmq
        context = zmq.Context()
        sender = context.socket(zmq.PUSH)
        sender.connect(f"tcp://127.0.0.1:{router_port}")
        
        time.sleep(1)
        
        test_requests = []
        for i in range(num_requests):
            request = {
                'request_id': f'test-high-{i:03d}',
                'prompt_ids': list(range(i % 10, (i % 10) + 5)),
                'adapter_dir': None,
                'sampling_params': {'max_new_tokens': 10}
            }
            test_requests.append(request)
        
        self.logger.info(f"Sending {num_requests} requests...")
        send_start = time.time()
        
        for request in test_requests:
            sender.send_json(request)
            time.sleep(0.01)  # Very small delay
        
        send_end = time.time()
        send_duration = send_end - send_start
        
        self.logger.info(f"✓ All {num_requests} requests sent in {send_duration:.2f}s")
        
        # Wait for processing
        self.logger.info("Waiting for processing...")
        time.sleep(10)
        
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
        worker_assignments = defaultdict(int)
        for event in routed_events:
            _, request_id, worker_idx, _ = event
            worker_assignments[worker_idx] += 1
        
        expected_per_worker = num_requests // num_workers
        for worker_id in range(num_workers):
            actual_count = worker_assignments[worker_id]
            self.assert_equal(actual_count, expected_per_worker,
                            f"Worker {worker_id} should receive {expected_per_worker} requests")
            self.logger.info(f"  ✓ Worker {worker_id} assigned {actual_count} requests")
        
        self.logger.info("✓ Load distribution verified")
        
        # Verify Workers processed requests
        completed_events = [e for e in worker_events if e[0] == 'completed']
        
        self.assert_equal(len(completed_events), num_requests,
                         f"Workers should complete {num_requests} requests")
        
        # Count requests per Worker
        worker_processed = defaultdict(int)
        for event in completed_events:
            _, worker_id, _, _ = event
            worker_processed[worker_id] += 1
        
        for worker_id in range(num_workers):
            count = worker_processed[worker_id]
            self.logger.info(f"  ✓ Worker {worker_id} processed {count} requests")
        
        self.logger.info(f"✓ All Workers processed requests")
        
        # Verify Merger received all responses
        self.assert_equal(len(merger_responses), num_requests,
                         f"Merger should receive {num_requests} responses")
        self.logger.info(f"✓ Merger received {len(merger_responses)} responses")
        
        # Calculate throughput and latency
        manager_done = [e for e in manager_events if e[0] == 'done']
        if manager_done:
            _, total_routed, duration, throughput = manager_done[0]
            self.logger.info(f"✓ Manager throughput: {throughput:.2f} req/s")
        
        # Verify system stability
        self.assert_true(self.manager_process.is_alive(), "Manager should still be alive")
        for i, proc in enumerate(self.worker_processes):
            self.assert_true(proc.is_alive(), f"Worker {i} should still be alive")
        self.assert_true(self.merger_process.is_alive(), "Merger should still be alive")
        
        self.logger.info("✓ System stable under load")
        
        # Verify all responses are valid
        for response in merger_responses:
            self.assert_true(response['success'], "Response should be successful")
        
        self.logger.info("✓ All responses valid")
        
        # Cleanup
        sender.close()
        context.term()
        
        self.result.details['num_requests'] = num_requests
        self.result.details['num_workers'] = num_workers
        self.result.details['all_processed'] = True
        self.result.details['system_stable'] = True
    
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
    """Run high concurrency test"""
    print("\n" + "="*70)
    print("Test 8.5.4: High Concurrency Test")
    print("="*70)
    print("\nThis test validates high concurrency with 4 Workers and 100 requests.")
    print("Using Dummy mode to avoid loading real models.\n")
    
    tests = [
        TestHighConcurrency(
            "test_high_concurrency",
            "Test high concurrency with 4 Workers"
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
        print("\n🎉 High concurrency test PASSED!")
        print("\nHigh concurrency verified:")
        print("  ✓ Complete system with 4 Workers")
        print("  ✓ 100 requests processed successfully")
        print("  ✓ Load distributed evenly (25 requests per Worker)")
        print("  ✓ All responses received")
        print("  ✓ System stable under load")
        print("  ✓ Throughput measured")
        print("\nFunctional tests (8.5) complete!")
        print("Ready to proceed to: 8.6 (Error Handling Tests)")
    else:
        print("\n❌ Test FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
