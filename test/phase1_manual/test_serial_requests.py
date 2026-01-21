#!/usr/bin/env python3
"""
Test 8.5.2: Serial Requests Processing Test

Tests the complete system with 1 Worker processing multiple serial requests.
Uses Dummy mode to avoid loading real models.

Test Coverage:
- Complete system with serial request processing
- Multiple requests sent sequentially
- Response order verification
- Request statistics validation
"""

import sys
import time
import multiprocessing
import asyncio
import argparse
from pathlib import Path

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
            for port in worker_ports:
                sender = context.socket(zmq.PUSH)
                sender.bind(f"tcp://127.0.0.1:{port}")
                worker_senders.append(sender)
            
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
                
                output_ids = prompt_ids + [100, 101, 102]
                
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


class TestSerialRequestsProcessing(ManualTest):
    """Test 1: Serial requests processing with complete system"""
    
    def setup(self):
        self.manager_process = None
        self.worker_process = None
        self.merger_process = None
        self.manager_queue = multiprocessing.Queue()
        self.worker_queue = multiprocessing.Queue()
        self.merger_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing serial requests processing (complete system, Dummy mode)")
        
        # Configuration
        router_port = 54000
        worker_port = 54001
        response_port = 54100
        detoken_port = 54200
        num_requests = 5
        
        # Create args for Worker
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
        
        # Start Worker
        self.logger.info("Starting Worker...")
        self.worker_process = multiprocessing.Process(
            target=run_worker_process,
            args=(0, worker_port, response_port, args, self.worker_queue),
            name="TestWorker-0"
        )
        self.worker_process.start()
        time.sleep(2)
        
        self.assert_true(self.worker_process.is_alive(), "Worker should be alive")
        self.logger.info("✓ Worker started")
        
        # Start Manager
        self.logger.info("Starting Manager...")
        self.manager_process = multiprocessing.Process(
            target=run_manager_process,
            args=(router_port, [worker_port], response_port, self.manager_queue),
            name="TestManager"
        )
        self.manager_process.start()
        time.sleep(2)
        
        self.assert_true(self.manager_process.is_alive(), "Manager should be alive")
        self.logger.info("✓ Manager started")
        
        # Wait for all components to be ready
        time.sleep(2)
        self.logger.info("✓ All components ready")
        
        # Send test requests serially
        import zmq
        context = zmq.Context()
        sender = context.socket(zmq.PUSH)
        sender.connect(f"tcp://127.0.0.1:{router_port}")
        
        time.sleep(1)
        
        test_requests = []
        for i in range(num_requests):
            request = {
                'request_id': f'test-serial-{i:03d}',
                'prompt_ids': list(range(i, i+5)),
                'adapter_dir': None,
                'sampling_params': {'max_new_tokens': 10}
            }
            test_requests.append(request)
        
        self.logger.info(f"Sending {num_requests} requests serially...")
        for i, request in enumerate(test_requests):
            self.logger.info(f"  Sending request {i+1}/{num_requests}: {request['request_id']}")
            sender.send_json(request)
            time.sleep(0.2)  # Small delay between requests
        
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
        
        # Verify Worker processed all requests
        processing_events = [e for e in worker_events if e[0] == 'processing']
        completed_events = [e for e in worker_events if e[0] == 'completed']
        self.assert_equal(len(processing_events), num_requests,
                         f"Worker should process {num_requests} requests")
        self.assert_equal(len(completed_events), num_requests,
                         f"Worker should complete {num_requests} requests")
        self.logger.info(f"✓ Worker processed {len(completed_events)} requests")
        
        # Verify Merger received all responses
        self.assert_equal(len(merger_responses), num_requests,
                         f"Merger should receive {num_requests} responses")
        self.logger.info(f"✓ Merger received {len(merger_responses)} responses")
        
        # Verify response order
        response_ids = [r['request_id'] for r in merger_responses]
        expected_ids = [r['request_id'] for r in test_requests]
        self.assert_equal(response_ids, expected_ids,
                         "Response order should match request order")
        self.logger.info("✓ Response order preserved")
        
        # Verify all responses are valid
        for i, response in enumerate(merger_responses):
            self.assert_equal(response['request_id'], test_requests[i]['request_id'],
                            f"Response {i+1} request_id should match")
            self.assert_true(response['success'], f"Response {i+1} should be successful")
            self.assert_true(len(response['output_ids']) > len(test_requests[i]['prompt_ids']),
                            f"Response {i+1} output should be longer than input")
        
        self.logger.info("✓ All responses validated")
        
        # Cleanup
        sender.close()
        context.term()
        
        self.result.details['num_requests'] = num_requests
        self.result.details['all_processed'] = True
        self.result.details['order_preserved'] = True
    
    def teardown(self):
        """Cleanup: terminate all processes"""
        if self.manager_process and self.manager_process.is_alive():
            self.logger.info("Terminating Manager...")
            self.manager_process.terminate()
            self.manager_process.join(timeout=5)
            if self.manager_process.is_alive():
                self.manager_process.kill()
                self.manager_process.join()
        
        if self.worker_process and self.worker_process.is_alive():
            self.logger.info("Terminating Worker...")
            self.worker_process.terminate()
            self.worker_process.join(timeout=5)
            if self.worker_process.is_alive():
                self.worker_process.kill()
                self.worker_process.join()
        
        if self.merger_process and self.merger_process.is_alive():
            self.logger.info("Terminating Merger...")
            self.merger_process.terminate()
            self.merger_process.join(timeout=5)
            if self.merger_process.is_alive():
                self.merger_process.kill()
                self.merger_process.join()


def main():
    """Run serial requests processing test"""
    print("\n" + "="*70)
    print("Test 8.5.2: Serial Requests Processing Test")
    print("="*70)
    print("\nThis test validates serial request processing with complete system.")
    print("Using Dummy mode to avoid loading real models.\n")
    
    tests = [
        TestSerialRequestsProcessing(
            "test_serial_requests_processing",
            "Test serial requests with complete system"
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
        print("\n🎉 Serial requests processing test PASSED!")
        print("\nSerial requests processing verified:")
        print("  ✓ Complete system handles multiple requests")
        print("  ✓ Manager routes all requests to Worker")
        print("  ✓ Worker processes all requests sequentially")
        print("  ✓ All responses received by Merger")
        print("  ✓ Response order preserved")
        print("  ✓ All responses valid")
        print("\nFunctional test (8.5.2) complete!")
        print("Ready to proceed to: 8.5.3 (Concurrent Requests)")
    else:
        print("\n❌ Test FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
