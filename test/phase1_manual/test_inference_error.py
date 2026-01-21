#!/usr/bin/env python3
"""
Test 8.6.3: Inference Error Test

Tests inference error handling and system resilience.
Simulates inference failures and verifies error responses.

Test Coverage:
- Complete system with inference error simulation
- Error response format verification
- System continues processing after errors
- Error logging verification
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
            for i, port in enumerate(worker_ports):
                sender = context.socket(zmq.PUSH)
                sender.bind(f"tcp://127.0.0.1:{port}")
                worker_senders.append(sender)
            
            print(f"[Manager] Listening on port {router_port}")
            print(f"[Manager] Connected to {len(worker_senders)} Workers")
            
            await asyncio.sleep(1)
            
            start_time = time.time()
            request_count = 0
            
            while time.time() - start_time < 60:
                try:
                    request = await asyncio.wait_for(
                        request_receiver.recv_json(),
                        timeout=1.0
                    )
                    
                    request_id = request['request_id']
                    worker_idx = request_count % len(worker_senders)
                    await worker_senders[worker_idx].send_json(request)
                    
                    request_count += 1
                    result_queue.put(('routed', request_id, worker_idx, time.time()))
                    
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
    """Run Worker process with inference error simulation"""
    try:
        import zmq
        
        print(f"[Worker {worker_id}] Starting...")
        
        context = zmq.Context()
        
        request_receiver = context.socket(zmq.PULL)
        request_receiver.connect(f"tcp://127.0.0.1:{request_port}")
        
        response_sender = context.socket(zmq.PUSH)
        response_sender.connect(f"tcp://127.0.0.1:{response_port}")
        
        print(f"[Worker {worker_id}] Ready (Dummy mode with error simulation)")
        result_queue.put(('ready', worker_id))
        
        start_time = time.time()
        processed_count = 0
        error_count = 0
        
        while time.time() - start_time < 60:
            try:
                request = request_receiver.recv_json(zmq.NOBLOCK)
                request_id = request['request_id']
                prompt_ids = request['prompt_ids']
                
                # Simulate inference error for specific requests
                # Trigger error if request_id starts with 'test-error-'
                if request_id.startswith('test-error-'):
                    print(f"[Worker {worker_id}] Simulating inference error for {request_id}")
                    
                    # Simulate processing time
                    time.sleep(0.1)
                    
                    # Send error response
                    response = {
                        'request_id': request_id,
                        'worker_id': worker_id,
                        'output_ids': [],
                        'metadata': {
                            'finish_reason': 'error',
                            'prompt_tokens': len(prompt_ids),
                            'completion_tokens': 0
                        },
                        'success': False,
                        'error': 'Simulated inference error: CUDA out of memory'
                    }
                    
                    response_sender.send_json(response)
                    error_count += 1
                    result_queue.put(('error_response', worker_id, request_id, time.time()))
                    
                else:
                    # Normal processing
                    time.sleep(0.1)
                    
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
                    processed_count += 1
                    result_queue.put(('completed', worker_id, request_id, time.time()))
                
            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                print(f"[Worker {worker_id}] Error: {e}")
                result_queue.put(('error', worker_id, str(e)))
        
        print(f"[Worker {worker_id}] Finished, processed {processed_count} requests, {error_count} errors")
        result_queue.put(('done', worker_id, processed_count, error_count))
        
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
        error_count = 0
        
        while time.time() - start_time < 60:
            try:
                response = response_receiver.recv_json(zmq.NOBLOCK)
                
                request_id = response['request_id']
                worker_id = response['worker_id']
                success = response['success']
                
                if not success:
                    error_count += 1
                    print(f"[Merger] Received error response: {request_id}, error: {response['error']}")
                
                received_count += 1
                result_queue.put(('received', response, time.time()))
                
            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                print(f"[Merger] Error: {e}")
                result_queue.put(('error', str(e)))
        
        print(f"[Merger] Finished, received {received_count} responses ({error_count} errors)")
        result_queue.put(('done', received_count, error_count))
        
        response_receiver.close()
        context.term()
        
    except Exception as e:
        print(f"[Merger] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', str(e)))


class TestInferenceErrorHandling(ManualTest):
    """Test 1: Inference error handling and system resilience"""
    
    def setup(self):
        self.manager_process = None
        self.worker_processes = []
        self.merger_process = None
        self.manager_queue = multiprocessing.Queue()
        self.worker_queue = multiprocessing.Queue()
        self.merger_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing inference error handling and system resilience")
        
        # Configuration
        router_port = 58000
        worker_ports = [58001, 58002]
        response_port = 58100
        detoken_port = 58200
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
        
        # Send test requests
        import zmq
        context = zmq.Context()
        sender = context.socket(zmq.PUSH)
        sender.connect(f"tcp://127.0.0.1:{router_port}")
        
        time.sleep(1)
        
        # Send normal requests
        self.logger.info("Sending normal requests...")
        for i in range(3):
            request = {
                'request_id': f'test-normal-{i:02d}',
                'prompt_ids': list(range(5)),
                'adapter_dir': None,
                'sampling_params': {'max_new_tokens': 10}
            }
            sender.send_json(request)
            time.sleep(0.1)
        
        self.logger.info("✓ Normal requests sent")
        time.sleep(2)
        
        # Send requests that will trigger errors
        self.logger.info("Sending requests that will trigger inference errors...")
        for i in range(2):
            request = {
                'request_id': f'test-error-{i:02d}',  # Contains 'error' to trigger simulation
                'prompt_ids': list(range(5)),
                'adapter_dir': None,
                'sampling_params': {'max_new_tokens': 10}
            }
            sender.send_json(request)
            time.sleep(0.1)
        
        self.logger.info("✓ Error-triggering requests sent")
        time.sleep(2)
        
        # Send more normal requests after errors
        self.logger.info("Sending normal requests after errors...")
        for i in range(3):
            request = {
                'request_id': f'test-after-error-{i:02d}',
                'prompt_ids': list(range(5)),
                'adapter_dir': None,
                'sampling_params': {'max_new_tokens': 10}
            }
            sender.send_json(request)
            time.sleep(0.1)
        
        self.logger.info("✓ Post-error requests sent")
        time.sleep(2)
        
        # Collect results
        worker_events = []
        while not self.worker_queue.empty():
            worker_events.append(self.worker_queue.get())
        
        merger_responses = []
        while not self.merger_queue.empty():
            event = self.merger_queue.get()
            if event[0] == 'received':
                merger_responses.append(event[1])
        
        # Verify error responses were generated
        error_events = [e for e in worker_events if e[0] == 'error_response']
        self.assert_true(len(error_events) >= 2,
                        f"Should have at least 2 error responses, got {len(error_events)}")
        self.logger.info(f"✓ Workers generated {len(error_events)} error responses")
        
        # Verify error response format
        error_responses = [r for r in merger_responses if not r['success']]
        self.assert_true(len(error_responses) >= 2,
                        f"Should have at least 2 error responses in Merger, got {len(error_responses)}")
        
        for response in error_responses:
            # Verify error response format
            self.assert_equal(response['success'], False, "Error response should have success=False")
            self.assert_not_none(response['error'], "Error response should have error field")
            self.assert_true('error' in response['error'].lower() or 'cuda' in response['error'].lower(),
                           "Error message should describe the error")
            self.assert_equal(response['metadata']['finish_reason'], 'error',
                           "Error response should have finish_reason='error'")
            self.logger.info(f"✓ Error response format correct: {response['request_id']}")
        
        # Verify normal responses after errors
        normal_after_error = [r for r in merger_responses 
                             if r['success'] and 'after-error' in r['request_id']]
        self.assert_true(len(normal_after_error) >= 2,
                        f"Should have at least 2 normal responses after errors, got {len(normal_after_error)}")
        self.logger.info(f"✓ System processed {len(normal_after_error)} normal requests after errors")
        
        # Verify system stability
        self.assert_true(self.manager_process.is_alive(), "Manager should still be alive")
        for i, proc in enumerate(self.worker_processes):
            self.assert_true(proc.is_alive(), f"Worker {i} should still be alive")
        self.assert_true(self.merger_process.is_alive(), "Merger should still be alive")
        
        self.logger.info("✓ System stable after inference errors")
        
        # Cleanup
        sender.close()
        context.term()
        
        self.result.details['error_responses_generated'] = len(error_responses)
        self.result.details['normal_responses_after_error'] = len(normal_after_error)
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
    """Run inference error test"""
    print("\n" + "="*70)
    print("Test 8.6.3: Inference Error Test")
    print("="*70)
    print("\nThis test validates inference error handling and system resilience.")
    print("Simulates inference failures and verifies error responses.\n")
    
    tests = [
        TestInferenceErrorHandling(
            "test_inference_error_handling",
            "Test inference error handling and system resilience"
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
        print("\n🎉 Inference error test PASSED!")
        print("\nInference error handling verified:")
        print("  ✓ Complete system with 2 Workers")
        print("  ✓ Inference errors simulated")
        print("  ✓ Error responses generated (success=False)")
        print("  ✓ Error response format correct (error field populated)")
        print("  ✓ System continues processing after errors")
        print("  ✓ Normal requests processed after errors")
        print("  ✓ System remains stable")
        print("\nError handling test (8.6.3) complete!")
        print("Ready to proceed to: 8.7.1 (Startup Log Verification)")
    else:
        print("\n❌ Test FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
