#!/usr/bin/env python3
"""
Test 8.4.3: Full Communication Chain Test

Tests the complete communication flow:
Manager → Worker → Merger

Test Coverage:
- Manager sends requests to multiple Workers
- Workers receive and process requests
- Workers send responses to Merger
- Merger receives all responses
- Round-robin routing works correctly
- All messages are delivered correctly
"""

import sys
import time
import multiprocessing
import asyncio
import zmq
import zmq.asyncio
from pathlib import Path
from collections import defaultdict

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))

from test_base import ManualTest, run_test_suite


def simple_worker_process(worker_id, request_port, response_port, result_queue):
    """
    Simple Worker that receives requests and sends responses
    
    This simulates a Worker that:
    1. Receives requests from Manager
    2. Processes them (adds worker_id to response)
    3. Sends responses to Merger
    """
    try:
        print(f"[Worker {worker_id}] Starting...")
        
        # Create ZMQ context
        context = zmq.Context()
        
        # Receive requests from Manager
        request_receiver = context.socket(zmq.PULL)
        request_receiver.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        request_receiver.connect(f"tcp://127.0.0.1:{request_port}")
        
        # Send responses to Merger
        response_sender = context.socket(zmq.PUSH)
        response_sender.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
        response_sender.connect(f"tcp://127.0.0.1:{response_port}")
        
        print(f"[Worker {worker_id}] Connected to request port {request_port} and response port {response_port}")
        print(f"[Worker {worker_id}] Ready to process requests...")
        
        # Process requests for 15 seconds
        start_time = time.time()
        processed_count = 0
        
        while time.time() - start_time < 15:
            try:
                # Receive request
                request = request_receiver.recv_json()
                processed_count += 1
                request_id = request['request_id']
                
                print(f"[Worker {worker_id}] Received request: {request_id}")
                result_queue.put(('received', worker_id, request_id))
                
                # Create response
                response = {
                    'request_id': request_id,
                    'worker_id': worker_id,
                    'output_ids': request['prompt_ids'] + [100 + worker_id],  # Add worker-specific token
                    'metadata': {
                        'finish_reason': 'eos',
                        'prompt_tokens': len(request['prompt_ids']),
                        'completion_tokens': 1
                    },
                    'success': True,
                    'error': None
                }
                
                # Send response
                response_sender.send_json(response)
                print(f"[Worker {worker_id}] Sent response: {request_id}")
                result_queue.put(('sent', worker_id, request_id))
                
            except zmq.Again:
                # Timeout, continue waiting
                pass
            except Exception as e:
                print(f"[Worker {worker_id}] Error: {e}")
                result_queue.put(('error', worker_id, str(e)))
                break
        
        print(f"[Worker {worker_id}] Finished, processed {processed_count} requests")
        result_queue.put(('done', worker_id, processed_count))
        
        # Cleanup
        request_receiver.close()
        response_sender.close()
        context.term()
        
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', worker_id, str(e)))


def simple_merger_process(response_port, result_queue):
    """
    Simple Merger that receives responses from Workers
    """
    try:
        print(f"[Merger] Starting...")
        
        # Create ZMQ context
        context = zmq.Context()
        
        # Receive responses from Workers
        receiver = context.socket(zmq.PULL)
        receiver.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        receiver.bind(f"tcp://127.0.0.1:{response_port}")
        
        print(f"[Merger] Bound to port {response_port}")
        print(f"[Merger] Waiting for responses...")
        
        # Receive responses for 15 seconds
        start_time = time.time()
        received_count = 0
        
        while time.time() - start_time < 15:
            try:
                response = receiver.recv_json()
                received_count += 1
                request_id = response['request_id']
                worker_id = response['worker_id']
                
                print(f"[Merger] Received response from Worker {worker_id}: {request_id}")
                result_queue.put(('received', response))
                
            except zmq.Again:
                # Timeout, continue waiting
                pass
            except Exception as e:
                print(f"[Merger] Error: {e}")
                result_queue.put(('error', str(e)))
                break
        
        print(f"[Merger] Finished, received {received_count} responses")
        result_queue.put(('done', received_count))
        
        # Cleanup
        receiver.close()
        context.term()
        
    except Exception as e:
        print(f"[Merger] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', str(e)))


async def simple_manager_sender(worker_ports, requests):
    """
    Simple Manager that sends requests to Workers using round-robin
    
    Args:
        worker_ports: List of Worker ports
        requests: List of requests to send
    
    Returns:
        Number of requests sent successfully
    """
    try:
        print(f"[Manager] Starting...")
        
        # Create ZMQ context
        context = zmq.asyncio.Context()
        
        # Create PUSH sockets for each Worker
        senders = []
        for i, port in enumerate(worker_ports):
            sender = context.socket(zmq.PUSH)
            sender.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
            sender.bind(f"tcp://127.0.0.1:{port}")
            senders.append(sender)
            print(f"[Manager] Bound to Worker {i} port {port}")
        
        # Wait for Workers to connect
        await asyncio.sleep(2)
        
        # Send requests using round-robin
        sent_count = 0
        for i, request in enumerate(requests):
            try:
                worker_idx = i % len(senders)
                request_id = request['request_id']
                
                print(f"[Manager] Sending request {request_id} to Worker {worker_idx}")
                await senders[worker_idx].send_json(request)
                sent_count += 1
                await asyncio.sleep(0.1)  # Small delay between requests
                
            except Exception as e:
                print(f"[Manager] Error sending request #{i+1}: {e}")
                break
        
        print(f"[Manager] Finished, sent {sent_count}/{len(requests)} requests")
        
        # Cleanup
        for sender in senders:
            sender.close()
        context.term()
        
        return sent_count
        
    except Exception as e:
        print(f"[Manager] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 0


class TestFullCommunicationBasic(ManualTest):
    """Test 1: Basic full communication chain with 2 Workers"""
    
    def setup(self):
        self.worker_processes = []
        self.merger_process = None
        self.worker_queue = multiprocessing.Queue()
        self.merger_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing full communication chain (Manager → Workers → Merger)")
        
        # Configuration
        num_workers = 2
        worker_ports = [52000, 52001]
        response_port = 52100
        num_requests = 10
        
        # Start Merger
        self.logger.info(f"Starting Merger on port {response_port}...")
        self.merger_process = multiprocessing.Process(
            target=simple_merger_process,
            args=(response_port, self.merger_queue),
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
                target=simple_worker_process,
                args=(i, worker_ports[i], response_port, self.worker_queue),
                name=f"TestWorker-{i}"
            )
            worker_process.start()
            self.worker_processes.append(worker_process)
        
        time.sleep(2)
        
        # Check all Workers are alive
        for i, proc in enumerate(self.worker_processes):
            self.assert_true(proc.is_alive(), f"Worker {i} should be alive")
        
        self.logger.info(f"✓ All {num_workers} Workers started")
        
        # Prepare test requests
        test_requests = []
        for i in range(num_requests):
            test_requests.append({
                'request_id': f'req-{i:03d}',
                'prompt_ids': [1, 2, 3, 4, 5],
                'adapter_dir': None,
                'sampling_params': {'max_new_tokens': 10}
            })
        
        # Send requests from Manager
        self.logger.info(f"Sending {num_requests} requests from Manager...")
        sent_count = asyncio.run(simple_manager_sender(worker_ports, test_requests))
        
        self.assert_equal(sent_count, num_requests,
                         f"Should send all {num_requests} requests")
        self.logger.info(f"✓ Manager sent {sent_count} requests")
        
        # Wait for processing
        time.sleep(3)
        
        # Collect Worker results
        worker_received = defaultdict(list)
        worker_sent = defaultdict(list)
        
        while not self.worker_queue.empty():
            event = self.worker_queue.get()
            event_type = event[0]
            if event_type == 'received':
                _, worker_id, request_id = event
                worker_received[worker_id].append(request_id)
            elif event_type == 'sent':
                _, worker_id, request_id = event
                worker_sent[worker_id].append(request_id)
        
        # Collect Merger results
        merger_responses = []
        merger_errors = []
        
        while not self.merger_queue.empty():
            event = self.merger_queue.get()
            event_type = event[0]
            if event_type == 'received':
                _, response = event
                merger_responses.append(response)
            elif event_type == 'error':
                _, error = event
                merger_errors.append(error)
        
        # Verify results
        self.assert_equal(len(merger_errors), 0, "Should have no errors")
        
        # Verify all Workers received requests
        total_received = sum(len(reqs) for reqs in worker_received.values())
        self.assert_equal(total_received, num_requests,
                         f"Workers should receive all {num_requests} requests")
        self.logger.info(f"✓ Workers received {total_received} requests")
        
        # Verify round-robin distribution
        for worker_id in range(num_workers):
            expected_count = num_requests // num_workers
            actual_count = len(worker_received[worker_id])
            self.assert_equal(actual_count, expected_count,
                            f"Worker {worker_id} should receive {expected_count} requests")
            self.logger.info(f"  ✓ Worker {worker_id} received {actual_count} requests")
        
        self.logger.info("✓ Round-robin distribution verified")
        
        # Verify all responses reached Merger
        self.assert_equal(len(merger_responses), num_requests,
                         f"Merger should receive all {num_requests} responses")
        self.logger.info(f"✓ Merger received {len(merger_responses)} responses")
        
        # Verify response content
        for response in merger_responses:
            self.assert_true('request_id' in response, "Response should have request_id")
            self.assert_true('worker_id' in response, "Response should have worker_id")
            self.assert_true('output_ids' in response, "Response should have output_ids")
            self.assert_true(response['success'], "Response should be successful")
        
        self.logger.info("✓ All response content verified")
        
        self.result.details['num_requests'] = num_requests
        self.result.details['num_workers'] = num_workers
        self.result.details['responses_received'] = len(merger_responses)
    
    def teardown(self):
        """Cleanup: terminate all processes"""
        if self.merger_process and self.merger_process.is_alive():
            self.logger.info("Terminating Merger...")
            self.merger_process.terminate()
            self.merger_process.join(timeout=5)
            if self.merger_process.is_alive():
                self.merger_process.kill()
                self.merger_process.join()
        
        for i, proc in enumerate(self.worker_processes):
            if proc and proc.is_alive():
                self.logger.info(f"Terminating Worker {i}...")
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()


class TestFullCommunicationLoad(ManualTest):
    """Test 2: Full communication chain under load"""
    
    def setup(self):
        self.worker_processes = []
        self.merger_process = None
        self.worker_queue = multiprocessing.Queue()
        self.merger_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing full communication chain under load")
        
        # Configuration
        num_workers = 2
        worker_ports = [52200, 52201]
        response_port = 52300
        num_requests = 20  # More requests
        
        # Start Merger
        self.logger.info(f"Starting Merger...")
        self.merger_process = multiprocessing.Process(
            target=simple_merger_process,
            args=(response_port, self.merger_queue),
            name="TestMerger"
        )
        self.merger_process.start()
        time.sleep(2)
        
        # Start Workers
        self.logger.info(f"Starting {num_workers} Workers...")
        for i in range(num_workers):
            worker_process = multiprocessing.Process(
                target=simple_worker_process,
                args=(i, worker_ports[i], response_port, self.worker_queue),
                name=f"TestWorker-{i}"
            )
            worker_process.start()
            self.worker_processes.append(worker_process)
        
        time.sleep(2)
        
        # Prepare test requests
        test_requests = []
        for i in range(num_requests):
            test_requests.append({
                'request_id': f'load-req-{i:03d}',
                'prompt_ids': list(range(i, i+5)),
                'adapter_dir': f'adapter-{i % 3}' if i % 2 == 0 else None,
                'sampling_params': {'max_new_tokens': 10 + i}
            })
        
        # Send requests rapidly
        self.logger.info(f"Sending {num_requests} requests rapidly...")
        sent_count = asyncio.run(simple_manager_sender(worker_ports, test_requests))
        
        self.assert_equal(sent_count, num_requests,
                         f"Should send all {num_requests} requests")
        self.logger.info(f"✓ Manager sent {sent_count} requests")
        
        # Wait for processing
        time.sleep(4)
        
        # Collect results
        worker_received = defaultdict(list)
        while not self.worker_queue.empty():
            event = self.worker_queue.get()
            if event[0] == 'received':
                _, worker_id, request_id = event
                worker_received[worker_id].append(request_id)
        
        merger_responses = []
        while not self.merger_queue.empty():
            event = self.merger_queue.get()
            if event[0] == 'received':
                _, response = event
                merger_responses.append(response)
        
        # Verify
        total_received = sum(len(reqs) for reqs in worker_received.values())
        self.assert_equal(total_received, num_requests,
                         f"Workers should receive all {num_requests} requests")
        
        self.assert_equal(len(merger_responses), num_requests,
                         f"Merger should receive all {num_requests} responses")
        
        # Calculate success rate
        success_rate = (len(merger_responses) / num_requests) * 100
        self.logger.info(f"✓ Success rate: {success_rate:.1f}%")
        self.logger.info("✓ No messages lost under load")
        
        self.result.details['num_requests'] = num_requests
        self.result.details['responses_received'] = len(merger_responses)
        self.result.details['success_rate'] = success_rate
    
    def teardown(self):
        if self.merger_process and self.merger_process.is_alive():
            self.merger_process.terminate()
            self.merger_process.join(timeout=5)
            if self.merger_process.is_alive():
                self.merger_process.kill()
                self.merger_process.join()
        
        for proc in self.worker_processes:
            if proc and proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()


def main():
    """Run all full communication chain tests"""
    print("\n" + "="*70)
    print("Test 8.4.3: Full Communication Chain Test")
    print("="*70)
    print("\nThis test validates the complete communication flow:")
    print("Manager → Workers → Merger")
    print("\nTesting round-robin routing and end-to-end message delivery.\n")
    
    tests = [
        TestFullCommunicationBasic(
            "test_full_communication_basic",
            "Test basic full communication chain"
        ),
        TestFullCommunicationLoad(
            "test_full_communication_load",
            "Test full communication under load"
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
        print("\n🎉 All full communication chain tests PASSED!")
        print("\nFull communication chain verified:")
        print("  ✓ Manager sends requests to Workers")
        print("  ✓ Round-robin routing works correctly")
        print("  ✓ Workers receive and process requests")
        print("  ✓ Workers send responses to Merger")
        print("  ✓ Merger receives all responses")
        print("  ✓ No messages lost in the chain")
        print("  ✓ Communication reliable under load")
        print("\nCommunication tests (8.4) complete!")
        print("Ready to proceed to: 8.5 (Functional Tests)")
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
