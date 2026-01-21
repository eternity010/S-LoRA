#!/usr/bin/env python3
"""
Test 8.4.2: Worker → Merger Communication Test

Tests that Workers can successfully send responses to Response Merger
and Merger can receive them correctly.

Test Coverage:
- Worker can send test responses to Merger
- Merger receives responses correctly
- Response content is correct
- Communication is reliable
"""

import sys
import time
import multiprocessing
import asyncio
import zmq
import zmq.asyncio
from pathlib import Path
import argparse

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))

from test_base import ManualTest, run_test_suite


def simple_merger_receiver(response_port, result_queue):
    """
    Simple Merger that receives responses and puts them in a queue
    
    This is a simplified Merger that only receives responses without
    full initialization, used for testing communication.
    """
    try:
        print(f"[TestMerger] Starting simple receiver...")
        
        # Create ZMQ context and socket
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        receiver.bind(f"tcp://127.0.0.1:{response_port}")
        
        print(f"[TestMerger] Bound to port {response_port}")
        print(f"[TestMerger] Waiting for responses...")
        
        # Receive responses for 10 seconds
        start_time = time.time()
        received_count = 0
        
        while time.time() - start_time < 10:
            try:
                response = receiver.recv_json()
                received_count += 1
                print(f"[TestMerger] Received response #{received_count}: {response}")
                result_queue.put(('received', response))
            except zmq.Again:
                # Timeout, continue waiting
                pass
            except Exception as e:
                print(f"[TestMerger] Error receiving: {e}")
                result_queue.put(('error', str(e)))
                break
        
        print(f"[TestMerger] Finished, received {received_count} responses")
        result_queue.put(('done', received_count))
        
        # Cleanup
        receiver.close()
        context.term()
        
    except Exception as e:
        print(f"[TestMerger] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', str(e)))


async def simple_worker_sender(worker_id, response_port, responses):
    """
    Simple Worker that sends test responses to Merger
    
    Args:
        worker_id: Worker ID
        response_port: Port to send responses to
        responses: List of responses to send
    
    Returns:
        Number of responses sent successfully
    """
    try:
        print(f"[TestWorker {worker_id}] Starting simple sender...")
        
        # Create ZMQ context and socket
        context = zmq.asyncio.Context()
        sender = context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
        sender.connect(f"tcp://127.0.0.1:{response_port}")
        
        print(f"[TestWorker {worker_id}] Connected to port {response_port}")
        
        # Wait a bit for Merger to bind
        await asyncio.sleep(1)
        
        # Send responses
        sent_count = 0
        for i, response in enumerate(responses):
            try:
                print(f"[TestWorker {worker_id}] Sending response #{i+1}: {response}")
                await sender.send_json(response)
                sent_count += 1
                await asyncio.sleep(0.1)  # Small delay between responses
            except Exception as e:
                print(f"[TestWorker {worker_id}] Error sending response #{i+1}: {e}")
                break
        
        print(f"[TestWorker {worker_id}] Finished, sent {sent_count}/{len(responses)} responses")
        
        # Cleanup
        sender.close()
        context.term()
        
        return sent_count
        
    except Exception as e:
        print(f"[TestWorker {worker_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 0


class TestWorkerToMergerBasic(ManualTest):
    """Test 1: Basic Worker → Merger communication"""
    
    def setup(self):
        self.merger_process = None
        self.result_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing basic Worker → Merger communication")
        
        # Configuration
        worker_id = 0
        response_port = 51000
        
        # Start Merger receiver
        self.logger.info(f"Starting Merger receiver on port {response_port}...")
        self.merger_process = multiprocessing.Process(
            target=simple_merger_receiver,
            args=(response_port, self.result_queue),
            name="TestMerger"
        )
        self.merger_process.start()
        
        # Wait for Merger to initialize
        time.sleep(2)
        
        # Check Merger is alive
        self.assert_true(self.merger_process.is_alive(), 
                        "Merger process should be alive")
        self.logger.info("✓ Merger receiver started")
        
        # Prepare test responses
        test_responses = [
            {
                'request_id': 'test-001',
                'worker_id': worker_id,
                'output_ids': [1, 2, 3, 4, 5],
                'metadata': {
                    'finish_reason': 'generating',
                    'prompt_tokens': 5,
                    'completion_tokens': 5
                },
                'success': True,
                'error': None
            },
            {
                'request_id': 'test-002',
                'worker_id': worker_id,
                'output_ids': [10, 20, 30, 40],
                'metadata': {
                    'finish_reason': 'eos',
                    'prompt_tokens': 3,
                    'completion_tokens': 4
                },
                'success': True,
                'error': None
            },
            {
                'request_id': 'test-003',
                'worker_id': worker_id,
                'output_ids': [100, 200, 300],
                'metadata': {
                    'finish_reason': 'length',
                    'prompt_tokens': 2,
                    'completion_tokens': 3
                },
                'success': True,
                'error': None
            }
        ]
        
        # Send responses from Worker
        self.logger.info(f"Sending {len(test_responses)} test responses...")
        sent_count = asyncio.run(simple_worker_sender(worker_id, response_port, test_responses))
        
        self.assert_equal(sent_count, len(test_responses),
                         f"Should send all {len(test_responses)} responses")
        self.logger.info(f"✓ Worker sent {sent_count} responses")
        
        # Wait for Merger to receive
        time.sleep(2)
        
        # Collect results from Merger
        received_responses = []
        errors = []
        done_count = 0
        
        while not self.result_queue.empty():
            event_type, data = self.result_queue.get()
            if event_type == 'received':
                received_responses.append(data)
            elif event_type == 'error':
                errors.append(data)
            elif event_type == 'done':
                done_count = data
        
        # Verify results
        self.assert_equal(len(errors), 0, "Should have no errors")
        self.assert_equal(len(received_responses), len(test_responses),
                         f"Merger should receive all {len(test_responses)} responses")
        
        self.logger.info(f"✓ Merger received {len(received_responses)} responses")
        
        # Verify response content
        for i, (sent, received) in enumerate(zip(test_responses, received_responses)):
            self.assert_equal(received['request_id'], sent['request_id'],
                            f"Response {i+1} request_id should match")
            self.assert_equal(received['worker_id'], sent['worker_id'],
                            f"Response {i+1} worker_id should match")
            self.assert_equal(received['output_ids'], sent['output_ids'],
                            f"Response {i+1} output_ids should match")
            self.assert_equal(received['success'], sent['success'],
                            f"Response {i+1} success should match")
            self.logger.info(f"  ✓ Response {i+1} content verified")
        
        self.logger.info("✓ All response content verified")
        
        self.result.details['sent_count'] = sent_count
        self.result.details['received_count'] = len(received_responses)
        self.result.details['errors'] = len(errors)
    
    def teardown(self):
        """Cleanup: terminate Merger process"""
        if self.merger_process and self.merger_process.is_alive():
            self.logger.info("Terminating Merger process...")
            self.merger_process.terminate()
            self.merger_process.join(timeout=5)
            if self.merger_process.is_alive():
                self.merger_process.kill()
                self.merger_process.join()


class TestWorkerToMergerMultiple(ManualTest):
    """Test 2: Worker sends multiple responses to Merger"""
    
    def setup(self):
        self.merger_process = None
        self.result_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing Worker sending multiple responses to Merger")
        
        # Configuration
        worker_id = 0
        response_port = 51100
        num_responses = 10
        
        # Start Merger receiver
        self.logger.info(f"Starting Merger receiver...")
        self.merger_process = multiprocessing.Process(
            target=simple_merger_receiver,
            args=(response_port, self.result_queue),
            name="TestMerger"
        )
        self.merger_process.start()
        time.sleep(2)
        
        # Prepare test responses
        test_responses = []
        for i in range(num_responses):
            test_responses.append({
                'request_id': f'test-{i:03d}',
                'worker_id': worker_id,
                'output_ids': list(range(i, i+5)),
                'metadata': {
                    'finish_reason': 'generating' if i < num_responses - 1 else 'eos',
                    'prompt_tokens': 5,
                    'completion_tokens': i + 1
                },
                'success': True,
                'error': None
            })
        
        # Send responses
        self.logger.info(f"Sending {num_responses} responses...")
        sent_count = asyncio.run(simple_worker_sender(worker_id, response_port, test_responses))
        
        self.assert_equal(sent_count, num_responses,
                         f"Should send all {num_responses} responses")
        self.logger.info(f"✓ Sent {sent_count} responses")
        
        # Wait for Merger to receive
        time.sleep(2)
        
        # Collect results
        received_responses = []
        while not self.result_queue.empty():
            event_type, data = self.result_queue.get()
            if event_type == 'received':
                received_responses.append(data)
        
        # Verify
        self.assert_equal(len(received_responses), num_responses,
                         f"Merger should receive all {num_responses} responses")
        
        self.logger.info(f"✓ Merger received {len(received_responses)} responses")
        
        # Verify order (request_ids should match)
        for i, (sent, received) in enumerate(zip(test_responses, received_responses)):
            self.assert_equal(received['request_id'], sent['request_id'],
                            f"Response {i+1} order should be preserved")
        
        self.logger.info("✓ Response order preserved")
        
        self.result.details['num_responses'] = num_responses
        self.result.details['all_received'] = len(received_responses) == num_responses
    
    def teardown(self):
        if self.merger_process and self.merger_process.is_alive():
            self.merger_process.terminate()
            self.merger_process.join(timeout=5)
            if self.merger_process.is_alive():
                self.merger_process.kill()
                self.merger_process.join()


class TestWorkerToMergerReliability(ManualTest):
    """Test 3: Communication reliability under load"""
    
    def setup(self):
        self.merger_process = None
        self.result_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing communication reliability under load")
        
        # Configuration
        worker_id = 0
        response_port = 51200
        num_responses = 50  # More responses to test reliability
        
        # Start Merger receiver
        self.logger.info(f"Starting Merger receiver...")
        self.merger_process = multiprocessing.Process(
            target=simple_merger_receiver,
            args=(response_port, self.result_queue),
            name="TestMerger"
        )
        self.merger_process.start()
        time.sleep(2)
        
        # Prepare test responses with varying sizes
        test_responses = []
        for i in range(num_responses):
            # Vary output length
            output_len = 5 + (i % 20)
            test_responses.append({
                'request_id': f'load-test-{i:03d}',
                'worker_id': worker_id,
                'output_ids': list(range(i, i + output_len)),
                'metadata': {
                    'finish_reason': 'generating' if i < num_responses - 1 else 'eos',
                    'prompt_tokens': 5 + (i % 10),
                    'completion_tokens': output_len,
                    'gen_metadata': {
                        'logprobs': [0.1 * j for j in range(output_len)]
                    }
                },
                'success': True,
                'error': None
            })
        
        # Send responses rapidly
        self.logger.info(f"Sending {num_responses} responses rapidly...")
        sent_count = asyncio.run(simple_worker_sender(worker_id, response_port, test_responses))
        
        self.assert_equal(sent_count, num_responses,
                         f"Should send all {num_responses} responses")
        self.logger.info(f"✓ Sent {sent_count} responses")
        
        # Wait for Merger to receive all
        time.sleep(3)
        
        # Collect results
        received_responses = []
        errors = []
        while not self.result_queue.empty():
            event_type, data = self.result_queue.get()
            if event_type == 'received':
                received_responses.append(data)
            elif event_type == 'error':
                errors.append(data)
        
        # Verify
        self.assert_equal(len(errors), 0, "Should have no errors")
        self.assert_equal(len(received_responses), num_responses,
                         f"Merger should receive all {num_responses} responses")
        
        self.logger.info(f"✓ Merger received {len(received_responses)}/{num_responses} responses")
        self.logger.info("✓ No responses lost under load")
        
        # Calculate success rate
        success_rate = (len(received_responses) / num_responses) * 100
        self.logger.info(f"✓ Success rate: {success_rate:.1f}%")
        
        self.result.details['num_responses'] = num_responses
        self.result.details['received'] = len(received_responses)
        self.result.details['success_rate'] = success_rate
    
    def teardown(self):
        if self.merger_process and self.merger_process.is_alive():
            self.merger_process.terminate()
            self.merger_process.join(timeout=5)
            if self.merger_process.is_alive():
                self.merger_process.kill()
                self.merger_process.join()


def main():
    """Run all Worker → Merger communication tests"""
    print("\n" + "="*70)
    print("Test 8.4.2: Worker → Merger Communication Test")
    print("="*70)
    print("\nThis test validates Worker → Merger communication.")
    print("Testing response sending, receiving, and reliability.\n")
    
    tests = [
        TestWorkerToMergerBasic(
            "test_worker_to_merger_basic",
            "Test basic Worker → Merger communication"
        ),
        TestWorkerToMergerMultiple(
            "test_worker_to_merger_multiple",
            "Test Worker sending multiple responses"
        ),
        TestWorkerToMergerReliability(
            "test_worker_to_merger_reliability",
            "Test communication reliability under load"
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
        print("\n🎉 All Worker → Merger communication tests PASSED!")
        print("\nWorker → Merger communication verified:")
        print("  ✓ Worker can send responses to Merger")
        print("  ✓ Merger receives responses correctly")
        print("  ✓ Response content is correct")
        print("  ✓ Communication is reliable under load")
        print("  ✓ No responses lost")
        print("\nCommunication tests (8.4.2) complete!")
        print("Ready to proceed to: 8.4.3 (Full Communication Chain)")
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
