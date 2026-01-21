#!/usr/bin/env python3
"""
Test 8.4.1: Manager → Worker Communication Test

Tests that the Manager can successfully send messages to Workers
and Workers can receive them correctly.

Test Coverage:
- Manager can send test messages to Worker
- Worker receives messages correctly
- Message content is correct
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


def simple_worker_receiver(worker_id, request_port, result_queue):
    """
    Simple Worker that receives messages and puts them in a queue
    
    This is a simplified Worker that only receives messages without
    full initialization, used for testing communication.
    """
    try:
        print(f"[TestWorker {worker_id}] Starting simple receiver...")
        
        # Create ZMQ context and socket
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        receiver.connect(f"tcp://127.0.0.1:{request_port}")
        
        print(f"[TestWorker {worker_id}] Connected to port {request_port}")
        print(f"[TestWorker {worker_id}] Waiting for messages...")
        
        # Receive messages for 10 seconds
        start_time = time.time()
        received_count = 0
        
        while time.time() - start_time < 10:
            try:
                message = receiver.recv_json()
                received_count += 1
                print(f"[TestWorker {worker_id}] Received message #{received_count}: {message}")
                result_queue.put(('received', message))
            except zmq.Again:
                # Timeout, continue waiting
                pass
            except Exception as e:
                print(f"[TestWorker {worker_id}] Error receiving: {e}")
                result_queue.put(('error', str(e)))
                break
        
        print(f"[TestWorker {worker_id}] Finished, received {received_count} messages")
        result_queue.put(('done', received_count))
        
        # Cleanup
        receiver.close()
        context.term()
        
    except Exception as e:
        print(f"[TestWorker {worker_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('error', str(e)))


async def simple_manager_sender(worker_port, messages):
    """
    Simple Manager that sends test messages to a Worker
    
    Args:
        worker_port: Port to send messages to
        messages: List of messages to send
    
    Returns:
        Number of messages sent successfully
    """
    try:
        print(f"[TestManager] Starting simple sender...")
        
        # Create ZMQ context and socket
        context = zmq.asyncio.Context()
        sender = context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
        sender.bind(f"tcp://127.0.0.1:{worker_port}")
        
        print(f"[TestManager] Bound to port {worker_port}")
        
        # Wait a bit for Worker to connect
        await asyncio.sleep(1)
        
        # Send messages
        sent_count = 0
        for i, message in enumerate(messages):
            try:
                print(f"[TestManager] Sending message #{i+1}: {message}")
                await sender.send_json(message)
                sent_count += 1
                await asyncio.sleep(0.1)  # Small delay between messages
            except Exception as e:
                print(f"[TestManager] Error sending message #{i+1}: {e}")
                break
        
        print(f"[TestManager] Finished, sent {sent_count}/{len(messages)} messages")
        
        # Cleanup
        sender.close()
        context.term()
        
        return sent_count
        
    except Exception as e:
        print(f"[TestManager] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 0


class TestManagerToWorkerBasic(ManualTest):
    """Test 1: Basic Manager → Worker communication"""
    
    def setup(self):
        self.worker_process = None
        self.result_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing basic Manager → Worker communication")
        
        # Configuration
        worker_id = 0
        request_port = 50000
        
        # Start Worker receiver
        self.logger.info(f"Starting Worker receiver on port {request_port}...")
        self.worker_process = multiprocessing.Process(
            target=simple_worker_receiver,
            args=(worker_id, request_port, self.result_queue),
            name="TestWorker-0"
        )
        self.worker_process.start()
        
        # Wait for Worker to initialize
        time.sleep(2)
        
        # Check Worker is alive
        self.assert_true(self.worker_process.is_alive(), 
                        "Worker process should be alive")
        self.logger.info("✓ Worker receiver started")
        
        # Prepare test messages
        test_messages = [
            {
                'request_id': 'test-001',
                'prompt_ids': [1, 2, 3, 4, 5],
                'adapter_dir': None,
                'sampling_params': {'max_new_tokens': 10}
            },
            {
                'request_id': 'test-002',
                'prompt_ids': [10, 20, 30],
                'adapter_dir': 'adapter-1',
                'sampling_params': {'max_new_tokens': 20}
            },
            {
                'request_id': 'test-003',
                'prompt_ids': [100, 200],
                'adapter_dir': None,
                'sampling_params': {'max_new_tokens': 5}
            }
        ]
        
        # Send messages from Manager
        self.logger.info(f"Sending {len(test_messages)} test messages...")
        sent_count = asyncio.run(simple_manager_sender(request_port, test_messages))
        
        self.assert_equal(sent_count, len(test_messages),
                         f"Should send all {len(test_messages)} messages")
        self.logger.info(f"✓ Manager sent {sent_count} messages")
        
        # Wait for Worker to receive
        time.sleep(2)
        
        # Collect results from Worker
        received_messages = []
        errors = []
        done_count = 0
        
        while not self.result_queue.empty():
            event_type, data = self.result_queue.get()
            if event_type == 'received':
                received_messages.append(data)
            elif event_type == 'error':
                errors.append(data)
            elif event_type == 'done':
                done_count = data
        
        # Verify results
        self.assert_equal(len(errors), 0, "Should have no errors")
        self.assert_equal(len(received_messages), len(test_messages),
                         f"Worker should receive all {len(test_messages)} messages")
        
        self.logger.info(f"✓ Worker received {len(received_messages)} messages")
        
        # Verify message content
        for i, (sent, received) in enumerate(zip(test_messages, received_messages)):
            self.assert_equal(received['request_id'], sent['request_id'],
                            f"Message {i+1} request_id should match")
            self.assert_equal(received['prompt_ids'], sent['prompt_ids'],
                            f"Message {i+1} prompt_ids should match")
            self.logger.info(f"  ✓ Message {i+1} content verified")
        
        self.logger.info("✓ All message content verified")
        
        self.result.details['sent_count'] = sent_count
        self.result.details['received_count'] = len(received_messages)
        self.result.details['errors'] = len(errors)
    
    def teardown(self):
        """Cleanup: terminate Worker process"""
        if self.worker_process and self.worker_process.is_alive():
            self.logger.info("Terminating Worker process...")
            self.worker_process.terminate()
            self.worker_process.join(timeout=5)
            if self.worker_process.is_alive():
                self.worker_process.kill()
                self.worker_process.join()


class TestManagerToWorkerMultiple(ManualTest):
    """Test 2: Manager sends multiple messages to Worker"""
    
    def setup(self):
        self.worker_process = None
        self.result_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing Manager sending multiple messages to Worker")
        
        # Configuration
        worker_id = 0
        request_port = 50100
        num_messages = 10
        
        # Start Worker receiver
        self.logger.info(f"Starting Worker receiver...")
        self.worker_process = multiprocessing.Process(
            target=simple_worker_receiver,
            args=(worker_id, request_port, self.result_queue),
            name="TestWorker-0"
        )
        self.worker_process.start()
        time.sleep(2)
        
        # Prepare test messages
        test_messages = []
        for i in range(num_messages):
            test_messages.append({
                'request_id': f'test-{i:03d}',
                'prompt_ids': list(range(i, i+5)),
                'adapter_dir': f'adapter-{i % 3}' if i % 2 == 0 else None,
                'sampling_params': {'max_new_tokens': 10 + i}
            })
        
        # Send messages
        self.logger.info(f"Sending {num_messages} messages...")
        sent_count = asyncio.run(simple_manager_sender(request_port, test_messages))
        
        self.assert_equal(sent_count, num_messages,
                         f"Should send all {num_messages} messages")
        self.logger.info(f"✓ Sent {sent_count} messages")
        
        # Wait for Worker to receive
        time.sleep(2)
        
        # Collect results
        received_messages = []
        while not self.result_queue.empty():
            event_type, data = self.result_queue.get()
            if event_type == 'received':
                received_messages.append(data)
        
        # Verify
        self.assert_equal(len(received_messages), num_messages,
                         f"Worker should receive all {num_messages} messages")
        
        self.logger.info(f"✓ Worker received {len(received_messages)} messages")
        
        # Verify order (request_ids should match)
        for i, (sent, received) in enumerate(zip(test_messages, received_messages)):
            self.assert_equal(received['request_id'], sent['request_id'],
                            f"Message {i+1} order should be preserved")
        
        self.logger.info("✓ Message order preserved")
        
        self.result.details['num_messages'] = num_messages
        self.result.details['all_received'] = len(received_messages) == num_messages
    
    def teardown(self):
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join(timeout=5)
            if self.worker_process.is_alive():
                self.worker_process.kill()
                self.worker_process.join()


class TestManagerToWorkerReliability(ManualTest):
    """Test 3: Communication reliability under load"""
    
    def setup(self):
        self.worker_process = None
        self.result_queue = multiprocessing.Queue()
    
    def execute(self):
        self.logger.info("Testing communication reliability under load")
        
        # Configuration
        worker_id = 0
        request_port = 50200
        num_messages = 50  # More messages to test reliability
        
        # Start Worker receiver
        self.logger.info(f"Starting Worker receiver...")
        self.worker_process = multiprocessing.Process(
            target=simple_worker_receiver,
            args=(worker_id, request_port, self.result_queue),
            name="TestWorker-0"
        )
        self.worker_process.start()
        time.sleep(2)
        
        # Prepare test messages with varying sizes
        test_messages = []
        for i in range(num_messages):
            # Vary prompt length
            prompt_len = 5 + (i % 20)
            test_messages.append({
                'request_id': f'load-test-{i:03d}',
                'prompt_ids': list(range(i, i + prompt_len)),
                'adapter_dir': f'adapter-{i % 5}' if i % 3 == 0 else None,
                'sampling_params': {
                    'max_new_tokens': 10 + (i % 50),
                    'temperature': 0.5 + (i % 10) * 0.05
                }
            })
        
        # Send messages rapidly
        self.logger.info(f"Sending {num_messages} messages rapidly...")
        sent_count = asyncio.run(simple_manager_sender(request_port, test_messages))
        
        self.assert_equal(sent_count, num_messages,
                         f"Should send all {num_messages} messages")
        self.logger.info(f"✓ Sent {sent_count} messages")
        
        # Wait for Worker to receive all
        time.sleep(3)
        
        # Collect results
        received_messages = []
        errors = []
        while not self.result_queue.empty():
            event_type, data = self.result_queue.get()
            if event_type == 'received':
                received_messages.append(data)
            elif event_type == 'error':
                errors.append(data)
        
        # Verify
        self.assert_equal(len(errors), 0, "Should have no errors")
        self.assert_equal(len(received_messages), num_messages,
                         f"Worker should receive all {num_messages} messages")
        
        self.logger.info(f"✓ Worker received {len(received_messages)}/{num_messages} messages")
        self.logger.info("✓ No messages lost under load")
        
        # Calculate success rate
        success_rate = (len(received_messages) / num_messages) * 100
        self.logger.info(f"✓ Success rate: {success_rate:.1f}%")
        
        self.result.details['num_messages'] = num_messages
        self.result.details['received'] = len(received_messages)
        self.result.details['success_rate'] = success_rate
    
    def teardown(self):
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join(timeout=5)
            if self.worker_process.is_alive():
                self.worker_process.kill()
                self.worker_process.join()


def main():
    """Run all Manager → Worker communication tests"""
    print("\n" + "="*70)
    print("Test 8.4.1: Manager → Worker Communication Test")
    print("="*70)
    print("\nThis test validates Manager → Worker communication.")
    print("Testing message sending, receiving, and reliability.\n")
    
    tests = [
        TestManagerToWorkerBasic(
            "test_manager_to_worker_basic",
            "Test basic Manager → Worker communication"
        ),
        TestManagerToWorkerMultiple(
            "test_manager_to_worker_multiple",
            "Test Manager sending multiple messages"
        ),
        TestManagerToWorkerReliability(
            "test_manager_to_worker_reliability",
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
        print("\n🎉 All Manager → Worker communication tests PASSED!")
        print("\nManager → Worker communication verified:")
        print("  ✓ Manager can send messages to Worker")
        print("  ✓ Worker receives messages correctly")
        print("  ✓ Message content is correct")
        print("  ✓ Communication is reliable under load")
        print("  ✓ No messages lost")
        print("\nCommunication tests (8.4.1) complete!")
        print("Ready to proceed to: 8.4.2 (Worker → Merger Communication)")
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
