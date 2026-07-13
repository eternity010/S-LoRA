"""
Tests for WorkerStateReporter

Tests the Worker-side state reporter that sends state updates to the Router.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from slora.server.router.worker_state_reporter import WorkerStateReporter
from slora.server.router.worker_state import WorkerState


class TestWorkerStateReporterInit:
    """Tests for WorkerStateReporter initialization"""
    
    def test_initialization_defaults(self):
        """Test default initialization"""
        reporter = WorkerStateReporter(worker_id=0)
        
        assert reporter.worker_id == 0
        assert reporter.report_interval_ms == 100
        assert reporter.router_address is None
        assert reporter._state_getter is None
        assert not reporter.is_running
        assert reporter.report_count == 0
    
    def test_initialization_custom_values(self):
        """Test initialization with custom values"""
        def getter():
            return {'cached_adapters': set(), 'queue_length': 0}
        
        reporter = WorkerStateReporter(
            worker_id=5,
            report_interval_ms=200,
            router_address="tcp://127.0.0.1:5555",
            state_getter=getter
        )
        
        assert reporter.worker_id == 5
        assert reporter.report_interval_ms == 200
        assert reporter.router_address == "tcp://127.0.0.1:5555"
        assert reporter._state_getter is getter
    
    def test_initialization_invalid_worker_id(self):
        """Test that negative worker_id raises error"""
        with pytest.raises(ValueError, match="worker_id must be non-negative"):
            WorkerStateReporter(worker_id=-1)
    
    def test_initialization_invalid_interval(self):
        """Test that non-positive interval raises error"""
        with pytest.raises(ValueError, match="report_interval_ms must be positive"):
            WorkerStateReporter(worker_id=0, report_interval_ms=0)
        
        with pytest.raises(ValueError, match="report_interval_ms must be positive"):
            WorkerStateReporter(worker_id=0, report_interval_ms=-100)


class TestStateGetter:
    """Tests for state getter functionality"""
    
    def test_set_state_getter(self):
        """Test setting state getter after initialization"""
        reporter = WorkerStateReporter(worker_id=0)
        
        def getter():
            return {'cached_adapters': {'adapter1'}, 'queue_length': 5}
        
        reporter.set_state_getter(getter)
        assert reporter._state_getter is getter
    
    def test_get_current_state_with_getter(self):
        """Test getting current state with a getter"""
        def getter():
            return {
                'cached_adapters': {'adapter1', 'adapter2'},
                'queue_length': 10,
                'gpu_memory_free': 1000000,
                'pending_raw_tokens': 1234,
                'waiting_request_count': 3,
                'current_batch_size': 4,
                'current_batch_prompt_tokens': 567,
            }
        
        reporter = WorkerStateReporter(worker_id=3, state_getter=getter)
        state = reporter.get_current_state()
        
        assert state.worker_id == 3
        assert state.cached_adapters == {'adapter1', 'adapter2'}
        assert state.queue_length == 10
        assert state.gpu_memory_free == 1000000
        assert state.pending_raw_tokens == 1234
        assert state.waiting_request_count == 3
        assert state.current_batch_size == 4
        assert state.current_batch_prompt_tokens == 567
        assert state.is_healthy is True
    
    def test_get_current_state_with_list_adapters(self):
        """Test that list adapters are converted to set"""
        def getter():
            return {
                'cached_adapters': ['adapter1', 'adapter2'],  # List instead of set
                'queue_length': 5
            }
        
        reporter = WorkerStateReporter(worker_id=0, state_getter=getter)
        state = reporter.get_current_state()
        
        assert isinstance(state.cached_adapters, set)
        assert state.cached_adapters == {'adapter1', 'adapter2'}
    
    def test_get_current_state_without_getter(self):
        """Test getting current state without a getter returns defaults"""
        reporter = WorkerStateReporter(worker_id=2)
        state = reporter.get_current_state()
        
        assert state.worker_id == 2
        assert state.cached_adapters == set()
        assert state.queue_length == 0
        assert state.gpu_memory_free == 0
        assert state.is_healthy is True
    
    def test_get_current_state_getter_error(self):
        """Test that getter errors return default state"""
        def bad_getter():
            raise RuntimeError("Getter error")
        
        reporter = WorkerStateReporter(worker_id=0, state_getter=bad_getter)
        state = reporter.get_current_state()
        
        # Should return default state on error
        assert state.worker_id == 0
        assert state.cached_adapters == set()
        assert state.queue_length == 0


class TestStateMessage:
    """Tests for state message creation"""
    
    def test_create_state_message(self):
        """Test creating state message from WorkerState"""
        reporter = WorkerStateReporter(worker_id=1)
        
        state = WorkerState(
            worker_id=1,
            cached_adapters={'adapter1', 'adapter2'},
            queue_length=5,
            gpu_memory_free=2000000,
            pending_raw_tokens=900,
            waiting_request_count=2,
            current_batch_size=3,
            current_batch_prompt_tokens=700,
        )
        
        message = reporter._create_state_message(state)
        
        assert message['type'] == 'worker_state'
        assert message['worker_id'] == 1
        assert set(message['cached_adapters']) == {'adapter1', 'adapter2'}
        assert message['queue_length'] == 5
        assert message['pending_raw_tokens'] == 900
        assert message['waiting_request_count'] == 2
        assert message['current_batch_size'] == 3
        assert message['current_batch_prompt_tokens'] == 700
        assert message['report_seq'] == 1
        assert message['worker_report_time'] == message['timestamp']
        assert message['gpu_memory_free'] == 2000000
        assert 'timestamp' in message
        assert isinstance(message['timestamp'], float)
    
    def test_create_state_message_empty_adapters(self):
        """Test creating message with no cached adapters"""
        reporter = WorkerStateReporter(worker_id=0)
        
        state = WorkerState(worker_id=0)
        message = reporter._create_state_message(state)
        
        assert message['cached_adapters'] == []
        assert message['queue_length'] == 0


class TestStats:
    """Tests for reporter statistics"""
    
    def test_get_stats_initial(self):
        """Test initial stats"""
        reporter = WorkerStateReporter(worker_id=0)
        stats = reporter.get_stats()
        
        assert stats['worker_id'] == 0
        assert stats['report_count'] == 0
        assert stats['error_count'] == 0
        assert stats['running'] is False
        assert stats['has_socket'] is False
    
    def test_is_running_property(self):
        """Test is_running property"""
        reporter = WorkerStateReporter(worker_id=0)
        assert reporter.is_running is False
    
    def test_report_count_property(self):
        """Test report_count property"""
        reporter = WorkerStateReporter(worker_id=0)
        assert reporter.report_count == 0


class TestStartStop:
    """Tests for start/stop functionality"""
    
    def test_start_without_zmq_address(self):
        """Test starting without ZMQ address (for testing)"""
        async def run_test():
            reporter = WorkerStateReporter(worker_id=0)
            
            result = await reporter.start()
            
            # Should start even without ZMQ (for testing purposes)
            assert result is True
            assert reporter.is_running is True
            
            # Clean up
            await reporter.stop()
            assert reporter.is_running is False
        
        asyncio.run(run_test())
    
    def test_start_already_running(self):
        """Test starting when already running"""
        async def run_test():
            reporter = WorkerStateReporter(worker_id=0)
            
            await reporter.start()
            result = await reporter.start()  # Start again
            
            assert result is True  # Should return True but not start again
            
            await reporter.stop()
        
        asyncio.run(run_test())
    
    def test_stop_not_running(self):
        """Test stopping when not running"""
        async def run_test():
            reporter = WorkerStateReporter(worker_id=0)
            
            # Should not raise error
            await reporter.stop()
            assert reporter.is_running is False
        
        asyncio.run(run_test())


class TestReportNow:
    """Tests for immediate reporting"""
    
    def test_report_now_without_socket(self):
        """Test report_now without socket returns False"""
        async def run_test():
            reporter = WorkerStateReporter(worker_id=0)
            
            result = await reporter.report_now()
            
            assert result is False
        
        asyncio.run(run_test())
    
    def test_report_now_sync_without_socket(self):
        """Test synchronous report_now without socket"""
        reporter = WorkerStateReporter(worker_id=0)
        
        # Should not raise error
        reporter.report_now_sync()

    @pytest.mark.asyncio
    async def test_report_if_due_only_sends_after_interval(self):
        reporter = WorkerStateReporter(worker_id=0, report_interval_ms=100)
        reporter.report_now = AsyncMock(return_value=True)
        reporter._last_report_time = time.time()

        assert await reporter.report_if_due() is False
        reporter.report_now.assert_not_awaited()

        reporter._last_report_time = time.time() - 0.2
        assert await reporter.report_if_due() is True
        reporter.report_now.assert_awaited_once_with()


class TestIntegration:
    """Integration tests for WorkerStateReporter"""
    
    def test_reporter_lifecycle(self):
        """Test full reporter lifecycle"""
        async def run_test():
            call_count = 0
            
            def getter():
                nonlocal call_count
                call_count += 1
                return {
                    'cached_adapters': {'adapter1'},
                    'queue_length': call_count
                }
            
            reporter = WorkerStateReporter(
                worker_id=0,
                report_interval_ms=50,  # Fast interval for testing
                state_getter=getter
            )
            
            # Start reporter
            await reporter.start()
            assert reporter.is_running is True
            
            # Wait a bit for some reports
            await asyncio.sleep(0.15)  # Should trigger ~3 reports
            
            # Stop reporter
            await reporter.stop()
            assert reporter.is_running is False
            
            # Getter should have been called multiple times
            assert call_count >= 2
        
        asyncio.run(run_test())
    
    def test_reporter_state_changes(self):
        """Test that reporter captures state changes"""
        async def run_test():
            current_queue = [0]  # Use list to allow mutation in closure
            
            def getter():
                return {
                    'cached_adapters': set(),
                    'queue_length': current_queue[0]
                }
            
            reporter = WorkerStateReporter(
                worker_id=0,
                report_interval_ms=50,
                state_getter=getter
            )
            
            # Get initial state
            state1 = reporter.get_current_state()
            assert state1.queue_length == 0
            
            # Change queue length
            current_queue[0] = 10
            
            # Get updated state
            state2 = reporter.get_current_state()
            assert state2.queue_length == 10
        
        asyncio.run(run_test())
