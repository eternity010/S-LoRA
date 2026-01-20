"""
测试 ZMQ 通信超时处理

验证 Router Manager、GPU Worker 和 Response Merger 的 ZMQ 超时配置和重试逻辑。
"""

import pytest
import asyncio
import argparse
import zmq
import zmq.asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_router_manager_zmq_timeout_configuration():
    """
    测试 Router Manager 的 ZMQ 超时配置
    
    验证：
    1. PULL socket 设置了 RCVTIMEO
    2. PUSH sockets 设置了 SNDTIMEO
    3. 所有 sockets 设置了 LINGER=0
    
    Requirements: 7.4
    """
    from slora.server.router.dp_manager import DataParallelRouterManager
    
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        num_workers=2,
        gpu_ids='0,1',
        lora_dirs=None
    )
    
    manager = DataParallelRouterManager(args, 40000, 40001, 40002)
    manager._allocate_ports()
    
    # 模拟 socket 创建
    mock_context = Mock()
    mock_pull_socket = Mock()
    mock_push_socket1 = Mock()
    mock_push_socket2 = Mock()
    
    socket_call_count = [0]
    def mock_socket(socket_type):
        socket_call_count[0] += 1
        if socket_call_count[0] == 1:
            return mock_pull_socket
        elif socket_call_count[0] == 2:
            return mock_push_socket1
        else:
            return mock_push_socket2
    
    mock_context.socket = mock_socket
    
    with patch('zmq.asyncio.Context', return_value=mock_context):
        manager._setup_zmq()
    
    # 验证 PULL socket 超时配置
    mock_pull_socket.setsockopt.assert_any_call(zmq.RCVTIMEO, 30000)
    mock_pull_socket.setsockopt.assert_any_call(zmq.LINGER, 0)
    
    # 验证 PUSH sockets 超时配置
    mock_push_socket1.setsockopt.assert_any_call(zmq.SNDTIMEO, 30000)
    mock_push_socket1.setsockopt.assert_any_call(zmq.LINGER, 0)
    mock_push_socket2.setsockopt.assert_any_call(zmq.SNDTIMEO, 30000)
    mock_push_socket2.setsockopt.assert_any_call(zmq.LINGER, 0)


@pytest.mark.asyncio
async def test_router_manager_route_request_retry_on_timeout():
    """
    测试 Router Manager 的请求路由重试逻辑
    
    验证：
    1. ZMQ 超时时捕获 zmq.Again 异常
    2. 重试最多 3 次
    3. 每次重试间隔 0.1 秒
    4. 超过重试次数后抛出异常
    
    Requirements: 7.4
    """
    from slora.server.router.dp_manager import DataParallelRouterManager
    
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        num_workers=2,
        gpu_ids='0,1',
        lora_dirs=None
    )
    
    manager = DataParallelRouterManager(args, 40000, 40001, 40002)
    manager._allocate_ports()
    manager.request_senders = [AsyncMock(), AsyncMock()]
    
    # 模拟 router 始终选择 Worker 0
    manager.router.select_worker = Mock(return_value=0)
    
    # 模拟第一个 Worker 的 socket 超时 2 次，第 3 次成功
    call_count = [0]
    async def mock_send_json(request):
        call_count[0] += 1
        if call_count[0] < 3:
            raise zmq.Again("Send timeout")
    
    manager.request_senders[0].send_json = mock_send_json
    
    # 调用 route_request
    request = {'request_id': 'test_req', 'prompt_ids': [1, 2, 3]}
    await manager.route_request(request)
    
    # 验证重试了 3 次
    assert call_count[0] == 3


@pytest.mark.asyncio
async def test_router_manager_route_request_max_retries_exceeded():
    """
    测试 Router Manager 超过最大重试次数
    
    验证：
    1. 重试 3 次后仍然失败
    2. 抛出异常
    3. 错误消息包含重试次数
    
    Requirements: 7.4
    """
    from slora.server.router.dp_manager import DataParallelRouterManager
    
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        num_workers=2,
        gpu_ids='0,1',
        lora_dirs=None
    )
    
    manager = DataParallelRouterManager(args, 40000, 40001, 40002)
    manager._allocate_ports()
    manager.request_senders = [AsyncMock(), AsyncMock()]
    
    # 模拟 router 始终选择 Worker 0
    manager.router.select_worker = Mock(return_value=0)
    
    # 模拟 socket 一直超时
    async def mock_send_json(request):
        raise zmq.Again("Send timeout")
    
    manager.request_senders[0].send_json = mock_send_json
    
    # 调用 route_request，应该抛出异常
    request = {'request_id': 'test_req', 'prompt_ids': [1, 2, 3]}
    with pytest.raises(Exception) as exc_info:
        await manager.route_request(request)
    
    # 验证错误消息
    assert "after 3 attempts" in str(exc_info.value)
    assert "ZMQ timeout" in str(exc_info.value)


@pytest.mark.asyncio
async def test_router_manager_route_request_retry_on_other_error():
    """
    测试 Router Manager 处理其他错误的重试逻辑
    
    验证：
    1. 非 zmq.Again 异常也会重试
    2. 重试最多 3 次
    3. 超过重试次数后抛出异常
    
    Requirements: 7.4
    """
    from slora.server.router.dp_manager import DataParallelRouterManager
    
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        num_workers=2,
        gpu_ids='0,1',
        lora_dirs=None
    )
    
    manager = DataParallelRouterManager(args, 40000, 40001, 40002)
    manager._allocate_ports()
    manager.request_senders = [AsyncMock(), AsyncMock()]
    
    # 模拟 router 始终选择 Worker 0
    manager.router.select_worker = Mock(return_value=0)
    
    # 模拟其他错误
    call_count = [0]
    async def mock_send_json(request):
        call_count[0] += 1
        raise RuntimeError("Connection error")
    
    manager.request_senders[0].send_json = mock_send_json
    
    # 调用 route_request，应该抛出异常
    request = {'request_id': 'test_req', 'prompt_ids': [1, 2, 3]}
    with pytest.raises(Exception) as exc_info:
        await manager.route_request(request)
    
    # 验证重试了 3 次
    assert call_count[0] == 3
    assert "after 3 attempts" in str(exc_info.value)


@pytest.mark.asyncio
async def test_gpu_worker_zmq_timeout_configuration():
    """
    测试 GPU Worker 的 ZMQ 超时配置
    
    验证：
    1. PULL socket 设置了 RCVTIMEO
    2. PUSH socket 设置了 SNDTIMEO
    3. 所有 sockets 设置了 LINGER=0
    
    Requirements: 7.4
    """
    from slora.server.router.gpu_worker import GPUWorker
    
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        lora_dirs=None,
        no_lora=True
    )
    
    with patch('slora.server.router.gpu_worker.torch'):
        with patch('slora.server.router.gpu_worker.os.environ'):
            worker = GPUWorker(0, 0, args)
    
    # 模拟 socket 创建
    mock_context = Mock()
    mock_pull_socket = Mock()
    mock_push_socket = Mock()
    
    socket_call_count = [0]
    def mock_socket(socket_type):
        socket_call_count[0] += 1
        if socket_call_count[0] == 1:
            return mock_pull_socket
        else:
            return mock_push_socket
    
    mock_context.socket = mock_socket
    
    with patch('zmq.asyncio.Context', return_value=mock_context):
        worker._setup_zmq(50000, 50001)
    
    # 验证 PULL socket 超时配置
    mock_pull_socket.setsockopt.assert_any_call(zmq.RCVTIMEO, 30000)
    mock_pull_socket.setsockopt.assert_any_call(zmq.LINGER, 0)
    
    # 验证 PUSH socket 超时配置
    mock_push_socket.setsockopt.assert_any_call(zmq.SNDTIMEO, 30000)
    mock_push_socket.setsockopt.assert_any_call(zmq.LINGER, 0)


@pytest.mark.asyncio
async def test_response_merger_zmq_timeout_configuration():
    """
    测试 Response Merger 的 ZMQ 超时配置
    
    验证：
    1. PULL socket 设置了 RCVTIMEO
    2. PUSH socket 设置了 SNDTIMEO
    3. 所有 sockets 设置了 LINGER=0
    
    Requirements: 7.4
    """
    from slora.server.router.response_merger import ResponseMerger
    
    merger = ResponseMerger(50001, 50002)
    
    # 模拟 socket 创建
    mock_context = Mock()
    mock_pull_socket = Mock()
    mock_push_socket = Mock()
    
    socket_call_count = [0]
    def mock_socket(socket_type):
        socket_call_count[0] += 1
        if socket_call_count[0] == 1:
            return mock_pull_socket
        else:
            return mock_push_socket
    
    mock_context.socket = mock_socket
    
    with patch('zmq.asyncio.Context', return_value=mock_context):
        merger._setup_zmq()
    
    # 验证 PULL socket 超时配置
    mock_pull_socket.setsockopt.assert_any_call(zmq.RCVTIMEO, 30000)
    mock_pull_socket.setsockopt.assert_any_call(zmq.LINGER, 0)
    
    # 验证 PUSH socket 超时配置
    mock_push_socket.setsockopt.assert_any_call(zmq.SNDTIMEO, 30000)
    mock_push_socket.setsockopt.assert_any_call(zmq.LINGER, 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
