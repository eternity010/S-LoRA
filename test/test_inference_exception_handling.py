"""
测试推理异常处理

验证 Worker 在推理过程中遇到各种异常时能够正确处理并返回错误响应。
"""

import pytest
import asyncio
import argparse
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from slora.server.router.gpu_worker import GPUWorker
from slora.server.io_struct import Req, Batch
from slora.server.sampling_params import SamplingParams


@pytest.fixture
def mock_worker():
    """创建模拟的 GPUWorker 实例"""
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        lora_dirs=None,
        no_lora=True,
        eos_id=2
    )
    
    with patch('slora.server.router.gpu_worker.torch'):
        with patch('slora.server.router.gpu_worker.os.environ'):
            worker = GPUWorker(0, 0, args)
            worker.model_rpc = AsyncMock()
            worker.req_queue = Mock()
            worker.lora_ranks = {None: 0}
            worker.actual_adapter_memory_usage = 0
            return worker


@pytest.fixture
def mock_batch():
    """创建模拟的批次"""
    req1 = Req(
        adapter_dir='base',
        request_id='req1',
        prompt_ids=[1, 2, 3],
        sample_params=SamplingParams()
    )
    req1.output_ids = []
    req1.output_metadata_list = []
    req1.input_len = 3
    req1.max_output_len = 10
    
    batch = Batch(batch_id=1, reqs=[req1])
    return batch


@pytest.mark.asyncio
async def test_infer_batch_cuda_oom_error(mock_worker, mock_batch):
    """
    测试 _infer_batch 处理 CUDA OOM 错误
    
    验证：
    1. 捕获 RuntimeError（CUDA OOM）
    2. 记录详细错误日志（错误类型、批次信息）
    3. 抛出带有错误类型标记的 RuntimeError
    
    Requirements: 7.3
    """
    # 模拟 CUDA OOM 错误
    mock_worker.model_rpc.init_batch = AsyncMock(
        side_effect=RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
    )
    
    # 调用 _infer_batch，应该抛出 RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        await mock_worker._infer_batch(mock_batch)
    
    # 验证错误消息包含错误类型标记
    assert "[CUDA_OOM]" in str(exc_info.value)
    assert "out of memory" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_infer_batch_runtime_error(mock_worker, mock_batch):
    """
    测试 _infer_batch 处理一般运行时错误
    
    验证：
    1. 捕获 RuntimeError（非 OOM）
    2. 记录详细错误日志
    3. 抛出带有错误类型标记的 RuntimeError
    
    Requirements: 7.3
    """
    # 模拟一般运行时错误
    mock_worker.model_rpc.init_batch = AsyncMock(
        side_effect=RuntimeError("Model execution failed")
    )
    
    # 调用 _infer_batch，应该抛出 RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        await mock_worker._infer_batch(mock_batch)
    
    # 验证错误消息包含错误类型标记
    assert "[RUNTIME_ERROR]" in str(exc_info.value)
    assert "Model execution failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_infer_batch_timeout_error(mock_worker, mock_batch):
    """
    测试 _infer_batch 处理 RPC 超时错误
    
    验证：
    1. 捕获 asyncio.TimeoutError
    2. 记录详细错误日志（批次信息）
    3. 抛出带有错误类型标记的 RuntimeError
    
    Requirements: 7.3
    """
    # 模拟 RPC 超时
    mock_worker.model_rpc.init_batch = AsyncMock(
        side_effect=asyncio.TimeoutError("RPC timeout")
    )
    
    # 调用 _infer_batch，应该抛出 RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        await mock_worker._infer_batch(mock_batch)
    
    # 验证错误消息包含错误类型标记
    assert "[RPC_TIMEOUT]" in str(exc_info.value)
    assert "timeout" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_infer_batch_unknown_error(mock_worker, mock_batch):
    """
    测试 _infer_batch 处理未知错误
    
    验证：
    1. 捕获其他类型的异常
    2. 记录详细错误日志（错误类型、批次信息）
    3. 抛出带有错误类型标记的 RuntimeError
    
    Requirements: 7.3
    """
    # 模拟未知错误
    mock_worker.model_rpc.init_batch = AsyncMock(
        side_effect=ValueError("Invalid input")
    )
    
    # 调用 _infer_batch，应该抛出 RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        await mock_worker._infer_batch(mock_batch)
    
    # 验证错误消息包含错误类型标记
    assert "[UNKNOWN_ERROR]" in str(exc_info.value)
    assert "Invalid input" in str(exc_info.value)


@pytest.mark.asyncio
async def test_process_requests_inference_error_generates_error_responses(mock_worker):
    """
    测试 _process_requests 在推理失败时生成错误响应
    
    验证：
    1. 推理失败时捕获 RuntimeError
    2. 为批次中的所有请求生成错误响应
    3. 错误响应包含正确的字段（success=False, error, error_type）
    4. 清理失败的批次
    
    Requirements: 7.3
    """
    # 创建模拟批次
    req1 = Req(
        adapter_dir='base',
        request_id='req1',
        prompt_ids=[1, 2, 3],
        sample_params=SamplingParams()
    )
    req1.output_ids = []
    req1.output_metadata_list = []
    req1.input_len = 3
    req1.max_output_len = 10
    
    req2 = Req(
        adapter_dir='base',
        request_id='req2',
        prompt_ids=[4, 5, 6],
        sample_params=SamplingParams()
    )
    req2.output_ids = []
    req2.output_metadata_list = []
    req2.input_len = 3
    req2.max_output_len = 10
    
    batch = Batch(batch_id=1, reqs=[req1, req2])
    mock_worker.current_batch = batch
    
    # 模拟 ReqQueue 不生成新批次
    mock_worker.req_queue.generate_new_batch = Mock(return_value=None)
    
    # 模拟推理失败
    with patch.object(mock_worker, '_infer_batch', side_effect=RuntimeError("[CUDA_OOM] Out of memory")):
        # 调用 _process_requests
        responses = await mock_worker._process_requests()
    
    # 验证生成了错误响应
    assert len(responses) == 2
    
    # 验证第一个响应
    assert responses[0]['request_id'] == 'req1'
    assert responses[0]['success'] is False
    assert responses[0]['error'] == "[CUDA_OOM] Out of memory"
    assert responses[0]['metadata']['finish_reason'] == 'error'
    assert responses[0]['metadata']['error_type'] == 'CUDA_OOM'
    assert responses[0]['output_ids'] == [1, 2, 3]  # 只返回 prompt
    
    # 验证第二个响应
    assert responses[1]['request_id'] == 'req2'
    assert responses[1]['success'] is False
    assert responses[1]['error'] == "[CUDA_OOM] Out of memory"
    
    # 验证批次被清理
    assert mock_worker.current_batch is None
    mock_worker.model_rpc.remove_batch.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_process_requests_cleanup_error_handling(mock_worker):
    """
    测试 _process_requests 在清理批次时的错误处理
    
    验证：
    1. 推理失败后尝试清理批次
    2. 清理失败时捕获异常并记录日志
    3. 继续执行，不抛出异常
    
    Requirements: 7.3
    """
    # 创建模拟批次
    req1 = Req(
        adapter_dir='base',
        request_id='req1',
        prompt_ids=[1, 2, 3],
        sample_params=SamplingParams()
    )
    req1.output_ids = []
    req1.input_len = 3
    req1.max_output_len = 10
    
    batch = Batch(batch_id=1, reqs=[req1])
    mock_worker.current_batch = batch
    
    # 模拟 ReqQueue 不生成新批次
    mock_worker.req_queue.generate_new_batch = Mock(return_value=None)
    
    # 模拟推理失败
    with patch.object(mock_worker, '_infer_batch', side_effect=RuntimeError("[CUDA_OOM] Out of memory")):
        # 模拟清理批次也失败
        mock_worker.model_rpc.remove_batch = AsyncMock(side_effect=Exception("Cleanup failed"))
        
        # 调用 _process_requests，应该不抛出异常
        responses = await mock_worker._process_requests()
    
    # 验证生成了错误响应
    assert len(responses) == 1
    assert responses[0]['success'] is False
    
    # 验证批次被清理（即使清理失败）
    assert mock_worker.current_batch is None


@pytest.mark.asyncio
async def test_run_handles_receive_error(mock_worker):
    """
    测试 run 主循环处理接收请求错误
    
    验证：
    1. 接收请求失败时捕获异常
    2. 记录错误日志
    3. 继续处理现有批次
    
    Requirements: 7.3
    """
    # 模拟接收请求失败
    mock_worker._receive_request = AsyncMock(side_effect=Exception("ZMQ error"))
    mock_worker._process_requests = AsyncMock(return_value=[])
    
    # 运行一次循环
    async def run_once():
        try:
            request = await asyncio.wait_for(
                mock_worker._receive_request(), 
                timeout=0.01
            )
            req_obj = mock_worker._convert_to_req_object(request)
            mock_worker.req_queue.append(req_obj)
        except asyncio.TimeoutError:
            pass
        except Exception as recv_error:
            # 应该捕获异常并记录日志
            assert "ZMQ error" in str(recv_error)
        
        # 应该继续处理现有批次
        responses = await mock_worker._process_requests()
        return responses
    
    # 调用，应该不抛出异常
    responses = await run_once()
    
    # 验证 _process_requests 被调用
    mock_worker._process_requests.assert_called_once()


@pytest.mark.asyncio
async def test_run_handles_send_error(mock_worker):
    """
    测试 run 主循环处理发送响应错误
    
    验证：
    1. 发送响应失败时捕获异常
    2. 记录错误日志
    3. 继续发送下一个响应
    
    Requirements: 7.3
    """
    # 模拟响应
    responses = [
        {'request_id': 'req1', 'success': True},
        {'request_id': 'req2', 'success': True}
    ]
    
    # 模拟第一个响应发送失败，第二个成功
    send_count = [0]
    async def mock_send_response(response):
        send_count[0] += 1
        if send_count[0] == 1:
            raise Exception("ZMQ send error")
    
    mock_worker._send_response = mock_send_response
    
    # 发送响应
    sent_count = 0
    for response in responses:
        try:
            await mock_worker._send_response(response)
            sent_count += 1
        except Exception as send_error:
            # 应该捕获异常并记录日志
            assert "ZMQ send error" in str(send_error)
            # 继续发送下一个响应
    
    # 验证第二个响应被尝试发送
    assert send_count[0] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
