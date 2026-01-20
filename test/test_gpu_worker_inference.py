"""
测试 GPU Worker 的实际推理逻辑

Task 2.9.2: 实现实际推理逻辑
"""

import pytest
import argparse
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from slora.server.router.gpu_worker import GPUWorker
from slora.server.io_struct import Req, Batch
from slora.server.sampling_params import SamplingParams


@pytest.fixture
def mock_args():
    """创建模拟的命令行参数"""
    args = argparse.Namespace()
    args.model_dir = '/path/to/model'
    args.lora_dirs = []
    args.max_total_token_num = 10000
    args.batch_max_tokens = 2000
    args.running_max_req_size = 10
    args.max_req_total_len = 2048
    args.no_lora = False
    args.eos_id = 2
    return args


@pytest.fixture
def mock_batch():
    """创建模拟的批次"""
    sample_params = SamplingParams(
        do_sample=False,
        max_new_tokens=10
    )
    
    req1 = Req(
        adapter_dir='base',
        request_id='req1',
        prompt_ids=[1, 2, 3, 4, 5],
        sample_params=sample_params
    )
    
    req2 = Req(
        adapter_dir='base',
        request_id='req2',
        prompt_ids=[6, 7, 8],
        sample_params=sample_params
    )
    
    batch = Batch(batch_id=1, reqs=[req1, req2])
    return batch


@pytest.mark.asyncio
async def test_infer_batch_prefill_mode(mock_args, mock_batch):
    """测试 _infer_batch 在 prefill 模式下的行为"""
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.init_batch = AsyncMock()
    worker.model_rpc.prefill_batch = AsyncMock(return_value={
        'req1': (10, {'logprob': -0.5}),
        'req2': (11, {'logprob': -0.3})
    })
    
    # 调用 _infer_batch（prefill 模式）
    result = await worker._infer_batch(mock_batch)
    
    # 验证 init_batch 被调用
    worker.model_rpc.init_batch.assert_called_once()
    call_args = worker.model_rpc.init_batch.call_args
    assert call_args[0][0] == 1  # batch_id
    assert len(call_args[0][1]) == 2  # 2 个请求
    
    # 验证 prefill_batch 被调用
    worker.model_rpc.prefill_batch.assert_called_once_with(1)
    
    # 验证返回结果
    assert 'req1' in result
    assert 'req2' in result
    assert result['req1'] == (10, {'logprob': -0.5})
    assert result['req2'] == (11, {'logprob': -0.3})


@pytest.mark.asyncio
async def test_infer_batch_decode_mode(mock_args, mock_batch):
    """测试 _infer_batch 在 decode 模式下的行为"""
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.update_adapter_stats = AsyncMock()
    worker.model_rpc.decode_batch = AsyncMock(return_value={
        'req1': (12, {'logprob': -0.4}),
        'req2': (13, {'logprob': -0.6})
    })
    
    # 设置批次为 decode 模式（已有输出 token）
    mock_batch.reqs[0].output_ids = [10]
    mock_batch.reqs[1].output_ids = [11]
    
    # 调用 _infer_batch（decode 模式）
    result = await worker._infer_batch(mock_batch)
    
    # 验证 update_adapter_stats 被调用
    worker.model_rpc.update_adapter_stats.assert_called_once()
    
    # 验证 decode_batch 被调用
    worker.model_rpc.decode_batch.assert_called_once_with(1)
    
    # 验证返回结果
    assert 'req1' in result
    assert 'req2' in result
    assert result['req1'] == (12, {'logprob': -0.4})
    assert result['req2'] == (13, {'logprob': -0.6})


@pytest.mark.asyncio
async def test_process_requests_with_inference(mock_args):
    """测试 _process_requests 使用实际推理"""
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 设置 req_queue
    worker.req_queue = Mock()
    worker.lora_ranks = {'base': 0}
    worker.actual_adapter_memory_usage = 0
    
    # 模拟 _update_actual_adapter_usage 和 _load_adapters
    worker._update_actual_adapter_usage = AsyncMock()
    worker._load_adapters = AsyncMock()
    
    # 创建模拟批次
    sample_params = SamplingParams(
        do_sample=False,
        max_new_tokens=10,
        stop_sequences=[]  # 添加空列表而不是 None
    )
    req = Req(
        adapter_dir='base',
        request_id='req1',
        prompt_ids=[1, 2, 3],
        sample_params=sample_params
    )
    batch = Batch(batch_id=1, reqs=[req])
    
    # 模拟 generate_new_batch 返回新批次
    worker.req_queue.generate_new_batch = Mock(return_value=batch)
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.init_batch = AsyncMock()
    worker.model_rpc.prefill_batch = AsyncMock(return_value={
        'req1': (10, {'logprob': -0.5})
    })
    worker.model_rpc.remove_batch = AsyncMock()
    
    # 调用 _process_requests
    responses = await worker._process_requests()
    
    # 验证推理被调用
    worker.model_rpc.init_batch.assert_called_once()
    worker.model_rpc.prefill_batch.assert_called_once()
    
    # 验证响应
    assert len(responses) == 1
    assert responses[0]['request_id'] == 'req1'
    assert responses[0]['worker_id'] == 0
    assert responses[0]['success'] == True
    assert 10 in responses[0]['output_ids']  # 新生成的 token
    assert responses[0]['metadata']['prompt_tokens'] == 3
    assert responses[0]['metadata']['completion_tokens'] == 1


@pytest.mark.asyncio
async def test_process_requests_marks_finished(mock_args):
    """测试 _process_requests 正确标记完成的请求"""
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 设置 req_queue
    worker.req_queue = Mock()
    worker.lora_ranks = {'base': 0}
    worker.actual_adapter_memory_usage = 0
    
    # 模拟 _update_actual_adapter_usage 和 _load_adapters
    worker._update_actual_adapter_usage = AsyncMock()
    worker._load_adapters = AsyncMock()
    
    # 创建模拟批次（max_output_len=1，生成一个 token 后就完成）
    sample_params = SamplingParams(
        do_sample=False,
        max_new_tokens=1,
        stop_sequences=[]  # 添加空列表而不是 None
    )
    req = Req(
        adapter_dir='base',
        request_id='req1',
        prompt_ids=[1, 2, 3],
        sample_params=sample_params
    )
    batch = Batch(batch_id=1, reqs=[req])
    
    # 模拟 generate_new_batch 返回新批次
    worker.req_queue.generate_new_batch = Mock(return_value=batch)
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.init_batch = AsyncMock()
    worker.model_rpc.prefill_batch = AsyncMock(return_value={
        'req1': (10, {'logprob': -0.5})
    })
    worker.model_rpc.remove_batch = AsyncMock()
    
    # 调用 _process_requests
    responses = await worker._process_requests()
    
    # 验证请求被标记为完成
    assert req.has_generate_finished == True
    
    # 验证批次被移除
    worker.model_rpc.remove_batch.assert_called_once_with(1)
    
    # 验证 current_batch 被重置
    assert worker.current_batch is None


@pytest.mark.asyncio
async def test_process_requests_filters_batch(mock_args):
    """测试 _process_requests 正确过滤批次"""
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 设置 req_queue
    worker.req_queue = Mock()
    worker.lora_ranks = {'base': 0}
    worker.actual_adapter_memory_usage = 0
    
    # 模拟 _update_actual_adapter_usage 和 _load_adapters
    worker._update_actual_adapter_usage = AsyncMock()
    worker._load_adapters = AsyncMock()
    
    # 创建模拟批次（2个请求，一个完成，一个未完成）
    sample_params1 = SamplingParams(
        do_sample=False,
        max_new_tokens=1,
        stop_sequences=[]  # 添加空列表而不是 None
    )
    sample_params2 = SamplingParams(
        do_sample=False,
        max_new_tokens=10,
        stop_sequences=[]  # 添加空列表而不是 None
    )
    
    req1 = Req(adapter_dir='base', request_id='req1', prompt_ids=[1, 2, 3], sample_params=sample_params1)
    req2 = Req(adapter_dir='base', request_id='req2', prompt_ids=[4, 5, 6], sample_params=sample_params2)
    
    batch = Batch(batch_id=1, reqs=[req1, req2])
    
    # 模拟 generate_new_batch 返回新批次
    worker.req_queue.generate_new_batch = Mock(return_value=batch)
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.init_batch = AsyncMock()
    worker.model_rpc.prefill_batch = AsyncMock(return_value={
        'req1': (10, {'logprob': -0.5}),
        'req2': (11, {'logprob': -0.3})
    })
    worker.model_rpc.filter_batch = AsyncMock()
    
    # 调用 _process_requests
    responses = await worker._process_requests()
    
    # 验证 req1 被标记为完成，req2 未完成
    assert req1.has_generate_finished == True
    assert req2.has_generate_finished == False
    
    # 验证批次被过滤（调用 filter_batch）
    worker.model_rpc.filter_batch.assert_called_once()
    call_args = worker.model_rpc.filter_batch.call_args
    assert call_args[0][0] == 1  # batch_id
    assert call_args[0][1] == ['req2']  # 只保留 req2
    
    # 验证 current_batch 还存在且只包含 req2
    assert worker.current_batch is not None
    assert len(worker.current_batch.reqs) == 1
    assert worker.current_batch.reqs[0].request_id == 'req2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
