"""
测试 GPU Worker 的批次管理功能

Task 2.9.3: 集成完整的批次管理
"""

import pytest
import argparse
from unittest.mock import Mock, AsyncMock, patch
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
    args.no_lora = False
    args.eos_id = 2
    return args


@pytest.mark.asyncio
async def test_handle_finish_req_with_finished_requests(mock_args):
    """测试 _handle_finish_req 处理已完成的请求"""
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 创建模拟批次
    sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
    req1 = Req(adapter_dir='lora1', request_id='req1', prompt_ids=[1, 2, 3], sample_params=sample_params)
    req2 = Req(adapter_dir='lora2', request_id='req2', prompt_ids=[4, 5, 6], sample_params=sample_params)
    
    # 标记 req1 为已完成
    req1.has_generate_finished = True
    req2.has_generate_finished = False
    
    batch = Batch(batch_id=1, reqs=[req1, req2])
    worker.current_batch = batch
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.decrease_request_counts = AsyncMock()
    worker.model_rpc.filter_batch = AsyncMock()
    
    # 调用 _handle_finish_req
    await worker._handle_finish_req(batch, has_new_finished_req=True)
    
    # 验证 decrease_request_counts 被调用
    worker.model_rpc.decrease_request_counts.assert_called_once_with(['lora1'])
    
    # 验证批次被过滤
    assert len(batch.reqs) == 1
    assert batch.reqs[0].request_id == 'req2'
    
    # 验证 filter_batch 被调用
    worker.model_rpc.filter_batch.assert_called_once_with(1, ['req2'])


@pytest.mark.asyncio
async def test_handle_finish_req_batch_becomes_empty(mock_args):
    """测试 _handle_finish_req 当批次变空时"""
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 创建模拟批次（所有请求都完成）
    sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
    req1 = Req(adapter_dir='lora1', request_id='req1', prompt_ids=[1, 2, 3], sample_params=sample_params)
    req2 = Req(adapter_dir='lora2', request_id='req2', prompt_ids=[4, 5, 6], sample_params=sample_params)
    
    # 标记所有请求为已完成
    req1.has_generate_finished = True
    req2.has_generate_finished = True
    
    batch = Batch(batch_id=1, reqs=[req1, req2])
    worker.current_batch = batch
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.decrease_request_counts = AsyncMock()
    worker.model_rpc.remove_batch = AsyncMock()
    
    # 调用 _handle_finish_req
    await worker._handle_finish_req(batch, has_new_finished_req=True)
    
    # 验证 decrease_request_counts 被调用
    worker.model_rpc.decrease_request_counts.assert_called_once()
    call_args = worker.model_rpc.decrease_request_counts.call_args[0][0]
    assert set(call_args) == {'lora1', 'lora2'}
    
    # 验证批次为空
    assert batch.is_clear()
    
    # 验证 remove_batch 被调用
    worker.model_rpc.remove_batch.assert_called_once_with(1)
    
    # 验证 current_batch 被重置
    assert worker.current_batch is None


@pytest.mark.asyncio
async def test_handle_finish_req_no_finished_requests(mock_args):
    """测试 _handle_finish_req 当没有完成的请求时"""
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 创建模拟批次（没有完成的请求）
    sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
    req1 = Req(adapter_dir='lora1', request_id='req1', prompt_ids=[1, 2, 3], sample_params=sample_params)
    req2 = Req(adapter_dir='lora2', request_id='req2', prompt_ids=[4, 5, 6], sample_params=sample_params)
    
    batch = Batch(batch_id=1, reqs=[req1, req2])
    worker.current_batch = batch
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.decrease_request_counts = AsyncMock()
    worker.model_rpc.filter_batch = AsyncMock()
    worker.model_rpc.remove_batch = AsyncMock()
    
    # 调用 _handle_finish_req（没有完成的请求）
    await worker._handle_finish_req(batch, has_new_finished_req=False)
    
    # 验证没有调用任何 RPC 方法
    worker.model_rpc.decrease_request_counts.assert_not_called()
    worker.model_rpc.filter_batch.assert_not_called()
    worker.model_rpc.remove_batch.assert_not_called()
    
    # 验证批次状态不变
    assert len(batch.reqs) == 2
    assert worker.current_batch is not None


@pytest.mark.asyncio
async def test_handle_finish_req_with_no_lora(mock_args):
    """测试 _handle_finish_req 在禁用 LoRA 时"""
    mock_args.no_lora = True
    
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 创建模拟批次
    sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
    req1 = Req(adapter_dir='base', request_id='req1', prompt_ids=[1, 2, 3], sample_params=sample_params)
    
    # 标记请求为已完成
    req1.has_generate_finished = True
    
    batch = Batch(batch_id=1, reqs=[req1])
    worker.current_batch = batch
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.decrease_request_counts = AsyncMock()
    worker.model_rpc.remove_batch = AsyncMock()
    
    # 调用 _handle_finish_req
    await worker._handle_finish_req(batch, has_new_finished_req=True)
    
    # 验证 decrease_request_counts 没有被调用（因为 no_lora=True）
    worker.model_rpc.decrease_request_counts.assert_not_called()
    
    # 验证 remove_batch 被调用（批次为空）
    worker.model_rpc.remove_batch.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_handle_finish_req_multiple_adapters(mock_args):
    """测试 _handle_finish_req 处理多个不同的 adapters"""
    with patch.object(GPUWorker, '_setup_gpu'):
        with patch.object(GPUWorker, '_setup_adapter_config'):
            worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
    
    # 创建模拟批次（多个请求使用不同的 adapters）
    sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
    req1 = Req(adapter_dir='lora1', request_id='req1', prompt_ids=[1, 2, 3], sample_params=sample_params)
    req2 = Req(adapter_dir='lora1', request_id='req2', prompt_ids=[4, 5, 6], sample_params=sample_params)
    req3 = Req(adapter_dir='lora2', request_id='req3', prompt_ids=[7, 8, 9], sample_params=sample_params)
    
    # 标记 req1 和 req3 为已完成
    req1.has_generate_finished = True
    req2.has_generate_finished = False
    req3.has_generate_finished = True
    
    batch = Batch(batch_id=1, reqs=[req1, req2, req3])
    worker.current_batch = batch
    
    # 模拟 model_rpc
    worker.model_rpc = AsyncMock()
    worker.model_rpc.decrease_request_counts = AsyncMock()
    worker.model_rpc.filter_batch = AsyncMock()
    
    # 调用 _handle_finish_req
    await worker._handle_finish_req(batch, has_new_finished_req=True)
    
    # 验证 decrease_request_counts 被调用，包含两个 adapters
    worker.model_rpc.decrease_request_counts.assert_called_once()
    call_args = worker.model_rpc.decrease_request_counts.call_args[0][0]
    assert 'lora1' in call_args
    assert 'lora2' in call_args
    
    # 验证批次只保留 req2
    assert len(batch.reqs) == 1
    assert batch.reqs[0].request_id == 'req2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
