"""
测试 GPU Worker 的模型 RPC 初始化功能

Task 2.9.1: 实现模型 RPC 初始化
"""

import pytest
import argparse
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from slora.server.router.gpu_worker import GPUWorker


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
    args.pool_size_lora = 0
    args.swap = False
    args.prefetch = False
    args.prefetch_size = 0
    args.scheduler = 'slora'
    args.profile = False
    args.batch_num_adapters = None
    args.enable_abort = False
    args.dummy = False
    args.no_lora_compute = False
    args.no_lora_swap = False
    args.no_kernel = False
    args.no_mem_pool = False
    args.bmm = False
    args.no_lora = False
    args.fair_weights = None
    args.evict_interval_threshold = 0.9
    args.evict_interval_ratio = 0.5
    args.evict_idle_threshold = 0.8
    args.evict_idle_ratio = 0.7
    args.max_lora_ratio = 0.5
    args.load_way = 'HF'
    args.mode = []
    return args


@pytest.mark.asyncio
async def test_init_model_rpc_creates_client(mock_args):
    """测试 _init_model_rpc 创建 ModelRpcClient"""
    with patch('slora.server.router.gpu_worker.start_model_process') as mock_start:
        # 模拟 start_model_process 返回一个 ModelRpcClient
        mock_rpc_client = AsyncMock()
        mock_rpc_client.init_model = AsyncMock()
        mock_start.return_value = mock_rpc_client
        
        # 创建 Worker（不设置 GPU，避免 CUDA 依赖）
        with patch.object(GPUWorker, '_setup_gpu'):
            with patch.object(GPUWorker, '_setup_adapter_config'):
                worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
        
        # 调用 _init_model_rpc
        await worker._init_model_rpc()
        
        # 验证 start_model_process 被调用
        mock_start.assert_called_once_with(port=None, world_size=1)
        
        # 验证 model_rpc 被设置
        assert worker.model_rpc is not None
        assert worker.model_rpc == mock_rpc_client


@pytest.mark.asyncio
async def test_init_model_rpc_calls_init_model(mock_args):
    """测试 _init_model_rpc 调用 init_model"""
    with patch('slora.server.router.gpu_worker.start_model_process') as mock_start:
        # 模拟 start_model_process 返回一个 ModelRpcClient
        mock_rpc_client = AsyncMock()
        mock_rpc_client.init_model = AsyncMock()
        mock_start.return_value = mock_rpc_client
        
        # 创建 Worker
        with patch.object(GPUWorker, '_setup_gpu'):
            with patch.object(GPUWorker, '_setup_adapter_config'):
                worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
        
        # 调用 _init_model_rpc
        await worker._init_model_rpc()
        
        # 验证 init_model 被调用
        mock_rpc_client.init_model.assert_called_once()
        
        # 验证调用参数
        call_args = mock_rpc_client.init_model.call_args
        assert call_args.kwargs['rank_id'] == 0
        assert call_args.kwargs['world_size'] == 1
        assert call_args.kwargs['weight_dir'] == '/path/to/model'
        assert call_args.kwargs['adapter_dirs'] == []
        assert call_args.kwargs['max_total_token_num'] == 10000
        assert call_args.kwargs['load_way'] == 'HF'
        assert call_args.kwargs['mode'] == []
        assert call_args.kwargs['prefetch_stream'] is None


@pytest.mark.asyncio
async def test_init_model_rpc_creates_input_params(mock_args):
    """测试 _init_model_rpc 创建 InputParams 对象"""
    with patch('slora.server.router.gpu_worker.start_model_process') as mock_start:
        # 模拟 start_model_process 返回一个 ModelRpcClient
        mock_rpc_client = AsyncMock()
        mock_rpc_client.init_model = AsyncMock()
        mock_start.return_value = mock_rpc_client
        
        # 创建 Worker
        with patch.object(GPUWorker, '_setup_gpu'):
            with patch.object(GPUWorker, '_setup_adapter_config'):
                worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
        
        # 调用 _init_model_rpc
        await worker._init_model_rpc()
        
        # 验证 init_model 被调用
        mock_rpc_client.init_model.assert_called_once()
        
        # 验证 input_params 参数
        call_args = mock_rpc_client.init_model.call_args
        input_params = call_args.kwargs['input_params']
        
        # 验证 InputParams 的关键属性
        assert input_params.max_req_total_len == 2048
        assert input_params.max_total_token_num == 10000
        assert input_params.batch_max_tokens == 2000
        assert input_params.running_max_req_size == 10
        assert input_params.scheduler == 'slora'
        assert input_params.no_lora == False
        assert input_params.evict_interval_threshold == 0.9
        assert input_params.evict_interval_ratio == 0.5


@pytest.mark.asyncio
async def test_init_model_rpc_handles_error(mock_args):
    """测试 _init_model_rpc 处理错误"""
    with patch('slora.server.router.gpu_worker.start_model_process') as mock_start:
        # 模拟 start_model_process 抛出异常
        mock_start.side_effect = Exception("Failed to start model process")
        
        # 创建 Worker
        with patch.object(GPUWorker, '_setup_gpu'):
            with patch.object(GPUWorker, '_setup_adapter_config'):
                worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
        
        # 调用 _init_model_rpc 应该抛出异常
        with pytest.raises(Exception, match="Failed to start model process"):
            await worker._init_model_rpc()


@pytest.mark.asyncio
async def test_init_model_rpc_with_lora_dirs(mock_args):
    """测试 _init_model_rpc 处理 LoRA 目录"""
    # 添加 LoRA 目录
    mock_args.lora_dirs = ['/path/to/lora1', '/path/to/lora2']
    
    with patch('slora.server.router.gpu_worker.start_model_process') as mock_start:
        # 模拟 start_model_process 返回一个 ModelRpcClient
        mock_rpc_client = AsyncMock()
        mock_rpc_client.init_model = AsyncMock()
        mock_start.return_value = mock_rpc_client
        
        # 创建 Worker
        with patch.object(GPUWorker, '_setup_gpu'):
            with patch.object(GPUWorker, '_setup_adapter_config'):
                worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
        
        # 调用 _init_model_rpc
        await worker._init_model_rpc()
        
        # 验证 init_model 被调用，并传递了 adapter_dirs
        call_args = mock_rpc_client.init_model.call_args
        assert call_args.kwargs['adapter_dirs'] == ['/path/to/lora1', '/path/to/lora2']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
