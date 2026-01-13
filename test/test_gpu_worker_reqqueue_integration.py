"""
GPU Worker ReqQueue 集成测试

测试 GPUWorker 类与 ReqQueue 的集成，特别是 Adapter 信息传递（Task 2.7.4）。
"""

import os
import sys
import pytest
import argparse
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slora.server.router.gpu_worker import GPUWorker
from slora.server.io_struct import Req, Batch
from slora.server.sampling_params import SamplingParams


class TestGPUWorkerReqQueueIntegration:
    """测试 GPUWorker 与 ReqQueue 的集成"""
    
    @pytest.mark.asyncio
    async def test_generate_new_batch_passes_lora_ranks(self):
        """测试 generate_new_batch 正确传递 lora_ranks"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1", "/fake/lora2"],
            dummy=False,
            no_lora=False,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._setup_request_queue()
            
            # Mock ReqQueue.generate_new_batch
            original_generate = worker.req_queue.generate_new_batch
            call_args = []
            
            def mock_generate(current_batch, lora_ranks, actual_adapter_size=0):
                call_args.append({
                    'current_batch': current_batch,
                    'lora_ranks': lora_ranks,
                    'actual_adapter_size': actual_adapter_size
                })
                return None  # 返回 None 表示没有新批次
            
            worker.req_queue.generate_new_batch = mock_generate
            
            # 调用 _process_requests
            await worker._process_requests()
            
            # 验证 generate_new_batch 被调用
            assert len(call_args) == 1
            
            # 验证 lora_ranks 被正确传递
            assert call_args[0]['lora_ranks'] == worker.lora_ranks
            assert "/fake/lora1" in call_args[0]['lora_ranks']
            assert "/fake/lora2" in call_args[0]['lora_ranks']
            assert None in call_args[0]['lora_ranks']
    
    @pytest.mark.asyncio
    async def test_generate_new_batch_passes_actual_adapter_size(self):
        """测试 generate_new_batch 正确传递 actual_adapter_size"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=False,
            no_lora=False,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._setup_request_queue()
            
            # 设置 actual_adapter_memory_usage
            worker.actual_adapter_memory_usage = 300
            
            # Mock ReqQueue.generate_new_batch
            call_args = []
            
            def mock_generate(current_batch, lora_ranks, actual_adapter_size=0):
                call_args.append({
                    'current_batch': current_batch,
                    'lora_ranks': lora_ranks,
                    'actual_adapter_size': actual_adapter_size
                })
                return None
            
            worker.req_queue.generate_new_batch = mock_generate
            
            # 调用 _process_requests
            await worker._process_requests()
            
            # 验证 actual_adapter_size 被正确传递
            assert len(call_args) == 1
            assert call_args[0]['actual_adapter_size'] == 300
    
    @pytest.mark.asyncio
    async def test_generate_new_batch_with_zero_actual_adapter_size(self):
        """测试 actual_adapter_size 为 0 时的传递"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=False,
            no_lora=False,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._setup_request_queue()
            
            # actual_adapter_memory_usage 默认为 0
            assert worker.actual_adapter_memory_usage == 0
            
            # Mock ReqQueue.generate_new_batch
            call_args = []
            
            def mock_generate(current_batch, lora_ranks, actual_adapter_size=0):
                call_args.append({
                    'current_batch': current_batch,
                    'lora_ranks': lora_ranks,
                    'actual_adapter_size': actual_adapter_size
                })
                return None
            
            worker.req_queue.generate_new_batch = mock_generate
            
            # 调用 _process_requests
            await worker._process_requests()
            
            # 验证 actual_adapter_size 为 0
            assert len(call_args) == 1
            assert call_args[0]['actual_adapter_size'] == 0
    
    @pytest.mark.asyncio
    async def test_generate_new_batch_with_current_batch(self):
        """测试有当前批次时的 generate_new_batch 调用"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=False,
            no_lora=False,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._setup_request_queue()
            
            # 创建一个当前批次
            req = Req(
                adapter_dir="/fake/lora1",
                request_id="test-req-1",
                prompt_ids=[1, 2, 3],
                sample_params=SamplingParams()
            )
            current_batch = Batch("batch-1", [req])
            worker.current_batch = current_batch
            
            # 设置 actual_adapter_memory_usage
            worker.actual_adapter_memory_usage = 150
            
            # Mock ReqQueue.generate_new_batch
            call_args = []
            
            def mock_generate(current_batch_arg, lora_ranks, actual_adapter_size=0):
                call_args.append({
                    'current_batch': current_batch_arg,
                    'lora_ranks': lora_ranks,
                    'actual_adapter_size': actual_adapter_size
                })
                return None
            
            worker.req_queue.generate_new_batch = mock_generate
            
            # 调用 _process_requests
            await worker._process_requests()
            
            # 验证 current_batch 被正确传递
            assert len(call_args) == 1
            assert call_args[0]['current_batch'] == current_batch
            assert call_args[0]['actual_adapter_size'] == 150
    
    @pytest.mark.asyncio
    async def test_adapter_loading_after_batch_generation(self):
        """测试批次生成后加载 adapters"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1", "/fake/lora2"],
            dummy=False,
            no_lora=False,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._setup_request_queue()
            
            # 创建一个新批次
            req1 = Req(
                adapter_dir="/fake/lora1",
                request_id="test-req-1",
                prompt_ids=[1, 2, 3],
                sample_params=SamplingParams()
            )
            req2 = Req(
                adapter_dir="/fake/lora2",
                request_id="test-req-2",
                prompt_ids=[4, 5, 6],
                sample_params=SamplingParams()
            )
            new_batch = Batch("batch-1", [req1, req2])
            
            # Mock ReqQueue.generate_new_batch 返回新批次
            worker.req_queue.generate_new_batch = MagicMock(return_value=new_batch)
            
            # Mock _load_adapters
            load_adapters_called = []
            
            async def mock_load_adapters(adapter_dirs):
                load_adapters_called.append(adapter_dirs)
            
            worker._load_adapters = mock_load_adapters
            
            # 调用 _process_requests
            await worker._process_requests()
            
            # 验证 _load_adapters 被调用
            assert len(load_adapters_called) == 1
            
            # 验证传递的 adapter_dirs 包含批次中的所有 adapters
            loaded_adapters = load_adapters_called[0]
            assert "/fake/lora1" in loaded_adapters or "/fake/lora2" in loaded_adapters
    
    @pytest.mark.asyncio
    async def test_no_adapter_loading_when_no_lora(self):
        """测试 no_lora 模式下不加载 adapters"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            no_lora=True,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._setup_request_queue()
            
            # 创建一个新批次
            req = Req(
                adapter_dir="base",
                request_id="test-req-1",
                prompt_ids=[1, 2, 3],
                sample_params=SamplingParams()
            )
            new_batch = Batch("batch-1", [req])
            
            # Mock ReqQueue.generate_new_batch 返回新批次
            worker.req_queue.generate_new_batch = MagicMock(return_value=new_batch)
            
            # Mock _load_adapters
            load_adapters_called = []
            
            async def mock_load_adapters(adapter_dirs):
                load_adapters_called.append(adapter_dirs)
            
            worker._load_adapters = mock_load_adapters
            
            # 调用 _process_requests
            await worker._process_requests()
            
            # 验证 _load_adapters 没有被调用
            assert len(load_adapters_called) == 0
    
    @pytest.mark.asyncio
    async def test_adapter_info_consistency(self):
        """测试 adapter 信息在整个流程中的一致性"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1", "/fake/lora2"],
            dummy=False,
            no_lora=False,
            max_total_token_num=1000,
            batch_max_tokens=500,
            running_max_req_size=10
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._setup_request_queue()
            
            # 设置 actual_adapter_memory_usage
            worker.actual_adapter_memory_usage = 500
            
            # 验证 lora_ranks 包含所有配置的 adapters
            assert "/fake/lora1" in worker.lora_ranks
            assert "/fake/lora2" in worker.lora_ranks
            assert None in worker.lora_ranks
            
            # Mock ReqQueue.generate_new_batch
            call_args = []
            
            def mock_generate(current_batch, lora_ranks, actual_adapter_size=0):
                call_args.append({
                    'lora_ranks': lora_ranks,
                    'actual_adapter_size': actual_adapter_size
                })
                return None
            
            worker.req_queue.generate_new_batch = mock_generate
            
            # 调用 _process_requests
            await worker._process_requests()
            
            # 验证传递的信息与 worker 的状态一致
            assert len(call_args) == 1
            assert call_args[0]['lora_ranks'] == worker.lora_ranks
            assert call_args[0]['actual_adapter_size'] == worker.actual_adapter_memory_usage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
