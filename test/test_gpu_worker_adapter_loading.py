"""
GPU Worker Adapter 加载测试

测试 GPUWorker 类的 Adapter 加载/卸载功能（Task 2.7.3）。
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


class TestGPUWorkerAdapterLoading:
    """测试 GPUWorker Adapter 加载功能"""
    
    @pytest.mark.asyncio
    async def test_load_adapters_with_no_lora(self):
        """测试 no_lora 模式下的 adapter 加载"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            no_lora=True
        )
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 调用加载方法（应该直接返回，不执行任何操作）
            adapter_dirs = {"/fake/lora1", "/fake/lora2"}
            await worker._load_adapters(adapter_dirs)
            
            # 验证没有报错，正常返回
            assert True
    
    @pytest.mark.asyncio
    async def test_load_adapters_without_model_rpc(self):
        """测试没有 model_rpc 时的 adapter 加载"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=False,
            no_lora=False
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 确保没有 model_rpc
            assert not hasattr(worker, 'model_rpc') or worker.model_rpc is None
            
            # 调用加载方法（应该输出警告并返回）
            adapter_dirs = {"/fake/lora1"}
            await worker._load_adapters(adapter_dirs)
            
            # 验证没有报错，正常返回
            assert True
    
    @pytest.mark.asyncio
    async def test_load_adapters_with_empty_set(self):
        """测试空 adapter 集合的加载"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=False,
            no_lora=False
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Mock model_rpc
            mock_rpc = MagicMock()
            mock_rpc.load_adapters = AsyncMock()
            worker.model_rpc = mock_rpc
            
            # 调用加载方法（空集合）
            adapter_dirs = set()
            await worker._load_adapters(adapter_dirs)
            
            # 验证 load_adapters 没有被调用
            mock_rpc.load_adapters.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_load_adapters_success(self):
        """测试成功加载 adapters"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1", "/fake/lora2"],
            dummy=False,
            no_lora=False
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Mock model_rpc
            mock_rpc = MagicMock()
            mock_rpc.load_adapters = AsyncMock()
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [100, 200],
                'used_cells': 400,
                'total_cells': 1000,
                'usage_ratio': 0.4
            })
            worker.model_rpc = mock_rpc
            
            # 调用加载方法
            adapter_dirs = {"/fake/lora1", "/fake/lora2"}
            await worker._load_adapters(adapter_dirs)
            
            # 验证 load_adapters 被调用
            mock_rpc.load_adapters.assert_called_once_with(adapter_dirs)
            
            # 验证 check_lora_memory 被调用（更新实际占用）
            mock_rpc.check_lora_memory.assert_called_once()
            
            # 验证实际占用被更新
            assert worker.actual_adapter_memory_usage == 300
    
    @pytest.mark.asyncio
    async def test_load_adapters_with_single_adapter(self):
        """测试加载单个 adapter"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=False,
            no_lora=False
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Mock model_rpc
            mock_rpc = MagicMock()
            mock_rpc.load_adapters = AsyncMock()
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [150],
                'used_cells': 250,
                'total_cells': 1000,
                'usage_ratio': 0.25
            })
            worker.model_rpc = mock_rpc
            
            # 调用加载方法
            adapter_dirs = {"/fake/lora1"}
            await worker._load_adapters(adapter_dirs)
            
            # 验证 load_adapters 被调用
            mock_rpc.load_adapters.assert_called_once_with(adapter_dirs)
            
            # 验证实际占用被更新
            assert worker.actual_adapter_memory_usage == 150
    
    @pytest.mark.asyncio
    async def test_load_adapters_with_exception(self):
        """测试加载 adapter 时发生异常"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=False,
            no_lora=False
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Mock model_rpc 抛出异常
            mock_rpc = MagicMock()
            mock_rpc.load_adapters = AsyncMock(side_effect=Exception("RPC error"))
            worker.model_rpc = mock_rpc
            
            # 调用加载方法（不应该抛出异常）
            adapter_dirs = {"/fake/lora1"}
            await worker._load_adapters(adapter_dirs)
            
            # 验证 load_adapters 被调用
            mock_rpc.load_adapters.assert_called_once_with(adapter_dirs)
            
            # 验证没有崩溃，正常返回
            assert True
    
    @pytest.mark.asyncio
    async def test_load_adapters_multiple_times(self):
        """测试多次加载不同的 adapters"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1", "/fake/lora2", "/fake/lora3"],
            dummy=False,
            no_lora=False
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Mock model_rpc
            mock_rpc = MagicMock()
            mock_rpc.load_adapters = AsyncMock()
            worker.model_rpc = mock_rpc
            
            # 第一次加载
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [100],
                'used_cells': 200,
                'total_cells': 1000,
                'usage_ratio': 0.2
            })
            await worker._load_adapters({"/fake/lora1"})
            assert worker.actual_adapter_memory_usage == 100
            
            # 第二次加载
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [100, 150],
                'used_cells': 350,
                'total_cells': 1000,
                'usage_ratio': 0.35
            })
            await worker._load_adapters({"/fake/lora2"})
            assert worker.actual_adapter_memory_usage == 250
            
            # 第三次加载
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [100, 150, 200],
                'used_cells': 550,
                'total_cells': 1000,
                'usage_ratio': 0.55
            })
            await worker._load_adapters({"/fake/lora3"})
            assert worker.actual_adapter_memory_usage == 450
            
            # 验证 load_adapters 被调用了 3 次
            assert mock_rpc.load_adapters.call_count == 3
    
    @pytest.mark.asyncio
    async def test_load_adapters_updates_actual_adapter_size(self):
        """测试加载 adapter 后 actual_adapter_size 被同步更新"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=False,
            no_lora=False
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 初始值应该为 0
            assert worker.actual_adapter_size == 0
            assert worker.actual_adapter_memory_usage == 0
            
            # Mock model_rpc
            mock_rpc = MagicMock()
            mock_rpc.load_adapters = AsyncMock()
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [300],
                'used_cells': 400,
                'total_cells': 1000,
                'usage_ratio': 0.4
            })
            worker.model_rpc = mock_rpc
            
            # 调用加载方法
            adapter_dirs = {"/fake/lora1"}
            await worker._load_adapters(adapter_dirs)
            
            # 验证两个属性都被更新
            assert worker.actual_adapter_memory_usage == 300
            assert worker.actual_adapter_size == 300
    
    @pytest.mark.asyncio
    async def test_load_adapters_with_large_adapter_set(self):
        """测试加载大量 adapters"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=[f"/fake/lora{i}" for i in range(10)],
            dummy=False,
            no_lora=False
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Mock model_rpc
            mock_rpc = MagicMock()
            mock_rpc.load_adapters = AsyncMock()
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [100] * 10,  # 10 个 adapter，每个 100 cells
                'used_cells': 1100,
                'total_cells': 2000,
                'usage_ratio': 0.55
            })
            worker.model_rpc = mock_rpc
            
            # 调用加载方法
            adapter_dirs = {f"/fake/lora{i}" for i in range(10)}
            await worker._load_adapters(adapter_dirs)
            
            # 验证 load_adapters 被调用
            mock_rpc.load_adapters.assert_called_once()
            
            # 验证实际占用被正确计算（10 * 100 = 1000）
            assert worker.actual_adapter_memory_usage == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
