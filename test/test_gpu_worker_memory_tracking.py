"""
GPU Worker 内存跟踪测试

测试 GPUWorker 类的实际内存占用跟踪功能（Task 2.7.2）。
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


class TestGPUWorkerMemoryTracking:
    """测试 GPUWorker 内存跟踪功能"""
    
    def test_actual_adapter_memory_usage_initialization(self):
        """测试 actual_adapter_memory_usage 初始化为 0"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=False
        )
        
        mock_config = ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证 actual_adapter_memory_usage 初始化为 0
            assert worker.actual_adapter_memory_usage == 0
    
    @pytest.mark.asyncio
    async def test_update_actual_adapter_usage_with_no_lora(self):
        """测试 no_lora 模式下的内存占用更新"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            no_lora=True
        )
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 调用更新方法
            await worker._update_actual_adapter_usage()
            
            # 验证内存占用为 0
            assert worker.actual_adapter_memory_usage == 0
    
    @pytest.mark.asyncio
    async def test_update_actual_adapter_usage_without_model_rpc(self):
        """测试没有 model_rpc 时的内存占用更新"""
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
            
            # 调用更新方法（应该直接返回，不报错）
            await worker._update_actual_adapter_usage()
            
            # 验证内存占用保持为 0（因为无法查询）
            assert worker.actual_adapter_memory_usage == 0
    
    @pytest.mark.asyncio
    async def test_update_actual_adapter_usage_with_valid_memory_info(self):
        """测试有效内存信息时的内存占用更新"""
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
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [100, 200, 150],  # 3 个 adapter 的占用
                'used_cells': 500,
                'total_cells': 1000,
                'usage_ratio': 0.5
            })
            worker.model_rpc = mock_rpc
            
            # 调用更新方法
            await worker._update_actual_adapter_usage()
            
            # 验证内存占用被正确计算（100 + 200 + 150 = 450）
            assert worker.actual_adapter_memory_usage == 450
            assert worker.actual_adapter_size == 450  # 应该同步更新
    
    @pytest.mark.asyncio
    async def test_update_actual_adapter_usage_with_empty_adapter_cells(self):
        """测试 adapter_cells 为空时的内存占用更新"""
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
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [],  # 空列表
                'used_cells': 100,
                'total_cells': 1000,
                'usage_ratio': 0.1
            })
            worker.model_rpc = mock_rpc
            
            # 调用更新方法
            await worker._update_actual_adapter_usage()
            
            # 验证内存占用为 0
            assert worker.actual_adapter_memory_usage == 0
            assert worker.actual_adapter_size == 0
    
    @pytest.mark.asyncio
    async def test_update_actual_adapter_usage_with_none_memory_info(self):
        """测试 check_lora_memory 返回 None 时的处理"""
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
            
            # 设置初始值
            worker.actual_adapter_memory_usage = 100
            worker.actual_adapter_size = 100
            
            # Mock model_rpc 返回 None
            mock_rpc = MagicMock()
            mock_rpc.check_lora_memory = AsyncMock(return_value=None)
            worker.model_rpc = mock_rpc
            
            # 调用更新方法
            await worker._update_actual_adapter_usage()
            
            # 验证内存占用保持不变（保守估计）
            assert worker.actual_adapter_memory_usage == 100
            assert worker.actual_adapter_size == 100
    
    @pytest.mark.asyncio
    async def test_update_actual_adapter_usage_with_exception(self):
        """测试查询异常时的处理"""
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
            
            # 设置初始值
            worker.actual_adapter_memory_usage = 200
            worker.actual_adapter_size = 200
            
            # Mock model_rpc 抛出异常
            mock_rpc = MagicMock()
            mock_rpc.check_lora_memory = AsyncMock(side_effect=Exception("RPC error"))
            worker.model_rpc = mock_rpc
            
            # 调用更新方法（不应该抛出异常）
            await worker._update_actual_adapter_usage()
            
            # 验证内存占用保持不变（保守估计）
            assert worker.actual_adapter_memory_usage == 200
            assert worker.actual_adapter_size == 200
    
    @pytest.mark.asyncio
    async def test_update_actual_adapter_usage_with_single_adapter(self):
        """测试单个 adapter 时的内存占用更新"""
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
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [300],  # 单个 adapter
                'used_cells': 400,
                'total_cells': 1000,
                'usage_ratio': 0.4
            })
            worker.model_rpc = mock_rpc
            
            # 调用更新方法
            await worker._update_actual_adapter_usage()
            
            # 验证内存占用被正确计算
            assert worker.actual_adapter_memory_usage == 300
            assert worker.actual_adapter_size == 300
    
    @pytest.mark.asyncio
    async def test_update_actual_adapter_usage_multiple_times(self):
        """测试多次更新内存占用"""
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
            worker.model_rpc = mock_rpc
            
            # 第一次更新：加载了 2 个 adapter
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [100, 200],
                'used_cells': 400,
                'total_cells': 1000,
                'usage_ratio': 0.4
            })
            await worker._update_actual_adapter_usage()
            assert worker.actual_adapter_memory_usage == 300
            
            # 第二次更新：加载了 3 个 adapter
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [100, 200, 150],
                'used_cells': 550,
                'total_cells': 1000,
                'usage_ratio': 0.55
            })
            await worker._update_actual_adapter_usage()
            assert worker.actual_adapter_memory_usage == 450
            
            # 第三次更新：卸载了一些 adapter
            mock_rpc.check_lora_memory = AsyncMock(return_value={
                'adapter_cells': [100],
                'used_cells': 200,
                'total_cells': 1000,
                'usage_ratio': 0.2
            })
            await worker._update_actual_adapter_usage()
            assert worker.actual_adapter_memory_usage == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
