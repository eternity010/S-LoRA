"""
GPU Worker Adapter 配置测试

测试 GPUWorker 类的 Adapter rank 配置功能（Task 2.7.1）。
"""

import os
import sys
import pytest
import argparse
from unittest.mock import patch, MagicMock

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slora.server.router.gpu_worker import GPUWorker


class TestGPUWorkerAdapterConfig:
    """测试 GPUWorker Adapter 配置"""
    
    def test_adapter_config_with_lora_dirs(self):
        """测试有 lora_dirs 时的 Adapter 配置初始化"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1", "/fake/lora2"],
            dummy=False
        )
        
        # Mock get_lora_config 返回不同的 rank 值
        mock_configs = [
            ({"r": 8, "lora_alpha": 16}, "/fake/lora1"),
            ({"r": 16, "lora_alpha": 32}, "/fake/lora2"),
        ]
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', side_effect=mock_configs):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证 lora_ranks 字典被正确初始化
            assert "/fake/lora1" in worker.lora_ranks
            assert "/fake/lora2" in worker.lora_ranks
            assert worker.lora_ranks["/fake/lora1"] == 8
            assert worker.lora_ranks["/fake/lora2"] == 16
            
            # 验证 None 键被添加（base 模型）
            assert None in worker.lora_ranks
            assert worker.lora_ranks[None] == 0
            
            # 验证总共有 3 个条目（2 个 adapter + 1 个 None）
            assert len(worker.lora_ranks) == 3
    
    def test_adapter_config_without_lora_dirs(self):
        """测试没有 lora_dirs 时的 Adapter 配置初始化"""
        args = argparse.Namespace(
            model_dir="/fake/path"
        )
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证只有 None 键（base 模型）
            assert None in worker.lora_ranks
            assert worker.lora_ranks[None] == 0
            assert len(worker.lora_ranks) == 1
    
    def test_adapter_config_with_empty_lora_dirs(self):
        """测试 lora_dirs 为空列表时的 Adapter 配置初始化"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=[],
            dummy=False
        )
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证只有 None 键（base 模型）
            assert None in worker.lora_ranks
            assert worker.lora_ranks[None] == 0
            assert len(worker.lora_ranks) == 1
    
    def test_adapter_config_with_dummy_mode(self):
        """测试 dummy 模式下的 Adapter 配置初始化"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1"],
            dummy=True
        )
        
        mock_config = ({"r": 32, "lora_alpha": 64}, "/fake/lora1")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', return_value=mock_config) as mock_get_config:
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证 get_lora_config 被调用时传递了 dummy=True
            mock_get_config.assert_called_once_with("/fake/lora1", True)
            
            # 验证 rank 被正确读取
            assert worker.lora_ranks["/fake/lora1"] == 32
            assert worker.lora_ranks[None] == 0
    
    def test_adapter_config_with_failed_loading(self):
        """测试 Adapter 配置加载失败时的处理"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=["/fake/lora1", "/fake/lora2"],
            dummy=False
        )
        
        # Mock get_lora_config: 第一个成功，第二个失败
        def mock_get_lora_config(lora_dir, dummy):
            if lora_dir == "/fake/lora1":
                return ({"r": 8, "lora_alpha": 16}, "/fake/lora1")
            else:
                raise Exception("Failed to load config")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', side_effect=mock_get_lora_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证第一个 adapter 成功加载
            assert worker.lora_ranks["/fake/lora1"] == 8
            
            # 验证第二个 adapter 使用默认 rank
            assert worker.lora_ranks["/fake/lora2"] == 8  # 默认值
            
            # 验证 None 键存在
            assert worker.lora_ranks[None] == 0
            
            # 验证总共有 3 个条目
            assert len(worker.lora_ranks) == 3
    
    def test_adapter_config_with_multiple_adapters(self):
        """测试多个 Adapter 的配置初始化"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            lora_dirs=[f"/fake/lora{i}" for i in range(5)],
            dummy=False
        )
        
        # Mock get_lora_config 返回不同的 rank 值
        def mock_get_lora_config(lora_dir, dummy):
            idx = int(lora_dir.split("lora")[1])
            return ({"r": 8 * (idx + 1), "lora_alpha": 16 * (idx + 1)}, lora_dir)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_lora_config', side_effect=mock_get_lora_config):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证所有 adapter 的 rank 被正确读取
            for i in range(5):
                lora_dir = f"/fake/lora{i}"
                assert lora_dir in worker.lora_ranks
                assert worker.lora_ranks[lora_dir] == 8 * (i + 1)
            
            # 验证 None 键存在
            assert worker.lora_ranks[None] == 0
            
            # 验证总共有 6 个条目（5 个 adapter + 1 个 None）
            assert len(worker.lora_ranks) == 6
    
    def test_adapter_config_actual_adapter_size_initialization(self):
        """测试 actual_adapter_size 初始化为 0"""
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
            
            # 验证 actual_adapter_size 初始化为 0
            assert worker.actual_adapter_size == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
