"""
GPU Worker 初始化测试

测试 GPUWorker 类的初始化和 GPU 环境设置功能。
"""

import os
import sys
import pytest
import argparse
from unittest.mock import patch, MagicMock

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slora.server.router.gpu_worker import GPUWorker


class TestGPUWorkerInit:
    """测试 GPUWorker 初始化"""
    
    def test_worker_initialization(self):
        """测试 Worker 基本初始化"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device') as mock_set_device:
            
            worker = GPUWorker(worker_id=0, gpu_id=2, args=args)
            
            # 验证属性设置
            assert worker.worker_id == 0
            assert worker.gpu_id == 2
            assert worker.args == args
            
            # 验证初始状态
            assert worker.model is None
            assert worker.adapter_cache == {}
            assert worker.context is None
            assert worker.request_receiver is None
            assert worker.response_sender is None
    
    def test_gpu_environment_setup(self):
        """测试 GPU 环境设置"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device') as mock_set_device:
            
            worker = GPUWorker(worker_id=1, gpu_id=3, args=args)
            
            # 验证 CUDA_VISIBLE_DEVICES 被正确设置
            assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
            
            # 验证 torch.cuda.set_device 被调用
            mock_set_device.assert_called_once_with(0)
    
    def test_multiple_workers_gpu_isolation(self):
        """测试多个 Worker 的 GPU 隔离"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            # 创建第一个 Worker
            worker1 = GPUWorker(worker_id=0, gpu_id=0, args=args)
            assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
            
            # 创建第二个 Worker（在实际场景中会在不同进程）
            worker2 = GPUWorker(worker_id=1, gpu_id=1, args=args)
            assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
            
            # 验证 Worker ID 不同
            assert worker1.worker_id != worker2.worker_id
            assert worker1.gpu_id != worker2.gpu_id
    
    def test_cuda_not_available_error(self):
        """测试 CUDA 不可用时的错误处理"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                GPUWorker(worker_id=0, gpu_id=0, args=args)
    
    def test_worker_id_and_gpu_id_assignment(self):
        """测试 Worker ID 和 GPU ID 的正确分配"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        test_cases = [
            (0, 0),
            (1, 1),
            (2, 3),
            (5, 7),
        ]
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            for worker_id, gpu_id in test_cases:
                worker = GPUWorker(worker_id=worker_id, gpu_id=gpu_id, args=args)
                assert worker.worker_id == worker_id
                assert worker.gpu_id == gpu_id
                assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
