"""
测试 DataParallelRouterManager 类框架

测试覆盖：
- GPU 检测逻辑
- GPU ID 解析
- 端口分配
- 初始化参数验证
"""

import pytest
import torch
import argparse
from unittest.mock import patch, MagicMock

from slora.server.router.dp_manager import DataParallelRouterManager


class TestDataParallelRouterManagerFramework:
    """测试 DataParallelRouterManager 类框架"""
    
    def test_init_with_num_workers(self):
        """测试指定 num_workers 的初始化"""
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,1,2"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        assert manager.num_workers == 3
        assert manager.gpu_ids == [0, 1, 2]
        assert manager.router_port == 10000
        assert manager.response_port == 10001
        assert manager.detoken_port == 10002
        assert len(manager.workers) == 0  # 尚未启动
        assert len(manager.worker_ports) == 0  # 尚未分配
        assert manager.merger_process is None  # 尚未启动
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_init_auto_detect_gpus(self, mock_is_available, mock_device_count):
        """测试自动检测 GPU 数量"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 4
        
        args = argparse.Namespace()
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        assert manager.num_workers == 4
        assert manager.gpu_ids == [0, 1, 2, 3]
    
    @patch('torch.cuda.is_available')
    def test_detect_gpus_no_cuda(self, mock_is_available):
        """测试 CUDA 不可用时的错误处理"""
        mock_is_available.return_value = False
        
        args = argparse.Namespace()
        
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_detect_gpus_no_devices(self, mock_is_available, mock_device_count):
        """测试没有检测到 GPU 时的错误处理"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 0
        
        args = argparse.Namespace()
        
        with pytest.raises(RuntimeError, match="No GPUs detected"):
            DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
    
    def test_parse_gpu_ids_valid(self):
        """测试解析有效的 GPU ID 列表"""
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,1,2"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        assert manager.gpu_ids == [0, 1, 2]
    
    def test_parse_gpu_ids_with_spaces(self):
        """测试解析带空格的 GPU ID 列表"""
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0, 1, 2"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        assert manager.gpu_ids == [0, 1, 2]
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_parse_gpu_ids_default(self, mock_is_available, mock_device_count):
        """测试未指定 GPU IDs 时使用默认值"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 3
        
        args = argparse.Namespace()
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        assert manager.gpu_ids == [0, 1, 2]
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_parse_gpu_ids_invalid_format(self, mock_is_available, mock_device_count):
        """测试无效的 GPU ID 格式"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 4
        
        args = argparse.Namespace(
            num_workers=2,
            gpu_ids="0,abc"
        )
        
        with pytest.raises(ValueError, match="Failed to parse GPU IDs"):
            DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_parse_gpu_ids_out_of_range(self, mock_is_available, mock_device_count):
        """测试 GPU ID 超出范围"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,1,5"
        )
        
        with pytest.raises(ValueError, match="Invalid GPU ID 5"):
            DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
    
    def test_gpu_ids_count_mismatch(self):
        """测试 GPU ID 数量与 Worker 数量不匹配"""
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,1"  # 只有 2 个 GPU ID
        )
        
        with pytest.raises(ValueError, match="Number of GPU IDs .* does not match"):
            DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
    
    def test_allocate_ports(self):
        """测试端口分配"""
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,1,2"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        manager._allocate_ports()
        
        assert len(manager.worker_ports) == 3
        assert manager.worker_ports == [50000, 50001, 50002]
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_allocate_ports_multiple_workers(self, mock_is_available, mock_device_count):
        """测试为多个 Worker 分配端口"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 5
        
        args = argparse.Namespace(
            num_workers=5,
            gpu_ids="0,1,2,3,4"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        manager._allocate_ports()
        
        assert len(manager.worker_ports) == 5
        assert manager.worker_ports == [50000, 50001, 50002, 50003, 50004]
        # 验证端口唯一性
        assert len(set(manager.worker_ports)) == 5
    
    def test_router_initialization(self):
        """测试路由器初始化"""
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,1,2"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        assert manager.router is not None
        assert manager.router.num_workers == 3
    
    def test_attributes_initialization(self):
        """测试所有属性正确初始化"""
        args = argparse.Namespace(
            num_workers=2,
            gpu_ids="0,1",
            model_dir="/path/to/model"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # 验证基本属性
        assert manager.args == args
        assert manager.router_port == 10000
        assert manager.response_port == 10001
        assert manager.detoken_port == 10002
        assert manager.num_workers == 2
        assert manager.gpu_ids == [0, 1]
        
        # 验证列表初始化
        assert isinstance(manager.workers, list)
        assert isinstance(manager.worker_ports, list)
        assert isinstance(manager.request_senders, list)
        
        # 验证 ZMQ 相关属性初始化为 None
        assert manager.context is None
        assert manager.request_receiver is None
        assert manager.merger_process is None
    
    @patch('slora.server.router.dp_manager.mp.Process')
    def test_start_response_merger(self, mock_process):
        """测试启动 Response Merger 进程"""
        args = argparse.Namespace(
            num_workers=2,
            gpu_ids="0,1"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # 创建 mock 进程实例
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        
        # 调用 _start_response_merger
        manager._start_response_merger()
        
        # 验证进程被创建
        mock_process.assert_called_once()
        call_args = mock_process.call_args
        
        # 验证参数
        assert call_args[1]['args'] == (10001, 10002)  # response_port, detoken_port
        assert call_args[1]['name'] == "ResponseMerger"
        
        # 验证进程被启动
        mock_process_instance.start.assert_called_once()
        
        # 验证 merger_process 被设置
        assert manager.merger_process == mock_process_instance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
