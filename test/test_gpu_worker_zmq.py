"""
GPU Worker ZMQ 通信测试

测试 GPUWorker 类的 ZMQ 通信设置功能。
"""

import os
import sys
import pytest
import argparse
from unittest.mock import patch, MagicMock, call
import zmq

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slora.server.router.gpu_worker import GPUWorker


class TestGPUWorkerZMQ:
    """测试 GPUWorker ZMQ 通信设置"""
    
    def test_zmq_setup_creates_sockets(self):
        """测试 ZMQ 设置创建正确的 socket"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            # 创建 mock context 和 sockets
            mock_context = MagicMock()
            mock_pull_socket = MagicMock()
            mock_push_socket = MagicMock()
            
            # 配置 socket 方法返回不同的 mock
            def socket_side_effect(socket_type):
                if socket_type == zmq.PULL:
                    return mock_pull_socket
                elif socket_type == zmq.PUSH:
                    return mock_push_socket
                return MagicMock()
            
            mock_context.socket.side_effect = socket_side_effect
            
            # 创建 Worker
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Patch zmq.asyncio.Context to return our mock
            with patch('slora.server.router.gpu_worker.zmq.asyncio.Context', return_value=mock_context):
                worker._setup_zmq(request_port=50000, response_port=50001)
            
            # 验证创建了两个 socket
            assert mock_context.socket.call_count == 2
            
            # 验证 request_receiver 和 response_sender 被设置
            assert worker.request_receiver is mock_pull_socket
            assert worker.response_sender is mock_push_socket
            
            # 验证 context 被存储
            assert worker.context is mock_context
    
    def test_zmq_pull_socket_connects_to_request_port(self):
        """测试 PULL socket 连接到正确的请求端口"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            mock_context = MagicMock()
            mock_pull_socket = MagicMock()
            mock_push_socket = MagicMock()
            
            def socket_side_effect(socket_type):
                if socket_type == zmq.PULL:
                    return mock_pull_socket
                elif socket_type == zmq.PUSH:
                    return mock_push_socket
                return MagicMock()
            
            mock_context.socket.side_effect = socket_side_effect
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            with patch('slora.server.router.gpu_worker.zmq.asyncio.Context', return_value=mock_context):
                worker._setup_zmq(request_port=50000, response_port=50001)
            
            # 验证 PULL socket 连接到请求端口
            mock_pull_socket.connect.assert_called_once_with("tcp://127.0.0.1:50000")
    
    def test_zmq_push_socket_connects_to_response_port(self):
        """测试 PUSH socket 连接到正确的响应端口"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            mock_context = MagicMock()
            mock_pull_socket = MagicMock()
            mock_push_socket = MagicMock()
            
            def socket_side_effect(socket_type):
                if socket_type == zmq.PULL:
                    return mock_pull_socket
                elif socket_type == zmq.PUSH:
                    return mock_push_socket
                return MagicMock()
            
            mock_context.socket.side_effect = socket_side_effect
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            with patch('slora.server.router.gpu_worker.zmq.asyncio.Context', return_value=mock_context):
                worker._setup_zmq(request_port=50000, response_port=50001)
            
            # 验证 PUSH socket 连接到响应端口
            mock_push_socket.connect.assert_called_once_with("tcp://127.0.0.1:50001")
    
    def test_zmq_setup_with_different_ports(self):
        """测试使用不同端口设置 ZMQ"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        test_cases = [
            (50000, 50001),
            (50100, 50101),
            (60000, 60001),
        ]
        
        for request_port, response_port in test_cases:
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.set_device'):
                
                mock_context = MagicMock()
                mock_pull_socket = MagicMock()
                mock_push_socket = MagicMock()
                
                def socket_side_effect(socket_type):
                    if socket_type == zmq.PULL:
                        return mock_pull_socket
                    elif socket_type == zmq.PUSH:
                        return mock_push_socket
                    return MagicMock()
                
                mock_context.socket.side_effect = socket_side_effect
                
                worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
                
                with patch('slora.server.router.gpu_worker.zmq.asyncio.Context', return_value=mock_context):
                    worker._setup_zmq(request_port=request_port, response_port=response_port)
                
                # 验证连接到正确的端口
                mock_pull_socket.connect.assert_called_with(f"tcp://127.0.0.1:{request_port}")
                mock_push_socket.connect.assert_called_with(f"tcp://127.0.0.1:{response_port}")
    
    def test_zmq_context_stored_in_worker(self):
        """测试 ZMQ context 被正确存储在 Worker 中"""
        args = argparse.Namespace(model_dir="/fake/path")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            mock_context = MagicMock()
            mock_pull_socket = MagicMock()
            mock_push_socket = MagicMock()
            
            def socket_side_effect(socket_type):
                if socket_type == zmq.PULL:
                    return mock_pull_socket
                elif socket_type == zmq.PUSH:
                    return mock_push_socket
                return MagicMock()
            
            mock_context.socket.side_effect = socket_side_effect
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证初始状态
            assert worker.context is None
            
            # 设置 ZMQ
            with patch('slora.server.router.gpu_worker.zmq.asyncio.Context', return_value=mock_context):
                worker._setup_zmq(request_port=50000, response_port=50001)
            
            # 验证 context 被存储
            assert worker.context is mock_context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
