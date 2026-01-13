"""
GPU Worker 综合测试

整合所有 GPU Worker 功能的综合测试套件，包括：
- GPU 环境设置
- ZMQ 通信
- 模型加载
- 请求处理
- 消息接收和发送
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


class TestGPUWorkerComprehensive:
    """GPU Worker 综合测试"""
    
    # ==================== GPU 环境设置测试 ====================
    
    def test_worker_initialization_complete(self):
        """测试 Worker 完整初始化流程"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            max_total_token_num=1000
        )
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device') as mock_set_device:
            
            worker = GPUWorker(worker_id=0, gpu_id=2, args=args)
            
            # 验证所有属性正确初始化
            assert worker.worker_id == 0
            assert worker.gpu_id == 2
            assert worker.args == args
            assert worker.model is None
            assert worker.adapter_cache == {}
            assert worker.context is None
            assert worker.request_receiver is None
            assert worker.response_sender is None
            
            # 验证 GPU 设置
            assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
            mock_set_device.assert_called_once_with(0)
    
    def test_gpu_isolation_between_workers(self):
        """测试多个 Worker 之间的 GPU 隔离"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            # 创建多个 Worker，验证 GPU 隔离
            worker1 = GPUWorker(worker_id=0, gpu_id=0, args=args)
            assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
            
            worker2 = GPUWorker(worker_id=1, gpu_id=3, args=args)
            assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
            
            worker3 = GPUWorker(worker_id=2, gpu_id=7, args=args)
            assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'
    
    def test_cuda_unavailable_raises_error(self):
        """测试 CUDA 不可用时抛出错误"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                GPUWorker(worker_id=0, gpu_id=0, args=args)
    
    # ==================== ZMQ 通信测试 ====================
    
    def test_zmq_setup_complete(self):
        """测试 ZMQ 完整设置"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            mock_context = MagicMock()
            mock_pull_socket = MagicMock()
            mock_push_socket = MagicMock()
            
            def socket_side_effect(socket_type):
                import zmq
                if socket_type == zmq.PULL:
                    return mock_pull_socket
                elif socket_type == zmq.PUSH:
                    return mock_push_socket
                return MagicMock()
            
            mock_context.socket.side_effect = socket_side_effect
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            with patch('slora.server.router.gpu_worker.zmq.asyncio.Context', return_value=mock_context):
                worker._setup_zmq(request_port=50000, response_port=50001)
            
            # 验证 sockets 创建和连接
            assert worker.context is mock_context
            assert worker.request_receiver is mock_pull_socket
            assert worker.response_sender is mock_push_socket
            mock_pull_socket.connect.assert_called_once_with("tcp://127.0.0.1:50000")
            mock_push_socket.connect.assert_called_once_with("tcp://127.0.0.1:50001")
    
    def test_zmq_different_ports(self):
        """测试不同端口配置的 ZMQ 设置"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        port_configs = [
            (50000, 50001),
            (60000, 60001),
            (55555, 55556),
        ]
        
        for request_port, response_port in port_configs:
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.set_device'):
                
                mock_context = MagicMock()
                mock_pull_socket = MagicMock()
                mock_push_socket = MagicMock()
                
                def socket_side_effect(socket_type):
                    import zmq
                    if socket_type == zmq.PULL:
                        return mock_pull_socket
                    elif socket_type == zmq.PUSH:
                        return mock_push_socket
                    return MagicMock()
                
                mock_context.socket.side_effect = socket_side_effect
                
                worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
                
                with patch('slora.server.router.gpu_worker.zmq.asyncio.Context', return_value=mock_context):
                    worker._setup_zmq(request_port=request_port, response_port=response_port)
                
                mock_pull_socket.connect.assert_called_with(f"tcp://127.0.0.1:{request_port}")
                mock_push_socket.connect.assert_called_with(f"tcp://127.0.0.1:{response_port}")
    
    # ==================== 模型加载测试 ====================
    
    def test_load_llama_model_complete(self):
        """测试完整的 Llama 模型加载流程"""
        args = argparse.Namespace(
            model_dir="/fake/llama/path",
            max_total_token_num=2000,
            mem_adapter_size=200,
            load_way='HF',
            mode=[],
            dummy=False
        )
        
        mock_model_cfg = {
            "model_type": "llama",
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_hidden_layers": 32
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg), \
             patch('slora.server.router.gpu_worker.LlamaTpPartModel') as mock_llama_model:
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._load_model()
            
            # 验证模型加载参数
            mock_llama_model.assert_called_once()
            call_kwargs = mock_llama_model.call_args[1]
            assert call_kwargs['tp_rank'] == 0
            assert call_kwargs['world_size'] == 1
            assert call_kwargs['weight_dir'] == "/fake/llama/path"
            assert call_kwargs['max_total_token_num'] == 2000
            assert call_kwargs['mem_adapter_size'] == 200
            assert call_kwargs['load_way'] == 'HF'
            assert call_kwargs['mode'] == []
            assert call_kwargs['dummy'] == False
    
    def test_load_llama2_model_with_gqa(self):
        """测试 Llama2 模型（GQA）加载"""
        args = argparse.Namespace(
            model_dir="/fake/llama2/path",
            max_total_token_num=1000,
            mem_adapter_size=100,
            load_way='HF',
            mode=[],
            dummy=False
        )
        
        mock_model_cfg = {
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,  # GQA 特征
            "hidden_size": 4096
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg), \
             patch('slora.server.router.gpu_worker.Llama2TpPartModel') as mock_llama2_model:
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._load_model()
            
            # 验证 Llama2 模型被调用
            mock_llama2_model.assert_called_once()
            assert worker.model is not None
    
    def test_model_loading_error_handling(self):
        """测试模型加载错误处理"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            max_total_token_num=1000,
            dummy=False
        )
        
        mock_model_cfg = {
            "model_type": "llama",
            "num_attention_heads": 32
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg), \
             patch('slora.server.router.gpu_worker.LlamaTpPartModel', side_effect=RuntimeError("Model load failed")):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            with pytest.raises(RuntimeError, match="Model load failed"):
                worker._load_model()
    
    # ==================== 请求处理测试 ====================
    
    @pytest.mark.asyncio
    async def test_complete_request_response_cycle(self):
        """测试完整的请求-响应周期"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Mock ZMQ sockets
            mock_receiver = AsyncMock()
            mock_sender = AsyncMock()
            
            test_request = {
                'request_id': 'req_cycle_test',
                'adapter_dir': None,
                'prompt_ids': [10, 20, 30, 40],
                'sampling_params': {'temperature': 0.8}
            }
            
            mock_receiver.recv_json.return_value = test_request
            worker.request_receiver = mock_receiver
            worker.response_sender = mock_sender
            
            # 执行完整周期
            request = await worker._receive_request()
            response = await worker._process_request(request)
            await worker._send_response(response)
            
            # 验证
            assert request == test_request
            assert response['request_id'] == 'req_cycle_test'
            assert response['worker_id'] == 0
            assert response['success'] is True
            mock_sender.send_json.assert_called_once_with(response)
    
    @pytest.mark.asyncio
    async def test_request_id_preservation(self):
        """测试请求 ID 在整个流程中保持一致"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 测试多个不同的 request_id
            test_ids = ['req_001', 'req_abc', 'req_xyz_123', 'test-request-456']
            
            for request_id in test_ids:
                request = {
                    'request_id': request_id,
                    'adapter_dir': None,
                    'prompt_ids': [1, 2, 3],
                    'sampling_params': {}
                }
                
                response = await worker._process_request(request)
                
                # 验证 request_id 保持一致
                assert response['request_id'] == request_id
    
    @pytest.mark.asyncio
    async def test_worker_id_in_all_responses(self):
        """测试所有响应都包含正确的 worker_id"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        # 测试不同的 worker_id
        for worker_id in [0, 1, 2, 5, 10]:
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.set_device'):
                
                worker = GPUWorker(worker_id=worker_id, gpu_id=0, args=args)
                
                request = {
                    'request_id': 'test_req',
                    'adapter_dir': None,
                    'prompt_ids': [1, 2, 3],
                    'sampling_params': {}
                }
                
                response = await worker._process_request(request)
                
                assert response['worker_id'] == worker_id
    
    @pytest.mark.asyncio
    async def test_error_response_format(self):
        """测试错误响应的格式"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 无效请求（缺少必需字段）
            invalid_request = {
                'adapter_dir': None
                # 缺少 request_id 和 prompt_ids
            }
            
            response = await worker._process_request(invalid_request)
            
            # 验证错误响应格式
            assert response['success'] is False
            assert response['error'] is not None
            assert isinstance(response['error'], str)
            assert response['worker_id'] == 0
            assert 'request_id' in response
            assert 'output_ids' in response
            assert 'metadata' in response
    
    @pytest.mark.asyncio
    async def test_response_message_completeness(self):
        """测试响应消息的完整性"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            request = {
                'request_id': 'complete_test',
                'adapter_dir': '/path/to/adapter',
                'prompt_ids': [100, 200, 300],
                'sampling_params': {'top_p': 0.95, 'temperature': 0.7}
            }
            
            response = await worker._process_request(request)
            
            # 验证所有必需字段存在
            required_fields = [
                'request_id',
                'worker_id',
                'output_ids',
                'metadata',
                'success',
                'error'
            ]
            
            for field in required_fields:
                assert field in response, f"Missing required field: {field}"
            
            # 验证字段类型
            assert isinstance(response['request_id'], str)
            assert isinstance(response['worker_id'], int)
            assert isinstance(response['output_ids'], list)
            assert isinstance(response['metadata'], dict)
            assert isinstance(response['success'], bool)
    
    @pytest.mark.asyncio
    async def test_metadata_contains_useful_info(self):
        """测试元数据包含有用信息"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            request = {
                'request_id': 'metadata_test',
                'adapter_dir': None,
                'prompt_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'sampling_params': {}
            }
            
            response = await worker._process_request(request)
            
            # 验证 metadata 存在且有内容
            assert 'metadata' in response
            assert isinstance(response['metadata'], dict)
            assert len(response['metadata']) > 0
    
    # ==================== 集成测试 ====================
    
    def test_worker_full_initialization_sequence(self):
        """测试 Worker 完整初始化序列"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            max_total_token_num=1000,
            mem_adapter_size=100,
            load_way='HF',
            mode=[],
            dummy=False
        )
        
        mock_model_cfg = {
            "model_type": "llama",
            "num_attention_heads": 32,
            "hidden_size": 4096
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg), \
             patch('slora.server.router.gpu_worker.LlamaTpPartModel'):
            
            # 1. 创建 Worker（初始化 + GPU 设置）
            worker = GPUWorker(worker_id=0, gpu_id=1, args=args)
            assert worker.worker_id == 0
            assert worker.gpu_id == 1
            
            # 2. 设置 ZMQ
            mock_context = MagicMock()
            mock_pull_socket = MagicMock()
            mock_push_socket = MagicMock()
            
            def socket_side_effect(socket_type):
                import zmq
                if socket_type == zmq.PULL:
                    return mock_pull_socket
                elif socket_type == zmq.PUSH:
                    return mock_push_socket
                return MagicMock()
            
            mock_context.socket.side_effect = socket_side_effect
            
            with patch('slora.server.router.gpu_worker.zmq.asyncio.Context', return_value=mock_context):
                worker._setup_zmq(request_port=50000, response_port=50001)
            
            assert worker.context is not None
            assert worker.request_receiver is not None
            assert worker.response_sender is not None
            
            # 3. 加载模型
            worker._load_model()
            assert worker.model is not None
            
            # Worker 现在已完全初始化，可以处理请求


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
