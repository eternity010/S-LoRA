"""
GPU Worker 请求处理测试

测试 GPUWorker 类的请求处理循环功能。
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


class TestGPUWorkerRequestProcessing:
    """测试 GPUWorker 请求处理"""
    
    @pytest.mark.asyncio
    async def test_receive_request(self):
        """测试接收请求"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Mock ZMQ socket
            mock_receiver = AsyncMock()
            test_request = {
                'request_id': 'req_123',
                'adapter_dir': None,
                'prompt_ids': [1, 2, 3],
                'sampling_params': {}
            }
            mock_receiver.recv_json.return_value = test_request
            worker.request_receiver = mock_receiver
            
            # 测试接收请求
            received = await worker._receive_request()
            
            assert received == test_request
            mock_receiver.recv_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_request_success(self):
        """测试成功处理请求"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            test_request = {
                'request_id': 'req_123',
                'adapter_dir': None,
                'prompt_ids': [1, 2, 3, 4, 5],
                'sampling_params': {'temperature': 0.7}
            }
            
            # 处理请求
            response = await worker._process_request(test_request)
            
            # 验证响应格式
            assert response['request_id'] == 'req_123'
            assert response['worker_id'] == 0
            assert response['success'] is True
            assert response['error'] is None
            assert 'output_ids' in response
            assert 'metadata' in response
            assert isinstance(response['output_ids'], list)
            assert isinstance(response['metadata'], dict)
    
    @pytest.mark.asyncio
    async def test_process_request_with_adapter(self):
        """测试带 adapter 的请求处理"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            test_request = {
                'request_id': 'req_456',
                'adapter_dir': '/path/to/adapter',
                'prompt_ids': [10, 20, 30],
                'sampling_params': {}
            }
            
            # 处理请求
            response = await worker._process_request(test_request)
            
            # 验证响应
            assert response['request_id'] == 'req_456'
            assert response['success'] is True
    
    @pytest.mark.asyncio
    async def test_process_request_error_handling(self):
        """测试请求处理的错误处理"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 缺少必需字段的请求
            invalid_request = {
                'adapter_dir': None,
                'sampling_params': {}
                # 缺少 request_id 和 prompt_ids
            }
            
            # 处理无效请求
            response = await worker._process_request(invalid_request)
            
            # 验证错误响应
            assert response['success'] is False
            assert response['error'] is not None
            assert 'request_id' in response
            assert response['worker_id'] == 0
    
    @pytest.mark.asyncio
    async def test_send_response(self):
        """测试发送响应"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Mock ZMQ socket
            mock_sender = AsyncMock()
            worker.response_sender = mock_sender
            
            test_response = {
                'request_id': 'req_123',
                'worker_id': 0,
                'output_ids': [1, 2, 3],
                'metadata': {},
                'success': True,
                'error': None
            }
            
            # 发送响应
            await worker._send_response(test_response)
            
            # 验证调用
            mock_sender.send_json.assert_called_once_with(test_response)
    
    @pytest.mark.asyncio
    async def test_response_message_format(self):
        """测试响应消息格式"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=1, gpu_id=2, args=args)
            
            test_request = {
                'request_id': 'req_789',
                'adapter_dir': None,
                'prompt_ids': [100, 200],
                'sampling_params': {'top_p': 0.9}
            }
            
            response = await worker._process_request(test_request)
            
            # 验证所有必需字段
            required_fields = ['request_id', 'worker_id', 'output_ids', 'metadata', 'success', 'error']
            for field in required_fields:
                assert field in response, f"Missing required field: {field}"
            
            # 验证字段类型
            assert isinstance(response['request_id'], str)
            assert isinstance(response['worker_id'], int)
            assert isinstance(response['output_ids'], list)
            assert isinstance(response['metadata'], dict)
            assert isinstance(response['success'], bool)
            assert response['error'] is None or isinstance(response['error'], str)
    
    @pytest.mark.asyncio
    async def test_request_id_consistency(self):
        """测试请求 ID 一致性（Property 3）"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 测试多个不同的 request_id
            test_request_ids = ['req_001', 'req_002', 'req_003', 'req_xyz', 'req_abc']
            
            for request_id in test_request_ids:
                test_request = {
                    'request_id': request_id,
                    'adapter_dir': None,
                    'prompt_ids': [1, 2, 3],
                    'sampling_params': {}
                }
                
                response = await worker._process_request(test_request)
                
                # 验证响应中的 request_id 与请求中的一致
                assert response['request_id'] == request_id, \
                    f"Response request_id {response['request_id']} does not match request {request_id}"
    
    @pytest.mark.asyncio
    async def test_worker_id_in_response(self):
        """测试响应中包含正确的 worker_id"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        # 测试不同的 worker_id
        test_worker_ids = [0, 1, 2, 5, 10]
        
        for worker_id in test_worker_ids:
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.set_device'):
                
                worker = GPUWorker(worker_id=worker_id, gpu_id=0, args=args)
                
                test_request = {
                    'request_id': 'req_test',
                    'adapter_dir': None,
                    'prompt_ids': [1, 2, 3],
                    'sampling_params': {}
                }
                
                response = await worker._process_request(test_request)
                
                # 验证响应中的 worker_id 正确
                assert response['worker_id'] == worker_id, \
                    f"Response worker_id {response['worker_id']} does not match expected {worker_id}"
    
    @pytest.mark.asyncio
    async def test_metadata_in_response(self):
        """测试响应中包含元数据"""
        args = argparse.Namespace(model_dir="/fake/path", max_total_token_num=1000)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            test_request = {
                'request_id': 'req_meta',
                'adapter_dir': None,
                'prompt_ids': [1, 2, 3, 4, 5],
                'sampling_params': {}
            }
            
            response = await worker._process_request(test_request)
            
            # 验证 metadata 存在且包含有用信息
            assert 'metadata' in response
            assert isinstance(response['metadata'], dict)
            # Phase 1 简化实现应该至少有一些基本信息
            assert len(response['metadata']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
