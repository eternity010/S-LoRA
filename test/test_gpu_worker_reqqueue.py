"""
GPU Worker ReqQueue Integration Tests

测试 GPU Worker 与 ReqQueue 的集成，验证批处理功能。
"""

import pytest
import argparse
import asyncio
from unittest.mock import Mock, patch, MagicMock

from slora.server.router.gpu_worker import GPUWorker
from slora.server.io_struct import Req, Batch
from slora.server.sampling_params import SamplingParams


@pytest.fixture
def mock_args():
    """创建模拟的命令行参数"""
    args = argparse.Namespace()
    args.model_dir = "/fake/model"
    args.max_total_token_num = 1000
    args.batch_max_tokens = 500
    args.running_max_req_size = 10
    args.mem_adapter_size = 0
    args.load_way = 'HF'
    args.mode = []
    args.dummy = True
    return args


@pytest.fixture
def worker(mock_args):
    """创建 GPU Worker 实例（跳过 GPU 设置）"""
    with patch('slora.server.router.gpu_worker.torch.cuda.is_available', return_value=True), \
         patch('slora.server.router.gpu_worker.torch.cuda.set_device'):
        worker = GPUWorker(worker_id=0, gpu_id=0, args=mock_args)
        return worker


class TestSetupRequestQueue:
    """测试 _setup_request_queue() 方法"""
    
    def test_setup_request_queue_creates_instance(self, worker):
        """测试创建 ReqQueue 实例"""
        worker._setup_request_queue()
        
        assert worker.req_queue is not None
        assert worker.req_queue.max_total_tokens == 1000
        assert worker.req_queue.batch_max_tokens == 500
        assert worker.req_queue.running_max_req_size == 10
    
    def test_setup_request_queue_initializes_empty_waiting_list(self, worker):
        """测试初始化空的等待列表"""
        worker._setup_request_queue()
        
        assert len(worker.req_queue.waiting_req_list) == 0


class TestConvertToReqObject:
    """测试 _convert_to_req_object() 方法"""
    
    def test_convert_basic_request(self, worker):
        """测试转换基本请求"""
        request = {
            'request_id': 'req-001',
            'adapter_dir': 'adapter-1',
            'prompt_ids': [1, 2, 3, 4, 5],
            'sampling_params': {
                'do_sample': True,  # 需要设置为 True 才能保留 temperature 和 top_p
                'max_new_tokens': 100,
                'temperature': 0.8,
                'top_p': 0.9
            }
        }
        
        req = worker._convert_to_req_object(request)
        
        assert isinstance(req, Req)
        assert req.request_id == 'req-001'
        assert req.adapter_dir == 'adapter-1'
        assert req.prompt_ids == [1, 2, 3, 4, 5]
        assert req.input_len == 5
        assert req.max_output_len == 100
        assert req.sample_params.temperature == 0.8
        assert req.sample_params.top_p == 0.9
    
    def test_convert_request_with_default_adapter(self, worker):
        """测试转换没有 adapter_dir 的请求（使用默认值）"""
        request = {
            'request_id': 'req-002',
            'prompt_ids': [1, 2, 3],
            'sampling_params': {}
        }
        
        req = worker._convert_to_req_object(request)
        
        assert req.adapter_dir == 'base'
    
    def test_convert_request_with_default_sampling_params(self, worker):
        """测试转换没有 sampling_params 的请求（使用默认值）"""
        request = {
            'request_id': 'req-003',
            'prompt_ids': [1, 2, 3]
        }
        
        req = worker._convert_to_req_object(request)
        
        assert req.max_output_len == 128  # 默认值
        assert req.sample_params.temperature == 1.0
        assert req.sample_params.top_p == 1.0
    
    def test_convert_request_with_stop_sequences(self, worker):
        """测试转换包含 stop_sequences 的请求"""
        request = {
            'request_id': 'req-004',
            'prompt_ids': [1, 2, 3],
            'sampling_params': {
                'stop_sequences': [[13], [13, 13]]
            }
        }
        
        req = worker._convert_to_req_object(request)
        
        assert req.sample_params.stop_sequences == [[13], [13, 13]]


class TestProcessRequests:
    """测试 _process_requests() 方法（批处理）"""
    
    @pytest.mark.asyncio
    async def test_process_empty_queue(self, worker):
        """测试处理空队列"""
        worker._setup_request_queue()
        
        responses = await worker._process_requests()
        
        assert responses == []
        assert worker.current_batch is None
    
    @pytest.mark.asyncio
    async def test_process_single_request(self, worker):
        """测试处理单个请求"""
        worker._setup_request_queue()
        
        # 添加一个请求到队列
        req = Req(
            adapter_dir='base',
            request_id='req-001',
            prompt_ids=[1, 2, 3],
            sample_params=SamplingParams(max_new_tokens=10)
        )
        worker.req_queue.append(req)
        
        # 处理请求
        responses = await worker._process_requests()
        
        assert len(responses) == 1
        assert responses[0]['request_id'] == 'req-001'
        assert responses[0]['worker_id'] == 0
        assert responses[0]['success'] is True
        assert len(responses[0]['output_ids']) > 3  # 应该有输出
    
    @pytest.mark.asyncio
    async def test_process_multiple_requests_batch(self, worker):
        """测试批处理多个请求"""
        worker._setup_request_queue()
        
        # 添加多个请求到队列
        for i in range(3):
            req = Req(
                adapter_dir='base',
                request_id=f'req-{i:03d}',
                prompt_ids=[1, 2, 3],
                sample_params=SamplingParams(max_new_tokens=10)
            )
            worker.req_queue.append(req)
        
        # 处理请求
        responses = await worker._process_requests()
        
        assert len(responses) == 3
        request_ids = [r['request_id'] for r in responses]
        assert 'req-000' in request_ids
        assert 'req-001' in request_ids
        assert 'req-002' in request_ids
    
    @pytest.mark.asyncio
    async def test_process_requests_updates_batch_state(self, worker):
        """测试处理请求后更新批次状态"""
        worker._setup_request_queue()
        
        # 添加请求
        req = Req(
            adapter_dir='base',
            request_id='req-001',
            prompt_ids=[1, 2, 3],
            sample_params=SamplingParams(max_new_tokens=10)
        )
        worker.req_queue.append(req)
        
        # 第一次处理
        responses1 = await worker._process_requests()
        assert len(responses1) == 1
        
        # 批次应该被清空（因为请求已完成）
        assert worker.current_batch is None
    
    @pytest.mark.asyncio
    async def test_process_requests_with_lora_ranks(self, worker):
        """测试处理带有 LoRA ranks 的请求"""
        worker._setup_request_queue()
        worker.lora_ranks = {'adapter-1': 8, 'adapter-2': 16}
        
        # 添加使用不同 adapter 的请求
        req1 = Req(
            adapter_dir='adapter-1',
            request_id='req-001',
            prompt_ids=[1, 2, 3],
            sample_params=SamplingParams(max_new_tokens=10)
        )
        req2 = Req(
            adapter_dir='adapter-2',
            request_id='req-002',
            prompt_ids=[4, 5, 6],
            sample_params=SamplingParams(max_new_tokens=10)
        )
        worker.req_queue.append(req1)
        worker.req_queue.append(req2)
        
        # 处理请求
        responses = await worker._process_requests()
        
        # 应该能处理多个 adapter 的请求
        assert len(responses) >= 1


class TestRunWithReqQueue:
    """测试 run() 方法与 ReqQueue 的集成"""
    
    @pytest.mark.asyncio
    async def test_run_receives_and_processes_requests(self, worker):
        """测试主循环接收并处理请求"""
        worker._setup_request_queue()
        
        # 模拟 ZMQ 接收
        requests = [
            {
                'request_id': 'req-001',
                'prompt_ids': [1, 2, 3],
                'sampling_params': {'max_new_tokens': 10}
            },
            {
                'request_id': 'req-002',
                'prompt_ids': [4, 5, 6],
                'sampling_params': {'max_new_tokens': 10}
            }
        ]
        
        request_iter = iter(requests)
        responses_sent = []
        
        async def mock_receive():
            try:
                return next(request_iter)
            except StopIteration:
                # 模拟超时
                await asyncio.sleep(1)
                raise asyncio.TimeoutError()
        
        async def mock_send(response):
            responses_sent.append(response)
        
        worker._receive_request = mock_receive
        worker._send_response = mock_send
        
        # 运行几次循环
        async def run_limited():
            for _ in range(5):  # 运行 5 次循环
                try:
                    # 尝试接收新请求
                    try:
                        request = await asyncio.wait_for(
                            worker._receive_request(), 
                            timeout=0.01
                        )
                        req_obj = worker._convert_to_req_object(request)
                        worker.req_queue.append(req_obj)
                    except asyncio.TimeoutError:
                        pass
                    
                    # 处理请求批次
                    responses = await worker._process_requests()
                    
                    # 发送响应
                    for response in responses:
                        await worker._send_response(response)
                except Exception:
                    pass
        
        await run_limited()
        
        # 验证响应
        assert len(responses_sent) >= 1
        request_ids = [r['request_id'] for r in responses_sent]
        assert 'req-001' in request_ids or 'req-002' in request_ids


class TestReqQueueIntegration:
    """测试 ReqQueue 集成的完整性"""
    
    def test_reqqueue_append_and_generate_batch(self, worker):
        """测试 ReqQueue 的 append 和 generate_new_batch 功能"""
        worker._setup_request_queue()
        
        # 添加请求
        req1 = Req(
            adapter_dir='base',
            request_id='req-001',
            prompt_ids=[1, 2, 3],
            sample_params=SamplingParams(max_new_tokens=10)
        )
        req2 = Req(
            adapter_dir='base',
            request_id='req-002',
            prompt_ids=[4, 5, 6],
            sample_params=SamplingParams(max_new_tokens=10)
        )
        
        worker.req_queue.append(req1)
        worker.req_queue.append(req2)
        
        # 生成批次（需要提供 lora_ranks）
        batch = worker.req_queue.generate_new_batch(None, worker.lora_ranks, 0)
        
        assert batch is not None
        assert len(batch.reqs) == 2
        assert batch.reqs[0].request_id == 'req-001'
        assert batch.reqs[1].request_id == 'req-002'
    
    def test_reqqueue_respects_batch_max_tokens(self, worker):
        """测试 ReqQueue 遵守 batch_max_tokens 限制"""
        worker._setup_request_queue()
        
        # 创建一个超过 batch_max_tokens 的请求
        large_req = Req(
            adapter_dir='base',
            request_id='req-large',
            prompt_ids=[1] * 600,  # 600 tokens，超过 batch_max_tokens (500)
            sample_params=SamplingParams(max_new_tokens=10)
        )
        
        worker.req_queue.append(large_req)
        
        # 尝试生成批次（需要提供 lora_ranks）
        batch = worker.req_queue.generate_new_batch(None, worker.lora_ranks, 0)
        
        # 应该无法生成批次（因为超过限制）
        assert batch is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
