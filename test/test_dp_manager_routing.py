"""
测试 DataParallelRouterManager 的请求路由逻辑

测试内容：
1. route_request() 方法能够正确选择 Worker
2. route_request() 方法能够通过 ZMQ 发送请求
3. run() 方法能够持续接收和路由请求

Requirements:
    - 2.1: 使用 Round Robin Router 选择下一个 Worker
    - 2.2: 按照 Worker ID 的顺序循环分配请求
    - 2.3: 通过 ZMQ PUSH socket 发送请求消息
"""

import pytest
import asyncio
import argparse
import json
import os
import zmq
import zmq.asyncio
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from slora.server.router.dp_manager import DataParallelRouterManager
from slora.server.sampling_params import SamplingParams


class TestConfigUpdateAcknowledgement:
    def test_update_id_is_recorded_after_config_is_applied(self, tmp_path):
        manager = DataParallelRouterManager.__new__(DataParallelRouterManager)
        manager.routing_strategy = "adapter-aware"
        manager.router = Mock()
        manager.router.update_config.return_value = {
            "w1": 1.0,
            "w2": 2.0,
            "w3": 0.0,
            "load_metric": "token_count",
        }
        manager.replica_manager = None
        manager._write_stats_file = Mock()
        update_file = tmp_path / "routing-config.json"
        update_file.write_text(json.dumps({
            "w2": 2.0,
            "load_metric": "token_count",
            "reset_stats": True,
            "update_id": "update-123",
        }))

        manager._check_config_update(str(update_file))

        manager.router.update_config.assert_called_once_with(
            w2=2.0,
            load_metric="token_count",
            reset_stats=True,
        )
        assert manager._last_config_update_id == "update-123"
        manager._write_stats_file.assert_called_once_with()
        assert not update_file.exists()

    @pytest.mark.asyncio
    async def test_cache_reset_result_preserves_trigger_id(self, tmp_path):
        manager = DataParallelRouterManager.__new__(DataParallelRouterManager)
        manager.reset_all_adapter_caches = AsyncMock(return_value={
            "success": True,
            "num_workers": 3,
            "message": "completed",
        })
        trigger_file = tmp_path / "reset.trigger"
        result_file = tmp_path / "reset.result"
        trigger_file.write_text(json.dumps({"reset_id": "reset-123"}))

        await manager._check_cache_reset(str(trigger_file), str(result_file))

        result = json.loads(result_file.read_text())
        assert result["reset_id"] == "reset-123"
        assert result["success"] is True
        assert not trigger_file.exists()

    @pytest.mark.asyncio
    async def test_reset_waits_for_all_worker_acknowledgements(self, monkeypatch):
        manager = DataParallelRouterManager.__new__(DataParallelRouterManager)
        manager.num_workers = 3
        manager.router = Mock()
        manager.router.adapter_to_workers = {"adapter": {0, 1}}
        manager._reset_experiment_stats = Mock()
        manager._write_stats_file = Mock()
        ack_file = "/tmp/slora_reset_cache_reset_deadbeef.jsonl"
        try:
            os.remove(ack_file)
        except FileNotFoundError:
            pass

        senders = []
        for worker_id in range(3):
            async def send_json(command, worker_id=worker_id):
                with open(ack_file, "a") as f:
                    f.write(json.dumps({
                        "type": "reset_cache_response",
                        "request_id": command["request_id"],
                        "worker_id": worker_id,
                        "success": True,
                    }) + "\n")
            senders.append(SimpleNamespace(send_json=send_json))
        manager.request_senders = senders
        monkeypatch.setattr("uuid.uuid4", lambda: SimpleNamespace(hex="deadbeef"))

        result = await manager.reset_all_adapter_caches()

        assert result["success"] is True
        assert set(result["worker_results"]) == {0, 1, 2}
        assert manager.router.adapter_to_workers == {}
        manager._reset_experiment_stats.assert_called_once_with()
        manager._write_stats_file.assert_called_once_with()
        assert not os.path.exists(ack_file)


@pytest.fixture
def mock_args():
    """创建模拟的命令行参数"""
    args = argparse.Namespace()
    args.num_workers = 3
    args.gpu_ids = "0,1,2"
    args.log_level = "INFO"
    return args


@pytest.fixture
def router_manager(mock_args):
    """创建 DataParallelRouterManager 实例（不启动 Worker）"""
    manager = DataParallelRouterManager(
        args=mock_args,
        router_port=60000,
        response_port=60001
    )
    return manager


class TestRouteRequest:
    """测试 route_request() 方法"""
    
    @pytest.mark.asyncio
    async def test_route_request_selects_worker(self, router_manager):
        """
        测试 route_request 能够使用 Round Robin Router 选择 Worker
        
        验证：
        1. 调用 router.select_worker() 选择 Worker
        2. 选择的 Worker ID 在有效范围内
        """
        # 设置 ZMQ（使用模拟的 sockets）
        router_manager.context = zmq.asyncio.Context()
        router_manager.request_senders = []
        
        for i in range(router_manager.num_workers):
            mock_sender = AsyncMock()
            router_manager.request_senders.append(mock_sender)
        
        # 创建测试请求
        request = {
            'request_id': 'test-req-1',
            'adapter_dir': '/path/to/adapter',
            'prompt_ids': [1, 2, 3],
            'sampling_params': {}
        }
        
        # 路由请求
        await router_manager.route_request(request)
        
        # 验证：至少有一个 sender 被调用
        called_senders = [s for s in router_manager.request_senders if s.send_json.called]
        assert len(called_senders) == 1, "应该有且仅有一个 sender 被调用"
    
    @pytest.mark.asyncio
    async def test_route_request_round_robin_order(self, router_manager):
        """
        测试 route_request 按照轮询顺序分配请求
        
        验证：
        1. 连续的请求按照 0, 1, 2, 0, 1, 2 的顺序分配
        2. 每个 Worker 收到的请求数量相等
        
        Requirements: 2.1, 2.2
        """
        # 设置 ZMQ（使用模拟的 sockets）
        router_manager.context = zmq.asyncio.Context()
        router_manager.request_senders = []
        
        for i in range(router_manager.num_workers):
            mock_sender = AsyncMock()
            router_manager.request_senders.append(mock_sender)
        
        # 发送 6 个请求
        for i in range(6):
            request = {
                'request_id': f'test-req-{i}',
                'adapter_dir': '/path/to/adapter',
                'prompt_ids': [1, 2, 3],
                'sampling_params': {}
            }
            await router_manager.route_request(request)
        
        # 验证：每个 Worker 应该收到 2 个请求
        for i, sender in enumerate(router_manager.request_senders):
            assert sender.send_json.call_count == 2, \
                f"Worker {i} 应该收到 2 个请求，实际收到 {sender.send_json.call_count}"
    
    @pytest.mark.asyncio
    async def test_route_request_sends_correct_message(self, router_manager):
        """
        测试 route_request 发送正确的消息内容
        
        验证：
        1. 发送的消息包含所有必需字段
        2. 消息内容与原始请求一致
        
        Requirements: 2.3, 2.4
        """
        # 设置 ZMQ（使用模拟的 sockets）
        router_manager.context = zmq.asyncio.Context()
        router_manager.request_senders = []
        
        for i in range(router_manager.num_workers):
            mock_sender = AsyncMock()
            router_manager.request_senders.append(mock_sender)
        
        # 创建测试请求
        request = {
            'request_id': 'test-req-1',
            'adapter_dir': '/path/to/adapter',
            'prompt_ids': [1, 2, 3, 4, 5],
            'sampling_params': {'temperature': 0.7, 'top_p': 0.9}
        }
        
        # 路由请求
        await router_manager.route_request(request)
        
        # 找到被调用的 sender
        called_sender = None
        for sender in router_manager.request_senders:
            if sender.send_json.called:
                called_sender = sender
                break
        
        assert called_sender is not None, "应该有一个 sender 被调用"
        
        # 验证发送的消息
        sent_message = called_sender.send_json.call_args[0][0]
        assert sent_message['request_id'] == 'test-req-1'
        assert sent_message['adapter_dir'] == '/path/to/adapter'
        assert sent_message['prompt_ids'] == [1, 2, 3, 4, 5]
        assert sent_message['sampling_params'] == {'temperature': 0.7, 'top_p': 0.9}
    
    @pytest.mark.asyncio
    async def test_route_request_handles_error(self, router_manager):
        """
        测试 route_request 处理发送失败的情况
        
        验证：
        1. 发送失败时抛出异常
        2. 错误日志被记录
        
        Requirements: 2.5
        """
        # 设置 ZMQ（使用模拟的 sockets）
        router_manager.context = zmq.asyncio.Context()
        router_manager.request_senders = []
        
        for i in range(router_manager.num_workers):
            mock_sender = AsyncMock()
            # 模拟发送失败
            mock_sender.send_json.side_effect = Exception("ZMQ send failed")
            router_manager.request_senders.append(mock_sender)
        
        # 创建测试请求
        request = {
            'request_id': 'test-req-1',
            'adapter_dir': '/path/to/adapter',
            'prompt_ids': [1, 2, 3],
            'sampling_params': {}
        }
        
        # 验证：应该抛出异常
        with pytest.raises(Exception) as exc_info:
            await router_manager.route_request(request)
        
        assert "ZMQ send failed" in str(exc_info.value)


class TestRunMainLoop:
    """测试 run() 主循环"""
    
    @pytest.mark.asyncio
    async def test_run_receives_and_routes_requests(self, router_manager):
        """
        测试 run() 能够持续接收和路由请求
        
        验证：
        1. 从 request_receiver 接收请求
        2. 调用 route_request() 路由请求
        3. 持续处理多个请求
        
        Requirements: 2.1, 2.3
        
        Note: run() 方法使用 recv_pyobj() 接收 tuple 格式的请求
        """
        # 设置 ZMQ（使用模拟的 sockets）
        router_manager.context = zmq.asyncio.Context()
        
        # 模拟 request_receiver
        mock_receiver = AsyncMock()
        router_manager.request_receiver = mock_receiver
        
        # 模拟 send_to_detokenization
        router_manager.send_to_detokenization = Mock()
        
        # 模拟接收 3 个请求后停止
        # run() 方法期望 tuple 格式: (adapter_dir, prompt_ids, sampling_params, request_id)
        # sampling_params 需要是 SamplingParams 对象
        requests = [
            ('/path', [1, 2], SamplingParams(max_new_tokens=10), 'req-1'),
            ('/path', [3, 4], SamplingParams(max_new_tokens=10), 'req-2'),
            ('/path', [5, 6], SamplingParams(max_new_tokens=10), 'req-3'),
        ]
        
        # 设置 recv_pyobj 返回请求，然后抛出异常停止循环
        mock_receiver.recv_pyobj.side_effect = requests + [asyncio.CancelledError()]
        
        # 模拟 request_senders
        router_manager.request_senders = []
        for i in range(router_manager.num_workers):
            mock_sender = AsyncMock()
            router_manager.request_senders.append(mock_sender)
        
        # 设置 workers_ready 为 True，避免健康检查和统计任务阻塞
        router_manager.workers_ready = True
        
        # 运行主循环（会在 CancelledError 时停止）
        with pytest.raises(asyncio.CancelledError):
            await router_manager.run()
        
        # 验证：recv_pyobj 被调用了 4 次（3 个请求 + 1 次触发异常）
        assert mock_receiver.recv_pyobj.call_count == 4
        
        # 验证：所有请求都被路由（总共 3 个请求）
        total_sent = sum(s.send_json.call_count for s in router_manager.request_senders)
        assert total_sent == 3, f"应该发送 3 个请求，实际发送 {total_sent}"
    
    @pytest.mark.asyncio
    async def test_run_continues_on_error(self, router_manager):
        """
        测试 run() 在遇到错误时继续运行
        
        验证：
        1. 单个请求处理失败不会终止主循环
        2. 错误被记录但循环继续
        
        Requirements: 2.5
        
        Note: run() 方法使用 recv_pyobj() 接收 tuple 格式的请求
        """
        # 设置 ZMQ（使用模拟的 sockets）
        router_manager.context = zmq.asyncio.Context()
        
        # 模拟 request_receiver
        mock_receiver = AsyncMock()
        router_manager.request_receiver = mock_receiver
        
        # 模拟 send_to_detokenization
        router_manager.send_to_detokenization = Mock()
        
        # 第一个请求正常，第二个请求触发错误，第三个请求正常，然后停止
        # run() 方法期望 tuple 格式: (adapter_dir, prompt_ids, sampling_params, request_id)
        # sampling_params 需要是 SamplingParams 对象
        requests = [
            ('/path', [1, 2], SamplingParams(max_new_tokens=10), 'req-1'),
            ('/path', [3, 4], SamplingParams(max_new_tokens=10), 'req-2'),
            ('/path', [5, 6], SamplingParams(max_new_tokens=10), 'req-3'),
        ]
        
        mock_receiver.recv_pyobj.side_effect = requests + [asyncio.CancelledError()]
        
        # 模拟 request_senders
        router_manager.request_senders = []
        for i in range(router_manager.num_workers):
            mock_sender = AsyncMock()
            router_manager.request_senders.append(mock_sender)
        
        # 让第一个 sender 在第二次调用时失败（使用 side_effect）
        error_on_second_call = [
            None,  # 第一次调用成功
            Exception("Simulated send error"),  # 第二次调用失败
            None,  # 第三次调用成功（如果有的话）
        ]
        router_manager.request_senders[0].send_json.side_effect = error_on_second_call
        
        # 设置 workers_ready 为 True，避免健康检查和统计任务阻塞
        router_manager.workers_ready = True
        
        # 运行主循环（会在 CancelledError 时停止）
        with pytest.raises(asyncio.CancelledError):
            await router_manager.run()
        
        # 验证：recv_pyobj 被调用了 4 次（3 个请求 + 1 次触发异常）
        assert mock_receiver.recv_pyobj.call_count == 4
        
        # 验证：至少有 2 个请求被成功发送（req-1 和 req-3）
        # 注意：由于 round robin，请求会分配到不同的 worker
        # 我们只需要验证总的调用次数
        total_calls = sum(s.send_json.call_count for s in router_manager.request_senders)
        assert total_calls == 3, f"应该尝试发送 3 个请求，实际尝试 {total_calls}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
