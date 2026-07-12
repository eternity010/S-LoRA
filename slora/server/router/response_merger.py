"""
Response Merger 模块

该模块实现了数据并行模式下的响应合并器，负责：
1. 接收来自多个 GPU Worker 的响应
2. 将响应转换为 Detokenization 进程期望的格式
3. 转发响应到 Detokenization 进程

Requirements:
    - 4.1: Worker 通过 ZMQ PUSH socket 发送响应消息
    - 4.2: Response Merger 通过 ZMQ PULL socket 接收来自所有 Worker 的响应
    - 4.3: 根据 request_id 匹配原始请求
    - 4.4: 将响应转发到 Detokenization 进程
    - 4.5: 确保响应消息包含 request_id、worker_id、output_ids 和 success 状态
"""

import os
import zmq
import zmq.asyncio
import asyncio
import json
from typing import Dict, Any

from slora.server.io_struct import BatchTokenIdOut, ReqDetokenizationState


class ResponseMerger:
    """
    响应合并器
    
    负责收集来自多个 GPU Worker 的响应，并将其转发到 Detokenization 进程。
    Response Merger 作为 Worker 和 Detokenization 之间的桥梁，确保响应正确传递。
    
    Attributes:
        worker_response_port: 接收 Worker 响应的端口
        detoken_port: Detokenization 进程的端口
        context: ZMQ 上下文
        worker_receiver: 接收 Worker 响应的 PULL socket
        detoken_sender: 发送到 Detokenization 的 PUSH socket
    """
    
    def __init__(self, worker_response_port: int, detoken_port: int):
        """
        初始化 Response Merger
        
        Args:
            worker_response_port: 接收 Worker 响应的端口
                所有 Worker 通过 PUSH socket 连接到这个端口发送响应
            detoken_port: Detokenization 进程的端口
                Response Merger 通过 PUSH socket 连接到这个端口转发响应
        
        Requirements:
            - 4.1: Worker 通过 ZMQ PUSH socket 发送响应消息
            - 4.2: Response Merger 通过 ZMQ PULL socket 接收响应
        
        Note:
            ZMQ 的 PUSH/PULL 模式是多对一的通信模式：
            - 多个 Worker 可以 PUSH 到同一个 PULL socket
            - Response Merger 的 PULL socket 会自动负载均衡接收消息
        """
        self.worker_response_port = worker_response_port
        self.detoken_port = detoken_port
        
        # ZMQ 通信（将在 _setup_zmq 中初始化）
        self.context = None
        self.worker_receiver = None
        self.detoken_sender = None
        
        print(f"[ResponseMerger] Initialized with worker_response_port={worker_response_port}, "
              f"detoken_port={detoken_port}")
    
    def _setup_zmq(self) -> None:
        """
        设置 ZMQ 通信
        
        创建两个 socket：
        1. PULL socket：接收来自所有 Worker 的响应（bind 模式）
        2. PUSH socket：发送到 Detokenization 进程（connect 模式）
        
        配置超时参数以防止通信阻塞。
        
        Requirements:
            - 4.1: Worker 通过 ZMQ PUSH socket 发送响应消息
            - 4.2: Response Merger 通过 ZMQ PULL socket 接收响应
            - 7.4: 设置 socket 超时防止通信阻塞
        
        Note:
            - PULL socket 使用 bind 模式，Worker 使用 connect 模式
            - PUSH socket 使用 connect 模式，Detokenization 使用 bind 模式
            - 这种模式确保了消息的可靠传递
            
            超时配置：
            - RCVTIMEO: 30000ms (30秒) - 接收超时
            - SNDTIMEO: 30000ms (30秒) - 发送超时
            - LINGER: 0 - 关闭时立即丢弃未发送消息
        """
        # 创建 ZMQ 异步上下文
        self.context = zmq.asyncio.Context()
        
        # 接收 Worker 响应
        # 使用 PULL socket 接收来自多个 Worker 的 PUSH 消息
        self.worker_receiver = self.context.socket(zmq.PULL)
        
        # 设置接收超时（30秒）
        self.worker_receiver.setsockopt(zmq.RCVTIMEO, 30000)
        # 设置 LINGER 为 0，关闭时立即丢弃未发送消息
        self.worker_receiver.setsockopt(zmq.LINGER, 0)
        # 设置 RCVHWM 为 0 表示无限制，防止消息丢失
        self.worker_receiver.setsockopt(zmq.RCVHWM, 0)
        
        self.worker_receiver.bind(f"tcp://127.0.0.1:{self.worker_response_port}")
        
        # 发送到 Detokenization
        # 使用 PUSH socket 连接到 Detokenization 的 PULL socket
        self.detoken_sender = self.context.socket(zmq.PUSH)
        
        # 设置发送超时（30秒）
        self.detoken_sender.setsockopt(zmq.SNDTIMEO, 30000)
        # 设置 LINGER 为 0，关闭时立即丢弃未发送消息
        self.detoken_sender.setsockopt(zmq.LINGER, 0)
        # 设置 SNDHWM 为 0 表示无限制，防止消息丢失
        self.detoken_sender.setsockopt(zmq.SNDHWM, 0)
        
        self.detoken_sender.connect(f"tcp://127.0.0.1:{self.detoken_port}")
        
        print(f"[ResponseMerger] ZMQ communication setup complete (timeout=30s)")
        print(f"[ResponseMerger] Listening for worker responses on port {self.worker_response_port}")
        print(f"[ResponseMerger] Forwarding to detokenization on port {self.detoken_port}")
    
    def _convert_to_detoken_format(self, response: Dict[str, Any]) -> BatchTokenIdOut:
        """
        将 Worker 响应转换为 Detokenization 期望的格式
        
        Worker 响应格式：
        {
            'request_id': str,
            'worker_id': int,
            'output_ids': List[int],
            'metadata': {
                'finish_reason': str,
                'prompt_tokens': int,
                'completion_tokens': int,
                'gen_metadata': dict  # 包含 id, logprob 等
            },
            'success': bool,
            'error': Optional[str],
            'finished': bool
        }
        
        Detokenization 期望格式：
        BatchTokenIdOut 对象，包含：
        - reqs_infs: List[Tuple[req_id, new_token_id, gen_metadata, finished_state, abort_state]]
        
        注意：gen_metadata 应该是 Worker 返回的 metadata['gen_metadata']，
        而不是整个 metadata 对象。这样 detokenization 进程才能正确处理。
        
        Args:
            response: Worker 响应字典
        
        Returns:
            BatchTokenIdOut: Detokenization 期望的批次输出格式
        
        Requirements:
            - 4.3: 根据 request_id 匹配原始请求
            - 4.5: 确保响应消息包含必要字段
        """
        batch_out = BatchTokenIdOut()
        
        request_id = response.get('request_id')
        output_ids = response.get('output_ids')
        metadata = response.get('metadata', {})
        success = response.get('success', False)
        finished = response.get('finished', False)
        
        # 如果缺少关键字段，跳过（可能是错误响应或心跳消息）
        if request_id is None or output_ids is None:
            return batch_out
        
        # 提取真正的 gen_metadata（包含 id, logprob 等）
        # 这是 detokenization 进程期望的格式
        gen_metadata = metadata.get('gen_metadata', {})
        
        # 对于流式输出，我们只取最后一个 token
        if output_ids and len(output_ids) > 0:
            new_token_id = output_ids[-1]
            # finished_state: 请求失败或已完成
            finished_state = not success or finished
            abort_state = False  # 数据并行模式暂不支持 abort
            
            batch_out.reqs_infs.append((
                request_id,
                new_token_id,
                gen_metadata,  # 使用正确的 gen_metadata
                finished_state,
                abort_state
            ))
        
        return batch_out
    
    async def _forward_to_detokenization(self, response: Dict[str, Any]) -> None:
        """
        转发响应到 Detokenization 进程
        """
        # 转换为 Detokenization 格式
        detoken_msg = self._convert_to_detoken_format(response)
        
        # 只在 DEBUG 模式下打印完成日志
        finished = response.get('finished', False)
        if finished and os.environ.get('DEBUG', '0') == '1':
            request_id = response['request_id']
            worker_id = response['worker_id']
            print(f"[ResponseMerger] FINISHED: req={request_id[:8]}..., worker={worker_id}")
        
        # 发送到 Detokenization
        await self.detoken_sender.send_pyobj(detoken_msg)

    @staticmethod
    def _record_reset_cache_response(response: Dict[str, Any]) -> None:
        """Persist a Worker cache-reset acknowledgement for the router manager."""
        request_id = response.get('request_id')
        if not request_id:
            return
        ack_file = f"/tmp/slora_reset_cache_{request_id}.jsonl"
        with open(ack_file, 'a') as f:
            f.write(json.dumps(response) + "\n")
    
    async def run(self) -> None:
        """
        主循环
        
        持续接收来自 Worker 的响应，并转发到 Detokenization 进程。
        
        Requirements:
            - 4.2: 通过 ZMQ PULL socket 接收来自所有 Worker 的响应
            - 4.4: 将响应转发到 Detokenization 进程
        
        Note:
            这是一个无限循环，会一直运行直到进程被终止。
            在生产环境中，应该添加优雅关闭机制。
        """
        print(f"[ResponseMerger] Starting main loop...")
        
        while True:
            try:
                # 接收 Worker 响应（JSON 格式）
                response = await self.worker_receiver.recv_json()

                if response.get('type') == 'reset_cache_response':
                    self._record_reset_cache_response(response)
                    continue
                
                # 转发到 Detokenization
                await self._forward_to_detokenization(response)
                
            except zmq.error.Again:
                # ZMQ 超时是正常的，不需要打印错误
                # 继续处理下一个响应
                continue
            except Exception as e:
                print(f"[ResponseMerger] Error processing response: {str(e)}")
                import traceback
                traceback.print_exc()
                # 继续处理下一个响应，不中断主循环


def run_response_merger_process(worker_response_port: int, detoken_port: int):
    """
    Response Merger 进程入口函数
    
    这个函数在独立的进程中运行，创建并启动 Response Merger。
    
    Args:
        worker_response_port: 接收 Worker 响应的端口
        detoken_port: Detokenization 进程的端口
    
    Note:
        这个函数会被 multiprocessing.Process 调用
    """
    try:
        # 创建 Response Merger
        merger = ResponseMerger(worker_response_port, detoken_port)
        
        # 设置 ZMQ 通信
        merger._setup_zmq()
        
        # 运行主循环
        asyncio.run(merger.run())
        
    except Exception as e:
        print(f"[ResponseMerger] Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
