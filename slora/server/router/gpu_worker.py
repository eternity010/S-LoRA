"""
GPU Worker 进程模块

该模块实现了数据并行模式下的 GPU Worker，每个 Worker 在独立的进程中运行，
在指定的 GPU 上加载完整的基座模型并处理推理请求。
"""

import os
import torch
import argparse
import zmq
import zmq.asyncio
import asyncio
from typing import Optional, List

from slora.models.llama.model import LlamaTpPartModel
from slora.models.llama2.model import Llama2TpPartModel
from slora.utils.model_utils import get_model_config
from slora.server.router.req_queue import ReqQueue
from slora.server.io_struct import Req, Batch
from slora.server.sampling_params import SamplingParams


class GPUWorker:
    """
    GPU Worker 进程，在指定 GPU 上处理推理请求
    
    每个 Worker 运行在独立的进程中，拥有完整的基座模型副本。
    Worker 通过 ZMQ 接收请求并返回响应。
    
    Attributes:
        worker_id: Worker 的唯一标识符
        gpu_id: 分配的 GPU ID
        args: 命令行参数
        model: 加载的基座模型（待实现）
        adapter_cache: Adapter 缓存（待实现）
    """
    
    def __init__(self, worker_id: int, gpu_id: int, args: argparse.Namespace):
        """
        初始化 GPU Worker
        
        Args:
            worker_id: Worker 的唯一标识符，用于日志和调试
            gpu_id: 分配的 GPU ID，Worker 将在此 GPU 上运行
            args: 命令行参数，包含模型路径、配置等信息
        
        Requirements:
            - 1.2: 为每个 Worker 分配唯一的 worker_id 和 gpu_id
            - 1.3: 设置 CUDA_VISIBLE_DEVICES 环境变量
        """
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.args = args
        
        # 模型相关（待后续任务实现）
        self.model = None
        self.adapter_cache = {}
        
        # 请求队列管理（复用 ReqQueue）
        self.req_queue = None  # 将在 _setup_request_queue() 中初始化
        self.current_batch = None
        self.lora_ranks = {'base': 0}  # LoRA rank 信息，用于显存管理（base 模型 rank 为 0）
        self.actual_adapter_size = 0  # 实际 adapter 占用的显存大小
        
        # ZMQ 通信相关（待后续任务实现）
        self.context = None
        self.request_receiver = None
        self.response_sender = None
        
        # 设置 GPU 环境
        self._setup_gpu()
    
    def _setup_gpu(self) -> None:
        """
        设置 GPU 环境
        
        通过设置 CUDA_VISIBLE_DEVICES 环境变量，确保 Worker 只能看到分配给它的 GPU。
        然后将 PyTorch 的默认设备设置为该 GPU。
        
        Requirements:
            - 1.3: 设置 CUDA_VISIBLE_DEVICES 环境变量为对应的 gpu_id
        
        Note:
            设置 CUDA_VISIBLE_DEVICES 后，从 Worker 的视角看，只有一个 GPU (索引为 0)。
            这样可以确保不同 Worker 之间的 GPU 隔离。
        """
        # 设置环境变量，限制可见的 GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        # 设置 PyTorch 默认设备为 GPU 0（因为 CUDA_VISIBLE_DEVICES 已经限制了可见 GPU）
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"[Worker {self.worker_id}] GPU environment setup complete: "
                  f"CUDA_VISIBLE_DEVICES={self.gpu_id}, torch device=cuda:0")
        else:
            raise RuntimeError(f"[Worker {self.worker_id}] CUDA is not available on GPU {self.gpu_id}")
    
    def _setup_zmq(self, request_port: int, response_port: int) -> None:
        """
        设置 ZMQ 通信
        
        创建两个 ZMQ socket：
        1. PULL socket: 从 Router Manager 接收请求
        2. PUSH socket: 向 Response Merger 发送响应
        
        Args:
            request_port: 接收请求的端口号（Router Manager 的 PUSH socket 绑定的端口）
            response_port: 发送响应的端口号（Response Merger 的 PULL socket 绑定的端口）
        
        Requirements:
            - 2.3: 通过 ZMQ PUSH socket 发送请求消息
            - 4.1: 通过 ZMQ PUSH socket 发送响应消息
        
        Note:
            使用 zmq.asyncio.Context 以支持异步操作。
            Worker 使用 PULL socket 接收（多对一），使用 PUSH socket 发送（一对一）。
        """
        # 创建异步 ZMQ context
        self.context = zmq.asyncio.Context()
        
        # 创建 PULL socket 接收请求
        # Router Manager 使用 PUSH 发送，Worker 使用 PULL 接收
        self.request_receiver = self.context.socket(zmq.PULL)
        self.request_receiver.connect(f"tcp://127.0.0.1:{request_port}")
        
        # 创建 PUSH socket 发送响应
        # Worker 使用 PUSH 发送，Response Merger 使用 PULL 接收
        self.response_sender = self.context.socket(zmq.PUSH)
        self.response_sender.connect(f"tcp://127.0.0.1:{response_port}")
        
        print(f"[Worker {self.worker_id}] ZMQ communication setup complete: "
              f"request_port={request_port}, response_port={response_port}")
    
    def _load_model(self) -> None:
        """
        加载基座模型
        
        复用 slora/common/basemodel/ 中的模型加载逻辑。
        根据模型配置自动选择合适的模型类（Llama 或 Llama2）。
        
        Requirements:
            - 1.4: 在指定 GPU 上加载模型
        
        Raises:
            Exception: 如果模型类型不支持或加载失败
        
        Note:
            - 数据并行模式下，每个 Worker 加载完整的基座模型
            - tp_rank=0, world_size=1 表示单 GPU 模式（无张量并行）
            - 模型会自动加载到当前 GPU（由 _setup_gpu 设置）
        """
        try:
            # 获取模型配置
            model_cfg = get_model_config(
                self.args.model_dir,
                dummy=getattr(self.args, 'dummy', False)
            )
            
            model_type = model_cfg.get("model_type", "llama")
            print(f"[Worker {self.worker_id}] Loading model type: {model_type}")
            
            # 根据模型类型选择对应的模型类
            if model_type == "llama":
                # 检查是否是 Llama2（通过 num_key_value_heads 判断）
                if "num_key_value_heads" in model_cfg:
                    print(f"[Worker {self.worker_id}] Detected Llama2 model (GQA)")
                    self.model = Llama2TpPartModel(
                        tp_rank=0,  # 数据并行模式，每个 Worker 独立，无张量并行
                        world_size=1,  # 单 GPU 模式
                        weight_dir=self.args.model_dir,
                        max_total_token_num=self.args.max_total_token_num,
                        mem_adapter_size=getattr(self.args, 'mem_adapter_size', 0),
                        load_way=getattr(self.args, 'load_way', 'HF'),
                        mode=getattr(self.args, 'mode', []),
                        dummy=getattr(self.args, 'dummy', False)
                    )
                else:
                    print(f"[Worker {self.worker_id}] Detected Llama model")
                    self.model = LlamaTpPartModel(
                        tp_rank=0,
                        world_size=1,
                        weight_dir=self.args.model_dir,
                        max_total_token_num=self.args.max_total_token_num,
                        mem_adapter_size=getattr(self.args, 'mem_adapter_size', 0),
                        load_way=getattr(self.args, 'load_way', 'HF'),
                        mode=getattr(self.args, 'mode', []),
                        dummy=getattr(self.args, 'dummy', False)
                    )
            else:
                raise Exception(f"[Worker {self.worker_id}] Unsupported model type: {model_type}")
            
            print(f"[Worker {self.worker_id}] Model loaded successfully on GPU {self.gpu_id}")
            
        except Exception as e:
            print(f"[Worker {self.worker_id}] Failed to load model: {str(e)}")
            raise
    
    def _setup_request_queue(self) -> None:
        """
        初始化请求队列（复用 ReqQueue）
        
        ReqQueue 提供：
        - 批处理逻辑（generate_new_batch）
        - 显存管理（_can_add_new_req）
        - Adapter 调度（adapter_size 计算）
        
        Requirements:
            - 3.1: 持续监听 ZMQ PULL socket 接收请求
            - 3.2: 解析请求消息并提取必要的参数
            - 3.4: 使用现有的模型推理逻辑处理请求
        
        Note:
            复用张量并行模式中的 ReqQueue 进行请求管理。
            每个 Worker 本质上是一个"单 GPU 的张量并行系统"。
        """
        self.req_queue = ReqQueue(
            max_total_tokens=self.args.max_total_token_num,
            batch_max_tokens=self.args.batch_max_tokens,
            running_max_req_size=self.args.running_max_req_size
        )
        print(f"[Worker {self.worker_id}] Request queue initialized: "
              f"max_total_tokens={self.args.max_total_token_num}, "
              f"batch_max_tokens={self.args.batch_max_tokens}, "
              f"running_max_req_size={self.args.running_max_req_size}")
    
    def _convert_to_req_object(self, request: dict) -> Req:
        """
        将 ZMQ 消息转换为 Req 对象
        
        Args:
            request: ZMQ 请求消息，包含：
                - request_id: 请求唯一标识符
                - adapter_dir: Adapter 路径（可选）
                - prompt_ids: 输入 token IDs
                - sampling_params: 采样参数
        
        Returns:
            Req: 请求对象，可以被 ReqQueue 管理
        
        Requirements:
            - 3.2: 解析请求消息并提取必要的参数
        
        Note:
            将 ZMQ 消息格式转换为 ReqQueue 所需的 Req 对象格式。
        """
        # 提取采样参数
        sampling_params_dict = request.get('sampling_params', {})
        sample_params = SamplingParams(
            do_sample=sampling_params_dict.get('do_sample', False),
            presence_penalty=sampling_params_dict.get('presence_penalty', 0.0),
            frequency_penalty=sampling_params_dict.get('frequency_penalty', 0.0),
            temperature=sampling_params_dict.get('temperature', 1.0),
            top_p=sampling_params_dict.get('top_p', 1.0),
            top_k=sampling_params_dict.get('top_k', -1),
            ignore_eos=sampling_params_dict.get('ignore_eos', False),
            max_new_tokens=sampling_params_dict.get('max_new_tokens', 128),
            stop_sequences=sampling_params_dict.get('stop_sequences', None)
        )
        
        # 创建 Req 对象
        req = Req(
            adapter_dir=request.get('adapter_dir', 'base'),
            request_id=request['request_id'],
            prompt_ids=request['prompt_ids'],
            sample_params=sample_params
        )
        
        return req
    
    async def _receive_request(self) -> dict:
        """
        接收请求
        
        从 ZMQ PULL socket 接收来自 Router Manager 的请求消息。
        
        Returns:
            dict: 请求消息，包含 request_id, adapter_dir, prompt_ids, sampling_params
        
        Requirements:
            - 3.1: 持续监听 ZMQ PULL socket 接收请求
        
        Note:
            这是一个异步方法，会阻塞直到收到请求。
        """
        request_json = await self.request_receiver.recv_json()
        return request_json
    
    async def _process_request(self, request: dict) -> dict:
        """
        处理单个请求（向后兼容方法）
        
        这是一个向后兼容的方法，用于支持旧的测试代码。
        实际的批处理逻辑在 _process_requests() 中实现。
        
        Args:
            request: 请求消息
        
        Returns:
            dict: 响应消息
        
        Note:
            这个方法主要用于测试和向后兼容。
            生产环境应该使用 _process_requests() 进行批处理。
        """
        try:
            request_id = request['request_id']
            adapter_dir = request.get('adapter_dir')
            prompt_ids = request['prompt_ids']
            sampling_params = request.get('sampling_params', {})
            
            # TODO: Phase 1 简化实现 - 实际推理逻辑将在后续任务中完善
            # 这里先返回一个占位响应，确保消息流通
            
            # 加载 adapter（如果需要且未加载）
            if adapter_dir and adapter_dir not in self.adapter_cache:
                # TODO: 实现 adapter 加载逻辑
                print(f"[Worker {self.worker_id}] Adapter loading not yet implemented: {adapter_dir}")
            
            # 执行推理
            # TODO: 调用模型的 forward 方法进行实际推理
            # 目前返回占位数据
            output_ids = prompt_ids + [1, 2, 3]  # 占位：简单追加一些 token
            metadata = {
                'finish_reason': 'length',
                'prompt_tokens': len(prompt_ids),
                'completion_tokens': 3
            }
            
            return {
                'request_id': request_id,
                'worker_id': self.worker_id,
                'output_ids': output_ids,
                'metadata': metadata,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            # 捕获异常并返回错误响应
            print(f"[Worker {self.worker_id}] Error processing request {request.get('request_id', 'unknown')}: {str(e)}")
            return {
                'request_id': request.get('request_id', 'unknown'),
                'worker_id': self.worker_id,
                'output_ids': [],
                'metadata': {},
                'success': False,
                'error': str(e)
            }
    
    async def _process_requests(self) -> List[dict]:
        """
        处理请求批次（使用 ReqQueue 管理）
        
        使用 ReqQueue 生成新批次，合并到当前批次，执行推理，并生成响应。
        
        Returns:
            List[dict]: 响应列表，每个响应包含：
                - request_id: 请求唯一标识符
                - worker_id: Worker ID
                - output_ids: 输出 token IDs
                - metadata: 元数据
                - success: 是否成功
                - error: 错误信息（如果失败）
        
        Requirements:
            - 3.1: 持续监听 ZMQ PULL socket 接收请求
            - 3.2: 解析请求消息并提取必要的参数
            - 3.4: 使用现有的模型推理逻辑处理请求
            - 3.5: 生成包含 output_ids 和 metadata 的响应消息
        
        Note:
            Phase 1 简化实现：
            - 使用 ReqQueue 管理批处理
            - 暂不实现实际推理逻辑（使用占位数据）
            - 完整的推理逻辑将在后续 Phase 中实现
        """
        try:
            # 使用 ReqQueue 生成新批次
            new_batch = self.req_queue.generate_new_batch(
                self.current_batch,
                self.lora_ranks,
                self.actual_adapter_size
            )
            
            if new_batch is not None:
                # 合并到当前批次
                if self.current_batch is None:
                    self.current_batch = new_batch
                else:
                    self.current_batch.merge(new_batch)
            
            # 执行推理
            if self.current_batch is not None and len(self.current_batch.reqs) > 0:
                # TODO: Phase 1 简化实现 - 实际推理逻辑将在后续任务中完善
                # 这里先返回占位响应，确保批处理流程正确
                
                responses = []
                for req in self.current_batch.reqs:
                    # 占位：简单追加一些 token
                    output_ids = req.prompt_ids + req.output_ids + [1, 2, 3]
                    req.output_ids.extend([1, 2, 3])
                    
                    metadata = {
                        'finish_reason': 'length',
                        'prompt_tokens': req.input_len,
                        'completion_tokens': len(req.output_ids)
                    }
                    
                    responses.append({
                        'request_id': req.request_id,
                        'worker_id': self.worker_id,
                        'output_ids': output_ids,
                        'metadata': metadata,
                        'success': True,
                        'error': None
                    })
                    
                    # 标记请求完成（简化实现）
                    req.has_generate_finished = True
                
                # 更新批次状态（移除已完成的请求）
                self.current_batch.filter_finished()
                
                # 如果批次为空，重置
                if self.current_batch.is_clear():
                    self.current_batch = None
                
                return responses
            
            return []
            
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error processing requests: {str(e)}")
            return []
    
    async def _send_response(self, response: dict) -> None:
        """
        发送响应
        
        通过 ZMQ PUSH socket 将响应发送到 Response Merger。
        
        Args:
            response: 响应消息
        
        Requirements:
            - 4.1: 通过 ZMQ PUSH socket 发送响应消息
        
        Note:
            使用 send_json 发送 JSON 格式的响应。
        """
        await self.response_sender.send_json(response)
    
    async def run(self) -> None:
        """
        主循环
        
        持续接收请求、处理请求批次并发送响应。
        使用 ReqQueue 进行批处理管理。
        
        Requirements:
            - 3.1: 持续监听 ZMQ PULL socket 接收请求
            - 3.5: 生成包含 output_ids 和 metadata 的响应消息
            - 8.3: 输出 Worker 就绪日志
        
        Note:
            这是一个无限循环，Worker 会一直运行直到进程被终止。
            使用 ReqQueue 管理批处理，支持多个请求并发处理。
        """
        print(f"Worker {self.worker_id} ready on GPU {self.gpu_id}")
        
        while True:
            try:
                # 尝试接收新请求并添加到队列（非阻塞）
                try:
                    request = await asyncio.wait_for(
                        self._receive_request(), 
                        timeout=0.01  # 10ms 超时
                    )
                    req_obj = self._convert_to_req_object(request)
                    self.req_queue.append(req_obj)
                except asyncio.TimeoutError:
                    # 超时是正常的，继续处理现有批次
                    pass
                
                # 处理请求批次
                responses = await self._process_requests()
                
                # 发送响应
                for response in responses:
                    await self._send_response(response)
                
            except Exception as e:
                print(f"[Worker {self.worker_id}] Error in main loop: {str(e)}")
                # 继续运行，不因单个错误而终止
