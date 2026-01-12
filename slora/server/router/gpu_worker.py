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
from typing import Optional


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
