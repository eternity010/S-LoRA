"""
Data Parallel Router Manager 模块

该模块实现了数据并行模式下的路由管理器，负责：
1. 检测和管理 GPU 资源
2. 启动和管理多个 GPU Worker 进程
3. 路由请求到不同的 Worker
4. 管理 ZMQ 通信

Requirements:
    - 1.1: 根据配置创建指定数量的 GPU Worker 进程
    - 5.2: 支持通过 --num-workers 参数指定 Worker 数量
    - 5.3: 支持通过 --gpu-ids 参数指定使用的 GPU 列表
    - 5.4: 自动检测可用 GPU 数量
    - 5.5: 使用所有可用的 GPU
"""

import os
import torch
import argparse
import asyncio
import zmq
import zmq.asyncio
import multiprocessing as mp
from typing import List, Optional

from slora.server.router.round_robin_router import RoundRobinRouter


class DataParallelRouterManager:
    """
    数据并行路由管理器
    
    负责管理多个 GPU Worker 进程，并将请求路由到不同的 Worker。
    每个 Worker 运行在独立的进程中，拥有完整的基座模型副本。
    
    Attributes:
        args: 命令行参数
        router_port: 接收 HTTP 请求的端口
        response_port: 接收 Worker 响应的端口
        num_workers: Worker 数量
        gpu_ids: GPU ID 列表
        workers: Worker 进程列表
        worker_ports: Worker 端口列表
        router: 路由器实例
    """
    
    def __init__(self, args: argparse.Namespace, 
                 router_port: int, response_port: int):
        """
        初始化 Data Parallel Router Manager
        
        Args:
            args: 命令行参数，包含：
                - num_workers: Worker 数量（可选）
                - gpu_ids: GPU ID 列表字符串（可选，如 "0,1,2"）
                - 其他模型和推理相关参数
            router_port: 接收来自 API Server 的请求的端口
            response_port: 接收来自 Worker 的响应的端口
        
        Requirements:
            - 1.1: 根据配置创建指定数量的 GPU Worker 进程
            - 5.2: 支持通过 --num-workers 参数指定 Worker 数量
            - 5.3: 支持通过 --gpu-ids 参数指定使用的 GPU 列表
        
        Note:
            如果未指定 num_workers，将自动检测可用 GPU 数量。
            如果未指定 gpu_ids，将使用所有可用的 GPU。
        """
        self.args = args
        self.router_port = router_port
        self.response_port = response_port
        
        # Worker 管理
        # 如果未指定 num_workers，自动检测 GPU 数量
        self.num_workers = getattr(args, 'num_workers', None) or self._detect_gpus()
        
        # 解析 GPU ID 列表
        gpu_ids_str = getattr(args, 'gpu_ids', None)
        self.gpu_ids = self._parse_gpu_ids(gpu_ids_str)
        
        # 验证 GPU ID 数量与 Worker 数量匹配
        if len(self.gpu_ids) != self.num_workers:
            raise ValueError(
                f"Number of GPU IDs ({len(self.gpu_ids)}) does not match "
                f"number of workers ({self.num_workers})"
            )
        
        # Worker 进程和端口
        self.workers: List[mp.Process] = []
        self.worker_ports: List[int] = []
        
        # 路由器
        self.router = RoundRobinRouter(self.num_workers)
        
        # ZMQ 通信（将在 _setup_zmq 中初始化）
        self.context = None
        self.request_receiver = None
        self.request_senders = []
        
        print(f"[DataParallelRouterManager] Initialized with {self.num_workers} workers")
        print(f"[DataParallelRouterManager] GPU IDs: {self.gpu_ids}")
        print(f"[DataParallelRouterManager] Router port: {router_port}, Response port: {response_port}")
    
    def _detect_gpus(self) -> int:
        """
        自动检测可用 GPU 数量
        
        使用 PyTorch 的 cuda.device_count() 检测系统中可用的 GPU 数量。
        
        Returns:
            int: 可用 GPU 数量
        
        Requirements:
            - 5.4: 自动检测可用 GPU 数量并创建对应数量的 Worker
        
        Raises:
            RuntimeError: 如果没有可用的 GPU
        
        Note:
            这个方法在未指定 num_workers 参数时被调用。
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Data parallel mode requires GPUs.")
        
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            raise RuntimeError("No GPUs detected. Data parallel mode requires at least one GPU.")
        
        print(f"[DataParallelRouterManager] Detected {num_gpus} available GPUs")
        return num_gpus
    
    def _parse_gpu_ids(self, gpu_ids_str: Optional[str]) -> List[int]:
        """
        解析 GPU ID 列表
        
        将逗号分隔的 GPU ID 字符串解析为整数列表。
        如果未指定，则使用 0 到 num_workers-1 的所有 GPU。
        
        Args:
            gpu_ids_str: GPU ID 字符串，格式如 "0,1,2" 或 None
        
        Returns:
            List[int]: GPU ID 列表
        
        Requirements:
            - 5.3: 支持通过 --gpu-ids 参数指定使用的 GPU 列表
            - 5.5: 未指定 gpu-ids 时使用所有可用的 GPU
        
        Raises:
            ValueError: 如果 GPU ID 格式无效或 GPU ID 超出范围
        
        Examples:
            >>> self._parse_gpu_ids("0,1,2")
            [0, 1, 2]
            >>> self._parse_gpu_ids(None)  # num_workers=3
            [0, 1, 2]
        """
        if gpu_ids_str:
            try:
                # 解析逗号分隔的 GPU ID
                gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')]
                
                # 验证 GPU ID 有效性
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is not available")
                
                num_available_gpus = torch.cuda.device_count()
                for gpu_id in gpu_ids:
                    if gpu_id < 0 or gpu_id >= num_available_gpus:
                        raise ValueError(
                            f"Invalid GPU ID {gpu_id}. "
                            f"Available GPUs: 0-{num_available_gpus-1}"
                        )
                
                print(f"[DataParallelRouterManager] Using specified GPU IDs: {gpu_ids}")
                return gpu_ids
                
            except ValueError as e:
                raise ValueError(f"Failed to parse GPU IDs '{gpu_ids_str}': {str(e)}")
        else:
            # 未指定 GPU IDs，使用 0 到 num_workers-1
            gpu_ids = list(range(self.num_workers))
            print(f"[DataParallelRouterManager] Using default GPU IDs: {gpu_ids}")
            return gpu_ids
    
    def _allocate_ports(self) -> None:
        """
        为每个 Worker 分配端口
        
        为每个 Worker 分配唯一的端口号，用于 ZMQ 通信。
        端口从 50000 开始递增分配。
        
        Requirements:
            - 1.1: 根据配置创建指定数量的 GPU Worker 进程
        
        Note:
            端口分配策略：
            - Worker 0: 50000
            - Worker 1: 50001
            - Worker 2: 50002
            - ...
            
            这些端口用于 Router Manager 向 Worker 发送请求。
        """
        base_port = 50000
        self.worker_ports = []
        
        for i in range(self.num_workers):
            port = base_port + i
            self.worker_ports.append(port)
        
        print(f"[DataParallelRouterManager] Allocated ports for workers: {self.worker_ports}")
    
    async def start_workers(self) -> None:
        """
        启动所有 Worker 进程
        
        为每个 Worker 分配端口，然后启动所有 Worker 进程。
        等待所有 Worker 就绪后返回。
        
        Requirements:
            - 1.1: 根据配置创建指定数量的 GPU Worker 进程
            - 1.5: 确认所有 Worker 处于就绪状态
        
        Note:
            这是一个异步方法，会等待一段时间让 Worker 初始化。
            实际的就绪检测将在后续 Phase 中实现（通过心跳机制）。
        """
        # 分配端口
        self._allocate_ports()
        
        # 启动所有 Worker
        print(f"[DataParallelRouterManager] Starting {self.num_workers} workers...")
        for i in range(self.num_workers):
            worker = self._start_worker(i, self.gpu_ids[i])
            self.workers.append(worker)
            print(f"[DataParallelRouterManager] Started worker {i} on GPU {self.gpu_ids[i]}, "
                  f"port {self.worker_ports[i]}")
        
        # 等待所有 Worker 就绪
        # Phase 1 简化实现：固定等待时间
        # Phase 2 将实现心跳机制进行实际的就绪检测
        await asyncio.sleep(5)
        
        print(f"[DataParallelRouterManager] All {self.num_workers} workers started and ready")
    
    def _setup_zmq(self) -> None:
        """
        设置 ZMQ 通信
        
        创建 ZMQ context 和 sockets：
        1. PULL socket: 从 API Server 接收请求
        2. PUSH sockets: 向每个 Worker 发送请求
        
        Requirements:
            - 2.3: 通过 ZMQ PUSH socket 发送请求消息
        
        Note:
            Router Manager 使用 PULL socket 接收请求（多对一）
            Router Manager 使用多个 PUSH socket 向不同 Worker 发送请求（一对多）
            
            通信模式：
            - API Server → Router Manager: PUSH/PULL
            - Router Manager → Workers: PUSH/PULL (每个 Worker 一个 PUSH socket)
        """
        # 创建异步 ZMQ context
        self.context = zmq.asyncio.Context()
        
        # 创建 PULL socket 接收来自 API Server 的请求
        self.request_receiver = self.context.socket(zmq.PULL)
        self.request_receiver.bind(f"tcp://127.0.0.1:{self.router_port}")
        
        print(f"[DataParallelRouterManager] ZMQ PULL socket bound to port {self.router_port} "
              f"(receiving from API Server)")
        
        # 为每个 Worker 创建 PUSH socket
        self.request_senders = []
        for i, port in enumerate(self.worker_ports):
            sender = self.context.socket(zmq.PUSH)
            sender.bind(f"tcp://127.0.0.1:{port}")
            self.request_senders.append(sender)
            print(f"[DataParallelRouterManager] ZMQ PUSH socket bound to port {port} "
                  f"(sending to Worker {i})")
        
        print(f"[DataParallelRouterManager] ZMQ communication setup complete: "
              f"{len(self.request_senders)} worker sockets created")
    
    def _start_worker(self, worker_id: int, gpu_id: int) -> mp.Process:
        """
        启动单个 Worker 进程
        
        创建一个新的进程来运行 GPU Worker。
        Worker 将在指定的 GPU 上加载模型并处理请求。
        
        Args:
            worker_id: Worker 的唯一标识符
            gpu_id: 分配的 GPU ID
        
        Returns:
            mp.Process: Worker 进程对象
        
        Requirements:
            - 1.1: 根据配置创建指定数量的 GPU Worker 进程
            - 1.2: 为每个 Worker 分配唯一的 worker_id 和 gpu_id
        
        Note:
            Worker 进程的目标函数是 run_gpu_worker_process，
            它会创建 GPUWorker 实例并运行主循环。
        """
        proc = mp.Process(
            target=run_gpu_worker_process,
            args=(worker_id, gpu_id, self.args, 
                  self.worker_ports[worker_id], self.response_port),
            name=f"GPUWorker-{worker_id}"
        )
        proc.start()
        return proc
    
    async def route_request(self, request: dict) -> None:
        """
        路由请求到 Worker
        
        使用 Round Robin Router 选择一个 Worker，然后通过 ZMQ 发送请求。
        
        Args:
            request: 请求消息字典，包含：
                - request_id: 请求唯一标识符
                - adapter_dir: Adapter 路径
                - prompt_ids: 输入 token IDs
                - sampling_params: 采样参数
        
        Requirements:
            - 2.1: 使用 Round Robin Router 选择下一个 Worker
            - 2.2: 按照 Worker ID 的顺序循环分配请求
            - 2.3: 通过 ZMQ PUSH socket 发送请求消息
        
        Raises:
            Exception: 如果发送请求失败
        
        Note:
            这是一个异步方法，使用 ZMQ 的异步 API 发送消息。
            路由决策会在 DEBUG 级别记录日志（Requirement 8.4）。
        """
        try:
            # 使用 Round Robin Router 选择 Worker
            worker_id = self.router.select_worker()
            
            # DEBUG 级别记录路由决策（Requirement 8.4）
            if hasattr(self.args, 'log_level') and self.args.log_level == 'DEBUG':
                print(f"[DataParallelRouterManager] DEBUG: Routing request {request.get('request_id')} "
                      f"to Worker {worker_id}")
            
            # 通过 ZMQ PUSH socket 发送请求到选定的 Worker
            await self.request_senders[worker_id].send_json(request)
            
        except Exception as e:
            # Requirement 2.5: 发送请求失败时记录错误日志
            print(f"[DataParallelRouterManager] ERROR: Failed to route request "
                  f"{request.get('request_id', 'unknown')} to Worker {worker_id}: {str(e)}")
            raise
    
    async def run(self) -> None:
        """
        主循环 - 持续接收和路由请求
        
        从 API Server 接收请求，然后使用路由器选择 Worker 并发送请求。
        这个方法会一直运行，直到进程被终止。
        
        Requirements:
            - 2.1: 新请求到达时使用 Round Robin Router 选择 Worker
            - 2.3: 通过 ZMQ 发送请求到选定的 Worker
        
        Note:
            这是一个无限循环，会持续处理请求直到进程被终止。
            在实际部署中，应该添加优雅关闭机制（Phase 2）。
        """
        print(f"[DataParallelRouterManager] Starting main loop, listening on port {self.router_port}")
        
        while True:
            try:
                # 从 API Server 接收请求
                request = await self.request_receiver.recv_json()
                
                # 路由请求到 Worker
                await self.route_request(request)
                
            except Exception as e:
                # 记录错误但继续运行
                print(f"[DataParallelRouterManager] ERROR in main loop: {str(e)}")
                import traceback
                traceback.print_exc()
                # 继续处理下一个请求
                continue


def run_gpu_worker_process(worker_id: int, gpu_id: int, args: argparse.Namespace,
                           request_port: int, response_port: int) -> None:
    """
    Worker 进程的入口函数
    
    这个函数在独立的进程中运行，创建 GPUWorker 实例并启动主循环。
    
    Args:
        worker_id: Worker 的唯一标识符
        gpu_id: 分配的 GPU ID
        args: 命令行参数
        request_port: 接收请求的端口
        response_port: 发送响应的端口
    
    Note:
        这个函数必须在模块级别定义，因为 multiprocessing 需要能够 pickle 它。
    """
    from slora.server.router.gpu_worker import GPUWorker
    
    try:
        # 创建 Worker 实例
        worker = GPUWorker(worker_id, gpu_id, args)
        
        # 设置 ZMQ 通信
        worker._setup_zmq(request_port, response_port)
        
        # 初始化请求队列
        worker._setup_request_queue()
        
        # 初始化模型 RPC（Task 2.9.1）
        # 使用 RPC 方式加载模型，而不是直接加载
        asyncio.run(worker._init_model_rpc())
        
        # 运行主循环
        asyncio.run(worker.run())
        
    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
