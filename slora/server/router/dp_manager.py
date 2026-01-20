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
        detoken_port: Detokenization 进程的端口
        num_workers: Worker 数量
        gpu_ids: GPU ID 列表
        workers: Worker 进程列表
        worker_ports: Worker 端口列表
        merger_process: Response Merger 进程
        router: 路由器实例
    """
    
    def __init__(self, args: argparse.Namespace, 
                 router_port: int, response_port: int, detoken_port: int):
        """
        初始化 Data Parallel Router Manager
        
        Args:
            args: 命令行参数，包含：
                - num_workers: Worker 数量（可选）
                - gpu_ids: GPU ID 列表字符串（可选，如 "0,1,2"）
                - 其他模型和推理相关参数
            router_port: 接收来自 API Server 的请求的端口
            response_port: 接收来自 Worker 的响应的端口
            detoken_port: Detokenization 进程的端口
        
        Requirements:
            - 1.1: 根据配置创建指定数量的 GPU Worker 进程
            - 5.2: 支持通过 --num-workers 参数指定 Worker 数量
            - 5.3: 支持通过 --gpu-ids 参数指定使用的 GPU 列表
            - 4.1: Worker 通过 ZMQ PUSH socket 发送响应消息
            - 4.2: Response Merger 通过 ZMQ PULL socket 接收响应
            - 8.1: 输出详细的启动信息
        
        Note:
            如果未指定 num_workers，将自动检测可用 GPU 数量。
            如果未指定 gpu_ids，将使用所有可用的 GPU。
        """
        self.args = args
        self.router_port = router_port
        self.response_port = response_port
        self.detoken_port = detoken_port
        
        print(f"[DataParallelRouterManager] ========== Initialization Started ==========")
        
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
        
        # Response Merger 进程
        self.merger_process: Optional[mp.Process] = None
        
        # 路由器
        self.router = RoundRobinRouter(self.num_workers)
        
        # ZMQ 通信（将在 _setup_zmq 中初始化）
        self.context = None
        self.request_receiver = None
        self.request_senders = []
        
        # 请求统计（Requirement 8.2）
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'worker_request_counts': [0] * self.num_workers,  # 每个 Worker 的请求计数
            'start_time': None,  # 将在 run() 中设置
        }
        
        print(f"[DataParallelRouterManager] Configuration:")
        print(f"[DataParallelRouterManager]   Number of workers: {self.num_workers}")
        print(f"[DataParallelRouterManager]   GPU IDs: {self.gpu_ids}")
        print(f"[DataParallelRouterManager]   Router port: {router_port}")
        print(f"[DataParallelRouterManager]   Response port: {response_port}")
        print(f"[DataParallelRouterManager]   Detoken port: {detoken_port}")
        print(f"[DataParallelRouterManager]   Model directory: {args.model_dir}")
        print(f"[DataParallelRouterManager]   Max total tokens: {args.max_total_token_num}")
        print(f"[DataParallelRouterManager]   Batch max tokens: {args.batch_max_tokens}")
        print(f"[DataParallelRouterManager] ==========================================")
    
    def _detect_gpus(self) -> int:
        """
        自动检测可用 GPU 数量
        
        使用 PyTorch 的 cuda.device_count() 检测系统中可用的 GPU 数量。
        输出详细的 GPU 信息。
        
        Returns:
            int: 可用 GPU 数量
        
        Requirements:
            - 5.4: 自动检测可用 GPU 数量并创建对应数量的 Worker
            - 8.1: 输出详细的启动信息（GPU 型号、内存等）
        
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
        
        print(f"[DataParallelRouterManager] ========== GPU Detection ==========")
        print(f"[DataParallelRouterManager] Detected {num_gpus} available GPU(s)")
        
        # 输出每个 GPU 的详细信息
        for i in range(num_gpus):
            try:
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_name = gpu_props.name
                gpu_memory_gb = gpu_props.total_memory / (1024 ** 3)
                gpu_compute_capability = f"{gpu_props.major}.{gpu_props.minor}"
                
                print(f"[DataParallelRouterManager] GPU {i}:")
                print(f"[DataParallelRouterManager]   Name: {gpu_name}")
                print(f"[DataParallelRouterManager]   Memory: {gpu_memory_gb:.2f} GB")
                print(f"[DataParallelRouterManager]   Compute Capability: {gpu_compute_capability}")
            except Exception as e:
                print(f"[DataParallelRouterManager] GPU {i}: Unable to get properties ({str(e)})")
        
        print(f"[DataParallelRouterManager] =======================================")
        
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
        启动所有 Worker 进程和 Response Merger
        
        为每个 Worker 分配端口，然后启动所有 Worker 进程和 Response Merger 进程。
        等待所有 Worker 就绪后返回。检测 Worker 启动失败并终止所有进程。
        输出详细的启动进度日志。
        
        Requirements:
            - 1.1: 根据配置创建指定数量的 GPU Worker 进程
            - 1.5: 确认所有 Worker 处于就绪状态
            - 4.1: Worker 通过 ZMQ PUSH socket 发送响应消息
            - 4.2: Response Merger 通过 ZMQ PULL socket 接收响应
            - 7.1: Worker 启动失败时记录详细错误日志
            - 7.2: 检测 Worker 启动失败并终止所有进程
            - 8.1: 输出详细的启动信息
            - 8.3: 输出 Worker 就绪日志
        
        Raises:
            RuntimeError: 如果任何 Worker 启动失败
        
        Note:
            这是一个异步方法，会等待一段时间让 Worker 初始化。
            实际的就绪检测将在后续 Phase 中实现（通过心跳机制）。
        """
        import sys
        
        print(f"[DataParallelRouterManager] ========== Starting Workers ==========")
        
        # 分配端口
        self._allocate_ports()
        
        # 启动 Response Merger 进程
        print(f"[DataParallelRouterManager] Starting Response Merger...")
        self._start_response_merger()
        
        # 启动所有 Worker
        print(f"[DataParallelRouterManager] Starting {self.num_workers} worker(s)...")
        for i in range(self.num_workers):
            print(f"[DataParallelRouterManager] Starting Worker {i}...")
            print(f"[DataParallelRouterManager]   GPU ID: {self.gpu_ids[i]}")
            print(f"[DataParallelRouterManager]   Request port: {self.worker_ports[i]}")
            print(f"[DataParallelRouterManager]   Response port: {self.response_port}")
            
            worker = self._start_worker(i, self.gpu_ids[i])
            self.workers.append(worker)
            
            print(f"[DataParallelRouterManager] Worker {i} process started (PID: {worker.pid})")
        
        # 等待所有 Worker 就绪，同时检测启动失败
        # Phase 1 简化实现：固定等待时间 + 进程状态检查
        # Phase 2 将实现心跳机制进行实际的就绪检测
        print(f"[DataParallelRouterManager] Waiting for workers to initialize...")
        print(f"[DataParallelRouterManager] This may take a few minutes (loading models)...")
        
        # 分多次检查，每次等待 1 秒，总共等待 5 秒
        for check_round in range(5):
            await asyncio.sleep(1)
            
            # 检查所有 Worker 进程状态
            failed_workers = []
            for i, worker in enumerate(self.workers):
                if not worker.is_alive():
                    exitcode = worker.exitcode
                    failed_workers.append((i, exitcode))
            
            # 如果有 Worker 启动失败，终止所有进程并退出
            if failed_workers:
                print(f"[DataParallelRouterManager] ========== Worker Startup Failed ==========")
                for worker_id, exitcode in failed_workers:
                    print(f"[DataParallelRouterManager] Worker {worker_id} (GPU {self.gpu_ids[worker_id]}) "
                          f"failed with exit code {exitcode}")
                
                # 终止所有 Worker 进程
                print(f"[DataParallelRouterManager] Terminating all workers...")
                for i, worker in enumerate(self.workers):
                    if worker.is_alive():
                        print(f"[DataParallelRouterManager] Terminating worker {i}...")
                        worker.terminate()
                        worker.join(timeout=5)
                        if worker.is_alive():
                            print(f"[DataParallelRouterManager] Force killing worker {i}...")
                            worker.kill()
                
                # 终止 Response Merger 进程
                if self.merger_process and self.merger_process.is_alive():
                    print(f"[DataParallelRouterManager] Terminating Response Merger...")
                    self.merger_process.terminate()
                    self.merger_process.join(timeout=5)
                    if self.merger_process.is_alive():
                        print(f"[DataParallelRouterManager] Force killing Response Merger...")
                        self.merger_process.kill()
                
                # 抛出异常并退出
                error_msg = f"Worker startup failed. Failed workers: {failed_workers}"
                print(f"[DataParallelRouterManager] {error_msg}")
                print(f"[DataParallelRouterManager] ==========================================")
                raise RuntimeError(error_msg)
            
            print(f"[DataParallelRouterManager] Health check {check_round + 1}/5: All workers alive")
        
        print(f"[DataParallelRouterManager] ========== All Workers Ready ==========")
        print(f"[DataParallelRouterManager] Successfully started {self.num_workers} worker(s)")
        for i in range(self.num_workers):
            print(f"[DataParallelRouterManager] Worker {i}: GPU {self.gpu_ids[i]}, "
                  f"PID {self.workers[i].pid}, Port {self.worker_ports[i]}")
        print(f"[DataParallelRouterManager] =======================================")
    
    def _start_response_merger(self) -> None:
        """
        启动 Response Merger 进程
        
        创建一个新的进程来运行 Response Merger。
        Response Merger 负责收集来自所有 Worker 的响应并转发到 Detokenization 进程。
        
        Requirements:
            - 4.1: Worker 通过 ZMQ PUSH socket 发送响应消息
            - 4.2: Response Merger 通过 ZMQ PULL socket 接收响应
        
        Note:
            Response Merger 进程的目标函数是 run_response_merger_process，
            它会创建 ResponseMerger 实例并运行主循环。
        """
        from slora.server.router.response_merger import run_response_merger_process
        
        self.merger_process = mp.Process(
            target=run_response_merger_process,
            args=(self.response_port, self.detoken_port),
            name="ResponseMerger"
        )
        self.merger_process.start()
        
        print(f"[DataParallelRouterManager] Started Response Merger process "
              f"(worker_response_port={self.response_port}, detoken_port={self.detoken_port})")
    
    def _setup_zmq(self) -> None:
        """
        设置 ZMQ 通信
        
        创建 ZMQ context 和 sockets：
        1. PULL socket: 从 API Server 接收请求
        2. PUSH sockets: 向每个 Worker 发送请求
        
        配置超时参数以防止通信阻塞。
        
        Requirements:
            - 2.3: 通过 ZMQ PUSH socket 发送请求消息
            - 7.4: 设置 socket 超时防止通信阻塞
        
        Note:
            Router Manager 使用 PULL socket 接收请求（多对一）
            Router Manager 使用多个 PUSH socket 向不同 Worker 发送请求（一对多）
            
            通信模式：
            - API Server → Router Manager: PUSH/PULL
            - Router Manager → Workers: PUSH/PULL (每个 Worker 一个 PUSH socket)
            
            超时配置：
            - RCVTIMEO: 30000ms (30秒) - 接收超时
            - SNDTIMEO: 30000ms (30秒) - 发送超时
            - LINGER: 0 - 关闭时立即丢弃未发送消息
        """
        # 创建异步 ZMQ context
        self.context = zmq.asyncio.Context()
        
        # 创建 PULL socket 接收来自 API Server 的请求
        self.request_receiver = self.context.socket(zmq.PULL)
        
        # 设置接收超时（30秒）
        self.request_receiver.setsockopt(zmq.RCVTIMEO, 30000)
        # 设置 LINGER 为 0，关闭时立即丢弃未发送消息
        self.request_receiver.setsockopt(zmq.LINGER, 0)
        
        self.request_receiver.bind(f"tcp://127.0.0.1:{self.router_port}")
        
        print(f"[DataParallelRouterManager] ZMQ PULL socket bound to port {self.router_port} "
              f"(receiving from API Server, timeout=30s)")
        
        # 为每个 Worker 创建 PUSH socket
        self.request_senders = []
        for i, port in enumerate(self.worker_ports):
            sender = self.context.socket(zmq.PUSH)
            
            # 设置发送超时（30秒）
            sender.setsockopt(zmq.SNDTIMEO, 30000)
            # 设置 LINGER 为 0，关闭时立即丢弃未发送消息
            sender.setsockopt(zmq.LINGER, 0)
            
            sender.bind(f"tcp://127.0.0.1:{port}")
            self.request_senders.append(sender)
            print(f"[DataParallelRouterManager] ZMQ PUSH socket bound to port {port} "
                  f"(sending to Worker {i}, timeout=30s)")
        
        print(f"[DataParallelRouterManager] ZMQ communication setup complete: "
              f"{len(self.request_senders)} worker sockets created with timeout=30s")
    
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
        包含重试逻辑和超时处理。
        
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
            - 7.4: 实现重试逻辑（最多 3 次）
        
        Raises:
            Exception: 如果发送请求失败（重试 3 次后）
        
        Note:
            这是一个异步方法，使用 ZMQ 的异步 API 发送消息。
            路由决策会在 DEBUG 级别记录日志（Requirement 8.4）。
            发送失败时会重试最多 3 次，每次重试间隔 0.1 秒。
        """
        worker_id = None
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 使用 Round Robin Router 选择 Worker
                worker_id = self.router.select_worker()
                
                # DEBUG 级别记录路由决策（Requirement 8.4）
                if hasattr(self.args, 'log_level') and self.args.log_level == 'DEBUG':
                    print(f"[DataParallelRouterManager] DEBUG: Routing request {request.get('request_id')} "
                          f"to Worker {worker_id} (attempt {retry_count + 1}/{max_retries})")
                
                # 通过 ZMQ PUSH socket 发送请求到选定的 Worker
                await self.request_senders[worker_id].send_json(request)
                
                # 更新统计信息（Requirement 8.2）
                self.stats['total_requests'] += 1
                self.stats['successful_requests'] += 1  # Phase 1: 假设成功路由即为成功
                self.stats['worker_request_counts'][worker_id] += 1
                
                # 发送成功，返回
                return
                
            except zmq.Again as e:
                # ZMQ 超时（SNDTIMEO 触发）
                retry_count += 1
                print(f"[DataParallelRouterManager] ZMQ timeout routing request "
                      f"{request.get('request_id', 'unknown')} to Worker {worker_id} "
                      f"(attempt {retry_count}/{max_retries})")
                
                if retry_count >= max_retries:
                    # 超过重试次数，更新失败统计并抛出异常
                    self.stats['failed_requests'] += 1
                    error_msg = (f"Failed to route request {request.get('request_id', 'unknown')} "
                                f"to Worker {worker_id} after {max_retries} attempts: ZMQ timeout")
                    print(f"[DataParallelRouterManager] ERROR: {error_msg}")
                    raise Exception(error_msg)
                
                # 等待一段时间后重试
                await asyncio.sleep(0.1)
                
            except Exception as e:
                # 其他错误，记录日志并抛出
                retry_count += 1
                print(f"[DataParallelRouterManager] Error routing request "
                      f"{request.get('request_id', 'unknown')} to Worker {worker_id} "
                      f"(attempt {retry_count}/{max_retries}):")
                print(f"[DataParallelRouterManager]   Error type: {type(e).__name__}")
                print(f"[DataParallelRouterManager]   Error: {str(e)}")
                
                if retry_count >= max_retries:
                    # 超过重试次数，更新失败统计并抛出异常
                    self.stats['failed_requests'] += 1
                    error_msg = (f"Failed to route request {request.get('request_id', 'unknown')} "
                                f"to Worker {worker_id} after {max_retries} attempts: {str(e)}")
                    print(f"[DataParallelRouterManager] ERROR: {error_msg}")
                    raise Exception(error_msg)
                
                # 等待一段时间后重试
                await asyncio.sleep(0.1)
    
    async def _check_worker_health(self) -> None:
        """
        定期检查 Worker 进程健康状态
        
        每 10 秒检查一次所有 Worker 进程是否存活。
        如果发现进程退出，记录详细日志（worker_id、exitcode、时间）。
        
        Requirements:
            - 7.5: 定期检测 Worker 进程是否存活，记录进程退出日志
        
        Note:
            这是一个后台任务，在主循环启动时创建。
            当前实现只记录日志，不进行自动重启（Phase 2 功能）。
        """
        import time
        
        print(f"[DataParallelRouterManager] Starting worker health check task (interval: 10s)")
        
        while True:
            try:
                # 等待 10 秒
                await asyncio.sleep(10)
                
                # 检查所有 Worker 进程状态
                for i, worker in enumerate(self.workers):
                    if not worker.is_alive():
                        # Worker 进程已退出
                        exitcode = worker.exitcode
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        print(f"[DataParallelRouterManager] WARNING: Worker {i} (GPU {self.gpu_ids[i]}) "
                              f"has exited unexpectedly!")
                        print(f"[DataParallelRouterManager]   Worker ID: {i}")
                        print(f"[DataParallelRouterManager]   GPU ID: {self.gpu_ids[i]}")
                        print(f"[DataParallelRouterManager]   Exit code: {exitcode}")
                        print(f"[DataParallelRouterManager]   Timestamp: {timestamp}")
                        
                        # 根据退出码提供更多信息
                        if exitcode == 0:
                            print(f"[DataParallelRouterManager]   Status: Normal exit (exit code 0)")
                        elif exitcode == 1:
                            print(f"[DataParallelRouterManager]   Status: Error exit (exit code 1) - "
                                  f"Check worker logs for details")
                        elif exitcode == -9:
                            print(f"[DataParallelRouterManager]   Status: Killed by signal 9 (SIGKILL) - "
                                  f"Possible OOM or manual kill")
                        elif exitcode == -15:
                            print(f"[DataParallelRouterManager]   Status: Terminated by signal 15 (SIGTERM) - "
                                  f"Graceful shutdown requested")
                        elif exitcode and exitcode < 0:
                            print(f"[DataParallelRouterManager]   Status: Killed by signal {-exitcode}")
                        else:
                            print(f"[DataParallelRouterManager]   Status: Unknown exit code {exitcode}")
                
            except Exception as e:
                # 记录错误但继续监控
                print(f"[DataParallelRouterManager] ERROR in worker health check: {str(e)}")
                import traceback
                traceback.print_exc()
    
    async def _print_statistics(self) -> None:
        """
        定期输出请求统计信息
        
        每 10 秒输出一次统计摘要，包括：
        - 总请求数、成功数、失败数
        - 每个 Worker 的请求分布
        - 平均吞吐量（requests/second）
        
        Requirements:
            - 8.2: 每 10 秒输出一次整体的请求处理统计
        
        Note:
            这是一个后台任务，在主循环启动时创建。
        """
        import time
        
        print(f"[DataParallelRouterManager] Starting statistics reporting task (interval: 10s)")
        
        while True:
            try:
                # 等待 10 秒
                await asyncio.sleep(10)
                
                # 计算运行时间
                if self.stats['start_time'] is not None:
                    elapsed_time = time.time() - self.stats['start_time']
                    throughput = self.stats['total_requests'] / elapsed_time if elapsed_time > 0 else 0
                else:
                    elapsed_time = 0
                    throughput = 0
                
                # 输出统计摘要
                print(f"[DataParallelRouterManager] ========== Statistics Summary ==========")
                print(f"[DataParallelRouterManager] Total Requests: {self.stats['total_requests']}")
                print(f"[DataParallelRouterManager] Successful Requests: {self.stats['successful_requests']}")
                print(f"[DataParallelRouterManager] Failed Requests: {self.stats['failed_requests']}")
                print(f"[DataParallelRouterManager] Average Throughput: {throughput:.2f} req/s")
                print(f"[DataParallelRouterManager] Running Time: {elapsed_time:.2f}s")
                
                # 输出每个 Worker 的请求分布
                print(f"[DataParallelRouterManager] Worker Request Distribution:")
                for i in range(self.num_workers):
                    count = self.stats['worker_request_counts'][i]
                    percentage = (count / self.stats['total_requests'] * 100) if self.stats['total_requests'] > 0 else 0
                    print(f"[DataParallelRouterManager]   Worker {i} (GPU {self.gpu_ids[i]}): "
                          f"{count} requests ({percentage:.1f}%)")
                
                print(f"[DataParallelRouterManager] ==========================================")
                
            except Exception as e:
                # 记录错误但继续统计
                print(f"[DataParallelRouterManager] ERROR in statistics reporting: {str(e)}")
                import traceback
                traceback.print_exc()
    
    async def run(self) -> None:
        """
        主循环 - 持续接收和路由请求
        
        从 API Server 接收请求，然后使用路由器选择 Worker 并发送请求。
        同时启动后台任务定期检查 Worker 进程健康状态。
        这个方法会一直运行，直到进程被终止。
        
        Requirements:
            - 2.1: 新请求到达时使用 Round Robin Router 选择 Worker
            - 2.3: 通过 ZMQ 发送请求到选定的 Worker
            - 7.5: 定期检测 Worker 进程是否存活，记录进程退出日志
        
        Note:
            这是一个无限循环，会持续处理请求直到进程被终止。
            在实际部署中，应该添加优雅关闭机制（Phase 2）。
            Worker 健康检查作为后台任务运行，每 10 秒检查一次。
        """
        print(f"[DataParallelRouterManager] Starting main loop, listening on port {self.router_port}")
        
        # 设置统计开始时间（Requirement 8.2）
        import time
        self.stats['start_time'] = time.time()
        
        # 启动 Worker 健康检查后台任务
        health_check_task = asyncio.create_task(self._check_worker_health())
        print(f"[DataParallelRouterManager] Worker health check task started")
        
        # 启动统计报告后台任务（Requirement 8.2）
        stats_task = asyncio.create_task(self._print_statistics())
        print(f"[DataParallelRouterManager] Statistics reporting task started")
        
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
    包含完善的异常捕获和错误日志记录。
    
    Args:
        worker_id: Worker 的唯一标识符
        gpu_id: 分配的 GPU ID
        args: 命令行参数
        request_port: 接收请求的端口
        response_port: 发送响应的端口
    
    Requirements:
        - 7.1: Worker 启动失败时记录详细错误日志
        - 7.2: 通过退出码通知 Router Manager 启动失败
    
    Note:
        这个函数必须在模块级别定义，因为 multiprocessing 需要能够 pickle 它。
        任何异常都会导致进程以非零退出码退出，通知 Router Manager 启动失败。
    """
    import sys
    import traceback
    from slora.server.router.gpu_worker import GPUWorker
    
    try:
        print(f"[Worker {worker_id}] Starting worker process on GPU {gpu_id}...")
        
        # 创建 Worker 实例
        print(f"[Worker {worker_id}] Creating GPUWorker instance...")
        worker = GPUWorker(worker_id, gpu_id, args)
        
        # 设置 ZMQ 通信
        print(f"[Worker {worker_id}] Setting up ZMQ communication...")
        worker._setup_zmq(request_port, response_port)
        
        # 初始化请求队列
        print(f"[Worker {worker_id}] Setting up request queue...")
        worker._setup_request_queue()
        
        # 初始化模型 RPC（Task 2.9.1）
        # 使用 RPC 方式加载模型，而不是直接加载
        print(f"[Worker {worker_id}] Initializing model RPC...")
        asyncio.run(worker._init_model_rpc())
        
        print(f"[Worker {worker_id}] Worker initialization complete, starting main loop...")
        
        # 运行主循环
        asyncio.run(worker.run())
        
    except KeyboardInterrupt:
        # 优雅处理 Ctrl+C
        print(f"[Worker {worker_id}] Received keyboard interrupt, shutting down...")
        sys.exit(0)
        
    except Exception as e:
        # 捕获所有异常，记录详细错误日志和堆栈跟踪
        print(f"[Worker {worker_id}] FATAL ERROR during worker startup/execution:")
        print(f"[Worker {worker_id}] Error type: {type(e).__name__}")
        print(f"[Worker {worker_id}] Error message: {str(e)}")
        print(f"[Worker {worker_id}] Full traceback:")
        traceback.print_exc()
        
        # 通过非零退出码通知 Router Manager 启动失败
        print(f"[Worker {worker_id}] Exiting with error code 1")
        sys.exit(1)
