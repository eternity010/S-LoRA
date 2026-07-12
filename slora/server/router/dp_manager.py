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
import time
import json
import torch
import argparse
import asyncio
import zmq
import zmq.asyncio
import multiprocessing as mp
import logging
from typing import Any, Dict, List, Optional, Union

from slora.server.router.round_robin_router import RoundRobinRouter
from slora.server.router.adapter_aware_router import AdapterAwareRouter
from slora.server.router.worker_state import WorkerState, RoutingConfig
from slora.server.router.worker_state_cache import WorkerStateCache
from slora.server.router.replica_manager import ReplicaManager
from slora.server.io_struct import AbortReq, Req, ReqDetokenizationState
from slora.server.sampling_params import SamplingParams


logger = logging.getLogger(__name__)


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
                 router_port: int, response_port: int, detoken_port: int = None):
        """
        初始化 Data Parallel Router Manager
        
        Args:
            args: 命令行参数，包含：
                - num_workers: Worker 数量（可选）
                - gpu_ids: GPU ID 列表字符串（可选，如 "0,1,2"）
                - routing_strategy: 路由策略 ('adapter-aware' or 'round-robin')
                - routing_w1: 缓存亲和性权重（可选）
                - routing_w2: 负载惩罚权重（可选）
                - max_queue_length: 最大队列长度阈值（可选）
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
            - 8.1 (Adapter-Aware): 支持路由策略选择参数
            - 8.2 (Adapter-Aware): 支持 Round-Robin 回退
        
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
        
        # 路由策略配置 (Requirements 8.1, 8.2)
        self.routing_strategy = getattr(args, 'routing_strategy', 'round-robin')
        
        # 初始化路由器
        self.router: Union[RoundRobinRouter, AdapterAwareRouter] = self._create_router()
        
        # 加载并传递 adapter ranks 到路由器（仅 adapter-aware 模式）
        # Requirements: 3.1 (Rank-Aware Routing)
        if self.routing_strategy == 'adapter-aware':
            adapter_ranks = self._load_adapter_ranks()
            if isinstance(self.router, AdapterAwareRouter):
                self.router.set_adapter_ranks(adapter_ranks)
                print(f"[DataParallelRouterManager] Loaded {len(adapter_ranks)} adapter ranks for rank-aware routing")
        
        # Worker 状态缓存（用于 adapter-aware 路由）
        self.worker_state_cache: Optional[WorkerStateCache] = None
        if self.routing_strategy == 'adapter-aware':
            routing_config = self._get_routing_config()
            self.worker_state_cache = WorkerStateCache(
                num_workers=self.num_workers,
                heartbeat_interval_ms=routing_config.heartbeat_interval_ms,
                heartbeat_timeout_ms=routing_config.heartbeat_timeout_ms
            )
        
        # ZMQ 通信（将在 _setup_zmq 中初始化）
        self.context = None
        self.request_receiver = None
        self.request_senders = []
        self.send_to_detokenization = None  # 发送到 detokenization 的 socket
        self.state_receiver = None  # 接收 Worker 状态上报的 socket
        self.state_receiver_port = None  # 状态接收端口
        
        # 请求统计（Requirement 8.2）
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'worker_request_counts': [0] * self.num_workers,  # 每个 Worker 的请求计数
            'start_time': None,  # 将在 run() 中设置
        }
        
        # Worker 就绪状态
        self.workers_ready = False
        self.ready_port = None  # 用于接收 Worker 就绪信号的端口
        
        # ReplicaManager（热门 Adapter 主动复制，仅 adapter-aware 模式 + --enable-replication）
        self.replica_manager: Optional[ReplicaManager] = None
        self._enable_replication = getattr(args, 'enable_replication', False)
        if self._enable_replication and self.routing_strategy == 'adapter-aware':
            self._init_replica_manager()

        self._routing_debug_file = os.environ.get("SLORA_ROUTING_DEBUG_FILE")
        if self._routing_debug_file:
            debug_dir = os.path.dirname(self._routing_debug_file)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
            with open(self._routing_debug_file, "w"):
                pass
        
        print(f"[DataParallelRouterManager] Configuration:")
        print(f"[DataParallelRouterManager]   Number of workers: {self.num_workers}")
        print(f"[DataParallelRouterManager]   GPU IDs: {self.gpu_ids}")
        print(f"[DataParallelRouterManager]   Routing strategy: {self.routing_strategy}")
        print(f"[DataParallelRouterManager]   Router port: {router_port}")
        print(f"[DataParallelRouterManager]   Response port: {response_port}")
        print(f"[DataParallelRouterManager]   Detoken port: {detoken_port}")
        if hasattr(args, 'model_dir'):
            print(f"[DataParallelRouterManager]   Model directory: {args.model_dir}")
        if hasattr(args, 'max_total_token_num'):
            print(f"[DataParallelRouterManager]   Max total tokens: {args.max_total_token_num}")
        if hasattr(args, 'batch_max_tokens'):
            print(f"[DataParallelRouterManager]   Batch max tokens: {args.batch_max_tokens}")
        if self.routing_strategy == 'adapter-aware':
            routing_config = self._get_routing_config()
            print(f"[DataParallelRouterManager]   Routing w1 (cache affinity): {routing_config.w1}")
            print(f"[DataParallelRouterManager]   Routing w2 (load penalty): {routing_config.w2}")
            print(f"[DataParallelRouterManager]   Routing w3 (rank mismatch penalty): {routing_config.w3}")
            print(f"[DataParallelRouterManager]   Default LoRA rank: {routing_config.default_lora_rank}")
            print(f"[DataParallelRouterManager]   Max rank diff: {routing_config.max_rank_diff}")
            print(f"[DataParallelRouterManager]   Max queue length: {routing_config.max_queue_length}")
            print(f"[DataParallelRouterManager]   Hot adapter threshold: {routing_config.hot_adapter_threshold} req/s")
            print(f"[DataParallelRouterManager]   Load metric: {routing_config.load_metric}")
        if self.replica_manager:
            print(f"[DataParallelRouterManager]   Replication: ENABLED")
            print(f"[DataParallelRouterManager]   Patrol interval: {self.replica_manager.patrol_interval_sec}s")
            print(f"[DataParallelRouterManager]   Cooldown: {self.replica_manager.cooldown_sec}s")
            print(f"[DataParallelRouterManager]   Protection: {self.replica_manager.protection_sec}s")
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
                
                # 注意：不在这里验证 CUDA 可用性，因为子进程可能还没有初始化 CUDA
                # Worker 进程会在启动时设置 CUDA_VISIBLE_DEVICES 并验证
                
                print(f"[DataParallelRouterManager] Using specified GPU IDs: {gpu_ids}")
                return gpu_ids
                
            except ValueError as e:
                raise ValueError(f"Failed to parse GPU IDs '{gpu_ids_str}': {str(e)}")
        else:
            # 未指定 GPU IDs，使用 0 到 num_workers-1
            gpu_ids = list(range(self.num_workers))
            print(f"[DataParallelRouterManager] Using default GPU IDs: {gpu_ids}")
            return gpu_ids
    
    def _get_routing_config(self) -> RoutingConfig:
        """
        获取路由配置
        
        从命令行参数中提取路由配置参数，创建 RoutingConfig 实例。
        
        Returns:
            RoutingConfig: 路由配置实例
        
        Requirements: 6.1-6.5 (Adapter-Aware), 6.1-6.4 (Rank-Aware)
        """
        return RoutingConfig(
            strategy=getattr(self.args, 'routing_strategy', 'round-robin'),
            w1=getattr(self.args, 'routing_w1', 1.0),
            w2=getattr(self.args, 'routing_w2', 1.0),
            w3=getattr(self.args, 'routing_w3', 0.0),  # Rank mismatch penalty weight
            default_lora_rank=getattr(self.args, 'default_lora_rank', 16),  # Default rank for unknown adapters
            max_rank_diff=getattr(self.args, 'max_rank_diff', 64),  # Max rank difference for normalization
            heartbeat_interval_ms=getattr(self.args, 'heartbeat_interval_ms', 100),
            heartbeat_timeout_ms=getattr(self.args, 'heartbeat_timeout_ms', 300),
            max_queue_length=getattr(self.args, 'max_queue_length', 100),
            hot_adapter_threshold=getattr(self.args, 'hot_adapter_threshold', 1e9),
            max_total_token_num=getattr(self.args, 'max_total_token_num', 6000),
            batch_max_tokens=getattr(self.args, 'batch_max_tokens', 1000),
            hidden_dim=getattr(self.args, 'hidden_dim', None) or 4096,  # Worker 会通过心跳自动更新
            decode_cost_alpha=getattr(self.args, 'decode_cost_alpha', None),
            load_metric=getattr(self.args, 'load_metric', 'rwpt'),
        )
    
    def _init_replica_manager(self) -> None:
        """
        初始化 ReplicaManager，注册 send_preload_to_worker 作为预加载回调。

        Requirements: 7.3, 7.4, 9.4
        """
        routing_config = self._get_routing_config()
        self.replica_manager = ReplicaManager(
            num_workers=self.num_workers,
            capacity=float(routing_config.batch_max_tokens),
            w1=routing_config.w1,
            w2=routing_config.w2,
            patrol_interval_sec=getattr(self.args, 'patrol_interval_sec', 1.0),
            cooldown_sec=getattr(self.args, 'cooldown_sec', 5.0),
            protection_sec=getattr(self.args, 'protection_sec', 30.0),
            congestion_threshold=getattr(self.args, 'replication_congestion_threshold', 1.0),
            ema_alpha=getattr(self.args, 'ema_alpha', 0.3),
            max_protected_per_worker=getattr(self.args, 'max_protected_per_worker', 2),
            preload_callback=self.send_preload_to_worker,
        )
        print(f"[DataParallelRouterManager] ReplicaManager initialized")

    def _load_adapter_ranks(self) -> dict:
        """
        加载 adapter rank 信息
        
        从 lora_dirs 配置中加载每个 adapter 的 rank 信息。
        使用 get_lora_config 读取 adapter_config.json 中的 rank 值。
        对于缺失或无效的配置，使用默认 rank 值。
        
        Returns:
            Dict[str, int]: adapter 路径到 rank 的映射
        
        Requirements: 3.1
        
        Note:
            这个方法在 adapter-aware 路由模式下被调用。
            默认 rank 值从 routing config 中获取。
        """
        from slora.models.peft.lora_adapter import get_lora_config
        
        adapter_ranks = {}
        lora_dirs = getattr(self.args, 'lora_dirs', [])
        
        if not lora_dirs:
            logger.info("No lora_dirs configured, adapter ranks will be empty")
            return adapter_ranks
        
        # 获取默认 rank 值
        routing_config = self._get_routing_config()
        default_rank = getattr(routing_config, 'default_lora_rank', 16)
        
        logger.info(f"Loading adapter ranks from {len(lora_dirs)} lora_dirs...")
        
        for i, lora_dir in enumerate(lora_dirs):
            try:
                # 使用 get_lora_config 读取配置
                config, _ = get_lora_config(lora_dir, getattr(self.args, 'dummy', False))
                rank = config.get("r", default_rank)
                adapter_ranks[lora_dir] = rank
                
                # 只在第一个、最后一个和每20个 adapter 时输出，避免日志过多
                if i == 0 or i == len(lora_dirs) - 1 or (i + 1) % 20 == 0:
                    logger.info(f"  [{i+1}/{len(lora_dirs)}] {lora_dir}: rank={rank}")
                
            except Exception as e:
                # 配置缺失或无效，使用默认 rank
                logger.warning(f"Failed to load rank for {lora_dir}: {e}, using default rank {default_rank}")
                adapter_ranks[lora_dir] = default_rank
        
        logger.info(f"Loaded ranks for {len(adapter_ranks)} adapters")
        return adapter_ranks

    def _optimistically_update_worker_load(self, worker_id: int, request: dict) -> None:
        """
        乐观更新 Router 侧缓存的 Worker 负载状态。

        该估计仅用于填补 Worker 状态上报之间的短暂窗口，后续会被真实上报覆盖。
        """
        if self.routing_strategy != 'adapter-aware' or not hasattr(self.router, 'worker_states'):
            return

        state = self.router.worker_states.get(worker_id)
        if state is None:
            return

        state.queue_length += 1

        routing_config = getattr(self.router, 'config', None)
        if routing_config is None:
            return

        prompt_ids = request.get('prompt_ids')
        if prompt_ids is None:
            return

        try:
            prompt_len = len(prompt_ids)
        except TypeError:
            return

        if prompt_len <= 0:
            return

        if routing_config.load_metric == 'token_count':
            state.pending_raw_tokens += prompt_len
            return

        if routing_config.load_metric != 'rwpt':
            return

        adapter_dir = request.get('adapter_dir')
        if not adapter_dir:
            return

        hidden_size = getattr(routing_config, 'hidden_dim', None) or 4096
        gamma = 2.0 / (3.0 * hidden_size)
        default_rank = getattr(routing_config, 'default_lora_rank', 16)
        adapter_ranks = getattr(self.router, 'adapter_ranks', {})
        rank = adapter_ranks.get(adapter_dir, default_rank)

        estimated_prefill = int(prompt_len * (1.0 + gamma * rank))
        state.pending_prefill_tokens += estimated_prefill

    def _build_routing_debug_record(self, worker_id: int, request: dict) -> Optional[Dict[str, Any]]:
        if self.routing_strategy != 'adapter-aware' or not hasattr(self.router, 'worker_states'):
            return None

        adapter_dir = request.get('adapter_dir', '')
        prompt_ids = request.get('prompt_ids')
        try:
            prompt_len = len(prompt_ids) if prompt_ids is not None else None
        except TypeError:
            prompt_len = None

        routing_config = getattr(self.router, 'config', None)
        capacity = getattr(self.router, '_capacity', 0)
        load_metric = getattr(routing_config, 'load_metric', 'rwpt') if routing_config else 'rwpt'
        worker_debug = {}

        for wid, state in sorted(self.router.worker_states.items()):
            if load_metric == 'queue_length':
                load_pressure = state.queue_length
            elif load_metric == 'token_count':
                load_pressure = state.pending_raw_tokens / capacity if capacity > 0 else 0.0
            else:
                load_pressure = state.pending_prefill_tokens / capacity if capacity > 0 else 0.0

            if routing_config is not None and hasattr(self.router, '_effective_cache_affinity_weight'):
                effective_w1 = self.router._effective_cache_affinity_weight(load_pressure)
            else:
                effective_w1 = getattr(routing_config, 'w1', 1.0) if routing_config else 1.0

            has_adapter = state.has_adapter(adapter_dir)
            cache_bonus = effective_w1 if has_adapter else 0.0
            try:
                score = self.router.calculate_score(wid, adapter_dir)
            except Exception:
                score = None

            worker_debug[str(wid)] = {
                'selected': wid == worker_id,
                'score': score,
                'has_adapter': has_adapter,
                'cache_bonus': cache_bonus,
                'load_pressure': load_pressure,
                'queue_length': state.queue_length,
                'pending_prefill_tokens': state.pending_prefill_tokens,
                'pending_raw_tokens': state.pending_raw_tokens,
                'active_decode_seqs': state.active_decode_seqs,
                'cached_adapters': len(state.cached_adapters),
                'is_healthy': state.is_healthy,
            }

        return {
            'timestamp': time.time(),
            'request_id': request.get('request_id'),
            'adapter_dir': adapter_dir,
            'adapter_name': adapter_dir.rstrip('/').split('/')[-1] if adapter_dir else '',
            'prompt_len': prompt_len,
            'selected_worker': worker_id,
            'load_metric': load_metric,
            'routing_w1': getattr(routing_config, 'w1', None) if routing_config else None,
            'routing_w2': getattr(routing_config, 'w2', None) if routing_config else None,
            'routing_w3': getattr(routing_config, 'w3', None) if routing_config else None,
            'capacity': capacity,
            'workers': worker_debug,
        }

    def _write_routing_debug_record(self, worker_id: int, request: dict) -> None:
        routing_debug_file = getattr(self, '_routing_debug_file', None)
        if not routing_debug_file:
            return

        record = self._build_routing_debug_record(worker_id, request)
        if record is None:
            return

        try:
            with open(routing_debug_file, "a") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write routing debug record: {e}")
    
    def _create_router(self) -> Union[RoundRobinRouter, AdapterAwareRouter]:
        """
        创建路由器实例
        
        根据配置的路由策略创建相应的路由器。
        
        Returns:
            路由器实例（RoundRobinRouter 或 AdapterAwareRouter）
        
        Requirements: 8.1, 8.2
        """
        if self.routing_strategy == 'adapter-aware':
            routing_config = self._get_routing_config()
            router = AdapterAwareRouter(
                num_workers=self.num_workers,
                config=routing_config
            )
            alpha_status = "auto-profiled" if routing_config.decode_cost_alpha is None else f"{routing_config.decode_cost_alpha}"
            print(f"[DataParallelRouterManager] Created AdapterAwareRouter with config: "
                  f"w1={routing_config.w1}, w2={routing_config.w2}, "
                  f"w3={routing_config.w3}, hidden_dim={routing_config.hidden_dim}, "
                  f"decode_cost_alpha={alpha_status}, "
                  f"max_queue={routing_config.max_queue_length}")
            return router
        else:
            # 默认使用 Round-Robin 路由
            router = RoundRobinRouter(self.num_workers)
            print(f"[DataParallelRouterManager] Created RoundRobinRouter")
            return router
    
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
            这是一个异步方法，会等待所有 Worker 发送就绪信号。
        """
        import sys
        import socket
        
        print(f"[DataParallelRouterManager] ========== Starting Workers ==========")
        
        # 分配一个端口用于接收 Worker 就绪信号
        def find_free_port(start_port, exclude_ports):
            port = start_port
            while port < start_port + 100:
                if port in exclude_ports:
                    port += 1
                    continue
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', port))
                        return port
                except OSError:
                    port += 1
            raise RuntimeError(f"Failed to find free port starting from {start_port}")
        
        exclude_ports = set(self.worker_ports) | {self.router_port, self.response_port, self.detoken_port}
        self.ready_port = find_free_port(51000, exclude_ports)
        print(f"[DataParallelRouterManager] Ready signal port: {self.ready_port}")
        
        # 创建 ZMQ socket 接收就绪信号
        ready_context = zmq.Context()
        ready_receiver = ready_context.socket(zmq.PULL)
        ready_receiver.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
        ready_receiver.bind(f"tcp://127.0.0.1:{self.ready_port}")
        
        # 启动 Response Merger 进程
        print(f"[DataParallelRouterManager] Starting Response Merger...")
        self._start_response_merger()
        
        # 顺序启动 Worker，减少并发 preload 对磁盘和 CPU 的争抢
        print(f"[DataParallelRouterManager] Starting {self.num_workers} worker(s) in serial preload mode...")
        print(f"[DataParallelRouterManager] Waiting for each worker to be ready before starting the next one...")

        ready_workers = set()
        max_wait_time = 7200  # 最多等待 120 分钟（真实 adapter 冷启动可能超过 60 分钟）
        start_wait = asyncio.get_event_loop().time()

        for i in range(self.num_workers):
            print(f"[DataParallelRouterManager] Starting Worker {i}...")
            print(f"[DataParallelRouterManager]   GPU ID: {self.gpu_ids[i]}")
            print(f"[DataParallelRouterManager]   Request port: {self.worker_ports[i]}")
            print(f"[DataParallelRouterManager]   Response port: {self.response_port}")

            worker = self._start_worker(i, self.gpu_ids[i])
            self.workers.append(worker)

            print(f"[DataParallelRouterManager] Worker {i} process started (PID: {worker.pid})")
            print(f"[DataParallelRouterManager] Waiting for Worker {i} to be ready...")
            print(f"[DataParallelRouterManager] (This may take a while while models and adapters are loading)")

            while i not in ready_workers:
                elapsed = asyncio.get_event_loop().time() - start_wait
                if elapsed > max_wait_time:
                    print(f"[DataParallelRouterManager] ERROR: Timeout waiting for workers to be ready")
                    break

                failed_workers = []
                for worker_id, worker_proc in enumerate(self.workers):
                    if not worker_proc.is_alive() and worker_id not in ready_workers:
                        exitcode = worker_proc.exitcode
                        failed_workers.append((worker_id, exitcode))

                if failed_workers:
                    print(f"[DataParallelRouterManager] ========== Worker Startup Failed ==========")
                    for worker_id, exitcode in failed_workers:
                        print(f"[DataParallelRouterManager] Worker {worker_id} (GPU {self.gpu_ids[worker_id]}) "
                              f"failed with exit code {exitcode}")

                    for worker_proc in self.workers:
                        if worker_proc.is_alive():
                            worker_proc.terminate()
                            worker_proc.join(timeout=5)

                    ready_receiver.close()
                    ready_context.term()
                    raise RuntimeError(f"Worker startup failed: {failed_workers}")

                try:
                    msg = ready_receiver.recv_json()
                    worker_id = msg.get('worker_id')
                    if worker_id is not None and worker_id not in ready_workers:
                        ready_workers.add(worker_id)
                        print(f"[DataParallelRouterManager] Worker {worker_id} is READY ({len(ready_workers)}/{self.num_workers})")
                except zmq.Again:
                    await asyncio.sleep(0.1)
                    continue

            if i not in ready_workers:
                break
        
        ready_receiver.close()
        ready_context.term()
        
        if len(ready_workers) < self.num_workers:
            raise RuntimeError(f"Only {len(ready_workers)}/{self.num_workers} workers became ready")
        
        self.workers_ready = True
        
        # 输出醒目的就绪信息
        print("")
        print("=" * 80)
        print("[DataParallelRouterManager] ★★★ ALL WORKERS READY ★★★")
        print("=" * 80)
        print(f"[DataParallelRouterManager] Successfully started {self.num_workers} worker(s)")
        for i in range(self.num_workers):
            print(f"[DataParallelRouterManager]   Worker {i}: GPU {self.gpu_ids[i]}, PID {self.workers[i].pid}")
        print("=" * 80)
        print("")
    
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
        3. PUSH socket: 向 Detokenization 发送初始请求状态
        4. PULL socket: 接收 Worker 状态上报（仅 adapter-aware 模式）
        
        配置超时参数以防止通信阻塞。
        
        Requirements:
            - 2.3: 通过 ZMQ PUSH socket 发送请求消息
            - 7.4: 设置 socket 超时防止通信阻塞
            - 1.4 (Adapter-Aware): 接收 Worker 状态上报
        
        Note:
            Router Manager 使用 PULL socket 接收请求（多对一）
            Router Manager 使用多个 PUSH socket 向不同 Worker 发送请求（一对多）
            Router Manager 使用 PUSH socket 向 Detokenization 发送初始状态
            Router Manager 使用 PULL socket 接收 Worker 状态上报（adapter-aware 模式）
            
            通信模式：
            - API Server → Router Manager: PUSH/PULL
            - Router Manager → Workers: PUSH/PULL (每个 Worker 一个 PUSH socket)
            - Router Manager → Detokenization: PUSH/PULL
            - Workers → Router Manager: PUSH/PULL (状态上报)
            
            超时配置：
            - RCVTIMEO: 30000ms (30秒) - 接收超时
            - SNDTIMEO: 30000ms (30秒) - 发送超时
            - LINGER: 0 - 关闭时立即丢弃未发送消息
        """
        import socket
        
        # 创建异步 ZMQ context
        self.context = zmq.asyncio.Context()
        
        # 创建 PULL socket 接收来自 API Server 的请求
        self.request_receiver = self.context.socket(zmq.PULL)
        
        # 设置接收超时（30秒）
        self.request_receiver.setsockopt(zmq.RCVTIMEO, 30000)
        # 设置 LINGER 为 0，关闭时立即丢弃未发送消息
        self.request_receiver.setsockopt(zmq.LINGER, 0)
        # 设置 RCVHWM 为 0 表示无限制，防止消息丢失
        self.request_receiver.setsockopt(zmq.RCVHWM, 0)
        
        self.request_receiver.bind(f"tcp://127.0.0.1:{self.router_port}")
        
        print(f"[DataParallelRouterManager] ZMQ PULL socket bound to port {self.router_port} "
              f"(receiving from API Server, timeout=30s)")
        
        # 创建 PUSH socket 发送到 Detokenization
        if self.detoken_port:
            self.send_to_detokenization = self.context.socket(zmq.PUSH)
            self.send_to_detokenization.setsockopt(zmq.SNDTIMEO, 30000)
            self.send_to_detokenization.setsockopt(zmq.LINGER, 0)
            self.send_to_detokenization.connect(f"tcp://127.0.0.1:{self.detoken_port}")
            
            print(f"[DataParallelRouterManager] ZMQ PUSH socket connected to detokenization port {self.detoken_port} "
                  f"(timeout=30s)")
        
        # 为每个 Worker 创建 PUSH socket
        self.request_senders = []
        for i, port in enumerate(self.worker_ports):
            sender = self.context.socket(zmq.PUSH)
            
            # 设置发送超时（30秒）
            sender.setsockopt(zmq.SNDTIMEO, 30000)
            # 设置 LINGER 为 0，关闭时立即丢弃未发送消息
            sender.setsockopt(zmq.LINGER, 0)
            # 设置 SNDHWM (Send High Water Mark) 为 0 表示无限制
            # 这样可以防止消息在队列满时被丢弃
            sender.setsockopt(zmq.SNDHWM, 0)
            
            sender.bind(f"tcp://127.0.0.1:{port}")
            self.request_senders.append(sender)
            print(f"[DataParallelRouterManager] ZMQ PUSH socket bound to port {port} "
                  f"(sending to Worker {i}, timeout=30s, HWM=unlimited)")
        
        # 设置状态接收 socket（仅 adapter-aware 模式）
        if self.routing_strategy == 'adapter-aware':
            self._setup_state_receiver()
        
        print(f"[DataParallelRouterManager] ZMQ communication setup complete: "
              f"{len(self.request_senders)} worker sockets created with timeout=30s")
    
    def _setup_state_receiver(self) -> None:
        """
        设置状态接收 socket
        
        创建 ZMQ PULL socket 用于接收 Worker 状态上报。
        
        Requirements: 1.4 (Adapter-Aware)
        """
        import socket as sock_module
        
        # 找一个可用端口
        def find_free_port(start_port, exclude_ports):
            port = start_port
            while port < start_port + 100:
                if port in exclude_ports:
                    port += 1
                    continue
                try:
                    with sock_module.socket(sock_module.AF_INET, sock_module.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', port))
                        return port
                except OSError:
                    port += 1
            raise RuntimeError(f"Failed to find free port starting from {start_port}")
        
        exclude_ports = set(self.worker_ports) | {self.router_port, self.response_port}
        if self.detoken_port:
            exclude_ports.add(self.detoken_port)
        if self.ready_port:
            exclude_ports.add(self.ready_port)
        
        self.state_receiver_port = find_free_port(52000, exclude_ports)
        
        # 创建 PULL socket 接收 Worker 状态上报
        self.state_receiver = self.context.socket(zmq.PULL)
        self.state_receiver.setsockopt(zmq.RCVTIMEO, 100)  # 100ms 超时，非阻塞
        self.state_receiver.setsockopt(zmq.LINGER, 0)
        self.state_receiver.setsockopt(zmq.RCVHWM, 0)
        self.state_receiver.bind(f"tcp://127.0.0.1:{self.state_receiver_port}")
        
        print(f"[DataParallelRouterManager] ZMQ PULL socket bound to port {self.state_receiver_port} "
              f"(receiving Worker state reports)")
    
    def get_state_receiver_address(self) -> Optional[str]:
        """
        获取状态接收地址
        
        Returns:
            状态接收地址，格式为 "tcp://host:port"，如果未启用则返回 None
        """
        if self.state_receiver_port:
            return f"tcp://127.0.0.1:{self.state_receiver_port}"
        return None
    
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
                  self.worker_ports[worker_id], self.response_port, self.ready_port,
                  self.state_receiver_port),  # 传递状态上报端口
            name=f"GPUWorker-{worker_id}"
        )
        proc.start()
        return proc
    
    async def route_request(self, request: dict) -> None:
        """
        路由请求到 Worker
        
        根据配置的路由策略选择一个 Worker，然后通过 ZMQ 发送请求。
        支持 Round-Robin 和 Adapter-Aware 两种路由策略。
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
            - 8.3 (Adapter-Aware): 保持现有 API 接口不变
        
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
        request_id_short = request.get('request_id', 'unknown')[:8]
        adapter_dir = request.get('adapter_dir', '')
        
        while retry_count < max_retries:
            try:
                # 根据路由策略选择 Worker
                if self.routing_strategy == 'adapter-aware':
                    # 使用 AdapterAwareRouter 选择 Worker
                    # AdapterAwareRouter.select_worker 需要 adapter_dir 参数
                    worker_id = self.router.select_worker(adapter_dir)
                else:
                    # 使用 Round Robin Router 选择 Worker
                    worker_id = self.router.select_worker()
                
                # DEBUG 模式：打印路由日志
                if os.environ.get('DEBUG', '0') == '1':
                    if self.routing_strategy == 'adapter-aware':
                        # 获取更详细的路由信息
                        cache_hit = False
                        if hasattr(self.router, 'worker_states'):
                            state = self.router.worker_states.get(worker_id)
                            if state:
                                cache_hit = state.has_adapter(adapter_dir)
                        print(f"[Router] {request_id_short}... -> Worker {worker_id} "
                              f"(adapter-aware, cache_hit={cache_hit})")
                    else:
                        print(f"[Router] {request_id_short}... -> Worker {worker_id} (round-robin)")
                
                # 通过 ZMQ PUSH socket 发送请求到选定的 Worker
                await self.request_senders[worker_id].send_json(request)

                self._write_routing_debug_record(worker_id, request)
                
                # 乐观更新 Router 端的 Worker 状态，缓解状态上报窗口带来的低估
                if self.routing_strategy == 'adapter-aware':
                    self._optimistically_update_worker_load(worker_id, request)
                
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
    
    async def reset_all_adapter_caches(self) -> dict:
        """
        重置所有 Worker 的 Adapter 缓存
        
        向所有 Worker 广播 reset_cache 命令，等待所有 Worker 完成后返回。
        同时重置路由器的 adapter_to_workers 倒排索引。
        
        用于实验间的 cache 重置，确保实验公平性。
        
        Returns:
            dict: 包含重置结果的字典
                - success: 是否全部成功
                - total_cleared: 总共清除的 adapter 数量
                - worker_results: 每个 Worker 的结果
                - error: 错误信息（如果有）
        """
        import json
        import os
        import uuid
        
        print(f"[DataParallelRouterManager] Resetting adapter caches on all {self.num_workers} workers...")
        
        # 生成唯一的请求 ID
        reset_request_id = f"reset_{uuid.uuid4().hex[:8]}"
        ack_file = f"/tmp/slora_reset_cache_{reset_request_id}.jsonl"
        try:
            os.remove(ack_file)
        except FileNotFoundError:
            pass
        
        # 向所有 Worker 发送 reset_cache 命令
        reset_command = {
            'type': 'reset_cache',
            'request_id': reset_request_id,
        }
        
        try:
            # 广播到所有 Worker
            for worker_id in range(self.num_workers):
                await self.request_senders[worker_id].send_json(reset_command)
            
            print(f"[DataParallelRouterManager] Reset commands sent to all workers, waiting for responses...")

            worker_results = {}
            deadline = time.time() + 15.0
            while time.time() < deadline and len(worker_results) < self.num_workers:
                if os.path.exists(ack_file):
                    try:
                        with open(ack_file, 'r') as f:
                            for line in f:
                                response = json.loads(line)
                                worker_id = response.get('worker_id')
                                if worker_id is not None:
                                    worker_results[int(worker_id)] = response
                    except (OSError, json.JSONDecodeError):
                        pass
                if len(worker_results) < self.num_workers:
                    await asyncio.sleep(0.05)

            missing_workers = sorted(set(range(self.num_workers)) - set(worker_results))
            failed_workers = sorted(
                worker_id for worker_id, result in worker_results.items()
                if not result.get('success', False)
            )
            if missing_workers or failed_workers:
                return {
                    'success': False,
                    'num_workers': self.num_workers,
                    'worker_results': worker_results,
                    'message': (
                        f"Cache reset incomplete: missing={missing_workers}, "
                        f"failed={failed_workers}"
                    ),
                    'error': 'Worker cache reset acknowledgement failed',
                }
            
            # 重置路由器的 adapter 索引
            if hasattr(self.router, 'adapter_to_workers'):
                self.router.adapter_to_workers.clear()
                print(f"[DataParallelRouterManager] Router adapter index cleared")
            
            # 重置本轮实验统计并立即刷新 stats 文件
            self._reset_experiment_stats()
            self._write_stats_file()
            
            print(f"[DataParallelRouterManager] Adapter cache reset completed")
            
            return {
                'success': True,
                'num_workers': self.num_workers,
                'worker_results': worker_results,
                'message': f'Reset completed on {self.num_workers} workers',
                'error': None
            }
            
        except Exception as e:
            error_msg = f"Error resetting adapter caches: {str(e)}"
            print(f"[DataParallelRouterManager] ERROR: {error_msg}")
            return {
                'success': False,
                'num_workers': self.num_workers,
                'message': error_msg,
                'error': str(e)
            }
        finally:
            try:
                os.remove(ack_file)
            except FileNotFoundError:
                pass

    def _reset_experiment_stats(self) -> None:
        """Reset per-experiment statistics kept by the router manager and router."""
        self.stats['total_requests'] = 0
        self.stats['successful_requests'] = 0
        self.stats['failed_requests'] = 0
        self.stats['worker_request_counts'] = [0] * self.num_workers
        self.stats['start_time'] = time.time()

        if hasattr(self.router, 'reset_stats'):
            self.router.reset_stats()

        print("[DataParallelRouterManager] Per-experiment statistics reset")

    async def send_preload_to_worker(self, worker_id: int, adapter_dir: str, protection_sec: float = 30.0) -> dict:
        """
        向指定 Worker 发送预加载 adapter 指令

        通过 ZMQ PUSH socket 发送 preload_adapter 消息，Worker 端在消息循环中处理。

        Args:
            worker_id: 目标 Worker ID
            adapter_dir: adapter 目录路径
            protection_sec: 淘汰保护时长（秒），默认 30

        Returns:
            {success: bool, error: str|None}

        Requirements: 2.6
        """
        try:
            if worker_id < 0 or worker_id >= self.num_workers:
                return {"success": False, "error": f"Invalid worker_id: {worker_id}"}

            message = {
                "type": "preload_adapter",
                "adapter_dir": adapter_dir,
                "protection_sec": protection_sec,
            }
            await self.request_senders[worker_id].send_json(message)
            logger.info(f"Sent preload_adapter to Worker {worker_id}: {adapter_dir}")
            return {"success": True, "error": None}

        except Exception as e:
            logger.error(f"Failed to send preload_adapter to Worker {worker_id}: {e}")
            return {"success": False, "error": str(e)}

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
            只有在所有 Worker 就绪后才开始检查。
        """
        import time
        
        # 等待所有 Worker 就绪
        while not self.workers_ready:
            await asyncio.sleep(1)
        
        print(f"[DataParallelRouterManager] Worker health check started (interval: 10s)")
        
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
    
    def _write_stats_file(self, stats_file: str = "/tmp/slora_routing_stats.json") -> None:
        """
        将当前路由统计写入 stats 文件（原子写入）
        
        用于定期刷新统计快照，以及在 reset_stats 后立即刷新，
        确保 runner 读到的是 reset 后的干净数据。
        """
        import time
        import json
        import os
        
        try:
            # 计算运行时间
            if self.stats['start_time'] is not None:
                elapsed_time = time.time() - self.stats['start_time']
                throughput = self.stats['total_requests'] / elapsed_time if elapsed_time > 0 else 0
            else:
                elapsed_time = 0
                throughput = 0
            
            # 计算缓存命中率（仅 adapter-aware 模式）
            cache_hit_rate = 0.0
            cache_hits = 0
            cache_misses = 0
            if self.routing_strategy == 'adapter-aware' and hasattr(self.router, 'get_stats'):
                router_stats = self.router.get_stats()
                cache_hit_rate = router_stats.get('cache_hit_rate', 0.0)
                cache_hits = router_stats.get('cache_hits', 0)
                cache_misses = router_stats.get('cache_misses', 0)
            
            stats_data = {
                'routing_strategy': self.routing_strategy,
                'total_requests': self.stats['total_requests'],
                'successful_requests': self.stats['successful_requests'],
                'failed_requests': self.stats['failed_requests'],
                'throughput': throughput,
                'elapsed_time': elapsed_time,
                'cache_hit_rate': cache_hit_rate,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'worker_request_counts': {str(i): self.stats['worker_request_counts'][i] for i in range(self.num_workers)},
                'num_workers': self.num_workers,
                'gpu_ids': self.gpu_ids,
                'timestamp': time.time(),
                'config_update_id': getattr(self, '_last_config_update_id', None),
            }

            routing_config = getattr(self.router, 'config', None)
            if routing_config is not None:
                stats_data['routing_config'] = {
                    'w1': routing_config.w1,
                    'w2': routing_config.w2,
                    'w3': routing_config.w3,
                    'load_metric': routing_config.load_metric,
                }
            
            temp_file = stats_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(stats_data, f)
            os.rename(temp_file, stats_file)
        except Exception as e:
            print(f"[DataParallelRouterManager] Warning: Failed to write stats file: {e}")

    def _check_config_update(self, config_update_file: str) -> None:
        """
        检查并应用路由配置更新
        
        从配置文件读取更新参数，调用 router.update_config() 应用更新，
        然后删除配置文件。
        
        Args:
            config_update_file: 配置更新文件路径 (如 /tmp/slora_routing_config_update.json)
        
        Note:
            - 仅在 adapter-aware 路由策略下有效
            - 配置文件格式: {"w1": float, "w2": float, "w3": float, "reset_stats": bool}
            - 所有参数都是可选的
        """
        import os
        import json
        
        try:
            if not os.path.exists(config_update_file):
                return
            
            # 读取配置文件
            with open(config_update_file, 'r') as f:
                update = json.load(f)

            update_id = update.pop('update_id', None)
            
            # 删除配置文件（无论更新是否成功）
            os.remove(config_update_file)
            
            # 仅 adapter-aware 路由支持配置更新
            if self.routing_strategy != 'adapter-aware':
                print(f"[DataParallelRouterManager] Config update ignored: "
                      f"routing strategy is '{self.routing_strategy}', not 'adapter-aware'")
                return
            
            if not hasattr(self.router, 'update_config'):
                print(f"[DataParallelRouterManager] Config update ignored: "
                      f"router does not support update_config()")
                return
            
            # 应用配置更新
            result = self.router.update_config(**update)
            self._last_config_update_id = update_id
            print(f"[DataParallelRouterManager] Routing config updated: {result}")
            
            # 同步更新 ReplicaManager 的 w1/w2 日志状态；复制触发阈值不跟随 w2 变化
            if self.replica_manager and ('w1' in update or 'w2' in update):
                new_w1 = result.get('w1', self.replica_manager.w1)
                new_w2 = result.get('w2', self.replica_manager.w2)
                self.replica_manager.update_config(new_w1, new_w2)
            
            # 如果执行了 reset_stats，立即刷新 stats 文件，
            # 避免 runner 读到 reset 前的旧快照（竞态修复）
            self._write_stats_file()
            
        except json.JSONDecodeError as e:
            print(f"[DataParallelRouterManager] ERROR: Invalid config file format: {e}")
            # 尝试删除损坏的配置文件
            try:
                os.remove(config_update_file)
            except:
                pass
        except Exception as e:
            print(f"[DataParallelRouterManager] ERROR in config update: {str(e)}")
            import traceback
            traceback.print_exc()
    
    async def _check_cache_reset(self, trigger_file: str, result_file: str) -> None:
        """
        检查并执行 adapter cache 重置
        
        从触发文件检测重置请求，执行重置操作，然后写入结果文件。
        
        Args:
            trigger_file: 触发文件路径 (如 /tmp/slora_reset_adapter_cache.trigger)
            result_file: 结果文件路径 (如 /tmp/slora_reset_adapter_cache.result)
        
        Note:
            - 触发文件存在时执行重置
            - 重置完成后写入结果文件
            - 触发文件由 API server 创建，结果文件由此方法创建
        """
        import os
        import json
        
        try:
            if not os.path.exists(trigger_file):
                return

            with open(trigger_file, 'r') as f:
                trigger = json.load(f)
            reset_id = trigger.get('reset_id')
            
            print(f"[DataParallelRouterManager] Cache reset triggered")
            
            # 执行重置
            result = await self.reset_all_adapter_caches()
            result['reset_id'] = reset_id
            
            # 写入结果文件
            result_temp_file = result_file + ".tmp"
            with open(result_temp_file, 'w') as f:
                json.dump(result, f)
            os.replace(result_temp_file, result_file)
            
            # 删除触发文件
            try:
                os.remove(trigger_file)
            except:
                pass
            
            print(f"[DataParallelRouterManager] Cache reset completed: {result.get('message', '')}")
            
        except Exception as e:
            print(f"[DataParallelRouterManager] ERROR in cache reset: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 写入错误结果
            try:
                with open(result_file, 'w') as f:
                    json.dump({
                        'success': False,
                        'message': f'Error: {str(e)}',
                        'error': str(e)
                    }, f)
            except:
                pass
    
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
            只有在所有 Worker 就绪后才开始输出统计。
        """
        import time
        import json
        import os
        
        # 统计文件路径
        stats_file = "/tmp/slora_routing_stats.json"
        
        # 等待所有 Worker 就绪
        while not self.workers_ready:
            await asyncio.sleep(1)
        
        print(f"[DataParallelRouterManager] Statistics reporting started (interval: 20s)")
        print(f"[DataParallelRouterManager] Stats file: {stats_file}")
        
        # 配置更新文件路径
        config_update_file = "/tmp/slora_routing_config_update.json"
        
        # Cache 重置触发文件路径
        cache_reset_trigger_file = "/tmp/slora_reset_adapter_cache.trigger"
        cache_reset_result_file = "/tmp/slora_reset_adapter_cache.result"
        
        while True:
            try:
                # 等待 5 秒（更频繁地更新统计文件）
                await asyncio.sleep(5)
                
                # 检查并应用配置更新
                self._check_config_update(config_update_file)
                
                # 检查并执行 cache 重置
                await self._check_cache_reset(cache_reset_trigger_file, cache_reset_result_file)
                
                await asyncio.sleep(5)
                
                # 计算运行时间
                if self.stats['start_time'] is not None:
                    elapsed_time = time.time() - self.stats['start_time']
                    throughput = self.stats['total_requests'] / elapsed_time if elapsed_time > 0 else 0
                else:
                    elapsed_time = 0
                    throughput = 0
                
                # 计算缓存命中率（仅 adapter-aware 模式）
                cache_hit_rate = 0.0
                if self.routing_strategy == 'adapter-aware' and hasattr(self.router, 'get_stats'):
                    router_stats = self.router.get_stats()
                    cache_hit_rate = router_stats.get('cache_hit_rate', 0.0)
                
                # 写入统计文件（复用 _write_stats_file）
                self._write_stats_file(stats_file)
                
                # 每 20 秒输出一次到控制台
                if int(elapsed_time) % 20 < 5:
                    print(f"[DataParallelRouterManager] ========== Statistics Summary ==========")
                    print(f"[DataParallelRouterManager] Total Requests: {self.stats['total_requests']}")
                    print(f"[DataParallelRouterManager] Successful Requests: {self.stats['successful_requests']}")
                    print(f"[DataParallelRouterManager] Failed Requests: {self.stats['failed_requests']}")
                    print(f"[DataParallelRouterManager] Average Throughput: {throughput:.2f} req/s")
                    print(f"[DataParallelRouterManager] Running Time: {elapsed_time:.2f}s")
                    if cache_hit_rate > 0:
                        print(f"[DataParallelRouterManager] Cache Hit Rate: {cache_hit_rate:.2%}")
                    
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
            - 1.4 (Adapter-Aware): 接收并处理 Worker 状态上报
        
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
        
        # 启动状态接收处理任务（仅 adapter-aware 模式）
        state_receiver_task = None
        if self.routing_strategy == 'adapter-aware' and self.state_receiver:
            state_receiver_task = asyncio.create_task(self._process_worker_states())
            print(f"[DataParallelRouterManager] Worker state receiver task started")
        
        # 启动 ReplicaManager 巡检线程（仅启用副本机制时）
        patrol_task = None
        if self.replica_manager:
            patrol_task = asyncio.create_task(self.replica_manager._start_patrol_loop())
            print(f"[DataParallelRouterManager] ReplicaManager patrol loop started "
                  f"(interval={self.replica_manager.patrol_interval_sec}s)")
        
        while True:
            try:
                # 从 API Server 接收请求
                # HTTP Server 使用 send_pyobj 发送 Python 对象（pickle 序列化）
                recv_req = await self.request_receiver.recv_pyobj()
                
                # 处理不同类型的请求
                if isinstance(recv_req, tuple) and len(recv_req) == 4:
                    # 正常请求：(adapter_dir, prompt_ids, sampling_params, request_id)
                    adapter_dir, prompt_ids, sampling_params, request_id = recv_req
                    
                    # 创建 Req 对象（与张量并行模式一致）
                    req = Req(adapter_dir, request_id, prompt_ids, sampling_params)
                    
                    # 立即发送初始状态到 detokenization（与张量并行模式一致）
                    # 这样 detokenization 进程就知道有新请求了
                    if self.send_to_detokenization:
                        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
                    
                    # 转换 sampling_params 为 dict（如果它是对象）
                    if hasattr(sampling_params, '__dict__'):
                        sampling_params_dict = sampling_params.__dict__
                    else:
                        sampling_params_dict = sampling_params
                    
                    # 转换为 dict 格式供 route_request 使用
                    request = {
                        'request_id': request_id,
                        'adapter_dir': adapter_dir,
                        'prompt_ids': prompt_ids,
                        'sampling_params': sampling_params_dict
                    }
                    
                    # 路由请求到 Worker
                    await self.route_request(request)
                    
                elif isinstance(recv_req, AbortReq):
                    # Abort 请求
                    abort_req = recv_req
                    request_id = abort_req.req_id
                    
                    # 发送 abort 到 detokenization（与张量并行模式一致）
                    if self.send_to_detokenization:
                        self.send_to_detokenization.send_pyobj(abort_req)
                    
                    # TODO: Phase 1 暂不支持向 Worker 发送 abort
                    # Phase 2 需要实现向 Worker 转发 abort 请求
                    # 注意：AbortReq 在正常请求完成后也会被调用（用于清理资源）
                    # 因此不输出日志，避免冗余
                else:
                    print(f"[DataParallelRouterManager] WARNING: Unknown request type: {type(recv_req)}")
                
            except zmq.error.Again:
                # ZMQ 超时是正常的，不需要打印错误
                # 继续处理下一个请求
                continue
            except Exception as e:
                # 记录错误但继续运行
                print(f"[DataParallelRouterManager] ERROR in main loop: {str(e)}")
                import traceback
                traceback.print_exc()
                # 继续处理下一个请求
                continue
    
    async def _process_worker_states(self) -> None:
        """
        处理 Worker 状态上报
        
        持续接收 Worker 状态上报消息，更新路由器的 Worker 状态缓存。
        
        Requirements: 1.4 (Adapter-Aware)
        """
        import os
        debug_mode = os.environ.get('DEBUG', '0') == '1'
        
        # 设置文件 logger
        dlog = None
        if debug_mode:
            from slora.server.router.adapter_aware_router import _get_debug_file_logger
            dlog = _get_debug_file_logger()
        
        print(f"[DataParallelRouterManager] Worker state processing started")
        
        # 统计接收到的状态上报数量
        state_report_count = 0
        
        while True:
            try:
                # 非阻塞接收状态消息
                message = await self.state_receiver.recv_json()
                
                # 验证消息类型
                if message.get('type') != 'worker_state':
                    logger.warning(f"Unknown state message type: {message.get('type')}")
                    continue
                
                worker_id = message.get('worker_id')
                if worker_id is None or worker_id < 0 or worker_id >= self.num_workers:
                    logger.warning(f"Invalid worker_id in state message: {worker_id}")
                    continue
                
                # 创建 WorkerState 对象
                state = WorkerState(
                    worker_id=worker_id,
                    cached_adapters=set(message.get('cached_adapters', [])),
                    queue_length=message.get('queue_length', 0),
                    gpu_memory_free=message.get('gpu_memory_free', 0),
                    last_heartbeat=message.get('timestamp', time.time()),
                    is_healthy=True,
                    # RWPT fields
                    pending_prefill_tokens=message.get('pending_prefill_tokens', 0),
                    pending_raw_tokens=message.get('pending_raw_tokens', 0),
                    active_decode_seqs=message.get('active_decode_seqs', 0),
                    pool_used_ratio=message.get('pool_used_ratio', 0.0),
                    # Hot adapter replication: top-K RWPT contributors
                    top_k_rwpt_adapters=[
                        tuple(t) for t in message.get('top_k_rwpt_adapters', [])
                    ],
                )
                
                # 缓存 profiled_alpha（一次性，首次收到后不再更新）
                profiled_alpha = message.get('profiled_alpha')
                if profiled_alpha is not None:
                    if not hasattr(self, '_worker_profiled_alphas'):
                        self._worker_profiled_alphas = {}
                    if worker_id not in self._worker_profiled_alphas:
                        self._worker_profiled_alphas[worker_id] = profiled_alpha
                        logger.info(f"Worker {worker_id} profiled_alpha={profiled_alpha}")
                        # 取所有已上报 Worker 的中位数更新 Router 的 decode_cost_alpha
                        alphas = sorted(self._worker_profiled_alphas.values())
                        median_alpha = alphas[len(alphas) // 2]
                        if isinstance(self.router, AdapterAwareRouter):
                            self.router.config.decode_cost_alpha = median_alpha
                            self.router._decode_cost_alpha = median_alpha
                            logger.info(f"Router decode_cost_alpha updated to {median_alpha} "
                                       f"(median of {len(alphas)} workers)")
                
                # 缓存 hidden_dim（一次性，首次收到后不再更新）
                hidden_dim = message.get('hidden_dim')
                if hidden_dim is not None:
                    if not hasattr(self, '_worker_hidden_dim'):
                        self._worker_hidden_dim = None
                    if self._worker_hidden_dim is None:
                        self._worker_hidden_dim = hidden_dim
                        if isinstance(self.router, AdapterAwareRouter):
                            self.router.config.hidden_dim = hidden_dim
                            self.router._gamma = 2.0 / (3.0 * hidden_dim)
                            logger.info(f"Router hidden_dim updated to {hidden_dim} "
                                       f"(auto-detected from Worker {worker_id}), "
                                       f"gamma={self.router._gamma:.6f}")
                
                # 更新路由器状态
                if isinstance(self.router, AdapterAwareRouter):
                    self.router.update_worker_state(worker_id, state)
                
                # 更新状态缓存
                if self.worker_state_cache:
                    self.worker_state_cache.update(worker_id, state)
                
                # 同步更新 ReplicaManager（热门 Adapter 主动复制）
                if self.replica_manager:
                    self.replica_manager.update_worker_state(worker_id, state)
                
                state_report_count += 1
                
                # DEBUG: 每 100 次上报写入文件
                if dlog and state_report_count % 100 == 0:
                    dlog.debug(f"[StateRecv] total={state_report_count}, "
                               f"W{worker_id}: queue={state.queue_length}, "
                               f"cached={len(state.cached_adapters)}")
                
                logger.debug(f"Updated worker {worker_id} state: "
                            f"queue={state.queue_length}, "
                            f"adapters={len(state.cached_adapters)}")
                
            except zmq.Again:
                # 超时，继续循环
                await asyncio.sleep(0.01)  # 短暂休眠避免 CPU 空转
                continue
            except asyncio.CancelledError:
                logger.info("Worker state processing cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing worker state: {e}")
                await asyncio.sleep(0.1)
                continue


def run_gpu_worker_process(worker_id: int, gpu_id: int, args: argparse.Namespace,
                           request_port: int, response_port: int, ready_port: int,
                           state_report_port: int = None) -> None:
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
        ready_port: 发送就绪信号的端口
        state_report_port: 状态上报端口（用于智能路由）
    
    Requirements:
        - 7.1: Worker 启动失败时记录详细错误日志
        - 7.2: 通过退出码通知 Router Manager 启动失败
    
    Note:
        这个函数必须在模块级别定义，因为 multiprocessing 需要能够 pickle 它。
        任何异常都会导致进程以非零退出码退出，通知 Router Manager 启动失败。
        
        重要：Router Manager 会等待所有 Worker 发送就绪信号后才开始路由请求，
        因此不需要在模型加载期间缓存请求。
    """
    import sys
    import traceback
    import os
    from slora.server.router.gpu_worker import GPUWorker
    
    # 强制 stdout 无缓冲，确保日志及时输出
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

    worker_cpu_threads = 8
    os.environ["OMP_NUM_THREADS"] = str(worker_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(worker_cpu_threads)

    torch.set_num_threads(worker_cpu_threads)
    torch.set_num_interop_threads(1)
    
    def send_ready_signal(ready_port: int, worker_id: int):
        """发送就绪信号到 Router Manager"""
        import zmq
        context = zmq.Context()
        sender = context.socket(zmq.PUSH)
        sender.setsockopt(zmq.LINGER, 1000)
        sender.connect(f"tcp://127.0.0.1:{ready_port}")
        sender.send_json({'worker_id': worker_id, 'status': 'ready'})
        sender.close()
        context.term()
    
    try:
        print(f"[Worker {worker_id}] Starting worker process on GPU {gpu_id}...")
        print(f"[Worker {worker_id}] CPU thread limit: {worker_cpu_threads}")
        sys.stdout.flush()
        
        # 创建 Worker 实例
        print(f"[Worker {worker_id}] Creating GPUWorker instance...")
        sys.stdout.flush()
        worker = GPUWorker(worker_id, gpu_id, args)
        
        # 初始化请求队列
        print(f"[Worker {worker_id}] Setting up request queue...")
        sys.stdout.flush()
        worker._setup_request_queue()
        
        # 初始化模型 RPC（Task 2.9.1）
        # 使用 RPC 方式加载模型，而不是直接加载
        print(f"[Worker {worker_id}] Initializing model RPC...")
        sys.stdout.flush()
        asyncio.run(worker._init_model_rpc())
        
        print(f"[Worker {worker_id}] Model loading complete")
        sys.stdout.flush()
        
        # 设置 ZMQ 通信
        print(f"[Worker {worker_id}] Setting up ZMQ communication...")
        sys.stdout.flush()
        worker._setup_zmq(request_port, response_port)
        
        # 设置状态上报器（用于智能路由）
        if state_report_port:
            print(f"[Worker {worker_id}] Setting up state reporter (port={state_report_port})...")
            sys.stdout.flush()
            worker._setup_state_reporter(state_report_port)
        
        # 发送就绪信号
        # Router Manager 收到所有 Worker 的就绪信号后才开始路由请求
        print(f"[Worker {worker_id}] Sending ready signal...")
        sys.stdout.flush()
        send_ready_signal(ready_port, worker_id)
        
        print(f"[Worker {worker_id}] Worker initialization complete, starting main loop...")
        sys.stdout.flush()
        
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
