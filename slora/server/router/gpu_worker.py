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
from slora.models.peft.lora_adapter import get_lora_config
from slora.server.router.model_infer.model_rpc import start_model_process, ModelRpcClient
from slora.server.input_params import InputParams


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
        self.model_rpc = None  # ModelRpcClient 实例
        self.adapter_cache = {}
        
        # Adapter 管理（借鉴 manager.py）
        self.lora_ranks = {}  # adapter_dir -> rank 映射，用于 ReqQueue 显存管理
        self.actual_adapter_size = 0  # 实际 adapter 占用的显存大小（cells）
        self.actual_adapter_memory_usage = 0  # 缓存实际的 adapter 内存占用（单位：cells）
        
        # 请求队列管理（复用 ReqQueue）
        self.req_queue = None  # 将在 _setup_request_queue() 中初始化
        self.current_batch = None
        
        # ZMQ 通信相关（待后续任务实现）
        self.context = None
        self.request_receiver = None
        self.response_sender = None
        
        # 设置 GPU 环境
        self._setup_gpu()
        
        # 初始化 Adapter rank 配置（Phase 1 必需）
        self._setup_adapter_config()
    
    def _setup_gpu(self) -> None:
        """
        设置 GPU 环境
        
        通过设置 CUDA_VISIBLE_DEVICES 环境变量，确保 Worker 只能看到分配给它的 GPU。
        然后将 PyTorch 的默认设备设置为该 GPU。
        输出详细的 GPU 信息（型号、内存、计算能力）。
        
        Requirements:
            - 1.3: 设置 CUDA_VISIBLE_DEVICES 环境变量为对应的 gpu_id
            - 8.1: 输出详细的启动信息（GPU 型号、内存等）
        
        Note:
            设置 CUDA_VISIBLE_DEVICES 后，从 Worker 的视角看，只有一个 GPU (索引为 0)。
            这样可以确保不同 Worker 之间的 GPU 隔离。
        """
        # 设置环境变量，限制可见的 GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        # 设置 PyTorch 默认设备为 GPU 0（因为 CUDA_VISIBLE_DEVICES 已经限制了可见 GPU）
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            
            # 获取 GPU 详细信息
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_name = gpu_props.name
            gpu_memory_gb = gpu_props.total_memory / (1024 ** 3)  # 转换为 GB
            gpu_compute_capability = f"{gpu_props.major}.{gpu_props.minor}"
            
            print(f"[Worker {self.worker_id}] ========== GPU Environment Setup ==========")
            print(f"[Worker {self.worker_id}] Physical GPU ID: {self.gpu_id}")
            print(f"[Worker {self.worker_id}] GPU Name: {gpu_name}")
            print(f"[Worker {self.worker_id}] Total Memory: {gpu_memory_gb:.2f} GB")
            print(f"[Worker {self.worker_id}] Compute Capability: {gpu_compute_capability}")
            print(f"[Worker {self.worker_id}] CUDA_VISIBLE_DEVICES: {self.gpu_id}")
            print(f"[Worker {self.worker_id}] PyTorch Device: cuda:0")
            print(f"[Worker {self.worker_id}] ==========================================")
        else:
            raise RuntimeError(f"[Worker {self.worker_id}] CUDA is not available on GPU {self.gpu_id}")
    
    def _setup_adapter_config(self) -> None:
        """
        初始化 Adapter 配置（借鉴 manager.py）
        
        读取所有 adapter 的 rank 配置，用于 ReqQueue 计算显存占用。
        这是 Phase 1 必需的功能，确保 ReqQueue 能够正确计算批次大小。
        
        Requirements:
            - 3.4: 使用现有的模型推理逻辑处理请求
        
        Note:
            - 借鉴 manager.py 的 lora_ranks 初始化逻辑
            - lora_ranks 字典用于 ReqQueue.generate_new_batch() 计算显存占用
            - None 键表示无 adapter 的情况（base 模型），rank 为 0
        """
        self.lora_ranks = {}
        
        # 检查是否有 lora_dirs 参数
        if hasattr(self.args, 'lora_dirs') and self.args.lora_dirs:
            lora_dirs = self.args.lora_dirs
            # 遍历所有 adapter 目录，读取配置并存储 rank
            for i, lora_dir in enumerate(lora_dirs):
                try:
                    config, _ = get_lora_config(lora_dir, getattr(self.args, 'dummy', False))
                    self.lora_ranks[lora_dir] = config["r"]
                    # 只在第一个、最后一个和每20个 adapter 时输出，避免日志过多
                    if i == 0 or i == len(lora_dirs) - 1 or (i + 1) % 20 == 0:
                        print(f"[Worker {self.worker_id}] Loaded adapter config: {lora_dir}, rank={config['r']}")
                except Exception as e:
                    print(f"[Worker {self.worker_id}] Warning: Failed to load adapter config from {lora_dir}: {e}")
                    # 如果加载失败，使用默认 rank
                    self.lora_ranks[lora_dir] = 8  # 默认 rank 值
        
        # 添加 None 键处理无 adapter 情况（base 模型）
        self.lora_ranks[None] = 0
        
        print(f"[Worker {self.worker_id}] Adapter rank configuration initialized: {len(self.lora_ranks)} adapters")
    
    async def _update_actual_adapter_usage(self) -> None:
        """
        查询并更新实际的 adapter 内存占用（借鉴 manager.py）
        
        通过 RPC 查询 LoRA 内存使用情况，提取 adapter 实际占用的 cells 数。
        用于并发控制的准确判断。
        
        Requirements:
            - 3.4: 使用现有的模型推理逻辑处理请求
        
        Note:
            - 借鉴 manager.py 的 _update_actual_adapter_usage() 方法
            - 在 adapter 加载/卸载后调用，更新实际占用
            - 如果查询失败，保持当前值不变（保守估计）
        """
        # 检查是否启用了 LoRA
        if getattr(self.args, 'no_lora', False):
            self.actual_adapter_memory_usage = 0
            return
        
        # 检查是否有 model_rpc（需要在模型加载后才能查询）
        if self.model_rpc is None:
            # 模型尚未加载，无法查询
            return
        
        try:
            # 查询 RPC 节点的 LoRA 内存使用情况
            memory_info = await self.model_rpc.check_lora_memory()
            
            if memory_info:
                # 计算实际的 adapter 占用：所有已加载 adapters 的总和
                adapter_cells_list = memory_info.get('adapter_cells', [])
                self.actual_adapter_memory_usage = sum(adapter_cells_list)
                
                # 同步更新 actual_adapter_size（用于 ReqQueue）
                self.actual_adapter_size = self.actual_adapter_memory_usage
                
                print(f"[Worker {self.worker_id}] Updated actual adapter usage: {self.actual_adapter_memory_usage} cells")
            else:
                # 查询失败，使用保守估计（当前值不变）
                print(f"[Worker {self.worker_id}] Warning: check_lora_memory() returned None")
        except Exception as e:
            # 查询出错，使用保守估计
            print(f"[Worker {self.worker_id}] Warning: Failed to query actual adapter usage: {e}")
    
    async def _load_adapters(self, adapter_dirs: set) -> None:
        """
        加载 Adapters（借鉴 manager.py）
        
        调用 RPC 的 load_adapters() 方法加载指定的 adapters。
        加载后调用 _update_actual_adapter_usage() 更新实际占用。
        
        Args:
            adapter_dirs: 需要加载的 adapter 目录集合
        
        Requirements:
            - 3.3: 根据请求中的 adapter_dir 加载对应的 Adapter
        
        Note:
            - 借鉴 manager.py 的 adapter 加载逻辑
            - 在批次生成后加载所需的 adapters
            - 加载后更新实际内存占用
            - Phase 1 简化实现：假设 model_rpc 已经初始化
        """
        # 检查是否启用了 LoRA
        if getattr(self.args, 'no_lora', False):
            return
        
        # 检查是否有 model_rpc（需要在模型加载后才能加载 adapter）
        if self.model_rpc is None:
            print(f"[Worker {self.worker_id}] Warning: model_rpc not initialized, cannot load adapters")
            return
        
        # 如果没有需要加载的 adapter，直接返回
        if not adapter_dirs:
            return
        
        try:
            # DEBUG 级别记录 adapter 加载详情（Requirement 8.5）
            import os
            if os.environ.get('DEBUG', '0') == '1':
                print(f"[Worker {self.worker_id}] DEBUG: Loading adapters:")
                for adapter_dir in adapter_dirs:
                    adapter_name = adapter_dir.split('/')[-1]
                    rank = self.lora_ranks.get(adapter_dir, 'unknown')
                    print(f"[Worker {self.worker_id}] DEBUG:   - {adapter_name} (rank={rank})")
            
            # 调用 RPC 加载 adapters
            await self.model_rpc.load_adapters(adapter_dirs)
            
            print(f"[Worker {self.worker_id}] Loaded {len(adapter_dirs)} adapters: "
                  f"{[d.split('/')[-1] for d in list(adapter_dirs)[:5]]}")
            
            # 加载后更新实际占用
            await self._update_actual_adapter_usage()
            
            # DEBUG 级别记录内存占用（Requirement 8.5）
            if os.environ.get('DEBUG', '0') == '1':
                print(f"[Worker {self.worker_id}] DEBUG: Adapter memory usage: "
                      f"{self.actual_adapter_memory_usage} cells")
            
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error loading adapters: {e}")
            # 加载失败不应该终止服务，继续运行
    
    def _setup_zmq(self, request_port: int, response_port: int) -> None:
        """
        设置 ZMQ 通信
        
        创建两个 ZMQ socket：
        1. PULL socket: 从 Router Manager 接收请求
        2. PUSH socket: 向 Response Merger 发送响应
        
        配置超时参数以防止通信阻塞。
        
        Args:
            request_port: 接收请求的端口号（Router Manager 的 PUSH socket 绑定的端口）
            response_port: 发送响应的端口号（Response Merger 的 PULL socket 绑定的端口）
        
        Requirements:
            - 2.3: 通过 ZMQ PUSH socket 发送请求消息
            - 4.1: 通过 ZMQ PUSH socket 发送响应消息
            - 7.4: 设置 socket 超时防止通信阻塞
        
        Note:
            使用 zmq.asyncio.Context 以支持异步操作。
            Worker 使用 PULL socket 接收（多对一），使用 PUSH socket 发送（一对一）。
            
            超时配置：
            - RCVTIMEO: 30000ms (30秒) - 接收超时
            - SNDTIMEO: 30000ms (30秒) - 发送超时
            - LINGER: 0 - 关闭时立即丢弃未发送消息
        """
        # 创建异步 ZMQ context
        self.context = zmq.asyncio.Context()
        
        # 创建 PULL socket 接收请求
        # Router Manager 使用 PUSH 发送，Worker 使用 PULL 接收
        self.request_receiver = self.context.socket(zmq.PULL)
        
        # 设置接收超时（30秒）
        self.request_receiver.setsockopt(zmq.RCVTIMEO, 30000)
        # 设置 LINGER 为 0，关闭时立即丢弃未发送消息
        self.request_receiver.setsockopt(zmq.LINGER, 0)
        # 设置 RCVHWM (Receive High Water Mark) 为 0 表示无限制
        # 这样可以防止消息在队列满时被丢弃
        self.request_receiver.setsockopt(zmq.RCVHWM, 0)
        
        self.request_receiver.connect(f"tcp://127.0.0.1:{request_port}")
        
        # 创建 PUSH socket 发送响应
        # Worker 使用 PUSH 发送，Response Merger 使用 PULL 接收
        self.response_sender = self.context.socket(zmq.PUSH)
        
        # 设置发送超时（30秒）
        self.response_sender.setsockopt(zmq.SNDTIMEO, 30000)
        # 设置 LINGER 为 0，关闭时立即丢弃未发送消息
        self.response_sender.setsockopt(zmq.LINGER, 0)
        # 设置 SNDHWM 为 0 表示无限制，防止消息丢失
        self.response_sender.setsockopt(zmq.SNDHWM, 0)
        
        self.response_sender.connect(f"tcp://127.0.0.1:{response_port}")
        
        print(f"[Worker {self.worker_id}] ZMQ communication setup complete: "
              f"request_port={request_port}, response_port={response_port}, timeout=30s")
    
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
    
    async def _init_model_rpc(self) -> None:
        """
        初始化模型 RPC 连接
        
        创建 ModelRpcClient 连接到模型进程。在数据并行模式下，每个 Worker
        运行在独立的进程中，使用 world_size=1 表示单 GPU 模式（无张量并行）。
        输出详细的模型加载进度日志。
        
        Requirements:
            - 1.4: 在指定 GPU 上加载模型
            - 3.4: 使用现有的模型推理逻辑处理请求
            - 8.1: 输出详细的启动信息（模型加载进度）
            - 8.3: 输出模型大小、加载时间等信息
        
        Note:
            - 数据并行模式下，每个 Worker 独立运行，world_size=1
            - 使用 start_model_process() 创建 ModelRpcClient
            - 单 GPU 模式下不使用 RPC，直接创建 ModelRpcServer 实例
            - 初始化后调用 init_model() 加载模型权重
        """
        import time
        
        try:
            print(f"[Worker {self.worker_id}] ========== Model Loading Started ==========")
            start_time = time.time()
            
            print(f"[Worker {self.worker_id}] Step 1/3: Creating Model RPC client...")
            
            # 创建 ModelRpcClient（world_size=1 表示单 GPU 模式）
            # 在单 GPU 模式下，start_model_process 会直接返回一个本地的 ModelRpcServer
            # 不会启动额外的 RPC 进程
            self.model_rpc = await start_model_process(
                port=None,  # 单 GPU 模式不需要端口
                world_size=1  # 数据并行模式，每个 Worker 独立
            )
            
            rpc_time = time.time() - start_time
            print(f"[Worker {self.worker_id}] Model RPC client created (took {rpc_time:.2f}s)")
            
            print(f"[Worker {self.worker_id}] Step 2/3: Preparing model initialization parameters...")
            
            # 创建 InputParams 对象（从 args 中提取参数）
            # 这是 init_model 所需的参数格式
            input_params = InputParams(
                max_req_total_len=getattr(self.args, 'max_req_total_len', 2048),
                max_total_token_num=self.args.max_total_token_num,
                pool_size_lora=getattr(self.args, 'pool_size_lora', 0),
                batch_max_tokens=self.args.batch_max_tokens,
                running_max_req_size=self.args.running_max_req_size,
                swap=getattr(self.args, 'swap', False),
                prefetch=getattr(self.args, 'prefetch', False),
                prefetch_size=getattr(self.args, 'prefetch_size', 0),
                scheduler=getattr(self.args, 'scheduler', 'slora'),
                profile=getattr(self.args, 'profile', False),
                batch_num_adapters=getattr(self.args, 'batch_num_adapters', None),
                enable_abort=getattr(self.args, 'enable_abort', False),
                dummy=getattr(self.args, 'dummy', False),
                no_lora_compute=getattr(self.args, 'no_lora_compute', False),
                no_lora_swap=getattr(self.args, 'no_lora_swap', False),
                no_kernel=getattr(self.args, 'no_kernel', False),
                no_mem_pool=getattr(self.args, 'no_mem_pool', False),
                bmm=getattr(self.args, 'bmm', False),
                no_lora=getattr(self.args, 'no_lora', False),
                fair_weights=getattr(self.args, 'fair_weights', None),
                evict_interval_threshold=getattr(self.args, 'evict_interval_threshold', 0.9),
                evict_interval_ratio=getattr(self.args, 'evict_interval_ratio', 0.5),
                evict_idle_threshold=getattr(self.args, 'evict_idle_threshold', 0.8),
                evict_idle_ratio=getattr(self.args, 'evict_idle_ratio', 0.7),
                max_lora_ratio=getattr(self.args, 'max_lora_ratio', 0.5),
            )
            
            print(f"[Worker {self.worker_id}] Model configuration:")
            print(f"[Worker {self.worker_id}]   Model directory: {self.args.model_dir}")
            print(f"[Worker {self.worker_id}]   Max total tokens: {self.args.max_total_token_num}")
            print(f"[Worker {self.worker_id}]   Batch max tokens: {self.args.batch_max_tokens}")
            print(f"[Worker {self.worker_id}]   Running max requests: {self.args.running_max_req_size}")
            print(f"[Worker {self.worker_id}]   Dummy mode: {getattr(self.args, 'dummy', False)}")
            print(f"[Worker {self.worker_id}]   LoRA enabled: {not getattr(self.args, 'no_lora', False)}")
            
            if hasattr(self.args, 'lora_dirs') and self.args.lora_dirs:
                print(f"[Worker {self.worker_id}]   Adapter directories: {len(self.args.lora_dirs)} adapters")
            
            print(f"[Worker {self.worker_id}] Step 3/3: Loading model weights...")
            model_load_start = time.time()
            
            # 初始化模型（加载权重）
            # 传递必要的参数给 RPC 服务器
            await self.model_rpc.init_model(
                rank_id=0,  # 单 GPU 模式，rank 始终为 0
                world_size=1,  # 单 GPU 模式
                weight_dir=self.args.model_dir,
                adapter_dirs=getattr(self.args, 'lora_dirs', []),
                max_total_token_num=self.args.max_total_token_num,
                load_way=getattr(self.args, 'load_way', 'HF'),
                mode=getattr(self.args, 'mode', []),
                input_params=input_params,  # 传递 InputParams 对象
                prefetch_stream=None  # 数据并行模式不使用 prefetch
            )
            
            model_load_time = time.time() - model_load_start
            total_time = time.time() - start_time
            
            # 获取模型大小信息（如果可用）
            try:
                # 尝试获取模型参数数量
                import os
                model_dir = self.args.model_dir
                # 估算模型大小（基于目录大小）
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(model_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.isfile(filepath):
                            total_size += os.path.getsize(filepath)
                
                model_size_gb = total_size / (1024 ** 3)
                print(f"[Worker {self.worker_id}] Model size: {model_size_gb:.2f} GB")
            except Exception as e:
                # 如果获取失败，不影响主流程
                pass
            
            print(f"[Worker {self.worker_id}] Model weights loaded (took {model_load_time:.2f}s)")
            print(f"[Worker {self.worker_id}] ========== Model Loading Complete ==========")
            print(f"[Worker {self.worker_id}] Total initialization time: {total_time:.2f}s")
            print(f"[Worker {self.worker_id}] Model RPC initialized successfully on GPU {self.gpu_id}")
            
        except Exception as e:
            print(f"[Worker {self.worker_id}] ========== Model Loading Failed ==========")
            print(f"[Worker {self.worker_id}] Failed to initialize model RPC: {str(e)}")
            import traceback
            traceback.print_exc()
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
        
        # DEBUG: 打印请求的 max_output_len
        print(f"[Worker {self.worker_id}] New request: id={req.request_id[:8]}..., "
              f"max_output_len={req.max_output_len}, ignore_eos={req.sample_params.ignore_eos}")
        
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
    
    async def _infer_batch(self, batch: Batch) -> dict:
        """
        执行批次推理
        
        调用 model_rpc 执行实际推理，处理推理结果并更新批次状态。
        包含完善的异常处理和错误分类。
        
        Args:
            batch: 要推理的批次
        
        Returns:
            dict: 请求ID到输出token的映射 {request_id: (token_id, metadata)}
        
        Requirements:
            - 3.4: 使用现有的模型推理逻辑处理请求
            - 3.5: 生成包含 output_ids 和 metadata 的响应消息
            - 7.3: 推理异常时记录详细错误日志
        
        Note:
            - 对于新批次（prefill），调用 init_batch + prefill_batch
            - 对于已有批次（decode），调用 decode_batch
            - 返回格式与 manager.py 保持一致
        
        Raises:
            RuntimeError: 推理失败时抛出，包含错误类型和详细信息
        """
        try:
            # 判断是 prefill 还是 decode
            # 如果批次中的请求还没有输出 token，则是 prefill
            is_prefill = all(len(req.output_ids) == 0 for req in batch.reqs)
            
            if is_prefill:
                # Prefill 阶段：初始化批次并执行 prefill
                # 1. 初始化批次（将请求信息传递给 RPC）
                reqs_rpc = [req.to_rpc_obj() for req in batch.reqs]
                await self.model_rpc.init_batch(batch.batch_id, reqs_rpc)
                
                # 2. 执行 prefill（处理 prompt）
                req_to_out_token_id = await self.model_rpc.prefill_batch(batch.batch_id)
            else:
                # Decode 阶段：生成下一个 token
                # 更新适配器使用统计信息（在推理前）
                if not getattr(self.args, 'no_lora', False):
                    adapter_dirs_list = list(batch.adapter_dirs)
                    await self.model_rpc.update_adapter_stats(adapter_dirs_list)
                
                # 执行 decode
                req_to_out_token_id = await self.model_rpc.decode_batch(batch.batch_id)
            
            return req_to_out_token_id
            
        except RuntimeError as e:
            # CUDA OOM 或其他运行时错误
            error_msg = str(e).lower()
            if 'out of memory' in error_msg or 'oom' in error_msg:
                error_type = "CUDA_OOM"
                print(f"[Worker {self.worker_id}] CUDA Out of Memory error in _infer_batch:")
                print(f"[Worker {self.worker_id}]   Batch ID: {batch.batch_id}")
                print(f"[Worker {self.worker_id}]   Batch size: {len(batch.reqs)} requests")
                print(f"[Worker {self.worker_id}]   Mode: {'prefill' if is_prefill else 'decode'}")
                print(f"[Worker {self.worker_id}]   Error: {str(e)}")
            else:
                error_type = "RUNTIME_ERROR"
                print(f"[Worker {self.worker_id}] Runtime error in _infer_batch:")
                print(f"[Worker {self.worker_id}]   Error type: {type(e).__name__}")
                print(f"[Worker {self.worker_id}]   Error: {str(e)}")
            
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"[{error_type}] Inference failed: {str(e)}")
            
        except asyncio.TimeoutError as e:
            # RPC 超时
            error_type = "RPC_TIMEOUT"
            print(f"[Worker {self.worker_id}] RPC timeout in _infer_batch:")
            print(f"[Worker {self.worker_id}]   Batch ID: {batch.batch_id}")
            print(f"[Worker {self.worker_id}]   Batch size: {len(batch.reqs)} requests")
            print(f"[Worker {self.worker_id}]   Mode: {'prefill' if is_prefill else 'decode'}")
            print(f"[Worker {self.worker_id}]   Error: {str(e)}")
            
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"[{error_type}] Inference timeout: {str(e)}")
            
        except Exception as e:
            # 其他未知错误
            error_type = "UNKNOWN_ERROR"
            print(f"[Worker {self.worker_id}] Unexpected error in _infer_batch:")
            print(f"[Worker {self.worker_id}]   Error type: {type(e).__name__}")
            print(f"[Worker {self.worker_id}]   Batch ID: {batch.batch_id}")
            print(f"[Worker {self.worker_id}]   Batch size: {len(batch.reqs)} requests")
            print(f"[Worker {self.worker_id}]   Mode: {'prefill' if is_prefill else 'decode'}")
            print(f"[Worker {self.worker_id}]   Error: {str(e)}")
            
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"[{error_type}] Inference failed: {str(e)}")
    
    async def _process_requests(self) -> List[dict]:
        """
        处理请求批次（使用 ReqQueue 管理）
        
        使用 ReqQueue 生成新批次，合并到当前批次，执行推理，并生成响应。
        包含完善的异常处理，确保错误时返回正确的错误响应。
        
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
            - 7.3: 推理异常时返回错误响应
        
        Note:
            Task 2.9.2 实现：
            - 使用 ReqQueue 管理批处理
            - 调用 model_rpc 执行实际推理
            - 处理 prefill 和 decode 两种模式
            - 推理失败时生成错误响应并清理批次
        """
        try:
            # 使用 ReqQueue 生成新批次
            new_batch = self.req_queue.generate_new_batch(
                self.current_batch,
                self.lora_ranks,
                actual_adapter_size=self.actual_adapter_memory_usage
            )
            
            if new_batch is not None:
                # 加载批次所需的 adapters
                if not getattr(self.args, 'no_lora', False) and new_batch.adapter_dirs:
                    await self._load_adapters(new_batch.adapter_dirs)
                
                # 先对新批次执行 prefill
                reqs_rpc = [req.to_rpc_obj() for req in new_batch.reqs]
                await self.model_rpc.init_batch(new_batch.batch_id, reqs_rpc)
                
                # 执行 prefill，获取第一个 token
                req_to_out_token_id = await self.model_rpc.prefill_batch(new_batch.batch_id)
                
                # 将第一个 token 添加到新批次的请求中
                for req_id, (new_token_id, new_gen_metadata) in req_to_out_token_id.items():
                    req = new_batch.id_to_reqs[req_id]
                    req.output_ids.append(new_token_id)
                    req.output_metadata_list.append(new_gen_metadata)
                
                # 标记新批次中已完成的请求
                eos_id = getattr(self.args, 'eos_id', 2)
                has_new_finished = new_batch.mark_finished_req(eos_id)
                
                # 处理新批次中已完成的请求
                if has_new_finished:
                    await self._handle_finish_req(new_batch, has_new_finished)
                
                # 合并到当前批次
                if self.current_batch is None:
                    self.current_batch = new_batch
                else:
                    if not new_batch.is_clear():
                        await self.model_rpc.merge_batch(self.current_batch.batch_id, new_batch.batch_id)
                        self.current_batch.merge(new_batch)
            
            # 执行推理
            if self.current_batch is not None and len(self.current_batch.reqs) > 0:
                try:
                    # 调用 model_rpc 执行实际推理
                    req_to_out_token_id = await self._infer_batch(self.current_batch)
                    
                    # 获取 eos_id
                    eos_id = getattr(self.args, 'eos_id', 2)
                    
                    # 将输出 token 添加到请求中
                    for req_id, (new_token_id, new_gen_metadata) in req_to_out_token_id.items():
                        req = self.current_batch.id_to_reqs[req_id]
                        req.output_ids.append(new_token_id)
                        req.output_metadata_list.append(new_gen_metadata)
                    
                    # 标记已完成的请求
                    has_new_finished_req = self.current_batch.mark_finished_req(eos_id)
                    
                    # 打印完成的请求
                    if has_new_finished_req:
                        for req in self.current_batch.reqs:
                            if req.has_generate_finished:
                                print(f"[Worker {self.worker_id}] FINISHED: {req.request_id[:8]}... (output_len={len(req.output_ids)})")
                    
                    # 生成响应
                    responses = []
                    for req_id, (new_token_id, new_gen_metadata) in req_to_out_token_id.items():
                        req = self.current_batch.id_to_reqs[req_id]
                        
                        # 判断请求是否完成（与 mark_finished_req 逻辑一致）
                        is_finished = req.has_generate_finished
                        
                        # 生成响应（包含完整的输出序列）
                        output_ids = req.prompt_ids + req.output_ids
                        metadata = {
                            'finish_reason': 'stop' if is_finished else 'generating',
                            'prompt_tokens': req.input_len,
                            'completion_tokens': len(req.output_ids),
                            'gen_metadata': new_gen_metadata
                        }
                        
                        responses.append({
                            'request_id': req.request_id,
                            'worker_id': self.worker_id,
                            'output_ids': output_ids,
                            'metadata': metadata,
                            'success': True,
                            'error': None,
                            'finished': is_finished  # 使用 mark_finished_req 设置的状态
                        })
                    
                    # 处理已完成的请求（从批次中移除）
                    await self._handle_finish_req(self.current_batch, has_new_finished_req)
                    
                    return responses
                    
                except RuntimeError as e:
                    # 推理失败，生成错误响应
                    error_msg = str(e)
                    print(f"[Worker {self.worker_id}] Inference failed, generating error responses")
                    
                    # 为批次中的所有请求生成错误响应
                    error_responses = []
                    for req in self.current_batch.reqs:
                        error_responses.append({
                            'request_id': req.request_id,
                            'worker_id': self.worker_id,
                            'output_ids': req.prompt_ids,  # 只返回 prompt
                            'metadata': {
                                'finish_reason': 'error',
                                'prompt_tokens': req.input_len,
                                'completion_tokens': 0,
                                'error_type': error_msg.split(']')[0].strip('[') if '[' in error_msg else 'UNKNOWN'
                            },
                            'success': False,
                            'error': error_msg
                        })
                    
                    # 清理失败的批次
                    if self.model_rpc:
                        try:
                            await self.model_rpc.remove_batch(self.current_batch.batch_id)
                        except Exception as cleanup_error:
                            print(f"[Worker {self.worker_id}] Error cleaning up batch: {cleanup_error}")
                    
                    self.current_batch = None
                    
                    return error_responses
            
            return []
            
        except Exception as e:
            # 捕获其他异常（如 ReqQueue 错误、adapter 加载错误等）
            print(f"[Worker {self.worker_id}] Error processing requests:")
            print(f"[Worker {self.worker_id}]   Error type: {type(e).__name__}")
            print(f"[Worker {self.worker_id}]   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 如果有当前批次，尝试清理
            if self.current_batch is not None:
                try:
                    if self.model_rpc:
                        await self.model_rpc.remove_batch(self.current_batch.batch_id)
                except Exception as cleanup_error:
                    print(f"[Worker {self.worker_id}] Error cleaning up batch: {cleanup_error}")
                
                self.current_batch = None
            
            return []
    
    async def _handle_finish_req(self, batch: Batch, has_new_finished_req: bool) -> None:
        """
        处理批次中已完成请求的逻辑
        
        当批次中有请求完成时：
        1. 记录完成的请求使用的适配器
        2. 过滤掉已完成的请求，更新批次状态
        3. 减少完成请求的适配器的当前请求计数
        4. 如果批次完全清空，则移除批次；否则过滤批次
        
        Args:
            batch: 当前运行的批次
            has_new_finished_req: 是否有新完成的请求
        
        Requirements:
            - 3.5: 生成包含 output_ids 和 metadata 的响应消息
        
        Note:
            借鉴 manager.py 的 _handle_finish_req 方法，简化为数据并行模式。
            Phase 1 不实现复杂的淘汰策略，只做基本的请求计数管理。
        """
        if has_new_finished_req:
            # 记录完成的请求使用的适配器（在 filter_finished 之前）
            finished_adapter_dirs = []
            if not getattr(self.args, 'no_lora', False):
                for req in batch.reqs:
                    if req.has_generate_finished:
                        finished_adapter_dirs.append(req.adapter_dir)
            
            # 过滤掉已完成的请求，只保留未完成的请求
            # 同时会更新 batch.adapter_dirs，只包含未完成请求使用的适配器
            batch.filter_finished()

            # 减少完成请求的适配器的当前请求计数
            # 这对于 adapter 使用统计和淘汰策略很重要
            if finished_adapter_dirs and not getattr(self.args, 'no_lora', False):
                await self.model_rpc.decrease_request_counts(finished_adapter_dirs)

            # 根据批次状态决定后续操作
            if batch.is_clear():
                # 批次完全清空，移除 RPC 端的批次
                await self.model_rpc.remove_batch(batch.batch_id)
                self.current_batch = None
            else:
                # 批次还有未完成的请求，过滤 RPC 端的批次
                req_id_list = [req.request_id for req in batch.reqs]
                await self.model_rpc.filter_batch(batch.batch_id, req_id_list)
    
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
    
    async def _receive_requests_loop(self) -> None:
        """
        请求接收协程 - 持续接收 ZMQ 消息并存入本地队列
        
        这个协程独立运行，专门负责从 ZMQ socket 接收请求并存入本地队列。
        这样可以：
        1. 避免 ZMQ 内部缓存导致的消息丢失问题
        2. 实现更可控的队列管理（为未来的多级队列、优先级调度做准备）
        3. 提供更好的可观测性（可以随时查看队列状态）
        
        Note:
            - 使用阻塞式 recv_json()，不会丢失消息
            - 接收到的请求立即转换为 Req 对象并存入 req_queue
            - 这个协程会一直运行，直到 Worker 进程终止
        """
        import sys
        
        print(f"[Worker {self.worker_id}] Request receiver loop started")
        sys.stdout.flush()
        
        while True:
            try:
                # 阻塞式接收请求（不会丢失消息）
                # ZMQ RCVTIMEO 设置为 30s，超时后会抛出 zmq.Again
                request = await self.request_receiver.recv_json()
                
                # 更新接收计数
                self._received_count += 1
                request_id = request.get('request_id', 'unknown')
                
                print(f"[Worker {self.worker_id}] Received #{self._received_count}: {request_id[:8]}...")
                sys.stdout.flush()
                
                # 转换为 Req 对象并存入队列
                req_obj = self._convert_to_req_object(request)
                self.req_queue.append(req_obj)
                
                # 未来扩展点：这里可以实现优先级判断，将请求存入不同的队列
                # 例如：
                # priority = self._calculate_priority(request)
                # self.multi_level_queue.enqueue(req_obj, priority)
                
            except zmq.Again:
                # ZMQ 超时（RCVTIMEO），继续等待
                # 这是正常的，不需要打印日志
                continue
            except asyncio.CancelledError:
                # 协程被取消，正常退出
                print(f"[Worker {self.worker_id}] Request receiver loop cancelled")
                break
            except Exception as e:
                # 其他错误，记录日志但继续运行
                print(f"[Worker {self.worker_id}] Recv error: {e}")
                import traceback
                traceback.print_exc()
                # 短暂等待后继续
                await asyncio.sleep(0.1)
    
    async def _process_loop(self) -> None:
        """
        请求处理协程 - 从本地队列取出请求并处理
        
        这个协程独立运行，负责：
        1. 从本地队列生成批次
        2. 执行推理
        3. 发送响应
        
        Note:
            - 与接收协程分离，实现接收和处理的并行
            - 当队列为空时，短暂等待后重试
        """
        import sys
        import time
        
        print(f"[Worker {self.worker_id}] Process loop started")
        sys.stdout.flush()
        
        last_heartbeat_time = time.time()
        
        while True:
            try:
                # 每 10 秒打印一次心跳
                now = time.time()
                if now - last_heartbeat_time >= 10:
                    queue_size = len(self.req_queue.waiting_req_list) if self.req_queue else 0
                    batch_size = len(self.current_batch.reqs) if self.current_batch else 0
                    print(f"[Worker {self.worker_id}] Heartbeat: received={self._received_count}, queue={queue_size}, batch={batch_size}")
                    sys.stdout.flush()
                    last_heartbeat_time = now
                
                # 处理请求批次
                responses = await self._process_requests()
                
                # 发送响应
                for response in responses:
                    try:
                        await self._send_response(response)
                    except Exception as send_error:
                        print(f"[Worker {self.worker_id}] Send error: {send_error}")
                
                # 如果没有待处理的请求，短暂等待
                if not self.req_queue.waiting_req_list and self.current_batch is None:
                    await asyncio.sleep(0.01)  # 10ms
                    
            except asyncio.CancelledError:
                # 协程被取消，正常退出
                print(f"[Worker {self.worker_id}] Process loop cancelled")
                break
            except Exception as e:
                print(f"[Worker {self.worker_id}] Process error: {e}")
                import traceback
                traceback.print_exc()
                # 短暂等待后继续
                await asyncio.sleep(0.1)
    
    async def run(self) -> None:
        """
        主循环 - 启动接收和处理两个独立的协程
        
        设计说明：
        - 接收协程 (_receive_requests_loop): 持续从 ZMQ 接收请求，存入本地队列
        - 处理协程 (_process_loop): 从本地队列取出请求，执行推理，发送响应
        
        这种设计的优点：
        1. 接收和处理并行，不会因为处理慢而丢失消息
        2. 本地队列可控，为未来的多级队列、优先级调度做准备
        3. 更好的可观测性和调试能力
        """
        import sys
        print(f"[Worker {self.worker_id}] Ready on GPU {self.gpu_id}")
        sys.stdout.flush()
        
        # 初始化接收计数器
        self._received_count = 0
        
        # 启动接收协程和处理协程
        receive_task = asyncio.create_task(self._receive_requests_loop())
        process_task = asyncio.create_task(self._process_loop())
        
        print(f"[Worker {self.worker_id}] Started receive and process loops")
        sys.stdout.flush()
        
        try:
            # 等待两个协程（正常情况下它们会一直运行）
            await asyncio.gather(receive_task, process_task)
        except asyncio.CancelledError:
            # 主循环被取消，取消子协程
            receive_task.cancel()
            process_task.cancel()
            try:
                await asyncio.gather(receive_task, process_task, return_exceptions=True)
            except:
                pass
            print(f"[Worker {self.worker_id}] Main loop cancelled")
        except Exception as e:
            print(f"[Worker {self.worker_id}] Main loop error: {e}")
            import traceback
            traceback.print_exc()
