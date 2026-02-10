"""
Worker State Reporter for Adapter-Aware Routing

This module implements the Worker-side state reporter that periodically
sends state updates to the Router for intelligent routing decisions.

Requirements: 1.1, 1.2, 1.3
"""

import asyncio
import time
import logging
from typing import Optional, Set, Callable, Any

try:
    import zmq
    import zmq.asyncio
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

from .worker_state import WorkerState


logger = logging.getLogger(__name__)


class WorkerStateReporter:
    """
    Worker 状态上报器
    
    负责定期向 Router 上报 Worker 状态，包括：
    - 已缓存的 Adapter 列表
    - 当前队列长度
    - 可用 GPU 显存
    
    支持两种上报模式：
    1. 定期心跳上报（默认 100ms 间隔）
    2. 事件触发上报（Adapter 加载/卸载时立即上报）
    
    Attributes:
        worker_id: Worker 唯一标识
        report_interval_ms: 上报间隔（毫秒）
        router_address: Router 状态接收地址
        _running: 是否正在运行
        _task: 定期上报任务
        _socket: ZMQ PUSH socket
        _context: ZMQ context
        _state_getter: 获取当前状态的回调函数
    
    Requirements: 1.1, 1.2, 1.3
    """
    
    def __init__(self,
                 worker_id: int,
                 report_interval_ms: int = 100,
                 router_address: Optional[str] = None,
                 state_getter: Optional[Callable[[], dict]] = None):
        """
        初始化状态上报器
        
        Args:
            worker_id: Worker 唯一标识
            report_interval_ms: 上报间隔（毫秒），默认 100ms
            router_address: Router 状态接收地址，格式为 "tcp://host:port"
            state_getter: 获取当前状态的回调函数，返回包含以下字段的字典：
                - cached_adapters: Set[str] 或 List[str]
                - queue_length: int
                - gpu_memory_free: int (可选)
        
        Requirements: 1.1, 1.2
        """
        if worker_id < 0:
            raise ValueError(f"worker_id must be non-negative, got {worker_id}")
        if report_interval_ms <= 0:
            raise ValueError(f"report_interval_ms must be positive, got {report_interval_ms}")
        
        self.worker_id = worker_id
        self.report_interval_ms = report_interval_ms
        self.router_address = router_address
        self._state_getter = state_getter
        
        # 运行状态
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # ZMQ 通信
        self._context: Optional[Any] = None
        self._socket: Optional[Any] = None
        
        # 统计信息
        self._report_count = 0
        self._last_report_time = 0.0
        self._error_count = 0
        
        logger.info(f"WorkerStateReporter initialized: worker_id={worker_id}, "
                   f"interval={report_interval_ms}ms, address={router_address}")
    
    def set_state_getter(self, getter: Callable[[], dict]) -> None:
        """
        设置状态获取回调函数
        
        Args:
            getter: 获取当前状态的回调函数
        """
        self._state_getter = getter
    
    def _setup_zmq(self) -> bool:
        """
        设置 ZMQ 通信
        
        Returns:
            是否成功设置
        """
        if not HAS_ZMQ:
            logger.warning("ZMQ not available, state reporting disabled")
            return False
        
        if not self.router_address:
            logger.warning("No router address configured, state reporting disabled")
            return False
        
        try:
            self._context = zmq.asyncio.Context()
            self._socket = self._context.socket(zmq.PUSH)
            
            # 设置 socket 选项
            self._socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1秒发送超时
            self._socket.setsockopt(zmq.LINGER, 0)  # 关闭时立即丢弃
            self._socket.setsockopt(zmq.SNDHWM, 100)  # 发送高水位
            
            self._socket.connect(self.router_address)
            
            logger.info(f"Worker {self.worker_id} connected to router at {self.router_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup ZMQ: {e}")
            self._cleanup_zmq()
            return False
    
    def _cleanup_zmq(self) -> None:
        """清理 ZMQ 资源"""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
        
        if self._context:
            try:
                self._context.term()
            except:
                pass
            self._context = None
    
    def get_current_state(self) -> WorkerState:
        """
        获取当前 Worker 状态
        
        Returns:
            当前 Worker 状态
        """
        if self._state_getter:
            try:
                state_dict = self._state_getter()
                cached_adapters = state_dict.get('cached_adapters', set())
                if isinstance(cached_adapters, list):
                    cached_adapters = set(cached_adapters)
                
                return WorkerState(
                    worker_id=self.worker_id,
                    cached_adapters=cached_adapters,
                    queue_length=state_dict.get('queue_length', 0),
                    gpu_memory_free=state_dict.get('gpu_memory_free', 0),
                    last_heartbeat=time.time(),
                    is_healthy=True
                )
            except Exception as e:
                logger.error(f"Error getting state: {e}")
        
        # 返回默认状态
        return WorkerState(
            worker_id=self.worker_id,
            cached_adapters=set(),
            queue_length=0,
            gpu_memory_free=0,
            last_heartbeat=time.time(),
            is_healthy=True
        )
    
    def _create_state_message(self, state: WorkerState) -> dict:
        """
        创建状态上报消息
        
        Args:
            state: Worker 状态
            
        Returns:
            状态消息字典
        """
        return {
            'type': 'worker_state',
            'worker_id': state.worker_id,
            'cached_adapters': list(state.cached_adapters),
            'queue_length': state.queue_length,
            'gpu_memory_free': state.gpu_memory_free,
            'timestamp': time.time()
        }
    
    async def _send_state(self, state: WorkerState) -> bool:
        """
        发送状态到 Router
        
        Args:
            state: Worker 状态
            
        Returns:
            是否发送成功
        """
        if not self._socket:
            return False
        
        try:
            message = self._create_state_message(state)
            await self._socket.send_json(message)
            
            self._report_count += 1
            self._last_report_time = time.time()
            
            logger.debug(f"Worker {self.worker_id} sent state: "
                        f"queue={state.queue_length}, "
                        f"adapters={len(state.cached_adapters)}")
            return True
            
        except zmq.Again:
            # 发送超时
            self._error_count += 1
            logger.warning(f"Worker {self.worker_id} state send timeout")
            return False
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Worker {self.worker_id} state send error: {e}")
            return False
    
    async def _report_loop(self) -> None:
        """
        定期上报循环
        
        Requirements: 1.3
        """
        interval_sec = self.report_interval_ms / 1000.0
        
        logger.info(f"Worker {self.worker_id} starting report loop "
                   f"(interval={self.report_interval_ms}ms)")
        
        while self._running:
            try:
                # 获取当前状态
                state = self.get_current_state()
                
                # 发送状态
                await self._send_state(state)
                
                # 等待下一个间隔
                await asyncio.sleep(interval_sec)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {self.worker_id} report loop cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} report loop error: {e}")
                await asyncio.sleep(interval_sec)
        
        logger.info(f"Worker {self.worker_id} report loop stopped")
    
    async def start(self) -> bool:
        """
        启动定期上报任务
        
        Returns:
            是否成功启动
        
        Requirements: 1.3
        """
        if self._running:
            logger.warning(f"Worker {self.worker_id} reporter already running")
            return True
        
        # 设置 ZMQ
        if not self._setup_zmq():
            logger.warning(f"Worker {self.worker_id} reporter started without ZMQ")
            # 即使没有 ZMQ，也可以启动（用于测试）
        
        self._running = True
        self._task = asyncio.create_task(self._report_loop())
        
        logger.info(f"Worker {self.worker_id} state reporter started")
        return True
    
    async def stop(self) -> None:
        """
        停止上报任务
        """
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        self._cleanup_zmq()
        
        logger.info(f"Worker {self.worker_id} state reporter stopped "
                   f"(total reports: {self._report_count}, errors: {self._error_count})")
    
    async def report_now(self) -> bool:
        """
        立即上报当前状态（用于 Adapter 加载/卸载事件）
        
        Returns:
            是否发送成功
        
        Requirements: 1.3
        """
        state = self.get_current_state()
        
        if self._socket:
            return await self._send_state(state)
        else:
            logger.debug(f"Worker {self.worker_id} report_now called but no socket")
            return False
    
    def report_now_sync(self) -> None:
        """
        同步版本的立即上报（用于非异步上下文）
        
        Note:
            这个方法会创建一个新的事件循环来执行异步上报。
            如果已经在异步上下文中，应该使用 report_now() 方法。
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建一个任务
                asyncio.create_task(self.report_now())
            else:
                # 否则直接运行
                loop.run_until_complete(self.report_now())
        except RuntimeError:
            # 没有事件循环，创建一个新的
            asyncio.run(self.report_now())
    
    def get_stats(self) -> dict:
        """
        获取上报统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'worker_id': self.worker_id,
            'report_count': self._report_count,
            'error_count': self._error_count,
            'last_report_time': self._last_report_time,
            'running': self._running,
            'has_socket': self._socket is not None
        }
    
    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running
    
    @property
    def report_count(self) -> int:
        """上报次数"""
        return self._report_count
