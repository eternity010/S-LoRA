# Design Document - Phase 1: 数据并行基础框架

## Overview

Phase 1 的设计目标是实现 S-LoRA 的数据并行基础架构，使多个 GPU 能够独立处理请求。本设计采用多进程架构，每个 GPU Worker 运行在独立的进程中，通过 ZeroMQ 进行进程间通信。核心设计原则是简单、可靠、易于扩展。

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Server (FastAPI)                  │
│                  (slora/server/api_server.py)            │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP Request
                         ▼
┌─────────────────────────────────────────────────────────┐
│         Data Parallel Router Manager                     │
│           (slora/server/router/dp_manager.py)            │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Round Robin Router                               │   │
│  │  - counter: int                                   │   │
│  │  - select_worker() -> int                         │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Worker Manager                                   │   │
│  │  - workers: List[Process]                         │   │
│  │  - start_workers()                                │   │
│  └──────────────────────────────────────────────────┘   │
└───┬─────────────┬─────────────┬─────────────────────────┘
    │ ZMQ PUSH    │ ZMQ PUSH    │ ZMQ PUSH
    ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│GPU Worker 0 │ │GPU Worker 1 │ │GPU Worker 2 │
│  (Process)  │ │  (Process)  │ │  (Process)  │
├─────────────┤ ├─────────────┤ ├─────────────┤
│ Base Model  │ │ Base Model  │ │ Base Model  │
│ (Llama-7B)  │ │ (Llama-7B)  │ │ (Llama-7B)  │
├─────────────┤ ├─────────────┤ ├─────────────┤
│Request Queue│ │Request Queue│ │Request Queue│
│  (Async)    │ │  (Async)    │ │  (Async)    │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │ ZMQ PUSH      │ ZMQ PUSH      │ ZMQ PUSH
       └───────────────┴───────────────┘
                       │
                       ▼
              ┌────────────────┐
              │Response Merger │
              │   (Async)      │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ Detokenization │
              │   (Existing)   │
              └────────────────┘
```

### Process Model

系统采用多进程架构，主要包含以下进程：

1. **Main Process**: API Server，处理 HTTP 请求
2. **Router Manager Process**: 管理 Worker 并路由请求
3. **GPU Worker Processes**: 每个 GPU 一个进程，独立处理请求
4. **Detokenization Process**: Token 解码（复用现有）

### Communication Protocol

使用 ZeroMQ 进行进程间通信：

**Router → Workers**:
- Pattern: PUSH/PULL
- Router 使用 PUSH socket 发送请求
- 每个 Worker 使用 PULL socket 接收请求

**Workers → Response Merger**:
- Pattern: PUSH/PULL
- 每个 Worker 使用 PUSH socket 发送响应
- Response Merger 使用 PULL socket 接收响应

## Components and Interfaces

### Component 1: GPU Worker

**File**: `slora/server/router/gpu_worker.py`

**Responsibility**: 在单张 GPU 上运行独立的推理实例

**Class Design**:

```python
class GPUWorker:
    """GPU Worker 进程，在指定 GPU 上处理请求"""
    
    def __init__(self, worker_id: int, gpu_id: int, args: argparse.Namespace):
        """
        初始化 Worker
        
        Args:
            worker_id: Worker 的唯一标识符
            gpu_id: 分配的 GPU ID
            args: 命令行参数
        """
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.args = args
        
        # ZMQ 通信
        self.context = None
        self.request_receiver = None
        self.response_sender = None
        
        # 模型相关
        self.model = None
        self.adapter_cache = {}
    
    def _setup_gpu(self) -> None:
        """设置 GPU 环境"""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.cuda.set_device(0)
    
    def _setup_zmq(self, request_port: int, response_port: int) -> None:
        """设置 ZMQ 通信"""
        self.context = zmq.asyncio.Context()
        
        # 接收请求
        self.request_receiver = self.context.socket(zmq.PULL)
        self.request_receiver.connect(f"tcp://127.0.0.1:{request_port}")
        
        # 发送响应
        self.response_sender = self.context.socket(zmq.PUSH)
        self.response_sender.connect(f"tcp://127.0.0.1:{response_port}")
    
    def _load_model(self) -> None:
        """加载基座模型（复用现有代码）"""
        # 复用 slora/common/basemodel/ 中的模型加载逻辑
        pass
    
    async def _receive_request(self) -> dict:
        """接收请求"""
        request_json = await self.request_receiver.recv_json()
        return request_json
    
    async def _process_request(self, request: dict) -> dict:
        """
        处理请求
        
        Args:
            request: 请求消息，包含 request_id, adapter_dir, prompt_ids, sampling_params
        
        Returns:
            响应消息，包含 request_id, worker_id, output_ids, metadata, success
        """
        try:
            # 加载 adapter（如果需要）
            adapter_dir = request.get('adapter_dir')
            if adapter_dir and adapter_dir not in self.adapter_cache:
                await self._load_adapter(adapter_dir)
            
            # 执行推理（复用现有逻辑）
            output_ids, metadata = await self._infer(request)
            
            return {
                'request_id': request['request_id'],
                'worker_id': self.worker_id,
                'output_ids': output_ids,
                'metadata': metadata,
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'request_id': request['request_id'],
                'worker_id': self.worker_id,
                'output_ids': [],
                'metadata': {},
                'success': False,
                'error': str(e)
            }
    
    async def _send_response(self, response: dict) -> None:
        """发送响应"""
        await self.response_sender.send_json(response)
    
    async def run(self) -> None:
        """主循环"""
        print(f"Worker {self.worker_id} ready on GPU {self.gpu_id}")
        
        while True:
            request = await self._receive_request()
            response = await self._process_request(request)
            await self._send_response(response)
```

**Key Methods**:
- `_setup_gpu()`: 设置 CUDA_VISIBLE_DEVICES
- `_load_model()`: 加载基座模型
- `_process_request()`: 处理单个请求
- `run()`: 主循环，持续接收和处理请求

### Component 2: Data Parallel Router Manager

**File**: `slora/server/router/dp_manager.py`

**Responsibility**: 管理多个 GPU Worker 并路由请求

**Class Design**:

```python
class DataParallelRouterManager:
    """数据并行路由管理器"""
    
    def __init__(self, args: argparse.Namespace, 
                 router_port: int, response_port: int):
        """
        初始化 Router Manager
        
        Args:
            args: 命令行参数
            router_port: 接收 HTTP 请求的端口
            response_port: 接收 Worker 响应的端口
        """
        self.args = args
        self.router_port = router_port
        self.response_port = response_port
        
        # Worker 管理
        self.num_workers = args.num_workers or self._detect_gpus()
        self.gpu_ids = self._parse_gpu_ids(args.gpu_ids)
        self.workers = []
        self.worker_ports = []
        
        # 路由器
        self.router = RoundRobinRouter(self.num_workers)
        
        # ZMQ 通信
        self.context = None
        self.request_receiver = None
        self.request_senders = []
    
    def _detect_gpus(self) -> int:
        """自动检测可用 GPU 数量"""
        return torch.cuda.device_count()
    
    def _parse_gpu_ids(self, gpu_ids_str: str) -> List[int]:
        """解析 GPU ID 列表"""
        if gpu_ids_str:
            return [int(x) for x in gpu_ids_str.split(',')]
        else:
            return list(range(self.num_workers))
    
    def _allocate_ports(self) -> None:
        """为每个 Worker 分配端口"""
        base_port = 50000
        for i in range(self.num_workers):
            self.worker_ports.append(base_port + i)
    
    async def start_workers(self) -> None:
        """启动所有 Worker 进程"""
        self._allocate_ports()
        
        for i in range(self.num_workers):
            worker = self._start_worker(i, self.gpu_ids[i])
            self.workers.append(worker)
        
        # 等待所有 Worker 就绪
        await asyncio.sleep(5)
        print(f"Started {self.num_workers} workers")
    
    def _start_worker(self, worker_id: int, gpu_id: int) -> mp.Process:
        """启动单个 Worker 进程"""
        proc = mp.Process(
            target=run_gpu_worker,
            args=(worker_id, gpu_id, self.args, 
                  self.worker_ports[worker_id], self.response_port)
        )
        proc.start()
        return proc
    
    def _setup_zmq(self) -> None:
        """设置 ZMQ 通信"""
        self.context = zmq.asyncio.Context()
        
        # 接收来自 API Server 的请求
        self.request_receiver = self.context.socket(zmq.PULL)
        self.request_receiver.bind(f"tcp://127.0.0.1:{self.router_port}")
        
        # 为每个 Worker 创建 PUSH socket
        for port in self.worker_ports:
            sender = self.context.socket(zmq.PUSH)
            sender.bind(f"tcp://127.0.0.1:{port}")
            self.request_senders.append(sender)
    
    async def route_request(self, request: dict) -> None:
        """路由请求到 Worker"""
        worker_id = self.router.select_worker()
        await self.request_senders[worker_id].send_json(request)
    
    async def run(self) -> None:
        """主循环"""
        while True:
            request = await self.request_receiver.recv_json()
            await self.route_request(request)
```

**Key Methods**:
- `start_workers()`: 启动所有 Worker 进程
- `route_request()`: 路由请求到选定的 Worker
- `run()`: 主循环，持续接收和路由请求

### Component 3: Round Robin Router

**File**: `slora/server/router/round_robin_router.py`

**Responsibility**: 实现轮询路由策略

**Class Design**:

```python
class RoundRobinRouter:
    """轮询路由器"""
    
    def __init__(self, num_workers: int):
        """
        初始化路由器
        
        Args:
            num_workers: Worker 数量
        """
        self.num_workers = num_workers
        self.counter = 0
    
    def select_worker(self) -> int:
        """
        选择下一个 Worker
        
        Returns:
            Worker ID
        """
        worker_id = self.counter % self.num_workers
        self.counter += 1
        return worker_id
```

### Component 4: Response Merger

**File**: `slora/server/router/response_merger.py`

**Responsibility**: 收集 Worker 响应并转发到 Detokenization

**Class Design**:

```python
class ResponseMerger:
    """响应合并器"""
    
    def __init__(self, worker_response_port: int, detoken_port: int):
        """
        初始化 Response Merger
        
        Args:
            worker_response_port: 接收 Worker 响应的端口
            detoken_port: Detokenization 进程的端口
        """
        self.worker_response_port = worker_response_port
        self.detoken_port = detoken_port
        
        # ZMQ 通信
        self.context = None
        self.worker_receiver = None
        self.detoken_sender = None
    
    def _setup_zmq(self) -> None:
        """设置 ZMQ 通信"""
        self.context = zmq.asyncio.Context()
        
        # 接收 Worker 响应
        self.worker_receiver = self.context.socket(zmq.PULL)
        self.worker_receiver.bind(f"tcp://127.0.0.1:{self.worker_response_port}")
        
        # 发送到 Detokenization
        self.detoken_sender = self.context.socket(zmq.PUSH)
        self.detoken_sender.connect(f"tcp://127.0.0.1:{self.detoken_port}")
    
    async def _forward_to_detokenization(self, response: dict) -> None:
        """转发响应到 Detokenization"""
        # 转换为 Detokenization 期望的格式
        detoken_msg = self._convert_to_detoken_format(response)
        await self.detoken_sender.send_pyobj(detoken_msg)
    
    async def run(self) -> None:
        """主循环"""
        while True:
            response = await self.worker_receiver.recv_json()
            await self._forward_to_detokenization(response)
```

## Data Models

### Request Message

```python
@dataclass
class RequestMessage:
    """请求消息"""
    request_id: str          # 请求唯一标识符
    adapter_dir: str         # Adapter 路径
    prompt_ids: List[int]    # 输入 token IDs
    sampling_params: dict    # 采样参数
    timestamp: float         # 时间戳
```

### Response Message

```python
@dataclass
class ResponseMessage:
    """响应消息"""
    request_id: str          # 请求唯一标识符
    worker_id: int           # 处理该请求的 Worker ID
    output_ids: List[int]    # 输出 token IDs
    metadata: dict           # 元数据（logprobs 等）
    success: bool            # 是否成功
    error: Optional[str]     # 错误信息（如果失败）
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Worker 启动完整性

*For any* 配置的 Worker 数量 N，系统启动后应该有恰好 N 个 Worker 进程在运行

**Validates: Requirements 1.1, 1.5**

### Property 2: 轮询路由公平性

*For any* 连续的 N 个请求（N = Worker 数量），每个 Worker 应该恰好收到 1 个请求

**Validates: Requirements 2.1, 2.2**

### Property 3: 请求响应一致性

*For any* 发送的请求，返回的响应中的 request_id 应该与原请求的 request_id 相同

**Validates: Requirements 4.3**

### Property 4: GPU 隔离性

*For any* Worker，其 CUDA_VISIBLE_DEVICES 应该只包含分配给它的 GPU ID

**Validates: Requirements 1.3**

### Property 5: 消息完整性

*For any* 请求消息，应该包含所有必需的字段（request_id, adapter_dir, prompt_ids, sampling_params）

**Validates: Requirements 2.4**

## Error Handling

### Worker 启动失败

**场景**: GPU 不可用或模型加载失败

**处理**:
1. Worker 记录详细错误日志
2. Worker 进程退出并返回非零状态码
3. Router Manager 检测到启动失败
4. Router Manager 记录错误并终止整个系统

### 推理异常

**场景**: 推理过程中发生 CUDA OOM 或其他异常

**处理**:
1. Worker 捕获异常
2. Worker 返回错误响应（success=False, error=异常信息）
3. Response Merger 转发错误响应
4. API Server 返回 HTTP 500 错误给客户端

### ZMQ 通信超时

**场景**: 消息发送或接收超时

**处理**:
1. 设置 ZMQ socket 超时（如 30 秒）
2. 超时后记录警告日志
3. 重试发送/接收（最多 3 次）
4. 如果仍然失败，返回错误响应

### Worker 进程意外退出

**场景**: Worker 进程崩溃

**处理**:
1. Router Manager 定期检查 Worker 进程状态
2. 检测到进程退出时记录错误日志
3. Phase 1 不自动重启（Phase 3 实现）
4. 系统继续使用剩余的 Worker

## Testing Strategy

### Unit Tests

**测试框架**: pytest

**测试内容**:
1. `test_round_robin_router.py`: 测试轮询路由逻辑
2. `test_gpu_worker_init.py`: 测试 Worker 初始化
3. `test_message_format.py`: 测试消息格式转换

**示例**:
```python
def test_round_robin_selection():
    """测试轮询路由按顺序选择 Worker"""
    router = RoundRobinRouter(num_workers=3)
    
    selections = [router.select_worker() for _ in range(6)]
    
    assert selections == [0, 1, 2, 0, 1, 2]
```

### Integration Tests

**测试框架**: pytest + pytest-asyncio

**测试内容**:
1. `test_single_worker_e2e.py`: 单 Worker 端到端测试
2. `test_multi_worker_concurrent.py`: 多 Worker 并发测试
3. `test_zmq_communication.py`: ZMQ 通信测试

**示例**:
```python
@pytest.mark.asyncio
async def test_multi_worker_processing():
    """测试多个 Worker 能够并发处理请求"""
    manager = DataParallelRouterManager(args, num_workers=2)
    await manager.start_workers()
    
    # 发送 10 个请求
    requests = [create_test_request(i) for i in range(10)]
    tasks = [manager.route_request(req) for req in requests]
    
    await asyncio.gather(*tasks)
    
    # 验证所有请求都被处理
    # （通过检查响应或日志）
```

### Property-Based Tests

**测试框架**: Hypothesis

**测试内容**:
1. 测试轮询路由的公平性
2. 测试消息序列化/反序列化的一致性

**示例**:
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=8))
def test_round_robin_fairness(num_workers):
    """
    Property: 对于任意数量的 Worker，
    连续 N 个请求应该均匀分配到所有 Worker
    """
    router = RoundRobinRouter(num_workers)
    
    selections = [router.select_worker() for _ in range(num_workers)]
    
    # 每个 Worker 应该被选中恰好一次
    assert sorted(selections) == list(range(num_workers))
```

### Performance Tests

**测试脚本**: `benchmarks/benchmark_phase1.py`

**测试内容**:
1. 吞吐量测试（不同 Worker 数量）
2. 延迟测试（P50, P99）
3. GPU 利用率测试

**目标**:
- 3 Workers 吞吐量 ≥ 2.5x 单 Worker
- 平均延迟增加 < 10%
- GPU 利用率 > 85%

---

**文档版本**: v1.0  
**创建日期**: 2025-01-12  
**状态**: 设计文档
