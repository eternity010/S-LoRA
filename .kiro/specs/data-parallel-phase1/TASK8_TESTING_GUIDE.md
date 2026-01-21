# Task 8 测试指南

## 概述

Task 8 是 Phase 1 的关键验证点，采用**从简单到复杂、从底层到上层**的阶段性测试策略。

## 测试层次

```
Level 1: 组件级测试（纯 Python，无进程）
   ↓
Level 2: 进程级测试（启动进程，不发送请求）
   ↓
Level 3: 通信级测试（测试 ZMQ 消息传递）
   ↓
Level 4: 功能级测试（完整请求-响应流程）
   ↓
Level 5: 错误处理测试（异常场景）
   ↓
Level 6: 监控和日志测试（验证输出）
```

## 测试准备

### 环境检查

```bash
# 1. 激活环境
conda activate slora

# 2. 检查 Python 版本
python --version  # 应该是 3.9+

# 3. 检查 GPU
nvidia-smi

# 4. 检查依赖
pip list | grep -E "torch|zmq|pytest"

# 5. 创建测试目录
mkdir -p test/phase1_manual
```

### 测试脚本模板

每个测试脚本应该包含：

```python
#!/usr/bin/env python3
"""
测试名称: XXX
测试目标: XXX
预期结果: XXX
"""

import sys
import time

def setup():
    """测试前准备"""
    print("=" * 60)
    print("测试开始: XXX")
    print("=" * 60)

def test_xxx():
    """主测试逻辑"""
    try:
        # 测试代码
        pass
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    print("✅ 测试通过")
    return True

def teardown():
    """测试后清理"""
    print("=" * 60)
    print("测试结束")
    print("=" * 60)

if __name__ == "__main__":
    setup()
    success = test_xxx()
    teardown()
    sys.exit(0 if success else 1)
```

## Level 1: 组件级测试

### 8.2.1 测试 Round Robin Router

**文件**: `test/phase1_manual/test_router.py`

**测试内容**:
```python
from slora.server.router.round_robin_router import RoundRobinRouter

def test_router():
    # 测试 3 个 Worker
    router = RoundRobinRouter(num_workers=3)
    
    # 测试轮询顺序
    selections = [router.select_worker() for _ in range(9)]
    expected = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    
    assert selections == expected, f"Expected {expected}, got {selections}"
    print(f"✅ 轮询顺序正确: {selections}")
    
    # 测试统计
    stats = router.get_stats()
    print(f"✅ 统计信息: {stats}")
    
    return True
```

**运行**:
```bash
python test/phase1_manual/test_router.py
```

**预期输出**:
```
============================================================
测试开始: Round Robin Router
============================================================
✅ 轮询顺序正确: [0, 1, 2, 0, 1, 2, 0, 1, 2]
✅ 统计信息: {'total_requests': 9, 'worker_0': 3, 'worker_1': 3, 'worker_2': 3}
✅ 测试通过
============================================================
测试结束
============================================================
```

### 8.2.2 测试 GPU Worker 初始化

**文件**: `test/phase1_manual/test_worker_init.py`

**测试内容**:
```python
from slora.server.router.gpu_worker import GPUWorker
import argparse

def test_worker_init():
    # 创建测试参数
    args = argparse.Namespace(
        worker_id=0,
        gpu_id=0,
        max_total_token_num=6000,
        batch_max_tokens=512,
        running_max_req_size=256,
        lora_dirs=[],
        no_lora=True,
        dummy=True
    )
    
    # 创建 Worker 实例
    worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
    
    # 验证初始化
    assert worker.worker_id == 0
    assert worker.gpu_id == 0
    print(f"✅ Worker 初始化成功: worker_id={worker.worker_id}, gpu_id={worker.gpu_id}")
    
    # 测试 GPU 环境设置
    worker._setup_gpu()
    print(f"✅ GPU 环境设置成功")
    
    return True
```

**运行**:
```bash
python test/phase1_manual/test_worker_init.py
```

### 8.2.3 测试 DataParallelRouterManager 初始化

**文件**: `test/phase1_manual/test_manager_init.py`

**测试内容**:
```python
from slora.server.router.dp_manager import DataParallelRouterManager
import argparse

def test_manager_init():
    args = argparse.Namespace(
        num_workers=2,
        gpu_ids="0,1",
        # ... 其他参数
    )
    
    manager = DataParallelRouterManager(
        args=args,
        router_port=50000,
        response_port=60000
    )
    
    # 验证 GPU 检测
    assert manager.num_workers == 2
    print(f"✅ GPU 检测成功: {manager.num_workers} workers")
    
    # 验证端口分配
    assert len(manager.worker_ports) == 2
    print(f"✅ 端口分配成功: {manager.worker_ports}")
    
    return True
```

## Level 2: 进程级测试

### 8.3.1 测试单个 Worker 进程启动

**文件**: `test/phase1_manual/test_single_worker_startup.py`

**测试内容**:
```python
import multiprocessing as mp
import time
from slora.server.router.gpu_worker import run_gpu_worker_process

def test_single_worker_startup():
    # 启动 Worker 进程
    proc = mp.Process(
        target=run_gpu_worker_process,
        args=(0, 0, args, 50000, 60000)
    )
    proc.start()
    
    # 等待启动
    time.sleep(5)
    
    # 检查进程状态
    assert proc.is_alive(), "Worker 进程未启动"
    print(f"✅ Worker 进程启动成功: PID={proc.pid}")
    
    # 优雅关闭
    proc.terminate()
    proc.join(timeout=5)
    print(f"✅ Worker 进程已关闭")
    
    return True
```

**运行**:
```bash
python test/phase1_manual/test_single_worker_startup.py
```

**预期日志**:
```
Worker 0 ready on GPU 0
```

### 8.3.2 测试多个 Worker 进程启动

**文件**: `test/phase1_manual/test_multi_worker_startup.py`

**测试内容**:
```python
def test_multi_worker_startup():
    workers = []
    
    # 启动 2 个 Worker
    for i in range(2):
        proc = mp.Process(
            target=run_gpu_worker_process,
            args=(i, i, args, 50000 + i, 60000)
        )
        proc.start()
        workers.append(proc)
    
    # 等待启动
    time.sleep(10)
    
    # 检查所有进程
    for i, proc in enumerate(workers):
        assert proc.is_alive(), f"Worker {i} 未启动"
        print(f"✅ Worker {i} 启动成功: PID={proc.pid}")
    
    # 优雅关闭
    for proc in workers:
        proc.terminate()
        proc.join(timeout=5)
    
    print(f"✅ 所有 Worker 已关闭")
    return True
```

## Level 3: 通信级测试

### 8.4.1 测试 Manager → Worker 通信

**文件**: `test/phase1_manual/test_manager_to_worker.py`

**测试内容**:
```python
import zmq
import zmq.asyncio
import asyncio

async def test_manager_to_worker():
    # Manager 端（PUSH）
    context = zmq.asyncio.Context()
    sender = context.socket(zmq.PUSH)
    sender.bind("tcp://127.0.0.1:50000")
    
    # Worker 端（PULL）
    receiver = context.socket(zmq.PULL)
    receiver.connect("tcp://127.0.0.1:50000")
    
    # 发送测试消息
    test_msg = {"request_id": "test_001", "prompt": "Hello"}
    await sender.send_json(test_msg)
    print(f"✅ Manager 发送消息: {test_msg}")
    
    # 接收消息
    received = await receiver.recv_json()
    print(f"✅ Worker 接收消息: {received}")
    
    # 验证
    assert received == test_msg
    print(f"✅ 消息内容正确")
    
    # 清理
    sender.close()
    receiver.close()
    context.term()
    
    return True

if __name__ == "__main__":
    asyncio.run(test_manager_to_worker())
```

### 8.4.3 测试完整通信链路

**文件**: `test/phase1_manual/test_full_communication.py`

**测试内容**:
- 启动 Manager、2 个 Worker、Response Merger
- 发送 10 个测试请求
- 验证轮询分配
- 验证所有响应被接收

## Level 4: 功能级测试

### 8.5.1 测试单个请求处理

**文件**: `test/phase1_manual/test_single_request.py`

**启动服务器**:
```bash
# 终端 1
python -m slora.server.api_server \
    --model-dir dummy \
    --parallel-mode data \
    --num-workers 1 \
    --dummy \
    --port 8000
```

**测试脚本**:
```python
import requests

def test_single_request():
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "prompt": "Hello, how are you?",
            "max_tokens": 50
        }
    )
    
    assert response.status_code == 200
    result = response.json()
    
    print(f"✅ 请求成功")
    print(f"   Request ID: {result.get('request_id')}")
    print(f"   Output: {result.get('output')}")
    
    return True
```

### 8.5.3 测试并发请求

**文件**: `test/phase1_manual/test_concurrent_requests.py`

**测试内容**:
```python
import requests
import concurrent.futures
import time

def send_request(i):
    start = time.time()
    response = requests.post(
        "http://localhost:8000/generate",
        json={"prompt": f"Request {i}", "max_tokens": 50}
    )
    latency = time.time() - start
    return i, response.status_code, latency

def test_concurrent_requests():
    num_requests = 10
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        results = [f.result() for f in futures]
    
    # 统计
    success_count = sum(1 for _, status, _ in results if status == 200)
    avg_latency = sum(lat for _, _, lat in results) / len(results)
    
    print(f"✅ 并发测试完成")
    print(f"   总请求数: {num_requests}")
    print(f"   成功数: {success_count}")
    print(f"   平均延迟: {avg_latency:.3f}s")
    
    assert success_count == num_requests
    return True
```

## Level 5: 错误处理测试

### 8.6.2 测试 Worker 进程崩溃

**文件**: `test/phase1_manual/test_worker_crash.py`

**测试步骤**:
1. 启动系统（2 Workers）
2. 发送几个请求验证正常工作
3. 手动杀死一个 Worker: `kill -9 <pid>`
4. 继续发送请求
5. 验证系统继续工作（使用剩余 Worker）
6. 检查 Manager 日志是否检测到崩溃

## Level 6: 监控和日志测试

### 8.7.1 验证启动日志

**检查清单**:
- [ ] 输出并行模式信息
- [ ] 输出 Worker 数量
- [ ] 输出 GPU 列表
- [ ] 每个 Worker 输出就绪消息
- [ ] 输出 GPU 信息（型号、内存）

### 8.7.2 验证请求统计

**检查清单**:
- [ ] 每 10 秒输出统计信息
- [ ] 包含总请求数
- [ ] 包含成功/失败数
- [ ] 包含每个 Worker 的请求分布

## 测试执行顺序

建议按照以下顺序执行测试：

```
Day 1: Level 1 组件级测试
  ├─ 8.2.1 Round Robin Router ✓
  ├─ 8.2.2 GPU Worker 初始化 ✓
  └─ 8.2.3 Manager 初始化 ✓

Day 2: Level 2 进程级测试
  ├─ 8.3.1 单 Worker 启动 ✓
  ├─ 8.3.2 多 Worker 启动 ✓
  └─ 8.3.3 Response Merger 启动 ✓

Day 3: Level 3 通信级测试
  ├─ 8.4.1 Manager → Worker ✓
  ├─ 8.4.2 Worker → Merger ✓
  └─ 8.4.3 完整通信链路 ✓

Day 4: Level 4 功能级测试
  ├─ 8.5.1 单个请求 ✓
  ├─ 8.5.2 串行请求 ✓
  ├─ 8.5.3 并发请求 ✓
  └─ 8.5.4 高并发 ✓

Day 5: Level 5-6 错误和监控测试
  ├─ 8.6.1 启动失败 ✓
  ├─ 8.6.2 进程崩溃 ✓
  ├─ 8.6.3 推理异常 ✓
  ├─ 8.7.1 启动日志 ✓
  ├─ 8.7.2 请求统计 ✓
  └─ 8.7.3 调试日志 ✓
```

## 测试结果记录

### 测试结果模板

```markdown
## Task 8.X.X 测试结果

**测试日期**: 2025-01-XX
**测试人**: XXX
**测试环境**: 
- GPU: XXX
- CUDA: XXX
- Python: XXX

**测试结果**: ✅ 通过 / ❌ 失败

**详细说明**:
- XXX

**发现的问题**:
1. XXX
2. XXX

**性能数据**:
- 吞吐量: XXX req/s
- 延迟: XXX ms
```

## 常见问题

### Q1: 测试脚本找不到模块
```bash
# 确保在项目根目录运行
cd /path/to/S-LoRA
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Q2: ZMQ 端口被占用
```bash
# 查找占用端口的进程
lsof -i :50000

# 杀死进程
kill -9 <pid>
```

### Q3: GPU 不可用
```bash
# 检查 GPU
nvidia-smi

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

**文档版本**: v1.0  
**创建日期**: 2025-01-13  
**维护者**: Phase 1 开发团队
