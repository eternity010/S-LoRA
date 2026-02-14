# 调试记录

## 请求丢失问题 (2026-02-02)

### 问题
客户端发送 60 个请求，Worker 只收到约 20 个，日志警告：
```
RuntimeWarning: Future <Future cancelled> completed while awaiting... A message has been dropped!
```

### 原因

1. **Worker 端**: 使用 `asyncio.wait_for(timeout=0.001)` 导致 Future 被取消，ZMQ 消息丢失
2. **Router Manager**: ZMQ 异步 context 在多个事件循环中使用，导致状态不一致

### 修复

**Worker 端 - 双协程架构**:
```python
# 接收协程：持续接收 ZMQ 消息 → 存入本地队列
async def _receive_requests_loop(self):
    while True:
        request = await self.request_receiver.recv_json()
        self.req_queue.append(self._convert_to_req_object(request))

# 处理协程：从队列取出 → 推理 → 发送响应
async def _process_loop(self):
    while True:
        responses = await self._process_requests()
        for response in responses:
            await self._send_response(response)
```

**Router Manager - 统一事件循环**:
```python
loop = asyncio.new_event_loop()
async def run_all():
    dp_manager._setup_zmq()
    await dp_manager.start_workers()
    await dp_manager.run()
loop.run_until_complete(run_all())
```

### 修改文件
- `slora/server/router/gpu_worker.py` - 双协程架构
- `slora/server/router/manager.py` - 统一事件循环
- `slora/server/router/dp_manager.py` - 移除 callback 参数

### 经验
1. 避免在异步 I/O 中使用 `asyncio.wait_for()` 的短超时
2. ZMQ 异步操作应在同一个事件循环中
3. 分离接收和处理逻辑，通过本地队列解耦

## 智能路由不生效 (2026-02-13)

### 问题
使用 `--routing-strategy adapter-aware` 启动后，请求仍均匀分配（33.3%），智能路由未生效。

### 原因
`run_gpu_worker_process` 没有将 `state_receiver_port` 传递给 Worker，也没有调用 `_setup_state_reporter()`。Worker 不上报状态，Router 不知道各 Worker 缓存了哪些 adapter，评分公式中 `cache_indicator` 始终为 0，退化为按队列长度均衡分配。

### 修复
`slora/server/router/dp_manager.py` 三处改动：

1. `_start_worker`: args 中传递 `self.state_receiver_port`
2. `run_gpu_worker_process` 签名: 添加 `state_report_port` 参数
3. `run_gpu_worker_process` 函数体: 调用 `worker._setup_state_reporter(state_report_port)`

### 验证
修复后测试（alpha=0.3 热点分布，100 adapter，3 Worker）：
- Worker 0: 51.3% / Worker 1: 23.0% / Worker 2: 25.7%
- `routing_debug.log` 确认 `has_target=True` 时 cache=1.00 加分生效

### 经验
跨进程通信的参数传递容易遗漏，尤其是 `multiprocessing.Process` 的 args 列表。功能实现后应检查完整调用链：端口创建 → 端口传递 → 子进程使用。
