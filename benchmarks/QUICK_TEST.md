# 快速测试指南

## 问题：每次重启加载模型太慢

每次重启服务器都需要加载模型（25GB+），在 3 个 GPU 上需要几分钟时间。

## 解决方案

### 方案 1: 使用 Dummy 模式（推荐用于功能测试）

Dummy 模式使用随机权重，跳过实际的模型加载，启动速度快 10-20 倍。

**适用场景：**
- 测试数据并行路由逻辑
- 测试请求分发和响应合并
- 测试 ZMQ 通信
- 快速验证代码修改

**不适用场景：**
- 测试实际推理结果
- 性能基准测试
- 生成质量评估

#### 启动命令

```bash
# 数据并行 + Dummy 模式（3 个 GPU）
python launch_server.py \
  --model-setting Real \
  --num-adapter 100 \
  --num-token 5000 \
  --dummy \
  --parallel-mode data \
  --num-workers 3 \
  --gpu-ids 0,1,2
```

**启动时间对比：**
- 真实模型：~3-5 分钟（每个 GPU 加载 25GB）
- Dummy 模式：~10-30 秒（只初始化结构）

#### 测试请求

```bash
# 在另一个终端发送测试请求
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "What is the capital of France?",
    "parameters": {
      "max_new_tokens": 50,
      "temperature": 1.0
    }
  }'
```

**注意：** Dummy 模式返回的是随机 token，不是有意义的文本。

---

### 方案 2: 保持服务器运行（推荐用于开发）

不要每次都重启服务器，只在必要时重启。

#### 什么时候需要重启？

**需要重启：**
- 修改了模型加载相关代码（`model_rpc.py`, `gpu_worker.py` 的模型初始化部分）
- 修改了进程启动逻辑（`dp_manager.py` 的 Worker 启动部分）
- 修改了命令行参数处理

**不需要重启：**
- 修改了路由逻辑（`dp_manager.py` 的 `route_request` 方法）
- 修改了响应处理（`response_merger.py`）
- 修改了日志输出
- 修改了统计信息

#### 热重载技巧

对于不需要重启的修改，可以使用 Python 的热重载：

```python
# 在代码中添加（仅用于开发）
import importlib
import slora.server.router.dp_manager
importlib.reload(slora.server.router.dp_manager)
```

---

### 方案 3: 使用更小的模型（用于快速迭代）

如果需要测试实际推理，可以使用更小的模型。

```bash
# 使用 1B 或 3B 模型（如果有的话）
python launch_server.py \
  --model-setting Real \
  --model-dir /path/to/smaller-model \
  --num-adapter 10 \
  --num-token 2000 \
  --parallel-mode data \
  --num-workers 3 \
  --gpu-ids 0,1,2
```

---

### 方案 4: 单 Worker 测试（最快）

如果只是测试功能，可以只启动 1 个 Worker。

```bash
# 单 Worker + Dummy 模式（最快启动）
python launch_server.py \
  --model-setting Real \
  --num-adapter 10 \
  --num-token 2000 \
  --dummy \
  --parallel-mode data \
  --num-workers 1 \
  --gpu-ids 0
```

**启动时间：** ~5-10 秒

---

## 开发工作流建议

### 阶段 1: 功能开发（使用 Dummy 模式）

```bash
# 启动服务器（Dummy 模式，3 个 Worker）
python launch_server.py \
  --model-setting Real \
  --num-adapter 100 \
  --num-token 5000 \
  --dummy \
  --parallel-mode data \
  --num-workers 3 \
  --gpu-ids 0,1,2

# 测试请求分发
python run_exp.py --debug --model-setting Real
```

**优点：**
- 启动快（10-30 秒）
- 可以快速验证路由逻辑
- 可以测试多 Worker 协作

### 阶段 2: 功能验证（使用真实模型，单 Worker）

```bash
# 启动服务器（真实模型，1 个 Worker）
python launch_server.py \
  --model-setting Real \
  --num-adapter 10 \
  --num-token 2000 \
  --parallel-mode data \
  --num-workers 1 \
  --gpu-ids 0

# 测试实际推理
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "What is the capital of France?",
    "parameters": {"max_new_tokens": 50}
  }'
```

**优点：**
- 加载时间减少 2/3（只加载 1 个模型）
- 可以验证实际推理结果
- 可以测试 LoRA adapter 加载

### 阶段 3: 性能测试（使用真实模型，多 Worker）

```bash
# 启动服务器（真实模型，3 个 Worker）
python launch_server.py \
  --model-setting Real \
  --num-adapter 100 \
  --num-token 5000 \
  --parallel-mode data \
  --num-workers 3 \
  --gpu-ids 0,1,2

# 运行性能测试
python run_exp.py --model-setting Real
```

**优点：**
- 完整的性能测试
- 真实的负载分布
- 准确的吞吐量测量

---

## 调试技巧

### 1. 检查服务器是否还在运行

```bash
# 检查进程
ps aux | grep api_server

# 检查端口
lsof -i :8000
```

### 2. 优雅关闭服务器

```bash
# 发送 SIGTERM（优雅关闭）
pkill -TERM -f api_server

# 如果卡住，强制关闭
pkill -KILL -f api_server
```

### 3. 查看日志

```bash
# 实时查看日志
tail -f /path/to/log/file

# 或者直接在终端查看输出
```

### 4. 快速重启脚本

创建一个重启脚本 `restart.sh`：

```bash
#!/bin/bash
# 关闭旧服务器
pkill -TERM -f api_server
sleep 2

# 启动新服务器（Dummy 模式）
python launch_server.py \
  --model-setting Real \
  --num-adapter 100 \
  --num-token 5000 \
  --dummy \
  --parallel-mode data \
  --num-workers 3 \
  --gpu-ids 0,1,2
```

使用：
```bash
chmod +x restart.sh
./restart.sh
```

---

## 总结

| 场景 | 推荐方案 | 启动时间 | 命令 |
|------|---------|---------|------|
| 功能开发 | Dummy + 3 Workers | ~30s | `--dummy --num-workers 3` |
| 快速测试 | Dummy + 1 Worker | ~10s | `--dummy --num-workers 1` |
| 推理验证 | 真实 + 1 Worker | ~1min | `--num-workers 1` |
| 性能测试 | 真实 + 3 Workers | ~3-5min | `--num-workers 3` |

**建议：** 在开发阶段使用 Dummy 模式，只在需要验证实际推理结果时才使用真实模型。
