# 淘汰策略优化测试验证指南

## 📋 测试目标

验证新的智能阈值淘汰策略是否正常工作，并对比旧策略的性能差异。

---

## 🧪 测试环境准备

### 1. 确认代码版本

确保以下文件已更新：
- ✅ `slora/server/router/manager.py` - 使用新的阈值淘汰策略
- ✅ `slora/server/api_server.py` - 添加了新的配置参数
- ✅ `slora/server/router/model_infer/infer_adapter.py` - 实现了阈值淘汰逻辑

### 2. 准备测试数据

```bash
cd /home/hzheng/S-LoRA/benchmarks

# 确保有测试用的 LoRA 适配器
# 如果没有，可以使用 --dummy 参数生成虚拟权重
```

---

## 🚀 测试场景

### 测试 1：基础功能验证（使用虚拟权重）

**目标**：验证新策略能正常运行，不会崩溃

**命令**：
```bash
python launch_server.py \
    --device a10g \
    --model-setting S1 \
    --backend slora \
    --num-adapter 100 \
    --num-token 10000 \
    --dummy \
    --evict-interval-threshold 0.85 \
    --evict-interval-ratio 0.3
```

**预期结果**：
- ✅ 服务器正常启动
- ✅ 加载 100 个虚拟适配器
- ✅ 能正常处理请求
- ✅ 日志中能看到淘汰信息（当内存使用率超过 85%）

**验证点**：
```bash
# 在另一个终端查看日志
tail -f server_logs.txt | grep "LoRA 内存使用率"
```

应该看到类似：
```
⚠️  LoRA 内存使用率 87.3% 超过阈值 85.0%，触发淘汰
   执行淘汰: 淘汰 15 个适配器
   淘汰完成: 使用率 87.3% → 72.1%
```

---

### 测试 2：不同阈值对比测试

**目标**：对比不同阈值配置对性能的影响

#### 配置 A：激进淘汰（低阈值）
```bash
python launch_server.py \
    --device a10g \
    --model-setting S1 \
    --backend slora \
    --num-adapter 100 \
    --num-token 10000 \
    --dummy \
    --evict-interval-threshold 0.70 \
    --evict-interval-ratio 0.5 \
    --evict-idle-threshold 0.80 \
    --evict-idle-ratio 0.6
```

**特点**：
- 更频繁触发淘汰
- 每次淘汰更多适配器
- 内存占用较低
- 可能有更多重复加载

#### 配置 B：保守淘汰（高阈值，推荐）
```bash
python launch_server.py \
    --device a10g \
    --model-setting S1 \
    --backend slora \
    --num-adapter 100 \
    --num-token 10000 \
    --dummy \
    --evict-interval-threshold 0.85 \
    --evict-interval-ratio 0.3 \
    --evict-idle-threshold 0.95 \
    --evict-idle-ratio 0.5
```

**特点**：
- 较少触发淘汰
- 保留更多适配器
- 内存占用较高
- 减少重复加载

#### 配置 C：极保守（接近旧策略）
```bash
python launch_server.py \
    --device a10g \
    --model-setting S1 \
    --backend slora \
    --num-adapter 100 \
    --num-token 10000 \
    --dummy \
    --evict-interval-threshold 0.99 \
    --evict-interval-ratio 0.1 \
    --evict-idle-threshold 0.99 \
    --evict-idle-ratio 0.1
```

**特点**：
- 几乎不触发淘汰
- 完全依赖加载前淘汰
- 可能导致 OOM

---

### 测试 3：压力测试（模拟高负载）

**目标**：验证在高并发场景下的稳定性

**命令**：
```bash
# 启动服务器
python launch_server.py \
    --device a10g \
    --model-setting S1 \
    --backend slora \
    --num-adapter 200 \
    --num-token 8000 \
    --dummy \
    --evict-interval-threshold 0.85 \
    --evict-interval-ratio 0.3

# 在另一个终端运行压力测试
python run_exp.py \
    --backend slora \
    --mode synthetic \
    --suite S1 \
    --output-file stress_test_results.jsonl
```

**监控指标**：
1. **淘汰频率**：`grep "触发淘汰" server_logs.txt | wc -l`
2. **适配器加载次数**：`grep "load.*adapters" server_logs.txt | wc -l`
3. **内存使用峰值**：查看日志中的 `usage_ratio`
4. **请求完成率**：查看 `stress_test_results.jsonl` 中的成功率

---

### 测试 4：热门适配器保留验证

**目标**：验证热门适配器确实被优先保留

**步骤**：

1. **启动服务器并开启详细日志**：
```bash
python launch_server.py \
    --device a10g \
    --model-setting S1 \
    --backend slora \
    --num-adapter 50 \
    --num-token 5000 \
    --dummy \
    --evict-interval-threshold 0.80 \
    --evict-interval-ratio 0.4
```

2. **运行有偏向性的测试**（某些适配器被频繁使用）：
```bash
# 修改 trace.py 或 exp_suite.py，设置 alpha 参数
# alpha 越大，请求分布越不均匀
python run_exp.py \
    --backend slora \
    --mode synthetic \
    --suite S1 \
    --output-file hotspot_test.jsonl
```

3. **分析日志**：
```bash
# 查看哪些适配器被淘汰
grep "淘汰列表" server_logs.txt

# 查看适配器使用统计（需要在代码中添加日志）
grep "adapter_.*分数" server_logs.txt
```

**预期结果**：
- 被淘汰的适配器分数较低
- 高分适配器（频繁使用的）不会被淘汰

---

## 📊 性能对比测试

### 对比场景：旧策略 vs 新策略

由于旧策略代码已被替换，可以通过调整阈值模拟：

#### 模拟旧策略（极高淘汰率）
```bash
# 设置极低阈值，每次请求完成都触发淘汰
python launch_server.py \
    --device a10g \
    --model-setting S1 \
    --backend slora \
    --num-adapter 100 \
    --num-token 10000 \
    --dummy \
    --evict-interval-threshold 0.01 \
    --evict-interval-ratio 0.99 \
    --evict-idle-threshold 0.01 \
    --evict-idle-ratio 0.99
```

#### 新策略（智能淘汰）
```bash
python launch_server.py \
    --device a10g \
    --model-setting S1 \
    --backend slora \
    --num-adapter 100 \
    --num-token 10000 \
    --dummy \
    --evict-interval-threshold 0.85 \
    --evict-interval-ratio 0.3 \
    --evict-idle-threshold 0.95 \
    --evict-idle-ratio 0.5
```

**对比指标**：

| 指标 | 获取方法 | 对比目标 |
|------|----------|----------|
| 吞吐量（req/s） | `time_stats.py` 分析结果 | 新策略应更高 |
| 平均延迟（ms） | 结果 jsonl 中的 `latency` | 新策略应更低 |
| P99 延迟（ms） | 计算 99 分位数 | 新策略应更低 |
| 适配器加载次数 | `grep "load.*adapters" \| wc -l` | 新策略应更少 |
| 内存平均使用率 | 计算日志中 `usage_ratio` 平均值 | 新策略应更高 |

---

## 🔍 详细测试步骤

### 步骤 1：基础功能测试

```bash
# 1. 进入测试目录
cd /home/hzheng/S-LoRA/benchmarks

# 2. 清理旧日志
rm -f server_logs.txt *.jsonl

# 3. 启动服务器（在后台运行，输出到日志）
nohup python launch_server.py \
    --device a10g \
    --model-setting S1 \
    --backend slora \
    --num-adapter 100 \
    --num-token 10000 \
    --dummy \
    --evict-interval-threshold 0.85 \
    --evict-interval-ratio 0.3 \
    > server_logs.txt 2>&1 &

# 记录进程 ID
SERVER_PID=$!
echo "服务器进程 ID: $SERVER_PID"

# 4. 等待服务器启动（大约 30-60 秒）
sleep 60

# 5. 检查服务器是否正常运行
curl http://127.0.0.1:8000/health || echo "服务器未就绪，请等待..."

# 6. 运行短期测试
python run_exp.py \
    --backend slora \
    --mode synthetic \
    --suite S1 \
    --output-file test_basic_results.jsonl

# 7. 等待测试完成
sleep 10

# 8. 停止服务器
kill $SERVER_PID

# 9. 分析结果
echo "=== 测试结果 ==="
python time_stats.py test_basic_results.jsonl

echo "=== 淘汰统计 ==="
grep "触发淘汰" server_logs.txt | wc -l
grep "淘汰完成" server_logs.txt
```

### 步骤 2：监控内存使用

创建监控脚本 `monitor_memory.sh`：

```bash
#!/bin/bash
# 保存为 monitor_memory.sh

echo "时间,淘汰触发次数,平均使用率"

while true; do
    timestamp=$(date +"%H:%M:%S")
    
    # 统计触发次数
    evict_count=$(grep "触发淘汰" server_logs.txt 2>/dev/null | wc -l)
    
    # 计算平均使用率（需要解析日志）
    avg_ratio=$(grep "usage_ratio" server_logs.txt 2>/dev/null | \
                tail -10 | \
                awk -F: '{sum+=$2; count++} END {if(count>0) print sum/count; else print 0}')
    
    echo "$timestamp,$evict_count,$avg_ratio"
    
    sleep 5
done
```

运行监控：
```bash
chmod +x monitor_memory.sh
./monitor_memory.sh > memory_monitor.csv &
MONITOR_PID=$!

# 运行测试...

# 停止监控
kill $MONITOR_PID
```

---

## 📈 结果分析

### 1. 分析吞吐量

```bash
# 使用内置的统计工具
python time_stats.py test_basic_results.jsonl

# 输出示例：
# Total requests: 1000
# Successful requests: 998
# Average latency: 245.6 ms
# P50 latency: 198.3 ms
# P99 latency: 567.2 ms
# Throughput: 12.4 req/s
```

### 2. 分析淘汰行为

```bash
# 统计淘汰事件
echo "总淘汰次数:"
grep "触发淘汰" server_logs.txt | wc -l

echo "平均每次淘汰的适配器数:"
grep "淘汰.*个适配器" server_logs.txt | \
    awk '{sum+=$NF; count++} END {print sum/count}'

echo "内存使用率变化:"
grep "使用率.*→" server_logs.txt | tail -20
```

### 3. 创建分析脚本

创建 `analyze_eviction.py`：

```python
#!/usr/bin/env python3
import re
import json
from collections import defaultdict

def analyze_eviction_log(log_file):
    """分析淘汰日志"""
    
    eviction_events = []
    adapter_evict_count = defaultdict(int)
    
    with open(log_file, 'r') as f:
        for line in f:
            # 解析触发淘汰的行
            if '触发淘汰' in line:
                match = re.search(r'使用率 ([\d.]+)%', line)
                if match:
                    ratio = float(match.group(1))
                    eviction_events.append({'trigger_ratio': ratio})
            
            # 解析淘汰列表
            elif '淘汰列表' in line:
                adapters = re.findall(r'adapter_\d+', line)
                for adapter in adapters:
                    adapter_evict_count[adapter] += 1
            
            # 解析淘汰完成
            elif '淘汰完成' in line:
                match = re.search(r'使用率 ([\d.]+)% → ([\d.]+)%', line)
                if match and eviction_events:
                    eviction_events[-1]['before_ratio'] = float(match.group(1))
                    eviction_events[-1]['after_ratio'] = float(match.group(2))
    
    # 输出统计
    print("=== 淘汰事件统计 ===")
    print(f"总淘汰事件数: {len(eviction_events)}")
    
    if eviction_events:
        avg_before = sum(e.get('before_ratio', 0) for e in eviction_events) / len(eviction_events)
        avg_after = sum(e.get('after_ratio', 0) for e in eviction_events) / len(eviction_events)
        print(f"平均触发前使用率: {avg_before:.1f}%")
        print(f"平均淘汰后使用率: {avg_after:.1f}%")
    
    print(f"\n=== 适配器淘汰频率 TOP 10 ===")
    top_adapters = sorted(adapter_evict_count.items(), key=lambda x: x[1], reverse=True)[:10]
    for adapter, count in top_adapters:
        print(f"{adapter}: {count} 次")

if __name__ == '__main__':
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'server_logs.txt'
    analyze_eviction_log(log_file)
```

运行分析：
```bash
python analyze_eviction.py server_logs.txt
```

---

## ✅ 验收标准

### 必须通过的测试

- [ ] **基础功能**：服务器能正常启动和运行
- [ ] **淘汰触发**：当内存使用率超过阈值时能触发淘汰
- [ ] **参数生效**：调整参数后行为有相应变化
- [ ] **无崩溃**：长时间运行不会崩溃或 OOM
- [ ] **请求成功**：测试请求成功率 > 95%

### 性能改进目标

- [ ] **吞吐量提升**：比旧策略提升 10-30%
- [ ] **延迟降低**：平均延迟降低 10-20%
- [ ] **加载次数减少**：适配器加载次数减少 50%+
- [ ] **内存利用率提高**：平均内存使用率提高 20-40%

---

## 🐛 常见问题排查

### 问题 1：服务器启动失败

**症状**：
```
AttributeError: 'InputParams' object has no attribute 'evict_interval_threshold'
```

**原因**：参数未正确传递到 `InputParams`

**解决**：检查 `api_server.py` 中是否正确设置了这些参数

---

### 问题 2：内存使用率一直很高，不触发淘汰

**可能原因**：
1. 阈值设置过高（如 0.99）
2. 所有适配器都在使用中（被保护）
3. 日志级别过低，看不到淘汰信息

**排查步骤**：
```bash
# 1. 检查参数
grep "evict.*threshold" server_logs.txt

# 2. 手动触发淘汰（需要修改代码添加 RPC 接口）

# 3. 降低阈值重新测试
```

---

### 问题 3：频繁 OOM

**症状**：
```
RuntimeError: CUDA out of memory
```

**原因**：
- 阈值设置过高
- `--num-token` 或 `--pool-size-lora` 设置过大
- 淘汰不够及时

**解决**：
```bash
# 降低阈值，更频繁淘汰
--evict-interval-threshold 0.75 \
--evict-interval-ratio 0.5

# 或减少适配器数量
--num-adapter 50

# 或增加 LoRA 专用空间
--pool-size-lora 20000
```

---

## 📝 测试报告模板

```markdown
# 淘汰策略优化测试报告

## 测试环境
- GPU: ___________
- 适配器数量: ___________
- 配置: ___________

## 测试结果

### 基础功能测试
- [x] 服务器正常启动
- [x] 淘汰正常触发
- [ ] 参数正确生效

### 性能对比
| 指标 | 旧策略 | 新策略 | 提升 |
|------|--------|--------|------|
| 吞吐量 | ___ req/s | ___ req/s | ___% |
| P50延迟 | ___ ms | ___ ms | ___% |
| P99延迟 | ___ ms | ___ ms | ___% |
| 加载次数 | ___ | ___ | ___% |

### 淘汰统计
- 总淘汰事件: ___
- 平均触发前使用率: ___%
- 平均淘汰后使用率: ___%

## 结论
___________

## 建议配置
___________
```

---

## 🚀 下一步优化方向

基于测试结果，可以考虑：

1. **自适应阈值**：根据负载动态调整阈值
2. **预测加载**：根据请求模式预测即将使用的适配器
3. **分级缓存**：将不常用的适配器放到 CPU 内存
4. **全局协调**：多卡场景下的协调淘汰

---

**文档版本**: 1.0  
**最后更新**: 2025-01-06  
**维护者**: S-LoRA Team

