# FIX-01 / FIX-02 / FIX-03 / CHECK-01 GPU 验证

这是正确性验证，不是性能实验。复用现有 Runner、HTTP 发请求、Worker、
模型 RPC 和真实内存池。服务器只启动一次，阶段之间等待全部请求完成，
保留 adapter 驻留状态，不调用缓存重置。

## 运行

先检查最终配置，不启动服务器、不创建结果目录：

```bash
cd /home/hzheng/S-LoRA/benchmarks
conda activate slora
python run_routing_comparison.py \
  --suite dp-gpu-fix-validation --gpu-ids 0,1 --dry-run
```

确认两张卡可用后运行。历史 `run1` 发现 FIX-03，修复后的 `run2` 已通过；
当前代码新增 CHECK-01，对账扩展尚待 GPU 验证，应使用新目录：

```bash
SLORA_MODEL_ROOT=/home/hzheng/models \
python run_routing_comparison.py \
  --suite dp-gpu-fix-validation \
  --gpu-ids 0,1 \
  --output-dir routing_comparison_results/check01_gpu_validation/run1
```

显式 `--gpu-ids` 覆盖套件并自动推断 worker 数；可额外指定 `--num-workers`，
但必须与 GPU 数量一致。普通套件未覆盖时保持原配置，默认仍为 GPU 1,2,3。
本验证套件默认使用 GPU 0,1、两个 Worker、4096-cell 内存池、10 个逻辑
adapter、`rwpt_active`、w2=0.4、beta=0.00845，不搜索参数、不启用主动复制。
默认模型为 `$SLORA_MODEL_ROOT/llama-7b`，两个 adapter 模板为
`alpaca-lora-7b` 和 `bactrian-x-llama-7b-lora`；其他位置可通过
`SLORA_REAL_MODEL`、`SLORA_REAL_ADAPTER_DIRS` 指定，后者以冒号分隔。
必须提供至少两种 rank 的可加载 adapter。启动前会检查配置文件和 rank。

## 阶段

| 阶段 | 两卡请求数 | 输入 / 输出长度 | 目的 |
|---|---:|---:|---|
| cold_load | 4 | 256 / 64 | 加载和 prefill 期间贡献持续可见 |
| resident_reuse | 8 | 256 / 128 | 重用上一阶段相同的高 rank adapter |
| capacity_boundary | 24 | 512 / 768 | 混合 rank、重复复用及内存边界等待 |

请求在每阶段内突发发送，长度为客户端构造值，日志按服务端实际 token 数验证。
每阶段最多等待 120 秒，超时立即停止，不会继续压入后续阶段。
本套件不读取旧 180 秒 trace，`duration` 在这里仅表示每阶段超时。
小内存池用于触发准入边界，不能与三卡 15000-cell 正式数据比较。
新增 CHECK-01 后，请求全部完成后最多再等 5 秒确认 Router 账本排空；这不是
增加第四个请求阶段，也不是固定延长压测。

## 输出与判定

套件目录下的 `diagnostics/server_<时间>/` 包含：

- `validation_manifest.json`：实际参数、协议及关键源文件 SHA-256。
- `validation_request_plan.json`：三个阶段的请求计划。
- `worker_0.jsonl`、`worker_1.jsonl`：接收/完成、阶段归属、驻留快照及准入账目。
- `worker_router.jsonl`：Router 实际接收并应用的 prefilling 上报。
- `route_decisions.jsonl`：逐请求路由评分和三阶段压力。
- `request_latencies_*.jsonl`：每阶段请求结果。
- `validation_summary.json`：自动检查结果，状态为 `passed`、`failed` 或 `incomplete`。

判定要求未完成请求没有丢失或重复贡献，原始输入长度和贡献不随生成变化，
prompt 缓冲区不被推理修改；完成后清零，Router 保留 prefilling
上报；驻留 adapter 与新增预留集合不重叠，预算保留原严格小于边界。同时必须
观察到驻留复用、新增预留、真实内存拒绝，以及“旧算法会拒绝、新预算允许”
的准入候选。最后一项是同一现场的预算对照，不代表旧版本实际运行的性能。

未触发目标场景是 `incomplete`，不是通过；错误或覆盖不足均返回非零退出码，
保留日志供分析。验证成功也不能据此声称吞吐改善。CLI 跳过常规性能分析和图表。

普通真实 trace 实验可显式加 `--validation-log` 记录同类证据；默认关闭，避免
把诊断开销带入正式性能对照。`--debug` 与它独立，不再依赖全局 `DEBUG=1`。

## CHECK-01 对账扩展

Worker 及 Router 需要使用配套新代码。请求携带本次启动标识和每 Worker 派发
编号；Worker 在入队/明确拒绝后推进确认水位，并与负载一起上报。不混用旧
Worker，也不支持 Worker 独立热重启后的自动恢复。

`worker_router.jsonl` 新增 `dispatch_reserved`、`dispatch_sent`、
`dispatch_rollback`、`dispatch_uncertain` 事件；已接受的 `worker_report` 增加
确认水位、原始/合成负载和剩余派发编号。Worker 的 received/lifecycle 记录
携带对应编号和快照水位。新 manifest 中 `dispatch_accounting=true`，源码
哈希列表扩展到 17 项。

新 GPU 运行必须检查派发与接收逐请求匹配、编号及确认水位单调、会话一致、
仅清除已覆盖请求、未覆盖贡献正确叠加、请求结束后两个 Worker 的账本清空。
未知发送结果或账目损坏不得算作通过；该轮仍不是性能实验。
summary 的 `dispatch_accounting.noncovering_reports_preserved` 记录未覆盖请求
被保留的现场次数，零次不能用来声称 GPU 已触发此交错。40 项新增 CPU 测试
已确定性覆盖关键交错及证据损坏，相关回归为 381 passed、6 个已知历史失败
排除。此轮没有启动 GPU 实验，旧 run2 不冒充新增协议的验证结果。

## 首轮失败与复验

2026-11-03 的 `server_20261103_172042_089730` 已完成全部 36 个请求，
目标覆盖无缺失，但跨阶段贡献守恒失败。单 GPU 本地推理共享并追加原始
prompt 列表，令输入长度与 RWPT 随 decode 增长。FIX-03 将推理历史独立复制，
Worker 负载读取固定 `Req.input_len`，并新增 prompt 实际缓冲区长度校验。

原 `run1` 的日志、manifest 和 `validation_summary.json` 保持原样，仍为失败；
不能以请求成功或改写历史记录代替修复后复验。上面的 `run2` 命令保持相同的
三阶段、GPU 和内存配置，不搜索参数。341 项相关 CPU 回归通过，6 项已知
历史失败排除。

用户随后执行 `run2`，诊断目录为
`run2/dp-gpu-fix-validation/diagnostics/server_20261103_173839_417643/`，
相对于 `routing_comparison_results/fix01_fix02_gpu_validation/`：

- 状态 `passed`，errors 与 missing_coverage 均为空，只读重分析与原结果一致。
- 三阶段请求 4/4、8/8、24/24 成功，两个 Worker 均排空。
- 6978 个生命周期快照通过检查；49058 个请求阶段样本无 prompt 长度漂移。
- 驻留复用 4253 次、新增预留 712 次、内存边界拒绝 4927 次，旧预算误拒
  候选 9 次；这些是重复候选检查次数，不是请求失败。
- Router 压力样本 72 个，收到并应用非零 prefilling 上报 72 次。
- 13 份关键源码哈希与检查时的工作区一致。

至此 FIX-01/FIX-02/FIX-03 通过当前两卡默认内存池路径的正确性套件。
控制台的 0.38 req/s、平均延迟 36.349 s 来自小内存池、长输出及突发积压，
不代表 1 RPS 的常规服务性能，也不与正式 trace 曲线比较。
manifest 保持 `performance_result=false`，未把该轮用于性能分析。

## 共享服务器

Runner 只清理自身启动的进程组，不再全局强杀 S-LoRA 或其他 GPU 的孤儿进程。
选定 GPU 已有计算进程时直接拒绝启动，不结束占用者。由于现有统计/配置文件
仍共用 `/tmp/slora_*`，Runner 使用互斥锁，发现其他 S-LoRA 服务时也会拒绝启动；
不同端口不代表这些共享文件已隔离。脱离本进程组的未知残留不自动清理。

通过本验证后再按相同硬件、trace、参数和起始缓存协议设计修复前后对照。
FIX-03 也会影响统计与响应，必须明确每组包含的修复；不能把差异全归因于
FIX-01/FIX-02。旧 rank 系数结果不能当作当前版本的严格基线。
