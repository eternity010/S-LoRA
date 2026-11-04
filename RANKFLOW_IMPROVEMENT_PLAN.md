# RankFlow 后续修正与研究计划

记录日期：2026-11-03。

本记录承接代码复查和研究方向讨论，用于逐项处理问题。已复现的问题、
尚待验证的实现风险、研究假设分别记录，不能相互替代。

## 当前基线与推进原则

- 仓库：`/home/hzheng/S-LoRA`。
- 分支：`scheduler-opt`。
- 基线提交：`63ee224966b68b9b5c6487518066614084a19399`。
- 复查前工作区干净；此前已核对本地提交与远端一致。
- 仓库尚无正式冻结标签；上述提交作为本轮修正前的版本依据。
- 本轮先完成代码复查、FIX-01/FIX-02 实现和 CPU 回归；用户随后运行首轮 GPU
  验证，发现独立的 prompt 列表共享问题。修复后用户执行 run2，三项修复通过
  当前两卡默认内存池路径的正确性套件；诊断、证据及适用范围见下文。
- FIX-01 已独立保存为 `cf58f22`；FIX-02、FIX-03、验证套件及验收记录随本轮
  后续提交一并保存。run2 的 13 份关键源码哈希与该批代码一致。
- 先修正实现，再建立当前版本的有效基线，随后只推进一个主要方法改进方向。
- 不预设修复必然提高性能，也不通过选择有利负载点来代替完整验证。
- 热点 adapter 主动复制暂缓研究，未启用或调整复制决策；FIX-02 仅补齐既有
  预加载入口的显存账目刷新和准入同步。

## 处理顺序

| 顺序 | 编号 | 事项 | 证据状态 | 处理状态 |
|---|---|---|---|---|
| 1 | FIX-01 | 补齐 adapter 加载及 prefill 阶段的负载可见性 | run2 6978 个生命周期快照及 72 次非零 prefilling 上报通过检查 | 当前两卡套件通过 |
| 2 | FIX-02 | 修正批次准入中已驻留 adapter 的显存重复计数 | run2 观察到 9 次旧预算误拒候选，无预算校验错误 | 当前两卡套件通过 |
| 3 | FIX-03 | 隔离推理 token 缓冲区与原始 prompt | run2 49058 个请求阶段样本无 prompt 长度漂移 | 当前两卡套件通过 |
| 4 | CHECK-01 | 验证乐观预增与 Worker 上报的因果对齐 | CPU 确定性复现并通过修复回归，真实影响未量化 | 实现完成，待 GPU 验证 |
| 5 | CHECK-02 | 核对 Router 缓存目录与 GPU 实际驻留集合 | 代码路径存在风险，触发频率未量化 | 待核查 |
| 6 | EXP-01 | 区分旧系数结果，建立修正后版本基线 | 已发现版本与数据时间差异 | 待处理 |
| 7 | METHOD-01 | 评估批次准入感知的请求等待代价路由 | 方法假设，尚未实现或验证 | 待设计 |

CHECK 项先做最小复现，确认问题后再确定修复范围。每完成一项，更新本表、
记录提交与验证结果，再进入下一项，不把多个机制改动混在同一次性能比较中。

## FIX-01：prefill 阶段负载统计空窗

### 证据与原因

相关代码：

- `slora/server/router/req_queue.py`：`generate_new_batch()`。
- `slora/server/router/gpu_worker.py`：`_process_requests()`、`_get_state_for_reporter()`。

请求生成新批次时已经离开 `waiting_req_list`，但新批次在完成 adapter 加载、
初始化和 prefill 后，才加入或合并到 `current_batch`。
状态采集只遍历等待队列和 `current_batch`，因此遗漏了中间阶段的请求。
这些阶段存在协作让出和状态上报机会，遗漏可能传递到 Router。

CPU 模拟使用 100 个 prompt tokens、rank 16、beta=0.00845 的单请求，
执行真实的队列准入和 Worker 调度路径，以模拟 RPC 替代 GPU 执行：

| 阶段 | Waiting RWPT | Active RWPT | 可见请求数 |
|---|---:|---:|---:|
| 等待中 | 113 | 0 | 1 |
| 已准入、尚未 prefill | 0 | 0 | 0 |
| prefill 调用期间 | 0 | 0 | 0 |
| 完成后 | 0 | 0 | 0 |

这证明统计遗漏存在；尚未测量它在真实 trace 中的持续时间和性能影响。

### 修正目标与验收

- 明确 waiting、已准入/加载/prefill、后续执行、完成等阶段的统计归属。
- 未完成请求在各阶段持续可见，阶段转换不丢失也不重复贡献。
- RWPT Active、Token Count Active 和请求数统计使用一致的请求集合。
- 覆盖已有执行批次时新增批次、单 token 完成、prefill 失败等路径。
- 保留 waiting-only 指标的定义，避免在修复过程中改变消融含义。
- 先用不依赖 GPU 的回归测试验证，再测量真实运行中的遗漏比例。

### 实现与验证（2026-11-03）

- Worker 新增 `prefilling_batch`，在真实队列准入后、第一次异步调用前登记，
  覆盖 adapter 加载、RPC 初始化、prefill 及 RPC 合并阶段。
- 新批次转交给 `current_batch` 后立即清除过渡引用，两步之间不让出协程；
  过渡统计排除已完成请求，失败和取消退出时也清除引用。
- 保留已有 waiting 和 current-batch 字段定义，新增 `prefilling_request_count`、
  `prefilling_raw_tokens`、`prefilling_rwpt_tokens`。
- 请求数和 Active 负载均包含过渡阶段；waiting-only 的 RWPT、Token Count
  继续只统计等待队列。未调整 beta、路由权重或批次准入逻辑。
- 新字段已贯通 WorkerState 序列化、reporter、Router 接收、状态缓存、评分及
  路由诊断记录；旧消息缺少字段时默认为零，但混用旧 Worker 无法修正其统计遗漏。

新增测试：`benchmarks/tests/test_routing_experiment/test_prefill_load_visibility.py`。
共 23 项 CPU 测试，覆盖真实 ReqQueue 准入、各阶段负载守恒、旧批次合并、
部分及单 token 完成、加载/初始化/prefill/merge/decode 失败、取消清理、
上报到 Router 评分与诊断的传递，以及旧消息兼容。
测试通过 mock 禁用 CUDA 可用性检查，未运行 GPU 推理。

100 prompt tokens、rank 16 的请求，在 waiting、加载、prefill、decode
阶段的可见总贡献均为 113 RWPT，完成后归零；与已有 rank-64 请求合并时，
总贡献保持 267 RWPT，不重复计数。

针对 Worker、RWPT、状态序列化、reporter/cache、Router/DP manager、
诊断等相关测试，最终结果为 **195 passed, 6 deselected**。
6 项排除项已在修复前的 `63ee224` 独立副本中全部复现失败：

- `test_worker_state.py` 中两项仍期望旧的热点阈值默认值 10，当前为 1e9。
- `test_gpu_worker_cooperative_scheduling.py` 中一项未设置 profiled beta，
  且仍期望理论 gamma 的旧公式。
- `test_weight_configuration_effect.py` 中三项按 queue-length 公式断言，
  但配置使用默认 RWPT 指标。

首次扩大检查还出现一项 Hypothesis 样本生成过慢的健康检查失败，单独重跑和
最终回归均通过。上述旧测试未在本次修复中修改。

仅验证统计正确性，尚未验证真实运行的性能收益。未重启已有服务器；本次修复、
测试和本记录已作为 FIX-01 独立提交 `cf58f22` 保存，未推送远端。
FIX-02 的后续实现和验证记录见下节。

## FIX-02：已驻留 adapter 显存重复计数

### 证据与原因

相关代码：

- `slora/server/router/req_queue.py`：`_init_cache_list()`、`_can_add_new_req()`。
- `slora/server/router/gpu_worker.py`：`actual_adapter_memory_usage` 的采集和传递。

`actual_adapter_size` 已包含所有已驻留 adapter 的实际占用，但准入逻辑还会
累加当前批次及候选请求的 adapter 占用，其中可能包含同一个已驻留 adapter。

CPU 模拟使用总容量 800 cells、一个已驻留 rank-64 adapter 占用 256 cells，
当前请求和候选请求均使用该 adapter：

- 两个请求的预计 KV 峰值为 398 cells。
- adapter 只计一次时可用 KV 容量为 544 cells，应可准入。
- 当前逻辑又增加 256 cells，将可用容量压到 288 cells，返回拒绝准入。

### 修正目标与验收

- adapter 占用按集合计算：实际驻留占用，加上尚未驻留且本次确实需要新增的占用。
- 同一个 adapter 被多个请求复用时只计一次；已驻留但不在当前批次的 adapter
  也不能在再次使用时重复计数。
- 核对 cells 的单位和底层分配器定义，不靠降低预留量掩盖计算问题。
- 覆盖复用已驻留 adapter、新增未驻留 adapter、混合 rank 和容量边界。
- 保留真实的容量约束，避免为了扩大并发引入显存超配。
- 性能实验关注批次准入拒绝、实际并发和延迟，不能预先承诺收益。

### 实现与验证（2026-11-03）

本项基于 FIX-01 提交 `cf58f22` 实现；实现和 CPU 验证阶段暂存工作区，
后随下文 GPU 复验通过的代码与套件一并提交。

- `InferAdapter.get_lora_memory_usage()` 新增 `adapter_cells_by_dir`，
  由同一份 `adapter_dirs` 与 `a_len` 构造；已有返回字段保留。
- 默认 `ReqQueue` 接受可选驻留映射。总占用按“快照中的实际占用 + 本轮所需但
  尚未驻留的 adapter 预留”计算；新 adapter 继续使用与底层一致的 `rank * 4`
  cells 规则，每个 adapter 仅预留一次，闲置缓存仍占用容量。
- Worker 的准入、adapter 加载及 RPC 批次注册使用同一把锁，与预加载、显式
  淘汰和清空串行化。加载前登记 `prefilling_batch`，保留 FIX-01 的连续可见性；
  RPC 批次注册后释放锁，prefill 和 decode 不持有此锁。
- 正常加载（含内部淘汰）、预加载、显式淘汰及清空后刷新驻留快照；操作部分失败
  或取消时也尝试刷新。清空操作在批次仍在途时拒绝执行。
- 驻留变更前使旧快照失效。查询失败、缺少映射或映射与占用不一致时，停止新增
  准入，已有批次继续执行；失败重试限频为约 0.5 秒，空闲等待时短暂让出协程。
  旧占用数字保留供诊断，不将查询失败替换成零占用。
- Python 3.9 的锁在 Worker 协程首次使用时创建，避免同步构造阶段绑定错误的
  事件循环。正常 decode 不增加逐步内存查询。
- 本次覆盖 RankFlow DP Worker 的默认内存池路径。旧的纯数字调用保持原行为；
  张量并行入口、其他队列、BMM 与无内存池模式没有因此获得驻留去重修复。
  旧后端缺少新增快照字段时，默认 DP 内存池路径会暂停新增准入，需要配套更新。
- 未改变 KV 峰值估计、严格小于容量的边界、batch token 上限、并发上限、
  beta、路由权重和淘汰选择策略。

新增测试：`benchmarks/tests/test_routing_experiment/test_adapter_memory_admission.py`，
共 **39 项 CPU 测试**，覆盖驻留/新增/闲置 adapter、混合 rank、共享复用、
容量边界、底层快照、查询失败和恢复、内部淘汰、预加载、清空、部分失败、
取消与两种方向的并发交错；旧接口与非内存池模式也有兼容测试。
更新了既有测试的 Worker 初始化字段、驻留快照及参数传递断言。

800-cell 复现中，已驻留 rank-64 adapter 只计 256 cells，KV 预算恢复为
544 cells，预计 KV 峰值 398 cells 的新请求可准入。总容量降至 654 cells 时，
KV 预算恰为 398 cells，仍按原严格边界拒绝；655 cells 时允许。

最终相关回归为 **267 passed, 12 deselected**，测试禁用 GPU 初始化，使用 CPU
数据与模拟 RPC。排除项均已在修复前版本确认失败：FIX-01 记录的 6 项，以及
`test_gpu_worker_reqqueue.py` 中另外 6 项（4 项没有有效推理模型仍期望正常响应，
2 项使用未配置 rank 的 `base` adapter）。新增快照接口所需的集成测试已更新，
未把接口变化导致的失败当作历史失败排除。

尚未执行 GPU 冒烟或性能对照，不能据此认定吞吐或延迟已经改善。
下一步可先联合验证 FIX-01/FIX-02，再推进 CHECK-01；CHECK-02 的 Router
缓存目录一致性仍为独立待核查项，本次准入不依赖该目录。

## GPU 验证准备（2026-11-03）

用户授权准备验证架构，本轮不启动 GPU 实验。新增 `dp-gpu-fix-validation`，
默认 GPU 0,1、两个 Worker、4096-cell 内存池；一次启动内依次完成冷加载、
驻留复用和混合 rank 容量边界验证。两卡共 36 个请求，各阶段等待全部完成后
继续，保留缓存，不作性能曲线数据。每阶段超时 120 秒。

- 修通 CLI 到实际配置的 GPU/worker 覆盖，GPU 列表可推断 worker 数；
  普通套件不显式覆盖时保留原来的 GPU 1,2,3。新增无启动副作用的 `--dry-run`。
- 只清理本 Runner 启动的进程组，不再全局匹配或强杀未知 GPU 孤儿进程。
  选定 GPU 被占用时拒绝启动；全局运行文件用互斥锁保护，发现已有 S-LoRA
  API 服务时拒绝并行运行。未知、脱离本进程组的残留不自动清理。
- Worker 验证日志默认关闭，验证套件自动打开，普通套件可用
  `--validation-log` 开启。逐请求生命周期、成功/拒绝准入预算与实际驻留快照
  分 Worker 保存；Router 单独记录实际接收并应用的 prefilling 上报。
- 验证不只检查请求成功或无 OOM：核对三阶段负载守恒、终止清零、Router
  prefilling 传递、驻留/新增集合去重、KV 预算和严格容量边界，并要求观察到
  真实内存拒绝及旧预算会拒绝、新预算允许且不受其他准入上限阻挡的候选。
  未覆盖为 `incomplete`，错误为 `failed`，均不作为成功返回。
- 保存请求计划、实际配置和关键源文件 SHA-256。成功套件跳过性能分析和作图；
  代码和预算见证不能替代修复前后相同协议的 GPU 性能对照。
- 已检查本机模型/adapter 配置和 CLI dry-run，未加载 GPU 模型。
  本轮最终相关 CPU 回归为 335 passed、6 deselected；排除的 6 项是 FIX-02
  已独立确认的 ReqQueue 历史失败。另一次更广的尝试停在旧进程管理测试：
  其模拟 Worker 不发送就绪信号，已终止该次测试，不将其计入通过结果。

运行命令、阶段和输出说明见 `benchmarks/GPU_FIX_VALIDATION.md`。FIX-02 与本轮
验证准备阶段尚未提交；后随 GPU 复验通过的代码与套件一并保存。
FIX-01 的独立提交保持不变。

## 首轮 GPU 验证与 FIX-03（2026-11-03）

用户运行 `dp-gpu-fix-validation`，GPU 0,1；诊断目录为
`benchmarks/routing_comparison_results/fix01_fix02_gpu_validation/run1/` 下的
`dp-gpu-fix-validation/diagnostics/server_20261103_172042_089730/`。
本次排查没有重新启动 GPU 模型，原始日志和失败 summary 保持不变。

### 实测结论

- 三阶段 HTTP 请求共 36 个，全部成功；两个 Worker 各接收和完成 18 个请求。
- 6920 个生命周期快照中记录到 19892 次贡献变化，全部为
  `request contribution changed across stages`，不是 19892 个失败请求。
- 同一请求接收时为 258 个输入 tokens、397 RWPT；prefill 后变为 259 个
  tokens、399 RWPT，decode 后继续增长。最大观测增量为 767 个 tokens：
  原始 514 变成 1281，rank-16 请求的 RWPT 从 583 变成 1454。
- 无缺失覆盖项；驻留复用候选检查 4191 次、新增预留检查 709 次、内存拒绝
  4862 次、旧预算误拒见证 9 次，Router 收到并应用非零 prefilling 上报 80 次。
  这些是重复采样/候选检查次数，不是独立请求数。
- 一次预算见证：池 4096 cells、驻留 384、新增预留 0、预计 KV 峰值 3711；
  新预算 3712 允许，旧预算 3328 拒绝，并且不受并发或 batch token 上限阻挡。
  这支持 FIX-02 的现场预算修正，不能证明性能改善或将整轮标记为通过。

### 根因与修复

单 GPU 的模型调用在 Worker 本地执行。`Req.to_rpc_obj()`、
`InferBatch.init_batch()` 和 `Req.prompt_ids` 原先共享同一列表；
`ModelRpcServer.forward()` 向 `all_input_ids` 追加生成 token，因而也修改了
原始 prompt。负载统计读取其动态长度，RWPT/Token Count 随 decode 增长；
响应中的 `req.prompt_ids + req.output_ids` 也有重复拼接输出的风险。

- `InferBatch.init_batch()` 为推理历史复制输入列表，推理仍正常累积生成 token，
  不再修改调用者 prompt。修改覆盖单 GPU 本地与 RPC 输入两种入口。
- Worker 的 waiting、prefilling、active、top-K 和验证记录统一读取构造请求时
  固定的 `req.input_len`；不改变 beta、w2、显存预算或路由公式。
- 验证记录额外保留 `prompt_buffer_len`，检查它等于原始输入长度；既有跨阶段
  贡献守恒检查不放宽。源文件 manifest 增加 InferBatch、model_rpc、io_struct。
- CPU 测试使用真实 `Req -> RPC 字典 -> InferBatch.init_batch`，仅将张量创建
  放到 CPU，模拟真实生成对 token 历史的追加。验证原始 prompt 不变、贡献
  守恒、完成后清零、响应不重复输出；另测三个阶段固定长度及污染检测。
  修复前复现失败，修复后相关回归 **341 passed、6 deselected**；仍为已确认
  的 6 项历史 ReqQueue 失败，未增加排除项。

首轮状态仍是 `failed`，没有通过重写旧日志或归一化旧贡献让它变成通过。
用户随后以同一配置运行 `run2`，结果见下节。该问题对历史性能数据的具体
影响尚未量化。
首轮诊断时 FIX-02、FIX-03 和验证套件尚未提交，随后完成下文 GPU 复验。

### run2：GPU 正确性验证通过

用户于 2026-11-03 执行修复后套件，诊断目录为
`benchmarks/routing_comparison_results/fix01_fix02_gpu_validation/run2/` 下的
`dp-gpu-fix-validation/diagnostics/server_20261103_173839_417643/`。
使用 GPU 0,1、两个 Worker、4096-cell 内存池，参数与三阶段协议不变。

- 原始 `validation_summary.json` 为 `passed`，errors 和 missing_coverage 均为空；
  以当前检查器只读重分析，完整 summary 与保存结果一致。
- cold_load 4/4、resident_reuse 8/8、capacity_boundary 24/24 请求成功；
  Worker 0 接收并完成 17 个，Worker 1 接收并完成 19 个，均最终排空。
- 6978 个生命周期快照通过请求集合守恒、逐请求贡献和上报总量检查；
  49058 个请求阶段样本都有 `prompt_buffer_len`，与原始 input_len 一致，
  没有 prompt 列表增长或输入长度漂移。
- 驻留复用检查 4253 次、新增预留检查 712 次、内存边界拒绝 4927 次，
  旧预算误拒候选 9 次。以上为候选检查/采样次数，不是失败请求数。
- 一次预算见证：Worker 0，池 4096、驻留 384、新增预留 0 cells；预计 KV
  峰值 3711，新预算 3712 允许，旧预算 3584 拒绝；并发和 batch token 上限
  均允许。新增预留、驻留集合去重及严格小于容量的校验均通过。
- Router 记录 72 个逐 worker 路由压力样本，实际接收并应用非零 prefilling
  上报 72 次；本套件的三阶段压力传递检查通过。
- manifest 的 13 份关键源码 SHA-256 均与检查时工作区一致。保留实际运行的
  日志、结果和 manifest，未修改原始 evidence，也未重新启动 GPU 模型。

本次通过范围是上述配置与覆盖场景的 GPU 正确性验证，不能推广为所有后端、
异常路径或性能均已验证。小内存池、长输出和各阶段突发发送刻意触发容量等待；
控制台的 0.38 req/s、平均延迟 36.349 s 不作为正式性能数据。
manifest 标记 `performance_result=false`，CLI 已跳过性能分析。

下一步先保存已验收修复的代码版本，再按计划处理 CHECK-01/CHECK-02；正式
性能对照需使用明确版本、相同硬件/trace/参数及统一缓存起始协议。
复验核对阶段仅更新验证记录，未新增代码修改或重跑 GPU 实验。经用户要求，
将 FIX-02、FIX-03、验证套件、相关测试和本记录一并提交；本次未推送远端。

## CHECK-01：乐观预增与状态上报的因果对齐

相关代码：

- `slora/server/router/dp_manager.py`：`_optimistically_update_worker_load()`、
  `_process_worker_states()`。
- `slora/server/router/adapter_aware_router.py`：`update_worker_state()`。

复查前版本派发会预增 Router 本地负载，后续上报则整体替换 Worker 状态。
如果某份上报尚未包含刚派发的请求，整体替换可能提前清除这部分预增。
现有 `report_seq` 是状态报告序号，不能单独证明某个派发请求已经被报告覆盖。

### 复现与实现（2026-11-03）

基于已提交的 `21ea7cc` 继续修改，没有启动 GPU、修改 beta/w2、显存准入或
主动复制策略。新测试先采集 queue=0 的状态，再派发一个 100-token、rank-64
请求，最后注入此前的状态；修复前实际把 queue=1 清成 0，修复后保留
queue=1、154 RWPT 和 1 个未确认请求。触发频率及性能影响仍未量化。

- 新增 `DispatchTracker`：分别保存不可被乐观预增修改的原始上报、每 Worker
  派发编号和未确认请求账本；条目记录 request_id、原始 tokens、RWPT、创建
  时间及发送结果是否不确定。评分状态由原始快照与尚未确认的贡献重新合成。
- 每次 Manager 启动生成 `routing_epoch` 并传给 Worker。普通 adapter-aware
  请求携带此标识及每 Worker 递增的 `dispatch_seq`；控制命令不占用派发编号。
  RR 保持原发送/选择语义，不使用乐观负载账本。
- Router 在发送 await 之前预记账；用协程锁保持并发调用的派发/发送顺序。
  Worker 将请求成功入队或明确拒绝后，同步推进 `ack_dispatch_seq`，与负载
  一起采集。`report_seq` 仍独立表示状态快照顺序。
- 仅接受同一会话、报告序号递增、确认水位不后退且不超出已分配编号的上报。
  只移除确认水位以内的条目，其余重新加到新快照上。重复/迟到/跨会话报告
  不更新 Router、cache 或 ReplicaManager。新版 adapter-aware 不静默回退到
  不带协议字段的旧 Worker；首次发现不兼容 Worker 时阻止向其派发。
- Router、状态缓存与 ReplicaManager 使用同一合成状态；原始上报单独保存。
  切换负载指标时从账本重算，重置统计或缓存不会清掉派发进度。
- ZMQ 明确未入队的 `Again` 撤销当次预留后最多重试三次，允许序号留下空号。
  其他发送异常/取消视为结果不确定：保留账目、不自动重发，阻止继续向该
  Worker 派发，直到后续上报确认交接。诊断日志故障不触发请求重发。
- 很快完成或失败的请求可以通过后续确认水位直接移交为零负载，不依赖
  Router 接收终止响应。采样失败不伪装为健康的空快照，不能据此清账。
  超过 10 秒仍未确认的请求在既有健康检查中告警，不按 TTL 删除账目。
  不新增可靠投递通道、自动热重启或完整取消机制；API Abort 不代表 GPU 已停止。

### 验证与下一步

新增 `benchmarks/tests/test_routing_experiment/test_dispatch_accounting.py`，共
40 项 CPU 检查，覆盖确定性交错、部分/全部确认、提前完成、两 Worker 独立
序号、发送等待期间确认、超时回滚与空号、未知结果不重试、并发顺序、协议
不兼容、序列化、采样失败和诊断故障；实际接收协程还覆盖无效请求的终止
错误响应和控制命令不推进确认水位。使用真实 Manager/Worker 的验证日志
测试对账分析器及损坏证据检测。相关回归 **381 passed、6 deselected**，仍只
排除此前独立确认的 6 项 ReqQueue 历史失败，没有新增排除项。

现有 `dp-gpu-fix-validation` 已增加派发/接收编号、原始与有效负载、确认水位
及待确认条目的日志。三个阶段不变，全部完成后最多再等 5 秒确认 Router
账本排空；失败或覆盖不足仍不能作为通过。新 manifest 标记
`dispatch_accounting=true`，并记录包括新增模块在内的 17 份关键源码。
summary 增加 `dispatch_accounting`，其中 `noncovering_reports_preserved` 表示
观察到未覆盖请求的报告且其贡献仍被保留的次数。若为零，不能声称该交错在
真实 GPU 运行中触发；对应顺序已由 CPU 确定性用例覆盖。

历史 run1/run2 原始日志和 summary 未修改；旧 run2 只证明当时 FIX-01/02/03
版本的既有套件通过，不证明新增 CHECK-01 已通过 GPU 验收。待用户使用新
输出目录运行扩展后的套件，命令见 `benchmarks/GPU_FIX_VALIDATION.md`。
本项实现、测试及说明随本次 CHECK-01 提交保存；GPU 复验仍待进行，未推送远端。

## CHECK-02：缓存命中信号是否反映 GPU 实际驻留

相关代码：

- `slora/server/router/gpu_worker.py`：`_load_adapters()`、`adapter_cache`、
  `_handle_finish_req()`。
- `slora/server/router/model_infer/infer_adapter.py`：`load_adapters()`、
  `check_and_evict_by_threshold()`。
- `slora/server/router/model_infer/model_rpc.py`：`exposed_load_adapters()`。

底层加载流程可以触发淘汰，但 Worker 的 `_load_adapters()` 主要向本地目录
添加新加载的 adapter。需确认底层加载触发的所有淘汰都能同步移除本地记录，
不能只检查请求完成时的显式淘汰路径。

先用模拟“加载 B 时淘汰 A”验证上报集合，再决定采用驻留集合快照还是变更回传。
区分 Router 观测命中与实际执行命中；不能把缓存元数据失真误判为缓存机制收益。
实际触发频率和对现有命中率数据的影响尚未量化。

## EXP-01：版本一致性与修正后基线

### 已确认的情况

- 现有主要 modelctx2048、9 RPS 结果的时间早于新 rank 系数代码提交。
- `45cbe6b` 于 2026-07-26 将运行时代价改为离线校准系数，
  当前默认 beta=0.00845。
- 已存在 `dp-realtrace-rwpt-active-9rps-w2-search-profiled` 搜索套件。
- 本轮检查的 `09_profiled_beta/rwpt_active_9rps_w2_search/` 目录下有服务器日志，
  未发现完整的 `results.jsonl`，不能据此宣布新系数搜索已完成。

### 后续要求

- 旧数据继续保留其历史版本身份，不当作当前系数版本的复测结果。
- 运行记录包含 commit、rank 系数、路由参数、trace、硬件和缓存初始状态。
- 修复影响准入等共享行为时，对照方法也需要使用修复后的共同执行基础。
- beta 改变负载尺度后，不能直接假定旧 w2 仍最优。
- 比较 TC Active 与 RWPT Active 时检查尺度和参数校准的公平性。
- 先在已有代表性负载点建立基线；若评价适用范围，仍需包含低负载与过载情况。
- 运行次数、服务器复用和启动成本在实际开跑前统一安排，不在本记录中启动实验。

## METHOD-01：批次准入感知的请求等待代价路由

### 当前方法的局限

当前主体评分为 `U_i = w_c * H_i - w_l * L_i / B`。
当 w_c=1、w_l=0.4、B=3072 时，缓存命中允许额外约 7680 个 RWPT 单位的负载差。
这个固定交换比例不直接反映 adapter 加载耗时、批次准入条件和显存阻塞。

现有结果提示缓存命中率与延迟不总同向：6 RPS 下命中率更高但 TTFT 更差，
9 RPS 下收益较明显，10 RPS 下 P90 TTFT 有退化。这些现象是研究动机，
尚不能证明某个具体原因具有因果作用。

### 推荐研究问题

对当前请求 x，哪个 Worker 能在可接受的系统代价下更早产生首 token？

保留 RWPT 作为代价估计的输入，研究下列条件对一次派发的共同影响：

1. 请求的 prompt 长度、adapter rank 和驻留状态。
2. 当前批次、等待请求及可用 KV/adapter 空间决定的准入机会。
3. 缓存未命中的实测加载时间，以及能否与现有执行重叠。
4. 派发对当前请求等待和已有请求执行的影响。

目标是用可解释的等待/加载时间比较建立路由决策依据，减少固定权重交换比例
对不同负载的依赖。避免将具有重叠关系的加载与执行时间直接重复相加。

### 需要避免的做法

- 不把增加评分项、换非线性函数或引入复杂学习模型本身当作创新。
- 不仅向每个候选 Worker 加同一个请求代价：同构线性模型中公共项会抵消，
  不会改变选择。要证明批次或显存条件使同一请求产生不同的边际等待代价。
- 不把完整 prompt RWPT 解释成精确剩余执行时间；active 中保留它目前只是代理信号。
- 不使用最终真实输出长度等在线决策不可得的信息。
- 不将简单串行队列模型直接当作连续批处理系统的精确 TTFT 模型。
- 不用同一份历史日志直接宣称替代路由的性能；改变派发会改变后续执行状态。
- 不预先宣称文献上的新颖性；预计延迟路由已有研究，应进一步核对相关工作。

### 进入实现前的验证门槛

- 先测量加载、排队、prefill 和准入受阻分别占多少时间。
- 验证 RWPT 相近的 Worker 是否因缓存/准入状态而产生显著不同的请求等待。
- 明确相对现有路由新增的状态、计算开销和适用边界。
- 只有证据支持后才实现最小版本，并与修正后的固定加权方法进行消融。
- 是否成为独立创新点，以机制、相关工作对比和实验结果决定。

## 本轮进展

- [x] 保存代码复查、CPU 复现结论与研究方向。
- [x] FIX-01：完成修复、CPU 回归及 run2 两卡 GPU 验证。
- [x] FIX-02：完成修复、CPU 回归及 run2 两卡 GPU 验证。
- [x] FIX-03：完成 prompt 隔离、CPU 回归及 run2 两卡 GPU 验证。
- [x] CHECK-01：完成 CPU 确定性复现、因果账本实现及相关回归。
- [ ] CHECK-01：完成扩展套件的 GPU 验证。
- [ ] CHECK-02：核查 GPU 驻留集合与上报缓存目录一致性。
- [ ] EXP-01：建立版本明确、修正后的实验基线。
- [ ] METHOD-01：完成测量与设计，再决定是否实现。

下一步为 CHECK-01 的 GPU 验证；其余核查、性能实验与方法研究尚未完成。
