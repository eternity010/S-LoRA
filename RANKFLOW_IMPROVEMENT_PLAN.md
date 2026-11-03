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
- 本轮进行代码复查、FIX-01 实现和 CPU 回归验证，没有启动 GPU 实验。
- 先修正实现，再建立当前版本的有效基线，随后只推进一个主要方法改进方向。
- 不预设修复必然提高性能，也不通过选择有利负载点来代替完整验证。
- 热点 adapter 主动复制暂缓研究，保留原型代码；本轮没有修改其开关或运行行为。

## 处理顺序

| 顺序 | 编号 | 事项 | 证据状态 | 处理状态 |
|---|---|---|---|---|
| 1 | FIX-01 | 补齐 adapter 加载及 prefill 阶段的负载可见性 | CPU 模拟复现并通过修复验证 | 代码完成，待 GPU 验证 |
| 2 | FIX-02 | 修正批次准入中已驻留 adapter 的显存重复计数 | CPU 模拟已复现错误拒绝 | 待处理 |
| 3 | CHECK-01 | 验证乐观预增与 Worker 上报的因果对齐 | 代码路径存在风险，影响未量化 | 待核查 |
| 4 | CHECK-02 | 核对 Router 缓存目录与 GPU 实际驻留集合 | 代码路径存在风险，触发频率未量化 | 待核查 |
| 5 | EXP-01 | 区分旧系数结果，建立修正后版本基线 | 已发现版本与数据时间差异 | 待处理 |
| 6 | METHOD-01 | 评估批次准入感知的请求等待代价路由 | 方法假设，尚未实现或验证 | 待设计 |

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
测试和本记录作为 FIX-01 独立提交保存，未推送远端。下一步仍需单独处理
FIX-02，再安排明确版本的 GPU 对照实验。

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

## CHECK-01：乐观预增与状态上报的因果对齐

相关代码：

- `slora/server/router/dp_manager.py`：`_optimistically_update_worker_load()`、
  `_process_worker_states()`。
- `slora/server/router/adapter_aware_router.py`：`update_worker_state()`。

当前派发会预增 Router 本地负载，后续上报则整体替换 Worker 状态。
如果某份上报尚未包含刚派发的请求，整体替换可能提前清除这部分预增。
现有 `report_seq` 是状态报告序号，不能单独证明某个派发请求已经被报告覆盖。

先构造“Worker 采集状态 -> Router 派发 -> Router 收到该状态”的最小复现，
再判断是否需要保留尚未被确认覆盖的派发账目，以及 Worker 接收序号/水位。
验收重点是迟到报告不擦除未被观察的请求、已覆盖请求不重复统计，
并处理请求完成与失败的清理。该项不是已证明的性能瓶颈。

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
- [x] FIX-01：完成加载及 prefill 阶段统计修复与 CPU 回归，GPU 验证待进行。
- [ ] FIX-02：修正 adapter 占用重复计数。
- [ ] CHECK-01：核查状态上报与乐观预增对齐。
- [ ] CHECK-02：核查 GPU 驻留集合与上报缓存目录一致性。
- [ ] EXP-01：建立版本明确、修正后的实验基线。
- [ ] METHOD-01：完成测量与设计，再决定是否实现。

下一项为 FIX-02；其余核查、实验与方法研究尚未完成。
