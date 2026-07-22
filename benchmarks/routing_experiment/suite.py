"""
Experiment suite module for predefined experiment configurations.
"""

from typing import List, Iterator, Dict, Any
from itertools import product

from .config import ExperimentConfig


class ExperimentSuite:
    """Experiment suite definition and configuration generator"""
    
    # Predefined experiment suites
    SUITES: Dict[str, Dict[str, List[Any]]] = {
        "rwpt-w2-search": {
            # RWPT 专用 w2 甜点搜索（移除 decode 项后）
            # load-metric-w2-sweep 结果显示 rwpt 在 w2=0.8 时最优（5.21/20.5s）
            # 但仍差于 queue_length/token_count，需要在更小 w2 范围细搜
            # RWPT/Capacity 值域 ~[0, 0.5]（纯 prefill token，无 decode 折算）
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [120],
            "routing_w1": [1.0],
            "routing_w2": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            "load_metric": ["rwpt"],
        },
        "ql-w2-search": {
            # queue_length 专用 w2 甜点搜索
            # queue_length 模式无 Capacity 归一化：Score -= w2 * QueueLen
            # QueueLen 通常 0~5，cache 命中得分 = w1 = 1.0
            # 要让负载惩罚与缓存亲和性竞争：w2 * QueueLen ≈ 1.0
            # → QueueLen=3 时 w2≈0.3 为平衡点
            # 搜索范围 w2 ∈ [0.05, 0.8]，覆盖"几乎纯缓存"到"强负载均衡"
            # 10 experiments
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [120],
            "routing_w1": [1.0],
            "routing_w2": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "load_metric": ["queue_length"],
        },
        "tc-w2-search": {
            # token_count 专用 w2 甜点搜索
            # token_count 模式：pending_raw_tokens / Capacity（无 rank 加权，无 decode 折算）
            # 与 rwpt 归一化方式相同（除以 batch_max_tokens≈2500），但不乘 (1+γ·r)
            # 值域与 rwpt 接近，预期甜点区间也接近
            # rwpt 甜点 w2∈[0.5, 2.5]，token_count 搜索范围覆盖 [0.3, 4.0]
            # 10 experiments
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [120],
            "routing_w1": [1.0],
            "routing_w2": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "load_metric": ["token_count"],
        },
        "dp-realtrace-w2-search": {
            # 真实 trace 下的 RWPT 小范围 w2 搜索
            # 目标：在保持真实请求到达序列不变的前提下，验证 w2 是否存在更合适的甜点位
            # 只保留 adapter-aware + rwpt；round-robin 不受 w2 影响，不纳入本套件
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps",
            ],
            "routing_w1": [1.0],
            "routing_w2": [1.0, 2.0, 3.0, 4.0],
            "load_metric": ["rwpt"],
        },
        "dp-realtrace-rwpt-w2-fine-search": {
            # Refine the post-state-fix RWPT optimum and probe beyond w2=4.0.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps",
            ],
            "routing_w1": [1.0],
            "routing_w2": [3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
            "load_metric": ["rwpt"],
        },
        "dp-realtrace-rwpt-active-w2-search-6rps": {
            # Calibrate active-request RWPT at the medium-load operating point.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-6rps-rwpt-active-w2-search"],
            "routing_w1": [1.0],
            "routing_w2": [0.5, 1.0, 1.5, 2.0],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-rwpt-active-w2-low-search-6rps": {
            # Refine active-request RWPT below the current w2=0.5 candidate.
            # Descending order gives w2=0.4 the fresh-server control point.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps-rwpt-active-w2-low-search",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4, 0.3, 0.2, 0.1],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-rwpt-active-w2-1-fresh-6rps": {
            # Fresh-server control for the w2 search ordering effect.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-6rps-rwpt-active-w2-1-fresh"],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-queue-length-w2-search": {
            # Calibrate queue_length on the same 6 rps real trace used for RWPT.
            # Queue length is not capacity-normalized, so its w2 scale is smaller.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.05, 0.10, 0.20, 0.30],
            "load_metric": ["queue_length"],
        },
        "dp-realtrace-token-count-w2-search": {
            # Calibrate token_count on the same 6 rps real trace used for RWPT.
            # token_count is capacity-normalized, so search around the RWPT scale.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps",
            ],
            "routing_w1": [1.0],
            "routing_w2": [2.0, 3.0, 4.0, 5.0],
            "load_metric": ["token_count"],
        },
        "dp-realtrace-token-count-state-debug": {
            # Short single-point run for validating worker state propagation.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [60],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_60s_capped2048_512_debug.jsonl",
            ],
            "workload_name": ["azure-http-top100-6rps-60s-debug"],
            "routing_w1": [1.0],
            "routing_w2": [3.0],
            "load_metric": ["token_count"],
        },
        "dp-realtrace-token-count-8rps-debug": {
            # Diagnose token_count blind spots under the 8 rps real trace.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-8rps-token-count-debug"],
            "routing_w1": [1.0],
            "routing_w2": [3.0],
            "load_metric": ["token_count"],
        },
        "dp-realtrace-token-count-8rps-repeat2": {
            # Repeat the calibrated 8 rps token_count point twice on one server.
            # Distinct workload names keep checkpoint identities independent.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-8rps-token-count-run1",
                "azure-http-top100-8rps-token-count-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [3.0],
            "load_metric": ["token_count"],
        },
        "dp-realtrace-queue-length-8rps-standalone": {
            # Fresh-server 8 rps validation without debug I/O or server reuse.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-8rps-queue-length-standalone"],
            "routing_w1": [1.0],
            "routing_w2": [0.20],
            "load_metric": ["queue_length"],
        },
        "dp-realtrace-queue-length-8rps-repeat2": {
            # Repeat the calibrated 8 rps QL point twice on one server.
            # Distinct workload names keep checkpoint identities independent.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-8rps-queue-length-run1",
                "azure-http-top100-8rps-queue-length-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.20],
            "load_metric": ["queue_length"],
        },
        "dp-realtrace-rwpt-8rps-no-decay-standalone": {
            # Isolate removal of RWPT cache-affinity decay on a fresh server.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-8rps-rwpt-no-decay-standalone"],
            "routing_w1": [1.0],
            "routing_w2": [3.0],
            "load_metric": ["rwpt"],
        },
        "dp-realtrace-rwpt-8rps-repeat2": {
            # Repeat the selected 8 rps RWPT point twice on one server.
            # Cache affinity is fixed; the removed decay mechanism is not active.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-8rps-rwpt-run1",
                "azure-http-top100-8rps-rwpt-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [3.0],
            "load_metric": ["rwpt"],
        },
        "dp-realtrace-roundrobin-8rps-repeat2": {
            # Run the same 8 rps round-robin baseline twice on one server.
            # Distinct workload names keep checkpoint identities independent.
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-8rps-roundrobin-run1",
                "azure-http-top100-8rps-roundrobin-run2",
            ],
        },
        "dp-realtrace-roundrobin-4rps-repeat2": {
            # Run 1 uses a fresh server; run 2 follows confirmed cache reset.
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [4.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_4rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-4rps-roundrobin-run1",
                "azure-http-top100-4rps-roundrobin-run2",
            ],
        },
        "dp-realtrace-rwpt-4rps-repeat2": {
            # Run 1 uses a fresh server; run 2 follows confirmed cache reset.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [4.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_4rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-4rps-rwpt-run1",
                "azure-http-top100-4rps-rwpt-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [3.0],
            "load_metric": ["rwpt"],
        },
        "dp-realtrace-rwpt-active-4rps-repeat2": {
            # Validate active-request RWPT at the low-load endpoint.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [4.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_4rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-4rps-rwpt-active-w2-0p4-run1",
                "azure-http-top100-4rps-rwpt-active-w2-0p4-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-roundrobin-6rps": {
            # 真实 trace 下的 round-robin 单点基线
            # 用于和 dp-realtrace-w2-search 的 6 rps RWPT 结果做同环境对比
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps",
            ],
        },
        "dp-realtrace-roundrobin-6rps-repeat2": {
            # Run 1 uses a fresh server; run 2 follows confirmed cache reset.
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps-roundrobin-run1",
                "azure-http-top100-6rps-roundrobin-run2",
            ],
        },
        "dp-realtrace-rwpt-6rps-repeat2": {
            # Run 1 uses a fresh server; run 2 follows confirmed cache reset.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps-rwpt-run1",
                "azure-http-top100-6rps-rwpt-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [3.0],
            "load_metric": ["rwpt"],
        },
        "dp-realtrace-rwpt-active-6rps-repeat2": {
            # Final medium-load validation for the selected active-request RWPT.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps-rwpt-active-w2-0p4-run1",
                "azure-http-top100-6rps-rwpt-active-w2-0p4-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-roundrobin-6rps-modelctx2048": {
            # Cross-system baseline trace: actual dummy prompt + output fits 2048.
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-6rps-modelctx2048-roundrobin"],
        },
        "dp-realtrace-rwpt-active-6rps-modelctx2048": {
            # Cross-system baseline trace using the selected final routing metric.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-6rps-modelctx2048-rwpt-active"],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-roundrobin-8rps-modelctx2048": {
            # Higher-pressure cross-system trace with a fresh round-robin server.
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-8rps-modelctx2048-roundrobin"],
        },
        "dp-realtrace-rwpt-active-8rps-modelctx2048": {
            # Higher-pressure cross-system trace using the selected RWPT Active metric.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-8rps-modelctx2048-rwpt-active"],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-roundrobin-9rps-modelctx2048": {
            # Locate the pressure crossover between the 8 and 10 RPS traces.
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [9.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_9rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-9rps-modelctx2048-roundrobin"],
        },
        "dp-realtrace-rwpt-9rps-modelctx2048": {
            # Waiting-only RWPT control for the active-request ablation.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [9.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_9rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-9rps-modelctx2048-rwpt"],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["rwpt"],
        },
        "dp-realtrace-rwpt-active-9rps-modelctx2048": {
            # RWPT Active at the intermediate 9 RPS pressure point.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [9.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_9rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-9rps-modelctx2048-rwpt-active"],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-token-count-9rps-modelctx2048": {
            # Waiting-only Token Count control for the active-request ablation.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [9.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_9rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-9rps-modelctx2048-token-count",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["token_count"],
        },
        "dp-realtrace-token-count-active-9rps-modelctx2048": {
            # Token Count Active ablation at the selected 9 RPS pressure point.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [9.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_9rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-9rps-modelctx2048-token-count-active",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["token_count_active"],
        },
        "dp-realtrace-queue-length-9rps-modelctx2048": {
            # Queue Length ablation at the selected 9 RPS pressure point.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [9.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_9rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-9rps-modelctx2048-queue-length",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.2],
            "load_metric": ["queue_length"],
        },
        "dp-realtrace-roundrobin-10rps-modelctx2048": {
            # Overload-point cross-system trace with round-robin routing.
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [10.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_10rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-10rps-modelctx2048-roundrobin"],
        },
        "dp-realtrace-rwpt-active-10rps-modelctx2048": {
            # Overload-point cross-system trace using RWPT Active.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [10.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_10rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": ["azure-http-top100-10rps-modelctx2048-rwpt-active"],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-rwpt-active-6rps-modelctx2048-w2-0p5": {
            # Recalibrate RWPT Active after enforcing the 2048-token model context.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps-modelctx2048-rwpt-active-w2-0p5",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.5],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-rwpt-active-6rps-modelctx2048-w2-coarse-search": {
            # Probe stronger load penalties on the cross-system 2048-token trace.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_modelctx2048_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps-modelctx2048-rwpt-active-w2-coarse-search",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.6, 0.8, 1.0],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-token-count-active-6rps-repeat2": {
            # Active-request token-count ablation using the RWPT Active weight.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-6rps-token-count-active-w2-0p4-run1",
                "azure-http-top100-6rps-token-count-active-w2-0p4-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["token_count_active"],
        },
        "dp-realtrace-token-count-active-8rps-repeat2": {
            # High-load active-request token-count ablation.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-8rps-token-count-active-w2-0p4-run1",
                "azure-http-top100-8rps-token-count-active-w2-0p4-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["token_count_active"],
        },
        "dp-realtrace-token-count-active-10rps-repeat2": {
            # Overload-point active-request token-count validation.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [10.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_10rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-10rps-token-count-active-w2-0p4-run1",
                "azure-http-top100-10rps-token-count-active-w2-0p4-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["token_count_active"],
        },
        "dp-realtrace-token-count-active-8rps-rank-swapped-repeat2": {
            # Rank ablation: preserve the trace but map the hottest adapters to rank 64.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_rank_swapped_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-8rps-rank-swapped-token-count-active-run1",
                "azure-http-top100-8rps-rank-swapped-token-count-active-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["token_count_active"],
        },
        "dp-realtrace-rwpt-active-8rps-rank-swapped-repeat2": {
            # Same rank-swapped trace as token-count-active for a controlled ablation.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_rank_swapped_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-8rps-rank-swapped-rwpt-active-run1",
                "azure-http-top100-8rps-rank-swapped-rwpt-active-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-rwpt-active-8rps-repeat2": {
            # Validate active-request RWPT at the high-load endpoint.
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-8rps-rwpt-active-w2-0p4-run1",
                "azure-http-top100-8rps-rwpt-active-w2-0p4-run2",
            ],
            "routing_w1": [1.0],
            "routing_w2": [0.4],
            "load_metric": ["rwpt_active"],
        },
        "dp-realtrace-rwpt-4rps-debug": {
            # 单点诊断：复现真实 trace 4 rps 下的 RWPT 路由偏斜。
            # 运行时配合 --debug，将逐请求路由决策写入 routing_debug/。
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [4.0],
            "duration": [180],
            "workload_type": ["trace"],
            "trace_file": [
                "real_workload/outputs/azure_llm_http_top100_4rps_180s_capped2048_512_v1.jsonl",
            ],
            "workload_name": [
                "azure-http-top100-4rps",
            ],
            "routing_w1": [1.0],
            "routing_w2": [3.0],
            "load_metric": ["rwpt"],
        },
        # R-LoRA round-robin 基线（与 S-LoRA 基线参数对齐）
        # DP=3 round-robin，用于对比 DP 架构本身的收益（无智能路由）
        # 3 alphas = 3 experiments
        "dp-roundrobin-baseline": {
            "routing_strategy": ["round-robin"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
        },

        # 端到端性能基线对比：固定 alpha=0.3，改变请求到达率
        # 用于绘制 round-robin 的吞吐-延迟曲线
        # 5 request rates = 5 experiments
        "dp-roundrobin-rate-scaling": {
            "routing_strategy": ["round-robin"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [2.0, 4.0, 6.0, 8.0],
            "duration": [180],
        },

        # 单点复核 round-robin 在强热点 8 req/s 下的高压表现
        "dp-roundrobin-rate8-validation": {
            "routing_strategy": ["round-robin"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
        },

        # 单点复核 round-robin 在中等强热点 8 req/s 下的高压表现
        "dp-roundrobin-rate8-alpha02-validation": {
            "routing_strategy": ["round-robin"],
            "alpha": [0.2],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
        },

        # 单点复核 round-robin 在中等热点 8 req/s 下的高压表现
        "dp-roundrobin-rate8-alpha03-validation": {
            "routing_strategy": ["round-robin"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
        },

        # 端到端性能基线对比：固定 alpha=0.3，改变请求到达率
        # 与 dp-roundrobin-rate-scaling 完全对齐，仅切换为 adapter-aware + rwpt
        # 3 request rates = 3 experiments
        "dp-rwpt-rate-scaling": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [2.0, 4.0, 6.0, 8.0],
            "duration": [180],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
        },

        # 验证强热点 8 req/s 下 RWPT 尾延迟表现
        # 固定 alpha=0.1、req_rate=8.0，仅保留当前主线 w2
        # 1 experiment
        "dp-rwpt-rate8-w2-search": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
        },

        # 验证中等强热点 8 req/s 下 RWPT 尾延迟表现
        # 固定 alpha=0.2、req_rate=8.0，仅保留当前主线 w2
        # 1 experiment
        "dp-rwpt-rate8-alpha02-validation": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.2],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
        },

        # 验证中等热点 8 req/s 下 RWPT 尾延迟表现
        # 固定 alpha=0.3、req_rate=8.0，仅保留当前主线 w2
        # 1 experiment
        "dp-rwpt-rate8-alpha03-validation": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.3],
            "num_adapters": [100],
            "req_rate": [8.0],
            "duration": [180],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
        },

        # 图 6 / 系统基线对比用的 RWPT 对照组
        # 与 dp-roundrobin-baseline 保持相同 alpha、req_rate、duration，仅切换为 adapter-aware + rwpt
        # 3 alphas = 3 experiments
        "dp-rwpt-baseline": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
        },

        # 图 6 / 系统基线对比用的 token_count 对照组
        # 与 dp-rwpt-baseline 保持相同 alpha、req_rate、duration，仅切换为 token_count + w2=3.0
        # 3 alphas = 3 experiments
        "dp-tokencount-baseline": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
            "routing_w1": [1.0],
            "routing_w2": [3.0],
            "load_metric": ["token_count"],
        },

        # 图 6 / 系统基线对比用的 queue_length 对照组
        # 与 dp-rwpt-baseline 保持相同 alpha、req_rate、duration，仅切换为 queue_length + w2=0.20
        # 3 alphas = 3 experiments
        "dp-queuelength-baseline": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
            "routing_w1": [1.0],
            "routing_w2": [0.20],
            "load_metric": ["queue_length"],
        },

        # 热门 Adapter 主动复制对比实验
        # 对照组 (enable_replication=False) vs 实验组 (enable_replication=True)
        # 在三种 Zipf 分布下对比 P90 TTFT、缓存命中率、吞吐量
        # 2 strategies × 3 alphas = 6 experiments
        "replication-comparison": {
            "routing_strategy": ["adapter-aware"],
            "alpha": [0.1, 0.3, 0.8],
            "num_adapters": [100],
            "req_rate": [6.0],
            "duration": [240],
            "routing_w1": [1.0],
            "routing_w2": [1.0],
            "load_metric": ["rwpt"],
            "enable_replication": [False, True],
        },
    }
    
    @classmethod
    def get_configs(cls, suite_name: str, **defaults) -> Iterator[ExperimentConfig]:
        """
        Generate all experiment configurations for a given suite.
        
        Args:
            suite_name: Name of the predefined suite
            **defaults: Default values for parameters not specified in suite
        
        Yields:
            ExperimentConfig objects for each parameter combination
        
        Raises:
            ValueError: If suite_name is not found
        """
        # Special suites with non-Cartesian configs
        if suite_name == "alpha-robustness":
            yield from cls.get_alpha_robustness_configs(**defaults)
            return
        if suite_name == "alpha-robustness-sla":
            yield from cls.get_alpha_robustness_sla_configs(**defaults)
            return
        if suite_name == "dp-realtrace-comparison":
            yield from cls.get_realtrace_comparison_configs(**defaults)
            return
        if suite_name == "dp-realtrace-comparison-2rps":
            yield from cls.get_realtrace_2rps_comparison_configs(**defaults)
            return
        if suite_name == "dp-realtrace-load-metric-comparison":
            yield from cls.get_realtrace_load_metric_comparison_configs(**defaults)
            return
        if suite_name == "dp-realtrace-8rps-load-metric-comparison":
            yield from cls.get_realtrace_8rps_load_metric_comparison_configs(**defaults)
            return
        if suite_name == "dp-realtrace-w2-search":
            yield from cls.get_realtrace_w2_search_configs(**defaults)
            return
        
        suite = cls.SUITES.get(suite_name)
        if suite is None:
            available = ", ".join(
                list(cls.SUITES.keys())
                + ["alpha-robustness", "alpha-robustness-sla", "dp-realtrace-comparison", "dp-realtrace-comparison-2rps", "dp-realtrace-load-metric-comparison", "dp-realtrace-8rps-load-metric-comparison", "dp-realtrace-w2-search"]
            )
            raise ValueError(
                f"Unknown suite: {suite_name}. Available suites: {available}"
            )
        
        # Get parameter names and their value lists
        keys = list(suite.keys())
        values = [suite[k] for k in keys]
        
        # Generate Cartesian product of all parameter combinations
        for combo in product(*values):
            params = dict(zip(keys, combo))
            
            # Merge with defaults
            config_params = {**defaults, **params}
            
            yield ExperimentConfig(**config_params)
    
    @classmethod
    def list_suites(cls) -> List[str]:
        """
        List all available predefined suites.

        Returns:
            List of suite names
        """
        return list(cls.SUITES.keys()) + [
            "alpha-robustness",
            "alpha-robustness-sla",
            "dp-realtrace-comparison",
            "dp-realtrace-comparison-2rps",
            "dp-realtrace-load-metric-comparison",
            "dp-realtrace-8rps-load-metric-comparison",
            "dp-realtrace-w2-search",
        ]
    
    @classmethod
    def get_suite_info(cls, suite_name: str) -> Dict[str, Any]:
        """
        Get information about a specific suite.
        
        Args:
            suite_name: Name of the suite
        
        Returns:
            Dictionary containing suite parameters and expected config count
        
        Raises:
            ValueError: If suite_name is not found
        """
        # Special suites
        if suite_name == "alpha-robustness":
            return {
                "name": suite_name,
                "parameters": {
                    "alpha": [0.1, 0.3, 0.8],
                    "metric_w2_pairs": [
                        ("rwpt", 0.5), ("token_count", 0.3), ("queue_length", 0.15),
                    ],
                },
                "config_count": 9,
            }
        if suite_name == "alpha-robustness-sla":
            return {
                "name": suite_name,
                "parameters": {
                    "alpha": [0.1, 0.3, 0.8],
                    "metric_w2_pairs": [
                        ("rwpt", 1.0), ("token_count", 3.0), ("queue_length", 0.20),
                    ],
                },
                "config_count": 9,
            }
        if suite_name == "dp-realtrace-comparison":
            return {
                "name": suite_name,
                "parameters": {
                    "routing_strategy": ["round-robin", "adapter-aware"],
                    "workload_name": ["azure-http-top100-4rps"],
                    "routing_w2": [3.0],
                    "num_adapters": [100],
                    "duration": [180],
                    "load_metric": ["rwpt"],
                },
                "config_count": 2,
            }
        if suite_name == "dp-realtrace-comparison-2rps":
            return {
                "name": suite_name,
                "parameters": {
                    "routing_strategy": ["round-robin", "adapter-aware"],
                    "workload_name": ["azure-http-top100-2rps"],
                    "routing_w2": [3.0],
                    "num_adapters": [100],
                    "duration": [180],
                    "load_metric": ["rwpt"],
                },
                "config_count": 2,
            }
        if suite_name == "dp-realtrace-load-metric-comparison":
            return {
                "name": suite_name,
                "parameters": {
                    "routing_strategy": ["adapter-aware"],
                    "workload_name": ["azure-http-top100-6rps"],
                    "metric_w2_pairs": [
                        ("queue_length", 0.20),
                        ("token_count", 3.0),
                        ("rwpt", 3.0),
                    ],
                    "num_adapters": [100],
                    "duration": [180],
                },
                "config_count": 3,
            }
        if suite_name == "dp-realtrace-8rps-load-metric-comparison":
            return {
                "name": suite_name,
                "parameters": {
                    "routing_strategy": ["round-robin", "adapter-aware"],
                    "workload_name": ["azure-http-top100-8rps"],
                    "strategy_metric_w2": [
                        ("round-robin", "rwpt", 1.0),
                        ("adapter-aware", "rwpt", 3.0),
                        ("adapter-aware", "token_count", 3.0),
                        ("adapter-aware", "queue_length", 0.20),
                    ],
                    "num_adapters": [100],
                    "duration": [180],
                },
                "config_count": 4,
            }
        if suite_name == "dp-realtrace-w2-search":
            return {
                "name": suite_name,
                "parameters": {
                    "routing_strategy": ["adapter-aware"],
                    "workload_name": ["azure-http-top100-6rps"],
                    "routing_w2": [1.0, 2.0, 3.0, 4.0],
                    "num_adapters": [100],
                    "duration": [180],
                    "load_metric": ["rwpt"],
                },
                "config_count": 4,
            }
        
        suite = cls.SUITES.get(suite_name)
        if suite is None:
            available = ", ".join(
                list(cls.SUITES.keys())
                + ["alpha-robustness", "alpha-robustness-sla", "dp-realtrace-comparison", "dp-realtrace-comparison-2rps", "dp-realtrace-load-metric-comparison", "dp-realtrace-8rps-load-metric-comparison", "dp-realtrace-w2-search"]
            )
            raise ValueError(
                f"Unknown suite: {suite_name}. Available suites: {available}"
            )
        
        # Calculate expected number of configurations
        config_count = 1
        for values in suite.values():
            config_count *= len(values)
        
        return {
            "name": suite_name,
            "parameters": suite,
            "config_count": config_count,
        }
    
    @classmethod
    def add_custom_suite(cls, suite_name: str, suite_def: Dict[str, List[Any]]) -> None:
        """
        Add a custom experiment suite.
        
        Args:
            suite_name: Name for the new suite
            suite_def: Dictionary mapping parameter names to value lists
        
        Raises:
            ValueError: If suite_name already exists
        """
        if suite_name in cls.SUITES:
            raise ValueError(f"Suite '{suite_name}' already exists")
        
        # Validate that required parameters are present
        required_params = {"routing_strategy", "alpha", "num_adapters", "req_rate", "duration"}
        provided_params = set(suite_def.keys())
        
        if not required_params.issubset(provided_params):
            missing = required_params - provided_params
            raise ValueError(f"Missing required parameters: {missing}")
        
        cls.SUITES[suite_name] = suite_def

    @classmethod
    def get_alpha_robustness_configs(cls, **defaults):
        """
        Alpha 鲁棒性实验（旧版，加权评分 w2）：固定每个 metric 的最优 w2，变化 alpha。
        3 metrics × 3 alphas = 9 experiments.

        w2 来源：早期加权评分选取（已被 SLA 版本替代，保留供对比）
          rwpt:         w2=0.5
          token_count:  w2=0.3
          queue_length: w2=0.15
        """
        metric_w2_pairs = [
            ("rwpt", 0.5),
            ("token_count", 0.3),
            ("queue_length", 0.15),
        ]
        alphas = [0.1, 0.3, 0.8]

        for alpha in alphas:
            for load_metric, w2 in metric_w2_pairs:
                yield ExperimentConfig(
                    routing_strategy="adapter-aware",
                    alpha=alpha,
                    num_adapters=100,
                    req_rate=6.0,
                    duration=120,
                    routing_w1=1.0,
                    routing_w2=w2,
                    routing_w3=0.0,
                    load_metric=load_metric,
                    **defaults,
                )

    @classmethod
    def get_realtrace_comparison_configs(cls, **defaults):
        traces = [
            (
                "azure-http-top100-4rps",
                4.0,
                "real_workload/outputs/azure_llm_http_top100_4rps_180s_capped2048_512_v1.jsonl",
            ),
        ]
        strategies = ["round-robin", "adapter-aware"]

        for workload_name, req_rate, trace_file in traces:
            for strategy in strategies:
                params = {
                    "routing_strategy": strategy,
                    "alpha": 0.1,
                    "num_adapters": 100,
                    "req_rate": req_rate,
                    "duration": 180,
                    "workload_type": "trace",
                    "trace_file": trace_file,
                    "workload_name": workload_name,
                    "routing_w1": 1.0,
                    "routing_w2": 3.0 if strategy == "adapter-aware" else 1.0,
                    "load_metric": "rwpt",
                }
                yield ExperimentConfig(**{**defaults, **params})

    @classmethod
    def get_realtrace_2rps_comparison_configs(cls, **defaults):
        trace_file = (
            "real_workload/outputs/"
            "azure_llm_http_top100_2rps_180s_capped2048_512_v1.jsonl"
        )
        for strategy in ("round-robin", "adapter-aware"):
            params = {
                "routing_strategy": strategy,
                "alpha": 0.1,
                "num_adapters": 100,
                "req_rate": 2.0,
                "duration": 180,
                "workload_type": "trace",
                "trace_file": trace_file,
                "workload_name": "azure-http-top100-2rps",
                "routing_w1": 1.0,
                "routing_w2": 3.0 if strategy == "adapter-aware" else 1.0,
                "load_metric": "rwpt",
            }
            yield ExperimentConfig(**{**defaults, **params})

    @classmethod
    def get_realtrace_load_metric_comparison_configs(cls, **defaults):
        trace_file = (
            "real_workload/outputs/"
            "azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl"
        )
        metric_w2_pairs = [
            ("queue_length", 0.20),
            ("token_count", 3.0),
            ("rwpt", 3.0),
        ]

        for load_metric, w2 in metric_w2_pairs:
            params = {
                "routing_strategy": "adapter-aware",
                "alpha": 0.1,
                "num_adapters": 100,
                "req_rate": 6.0,
                "duration": 180,
                "workload_type": "trace",
                "trace_file": trace_file,
                "workload_name": "azure-http-top100-6rps",
                "routing_w1": 1.0,
                "routing_w2": w2,
                "load_metric": load_metric,
            }
            yield ExperimentConfig(**{**defaults, **params})

    @classmethod
    def get_realtrace_8rps_load_metric_comparison_configs(cls, **defaults):
        trace_file = (
            "real_workload/outputs/"
            "azure_llm_http_top100_8rps_180s_capped2048_512_v1.jsonl"
        )
        strategy_metric_w2 = [
            ("round-robin", "rwpt", 1.0),
            ("adapter-aware", "rwpt", 3.0),
            ("adapter-aware", "token_count", 3.0),
            ("adapter-aware", "queue_length", 0.20),
        ]

        for strategy, load_metric, w2 in strategy_metric_w2:
            params = {
                "routing_strategy": strategy,
                "alpha": 0.1,
                "num_adapters": 100,
                "req_rate": 8.0,
                "duration": 180,
                "workload_type": "trace",
                "trace_file": trace_file,
                "workload_name": "azure-http-top100-8rps",
                "routing_w1": 1.0,
                "routing_w2": w2,
                "load_metric": load_metric,
            }
            yield ExperimentConfig(**{**defaults, **params})

    @classmethod
    def get_realtrace_w2_search_configs(cls, **defaults):
        traces = [
            (
                "azure-http-top100-6rps",
                6.0,
                "real_workload/outputs/azure_llm_http_top100_6rps_180s_capped2048_512_v1.jsonl",
            ),
        ]
        w2_values = [1.0, 2.0, 3.0, 4.0]

        for workload_name, req_rate, trace_file in traces:
            for w2 in w2_values:
                params = {
                    "routing_strategy": "adapter-aware",
                    "alpha": 0.1,
                    "num_adapters": 100,
                    "req_rate": req_rate,
                    "duration": 180,
                    "workload_type": "trace",
                    "trace_file": trace_file,
                    "workload_name": workload_name,
                    "routing_w1": 1.0,
                    "routing_w2": w2,
                    "load_metric": "rwpt",
                }
                yield ExperimentConfig(**{**defaults, **params})

    @classmethod
    def get_alpha_robustness_sla_configs(cls, **defaults):
        """
        Alpha 鲁棒性实验：固定每个 metric 的当前主线 w2，变化 alpha。
        3 metrics × 3 alphas = 9 experiments.

        token_count 和 queue_length 的 w2 来源于 6 rps 真实 trace 校准。
        当前主线设置：
          rwpt:         w2=1.0   (P90=14.14s, tput=5.553, cache=72.1%)
          token_count:  w2=3.0
          queue_length: w2=0.20
        """
        metric_w2_pairs = [
            ("rwpt", 1.0),
            ("token_count", 3.0),
            ("queue_length", 0.20),
        ]
        alphas = [0.1, 0.3, 0.8]

        for alpha in alphas:
            for load_metric, w2 in metric_w2_pairs:
                yield ExperimentConfig(
                    routing_strategy="adapter-aware",
                    alpha=alpha,
                    num_adapters=100,
                    req_rate=6.0,
                    duration=240,
                    routing_w1=1.0,
                    routing_w2=w2,
                    routing_w3=0.0,
                    load_metric=load_metric,
                    **defaults,
                )
