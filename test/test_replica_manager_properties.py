"""
Property-Based Tests for Hot Adapter Replication (Phase 1)

Property 1: WorkerState 序列化往返（含 top_k_rwpt_adapters）
Property 2: Top-K RWPT 聚合正确性

Feature: hot-adapter-replication
Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import heapq
import time
import pytest
from collections import defaultdict
from hypothesis import given, strategies as st, settings, assume
from typing import List, Tuple, Dict

from slora.server.router.worker_state import WorkerState


# ─── Strategies ───────────────────────────────────────────────────────────────

adapter_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/_-'),
    min_size=1, max_size=50
).filter(lambda x: len(x.strip()) > 0)

# top_k_rwpt_adapters: list of (adapter_dir, contribution) tuples
top_k_entry_strategy = st.tuples(
    adapter_path_strategy,
    st.floats(min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False),
)

worker_state_with_top_k_strategy = st.builds(
    WorkerState,
    worker_id=st.integers(min_value=0, max_value=1000),
    cached_adapters=st.frozensets(adapter_path_strategy, min_size=0, max_size=10).map(set),
    queue_length=st.integers(min_value=0, max_value=1000),
    gpu_memory_free=st.integers(min_value=0, max_value=10**12),
    last_heartbeat=st.floats(min_value=0.0, max_value=10**10, allow_nan=False, allow_infinity=False),
    is_healthy=st.booleans(),
    avg_rank=st.floats(min_value=0.0, max_value=256.0, allow_nan=False, allow_infinity=False),
    min_rank=st.integers(min_value=0, max_value=256),
    max_rank=st.integers(min_value=0, max_value=256),
    pending_prefill_tokens=st.integers(min_value=0, max_value=10**7),
    pending_raw_tokens=st.integers(min_value=0, max_value=10**7),
    active_decode_seqs=st.integers(min_value=0, max_value=1000),
    pool_used_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    top_k_rwpt_adapters=st.lists(top_k_entry_strategy, min_size=0, max_size=10),
)


# ─── Property 1: WorkerState 序列化往返 ──────────────────────────────────────

class TestProperty1WorkerStateRoundTrip:
    """
    Feature: hot-adapter-replication, Property 1: WorkerState 序列化往返

    对于任意包含 top_k_rwpt_adapters 字段的 WorkerState 实例，
    to_dict() → from_dict() 应产生等价对象。
    缺失字段时 from_dict() 应返回空列表。

    Validates: Requirements 1.1, 1.3, 1.4
    """

    @given(state=worker_state_with_top_k_strategy)
    @settings(max_examples=200, deadline=None)
    def test_round_trip_preserves_top_k(self, state: WorkerState):
        """to_dict → from_dict 往返保留 top_k_rwpt_adapters"""
        serialized = state.to_dict()
        restored = WorkerState.from_dict(serialized)

        assert restored.worker_id == state.worker_id
        assert restored.cached_adapters == state.cached_adapters
        assert restored.queue_length == state.queue_length
        assert restored.is_healthy == state.is_healthy
        assert restored.pending_prefill_tokens == state.pending_prefill_tokens
        assert restored.pending_raw_tokens == state.pending_raw_tokens
        assert restored.active_decode_seqs == state.active_decode_seqs
        # top_k_rwpt_adapters: tuple 列表应完整保留
        assert len(restored.top_k_rwpt_adapters) == len(state.top_k_rwpt_adapters)
        for (a1, v1), (a2, v2) in zip(restored.top_k_rwpt_adapters, state.top_k_rwpt_adapters):
            assert a1 == a2
            assert abs(v1 - v2) < 1e-6

    @given(state=worker_state_with_top_k_strategy)
    @settings(max_examples=50, deadline=None)
    def test_missing_top_k_defaults_to_empty(self, state: WorkerState):
        """旧格式数据（无 top_k_rwpt_adapters）反序列化时返回空列表"""
        serialized = state.to_dict()
        del serialized['top_k_rwpt_adapters']
        restored = WorkerState.from_dict(serialized)
        assert restored.top_k_rwpt_adapters == []

    @given(state=worker_state_with_top_k_strategy)
    @settings(max_examples=50, deadline=None)
    def test_serialized_format_is_list_of_lists(self, state: WorkerState):
        """to_dict 将 tuples 序列化为 list of lists（JSON 兼容）"""
        serialized = state.to_dict()
        for item in serialized['top_k_rwpt_adapters']:
            assert isinstance(item, list)
            assert len(item) == 2


# ─── Property 2: Top-K RWPT 聚合正确性 ──────────────────────────────────────

def compute_top_k_rwpt(
    requests: List[Tuple[str, int, int]],  # [(adapter_dir, input_len, rank), ...]
    hidden_dim: int,
    k: int,
) -> List[Tuple[str, float]]:
    """
    Pure-function reimplementation of GPUWorker._compute_top_k_rwpt_adapters
    for property testing. Same algorithm, no GPUWorker dependency.
    """
    if not requests:
        return []
    gamma = 2.0 / (3.0 * hidden_dim)
    adapter_contrib: Dict[str, float] = {}
    for adapter_dir, input_len, rank in requests:
        contrib = input_len * (1.0 + gamma * rank)
        adapter_contrib[adapter_dir] = adapter_contrib.get(adapter_dir, 0.0) + contrib
    return heapq.nlargest(k, adapter_contrib.items(), key=lambda x: x[1])


def reference_top_k_rwpt(
    requests: List[Tuple[str, int, int]],
    hidden_dim: int,
    k: int,
) -> List[Tuple[str, float]]:
    """
    Reference implementation using naive sort (no heapq) for cross-validation.
    """
    if not requests:
        return []
    gamma = 2.0 / (3.0 * hidden_dim)
    adapter_contrib: Dict[str, float] = defaultdict(float)
    for adapter_dir, input_len, rank in requests:
        adapter_contrib[adapter_dir] += input_len * (1.0 + gamma * rank)
    sorted_items = sorted(adapter_contrib.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:k]


# Strategy: generate a request queue
request_strategy = st.tuples(
    adapter_path_strategy,                                    # adapter_dir
    st.integers(min_value=1, max_value=2048),                 # input_len
    st.integers(min_value=0, max_value=128),                  # rank
)

request_queue_strategy = st.lists(request_strategy, min_size=0, max_size=50)


class TestProperty2TopKRWPTAggregation:
    """
    Feature: hot-adapter-replication, Property 2: Top-K RWPT 聚合正确性

    对于任意等待队列，计算 top-K RWPT 贡献后：
    (a) 返回列表长度 ≤ K
    (b) 列表按 RWPT 贡献降序排列
    (c) 每个 adapter 的贡献值 = Σ input_len × (1 + γ × rank)
    (d) 返回的 adapter 集合是贡献最大的 K 个

    Validates: Requirements 1.2
    """

    @given(
        requests=request_queue_strategy,
        k=st.integers(min_value=1, max_value=10),
        hidden_dim=st.integers(min_value=1024, max_value=8192),
    )
    @settings(max_examples=200, deadline=None)
    def test_length_le_k(self, requests, k, hidden_dim):
        """(a) 返回列表长度 ≤ K"""
        result = compute_top_k_rwpt(requests, hidden_dim, k)
        assert len(result) <= k

    @given(
        requests=request_queue_strategy,
        k=st.integers(min_value=1, max_value=10),
        hidden_dim=st.integers(min_value=1024, max_value=8192),
    )
    @settings(max_examples=200, deadline=None)
    def test_descending_order(self, requests, k, hidden_dim):
        """(b) 列表按 RWPT 贡献降序排列"""
        result = compute_top_k_rwpt(requests, hidden_dim, k)
        for i in range(len(result) - 1):
            assert result[i][1] >= result[i + 1][1]

    @given(
        requests=request_queue_strategy,
        k=st.integers(min_value=1, max_value=10),
        hidden_dim=st.integers(min_value=1024, max_value=8192),
    )
    @settings(max_examples=200, deadline=None)
    def test_contribution_values_correct(self, requests, k, hidden_dim):
        """(c) 每个 adapter 的贡献值计算正确"""
        result = compute_top_k_rwpt(requests, hidden_dim, k)
        gamma = 2.0 / (3.0 * hidden_dim)

        # 手动计算期望贡献
        expected: Dict[str, float] = defaultdict(float)
        for adapter_dir, input_len, rank in requests:
            expected[adapter_dir] += input_len * (1.0 + gamma * rank)

        for adapter_dir, contrib in result:
            assert abs(contrib - expected[adapter_dir]) < 1e-6

    @given(
        requests=request_queue_strategy,
        k=st.integers(min_value=1, max_value=10),
        hidden_dim=st.integers(min_value=1024, max_value=8192),
    )
    @settings(max_examples=200, deadline=None)
    def test_top_k_are_largest(self, requests, k, hidden_dim):
        """(d) 返回的 adapter 是贡献最大的 K 个"""
        result = compute_top_k_rwpt(requests, hidden_dim, k)
        ref = reference_top_k_rwpt(requests, hidden_dim, k)

        # 两种实现应返回相同的 adapter 集合和贡献值
        assert len(result) == len(ref)
        for (a1, v1), (a2, v2) in zip(result, ref):
            assert a1 == a2
            assert abs(v1 - v2) < 1e-6

    @given(
        requests=request_queue_strategy,
        hidden_dim=st.integers(min_value=1024, max_value=8192),
    )
    @settings(max_examples=100, deadline=None)
    def test_length_equals_min_k_unique_adapters(self, requests, hidden_dim):
        """返回长度 = min(K, unique_adapter_count)"""
        k = 5
        result = compute_top_k_rwpt(requests, hidden_dim, k)
        unique_adapters = len(set(r[0] for r in requests))
        assert len(result) == min(k, unique_adapters)

    def test_empty_queue_returns_empty(self):
        """空队列返回空列表"""
        assert compute_top_k_rwpt([], 4096, 5) == []


# ─── Property 10: 副本保护阻止淘汰 ─────────────────────────────────────────

import unittest.mock as mock


def _make_minimal_infer_adapter(adapter_dirs, protected_replicas=None):
    """
    Create a minimal InferAdapter-like object for testing protection logic.
    We mock out CUDA-dependent parts and only test the eviction candidate selection.
    """
    from unittest.mock import MagicMock
    from slora.server.router.model_infer.infer_adapter import InferAdapter
    import torch

    n = len(adapter_dirs)
    # Create a mock mem_manager
    mem_manager = MagicMock()
    mem_manager.tot_size = 10000

    adapter = InferAdapter(
        adapter_dirs=list(adapter_dirs),
        a_loc=torch.zeros(n * 4, dtype=torch.long),      # dummy
        a_start=torch.arange(0, n * 4, 4, dtype=torch.long),
        a_len=torch.full((n,), 4, dtype=torch.long),
        a_scaling=torch.ones(n, dtype=torch.float16),
        mem_manager=mem_manager,
        idx_map={d: i for i, d in enumerate(adapter_dirs)},
        prefetch_tag={},
        cur_tag=0,
        prefetch_stream=None,
        adapter_scores={d: float(i) for i, d in enumerate(adapter_dirs)},  # ascending scores
        score_update_counter={d: 1 for d in adapter_dirs},
        usage_timestamps={d: [time.time()] for d in adapter_dirs},
        usage_window=300.0,
        last_access_time={d: time.time() for d in adapter_dirs},
        current_request_count={d: 0 for d in adapter_dirs},
        load_time={d: time.time() for d in adapter_dirs},
        pending_adapter_counts={d: 0 for d in adapter_dirs},
        protected_replicas=dict(protected_replicas) if protected_replicas else {},
    )
    return adapter


class TestProperty10ReplicaProtectionBlocksEviction:
    """
    Feature: hot-adapter-replication, Property 10: 副本保护阻止淘汰

    对于任意处于保护期内的 adapter，select_eviction_candidates() 的返回列表
    不应包含该 adapter。保护期到期后，该 adapter 应可被正常淘汰。

    Validates: Requirements 2.4, 8.1, 8.2, 8.3
    """

    def test_protected_adapter_not_evicted(self):
        """保护期内的 adapter 不出现在淘汰候选中"""
        dirs = ["adapter_A", "adapter_B", "adapter_C", "adapter_D"]
        # adapter_A and adapter_B are protected (expire in 60s)
        protected = {
            "adapter_A": time.time() + 60,
            "adapter_B": time.time() + 60,
        }
        ia = _make_minimal_infer_adapter(dirs, protected)
        candidates = ia.select_eviction_candidates(evict_ratio=1.0)
        assert "adapter_A" not in candidates
        assert "adapter_B" not in candidates

    def test_expired_protection_allows_eviction(self):
        """保护期过期后 adapter 可被淘汰"""
        dirs = ["adapter_A", "adapter_B", "adapter_C"]
        # adapter_A protection already expired
        protected = {
            "adapter_A": time.time() - 1.0,
        }
        ia = _make_minimal_infer_adapter(dirs, protected)
        candidates = ia.select_eviction_candidates(evict_ratio=1.0)
        # adapter_A should be evictable now
        assert "adapter_A" in candidates

    def test_mixed_protected_and_unprotected(self):
        """混合场景：部分保护、部分不保护"""
        dirs = ["adapter_A", "adapter_B", "adapter_C", "adapter_D"]
        protected = {
            "adapter_B": time.time() + 60,  # protected
            "adapter_D": time.time() - 1,   # expired
        }
        ia = _make_minimal_infer_adapter(dirs, protected)
        candidates = ia.select_eviction_candidates(evict_ratio=1.0)
        assert "adapter_B" not in candidates
        # adapter_D expired, should be evictable
        assert "adapter_D" in candidates

    def test_all_protected_returns_empty(self):
        """所有 adapter 都在保护期内时，淘汰候选为空"""
        dirs = ["adapter_A", "adapter_B"]
        protected = {
            "adapter_A": time.time() + 60,
            "adapter_B": time.time() + 60,
        }
        ia = _make_minimal_infer_adapter(dirs, protected)
        candidates = ia.select_eviction_candidates(evict_ratio=1.0)
        assert candidates == []

    def test_no_protection_normal_eviction(self):
        """无保护时正常淘汰"""
        dirs = ["adapter_A", "adapter_B", "adapter_C"]
        ia = _make_minimal_infer_adapter(dirs)
        candidates = ia.select_eviction_candidates(evict_ratio=1.0)
        assert len(candidates) == 3

    @given(
        num_adapters=st.integers(min_value=1, max_value=20),
        num_protected=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_protected_never_in_candidates(self, num_adapters, num_protected):
        """Property: 保护期内的 adapter 永远不出现在淘汰候选中"""
        num_protected = min(num_protected, num_adapters)
        dirs = [f"adapter_{i}" for i in range(num_adapters)]
        protected = {
            dirs[i]: time.time() + 60
            for i in range(num_protected)
        }
        ia = _make_minimal_infer_adapter(dirs, protected)
        candidates = ia.select_eviction_candidates(evict_ratio=1.0)
        protected_set = set(protected.keys())
        for c in candidates:
            assert c not in protected_set

    def test_get_protected_count_cleans_expired(self):
        """get_protected_count 自动清理过期条目"""
        dirs = ["adapter_A", "adapter_B", "adapter_C"]
        protected = {
            "adapter_A": time.time() + 60,   # valid
            "adapter_B": time.time() - 1,     # expired
            "adapter_C": time.time() - 10,    # expired
        }
        ia = _make_minimal_infer_adapter(dirs, protected)
        count = ia.get_protected_count()
        assert count == 1
        assert "adapter_A" in ia.protected_replicas
        assert "adapter_B" not in ia.protected_replicas
        assert "adapter_C" not in ia.protected_replicas

    def test_add_and_remove_protection(self):
        """add_protection 和 remove_protection 基本功能"""
        dirs = ["adapter_A", "adapter_B"]
        ia = _make_minimal_infer_adapter(dirs)
        assert ia.get_protected_count() == 0

        ia.add_protection("adapter_A", 60.0)
        assert ia.get_protected_count() == 1

        ia.remove_protection("adapter_A")
        assert ia.get_protected_count() == 0
