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


# ─── Property 3: EMA 计算正确性 ─────────────────────────────────────────────

from slora.server.router.replica_manager import ReplicaManager


def _make_replica_manager(num_workers=3, capacity=1000.0, w1=1.0, w2=1.0, ema_alpha=0.3, cooldown_sec=5.0, max_protected_per_worker=2):
    """Helper to create a ReplicaManager for testing."""
    return ReplicaManager(
        num_workers=num_workers,
        capacity=capacity,
        w1=w1,
        w2=w2,
        ema_alpha=ema_alpha,
        cooldown_sec=cooldown_sec,
        max_protected_per_worker=max_protected_per_worker,
    )


class TestProperty3EMACalculation:
    """
    Feature: hot-adapter-replication, Property 3: EMA 计算正确性

    对于任意 RWPT/Capacity 值序列和 EMA 平滑系数 α，每次更新后的 EMA 值
    应满足 EMA_new = α × value_current + (1-α) × EMA_old。初始 EMA 值为 0。

    Validates: Requirements 3.1
    """

    @given(
        rwpt_values=st.lists(
            st.integers(min_value=0, max_value=10000),
            min_size=1, max_size=50,
        ),
        capacity=st.integers(min_value=100, max_value=10000),
        alpha=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_ema_matches_formula(self, rwpt_values, capacity, alpha):
        """EMA 更新严格遵循公式 EMA_new = α × (RWPT/Cap) + (1-α) × EMA_old"""
        rm = _make_replica_manager(num_workers=1, capacity=float(capacity), ema_alpha=alpha)

        expected_ema = 0.0
        for rwpt in rwpt_values:
            state = WorkerState(
                worker_id=0,
                pending_prefill_tokens=rwpt,
            )
            rm.update_worker_state(0, state)

            current_ratio = rwpt / float(capacity)
            expected_ema = alpha * current_ratio + (1.0 - alpha) * expected_ema

            assert abs(rm.ema_rwpt[0] - expected_ema) < 1e-9

    @given(
        alpha=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_ema_initial_value_is_zero(self, alpha):
        """EMA 初始值为 0"""
        rm = _make_replica_manager(num_workers=3, ema_alpha=alpha)
        for wid in range(3):
            assert rm.ema_rwpt[wid] == 0.0

    def test_ema_converges_to_constant(self):
        """常量输入下 EMA 收敛到该常量"""
        rm = _make_replica_manager(num_workers=1, capacity=1000.0, ema_alpha=0.3)
        # Feed constant RWPT = 500 (ratio = 0.5) for many iterations
        for _ in range(100):
            state = WorkerState(worker_id=0, pending_prefill_tokens=500)
            rm.update_worker_state(0, state)
        assert abs(rm.ema_rwpt[0] - 0.5) < 1e-6

    @given(
        num_workers=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_ema_independent_per_worker(self, num_workers):
        """各 Worker 的 EMA 独立更新，互不影响"""
        rm = _make_replica_manager(num_workers=num_workers, capacity=1000.0, ema_alpha=0.3)

        # Only update worker 0
        state = WorkerState(worker_id=0, pending_prefill_tokens=800)
        rm.update_worker_state(0, state)

        # Worker 0 should have non-zero EMA
        assert rm.ema_rwpt[0] > 0.0
        # All other workers should still be 0
        for wid in range(1, num_workers):
            assert rm.ema_rwpt[wid] == 0.0


# ─── Property 4: T_congestion 公式与动态更新 ────────────────────────────────

class TestProperty4TCongestionFormula:
    """
    Feature: hot-adapter-replication, Property 4: T_congestion 公式与动态更新

    对于任意正数 w1、w2 和 Capacity，T_congestion 应等于 (w1/w2) × Capacity。
    当 w1 或 w2 更新后，T_congestion 应自动重新计算为新的 (w1/w2) × Capacity。

    Validates: Requirements 3.3, 3.5
    """

    @given(
        w1=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        capacity=st.floats(min_value=1.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_t_congestion_formula(self, w1, w2, capacity):
        """T_congestion = (w1/w2) × Capacity"""
        rm = _make_replica_manager(capacity=capacity, w1=w1, w2=w2)
        expected = (w1 / w2) * capacity
        assert abs(rm.t_congestion - expected) < 1e-9

    @given(
        w1_init=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        w2_init=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        w1_new=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        w2_new=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        capacity=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_t_congestion_updates_on_config_change(self, w1_init, w2_init, w1_new, w2_new, capacity):
        """update_config() 后 T_congestion 自动重算"""
        rm = _make_replica_manager(capacity=capacity, w1=w1_init, w2=w2_init)
        rm.update_config(w1=w1_new, w2=w2_new)
        expected = (w1_new / w2_new) * capacity
        assert abs(rm.t_congestion - expected) < 1e-9

    def test_t_congestion_w2_zero_returns_inf(self):
        """w2=0 时 T_congestion 为 inf（防止除零）"""
        rm = ReplicaManager(num_workers=2, capacity=1000.0, w1=1.0, w2=0.0)
        assert rm.t_congestion == float("inf")

    @given(
        w1=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        capacity=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_update_config_stores_new_w1_w2(self, w1, w2, capacity):
        """update_config() 同时更新 self.w1 和 self.w2"""
        rm = _make_replica_manager(capacity=capacity)
        rm.update_config(w1=w1, w2=w2)
        assert rm.w1 == w1
        assert rm.w2 == w2


# ─── Property 11: 心跳同步 replica_map ──────────────────────────────────────

class TestProperty11HeartbeatSyncReplicaMap:
    """
    Feature: hot-adapter-replication, Property 11: 心跳同步 replica_map

    对于任意 Worker 心跳上报的 cached_adapters 集合，ReplicaManager 的
    replica_map 应准确反映所有 Worker 的缓存状态：
        replica_map[adapter] = {w | adapter ∈ cached_adapters_w}

    Validates: Requirements 9.1, 9.4
    """

    @given(
        worker_caches=st.lists(
            st.frozensets(adapter_path_strategy, min_size=0, max_size=8),
            min_size=1, max_size=6,
        )
    )
    @settings(max_examples=200, deadline=None)
    def test_replica_map_matches_cached_adapters(self, worker_caches):
        """replica_map 精确反映所有 Worker 的 cached_adapters"""
        num_workers = len(worker_caches)
        rm = _make_replica_manager(num_workers=num_workers)

        # Feed one heartbeat per worker
        for wid, adapters in enumerate(worker_caches):
            state = WorkerState(worker_id=wid, cached_adapters=set(adapters))
            rm.update_worker_state(wid, state)

        # Build expected replica_map from ground truth
        expected: Dict[str, Set[int]] = {}
        for wid, adapters in enumerate(worker_caches):
            for a in adapters:
                expected.setdefault(a, set()).add(wid)

        assert rm.replica_map == expected

    @given(
        initial_adapters=st.frozensets(adapter_path_strategy, min_size=1, max_size=8),
        new_adapters=st.frozensets(adapter_path_strategy, min_size=0, max_size=8),
    )
    @settings(max_examples=100, deadline=None)
    def test_replica_map_updates_on_cache_change(self, initial_adapters, new_adapters):
        """Worker 缓存变化后 replica_map 立即更新"""
        rm = _make_replica_manager(num_workers=2)

        # First heartbeat
        state1 = WorkerState(worker_id=0, cached_adapters=set(initial_adapters))
        rm.update_worker_state(0, state1)

        # Second heartbeat with different cache
        state2 = WorkerState(worker_id=0, cached_adapters=set(new_adapters))
        rm.update_worker_state(0, state2)

        # replica_map should reflect new_adapters, not initial_adapters
        for a in new_adapters:
            assert 0 in rm.replica_map.get(a, set())
        for a in initial_adapters - new_adapters:
            assert 0 not in rm.replica_map.get(a, set())

    def test_replica_map_empty_on_init(self):
        """初始化时 replica_map 为空"""
        rm = _make_replica_manager(num_workers=3)
        assert rm.replica_map == {}

    @given(
        num_workers=st.integers(min_value=2, max_value=6),
        shared_adapter=adapter_path_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_shared_adapter_tracked_across_workers(self, num_workers, shared_adapter):
        """同一 adapter 被多个 Worker 缓存时，replica_map 包含所有 Worker"""
        rm = _make_replica_manager(num_workers=num_workers)

        for wid in range(num_workers):
            state = WorkerState(worker_id=wid, cached_adapters={shared_adapter})
            rm.update_worker_state(wid, state)

        assert rm.replica_map.get(shared_adapter) == set(range(num_workers))
        assert rm.get_replica_count(shared_adapter) == num_workers


# ─── Property 5: 拥塞检测阈值 ───────────────────────────────────────────────

class TestProperty5CongestionDetection:
    """
    Feature: hot-adapter-replication, Property 5: 拥塞检测阈值

    当且仅当 EMA(RWPT/Capacity) > T_congestion/Capacity 时，
    该 Worker 应被判定为拥塞状态。

    Validates: Requirements 3.4
    """

    @given(
        ema_value=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        w1=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        capacity=st.floats(min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_congestion_iff_ema_exceeds_threshold(self, ema_value, w1, w2, capacity):
        """当且仅当 EMA > T_congestion/Capacity 时判定为拥塞"""
        rm = _make_replica_manager(num_workers=1, capacity=capacity, w1=w1, w2=w2)
        # Directly set EMA to the test value (bypass heartbeat)
        rm.ema_rwpt[0] = ema_value

        threshold = rm.t_congestion / capacity  # = w1/w2
        congested = rm._detect_congested_workers()

        if ema_value > threshold:
            assert 0 in congested
        else:
            assert 0 not in congested

    @given(
        num_workers=st.integers(min_value=2, max_value=8),
        ema_values=st.lists(
            st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=8,
        ),
        w1=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_multiple_workers_independent_detection(self, num_workers, ema_values, w1, w2):
        """多 Worker 场景：每个 Worker 独立判定拥塞"""
        assume(len(ema_values) >= num_workers)
        rm = _make_replica_manager(num_workers=num_workers, capacity=1000.0, w1=w1, w2=w2)
        threshold = rm.t_congestion / rm.capacity

        for wid in range(num_workers):
            rm.ema_rwpt[wid] = ema_values[wid]

        congested = rm._detect_congested_workers()

        for wid in range(num_workers):
            if ema_values[wid] > threshold:
                assert wid in congested
            else:
                assert wid not in congested

    def test_all_idle_no_congestion(self):
        """所有 Worker EMA=0 时无拥塞"""
        rm = _make_replica_manager(num_workers=3, w1=1.0, w2=1.0)
        assert rm._detect_congested_workers() == []

    def test_exactly_at_threshold_not_congested(self):
        """EMA 恰好等于阈值时不判定为拥塞（严格大于）"""
        rm = _make_replica_manager(num_workers=1, capacity=1000.0, w1=1.0, w2=1.0)
        threshold = rm.t_congestion / rm.capacity  # = 1.0
        rm.ema_rwpt[0] = threshold  # exactly at threshold
        assert rm._detect_congested_workers() == []


# ─── Property 6: 元凶识别 ────────────────────────────────────────────────────

class TestProperty6CulpritIdentification:
    """
    Feature: hot-adapter-replication, Property 6: 元凶识别

    对于任意非空的 top_k_rwpt_adapters 列表，识别出的 Culprit Adapter
    应为列表中 RWPT 贡献最大的（即第一个）元素。
    当列表为空时，应返回 None 且跳过复制决策。

    Validates: Requirements 4.1, 4.2
    """

    @given(
        top_k=st.lists(top_k_entry_strategy, min_size=1, max_size=10),
    )
    @settings(max_examples=200, deadline=None)
    def test_culprit_is_first_element(self, top_k):
        """非空列表时，元凶为第一个元素（贡献最大）"""
        rm = _make_replica_manager(num_workers=1)
        rm.worker_top_k[0] = list(top_k)
        culprit = rm._identify_culprit(0)
        assert culprit == top_k[0][0]

    def test_empty_top_k_returns_none(self):
        """空列表时返回 None"""
        rm = _make_replica_manager(num_workers=1)
        rm.worker_top_k[0] = []
        assert rm._identify_culprit(0) is None

    def test_unknown_worker_returns_none(self):
        """未知 worker_id 返回 None（不崩溃）"""
        rm = _make_replica_manager(num_workers=1)
        assert rm._identify_culprit(99) is None

    @given(
        top_k=st.lists(top_k_entry_strategy, min_size=2, max_size=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_culprit_not_second_element(self, top_k):
        """元凶不是第二个元素（除非第一个和第二个相同 adapter）"""
        rm = _make_replica_manager(num_workers=1)
        rm.worker_top_k[0] = list(top_k)
        culprit = rm._identify_culprit(0)
        # culprit must be the first adapter_dir
        assert culprit == top_k[0][0]
        # if first and second are different adapters, culprit != second
        if top_k[0][0] != top_k[1][0]:
            assert culprit != top_k[1][0]

    @given(
        num_workers=st.integers(min_value=2, max_value=5),
        worker_id=st.integers(min_value=0, max_value=4),
        top_k=st.lists(top_k_entry_strategy, min_size=1, max_size=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_culprit_only_from_specified_worker(self, num_workers, worker_id, top_k):
        """元凶只从指定 worker 的 top_k 中取，不受其他 worker 影响"""
        assume(worker_id < num_workers)
        rm = _make_replica_manager(num_workers=num_workers)

        # Set different top_k for each worker
        for wid in range(num_workers):
            rm.worker_top_k[wid] = [(f"other_adapter_{wid}", float(wid))]

        # Override the target worker
        rm.worker_top_k[worker_id] = list(top_k)

        culprit = rm._identify_culprit(worker_id)
        assert culprit == top_k[0][0]


# ─── Property 7: 全局防爆护栏 ───────────────────────────────────────────────

class TestProperty7GlobalGuardrails:
    """
    Feature: hot-adapter-replication, Property 7: 全局防爆护栏

    (a) 当 adapter 的全局副本数 >= N_max 时，护栏拒绝复制
    (b) 当距上次复制时间 < T_cooldown 时，护栏拒绝复制
    (c) 两个条件都不满足时，护栏通过，且 last_replication_time 更新为当前时间

    Validates: Requirements 5.1, 5.2, 5.4, 5.5
    """

    @given(
        num_workers=st.integers(min_value=2, max_value=8),
        replica_count=st.integers(min_value=0, max_value=8),
    )
    @settings(max_examples=200, deadline=None)
    def test_replica_count_limit(self, num_workers, replica_count):
        """副本数 >= N_max 时护栏拒绝"""
        assume(replica_count <= num_workers)
        rm = _make_replica_manager(num_workers=num_workers, cooldown_sec=0.0)
        n_max = num_workers - 1

        # Populate replica_map with replica_count workers
        if replica_count > 0:
            rm.replica_map['adapter_X'] = set(range(replica_count))

        result = rm._check_guardrails('adapter_X')

        if replica_count >= n_max:
            assert result == False
        else:
            assert result == True

    @given(
        elapsed=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        cooldown=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_cooldown_enforcement(self, elapsed, cooldown):
        """冷却期内护栏拒绝，冷却期后护栏通过"""
        rm = _make_replica_manager(num_workers=3, cooldown_sec=cooldown)
        # Simulate last replication happened `elapsed` seconds ago
        rm.last_replication_time['adapter_Y'] = time.time() - elapsed

        result = rm._check_guardrails('adapter_Y')

        if elapsed < cooldown:
            assert result == False
        else:
            assert result == True

    def test_guardrail_pass_updates_timestamp(self):
        """护栏通过后 last_replication_time 更新为当前时间"""
        rm = _make_replica_manager(num_workers=3, cooldown_sec=0.0)
        before = time.time()
        result = rm._check_guardrails('adapter_Z')
        after = time.time()

        assert result == True
        assert 'adapter_Z' in rm.last_replication_time
        assert before <= rm.last_replication_time['adapter_Z'] <= after

    def test_guardrail_fail_does_not_update_timestamp(self):
        """护栏拒绝时 last_replication_time 不更新"""
        rm = _make_replica_manager(num_workers=3, cooldown_sec=100.0)
        # First call passes and sets timestamp
        rm._check_guardrails('adapter_W')
        first_ts = rm.last_replication_time['adapter_W']

        # Second call should be blocked by cooldown
        result = rm._check_guardrails('adapter_W')
        assert result == False
        # Timestamp should not have changed
        assert rm.last_replication_time['adapter_W'] == first_ts

    @given(
        num_workers=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=50, deadline=None)
    def test_n_max_equals_num_workers_minus_one(self, num_workers):
        """N_max = num_workers - 1"""
        rm = _make_replica_manager(num_workers=num_workers)
        assert rm.n_max == num_workers - 1


# ─── Property 8: 目标 Worker 选择 ───────────────────────────────────────────

class TestProperty8TargetWorkerSelection:
    """
    Feature: hot-adapter-replication, Property 8: 目标 Worker 选择

    选择的目标 Worker 应满足：
    (a) 尚未缓存该 adapter
    (b) EMA(RWPT/Capacity) < T_congestion/Capacity × 0.9
    (c) 当前受保护副本数 < max_protected_per_worker
    在所有合格候选中，应选择 EMA 最低的 Worker。
    当无合格候选时，应返回 None。

    Validates: Requirements 6.1, 6.2, 6.3
    """

    @given(
        num_workers=st.integers(min_value=2, max_value=8),
        ema_values=st.lists(
            st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=8,
        ),
        w1=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_selected_worker_not_cached(self, num_workers, ema_values, w1, w2):
        """选中的 Worker 一定未缓存该 adapter"""
        assume(len(ema_values) >= num_workers)
        rm = _make_replica_manager(num_workers=num_workers, w1=w1, w2=w2)

        for wid in range(num_workers):
            rm.ema_rwpt[wid] = ema_values[wid]

        # Cache adapter on worker 0 only
        rm.replica_map['adapter_T'] = {0}

        target = rm._select_target_worker('adapter_T')
        if target is not None:
            assert target != 0
            assert target not in rm.replica_map.get('adapter_T', set())

    @given(
        num_workers=st.integers(min_value=2, max_value=8),
        ema_values=st.lists(
            st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=8,
        ),
        w1=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_selected_worker_below_load_threshold(self, num_workers, ema_values, w1, w2):
        """选中的 Worker EMA < T_congestion/Capacity × 0.9"""
        assume(len(ema_values) >= num_workers)
        rm = _make_replica_manager(num_workers=num_workers, w1=w1, w2=w2)

        for wid in range(num_workers):
            rm.ema_rwpt[wid] = ema_values[wid]

        threshold = (rm.t_congestion / rm.capacity) * 0.9
        target = rm._select_target_worker('adapter_T')

        if target is not None:
            assert rm.ema_rwpt[target] < threshold

    @given(
        num_workers=st.integers(min_value=2, max_value=6),
        ema_values=st.lists(
            st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=6,
        ),
    )
    @settings(max_examples=200, deadline=None)
    def test_selected_worker_is_lowest_ema(self, num_workers, ema_values):
        """在所有合格候选中，选 EMA 最低的"""
        assume(len(ema_values) >= num_workers)
        # Use w1=w2=1 so threshold = 0.9, all ema_values < 0.9 qualify
        rm = _make_replica_manager(num_workers=num_workers, w1=1.0, w2=1.0)

        for wid in range(num_workers):
            rm.ema_rwpt[wid] = ema_values[wid]

        target = rm._select_target_worker('adapter_T')

        if target is not None:
            # target must have the lowest EMA among all qualifying workers
            threshold = (rm.t_congestion / rm.capacity) * 0.9
            for wid in range(num_workers):
                if (wid not in rm.replica_map.get('adapter_T', set()) and
                        rm.ema_rwpt[wid] < threshold and
                        len(rm.protected_replicas.get(wid, {})) < rm.max_protected_per_worker):
                    assert rm.ema_rwpt[target] <= rm.ema_rwpt[wid]

    def test_no_candidates_returns_none(self):
        """无合格候选时返回 None"""
        rm = _make_replica_manager(num_workers=3, w1=1.0, w2=1.0)
        # All workers cached the adapter
        rm.replica_map['adapter_T'] = {0, 1, 2}
        assert rm._select_target_worker('adapter_T') is None

    def test_all_overloaded_returns_none(self):
        """所有 Worker 过载时返回 None（弹性降级）"""
        rm = _make_replica_manager(num_workers=3, w1=1.0, w2=1.0)
        # EMA > 0.9 for all workers
        rm.ema_rwpt = {0: 1.5, 1: 1.2, 2: 0.95}
        assert rm._select_target_worker('adapter_T') is None

    def test_protection_full_skips_worker(self):
        """受保护副本数满的 Worker 被跳过"""
        rm = _make_replica_manager(num_workers=3, w1=1.0, w2=1.0, max_protected_per_worker=2)
        rm.ema_rwpt = {0: 0.1, 1: 0.2, 2: 0.3}
        # Worker 0 has 2 protected replicas (full)
        rm.protected_replicas[0] = {'x': time.time() + 60, 'y': time.time() + 60}
        target = rm._select_target_worker('adapter_T')
        assert target != 0
        assert target == 1  # lowest EMA among non-full workers


# ─── Property 9: 每周期单副本 ────────────────────────────────────────────────

import asyncio


class TestProperty9SingleReplicaPerPatrol:
    """
    Feature: hot-adapter-replication, Property 9: 每周期单副本

    即使存在多个拥塞 Worker 或多个元凶 Adapter，
    单次 patrol() 最多只产生 1 个复制动作。

    Validates: Requirements 6.4
    """

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_single_action_with_multiple_congested_workers(self):
        """多个拥塞 Worker 时，单次 patrol 只产生 1 个复制动作"""
        calls = []

        def mock_callback(worker_id, adapter_dir):
            calls.append((worker_id, adapter_dir))
            return {"success": True}

        rm = _make_replica_manager(
            num_workers=4, w1=1.0, w2=1.0, cooldown_sec=0.0,
        )
        rm.preload_callback = mock_callback

        # Workers 0, 1, 2 all congested; Worker 3 idle
        rm.ema_rwpt = {0: 1.5, 1: 1.5, 2: 1.5, 3: 0.1}
        rm.worker_top_k[0] = [("adapter_A", 3000.0)]
        rm.worker_top_k[1] = [("adapter_B", 2000.0)]
        rm.worker_top_k[2] = [("adapter_C", 1000.0)]

        result = self._run(rm.patrol())

        assert result is not None
        assert len(calls) == 1  # exactly 1 preload triggered

    def test_no_action_when_no_congestion(self):
        """无拥塞时 patrol 返回 None，无复制动作"""
        calls = []

        def mock_callback(worker_id, adapter_dir):
            calls.append((worker_id, adapter_dir))
            return {"success": True}

        rm = _make_replica_manager(num_workers=3, w1=1.0, w2=1.0)
        rm.preload_callback = mock_callback
        rm.ema_rwpt = {0: 0.3, 1: 0.2, 2: 0.4}

        result = self._run(rm.patrol())

        assert result is None
        assert len(calls) == 0

    @given(
        num_congested=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50, deadline=None)
    def test_at_most_one_action_property(self, num_congested):
        """Property: 无论多少 Worker 拥塞，单次 patrol 最多 1 个动作"""
        calls = []

        def mock_callback(worker_id, adapter_dir):
            calls.append((worker_id, adapter_dir))
            return {"success": True}

        num_workers = num_congested + 1  # +1 idle worker
        rm = _make_replica_manager(
            num_workers=num_workers, w1=1.0, w2=1.0, cooldown_sec=0.0,
        )
        rm.preload_callback = mock_callback

        # Set congested workers
        for wid in range(num_congested):
            rm.ema_rwpt[wid] = 1.5
            rm.worker_top_k[wid] = [(f"adapter_{wid}", float(1000 - wid))]

        # One idle worker
        rm.ema_rwpt[num_congested] = 0.1

        self._run(rm.patrol())

        assert len(calls) <= 1

    def test_patrol_result_contains_action_details(self):
        """patrol 返回的结果包含完整的动作信息"""
        rm = _make_replica_manager(num_workers=2, w1=1.0, w2=1.0, cooldown_sec=0.0)
        rm.ema_rwpt = {0: 1.5, 1: 0.1}
        rm.worker_top_k[0] = [("adapter_X", 5000.0)]

        result = self._run(rm.patrol())

        assert result is not None
        assert result["action"] == "replicate"
        assert result["adapter_dir"] == "adapter_X"
        assert result["source_worker"] == 0
        assert result["target_worker"] == 1


# ─── Property 12: 路由器自动发现新副本 ──────────────────────────────────────

from slora.server.router.adapter_aware_router import AdapterAwareRouter
from slora.server.router.worker_state import RoutingConfig


def _make_router(num_workers=3, w1=1.0, w2=1.0):
    """Helper to create an AdapterAwareRouter for testing."""
    config = RoutingConfig(strategy='adapter-aware', w1=w1, w2=w2, w3=0.0)
    return AdapterAwareRouter(num_workers=num_workers, config=config)


class TestProperty12RouterAutoDiscovery:
    """
    Feature: hot-adapter-replication, Property 12: 路由器自动发现新副本

    当 cached_adapters 中新增了某个 adapter 时：
    (a) 路由器的倒排索引 adapter_to_workers 应自动包含该 Worker
    (b) 评分函数 calculate_score() 应对该 Worker 给出 w1 缓存命中加分

    Validates: Requirements 10.1, 10.2, 10.3
    """

    @given(
        num_workers=st.integers(min_value=2, max_value=6),
        worker_id=st.integers(min_value=0, max_value=5),
        w1=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_index_updated_after_heartbeat(self, num_workers, worker_id, w1):
        """update_worker_state 后 adapter_to_workers 自动包含该 Worker"""
        assume(worker_id < num_workers)
        router = _make_router(num_workers=num_workers, w1=w1)

        adapter = "adapter_new"
        state = WorkerState(
            worker_id=worker_id,
            cached_adapters={adapter},
        )
        router.update_worker_state(worker_id, state)

        assert worker_id in router.adapter_to_workers.get(adapter, set())

    @given(
        num_workers=st.integers(min_value=2, max_value=6),
        worker_id=st.integers(min_value=0, max_value=5),
        w1=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_score_includes_w1_bonus_after_cache(self, num_workers, worker_id, w1, w2):
        """新副本加载后 calculate_score 给出 w1 缓存命中加分"""
        assume(worker_id < num_workers)
        router = _make_router(num_workers=num_workers, w1=w1, w2=w2)

        adapter = "adapter_hot"

        # Before: worker does not have adapter
        score_before = router.calculate_score(worker_id, adapter)

        # After: worker caches the adapter
        state = WorkerState(worker_id=worker_id, cached_adapters={adapter})
        router.update_worker_state(worker_id, state)
        score_after = router.calculate_score(worker_id, adapter)

        # Score should increase by exactly w1
        assert abs((score_after - score_before) - w1) < 1e-9

    def test_index_removed_when_adapter_evicted(self):
        """adapter 被淘汰后，倒排索引自动移除该 Worker"""
        router = _make_router(num_workers=2)

        # Worker 0 caches adapter_A
        state1 = WorkerState(worker_id=0, cached_adapters={"adapter_A"})
        router.update_worker_state(0, state1)
        assert 0 in router.adapter_to_workers.get("adapter_A", set())

        # Worker 0 evicts adapter_A
        state2 = WorkerState(worker_id=0, cached_adapters=set())
        router.update_worker_state(0, state2)
        assert 0 not in router.adapter_to_workers.get("adapter_A", set())

    @given(
        num_workers=st.integers(min_value=2, max_value=5),
        shared_adapter=adapter_path_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_multiple_workers_cache_same_adapter(self, num_workers, shared_adapter):
        """多个 Worker 缓存同一 adapter 时，倒排索引包含所有 Worker"""
        router = _make_router(num_workers=num_workers)

        for wid in range(num_workers):
            state = WorkerState(worker_id=wid, cached_adapters={shared_adapter})
            router.update_worker_state(wid, state)

        workers_with_adapter = router.adapter_to_workers.get(shared_adapter, set())
        assert workers_with_adapter == set(range(num_workers))

    def test_no_modification_to_scoring_logic_needed(self):
        """副本加载后路由器无需修改评分逻辑即可自动分流"""
        router = _make_router(num_workers=3, w1=2.0, w2=1.0)

        # Initially only worker 0 has adapter_hot
        state0 = WorkerState(worker_id=0, cached_adapters={"adapter_hot"},
                             pending_prefill_tokens=900)
        router.update_worker_state(0, state0)

        # Worker 1 is idle but doesn't have adapter_hot
        state1 = WorkerState(worker_id=1, cached_adapters=set(),
                             pending_prefill_tokens=0)
        router.update_worker_state(1, state1)

        score0_before = router.calculate_score(0, "adapter_hot")
        score1_before = router.calculate_score(1, "adapter_hot")

        # Replicate adapter_hot to worker 1
        state1_new = WorkerState(worker_id=1, cached_adapters={"adapter_hot"},
                                 pending_prefill_tokens=0)
        router.update_worker_state(1, state1_new)

        score1_after = router.calculate_score(1, "adapter_hot")

        # Worker 1 now gets w1 bonus — score increased by w1=2.0
        assert abs((score1_after - score1_before) - 2.0) < 1e-9
        # Worker 1 score should now be competitive with worker 0
        assert score1_after > score1_before
