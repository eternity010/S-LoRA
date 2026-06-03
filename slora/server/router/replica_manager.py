"""
ReplicaManager — 热门 Adapter 主动复制决策模块

运行在 Router 端，通过周期性巡检检测 Worker 拥塞、识别元凶 Adapter、
选择目标 Worker 并触发预加载。

核心设计决策：
- 拥塞阈值 EMA(RWPT/Capacity) > 1.0，表示约一个 prefill batch 积压
- 副本上限 N_max = num_workers - 1
- EMA 平滑系数 α = 0.3（约 5 次心跳窗口）
- 巡检周期 0.5~2s，每周期最多 1 个复制动作

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import time
import asyncio
import logging
from typing import Dict, Set, List, Tuple, Optional, Callable

from slora.server.router.worker_state import WorkerState

logger = logging.getLogger(__name__)


class ReplicaManager:
    """
    副本管理器，运行在 Router 端。

    维护副本状态、EMA 平滑、拥塞检测、元凶识别、护栏检查、
    目标选择、触发预加载。

    Attributes:
        num_workers: Worker 总数
        capacity: batch_max_tokens（RWPT 归一化分母）
        w1: 路由评分函数的缓存亲和性权重
        w2: 路由评分函数的负载惩罚权重
        congestion_threshold: 归一化 RWPT 拥塞阈值，默认 1.0 batch
        n_max: 单 adapter 最大副本数 = num_workers - 1
        patrol_interval_sec: 巡检周期（秒）
        cooldown_sec: 同一 adapter 两次复制最小间隔（秒）
        protection_sec: 新副本淘汰保护时长（秒）
        ema_alpha: EMA 平滑系数
        max_protected_per_worker: 每 Worker 最大受保护副本数
        preload_callback: 预加载回调函数 (worker_id, adapter_dir) -> dict
    """

    def __init__(
        self,
        num_workers: int,
        capacity: float,
        w1: float,
        w2: float,
        patrol_interval_sec: float = 1.0,
        cooldown_sec: float = 5.0,
        protection_sec: float = 30.0,
        congestion_threshold: float = 1.0,
        ema_alpha: float = 0.3,
        max_protected_per_worker: int = 2,
        preload_callback: Optional[Callable] = None,
    ):
        # --- 配置参数 ---
        self.num_workers = num_workers
        self.capacity = capacity
        self.w1 = w1
        self.w2 = w2
        self.patrol_interval_sec = patrol_interval_sec
        self.cooldown_sec = cooldown_sec
        self.protection_sec = protection_sec
        if congestion_threshold <= 0:
            raise ValueError(
                f"congestion_threshold must be positive, got {congestion_threshold}"
            )
        self.congestion_threshold = congestion_threshold
        self.ema_alpha = ema_alpha
        self.max_protected_per_worker = max_protected_per_worker
        self.preload_callback = preload_callback

        # --- 派生参数 ---
        self.n_max = num_workers - 1  # 单 adapter 最大副本数
        # Historical scoring-boundary threshold, kept for observability only.
        self.t_congestion = self._compute_t_congestion(w1, w2, capacity)

        # --- 状态字典 ---
        # adapter → 缓存该 adapter 的 worker_id 集合
        self.replica_map: Dict[str, Set[int]] = {}
        # adapter → 上次复制时间戳
        self.last_replication_time: Dict[str, float] = {}
        # worker_id → EMA(RWPT / Capacity)
        self.ema_rwpt: Dict[int, float] = {i: 0.0 for i in range(num_workers)}
        # worker_id → top_k_rwpt_adapters
        self.worker_top_k: Dict[int, List[Tuple[str, float]]] = {
            i: [] for i in range(num_workers)
        }
        # worker_id → {adapter_dir: expire_time}
        self.protected_replicas: Dict[int, Dict[str, float]] = {
            i: {} for i in range(num_workers)
        }

        # --- 统计计数器 ---
        self._stats_total_replications: int = 0
        self._stats_guardrail_blocks: int = 0
        self._stats_no_target: int = 0
        self._stats_no_culprit: int = 0

        logger.info(
            f"ReplicaManager initialized: num_workers={num_workers}, "
            f"capacity={capacity}, w1={w1}, w2={w2}, "
            f"congestion_threshold={self.congestion_threshold:.2f}, "
            f"T_congestion_legacy={self.t_congestion:.2f}, N_max={self.n_max}, "
            f"cooldown={cooldown_sec}s, protection={protection_sec}s, "
            f"ema_alpha={ema_alpha}, patrol_interval={patrol_interval_sec}s"
        )

    # ------------------------------------------------------------------
    # 静态计算
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_t_congestion(w1: float, w2: float, capacity: float) -> float:
        """计算拥塞阈值 T_congestion = (w1/w2) × Capacity"""
        if w2 == 0:
            return float("inf")
        return (w1 / w2) * capacity

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def update_worker_state(self, worker_id: int, state: WorkerState) -> None:
        """
        心跳回调：更新 EMA、replica_map、worker_top_k。

        每次收到 Worker 心跳时由 DataParallelRouterManager 调用。

        Args:
            worker_id: Worker ID
            state: 该 Worker 的最新状态

        Requirements: 9.1, 9.4
        """
        # 1. 更新 EMA(RWPT / Capacity)
        if self.capacity > 0:
            current_ratio = state.pending_prefill_tokens / self.capacity
        else:
            current_ratio = 0.0

        old_ema = self.ema_rwpt.get(worker_id, 0.0)
        new_ema = self.ema_alpha * current_ratio + (1.0 - self.ema_alpha) * old_ema
        self.ema_rwpt[worker_id] = new_ema

        # 2. 更新 replica_map（基于 cached_adapters）
        #    先从所有 adapter 的集合中移除该 worker，再重新添加
        for adapter, workers in list(self.replica_map.items()):
            workers.discard(worker_id)
            if not workers:
                del self.replica_map[adapter]

        for adapter in state.cached_adapters:
            if adapter not in self.replica_map:
                self.replica_map[adapter] = set()
            self.replica_map[adapter].add(worker_id)

        # 3. 更新 worker_top_k
        self.worker_top_k[worker_id] = list(state.top_k_rwpt_adapters)

        # 4. 清理过期的 protected_replicas
        now = time.time()
        if worker_id in self.protected_replicas:
            expired = [
                a for a, t in self.protected_replicas[worker_id].items() if now >= t
            ]
            for a in expired:
                del self.protected_replicas[worker_id][a]

    def update_config(self, w1: float, w2: float) -> None:
        """
        动态更新 w1/w2 日志状态；复制触发阈值保持独立。

        Args:
            w1: 新的缓存亲和性权重
            w2: 新的负载惩罚权重

        Requirements: 9.2, 9.3
        """
        old_t = self.t_congestion
        self.w1 = w1
        self.w2 = w2
        self.t_congestion = self._compute_t_congestion(w1, w2, self.capacity)
        logger.info(
            f"ReplicaManager config updated: w1={w1}, w2={w2}, "
            f"T_congestion_legacy {old_t:.2f} -> {self.t_congestion:.2f}, "
            f"congestion_threshold={self.congestion_threshold:.2f}"
        )

    def get_replica_count(self, adapter_dir: str) -> int:
        """查询指定 adapter 的当前副本数。"""
        return len(self.replica_map.get(adapter_dir, set()))

    def get_replica_distribution(self, adapter_dir: str) -> Set[int]:
        """查询指定 adapter 的副本分布（缓存该 adapter 的 worker_id 集合）。"""
        return set(self.replica_map.get(adapter_dir, set()))

    def get_stats(self) -> Dict:
        """
        返回副本管理器的运行统计。

        Returns:
            包含复制次数、拒绝次数、当前副本分布等统计的字典

        Requirements: 11.2, 11.3
        """
        return {
            "total_replications": self._stats_total_replications,
            "guardrail_blocks": self._stats_guardrail_blocks,
            "no_target_count": self._stats_no_target,
            "no_culprit_count": self._stats_no_culprit,
            "replica_map": {
                adapter: sorted(workers)
                for adapter, workers in self.replica_map.items()
            },
            "ema_rwpt": dict(self.ema_rwpt),
            "t_congestion": self.t_congestion,
            "congestion_threshold": self.congestion_threshold,
            "n_max": self.n_max,
        }

    # ------------------------------------------------------------------
    # 拥塞检测与元凶识别 (Task 5.5)
    # ------------------------------------------------------------------

    def _detect_congested_workers(self) -> List[int]:
        """
        检测拥塞 Worker。

        判定条件：EMA(RWPT/Capacity) > congestion_threshold。
        默认阈值 1.0 表示约一个 prefill batch 的积压工作量。

        Returns:
            拥塞 Worker 的 worker_id 列表

        Requirements: 3.1, 3.2, 3.3, 3.4
        """
        congested = []
        for worker_id, ema in self.ema_rwpt.items():
            if ema > self.congestion_threshold:
                congested.append(worker_id)
        return congested

    def _identify_culprit(self, worker_id: int) -> Optional[str]:
        """
        从指定 Worker 的 top_k_rwpt_adapters 中识别元凶 adapter。

        取 RWPT 贡献最大的（列表第一个）元素。
        空列表时返回 None，跳过复制决策。

        Args:
            worker_id: 拥塞 Worker 的 ID

        Returns:
            元凶 adapter 的路径，或 None

        Requirements: 4.1, 4.2, 4.3
        """
        top_k = self.worker_top_k.get(worker_id, [])
        if not top_k:
            return None
        return top_k[0][0]  # (adapter_dir, contribution) 的第一个元素

    # ------------------------------------------------------------------
    # 全局防爆护栏 (Task 5.8)
    # ------------------------------------------------------------------

    def _check_guardrails(self, adapter_dir: str) -> bool:
        """
        护栏检查：副本数 < N_max 且冷却期已过。

        通过后更新 last_replication_time[adapter_dir] = now。

        Args:
            adapter_dir: 待复制的 adapter 路径

        Returns:
            True 表示允许复制，False 表示拒绝

        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        # 条件 1：副本数 < N_max
        if self.get_replica_count(adapter_dir) >= self.n_max:
            logger.debug(
                f"Guardrail blocked: {adapter_dir} replica count "
                f"{self.get_replica_count(adapter_dir)} >= N_max {self.n_max}"
            )
            self._stats_guardrail_blocks += 1
            return False

        # 条件 2：冷却期已过
        now = time.time()
        last_time = self.last_replication_time.get(adapter_dir, 0.0)
        if now - last_time < self.cooldown_sec:
            logger.debug(
                f"Guardrail blocked: {adapter_dir} cooldown not elapsed "
                f"({now - last_time:.1f}s < {self.cooldown_sec}s)"
            )
            self._stats_guardrail_blocks += 1
            return False

        # 通过护栏，更新时间戳
        self.last_replication_time[adapter_dir] = now
        return True

    # ------------------------------------------------------------------
    # 目标 Worker 选择 (Task 5.10)
    # ------------------------------------------------------------------

    def _select_target_worker(self, adapter_dir: str) -> Optional[int]:
        """
        选择目标 Worker：未缓存 + 负载低 + 保护数未满，选 EMA 最低的。

        三个筛选条件：
        1. 尚未缓存该 adapter
        2. EMA(RWPT/Capacity) < congestion_threshold × 0.9
        3. 当前受保护副本数 < max_protected_per_worker

        无合格候选时返回 None（弹性降级）。

        Args:
            adapter_dir: 待复制的 adapter 路径

        Returns:
            目标 Worker ID，或 None

        Requirements: 6.1, 6.2, 6.3
        """
        load_threshold = self.congestion_threshold * 0.9
        cached_workers = self.replica_map.get(adapter_dir, set())

        best_worker = None
        best_ema = float("inf")

        for worker_id in range(self.num_workers):
            # 条件 1：未缓存该 adapter
            if worker_id in cached_workers:
                continue

            # 条件 2：负载低
            ema = self.ema_rwpt.get(worker_id, 0.0)
            if ema >= load_threshold:
                continue

            # 条件 3：受保护副本数未满
            protected_count = len(self.protected_replicas.get(worker_id, {}))
            if protected_count >= self.max_protected_per_worker:
                continue

            # 选 EMA 最低的
            if ema < best_ema:
                best_ema = ema
                best_worker = worker_id

        return best_worker

    # ------------------------------------------------------------------
    # 巡检主循环 (Task 5.12)
    # ------------------------------------------------------------------

    async def patrol(self) -> Optional[Dict]:
        """
        巡检主循环：拥塞检测 → 元凶识别 → 护栏检查 → 目标选择 → 预加载。

        每次调用最多产生 1 个复制动作。

        Returns:
            复制动作的结果字典，或 None（无动作）

        Requirements: 7.1, 7.2, 7.3, 7.4, 6.4
        """
        congested = self._detect_congested_workers()
        if not congested:
            return None

        for worker_id in congested:
            # 识别元凶
            culprit = self._identify_culprit(worker_id)
            if culprit is None:
                self._stats_no_culprit += 1
                continue

            # 护栏检查
            if not self._check_guardrails(culprit):
                continue

            # 选择目标 Worker
            target = self._select_target_worker(culprit)
            if target is None:
                self._stats_no_target += 1
                continue

            # 记录保护期
            now = time.time()
            if target not in self.protected_replicas:
                self.protected_replicas[target] = {}
            self.protected_replicas[target][culprit] = now + self.protection_sec

            # 触发预加载回调
            result = {
                "action": "replicate",
                "adapter_dir": culprit,
                "source_worker": worker_id,
                "target_worker": target,
            }

            if self.preload_callback:
                try:
                    cb_result = self.preload_callback(target, culprit)
                    # 支持 async 和 sync 回调
                    if asyncio.iscoroutine(cb_result):
                        cb_result = await cb_result
                    result["callback_result"] = cb_result
                except Exception as e:
                    logger.error(f"Preload callback failed: {e}")
                    result["callback_result"] = {"success": False, "error": str(e)}

            self._stats_total_replications += 1
            logger.info(
                f"Patrol: replicate {culprit} from Worker {worker_id} "
                f"to Worker {target}"
            )

            # 每周期最多 1 个复制动作
            return result

        return None

    async def _start_patrol_loop(self) -> None:
        """
        启动巡检线程（asyncio Task），按 patrol_interval_sec 周期调用 patrol()。

        异常不会终止循环，仅记录日志。
        """
        logger.info(
            f"Patrol loop started (interval={self.patrol_interval_sec}s)"
        )
        while True:
            try:
                await self.patrol()
            except asyncio.CancelledError:
                logger.info("Patrol loop cancelled")
                break
            except Exception as e:
                logger.error(f"Patrol loop error: {e}")
            await asyncio.sleep(self.patrol_interval_sec)
