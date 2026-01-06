from dataclasses import dataclass
import numpy as np
import torch
from typing import List, Dict, Any
import time

from slora.common.mem_allocator import MemoryAllocator
from slora.utils.infer_utils import calculate_time, mark_start, mark_end


@dataclass
class InferAdapter:
    adapter_dirs: List[str]  # all adapters on the server
    a_loc: torch.Tensor  # a_loc[i] is a list of indices occupied by adapter i
    a_start: torch.Tensor  # a_start[i] is the start location of adapter i
    a_len: torch.Tensor  # a_len[i] is the number of cells occupied by adapter i
    a_scaling: torch.Tensor  # a_scaling[i] is the scaling factor of adapter i
    mem_manager: MemoryAllocator

    idx_map: Dict[str, int]
    prefetch_tag: Dict[str, int]
    cur_tag: int

    prefetch_stream: Any

    # LoRA 适配器分数相关数据结构
    adapter_scores: Dict[str, float]  # {adapter_dir: total_score} - 适配器综合分数
    score_update_counter: Dict[str, int]  # {adapter_dir: use_count} - 使用次数统计
    last_access_time: Dict[str, float]  # {adapter_dir: timestamp} - 最后访问时间
    total_use_duration: Dict[str, float]  # {adapter_dir: duration} - 累计使用时长（秒）
    current_request_count: Dict[str, int]  # {adapter_dir: count} - 当前使用该适配器的请求数
    load_time: Dict[str, float]  # {adapter_dir: timestamp} - 适配器加载时间

    @classmethod
    def init(cls, mem_manager, prefetch_stream):
        return cls(
            adapter_dirs=[],
            a_loc=torch.empty(0, dtype=torch.long, device="cuda"),
            a_start=torch.empty(0, dtype=torch.long, device="cuda"),
            a_len=torch.empty(0, dtype=torch.long, device="cuda"),
            a_scaling=torch.empty(0, dtype=torch.float16, device="cuda"),
            mem_manager=mem_manager,
            idx_map={},
            prefetch_tag={},
            cur_tag=0,
            prefetch_stream=prefetch_stream,
            # 初始化分数相关数据结构
            adapter_scores={},
            score_update_counter={},
            last_access_time={},
            total_use_duration={},
            current_request_count={},
            load_time={},
        )

    def update_adapter_stats_batch(self, batch_adapter_dirs: List[str]):
        """
        按批次更新显存中适配器的使用统计信息
        
        参数:
            batch_adapter_dirs: 当前批次中使用的所有适配器目录列表
        
        更新内容:
            - 使用次数 (score_update_counter)
            - 最后访问时间 (last_access_time)
            - 当前请求计数 (current_request_count)
        """
        current_time = time.time()
        
        # 统计批次中每个适配器出现的次数
        adapter_count = {}
        for adapter_dir in batch_adapter_dirs:
            if adapter_dir is None:
                continue
            # 只更新已加载到显存中的适配器
            if adapter_dir in self.idx_map:
                adapter_count[adapter_dir] = adapter_count.get(adapter_dir, 0) + 1
        
        # 批量更新统计信息
        for adapter_dir, count in adapter_count.items():
            # 更新使用次数
            self.score_update_counter[adapter_dir] = self.score_update_counter.get(adapter_dir, 0) + count
            
            # 更新最后访问时间
            self.last_access_time[adapter_dir] = current_time
            
            # 更新当前请求数（增量）
            self.current_request_count[adapter_dir] = self.current_request_count.get(adapter_dir, 0) + count
    
    def update_adapter_duration(self, adapter_dir: str, duration: float):
        """
        更新单个适配器的累计使用时长
        
        参数:
            adapter_dir: 适配器目录
            duration: 本次使用的时长（秒）
        """
        if adapter_dir is not None and adapter_dir in self.idx_map:
            self.total_use_duration[adapter_dir] = self.total_use_duration.get(adapter_dir, 0) + duration
    
    def decrease_request_count(self, adapter_dir: str, count: int = 1):
        """
        减少适配器的当前请求计数（请求完成时调用）
        
        参数:
            adapter_dir: 适配器目录
            count: 要减少的请求数量
        """
        if adapter_dir is not None and adapter_dir in self.idx_map:
            current_count = self.current_request_count.get(adapter_dir, 0)
            self.current_request_count[adapter_dir] = max(0, current_count - count)
    
    def calculate_adapter_score(self, adapter_dir: str, 
                                weight_usage: float = 0.3,
                                weight_recency: float = 0.3, 
                                weight_duration: float = 0.2,
                                weight_active: float = 0.2) -> float:
        """
        计算适配器的综合分数
        
        参数:
            adapter_dir: 适配器目录
            weight_usage: 使用次数权重
            weight_recency: 最近访问时间权重
            weight_duration: 累计使用时长权重
            weight_active: 当前活跃请求数权重
        
        返回:
            综合分数（越高越重要，越不应被淘汰）
        """
        if adapter_dir not in self.idx_map:
            return 0.0
        
        current_time = time.time()
        
        # 1. 使用次数分数（归一化）
        usage_count = self.score_update_counter.get(adapter_dir, 0)
        max_usage = max(self.score_update_counter.values()) if self.score_update_counter else 1
        usage_score = usage_count / max_usage if max_usage > 0 else 0
        
        # 2. 最近访问时间分数（越近越高）
        last_access = self.last_access_time.get(adapter_dir, 0)
        time_since_access = current_time - last_access if last_access > 0 else float('inf')
        # 使用指数衰减，1小时后分数降为0.37
        recency_score = np.exp(-time_since_access / 60)
        
        # 3. 累计使用时长分数（归一化）
        total_duration = self.total_use_duration.get(adapter_dir, 0)
        max_duration = max(self.total_use_duration.values()) if self.total_use_duration else 1
        duration_score = total_duration / max_duration if max_duration > 0 else 0
        
        # 4. 当前活跃请求数分数（归一化）
        active_requests = self.current_request_count.get(adapter_dir, 0)
        max_active = max(self.current_request_count.values()) if self.current_request_count else 1
        active_score = active_requests / max_active if max_active > 0 else 0
        
        # 综合分数
        total_score = (weight_usage * usage_score + 
                      weight_recency * recency_score + 
                      weight_duration * duration_score + 
                      weight_active * active_score)
        
        # 更新缓存的分数
        self.adapter_scores[adapter_dir] = total_score
        
        return total_score
    
    def get_adapters_by_score(self, top_k: int = None, ascending: bool = False) -> List[tuple]:
        """
        根据分数排序获取适配器列表
        
        参数:
            top_k: 返回前k个适配器，None表示返回全部
            ascending: True表示升序（分数低的优先），False表示降序（分数高的优先）
        
        返回:
            [(adapter_dir, score), ...] 按分数排序的列表
        """
        # 计算所有已加载适配器的分数
        scored_adapters = []
        for adapter_dir in self.adapter_dirs:
            score = self.calculate_adapter_score(adapter_dir)
            scored_adapters.append((adapter_dir, score))
        
        # 排序
        scored_adapters.sort(key=lambda x: x[1], reverse=not ascending)
        
        # 返回top_k
        if top_k is not None:
            return scored_adapters[:top_k]
        return scored_adapters
    
    def print_adapter_stats(self, top_k: int = 10):
        """
        打印适配器统计信息（用于调试）
        
        参数:
            top_k: 显示前k个适配器的详细信息
        """
        print(f"\n{'='*80}")
        print(f"适配器统计信息 (显存中共 {len(self.adapter_dirs)} 个适配器)")
        print(f"{'='*80}")
        
        scored_adapters = self.get_adapters_by_score(top_k=top_k, ascending=False)
        
        print(f"{'排名':<6}{'适配器':<40}{'分数':<10}{'使用次数':<10}{'活跃请求':<10}")
        print(f"{'-'*80}")
        
        for rank, (adapter_dir, score) in enumerate(scored_adapters, 1):
            adapter_short = adapter_dir[-35:] if len(adapter_dir) > 35 else adapter_dir
            usage = self.score_update_counter.get(adapter_dir, 0)
            active = self.current_request_count.get(adapter_dir, 0)
            print(f"{rank:<6}{adapter_short:<40}{score:<10.4f}{usage:<10}{active:<10}")
        
        print(f"{'='*80}\n")

    def get_lora_memory_usage(self) -> dict:
        """
        获取 LoRA 专用空间的使用情况
        
        返回:
            {
                'total_cells': 总内存空间（cells），包括 KV cache 和 LoRA 空间,
                'lora_cells': LoRA 专用空间大小（cells），0 表示与 KV cache 共享,
                'used_cells': 已使用空间（cells）,
                'available_cells': 可用空间（cells）,
                'usage_ratio': 使用率（0-1）,
                'num_adapters': 当前加载的适配器数量,
                'adapter_cells': 各适配器占用的 cells 列表
            }
        """
        # 从 MemoryAllocator 获取基础数据
        total_cells = self.mem_manager.tot_size
        lora_cells = total_cells - self.mem_manager.cache_size
        available_cells = self.mem_manager.can_use_mem_size
        used_cells = total_cells - available_cells
        
        # 计算使用率
        usage_ratio = used_cells / total_cells if total_cells > 0 else 0.0
        
        # 获取适配器信息
        num_adapters = len(self.adapter_dirs)
        adapter_cells = self.a_len.cpu().tolist() if num_adapters > 0 else []
        
        return {
            'total_cells': total_cells,
            'lora_cells': lora_cells,
            'used_cells': used_cells,
            'available_cells': available_cells,
            'usage_ratio': float(usage_ratio),
            'num_adapters': num_adapters,
            'adapter_cells': adapter_cells
        }

    def select_eviction_candidates(self, 
                                    evict_ratio: float = 0.2,
                                    preserve_adapters: set = None) -> List[str]:
        """
        选择要淘汰的适配器候选者
        
        参数:
            evict_ratio: 淘汰的比例（0-1），默认 0.2 (20%)
            preserve_adapters: 必须保留的适配器集合（当前批次使用的）
        
        返回:
            要淘汰的适配器目录列表
        """
        # 边界情况：没有适配器
        if len(self.adapter_dirs) == 0:
            return []
        
        # 边界情况：evict_ratio 为 0
        if evict_ratio <= 0:
            return []
        
        # 确保 evict_ratio 在有效范围内
        evict_ratio = min(evict_ratio, 1.0)
        
        # 初始化保护集合
        if preserve_adapters is None:
            preserve_adapters = set()
        
        # 获取所有适配器的分数（升序排列，分数低的在前）
        scored_adapters = self.get_adapters_by_score(ascending=True)
        
        # 过滤掉保护列表中的适配器
        evictable_adapters = []
        for adapter_dir, score in scored_adapters:
            if adapter_dir not in preserve_adapters:
                evictable_adapters.append((adapter_dir, score))
        
        # 边界情况：没有可淘汰的适配器
        if len(evictable_adapters) == 0:
            return []
        
        # 计算要淘汰的数量
        num_to_evict = int(len(evictable_adapters) * evict_ratio)
        
        # 确保至少淘汰 1 个（如果有可淘汰的且 evict_ratio > 0）
        if num_to_evict == 0 and evict_ratio > 0:
            num_to_evict = 1
        
        # 确保不超过可淘汰的总数
        num_to_evict = min(num_to_evict, len(evictable_adapters))
        
        # 选择低分适配器（已经按升序排列）
        candidates_to_evict = evictable_adapters[:num_to_evict]
        
        # 返回适配器目录列表
        eviction_list = [adapter_dir for adapter_dir, score in candidates_to_evict]
        
        return eviction_list

    def check_memory_threshold(self, threshold: float = 0.9) -> dict:
        """
        检查内存使用率是否超过阈值
        
        参数:
            threshold: 触发淘汰的阈值（0-1），默认 0.9 (90%)
        
        返回:
            {
                'over_threshold': 是否超过阈值 (bool),
                'current_ratio': 当前使用率 (float),
                'threshold': 设置的阈值 (float),
                'usage_info': 内存使用详情 (dict)
            }
        """
        # 参数验证：确保 threshold 在有效范围内
        if threshold < 0 or threshold > 1:
            raise ValueError(f"threshold 必须在 [0, 1] 范围内，当前值: {threshold}")
        
        # 获取当前内存使用情况
        usage_info = self.get_lora_memory_usage()
        
        # 提取当前使用率
        current_ratio = usage_info['usage_ratio']
        
        # 判断是否超过阈值
        over_threshold = current_ratio >= threshold
        
        # 构建返回结果
        result = {
            'over_threshold': over_threshold,
            'current_ratio': current_ratio,
            'threshold': threshold,
            'usage_info': usage_info
        }
        
        return result

    def execute_eviction(self, adapters_to_evict: List[str]) -> dict:
        """
        执行适配器淘汰并记录结果
        
        参数:
            adapters_to_evict: 要淘汰的适配器目录列表
        
        返回:
            {
                'before_usage': 淘汰前的内存使用情况,
                'after_usage': 淘汰后的内存使用情况,
                'evicted_adapters': 实际淘汰的适配器列表,
                'evicted_count': 淘汰的适配器数量,
                'cells_freed': 释放的 cells 数量
            }
        """
        # 记录淘汰前的内存使用情况
        before_usage = self.get_lora_memory_usage()
        
        # 初始化返回结果
        result = {
            'before_usage': before_usage,
            'after_usage': None,
            'evicted_adapters': [],
            'evicted_count': 0,
            'cells_freed': 0
        }
        
        # 边界情况：没有要淘汰的适配器
        if not adapters_to_evict or len(adapters_to_evict) == 0:
            result['after_usage'] = before_usage
            return result
        
        # 构建保留列表（当前所有适配器 - 要淘汰的）
        adapters_to_evict_set = set(adapters_to_evict)
        reserve_dirs = [d for d in self.adapter_dirs if d not in adapters_to_evict_set]
        
        # 记录实际要淘汰的适配器（只统计确实存在的）
        actual_evicted = [d for d in adapters_to_evict if d in self.adapter_dirs]
        
        # 如果没有实际要淘汰的适配器
        if len(actual_evicted) == 0:
            result['after_usage'] = before_usage
            return result
        
        # 打印淘汰信息
        print(f"\n   执行淘汰: 淘汰 {len(actual_evicted)} 个适配器")
        if len(actual_evicted) <= 5:
            print(f"   淘汰列表: {[d.split('/')[-1] for d in actual_evicted]}")
        else:
            print(f"   淘汰列表（前5个）: {[d.split('/')[-1] for d in actual_evicted[:5]]}...")
        
        # 调用 offload_adapters 执行淘汰
        self.offload_adapters(reserve_dirs)
        
        # 记录淘汰后的内存使用情况
        after_usage = self.get_lora_memory_usage()
        
        # 计算释放的空间
        cells_freed = before_usage['used_cells'] - after_usage['used_cells']
        
        # 更新返回结果
        result['after_usage'] = after_usage
        result['evicted_adapters'] = actual_evicted
        result['evicted_count'] = len(actual_evicted)
        result['cells_freed'] = cells_freed
        
        # 打印淘汰结果
        print(f"   淘汰完成: 使用率 {before_usage['usage_ratio']:.1%} → {after_usage['usage_ratio']:.1%}")
        print(f"   释放空间: {cells_freed} cells, 剩余 {after_usage['num_adapters']} 个适配器\n")
        
        return result

    def check_and_evict_by_threshold(self, 
                                      threshold: float = 0.9,
                                      evict_ratio: float = 0.2,
                                      preserve_adapters: set = None) -> dict:
        """
        检查空间使用率，超过阈值时淘汰低分适配器
        
        这是阈值淘汰的主入口方法，协调各子步骤完成完整的淘汰流程。
        
        参数:
            threshold: 触发淘汰的阈值（0-1），默认 0.9 (90%)
            evict_ratio: 淘汰的比例（0-1），默认 0.2 (20%)
            preserve_adapters: 必须保留的适配器集合（当前批次使用的）
        
        返回:
            {
                'triggered': 是否触发了淘汰检查,
                'evicted': 是否实际执行了淘汰,
                'reason': 未淘汰的原因（如果未淘汰）,
                'before_usage': 检查前的内存使用情况,
                'after_usage': 淘汰后的内存使用情况（如果淘汰了）,
                'evicted_adapters': 被淘汰的适配器列表,
                'evicted_count': 淘汰的适配器数量,
                'cells_freed': 释放的空间大小
            }
        """
        # 步骤 1：检查是否超过阈值
        check_result = self.check_memory_threshold(threshold)
        
        # 初始化返回结果
        result = {
            'triggered': True,
            'evicted': False,
            'reason': None,
            'before_usage': check_result['usage_info'],
            'after_usage': None,
            'evicted_adapters': [],
            'evicted_count': 0,
            'cells_freed': 0
        }
        
        # 步骤 2：判断是否超过阈值
        if not check_result['over_threshold']:
            # 未超过阈值，无需淘汰
            result['triggered'] = False
            result['reason'] = 'below_threshold'
            result['after_usage'] = check_result['usage_info']
            return result
        
        # 超过阈值，打印警告信息
        print(f"\n⚠️  LoRA 内存使用率 {check_result['current_ratio']:.1%} 超过阈值 {threshold:.1%}，触发淘汰")
        
        # 检查是否有适配器可以淘汰
        num_adapters = check_result['usage_info']['num_adapters']
        if num_adapters == 0:
            print("   没有加载任何适配器，无需淘汰")
            result['reason'] = 'no_adapters'
            result['after_usage'] = check_result['usage_info']
            return result
        
        # 步骤 3：选择淘汰候选者
        candidates = self.select_eviction_candidates(
            evict_ratio=evict_ratio,
            preserve_adapters=preserve_adapters
        )
        
        # 检查是否有候选者
        if len(candidates) == 0:
            print(f"   所有 {num_adapters} 个适配器都在使用中或受保护，无法淘汰")
            result['reason'] = 'no_candidates'
            result['after_usage'] = check_result['usage_info']
            return result
        
        # 步骤 4：执行淘汰
        eviction_result = self.execute_eviction(candidates)
        
        # 更新返回结果
        result['evicted'] = True
        result['reason'] = None
        result['after_usage'] = eviction_result['after_usage']
        result['evicted_adapters'] = eviction_result['evicted_adapters']
        result['evicted_count'] = eviction_result['evicted_count']
        result['cells_freed'] = eviction_result['cells_freed']
        
        return result

    def log_eviction_summary(self, evict_result: dict):
        """
        打印详细的淘汰统计信息
        
        参数:
            evict_result: check_and_evict_by_threshold 返回的结果字典
        """
        if not evict_result['triggered']:
            return
        
        print(f"\n{'='*80}")
        print(f"LoRA 阈值淘汰统计")
        print(f"{'='*80}")
        
        before = evict_result['before_usage']
        print(f"淘汰前状态:")
        print(f"  - 内存使用: {before['used_cells']}/{before['total_cells']} cells "
              f"({before['usage_ratio']:.1%})")
        print(f"  - 适配器数: {before['num_adapters']}")
        print(f"  - 可用空间: {before['available_cells']} cells")
        
        if evict_result['evicted']:
            after = evict_result['after_usage']
            print(f"\n淘汰操作:")
            print(f"  - 淘汰数量: {evict_result['evicted_count']}")
            print(f"  - 释放空间: {evict_result['cells_freed']} cells")
            
            if evict_result['evicted_adapters']:
                print(f"  - 淘汰列表（前5个）:")
                for adapter_dir in evict_result['evicted_adapters'][:5]:
                    adapter_name = adapter_dir.split('/')[-1]
                    score = self.adapter_scores.get(adapter_dir, 0.0)
                    print(f"      · {adapter_name} (分数: {score:.4f})")
            
            print(f"\n淘汰后状态:")
            print(f"  - 内存使用: {after['used_cells']}/{after['total_cells']} cells "
                  f"({after['usage_ratio']:.1%})")
            print(f"  - 适配器数: {after['num_adapters']}")
            print(f"  - 可用空间: {after['available_cells']} cells")
            print(f"  - 使用率变化: {before['usage_ratio']:.1%} → {after['usage_ratio']:.1%}")
        else:
            print(f"\n淘汰结果: 未执行淘汰")
            print(f"  - 原因: {evict_result['reason']}")
        
        print(f"{'='*80}\n")


    # @calculate_time(show=True, min_cost_ms=0)
    def load_lora_A(self, adapter, loc, prefetch=False):
        r = adapter.r
        h = adapter.network_config["hidden_size"]
        head_num = adapter.network_config["num_attention_heads"]
        head_dim = h // head_num

        for i in range(adapter.network_config["num_hidden_layers"]):
            adapter.layers[i].load_to_gpu(prefetch=prefetch)
            #self.mem_manager.key_buffer[i][loc[:r]] = adapter.layers[i].q_lora_A.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r:r * 2]] = adapter.layers[i].k_lora_A.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r * 2:r * 3]] = adapter.layers[i].v_lora_A.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r * 3:r * 4]] = adapter.layers[i].o_lora_A.transpose(0, 1).reshape(r, head_num, head_dim)

            w_combined = adapter.layers[i].w_combined
            self.mem_manager.key_buffer[i][loc] = w_combined[0]

            #self.mem_manager.key_buffer[i][loc[:r]] = w_combined[0].T.reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r:r * 2]] = w_combined[1].T.reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r * 2:r * 3]] = w_combined[2].T.reshape(r, head_num, head_dim)
            #self.mem_manager.key_buffer[i][loc[r * 3:r * 4]] = w_combined[3].T.reshape(r, head_num, head_dim)

            adapter.layers[i].offload_from_gpu()


    # @calculate_time(show=True, min_cost_ms=0)
    def load_lora_B(self, adapter, loc, prefetch=False):
        r = adapter.r
        h = adapter.network_config["hidden_size"]
        head_num = adapter.network_config["num_attention_heads"]
        head_dim = h // head_num
        for i in range(adapter.network_config["num_hidden_layers"]):
            adapter.layers[i].load_to_gpu(prefetch=prefetch)
            # this copy on gpu takes very few time, ~3ms for the following lines of copy
            #self.mem_manager.value_buffer[i][loc[:r]] = adapter.layers[i].q_lora_B.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r:r * 2]] = adapter.layers[i].k_lora_B.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r * 2:r * 3]] = adapter.layers[i].v_lora_B.transpose(0, 1).reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r * 3:r * 4]] = adapter.layers[i].o_lora_B.transpose(0, 1).reshape(r, head_num, head_dim)

            w_combined = adapter.layers[i].w_combined
            self.mem_manager.value_buffer[i][loc] = w_combined[1]

            #self.mem_manager.value_buffer[i][loc[:r]] = w_combined[4].reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r:r * 2]] = w_combined[5].reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r * 2:r * 3]] = w_combined[6].reshape(r, head_num, head_dim)
            #self.mem_manager.value_buffer[i][loc[r * 3:r * 4]] = w_combined[7].reshape(r, head_num, head_dim)

            adapter.layers[i].offload_from_gpu()

    # @calculate_time(show=True, min_cost_ms=0)
    def load_adapters(self, adapters, prefetch=False,
                      enable_threshold_eviction=True,
                      threshold=0.9,
                      evict_ratio=0.2):
        """
        加载 LoRA 适配器到 GPU 内存
        
        参数:
            adapters: 要加载的适配器列表
            prefetch: 是否为预取模式
            enable_threshold_eviction: 是否启用阈值淘汰（默认 True）
            threshold: 淘汰阈值（0-1），默认 0.9 (90%)
            evict_ratio: 淘汰比例（0-1），默认 0.2 (20%)
        """
        # func_name = "realload" if not prefetch else "prefetch"
        # mark_start(func_name)
        if len(adapters) == 0:
            print(f"load 0 adapters, {len(self.adapter_dirs)} in total")
            return

        if prefetch:
            self.cur_tag ^= 1
            capacity = self.mem_manager.can_use_mem_size
            new_adapters = []
            tot_size = 0
            # mark_start("load scan")
            for adapter in adapters:
                self.prefetch_tag[adapter.lora_dir] = self.cur_tag
                if adapter is not None and adapter.lora_dir not in self.idx_map:
                    if tot_size + adapter.r * 4 > capacity:
                        break
                    new_adapters.append(adapter)
                    tot_size += adapter.r * 4
            # mark_end("load scan")
            print(f"prefetch {len(new_adapters)} adapters, "
                  f"{len(self.adapter_dirs) + len(new_adapters)} in total")
        else:
            new_adapters = []
            tot_size = 0
            # mark_start("load scan")
            for adapter in adapters:
                if adapter is not None and adapter.lora_dir not in self.idx_map:
                    new_adapters.append(adapter)
                    tot_size += adapter.r * 4
            # mark_end("load scan")
            print(f"load {len(new_adapters)} adapters, {len(self.adapter_dirs) + len(new_adapters)} in total")

        # ===== 新增：加载前阈值检查 =====
        if not prefetch and enable_threshold_eviction and len(new_adapters) > 0:
            # 收集即将加载的适配器作为保护对象
            preserve_dirs = set()
            for adapter in new_adapters:
                if adapter is not None:
                    preserve_dirs.add(adapter.lora_dir)
            
            # 也保护当前预取标记中的适配器
            for adapter_dir in self.adapter_dirs:
                if adapter_dir in self.prefetch_tag and self.prefetch_tag[adapter_dir] == self.cur_tag:
                    preserve_dirs.add(adapter_dir)
            
            # 执行阈值检查和可能的淘汰
            evict_result = self.check_and_evict_by_threshold(
                threshold=threshold,
                evict_ratio=evict_ratio,
                preserve_adapters=preserve_dirs
            )
            
            # 淘汰后可选的日志输出（调试用）
            # if evict_result['evicted']:
            #     print(f"   阈值淘汰后剩余空间: {evict_result['after_usage']['available_cells']} cells")
        # ===== 新增结束 =====

        new_loc = self.mem_manager.alloc(tot_size)
        # assert len(new_loc) == tot_size
        start_offset = self.a_start.shape[0]
        self.a_start = torch.cat((self.a_start, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        len_offset = self.a_len.shape[0]
        self.a_len = torch.cat((self.a_len, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        loc_offset = self.a_loc.shape[0]
        self.a_loc = torch.cat((self.a_loc, torch.empty(tot_size, dtype=torch.long, device="cuda")))

        cum_loc = 0
        cum_loc_list = []
        current_time = time.time()
        for i, new_adapter in enumerate(new_adapters):
            cum_loc_list.append(cum_loc)
            self.idx_map[new_adapter.lora_dir] = len(self.adapter_dirs)
            self.adapter_dirs.append(new_adapter.lora_dir)
            self.a_start[start_offset + i] = loc_offset + cum_loc
            self.a_len[len_offset + i] = new_adapter.r * 4
            self.a_loc[loc_offset + cum_loc: loc_offset + cum_loc + new_adapter.r * 4] = (
                    new_loc[cum_loc: cum_loc + new_adapter.r * 4])
            cum_loc += new_adapter.r * 4
            
            # 记录加载时间
            if new_adapter.lora_dir not in self.load_time:
                self.load_time[new_adapter.lora_dir] = current_time
                # 初始化其他统计数据
                self.adapter_scores[new_adapter.lora_dir] = 0.0
                self.score_update_counter[new_adapter.lora_dir] = 0
                self.last_access_time[new_adapter.lora_dir] = current_time
                self.total_use_duration[new_adapter.lora_dir] = 0.0
                self.current_request_count[new_adapter.lora_dir] = 0
        self.a_scaling = torch.cat((self.a_scaling, torch.tensor([adapter.scaling for adapter in new_adapters], dtype=torch.float16, device="cuda")))

        #if prefetch:
        #    torch.cuda.synchronize()
        #    tic1 = time.time()

        if prefetch:
            with torch.cuda.stream(self.prefetch_stream):
                new_loc = new_loc.clone()
                for i, new_adapter in enumerate(new_adapters):
                    #self.idx_map[new_adapter.lora_dir] = len(self.adapter_dirs)
                    #self.adapter_dirs.append(new_adapter.lora_dir)
                    #self.a_start[start_offset + i] = loc_offset + cum_loc
                    #self.a_len[len_offset + i] = new_adapter.r * 4

                    cum_loc = cum_loc_list[i]
                    self.load_lora_A(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)
                    self.load_lora_B(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)

                    #self.load_lora_A(new_adapter, None, prefetch)
                    #self.load_lora_B(new_adapter, None, prefetch)
        else:
            for i, new_adapter in enumerate(new_adapters):
                cum_loc = cum_loc_list[i]
                self.load_lora_A(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)
                self.load_lora_B(new_adapter, new_loc[cum_loc: cum_loc + new_adapter.r * 4], prefetch)

            #if prefetch:
        #    tic2 = time.time()
        #    torch.cuda.synchronize()
        #    tic3 = time.time()
        #    print("launch time", tic2 - tic1, flush=True)
        #    print("total time", tic3 - tic1, flush=True)
        # mark_end(func_name)
        # print(f"current adapters on batch (loaded {len(new_adapters)})",
        #       len(self.adapter_dirs), self.adapter_dirs)
        # print(self.mem_manager.can_use_mem_size_suffix // 4 / 32)
    

    # @calculate_time(show=True, min_cost_ms=0)
    def offload_adapters(self, reserve_adapter_dirs):
        """
        卸载不需要的 LoRA 适配器，释放显存
        
        淘汰策略：
        1. 保留在 reserve_adapter_dirs 列表中的适配器
        2. 保留正在预取的适配器（通过 prefetch_tag 保护）
        3. 淘汰其他所有适配器
        
        参数:
            reserve_adapter_dirs: 需要保留的适配器目录列表
                                 如果为空列表，则卸载所有适配器
                                 如果包含所有已加载的适配器，则不卸载任何适配器
        """
        # 情况1：保留列表包含所有已加载的适配器，无需卸载
        if len(reserve_adapter_dirs) == len(self.adapter_dirs):
            print(f"offload 0 adapters, {len(self.adapter_dirs)} remains")
            return
        
        # 情况2：保留列表为空，卸载所有适配器
        if len(reserve_adapter_dirs) == 0:
            print(f"offload {len(self.adapter_dirs)} adapters, 0 remains")
            # 释放所有适配器占用的显存
            self.mem_manager.free(self.a_loc)
            # 清空所有适配器相关的数据结构
            self.adapter_dirs=[]
            self.a_loc=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_start=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_len=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_scaling=torch.empty(0, dtype=torch.float16, device="cuda")
            self.idx_map={}
            return

        # 情况3：部分保留，部分淘汰
        # mark_start("offload scan")
        remove_ind = []      # 存储需要释放的内存索引
        left_ind = []        # 存储需要保留的适配器索引
        new_adapter_dirs = [] # 保留的适配器目录列表
        removed_adapter_dirs = []  # 记录被移除的适配器（用于清理统计数据）
        self.idx_map = {}    # 重建索引映射
        
        # 扫描所有已加载的适配器，决定哪些保留，哪些淘汰
        for i, adapter_dir in enumerate(self.adapter_dirs):
            # 淘汰条件：不在保留列表中 且 （不在预取中 或 预取已完成）
            # 保留条件：在保留列表中 或 正在预取（prefetch_tag 匹配当前标签）
            if (adapter_dir not in reserve_adapter_dirs and
                (adapter_dir not in self.prefetch_tag or
                 self.prefetch_tag[adapter_dir] != self.cur_tag)):
                # 标记为淘汰：记录该适配器占用的内存索引
                remove_ind.append(self.a_loc[self.a_start[i]:self.a_start[i] + self.a_len[i]])
                removed_adapter_dirs.append(adapter_dir)
            else:
                # 标记为保留：记录索引并更新映射
                left_ind.append(i)
                self.idx_map[adapter_dir] = len(new_adapter_dirs)
                new_adapter_dirs.append(adapter_dir)
        
        # 如果没有需要淘汰的适配器，直接返回
        if len(remove_ind) == 0:
            return
        # mark_end("offload scan")
        
        # 清理被卸载适配器的统计数据（分数、使用次数、访问时间等）
        for adapter_dir in removed_adapter_dirs:
            self.adapter_scores.pop(adapter_dir, None)
            self.score_update_counter.pop(adapter_dir, None)
            self.last_access_time.pop(adapter_dir, None)
            self.total_use_duration.pop(adapter_dir, None)
            self.current_request_count.pop(adapter_dir, None)
            self.load_time.pop(adapter_dir, None)
        
        # 更新适配器目录列表
        self.adapter_dirs = new_adapter_dirs
        # 计算保留适配器占用的总大小
        tot_size = torch.sum(self.a_len[left_ind]).item()
        print(f"offload {len(remove_ind)} adapters, {len(left_ind)} remains")

        # 合并所有需要释放的内存索引
        # mark_start("offload cat")
        remove_ind = torch.cat(remove_ind)
        # mark_end("offload cat")
        
        # 释放被淘汰适配器占用的显存
        # mark_start("offload free mem manager")
        self.mem_manager.free(remove_ind)
        # mark_end("offload free mem manager")
        
        # 重建保留适配器的索引结构
        # mark_start("offload torch.empty")
        new_a_len = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_start = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_scaling = torch.empty(len(left_ind), dtype=torch.float16, device="cuda")
        new_a_loc = torch.empty(tot_size, dtype=torch.long, device="cuda")
        # mark_end("offload torch.empty")

        # 复制保留适配器的长度和缩放因子
        new_a_len[:] = self.a_len[left_ind]
        # 重新计算起始位置（从0开始，连续排列）
        new_a_start[0] = 0
        new_a_start[1:] = torch.cumsum(new_a_len, dim=0)[:-1]
        new_a_scaling[:] = self.a_scaling[left_ind]
        
        # 使用 Triton 内核高效地复制保留适配器的内存位置信息
        # mark_start("offload a_loc update")
        launch_var_len_copy_triton(self.a_start[left_ind], new_a_len,
                                   self.a_loc, new_a_start, new_a_loc)
        # mark_end("offload a_loc update")

        # 更新所有索引结构
        self.a_start = new_a_start
        self.a_len = new_a_len
        self.a_loc = new_a_loc
        self.a_scaling = new_a_scaling

        # print(f"current adapters on batch (offloaded {len(remove_ind)})",
        #       len(self.adapter_dirs), self.adapter_dirs)
        # print(self.mem_manager.can_use_mem_size_suffix // 4 / 32)


import triton
import triton.language as tl


@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    a_id = tl.program_id(0)
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + old_start + i + old_offset, mask=old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=new_offset < length)


def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)

    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_location, new_a_start, new_a_location, BLOCK_SIZE)


"""
from cupyx import jit
import cupy
import torch


@jit.rawkernel()
def var_len_copy_kernel(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                        BLOCK_SIZE):
    a_id = jit.blockIdx.x
    t_id = jit.threadIdx.x
    for i in range(t_id, old_a_len[a_id], BLOCK_SIZE):
        new_a_location[new_a_start[a_id] + i] = old_a_location[old_a_start[a_id] + i]


def launch_var_len_copy(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 128
    print(BLOCK_SIZE)
    grid_size = (len(old_a_start),)
    assert len(old_a_start) == len(new_a_start) == len(old_a_len)
    block_size = (BLOCK_SIZE,)

    var_len_copy_kernel(grid_size, block_size,
        (cupy.asarray(old_a_start),
         cupy.asarray(old_a_len),
         cupy.asarray(old_location),
         cupy.asarray(new_a_start),
         cupy.asarray(new_a_location),
         BLOCK_SIZE))
"""
