import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
import pickle
import time
import torch
import zmq
import zmq.asyncio
from typing import Dict, List, Optional

from ..sampling_params import SamplingParams
from ..io_struct import Req, Batch, BatchAbortReq
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from rpyc.utils.classic import obtain
from slora.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq
from .stats import Stats

from slora.server.input_params import InputParams
from slora.models.peft.lora_adapter import get_lora_config
from slora.server.router.profiler import AlphaModel, BetaModel
from slora.server.router.abort_req_queue import AbortReqQueue
from slora.server.router.cluster_req_queue import ClusterReqQueue
from slora.server.router.vtc_req_queue import VTCReqQueue
from slora.server.router.pets_req_queue import PETSReqQueue
from slora.server.router.peft_req_queue import PEFTReqQueue


def get_scheduler(input_params, adapter_dirs):
    if input_params.scheduler == "vtc_fair":
        return VTCReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.fair_weights)
    elif input_params.scheduler == "pets":
        return PETSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.scheduler == "peft":
        return PEFTReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.batch_num_adapters is not None:
        return ClusterReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                               input_params.running_max_req_size, input_params.batch_num_adapters)
    elif input_params.enable_abort:
        return AbortReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                             input_params.running_max_req_size)
    elif input_params.scheduler == "slora":
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    else:
        raise Exception("unrecognized scheduler")


class RouterManager:

    def __init__(self, weightdir, adapter_dirs, load_way, world_size, eos_id,
                 router_port, detokenization_port, model_rpc_ports,
                 input_params,
                 mode=[], log_stats=True, log_stats_interval=10):
        self.model_weightdir = weightdir
        self.adapter_dirs = adapter_dirs
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params

        if self.input_params.prefetch:
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.prefetch_stream = None

        # get adapter rank
        self.lora_ranks = {}
        for lora_dir in adapter_dirs:
            config, _ = get_lora_config(lora_dir, input_params.dummy)
            self.lora_ranks[lora_dir] = config["r"]
        self.lora_ranks[None] = 0

        self.req_queue = get_scheduler(input_params, adapter_dirs)

        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        
        # 缓存实际的 adapter 内存占用（单位：cells）
        # 在淘汰/加载后更新，用于并发控制的准确判断
        self.actual_adapter_memory_usage = 0
        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(log_stats, log_stats_interval)

    async def _update_actual_adapter_usage(self):
        """
        查询并更新实际的 adapter 内存占用
        
        通过 RPC 查询 LoRA 内存使用情况，提取 adapter 实际占用的 cells 数。
        用于并发控制的准确判断。
        """
        if self.input_params.no_lora:
            self.actual_adapter_memory_usage = 0
            return
        
        try:
            # 查询第一个 RPC 节点（所有节点的 adapter 占用应该相同）
            memory_info = await self.model_rpcs[0].check_lora_memory()
            
            if memory_info:
                # 计算实际的 adapter 占用：所有已加载 adapters 的总和
                adapter_cells_list = memory_info.get('adapter_cells', [])
                self.actual_adapter_memory_usage = sum(adapter_cells_list)
            else:
                # 查询失败，使用保守估计（当前值不变）
                pass
        except Exception as e:
            # 查询出错，使用保守估计
            print(f"警告：无法查询实际 adapter 占用: {e}")

    async def wait_to_model_ready(self):
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            init_model_ret.append(
                self.model_rpcs[rank_id].init_model(
                    rank_id,
                    self.world_size,
                    self.model_weightdir,
                    self.adapter_dirs,
                    self.input_params.max_total_token_num,
                    self.load_way,
                    self.mode,
                    input_params=self.input_params,
                    prefetch_stream=self.prefetch_stream,
                ))

        await asyncio.gather(*init_model_ret)
        
        # 新增：初始化后查询一次实际 adapter 占用
        await self._update_actual_adapter_usage()
        return
    
    async def profile_prefill(self):
        res = []
        for rank_id in range(self.world_size):  # async init model process
            res.append(
                self.model_rpcs[rank_id].profile_prefill())

        results = await asyncio.gather(*res)
        self.alpha_model = AlphaModel(results[0])
        self.beta_model = BetaModel(results[0])
        # check if the path exists else create it
        cache_dir = os.path.expanduser("~/.cache/slora")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_dir+"/profile_results.pkl", "wb") as f:
            pickle.dump(results[0], f)
        return


    def add_req(
        self,
        adapter_dir: str,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str
    ):
        req = Req(adapter_dir, request_id, prompt_ids, sampling_params)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    print("current batch size:", len(self.running_batch.reqs), "token used ratio:", self.running_batch.calcu_used_tokens() / self.input_params.max_total_token_num)
                    pass
                self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(
                self.running_batch, 
                self.lora_ranks,
                actual_adapter_size=self.actual_adapter_memory_usage  # 新增
            )
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch

                if not self.input_params.no_lora:
                    # load adapters
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                    await asyncio.gather(*ret)
                    
                    # 新增：加载后更新实际占用
                    await self._update_actual_adapter_usage()

                
                # merge adapter to base model
                if self.input_params.scheduler == "peft":
                    torch.cuda.synchronize()
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].merge_adapter())
                    await asyncio.gather(*ret)
            
                torch.cuda.synchronize()
                await self._prefill_batch(self.running_batch)
                await self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        if self.has_wait_tokens < self.max_wait_tokens:
            self.stats_tool.count_output_tokens(self.running_batch)
            # prefetch
            if (not self.input_params.no_lora and
                self.input_params.prefetch and (self.has_wait_tokens == self.max_wait_tokens // 2 or
                self.has_wait_tokens == self.max_wait_tokens - 3) and self.input_params.scheduler != "peft"):
                next_batch = self.req_queue.next_batch()
                if next_batch is not None:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(
                            next_batch.adapter_dirs, prefetch=True))
                    await asyncio.gather(*ret)
            await self._decode_batch(self.running_batch)
            await self._filter_runing_batch()

            self.has_wait_tokens += 1
            return
        else:
            new_mini_batch = self.req_queue.generate_new_batch(
                self.running_batch, 
                self.lora_ranks,
                actual_adapter_size=self.actual_adapter_memory_usage  # 新增
            )
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)

                if not self.input_params.no_lora:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                    await asyncio.gather(*ret)
                    
                    # 新增：加载后更新实际占用
                    await self._update_actual_adapter_usage()

                await self._prefill_batch(new_mini_batch, minibatch=True)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                self.has_wait_tokens = 0
            else:
                self.stats_tool.count_output_tokens(self.running_batch)
                await self._decode_batch(self.running_batch)
                await self._filter_runing_batch()
        

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _prefill_batch(self, batch, minibatch=True):
        await self._init_batch(batch)
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=True)
        return

    async def _decode_batch(self, batch:Batch):
        self.req_queue.update_counter(batch)
        
        # 更新适配器使用统计信息（在推理前）
        if not self.input_params.no_lora:
            adapter_dirs_list = list(batch.adapter_dirs)
            ret = []
            for tp_rank in range(self.world_size):
                ret.append(self.model_rpcs[tp_rank].update_adapter_stats(adapter_dirs_list))
            await asyncio.gather(*ret)
        
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        return

    async def _filter_batch(self, batch: Batch):
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False):
        """
        处理批次中已完成请求的逻辑
        
        当批次中有请求完成时：
        1. 过滤掉已完成的请求，更新批次状态
        2. 如果是 PEFT 调度器且批次已清空，则取消合并适配器
        3. 淘汰不在当前批次中的 LoRA 适配器（释放显存）
        4. 如果批次完全清空，则移除批次；否则过滤批次
        
        参数:
            batch: 当前运行的批次
            has_new_finished_req: 是否有新完成的请求
            minibatch: 是否为小批次（小批次不触发适配器淘汰，避免频繁操作）
        """
        if has_new_finished_req:
            # 记录完成的请求使用的适配器（在 filter_finished 之前）
            finished_adapter_dirs = []
            if not self.input_params.no_lora:
                for req in batch.reqs:
                    if req.has_generate_finished:
                        finished_adapter_dirs.append(req.adapter_dir)
            
            # 保存批次的原始 adapter_dirs（在 filter_finished 之前）
            # 这些是批次当前正在使用的所有 adapters，包括已完成请求的
            # 必须保护它们，因为 RPC 端的 batch 对象还持有对它们的引用
            original_adapter_dirs = batch.adapter_dirs.copy() if not self.input_params.no_lora else None
            
            # 过滤掉已完成的请求，只保留未完成的请求
            # 同时会更新 batch.adapter_dirs，只包含未完成请求使用的适配器
            batch.filter_finished()

            # 减少完成请求的适配器的当前请求计数
            if finished_adapter_dirs and not self.input_params.no_lora:
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].decrease_request_counts(finished_adapter_dirs))
                await asyncio.gather(*ret)

            # PEFT 调度器特殊处理：当批次完全清空时，需要取消合并适配器
            # PEFT 模式下适配器会合并到基础模型中，清空时需要恢复
            if self.input_params.scheduler == "peft" and batch.is_clear():
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                await asyncio.gather(*ret)

            # ===== 新的智能淘汰策略 =====
            # 策略：基于阈值触发淘汰，而非立即淘汰所有不在批次中的适配器
            # 优点：
            #   1. 保留热门适配器，减少重复加载
            #   2. 只在内存压力大时才淘汰
            #   3. 基于分数智能选择淘汰对象
            # 注意：minibatch 时不淘汰，避免频繁操作
            if not minibatch and not self.input_params.no_lora:
                # 输出淘汰前的状态
                await self._print_lora_status("请求完成时")
                
                # 调试日志：打印保护列表
                if original_adapter_dirs:
                    print(f"   🔒 保护的适配器 ({len(original_adapter_dirs)} 个): {[d.split('/')[-1] for d in list(original_adapter_dirs)[:10]]}")
                else:
                    print(f"   ⚠️  警告：original_adapter_dirs 为空或 None")
                
                ret = []
                for tp_rank in range(self.world_size):
                    # 使用阈值触发淘汰，保护批次的原始 adapter_dirs
                    # 重要：必须使用 original_adapter_dirs（filter_finished 之前的）
                    # 因为 RPC 端的 batch 对象还持有对这些 adapters 的引用
                    # 直到 filter_batch RPC 调用同步 RPC 端的状态
                    # 阈值和淘汰比例可通过命令行参数配置
                    ret.append(self.model_rpcs[tp_rank].trigger_threshold_eviction(
                        preserve_dirs=original_adapter_dirs,
                        threshold=self.input_params.evict_interval_threshold,
                        evict_ratio=self.input_params.evict_interval_ratio,
                        max_lora_ratio=self.input_params.max_lora_ratio
                    ))
                evict_results = await asyncio.gather(*ret)

                # 输出淘汰结果摘要
                if evict_results and evict_results[0]:
                    self._print_eviction_summary(evict_results[0], "请求完成时")
                    
                    # 新增：如果执行了淘汰，更新实际占用
                    if evict_results[0].get('evicted'):
                        await self._update_actual_adapter_usage()

            # 根据批次状态决定后续操作
            if batch.is_clear():
                # 批次完全清空，移除批次
                await self._remove_batch(batch)
            else:
                # 批次还有未完成的请求，过滤批次（移除已完成的请求）
                await self._filter_batch(batch)
        return

    async def _filter_runing_batch(self):
        """
        检查并清理运行中的批次
        
        当运行批次完全清空时（所有请求都已完成）：
        1. 触发阈值淘汰检查（而非立即全部卸载）
        2. 清空运行批次引用
        
        这个函数在每次推理步骤后都会被调用，用于及时清理已完成的批次
        
        优化说明：
        - 旧策略：批次清空时立即卸载所有适配器
        - 新策略：只在内存压力大时才淘汰，保留热门适配器
        - 优点：减少重复加载，提高吞吐量
        """
        if self.running_batch is not None and self.running_batch.is_clear():
            # 批次完全清空，但不立即卸载所有适配器
            # 使用阈值淘汰机制，只在内存使用率较高时才清理
            if not self.input_params.no_lora:
                # 输出淘汰前的状态
                await self._print_lora_status("批次空闲时")
                
                ret = []
                for tp_rank in range(self.world_size):
                    # 使用更高的阈值，只在接近满载时才淘汰
                    # 不保护任何适配器（preserve_dirs=None）
                    # 淘汰比例较大，释放更多空间
                    # 阈值和淘汰比例可通过命令行参数配置
                    ret.append(self.model_rpcs[tp_rank].trigger_threshold_eviction(
                        preserve_dirs=None,
                        threshold=self.input_params.evict_idle_threshold,
                        evict_ratio=self.input_params.evict_idle_ratio,
                        max_lora_ratio=self.input_params.max_lora_ratio
                    ))
                evict_results = await asyncio.gather(*ret)

                # 输出淘汰结果摘要
                if evict_results and evict_results[0]:
                    self._print_eviction_summary(evict_results[0], "批次空闲时")
                    
                    # 新增：如果执行了淘汰，更新实际占用
                    if evict_results[0].get('evicted'):
                        await self._update_actual_adapter_usage()

            # 清空运行批次引用，允许调度器生成新的批次
            self.running_batch = None
            return
    
    async def _print_lora_status(self, context: str = ""):
        """
        输出 LoRA 内存使用状态
        
        参数:
            context: 上下文信息（如"请求完成时"、"批次空闲时"等）
        """
        if self.input_params.no_lora:
            return
        
        try:
            # 查询第一个 GPU 的内存状态（多卡时通常状态相似）
            usage = await self.model_rpcs[0].check_lora_memory()
            if usage is None:
                return
            
            # 计算 LoRA 和 KV cache 的各自占用
            lora_occupied = sum(usage['adapter_cells']) if usage['adapter_cells'] else 0
            kv_occupied = usage['used_cells'] - lora_occupied
            
            context_str = f"[{context}] " if context else ""
            print(f"\n📊 {context_str}内存池状态:")
            print(f"   总池使用率: {usage['usage_ratio']:.1%} "
                  f"({usage['used_cells']}/{usage['total_cells']} cells)")
            
            # 计算 LoRA 使用率（基于 max_lora_ratio 上限）
            max_lora_ratio = self.input_params.max_lora_ratio if hasattr(self.input_params, 'max_lora_ratio') else None
            if max_lora_ratio and max_lora_ratio > 0:
                lora_max_cells = int(usage['total_cells'] * max_lora_ratio)
            else:
                lora_max_cells = usage['total_cells']
            lora_usage_pct = lora_occupied / lora_max_cells if lora_max_cells > 0 else 0.0
            
            print(f"   ├─ LoRA 占用: {lora_occupied}/{lora_max_cells} cells ({lora_usage_pct:.1%})")
            print(f"   └─ KV Cache 占用: {kv_occupied} cells ({kv_occupied/usage['total_cells']:.1%})")
            print(f"   已加载适配器: {usage['num_adapters']} 个")
        except Exception as e:
            # 静默失败，不影响主流程
            pass
    
    def _print_eviction_summary(self, evict_result: dict, context: str = ""):
        """
        输出淘汰结果摘要
        
        参数:
            evict_result: trigger_threshold_eviction 返回的结果字典
            context: 上下文信息（如"请求完成时"、"批次空闲时"等）
        """
        if not evict_result or not evict_result.get('triggered'):
            return
        
        context_str = f"[{context}] " if context else ""
        
        if evict_result.get('evicted'):
            before = evict_result.get('before_usage', {})
            after = evict_result.get('after_usage', {})
            print(f"\n✅ {context_str}阈值淘汰完成:")
            print(f"   淘汰数量: {evict_result.get('evicted_count', 0)} 个适配器")
            print(f"   释放空间: {evict_result.get('cells_freed', 0)} cells")
            if before and after:
                print(f"   LoRA 使用率变化: {before.get('lora_usage_ratio', 0):.1%} → {after.get('lora_usage_ratio', 0):.1%}")
        else:
            reason = evict_result.get('reason', 'unknown')
            reason_map = {
                'below_threshold': '内存使用率未超过阈值',
                'no_adapters': '没有加载任何适配器',
                'no_candidates': '所有适配器都在使用中或受保护'
            }
            reason_str = reason_map.get(reason, reason)
            print(f"\n⏭️  {context_str}未执行淘汰: {reason_str}")
    
    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
    
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 4:
                adapter_dir, prompt_ids, sampling_params, request_id = recv_req
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
    """
    启动路由进程 - 支持张量并行和数据并行模式
    
    根据 args.parallel_mode 参数选择启动逻辑：
    - 'tensor' 或默认: 使用原有的张量并行逻辑 (RouterManager)
    - 'data': 使用新的数据并行逻辑 (DataParallelRouterManager)
    
    Requirements:
        - 6.1: 未指定 parallel-mode 参数时默认使用张量并行模式
        - 6.2: 使用 --parallel-mode tensor 时使用原有的张量并行逻辑
        - 6.3: 使用 --parallel-mode data 时使用新的数据并行逻辑
    """
    # 获取并行模式，默认为 'tensor'
    parallel_mode = getattr(args, 'parallel_mode', 'tensor')
    
    # Requirement 6.5: 在启动日志中明确输出当前使用的并行模式
    print(f"[Router] Starting router process in {parallel_mode.upper()} parallel mode")
    
    # 根据并行模式选择启动逻辑
    if parallel_mode == 'data':
        # 数据并行模式 - 使用 DataParallelRouterManager
        _start_data_parallel_router(args, router_port, detokenization_port, pipe_writer)
    else:
        # 张量并行模式（默认）- 使用原有的 RouterManager
        _start_tensor_parallel_router(args, router_port, detokenization_port, 
                                      model_rpc_ports, mode, pipe_writer)


def _start_tensor_parallel_router(args, router_port, detokenization_port, 
                                  model_rpc_ports, mode, pipe_writer):
    """
    启动张量并行路由器（原有逻辑）
    
    这是原有的 start_router_process 函数的逻辑，保持向后兼容性。
    
    Requirements:
        - 6.2: 使用 --parallel-mode tensor 时使用原有的张量并行逻辑
        - 6.4: 保持现有的 API 接口不变
    """
    input_params = InputParams(max_req_total_len=args.max_req_total_len,
                               # kv cache manager parameters
                               max_total_token_num=args.max_total_token_num,
                               pool_size_lora=args.pool_size_lora,
                               batch_max_tokens=args.batch_max_tokens,
                               running_max_req_size=args.running_max_req_size,
                               # heuristic
                               swap=args.swap,
                               prefetch=args.prefetch,
                               prefetch_size=args.prefetch_size,
                               scheduler=args.scheduler,
                               profile=args.profile,
                               batch_num_adapters=args.batch_num_adapters,
                               enable_abort=args.enable_abort,
                               # mem_ratio=args.mem_ratio,
                               dummy=args.dummy,
                               no_lora_swap=args.no_lora_swap,
                               no_lora_compute=args.no_lora_compute,
                               no_kernel=args.no_kernel,
                               no_mem_pool=args.no_mem_pool,
                               bmm=args.bmm,
                               no_lora=args.no_lora,
                               fair_weights=args.fair_weights,
                               # eviction parameters
                               evict_interval_threshold=args.evict_interval_threshold,
                               evict_interval_ratio=args.evict_interval_ratio,
                               evict_idle_threshold=args.evict_idle_threshold,
                               evict_idle_ratio=args.evict_idle_ratio,
                               max_lora_ratio=args.max_lora_ratio,
                              )

    try:
        router = RouterManager(
            args.model_dir,
            args.lora_dirs,
            load_way="HF",
            world_size=args.tp,
            eos_id=args.eos_id,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            input_params=input_params,
            mode=mode,
            log_stats = not args.disable_log_stats,
            log_stats_interval = args.log_stats_interval,
        )
    
        asyncio.run(router.wait_to_model_ready())
        if input_params.profile:
            asyncio.run(router.profile_prefill())
        if input_params.scheduler == "pets" and input_params.profile:
            router.req_queue.alpha = router.alpha_model
            router.req_queue.beta = router.beta_model
        elif input_params.scheduler == "pets":
            # loading from file
            cache_dir = os.path.expanduser("~/.cache/slora")
            router.req_queue.alpha = AlphaModel.from_file(cache_dir+"/profile_results.pkl")
            router.req_queue.beta = BetaModel.from_file(cache_dir+"/profile_results.pkl")
    
    except Exception as e:
        import traceback
        import sys
        err_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return


def _start_data_parallel_router(args, router_port, detokenization_port, pipe_writer):
    """
    启动数据并行路由器（新逻辑）
    
    使用 DataParallelRouterManager 启动多个 GPU Worker 进程，
    并使用 Round Robin 策略路由请求。
    
    Requirements:
        - 6.3: 使用 --parallel-mode data 时使用新的数据并行逻辑
        - 1.1: 根据配置创建指定数量的 GPU Worker 进程
        - 5.2: 支持通过 --num-workers 参数指定 Worker 数量
        - 5.3: 支持通过 --gpu-ids 参数指定使用的 GPU 列表
        - 6.5: 在启动日志中输出当前并行模式
        - 8.1: 输出 Worker 就绪日志
    """
    from slora.server.router.dp_manager import DataParallelRouterManager
    from slora.utils.net_utils import alloc_can_use_network_port
    
    try:
        # 分配 response_port（用于 Worker 发送响应到 Response Merger）
        # 简单方案：使用 router_port + 1（如果被占用则继续尝试）
        import socket
        
        def find_free_port(start_port, exclude_ports):
            """查找可用端口"""
            port = start_port
            while port < start_port + 100:  # 最多尝试 100 个端口
                if port in exclude_ports:
                    port += 1
                    continue
                try:
                    # 尝试绑定端口
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', port))
                        return port
                except OSError:
                    port += 1
            raise RuntimeError(f"Failed to find free port starting from {start_port}")
        
        # 查找可用的 response_port
        response_port = find_free_port(
            router_port + 1, 
            exclude_ports={router_port, detokenization_port}
        )
        
        print(f"[DataParallelRouter] Allocated response_port: {response_port}")
        print(f"[DataParallelRouter] Router port: {router_port}")
        print(f"[DataParallelRouter] Detokenization port: {detokenization_port}")
        
        # Requirement 6.5 & 8.1: 输出启动配置信息
        # 在创建 manager 之前输出配置，因为 manager 初始化时会解析这些参数
        num_workers = getattr(args, 'num_workers', None)
        gpu_ids_str = getattr(args, 'gpu_ids', None)
        
        print("=" * 80)
        print("[DataParallelRouter] Starting Data Parallel Mode")
        print("=" * 80)
        if num_workers:
            print(f"[DataParallelRouter] Number of Workers: {num_workers} (specified)")
        else:
            print(f"[DataParallelRouter] Number of Workers: Auto-detect (using all available GPUs)")
        
        if gpu_ids_str:
            print(f"[DataParallelRouter] GPU IDs: {gpu_ids_str} (specified)")
        else:
            print(f"[DataParallelRouter] GPU IDs: Auto-assign (0, 1, 2, ...)")
        
        # 输出路由策略配置
        routing_strategy = getattr(args, 'routing_strategy', 'round-robin')
        print(f"[DataParallelRouter] Routing Strategy: {routing_strategy}")
        if routing_strategy == 'adapter-aware':
            routing_w1 = getattr(args, 'routing_w1', 1.0)
            routing_w2 = getattr(args, 'routing_w2', 4.0)
            routing_w3 = getattr(args, 'routing_w3', 0.0)
            default_lora_rank = getattr(args, 'default_lora_rank', 16)
            max_queue_length = getattr(args, 'max_queue_length', 100)
            hot_adapter_threshold = getattr(args, 'hot_adapter_threshold', 1e9)
            print(f"[DataParallelRouter]   w1 (cache affinity): {routing_w1}")
            print(f"[DataParallelRouter]   w2 (load penalty): {routing_w2}")
            print(f"[DataParallelRouter]   w3 (rank mismatch penalty): {routing_w3}")
            print(f"[DataParallelRouter]   default_lora_rank: {default_lora_rank}")
            print(f"[DataParallelRouter]   max_queue_length: {max_queue_length}")
            print(f"[DataParallelRouter]   hot_adapter_threshold: {hot_adapter_threshold} req/s")
        print("=" * 80)
        
        # 创建 DataParallelRouterManager 实例
        dp_manager = DataParallelRouterManager(
            args=args,
            router_port=router_port,
            response_port=response_port,
            detoken_port=detokenization_port
        )
        
        # 分配端口（必须在 _setup_zmq 之前）
        dp_manager._allocate_ports()
        
    except Exception as e:
        import traceback
        import sys
        err_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        pipe_writer.send(err_str)
        raise
    
    # 创建一个事件循环，所有异步操作都在这个循环中运行
    # 这样可以确保 ZMQ 异步 context 和 sockets 在同一个事件循环中使用
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def run_all():
        """在同一个事件循环中运行所有异步操作"""
        try:
            # 设置 ZMQ 通信（需要 worker_ports 已经分配）
            dp_manager._setup_zmq()
            
            # 启动所有 Worker 和 Response Merger
            await dp_manager.start_workers()
            
            # Requirement 6.5 & 8.1: 输出启动完成摘要
            print("=" * 80)
            print("[DataParallelRouter] Data Parallel Mode Started Successfully")
            print("=" * 80)
            print(f"[DataParallelRouter] Number of Workers: {dp_manager.num_workers}")
            print(f"[DataParallelRouter] GPU IDs: {dp_manager.gpu_ids}")
            print(f"[DataParallelRouter] Worker Ports: {dp_manager.worker_ports}")
            print(f"[DataParallelRouter] All workers are ready and accepting requests")
            print("=" * 80)
            
            # 发送 'init ok' 信号，通知主进程可以启动 HTTP Server
            pipe_writer.send('init ok')
            
            # 运行主循环
            await dp_manager.run()
            
        except Exception as e:
            import traceback
            import sys
            err_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            pipe_writer.send(err_str)
            raise
    
    loop.run_until_complete(run_all())
    return
