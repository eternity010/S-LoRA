from .sampling_params import SamplingParams
from typing import Dict, List, Optional, Tuple
import asyncio


class Req:
    def __init__(self, adapter_dir, request_id, prompt_ids, sample_params: SamplingParams):
        self.adapter_dir = adapter_dir
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.max_output_len = sample_params.max_new_tokens
        self.sample_params = sample_params
        self.output_ids = []
        self.output_metadata_list = []
        self.has_generate_finished = False
        self.aborted = False

    def to_rpc_obj(self):
        return {"adapter_dir": self.adapter_dir,
                "request_id": self.request_id,
                "input_id": self.prompt_ids,
                "output_len": self.max_output_len,
                "sampling_param": self.sample_params.to_dict() }

    def to_req_detokenization_state(self):
        out = ReqDetokenizationState(self.request_id, self.prompt_ids, self.max_output_len, self.sample_params.ignore_eos)
        if self.output_metadata_list:
            out.gen_metadata.update(self.output_metadata_list[-1])
        return out
    
    def stop_sequences_matched(self):
        for stop_token_ids in self.sample_params.stop_sequences:
            stop_len = len(stop_token_ids)
            if stop_len > 0:
                if len(self.output_ids) >= stop_len:
                    if all(self.output_ids[-(stop_len - i)] == stop_token_ids[i] for i in range(stop_len)):
                        return True
        return False

    def __repr__(self):
        return (f"request_id(n={self.request_id}, "
                f"adapter_dir={self.adapter_dir}, ")
                # f"prompt_ids={self.prompt_ids}, ")
        

class ReqDetokenizationState:
    def __init__(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_output_len: int,
        ignore_eos: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.max_output_len = max_output_len
        self.ignore_eos = ignore_eos
        self.gen_metadata = {}


class Batch:
    def __init__(self, batch_id, reqs: List[Req]):
        """
        初始化批次
        
        批次包含多个请求，每个请求可能使用不同的 LoRA 适配器。
        adapter_dirs 集合记录了当前批次中所有请求使用的适配器目录。
        这个集合在适配器淘汰时作为保留列表使用。
        """
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}

        # 收集批次中所有请求使用的适配器目录（去重）
        # 这个集合用于：
        # 1. 确定需要加载哪些适配器
        # 2. 在请求完成时，确定需要保留哪些适配器（淘汰策略）
        self.adapter_dirs = set()
        for req in reqs:
            self.adapter_dirs.add(req.adapter_dir)

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def calcu_max_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + req.max_output_len
        return tokens
    
    def calcu_used_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + len(req.output_ids)
        return tokens

    def mark_finished_req(self, eos_id):
        has_new_finish = False
        for req in self.reqs:
            if req.stop_sequences_matched():
                req.has_generate_finished = True
                has_new_finish = True
            # 检查是否有输出 token，避免空列表访问
            if len(req.output_ids) > 0 and req.output_ids[-1] == eos_id and req.sample_params.ignore_eos == False:
                req.has_generate_finished = True
                has_new_finish = True
            if len(req.output_ids) >= req.max_output_len or req.aborted:
                req.has_generate_finished = True
                has_new_finish = True
        return has_new_finish

    def filter_finished(self):
        """
        过滤掉已完成的请求，只保留未完成的请求
        
        这个方法在请求完成时被调用，用于：
        1. 从批次中移除已完成的请求
        2. 更新 adapter_dirs 集合，只包含未完成请求使用的适配器
        3. 更新后的 adapter_dirs 会作为适配器淘汰的保留列表
        
        注意：这个方法会更新 adapter_dirs，影响后续的适配器淘汰决策
        """
        # 筛选出未完成的请求
        unfinished_req = []
        for req in self.reqs:
            if not req.has_generate_finished:
                unfinished_req.append(req)
        
        # 更新请求列表和索引映射
        self.reqs = unfinished_req
        self.id_to_reqs = {req.request_id: req for req in self.reqs}

        # 重新计算 adapter_dirs，只包含未完成请求使用的适配器
        # 这个更新后的集合会被传递给 offload_adapters() 作为保留列表
        # 因此，只有未完成请求使用的适配器会被保留，其他适配器会被淘汰
        self.adapter_dirs = set()
        for req in self.reqs:
            self.adapter_dirs.add(req.adapter_dir)

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
            self.adapter_dirs.add(_req.adapter_dir)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def __repr__(self):
        return (f"batch_id={self.batch_id}, "
                # f"reqs={self.reqs}, "
                f"req_ids={self.id_to_reqs.keys()}")
        
class BatchTokenIdOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, int, Dict, bool, bool]] = []  # [req_id, new_token_id, gen_metadata, finished_state, abort_state]

class BatchStrOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, str, Dict, bool, bool]] = [] # [req_id, token_str, gen_metadata, finished_state, abort_state]
        
class AbortReq:
    def __init__(self, req_id):
        self.req_id = req_id

class BatchAbortReq:
    def __init__(self, req_ids):
        self.reqs: List[str] = req_ids
