# Adapted from vllm/entrypoints/api_server.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
import torch
import uvloop
import sys

from .build_prompt import build_prompt

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import argparse
import json
from http import HTTPStatus
import uuid
import multiprocessing as mp
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
from .sampling_params import SamplingParams
from .httpserver.manager import HttpServerManager
from .detokenization.manager import start_detokenization_process
from .router.manager import start_router_process

from slora.utils.net_utils import alloc_can_use_network_port
from slora.common.configs.config import setting
from .api_models import (
    ChatCompletionRequest,
    UsageInfo,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    DeltaMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
)

from slora.mprophet.measure import ModelProphet
from slora.mprophet.lora_stats import LoRAProphet


GB = 1024 ** 3
MB = 1024 ** 2

TIMEOUT_KEEP_ALIVE = 5  # seconds.

app = FastAPI()

isFirst = True


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code.value)


@app.get("/healthz")
@app.get("/health")
def healthcheck():
    return "OK"

@app.get("/routing_stats")
def routing_stats():
    """Get routing statistics from the router process"""
    import json
    import os
    
    stats_file = "/tmp/slora_routing_stats.json"
    
    if not os.path.exists(stats_file):
        return JSONResponse({
            "error": "Stats file not found",
            "message": "Router may not be running in data parallel mode or stats not yet available"
        }, status_code=404)
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "message": "Failed to read stats file"
        }, status_code=500)


@app.post("/update_routing_config")
async def update_routing_config(request: Request):
    """
    动态更新路由配置参数
    
    请求体 JSON 格式:
    {
        "w1": float,        # 缓存亲和性权重 (可选)
        "w2": float,        # 负载惩罚权重 (可选)
        "w3": float,        # Rank 不匹配惩罚权重 (可选)
        "reset_stats": bool # 是否重置统计信息 (可选, 默认 false)
    }
    
    返回:
    - 200: 配置更新请求已提交
    - 400: 请求参数无效
    - 500: 写入配置文件失败
    """
    import json
    import os
    
    config_update_file = "/tmp/slora_routing_config_update.json"
    
    try:
        request_dict = await request.json()
    except Exception as e:
        return JSONResponse({
            "error": "Invalid JSON",
            "message": str(e)
        }, status_code=400)
    
    # 验证参数
    valid_keys = {"w1", "w2", "w3", "load_metric", "reset_stats"}
    invalid_keys = set(request_dict.keys()) - valid_keys
    if invalid_keys:
        return JSONResponse({
            "error": "Invalid parameters",
            "message": f"Unknown parameters: {invalid_keys}. Valid parameters: {valid_keys}"
        }, status_code=400)
    
    # 验证数值参数非负
    for key in ["w1", "w2", "w3"]:
        if key in request_dict:
            value = request_dict[key]
            if not isinstance(value, (int, float)):
                return JSONResponse({
                    "error": "Invalid parameter type",
                    "message": f"{key} must be a number, got {type(value).__name__}"
                }, status_code=400)
            if value < 0:
                return JSONResponse({
                    "error": "Invalid parameter value",
                    "message": f"{key} must be non-negative, got {value}"
                }, status_code=400)
    
    # 验证 reset_stats 参数
    if "reset_stats" in request_dict:
        if not isinstance(request_dict["reset_stats"], bool):
            return JSONResponse({
                "error": "Invalid parameter type",
                "message": f"reset_stats must be a boolean"
            }, status_code=400)
    
    # 验证 load_metric 参数
    if "load_metric" in request_dict:
        valid_metrics = ("queue_length", "token_count", "rwpt")
        if request_dict["load_metric"] not in valid_metrics:
            return JSONResponse({
                "error": "Invalid load_metric value",
                "message": f"load_metric must be one of {valid_metrics}, got '{request_dict['load_metric']}'"
            }, status_code=400)
    
    # 写入配置文件
    try:
        with open(config_update_file, 'w') as f:
            json.dump(request_dict, f)
        
        return JSONResponse({
            "status": "submitted",
            "message": "Config update request submitted. Changes will be applied within 5 seconds.",
            "config": request_dict
        })
    except Exception as e:
        return JSONResponse({
            "error": "Failed to write config file",
            "message": str(e)
        }, status_code=500)


@app.post("/reset_adapter_cache")
async def reset_adapter_cache(request: Request):
    """
    重置所有 Worker 的 Adapter 缓存
    
    用于实验间的 cache 重置，确保实验公平性。
    清空所有 GPU 上已加载的 adapters，释放显存。
    
    通过文件通信触发 dp_manager 执行重置操作。
    
    请求体 JSON 格式 (可选):
    {
        "wait_seconds": float  # 等待完成的时间 (可选, 默认 3.0)
    }
    
    返回:
    - 200: 重置命令已提交
    - 500: 重置失败
    """
    import json
    import os
    
    reset_trigger_file = "/tmp/slora_reset_adapter_cache.trigger"
    reset_result_file = "/tmp/slora_reset_adapter_cache.result"
    
    try:
        # 解析可选参数
        try:
            request_dict = await request.json()
        except Exception:
            request_dict = {}
        
        wait_seconds = request_dict.get('wait_seconds', 3.0)
        
        # 清除旧的结果文件
        if os.path.exists(reset_result_file):
            os.remove(reset_result_file)
        
        # 写入触发文件
        with open(reset_trigger_file, 'w') as f:
            json.dump({'timestamp': time.time()}, f)
        
        # 等待 dp_manager 处理并写入结果
        await asyncio.sleep(wait_seconds)
        
        # 读取结果
        if os.path.exists(reset_result_file):
            with open(reset_result_file, 'r') as f:
                result = json.load(f)
            
            # 清理文件
            os.remove(reset_trigger_file)
            os.remove(reset_result_file)
            
            if result.get('success', False):
                return JSONResponse({
                    "status": "success",
                    "message": result.get('message', 'Adapter cache reset completed'),
                    "num_workers": result.get('num_workers', 0)
                })
            else:
                return JSONResponse({
                    "status": "error",
                    "message": result.get('message', 'Reset failed'),
                    "error": result.get('error')
                }, status_code=500)
        else:
            # 触发文件已写入，但没有结果文件，可能 dp_manager 还没处理
            return JSONResponse({
                "status": "submitted",
                "message": f"Reset command submitted. Check server logs for completion."
            })
            
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Failed to reset adapter cache: {str(e)}"
        }, status_code=500)


@app.post("/generate")
async def generate(request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    adapter_dir = request_dict["lora_dir"] if "lora_dir" in request_dict else None
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()

    if "req_id" in request_dict:
        request_id = request_dict["req_id"]
    else:
        request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(adapter_dir, prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = []
    count_output_tokens = 0
    tokens = []
    async for request_output, metadata, finished in results_generator:
        count_output_tokens += 1
        if finished == -1:
            return Response(status_code=499)
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            print(f"[API] Client disconnected, aborting request {request_id[:8]}...")
            await httpserver_manager.abort(request_id)
            return Response(status_code=499)
        final_output.append(request_output)
        if return_details:
            metadata["text"] = request_output
            tokens.append(metadata)

    assert final_output is not None
    ret = {
        "generated_text": ["".join(final_output)],
        "count_output_tokens": count_output_tokens,
    }
    if return_details:
        ret["tokens"] = tokens
    return Response(content=json.dumps(ret, ensure_ascii=False).encode("utf-8"))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    adapter_dir = request_dict["lora_dir"] if "lora_dir" in request_dict else None
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()

    if "req_id" in request_dict:
        request_id = request_dict["req_id"]
    else:
        request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(adapter_dir, prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output, metadata, finished in results_generator:
            ret = {
                "token": {
                    "id": metadata.get("id", None),
                    "text": request_output,
                    "logprob": metadata.get("logprob", None),
                    "special": False
                },
                "generated_text": None,
                "finished": finished,
                "details": None
            }

            yield ("data:" + json.dumps(ret, ensure_ascii=False) + f"\n\n").encode(
                "utf-8"
            )

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(
        stream_results(), media_type="text/event-stream", background=background_tasks
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest, raw_request: Request
) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    if request.n > 1:
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "The n parameter currently only supports 1"
        )

    if request.function_call != "none":
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "The function call feature is not supported"
        )

    created_time = int(time.time())
    prompt = await build_prompt(request)
    sampling_params = SamplingParams(
        do_sample=request.do_sample,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
        max_new_tokens=request.max_tokens,
        stop_sequences=request.stop
    )
    sampling_params.verify()

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    if not request.stream:
        final_output = []
        prompt_tokens = -1
        completion_tokens = 0
        async for request_output, metadata in results_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await httpserver_manager.abort(request_id)
                return Response(status_code=499)
            completion_tokens += 1
            if prompt_tokens == -1:
                prompt_tokens = metadata["prompt_tokens"]
            final_output.append(request_output)

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        chat_message = ChatMessage(role="assistant", content="".join(final_output))
        choice = ChatCompletionResponseChoice(index=0, message=chat_message)
        resp = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage
        )
        return resp

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output, metadata in results_generator:
            delta_message = DeltaMessage(role="assistant", content=request_output)

            stream_choice = ChatCompletionStreamResponseChoice(
                index=0, delta=delta_message
            )

            stream_resp = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[stream_choice],
            )
            yield ("data: " + stream_resp.json(ensure_ascii=False) + f"\n\n").encode("utf-8")

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(
        stream_results(), media_type="text/event-stream", background=background_tasks
    )


def print_mem_stats(args):
    model_dir = args.model_dir
    model_name = args.model_dir.split("/")[-1]
    try:
        fake_model = ModelProphet(model_name, model_dir=model_dir)
    except:
        fake_model = ModelProphet(model_name)
    model_size = fake_model.get_model_size()
    print(f"{model_name}: {model_size / GB:.2f} GB")
    peak_working_memory = fake_model.get_peak_working_memory(
            bs=20, context_len=512, tiling_dim=512)
    print(f"peak working mem for (bs=20, seqlen=512): {peak_working_memory / GB:.2f} GB")
    peak_working_memory = fake_model.get_peak_working_memory(
            bs=100, context_len=512, tiling_dim=512)
    print(f"peak working mem for (bs=100, seqlen=512): {peak_working_memory / GB:.2f} GB")
 
    tot_lora_size = 0
    for lora_dir in args.lora_dirs:
        lora_name = lora_dir.split("/")[-1]
        if args.dummy:
            fake_model = LoRAProphet(lora_name, model_name)
            try:
                fake_model = LoRAProphet(lora_name, model_name)
            except NotImplementedError as e:
                fake_model = LoRAProphet(lora_name, model_name,
                                         adapter_dir=lora_dir,
                                         base_model_dir=model_dir)
        else:
            fake_model = LoRAProphet(lora_name, model_name,
                                     adapter_dir=lora_dir,
                                     base_model_dir=model_dir)
        lora_size = fake_model.get_adapter_size()
        tot_lora_size += lora_size
        # print(f"{lora_name}, {base_name}: {lora_size / GB:.3f} GB")
    print(f"all adapters ({len(args.lora_dirs)}) estimated size: {tot_lora_size / GB:.2f} GB")
    print(f"avg adapter estimated size: {tot_lora_size / len(args.lora_dirs) / MB:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--model_dir", type=str, default=None,
                        help="the model weight dir path, the app will load config, weights and tokenizer from this dir")
    parser.add_argument("--tokenizer_mode", type=str, default="slow",
                        help="""tokenizer load mode, can be slow or auto, slow mode load fast but run slow, slow mode is good for debug and test, 
                        when you want to get best performance, try auto mode""")
    parser.add_argument("--max_total_token_num", type=int, default=6000,
                        help="the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)")
    parser.add_argument("--batch_max_tokens", type=int, default=None,
                        help="max tokens num for new cat batch, it control prefill batch size to Preventing OOM")
    parser.add_argument("--eos_id", type=int, default=2,
                        help="eos stop token id")
    parser.add_argument("--running_max_req_size", type=int, default=1000,
                        help="the max size for forward requests in the same time")
    parser.add_argument("--tp", type=int, default=1,
                        help="model tp parral size, the default is 1")
    parser.add_argument("--max_req_input_len", type=int, default=2048,
                        help="the max value for req input tokens num")
    parser.add_argument("--max_req_total_len", type=int, default=2048 + 1024,
                        help="the max value for req_input_len + req_output_len")
    parser.add_argument("--nccl_port", type=int, default=28765,
                        help="the nccl_port to build a distributed environment for PyTorch")
    parser.add_argument("--mode", type=str, default=[], nargs='+',
                        help="Model mode: [int8kv] [int8weight | int4weight]")
    parser.add_argument("--trust_remote_code", action='store_true',
                        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.")
    parser.add_argument("--disable_log_stats", action='store_true',
                        help="disable logging throughput stats.")
    parser.add_argument("--log_stats_interval", type=int, default=10,
                        help="log stats interval in second.")

    ''' slora arguments '''
    parser.add_argument("--lora-dirs", type=str, default=[], action="append",
                        help="the adapter weight dirs associate with base model dir")
    parser.add_argument("--fair-weights", type=int, default=[], action="append")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--pool-size-lora", type=int, default=0)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--prefetch-size", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="slora")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batch-num-adapters", type=int, default=None)
    parser.add_argument("--enable-abort", action="store_true")

    # Data parallel mode arguments
    parser.add_argument("--parallel-mode", type=str, default="tensor", choices=["tensor", "data"],
                        help="Parallel mode: 'tensor' for tensor parallelism (default), 'data' for data parallelism")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of GPU workers for data parallel mode. If not specified, uses all available GPUs")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, uses all available GPUs")
    
    # Routing strategy arguments (for data parallel mode)
    parser.add_argument("--routing-strategy", type=str, default="round-robin", 
                        choices=["round-robin", "adapter-aware"],
                        help="Routing strategy for data parallel mode: 'round-robin' (default) or 'adapter-aware'")
    parser.add_argument("--routing-w1", type=float, default=1.0,
                        help="Cache affinity weight for adapter-aware routing (default: 1.0)")
    parser.add_argument("--routing-w2", type=float, default=4.0,
                        help="Load penalty weight for adapter-aware routing (default: 1.0). "
                             "RWPT/Capacity is normalized to ~[0,1], so w2 should be comparable to w1.")
    parser.add_argument("--routing-w3", type=float, default=0.0,
                        help="Rank mismatch penalty weight for rank-aware routing (default: 0.0, disabled)")
    parser.add_argument("--default-lora-rank", type=int, default=16,
                        help="Default LoRA rank for unknown adapters in rank-aware routing (default: 16)")
    parser.add_argument("--max-queue-length", type=int, default=100,
                        help="Maximum queue length threshold for routing (default: 100)")
    parser.add_argument("--hot-adapter-threshold", type=float, default=10.0,
                        help="Hot adapter request rate threshold in req/s (default: 10.0)")
    
    # RWPT (Rank-Weighted Pending Tokens) 相关参数
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="模型隐藏层维度，用于计算 LoRA rank 加权系数 γ=2/(3·d) (默认: 从模型 config.json 自动检测，检测失败时回退 4096)")
    parser.add_argument("--decode-cost-alpha", type=float, default=None,
                        help="Decode 序列负载折算系数 (默认: None, 由 Worker 运行时 profiling 自动测量)")
    parser.add_argument("--load-metric", type=str, default="rwpt",
                        choices=["queue_length", "token_count", "rwpt"],
                        help="负载度量类型: queue_length (仅队列长度), "
                             "token_count (token 级无 rank 加权), rwpt (完整 RWPT, 默认)")

    # 阈值淘汰相关参数
    parser.add_argument("--evict-interval-threshold", type=float, default=0.85,
                        help="请求完成时触发淘汰的内存使用率阈值 (0-1)，默认 0.85 (85%%)")
    parser.add_argument("--evict-interval-ratio", type=float, default=0.3,
                        help="请求完成时的淘汰比例 (0-1)，默认 0.3 (30%%)")
    parser.add_argument("--evict-idle-threshold", type=float, default=0.95,
                        help="批次空闲时触发淘汰的内存使用率阈值 (0-1)，默认 0.95 (95%%)")
    parser.add_argument("--evict-idle-ratio", type=float, default=0.5,
                        help="批次空闲时的淘汰比例 (0-1)，默认 0.5 (50%%)")
    parser.add_argument("--max-lora-ratio", type=float, default=0.2,
                        help="LoRA 占用总内存的最大比例 (0-1)，剩余空间保留给 KV cache，默认 0.2 (20%%)")

    # debug parameters
    # do not use no-lora-swap, does not rule out the swap over MemAllocator
    parser.add_argument("--no-lora-swap", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-kernel", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    parser.add_argument("--bmm", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    ''' end of slora arguments '''

    args = parser.parse_args()

    assert args.max_req_input_len < args.max_req_total_len
    setting["max_req_total_len"] = args.max_req_total_len
    setting["nccl_port"] = args.nccl_port

    if args.batch_max_tokens is None:
        batch_max_tokens = int(1 / 6 * args.max_total_token_num)
        batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
        args.batch_max_tokens = batch_max_tokens
    else:
        assert (
            args.batch_max_tokens >= args.max_req_total_len
        ), "batch_max_tokens must >= max_req_total_len"

    can_use_ports = alloc_can_use_network_port(
        num=3 + args.tp, used_nccl_port=args.nccl_port
    )
    router_port, detokenization_port, httpserver_port = can_use_ports[0:3]
    model_rpc_ports = can_use_ports[3:]

    global httpserver_manager
    httpserver_manager = HttpServerManager(
        args.model_dir,
        args.tokenizer_mode,
        router_port=router_port,
        httpserver_port=httpserver_port,
        total_token_num=args.max_total_token_num,
        max_req_input_len=args.max_req_input_len,
        max_req_total_len=args.max_req_total_len,
        trust_remote_code=args.trust_remote_code,
        dummy=args.dummy,
    )
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)
    proc_router = mp.Process(
        target=start_router_process,
        args=(
            args,
            router_port,
            detokenization_port,
            model_rpc_ports,
            args.mode,
            pipe_router_writer,
        ),
    )
    proc_router.start()
    proc_detoken = mp.Process(
        target=start_detokenization_process,
        args=(
            args,
            detokenization_port,
            httpserver_port,
            pipe_detoken_writer,
            args.trust_remote_code,
        ),
    )
    proc_detoken.start()

    # wait load model ready
    router_init_state = pipe_router_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if router_init_state != "init ok" or detoken_init_state != "init ok":
        proc_router.kill()
        proc_detoken.kill()
        print(
            "router init state:",
            router_init_state,
            "detoken init state:",
            detoken_init_state,
        )
        sys.exit(1)

    assert proc_router.is_alive() and proc_detoken.is_alive()

    print_mem_stats(args)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        loop="uvloop",
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn'), # this code will not be ok for settings to fork to subprocess
    main()
