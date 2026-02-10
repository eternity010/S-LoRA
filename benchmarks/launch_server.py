import argparse
import os
import psutil
import sys

from exp_suite import BASE_MODEL, LORA_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="debug")
    parser.add_argument("--backend", type=str, default="slora",
                        choices=["slora", "vllm", "lightllm", "vllm-packed"])
    parser.add_argument("--model-setting", type=str, default="S1")

    parser.add_argument("--num-adapter", type=int)
    parser.add_argument("--num-token", type=int)

    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    parser.add_argument("--bmm", action="store_true")
    parser.add_argument("--batch-num-adapters", type=int, default=None)
    parser.add_argument("--enable-abort", action="store_true")
    parser.add_argument("--vllm-mem-ratio", type=float, default=0.95)
    
    # 数据并行相关参数
    parser.add_argument("--parallel-mode", type=str, default="tensor", 
                        choices=["tensor", "data"],
                        help="并行模式: tensor (张量并行) 或 data (数据并行)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="数据并行模式下的 Worker 数量")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="数据并行模式下使用的 GPU ID，逗号分隔，如 '0,1,2'")
    
    # 路由策略相关参数（数据并行模式）
    parser.add_argument("--routing-strategy", type=str, default="round-robin",
                        choices=["round-robin", "adapter-aware"],
                        help="路由策略: round-robin (轮询，默认) 或 adapter-aware (基于亲和性)")
    parser.add_argument("--routing-w1", type=float, default=1.0,
                        help="缓存亲和性权重 (默认: 1.0)")
    parser.add_argument("--routing-w2", type=float, default=0.1,
                        help="负载惩罚权重 (默认: 0.1)")
    parser.add_argument("--max-queue-length", type=int, default=100,
                        help="最大队列长度阈值 (默认: 100)")
    parser.add_argument("--hot-adapter-threshold", type=float, default=10.0,
                        help="热点 Adapter 请求率阈值，单位 req/s (默认: 10.0)")
    
    # 阈值淘汰相关参数
    parser.add_argument("--evict-interval-threshold", type=float, default=0.85,
                        help="请求完成时触发淘汰的内存使用率阈值 (0-1)")
    parser.add_argument("--evict-interval-ratio", type=float, default=0.3,
                        help="请求完成时的淘汰比例 (0-1)")
    parser.add_argument("--evict-idle-threshold", type=float, default=0.95,
                        help="批次空闲时触发淘汰的内存使用率阈值 (0-1)")
    parser.add_argument("--evict-idle-ratio", type=float, default=0.5,
                        help="批次空闲时的淘汰比例 (0-1)")
    parser.add_argument("--max-lora-ratio", type=float, default=0.4,
                        help="LoRA 占用总内存的最大比例 (0-1)")
    
    args = parser.parse_args()

    base_model = BASE_MODEL[args.model_setting]
    adapter_dirs = LORA_DIR[args.model_setting]

    if args.device == "a10g":
        if args.num_adapter is None: args.num_adapter = 200
        if args.num_token is None: args.num_token = 14000
    elif args.device == "h100":
        if args.num_adapter is None: args.num_adapter = 1000
        if args.num_token is None: args.num_token = 120000
    elif args.device == "debug":
        if args.num_adapter is None: args.num_adapter = 30
        if args.num_token is None: args.num_token = 14000
        if args.no_mem_pool:
            args.num_token -= 64 * 4 * 18
    

    if args.backend == "slora":
        cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
        cmd += f" --model {base_model}"
        cmd += f" --tokenizer_mode auto"

        num_iter = args.num_adapter // len(adapter_dirs) + 1
        for i in range(num_iter):
            for adapter_dir in adapter_dirs:
                cmd += f" --lora {adapter_dir}-{i}"

        if args.dummy:
            cmd += " --dummy"
        cmd += " --swap"
        # cmd += " --scheduler pets"
        # cmd += " --profile"
        if args.enable_abort:
            cmd += " --enable-abort"
        if args.batch_num_adapters:
            cmd += f" --batch-num-adapters {args.batch_num_adapters}"
        if args.no_lora_compute:
            cmd += " --no-lora-compute"
        if args.prefetch:
            cmd += " --prefetch"
        if args.no_mem_pool:
            cmd += " --no-mem-pool"
        # cmd += " --no-lora-copy"
        # cmd += " --no-kernel"
        if args.bmm:
            cmd += " --bmm"
        
        # 添加数据并行参数
        if args.parallel_mode:
            cmd += f" --parallel-mode {args.parallel_mode}"
        if args.num_workers:
            cmd += f" --num-workers {args.num_workers}"
        if args.gpu_ids:
            cmd += f" --gpu-ids {args.gpu_ids}"
        
        # 添加阈值淘汰参数
        cmd += f" --evict-interval-threshold {args.evict_interval_threshold}"
        cmd += f" --evict-interval-ratio {args.evict_interval_ratio}"
        cmd += f" --evict-idle-threshold {args.evict_idle_threshold}"
        cmd += f" --evict-idle-ratio {args.evict_idle_ratio}"
        cmd += f" --max-lora-ratio {args.max_lora_ratio}"

    elif args.backend == "lightllm":
        cmd = f"python -m lightllm.server.api_server" \
              f" --model_dir {base_model} --tp 1 --max_total_token_num {args.num_token}" \
              f" --tokenizer_mode auto" \
              f" --host 127.0.0.1 --port 8000"

    elif args.backend == "vllm":
        cmd = f"python -m vllm.entrypoints.api_server" \
              f" --model {base_model} --swap-space 16" \
              f" --disable-log-requests" \
              f" --host 127.0.0.1 --port 8000"

    elif args.backend == "vllm-packed":
        gpu_memory = args.vllm_mem_ratio / (args.num_adapter)

        for i in range(args.num_adapter):
            pid = os.fork()
            if pid == 0:
                cmd = f"python -m vllm.entrypoints.api_server" \
                    f" --model {base_model} --swap-space 0" \
                    f" --gpu-memory-utilization {gpu_memory}" \
                    f" --disable-log-requests" \
                    f" --host 127.0.0.1 --port {8000 + i}"
                os.system(cmd)
                sys.exit(0)

        for _ in range(args.num_adapter):
            os.wait()

        sys.exit(0)

    # print(cmd)
    os.system(cmd)
