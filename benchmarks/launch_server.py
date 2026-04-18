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
    parser.add_argument("--routing-w2", type=float, default=4.0,
                        help="负载惩罚权重 (默认: 1.0, RWPT/Capacity 归一化到 ~[0,1])")
    parser.add_argument("--routing-w3", type=float, default=0.0,
                        help="Rank 不匹配惩罚权重，用于 rank 感知路由 (默认: 0.0，禁用)")
    parser.add_argument("--load-metric", type=str, default="rwpt",
                        choices=["queue_length", "token_count", "rwpt"],
                        help="负载度量类型 (默认: rwpt)")
    parser.add_argument("--default-lora-rank", type=int, default=16,
                        help="未知 adapter 的默认 LoRA rank (默认: 16)")
    parser.add_argument("--max-queue-length", type=int, default=100,
                        help="最大队列长度阈值 (默认: 100)")
    parser.add_argument("--hot-adapter-threshold", type=float, default=10.0,
                        help="热点 Adapter 请求率阈值，单位 req/s (默认: 10.0)")
    
    # RWPT (Rank-Weighted Pending Tokens) 相关参数
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="模型隐藏层维度，用于计算 LoRA rank 加权系数 γ=2/(3·d) (默认: 从模型 config.json 自动检测，检测失败时回退 4096)")
    parser.add_argument("--decode-cost-alpha", type=float, default=None,
                        help="Decode 序列负载折算系数 (默认: None, 由 Worker 运行时 profiling 自动测量)")
    
    # 热门 Adapter 主动复制相关参数
    parser.add_argument("--enable-replication", action="store_true",
                        help="启用热门 Adapter 主动复制机制 (默认: 禁用)")
    parser.add_argument("--cooldown-sec", type=float, default=5.0,
                        help="同一 adapter 两次复制的最小间隔秒数 (默认: 5.0)")
    parser.add_argument("--patrol-interval-sec", type=float, default=1.0,
                        help="ReplicaManager 巡检周期秒数 (默认: 1.0)")
    parser.add_argument("--protection-sec", type=float, default=30.0,
                        help="新副本的淘汰保护时长秒数 (默认: 30.0)")
    
    # 阈值淘汰相关参数
    parser.add_argument("--evict-interval-threshold", type=float, default=0.85,
                        help="请求完成时触发淘汰的内存使用率阈值 (0-1)")
    parser.add_argument("--evict-interval-ratio", type=float, default=0.3,
                        help="请求完成时的淘汰比例 (0-1)")
    parser.add_argument("--evict-idle-threshold", type=float, default=0.95,
                        help="批次空闲时触发淘汰的内存使用率阈值 (0-1)")
    parser.add_argument("--evict-idle-ratio", type=float, default=0.5,
                        help="批次空闲时的淘汰比例 (0-1)")
    parser.add_argument("--max-lora-ratio", type=float, default=0.2,
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
        
        # 添加路由策略参数（数据并行模式）
        if args.routing_strategy:
            cmd += f" --routing-strategy {args.routing_strategy}"
        if args.routing_w1 != 1.0:
            cmd += f" --routing-w1 {args.routing_w1}"
        if args.routing_w2 != 4.0:
            cmd += f" --routing-w2 {args.routing_w2}"
        if args.routing_w3 != 0.0:
            cmd += f" --routing-w3 {args.routing_w3}"
        if hasattr(args, 'load_metric') and args.load_metric != "rwpt":
            cmd += f" --load-metric {args.load_metric}"
        if args.default_lora_rank != 16:
            cmd += f" --default-lora-rank {args.default_lora_rank}"
        if args.max_queue_length != 100:
            cmd += f" --max-queue-length {args.max_queue_length}"
        if args.hot_adapter_threshold != 10.0:
            cmd += f" --hot-adapter-threshold {args.hot_adapter_threshold}"
        
        # 添加 RWPT 参数
        if args.hidden_dim is not None:
            cmd += f" --hidden-dim {args.hidden_dim}"
        if args.decode_cost_alpha is not None:
            cmd += f" --decode-cost-alpha {args.decode_cost_alpha}"

        # 添加热门 Adapter 主动复制参数
        if args.enable_replication:
            cmd += " --enable-replication"
            cmd += f" --cooldown-sec {args.cooldown_sec}"
            cmd += f" --patrol-interval-sec {args.patrol_interval_sec}"
            cmd += f" --protection-sec {args.protection_sec}"

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
    import subprocess
    import shlex
    result = subprocess.run(shlex.split(cmd), check=False)
    sys.exit(result.returncode)
