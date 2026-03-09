"""
Experiment runner module for automated experiment execution.
"""

import asyncio
import subprocess
import time
import json
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import requests

from .config import ExperimentConfig
from .suite import ExperimentSuite
from .result import ExperimentResult, ExperimentRecord


class ExperimentRunner:
    """Experiment runner for automated execution"""
    
    def __init__(self,
                 output_dir: str = "routing_comparison_results",
                 model_setting: str = "Real",
                 server_host: str = "http://localhost:8000",
                 benchmarks_dir: str = ".",
                 model_dir: str = None,
                 adapter_dir: str = None,
                 debug: bool = False):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
            model_setting: Model setting ("Real" or "Dummy")
            server_host: Server URL
            benchmarks_dir: Path to benchmarks directory
            model_dir: Path to base model (optional)
            adapter_dir: Path to adapter directory (optional)
            debug: Enable debug logging to file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_setting = model_setting
        self.server_host = server_host
        self.benchmarks_dir = Path(benchmarks_dir)
        self.model_dir = model_dir
        self.adapter_dir = adapter_dir
        self.debug = debug
        
        self.server_process: Optional[subprocess.Popen] = None
        self.server_log = None
        self._current_log_file = self.output_dir / "server_log.txt"
        self._latest_log_file = self.output_dir / "server_log.txt"
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self._current_server_config = None  # Track current server configuration
        
        # Debug log file
        self._debug_log = None
        if self.debug:
            debug_file = Path(__file__).parent / "debug.log"
            self._debug_log = open(debug_file, 'w')
            self._log(f"ExperimentRunner initialized: output={output_dir}, model={model_setting}")
        
        # Clean up old logs at framework startup
        self._cleanup_old_logs()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _cleanup_old_logs(self) -> None:
        """Clean up old log files at framework startup"""
        log_dir = self.output_dir / "logs"
        if log_dir.exists():
            import shutil
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup and remove all existing log files
            for log_file in log_dir.glob("server_*.log"):
                # Skip backup files (those with timestamp in name)
                if log_file.stem.count('_') > 2:  # e.g., server_round-robin_adapters100_20240101_120000
                    continue
                backup_file = log_dir / f"{log_file.stem}_{timestamp}.log"
                shutil.copy(log_file, backup_file)
                log_file.unlink()
                self._log(f"backed up old log: {log_file.name} -> {backup_file.name}")
        
        # Clean up latest log symlink
        latest_log = self.output_dir / "server_log.txt"
        if latest_log.exists():
            latest_log.unlink()
    
    def _log(self, msg: str) -> None:
        """Write debug message to log file"""
        if self._debug_log:
            ts = time.strftime("%H:%M:%S")
            self._debug_log.write(f"[{ts}] {msg}\n")
            self._debug_log.flush()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        self._log("SIGINT received, cleaning up")
        print("\n\nReceived interrupt signal. Cleaning up...")
        self._stop_server()
        if self._debug_log:
            self._debug_log.close()
        sys.exit(0)
    
    def _get_server_config_key(self, config: ExperimentConfig) -> str:
        """
        Get a unique key for server configuration.
        
        Only parameters that affect server startup are included.
        Alpha is NOT included because it only affects client request distribution.
        w1/w2/w3 are NOT included because they can be updated dynamically via API.
        """
        return (
            f"{config.routing_strategy}_"
            f"workers{config.num_workers}_"
            f"gpus{config.gpu_ids}_"
            f"adapters{config.num_adapters}_"
            f"tokens{config.num_token}"
        )
    
    def _needs_server_restart(self, config: ExperimentConfig) -> bool:
        """
        Check if server needs to be restarted for this config.
        
        Returns True if:
        - No server is running
        - Server configuration has changed
        
        Returns False if:
        - Server is running with same configuration (alpha change doesn't matter)
        """
        if self.server_process is None:
            return True
        
        new_key = self._get_server_config_key(config)
        return new_key != self._current_server_config
    
    def run_suite(self, suite_name: str, resume: bool = False) -> None:
        """
        Run entire experiment suite with server reuse optimization.
        
        Server is only restarted when configuration changes (routing strategy,
        num_adapters, etc.). Alpha changes don't require restart since they
        only affect client request distribution.
        
        Execution flow:
        1. Start server for first experiment
        2. Run all experiments with same server config sequentially
        3. Stop server when config changes
        4. Start new server for next group of experiments
        5. Repeat until all experiments complete
        
        Args:
            suite_name: Name of the suite to run
            resume: Whether to resume from checkpoint
        """
        self._log(f"run_suite: {suite_name}, resume={resume}")
        configs = list(ExperimentSuite.get_configs(suite_name))
        total = len(configs)
        
        # Load checkpoint if resuming
        completed = set()
        if resume and self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                completed = set(checkpoint.get('completed', []))
            print(f"Resuming from checkpoint: {len(completed)} experiments already completed")
            self._log(f"resume: {len(completed)} completed")
        
        print(f"\nRunning suite '{suite_name}' with {total} experiments")
        print("=" * 70)
        
        try:
            for i, config in enumerate(configs, 1):
                # Create unique ID for this config (include w1/w2 for routing-weight-comparison)
                config_id = f"{config.routing_strategy}_alpha{config.alpha}_adapters{config.num_adapters}_w1{config.routing_w1}_w2{config.routing_w2}_lm{config.load_metric}"
                
                if config_id in completed:
                    print(f"\n[{i}/{total}] Skipping (already completed): {config_id}")
                    self._log(f"skip: {config_id}")
                    continue
                
                print(f"\n[{i}/{total}] Running: {config.routing_strategy}, "
                      f"alpha={config.alpha}, adapters={config.num_adapters}")
                print("-" * 70)
                self._log(f"exp[{i}/{total}]: {config_id}")
                
                try:
                    # Check if we need to restart server
                    needs_restart = self._needs_server_restart(config)
                    self._log(f"needs_restart={needs_restart}")
                    
                    # If server config changed, stop old server first
                    if needs_restart and self.server_process is not None:
                        print("  Server configuration changed, stopping old server...")
                        self._log("stopping old server")
                        self._stop_server()
                    
                    result = self.run_single_experiment(
                        config, 
                        start_server=needs_restart,
                        stop_server=False  # Keep server running for next experiment
                    )
                    
                    # Wait for server to finish processing and cool down
                    print("  Waiting for server to stabilize (5s)...")
                    time.sleep(5)
                    
                    self._log(f"success: tput={result.throughput:.2f}, lat={result.avg_latency:.3f}")
                    print(f"✓ Success: throughput={result.throughput:.2f} req/s, "
                          f"latency={result.avg_latency:.3f}s")
                    
                    # Update checkpoint
                    completed.add(config_id)
                    self._save_checkpoint(list(completed))
                    
                except Exception as e:
                    self._log(f"failed: {e}")
                    print(f"✗ Failed: {e}")
                    # On failure, stop server to ensure clean state for next experiment
                    self._stop_server()
                    continue
        
        finally:
            # Always stop server at the end of suite
            print("\n" + "=" * 70)
            print("Stopping server...")
            self._log("suite done, stopping server")
            self._stop_server()
            if self._debug_log:
                self._debug_log.close()
                self._debug_log = None
        
        print(f"Suite completed: {len(completed)}/{total} experiments successful")
    
    def run_single_experiment(self, config: ExperimentConfig, 
                             start_server: bool = True,
                             stop_server: bool = True) -> ExperimentResult:
        """
        Run single experiment with optional server management.
        
        Args:
            config: Experiment configuration
            start_server: Whether to start server (default: True)
            stop_server: Whether to stop server after experiment (default: True)
        
        Returns:
            ExperimentResult with collected metrics
        
        Raises:
            Exception: If experiment fails
        """
        config.validate()
        
        if start_server:
            print("  1. Starting server...")
            self._start_server(config)
            
            print("  2. Waiting for server to be ready...")
            self._wait_for_server()
        else:
            print("  1-2. Reusing existing server (no restart needed)")
            
            # 重置 adapter cache，确保实验公平性
            print("  1.5. Resetting adapter cache for fair comparison...")
            self._reset_adapter_cache()
        
        # 动态更新路由配置 (w1/w2/w3/load_metric)，无需重启服务器
        if config.routing_strategy == 'adapter-aware':
            print("  2.5. Updating routing config (w1/w2/w3/load_metric)...")
            self._update_routing_config(
                w1=config.routing_w1,
                w2=config.routing_w2,
                w3=config.routing_w3,
                load_metric=config.load_metric,
                reset_stats=True  # 重置统计以获得干净的实验数据
            )
        
        # Collect stats before benchmark to calculate delta
        stats_before = self._collect_routing_stats()
        
        print("  3. Running benchmark...")
        result = self._run_benchmark(config)
        
        print("  4. Collecting routing statistics...")
        stats_after = self._collect_routing_stats()
        
        # Calculate delta for worker_request_counts
        routing_stats = self._calculate_stats_delta(stats_before, stats_after)
        
        if stop_server:
            print("  5. Stopping server...")
            self._stop_server()
        else:
            print("  5. Keeping server running for next experiment...")
        
        print("  6. Saving results...")
        experiment_result = self._create_result(config, result, routing_stats)
        self._save_result(config, experiment_result)
        
        return experiment_result
    
    def _start_server(self, config: ExperimentConfig) -> None:
        """Start S-LoRA server after cleaning up any existing S-LoRA processes"""
        self._log(f"start_server: {config.routing_strategy}, adapters={config.num_adapters}")
        # Store gpu_ids for reference
        self.gpu_ids = config.gpu_ids
        
        # Clean up any existing S-LoRA processes (safe, won't kill other programs)
        print("     Cleaning up any existing S-LoRA processes...")
        self._cleanup_slora_processes(max_wait=30)
        
        cmd = [
            sys.executable,  # Use current Python interpreter
            "launch_server.py",
            "--model-setting", self.model_setting
        ]
        cmd.extend(config.to_server_args())
        
        # Add model and adapter paths if provided
        if self.model_dir:
            cmd.extend(["--model-dir", self.model_dir])
        if self.adapter_dir:
            cmd.extend(["--adapter-dir", self.adapter_dir])
        
        print(f"     Command: {' '.join(cmd)}")
        
        # Create log directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file based on SERVER configuration (not experiment config)
        # This way, multiple experiments with same server config share one log
        # Simplify the key for filename
        log_filename = f"server_{config.routing_strategy}_adapters{config.num_adapters}.log"
        log_file = log_dir / log_filename
        
        # Also maintain a latest log symlink/copy for easy monitoring
        latest_log = self.output_dir / "server_log.txt"
        
        # Note: Old logs are cleaned up at framework startup in _cleanup_old_logs()
        # Here we just append to existing log if server restarts during same run
        
        print(f"     Server log: {log_file}")
        print(f"     (This log will be shared by all experiments with same server config)")
        
        self.server_log = open(log_file, 'w')
        self._current_log_file = log_file
        self._latest_log_file = latest_log
        
        # Start server in a new process group so we can kill all children together
        self.server_process = subprocess.Popen(
            cmd,
            cwd=str(self.benchmarks_dir),
            stdout=self.server_log,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            start_new_session=True  # Create new process group for clean termination
        )
        
        # Track current server configuration
        self._current_server_config = self._get_server_config_key(config)
    
    def _wait_for_server(self, timeout: int = 4200) -> None:
        """
        Wait for server to be ready.
        
        S-LoRA has a built-in mechanism that ensures the HTTP server only starts
        after the model is fully loaded. The main process waits for "init ok" 
        messages from router and detokenization processes before starting Uvicorn.
        
        Therefore, when /health endpoint responds, the model is guaranteed to be ready.
        
        Args:
            timeout: Maximum wait time in seconds (default: 4200s = 70 minutes)
                    Loading 100 adapters can take 30-60 minutes on RTX 3090 in data parallel mode
        """
        start = time.time()
        log_file = self._current_log_file
        
        print(f"     Waiting for server (timeout: {timeout}s)...")
        print(f"     Server log: {log_file}")
        print(f"     Note: Server will only start after model is fully loaded")
        
        # Check if process started successfully
        time.sleep(5)
        if self.server_process.poll() is not None:
            raise RuntimeError("Server process terminated immediately")
        
        # Monitor log file for progress indicators
        last_log_size = 0
        last_progress_time = start
        
        while time.time() - start < timeout:
            elapsed = time.time() - start
            
            # Check if process is still running
            if self.server_process.poll() is not None:
                # Print last lines of log for debugging
                print(f"\n     ✗ Server process terminated unexpectedly after {elapsed:.1f}s")
                print(f"     Exit code: {self.server_process.returncode}")
                print(f"     Last 30 lines of server log:")
                print("     " + "=" * 60)
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-30:]:
                            print(f"     {line.rstrip()}")
                print("     " + "=" * 60)
                raise RuntimeError(f"Server process terminated unexpectedly after {elapsed:.1f}s")
            
            # Read and display log progress
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        f.seek(last_log_size)
                        new_content = f.read()
                        last_log_size = f.tell()
                        
                        if new_content:
                            # Display progress indicators
                            progress_keywords = [
                                ("Loading", "Loading model..."),
                                ("loading", "Loading..."),
                                ("Initializing", "Initializing..."),
                                ("initializing", "Initializing..."),
                                ("init ok", "Model loaded successfully"),
                                ("Uvicorn running", "HTTP server starting..."),
                            ]
                            
                            for keyword, message in progress_keywords:
                                if keyword in new_content:
                                    current_time = time.time()
                                    if current_time - last_progress_time > 5:  # Throttle output
                                        print(f"     [{elapsed:.0f}s] {message}")
                                        last_progress_time = current_time
                except Exception:
                    pass  # Ignore file read errors
            
            # Try to connect to health endpoint
            try:
                resp = requests.get(f"{self.server_host}/health", timeout=5)
                if resp.status_code == 200:
                    # Server is ready! (model is guaranteed to be loaded)
                    self._log(f"server ready in {elapsed:.1f}s")
                    print(f"     ✓ Server ready in {elapsed:.1f}s")
                    print(f"     ✓ Model loaded and HTTP server started")
                    return
            except requests.RequestException:
                pass
            
            time.sleep(5)
        
        # Timeout - print server logs for debugging
        print(f"\n     ✗ Server failed to start within {timeout}s")
        print(f"     Last 50 lines of server log:")
        print("     " + "=" * 60)
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(f"     {line.rstrip()}")
        print("     " + "=" * 60)
        
        raise TimeoutError(f"Server failed to start within {timeout}s")
    
    def _run_benchmark(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Run benchmark using run_exp.py logic.
        
        Generates synthetic requests based on config parameters and sends them
        to the server, collecting latency metrics.
        """
        import asyncio
        import numpy as np
        import aiohttp
        
        # Import from benchmarks modules
        sys.path.insert(0, str(self.benchmarks_dir))
        from exp_suite import BASE_MODEL, LORA_DIR
        from trace import generate_requests
        
        print(f"     Running {config.duration}s benchmark with {config.num_adapters} adapters...")
        print(f"     Request rate: {config.req_rate} req/s, Alpha: {config.alpha}")
        
        # Prepare adapter directories
        base_model = BASE_MODEL[self.model_setting]
        adapter_dirs_template = LORA_DIR[self.model_setting]
        
        # Generate adapter directory list (same logic as run_exp.py)
        adapter_dirs = []
        num_iter = config.num_adapters // len(adapter_dirs_template) + 1
        for i in range(num_iter):
            for adapter_dir in adapter_dirs_template:
                adapter_dirs.append(adapter_dir + f"-{i}")
        adapter_dirs = adapter_dirs[:config.num_adapters]
        adapter_dirs = [(base_model, adapter_dirs[i]) for i in range(config.num_adapters)]
        
        # Generate requests
        requests = generate_requests(
            num_adapters=config.num_adapters,
            alpha=config.alpha,
            req_rate=config.req_rate,
            cv=config.cv,
            duration=config.duration,
            input_range=config.input_range,
            output_range=config.output_range,
            adapter_dirs=adapter_dirs,
            seed=42
        )
        
        total_requests = len(requests)
        print(f"     Generated {total_requests} requests")
        
        # Run benchmark
        benchmark_start_time = time.time()
        per_req_latency = asyncio.run(self._async_benchmark(requests))
        benchmark_end_time = time.time()
        benchmark_time = benchmark_end_time - benchmark_start_time
        
        # Calculate statistics
        return self._calculate_benchmark_stats(per_req_latency, benchmark_time, config.req_rate)
    
    async def _async_benchmark(self, requests) -> list:
        """Run async benchmark, sending requests at specified times"""
        import aiohttp
        
        start = time.time()
        tasks = []
        
        for req in requests:
            # Wait until request time
            wait_time = start + req.req_time - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            task = asyncio.create_task(
                self._send_request(req.model_dir, req.adapter_dir, req.prompt,
                                   req.prompt_len, req.output_len)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                print(f"     Request failed: {r}")
                valid_results.append((0, 0, 0, None))  # Mark as failed
            else:
                valid_results.append(r)
        
        return valid_results
    
    async def _send_request(self, model_dir: str, adapter_dir: str, prompt: str,
                           prompt_len: int, output_len: int) -> tuple:
        """Send a single request to the server"""
        import aiohttp
        
        request_start_time = time.time()
        url = f"{self.server_host}/generate_stream"
        
        data = {
            'model_dir': model_dir,
            'lora_dir': adapter_dir,
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'ignore_eos': True,
                'max_new_tokens': output_len,
            }
        }
        
        first_token_latency = None
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=data) as response:
                    async for chunk, _ in response.content.iter_chunks():
                        if first_token_latency is None:
                            first_token_latency = time.time() - request_start_time
        except Exception as e:
            return (prompt_len, output_len, 0, None)
        
        request_latency = time.time() - request_start_time
        return (prompt_len, output_len, request_latency, first_token_latency)
    
    def _calculate_benchmark_stats(self, per_req_latency: list, benchmark_time: float,
                                   req_rate: float) -> Dict[str, Any]:
        """Calculate benchmark statistics from per-request latency data"""
        import numpy as np
        
        # Filter out failed requests
        num_abort = len([r for r in per_req_latency if r[3] is None])
        valid_latency = [r for r in per_req_latency if r[3] is not None]
        
        if not valid_latency:
            print(f"     ⚠ All {len(per_req_latency)} requests failed!")
            return {
                "throughput": 0.0,
                "strip_throughput": 0.0,
                "avg_latency": 0.0,
                "avg_first_token_latency": 0.0,
                "p50_latency": 0.0,
                "p90_latency": 0.0,
                "total_requests": len(per_req_latency),
                "num_abort": num_abort,
            }
        
        # Calculate throughput
        throughput = len(valid_latency) / benchmark_time
        
        # Strip throughput (exclude warmup)
        warmup_time = 10
        warmup_num = int(req_rate * warmup_time)
        if len(valid_latency) > warmup_num * 2:
            strip_throughput = (len(valid_latency) - warmup_num * 2) / (benchmark_time - warmup_time * 2)
        else:
            strip_throughput = throughput
        
        # Latency statistics
        latencies = [r[2] for r in valid_latency]
        first_token_latencies = [r[3] for r in valid_latency]
        
        avg_latency = np.mean(latencies)
        avg_first_token_latency = np.mean(first_token_latencies)
        
        # Total latency percentiles
        p50_latency = np.percentile(latencies, 50)
        p90_latency = np.percentile(latencies, 90)
        
        # First token latency percentiles
        p50_first_token_latency = np.percentile(first_token_latencies, 50)
        p90_first_token_latency = np.percentile(first_token_latencies, 90)
        
        print(f"     Completed: {len(valid_latency)}/{len(per_req_latency)} requests")
        print(f"     Throughput: {throughput:.2f} req/s (strip: {strip_throughput:.2f})")
        print(f"     Avg latency: {avg_latency:.2f}s, First token: {avg_first_token_latency:.2f}s")
        print(f"     Total P50/P90: {p50_latency:.2f}s / {p90_latency:.2f}s")
        print(f"     First token P50/P90: {p50_first_token_latency:.2f}s / {p90_first_token_latency:.2f}s")
        
        return {
            "throughput": throughput,
            "strip_throughput": strip_throughput,
            "avg_latency": avg_latency,
            "avg_first_token_latency": avg_first_token_latency,
            "p50_latency": p50_latency,
            "p90_latency": p90_latency,
            "p50_first_token_latency": p50_first_token_latency,
            "p90_first_token_latency": p90_first_token_latency,
            "total_requests": len(per_req_latency),
            "num_abort": num_abort,
        }
    
    def _collect_routing_stats(self) -> Dict[str, Any]:
        """Collect routing statistics from server"""
        try:
            resp = requests.get(f"{self.server_host}/routing_stats", timeout=10)
            if resp.status_code == 200:
                stats = resp.json()
                print(f"     Routing stats collected: {len(stats)} entries")
                return stats
        except requests.RequestException as e:
            print(f"     Warning: Could not collect routing stats: {e}")
        
        return {}
    
    def _update_routing_config(self, w1: float = None, w2: float = None, 
                               w3: float = None, load_metric: str = None,
                               reset_stats: bool = True) -> bool:
        """
        动态更新路由配置参数

        通过 POST /update_routing_config API 更新服务器的路由配置，
        无需重启服务器。

        Args:
            w1: 缓存亲和性权重 (可选)
            w2: 负载惩罚权重 (可选)
            w3: Rank 不匹配惩罚权重 (可选)
            load_metric: 负载度量类型 (可选, queue_length/token_count/rwpt)
            reset_stats: 是否重置统计信息 (默认 True)

        Returns:
            bool: 更新是否成功
        """
        # 构建请求体，只包含非 None 的参数
        data = {}
        if w1 is not None:
            data['w1'] = w1
        if w2 is not None:
            data['w2'] = w2
        if w3 is not None:
            data['w3'] = w3
        if load_metric is not None:
            data['load_metric'] = load_metric
        data['reset_stats'] = reset_stats

        if not data or (len(data) == 1 and 'reset_stats' in data):
            self._log("_update_routing_config: no parameters to update")
            return True

        self._log(f"_update_routing_config: {data}")

        try:
            resp = requests.post(
                f"{self.server_host}/update_routing_config",
                json=data,
                timeout=10
            )

            if resp.status_code == 200:
                result = resp.json()
                print(f"     Routing config updated: {result.get('config', {})}")
                self._log(f"config update success: {result}")

                # 等待配置生效 (服务器每 5 秒检查一次配置文件)
                # 使用 10 秒确保至少经过一个完整轮询周期，
                # 避免 stats_before 采集时 reset 尚未执行
                time.sleep(10)
                return True
            else:
                error = resp.json() if resp.content else {}
                print(f"     Warning: Failed to update routing config: {error}")
                self._log(f"config update failed: {resp.status_code} {error}")
                return False

        except requests.RequestException as e:
            print(f"     Warning: Could not update routing config: {e}")
            self._log(f"config update error: {e}")
            return False
    
    def _reset_adapter_cache(self) -> bool:
        """
        重置所有 Worker 的 Adapter 缓存
        
        用于实验间的 cache 重置，确保实验公平性。
        
        Returns:
            bool: 重置是否成功
        """
        self._log("_reset_adapter_cache: starting")
        
        try:
            resp = requests.post(
                f"{self.server_host}/reset_adapter_cache",
                json={},
                timeout=30  # 给足够的时间让所有 Worker 完成
            )
            
            if resp.status_code == 200:
                result = resp.json()
                print(f"     Adapter cache reset: {result.get('message', 'success')}")
                self._log(f"cache reset success: {result}")
                
                # 额外等待一小段时间确保状态同步
                time.sleep(1)
                return True
            else:
                error = resp.json() if resp.content else {}
                print(f"     Warning: Failed to reset adapter cache: {error}")
                self._log(f"cache reset failed: {resp.status_code} {error}")
                return False
                
        except requests.RequestException as e:
            print(f"     Warning: Could not reset adapter cache: {e}")
            self._log(f"cache reset error: {e}")
            return False
    
    def _calculate_stats_delta(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate delta between two stats snapshots for per-experiment values"""
        result = after.copy()
        
        # Calculate delta for worker_request_counts
        before_counts = before.get('worker_request_counts', {})
        after_counts = after.get('worker_request_counts', {})
        
        delta_counts = {}
        server_restarted = False
        
        for worker_id, count in after_counts.items():
            before_count = before_counts.get(worker_id, 0)
            # Handle both int and string keys
            if isinstance(before_count, str):
                before_count = int(before_count)
            if isinstance(count, str):
                count = int(count)
            
            delta = count - before_count
            # Detect server restart: if delta is negative, server was restarted
            if delta < 0:
                server_restarted = True
                delta = count  # Use current value directly
            delta_counts[worker_id] = delta
        
        # If server restarted, use after values directly for all counts
        if server_restarted:
            for worker_id, count in after_counts.items():
                if isinstance(count, str):
                    count = int(count)
                delta_counts[worker_id] = count
        
        result['worker_request_counts'] = delta_counts
        
        # Calculate delta for cache_hits and cache_misses
        before_hits = before.get('cache_hits', 0)
        after_hits = after.get('cache_hits', 0)
        before_misses = before.get('cache_misses', 0)
        after_misses = after.get('cache_misses', 0)
        
        delta_hits = after_hits - before_hits
        delta_misses = after_misses - before_misses
        
        # Detect server restart for cache stats
        if delta_hits < 0 or delta_misses < 0 or server_restarted:
            delta_hits = after_hits
            delta_misses = after_misses
        
        delta_total = delta_hits + delta_misses
        
        result['cache_hits'] = delta_hits
        result['cache_misses'] = delta_misses
        
        # Recalculate cache_hit_rate for this experiment only
        if delta_total > 0:
            result['cache_hit_rate'] = delta_hits / delta_total
        else:
            result['cache_hit_rate'] = 0.0
        
        return result
    
    def _stop_server(self) -> None:
        """Stop server and all child processes by killing the process group"""
        self._log("stop_server called")
        if self.server_process:
            pid = self.server_process.pid
            try:
                # Kill the entire process group (all children spawned by launch_server.py)
                import os
                pgid = os.getpgid(pid)
                self._log(f"killing pgid {pgid}")
                print(f"     Killing process group {pgid}...")
                os.killpg(pgid, signal.SIGTERM)
                
                # Wait for graceful termination
                self.server_process.wait(timeout=10)
                self._log("server stopped gracefully")
                print("     Server process group terminated gracefully")
            except subprocess.TimeoutExpired:
                self._log("SIGKILL needed")
                print("     Process group did not stop gracefully, sending SIGKILL...")
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    self.server_process.wait(timeout=5)
                except Exception:
                    pass
            except (ProcessLookupError, OSError) as e:
                self._log(f"already terminated: {e}")
                print(f"     Process already terminated: {e}")
            finally:
                self.server_process = None
                self._current_server_config = None  # Clear config tracking
        
        # Close log file if open
        if hasattr(self, 'server_log') and self.server_log:
            try:
                self.server_log.close()
                self.server_log = None
                
                # Copy to latest log for easy access
                if hasattr(self, '_current_log_file') and hasattr(self, '_latest_log_file'):
                    import shutil
                    if self._current_log_file.exists():
                        shutil.copy2(self._current_log_file, self._latest_log_file)
            except:
                pass
        
        # Clean up any remaining S-LoRA processes (safety net)
        print("     Verifying all S-LoRA processes are terminated...")
        self._cleanup_slora_processes(max_wait=30)
    

    def _get_slora_pids(self) -> set:
        """
        Get PIDs of S-LoRA related processes only.
        
        Only matches processes with S-LoRA specific patterns to avoid
        killing other GPU programs.
        """
        pids = set()
        
        # S-LoRA specific process patterns (based on actual module paths)
        slora_patterns = [
            'slora\\.server\\.api_server',           # Main API server
            'slora\\.server\\.router\\.manager',     # Router manager (tensor parallel)
            'slora\\.server\\.router\\.dp_manager',  # Data parallel router manager
            'slora\\.server\\.router\\.gpu_worker',  # GPU workers (data parallel)
            'slora\\.server\\.detokenization',       # Detokenization process
            'slora\\.mprophet\\.model_rpc',          # Model RPC workers
            'run_gpu_worker_process',                # GPU worker process function
        ]
        
        for pattern in slora_patterns:
            try:
                result = subprocess.run(
                    ['pgrep', '-f', pattern],
                    capture_output=True, text=True
                )
                if result.stdout.strip():
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            pids.add(line.strip())
            except Exception:
                pass
        
        # Also find orphaned multiprocessing spawn processes on our GPUs
        # These are left behind when parent process dies unexpectedly
        pids.update(self._get_orphan_gpu_processes())
        
        return pids
    
    def _get_orphan_gpu_processes(self) -> set:
        """
        Find orphaned Python multiprocessing spawn processes on our GPUs.
        
        These processes have ppid=1 (orphaned) and command contains
        'multiprocessing.spawn'. Only kills processes on GPUs we're using.
        """
        pids = set()
        gpu_ids = [int(g) for g in self.gpu_ids.split(',')] if hasattr(self, 'gpu_ids') and self.gpu_ids else []
        
        if not gpu_ids:
            return pids
        
        try:
            # Get PIDs on our GPUs
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode != 0 or not result.stdout.strip():
                return pids
            
            gpu_pids = set()
            for line in result.stdout.strip().split('\n'):
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    pid = parts[0].strip()
                    if pid:
                        gpu_pids.add(pid)
            
            # Check each GPU process
            for pid in gpu_pids:
                try:
                    # Check if it's an orphan (ppid=1) and multiprocessing spawn
                    result = subprocess.run(
                        ['ps', '-p', pid, '-o', 'ppid,cmd', '--no-headers'],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        parts = result.stdout.strip().split(None, 1)
                        if len(parts) >= 2:
                            ppid = parts[0]
                            cmd = parts[1]
                            # Orphan process with multiprocessing spawn
                            if ppid == '1' and 'multiprocessing.spawn' in cmd:
                                pids.add(pid)
                except Exception:
                    pass
        except Exception:
            pass
        
        return pids
    

    def _cleanup_slora_processes(self, max_wait: int = 30) -> None:
        """
        Clean up S-LoRA processes only (safe, won't kill other GPU programs).
        
        Kills only processes matching S-LoRA specific patterns, then waits
        until they are confirmed terminated.
        
        Args:
            max_wait: Maximum seconds to wait for processes to terminate
        """
        import os
        my_pid = str(os.getpid())
        
        # Step 1: Kill all S-LoRA processes
        pids = self._get_slora_pids()
        if pids:
            print(f"     Found {len(pids)} S-LoRA processes to clean up")
            for pid in pids:
                if pid != my_pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"     Killed S-LoRA process {pid}")
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass
        else:
            print(f"     No S-LoRA processes found")
            return
        
        # Step 2: Wait and verify processes are terminated
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(2)
            
            remaining_pids = self._get_slora_pids()
            remaining_pids.discard(my_pid)
            
            if not remaining_pids:
                elapsed = time.time() - start
                print(f"     ✓ All S-LoRA processes terminated in {elapsed:.1f}s")
                # Extra wait for GPU driver to release memory
                time.sleep(5)
                return
            
            print(f"     Still {len(remaining_pids)} S-LoRA processes remaining, killing again...")
            for pid in remaining_pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, ValueError):
                    pass
        
        # Timeout warning
        remaining = self._get_slora_pids()
        remaining.discard(my_pid)
        if remaining:
            print(f"     ⚠ {len(remaining)} S-LoRA processes still running after {max_wait}s: {remaining}")
    
    def _create_result(self,
                       config: ExperimentConfig,
                       benchmark_result: Dict[str, Any],
                       routing_stats: Dict[str, Any]) -> ExperimentResult:
        """Create ExperimentResult from benchmark data"""
        
        # Extract routing statistics (already delta values from _calculate_stats_delta)
        cache_hits = routing_stats.get('cache_hits', 0)
        cache_misses = routing_stats.get('cache_misses', 0)
        cache_hit_rate = routing_stats.get('cache_hit_rate', 0.0)
        total_requests = benchmark_result.get('total_requests', 0)
        
        # Extract worker load distribution (already delta values)
        worker_counts = routing_stats.get('worker_request_counts', {})
        if not worker_counts:
            # Default: assume equal distribution
            num_workers = config.num_workers
            per_worker = total_requests // num_workers if total_requests > 0 else 0
            worker_counts = {str(i): per_worker for i in range(num_workers)}
        
        return ExperimentResult(
            routing_strategy=config.routing_strategy,
            alpha=config.alpha,
            num_adapters=config.num_adapters,
            throughput=benchmark_result.get('throughput', 0.0),
            strip_throughput=benchmark_result.get('strip_throughput', 0.0),
            avg_latency=benchmark_result.get('avg_latency', 0.0),
            avg_first_token_latency=benchmark_result.get('avg_first_token_latency', 0.0),
            p50_latency=benchmark_result.get('p50_latency', 0.0),
            p90_latency=benchmark_result.get('p90_latency', 0.0),
            p50_first_token_latency=benchmark_result.get('p50_first_token_latency', 0.0),
            p90_first_token_latency=benchmark_result.get('p90_first_token_latency', 0.0),
            cache_hit_rate=cache_hit_rate,
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            worker_request_counts=worker_counts,
        )
    
    def _save_result(self, config: ExperimentConfig, result: ExperimentResult) -> None:
        """Save experiment result"""
        result_file = self.output_dir / "results.jsonl"
        
        record = ExperimentRecord(
            config=config.__dict__,
            result=result,
            timestamp=time.time()
        )
        
        record.save_to_jsonl(str(result_file), append=True)
        print(f"     Result saved to {result_file}")
    
    def _save_checkpoint(self, completed: list) -> None:
        """Save checkpoint for resume capability"""
        checkpoint = {
            'completed': completed,
            'timestamp': time.time()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_results(self) -> list:
        """Load all results from output directory"""
        result_file = self.output_dir / "results.jsonl"
        if not result_file.exists():
            return []
        
        return ExperimentRecord.load_from_jsonl(str(result_file))
