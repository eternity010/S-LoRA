"""
Experiment configuration module for routing strategy comparison.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ExperimentConfig:
    """Single experiment configuration"""
    
    # Routing strategy
    routing_strategy: str  # "round-robin" | "adapter-aware"
    
    # Workload parameters
    num_adapters: int
    alpha: float  # Power Law distribution parameter
    req_rate: float  # Request rate (req/s)
    duration: int  # Experiment duration (seconds)
    cv: float = 1.0  # Coefficient of variation
    
    # Token length ranges
    input_range: Tuple[int, int] = (128, 512)
    output_range: Tuple[int, int] = (64, 256)
    
    # Environment configuration
    gpu_ids: str = "1,2,3"
    num_workers: int = 3
    num_token: int = 15000
    
    # Router parameters (adapter-aware only)
    routing_w1: float = 1.0  # Cache affinity weight
    routing_w2: float = 4.0  # Load penalty weight (Load/Capacity, Load = prefill tokens only)
    routing_w3: float = 0.0  # Rank mismatch penalty

    # Memory parameters
    max_lora_ratio: float = 0.2  # Max LoRA memory ratio (0-1)
    
    # Load metric ablation
    load_metric: str = "rwpt"  # 'queue_length' | 'token_count' | 'rwpt'
    
    # Hot adapter replication
    enable_replication: bool = False  # 是否启用主动复制
    cooldown_sec: float = 5.0        # 同一 adapter 两次复制最小间隔
    patrol_interval_sec: float = 1.0  # 巡检周期
    protection_sec: float = 30.0      # 新副本淘汰保护时长
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.routing_strategy not in ["round-robin", "adapter-aware"]:
            raise ValueError(
                f"Invalid routing_strategy: {self.routing_strategy}, "
                f"must be 'round-robin' or 'adapter-aware'"
            )
        
        if not 0.1 <= self.alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0.1, 1.0], got {self.alpha}")
        
        if not 10 <= self.num_adapters <= 200:
            raise ValueError(
                f"num_adapters must be in [10, 200], got {self.num_adapters}"
            )
        
        if self.req_rate <= 0:
            raise ValueError(f"req_rate must be positive, got {self.req_rate}")
        
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")
        
        if self.num_token <= 0:
            raise ValueError(f"num_token must be positive, got {self.num_token}")

        if not 0.0 < self.max_lora_ratio < 1.0:
            raise ValueError(f"max_lora_ratio must be in (0, 1), got {self.max_lora_ratio}")
        
        if self.load_metric not in ("queue_length", "token_count", "rwpt"):
            raise ValueError(f"Invalid load_metric: {self.load_metric}")
    
    def to_server_args(self) -> List[str]:
        """Convert to launch_server.py command line arguments"""
        args = [
            "--parallel-mode", "data",
            "--num-workers", str(self.num_workers),
            "--gpu-ids", self.gpu_ids,
            "--num-adapter", str(self.num_adapters),
            "--num-token", str(self.num_token),
            "--routing-strategy", self.routing_strategy,
            "--max-lora-ratio", str(self.max_lora_ratio),
        ]
        
        if self.routing_strategy == "adapter-aware":
            args.extend([
                "--routing-w1", str(self.routing_w1),
                "--routing-w2", str(self.routing_w2),
                "--routing-w3", str(self.routing_w3),
            ])
        
        if self.load_metric != "rwpt":
            args.extend(["--load-metric", self.load_metric])
        
        if self.enable_replication:
            args.append("--enable-replication")
            args.extend(["--cooldown-sec", str(self.cooldown_sec)])
            args.extend(["--patrol-interval-sec", str(self.patrol_interval_sec)])
            args.extend(["--protection-sec", str(self.protection_sec)])
        
        return args
    
    def to_benchmark_args(self) -> dict:
        """Convert to run_exp.py parameters"""
        return {
            "num_adapters": self.num_adapters,
            "alpha": self.alpha,
            "req_rate": self.req_rate,
            "duration": self.duration,
            "cv": self.cv,
            "input_range": self.input_range,
            "output_range": self.output_range,
        }
