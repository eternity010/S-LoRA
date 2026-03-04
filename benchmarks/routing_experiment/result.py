"""
Experiment result data structures and persistence.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ExperimentResult:
    """Experiment result data structure"""
    
    # Configuration information
    routing_strategy: str
    alpha: float
    num_adapters: int
    
    # Throughput metrics
    throughput: float  # Total throughput (req/s)
    strip_throughput: float  # Throughput excluding warmup
    
    # Latency metrics (seconds)
    avg_latency: float
    avg_first_token_latency: float
    p50_latency: float
    p90_latency: float
    
    # Routing statistics
    cache_hit_rate: float
    total_requests: int
    cache_hits: int
    cache_misses: int
    
    # Worker load distribution
    worker_request_counts: Dict[str, int]
    
    # Optional fields with defaults (must come after required fields)
    p50_first_token_latency: float = 0.0
    p90_first_token_latency: float = 0.0
    p95_first_token_latency: float = 0.0
    avg_rank_mismatch: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentResult':
        """
        Create ExperimentResult from dictionary.
        
        Args:
            data: Dictionary containing result data
        
        Returns:
            ExperimentResult instance
        """
        return cls(**data)
    
    def save_to_jsonl(self, filepath: str, append: bool = True) -> None:
        """
        Save result to JSONL file.
        
        Args:
            filepath: Path to JSONL file
            append: If True, append to existing file; if False, overwrite
        """
        mode = 'a' if append else 'w'
        
        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, mode) as f:
            f.write(json.dumps(self.to_dict()) + '\n')
    
    @classmethod
    def load_from_jsonl(cls, filepath: str) -> List['ExperimentResult']:
        """
        Load results from JSONL file.
        
        Args:
            filepath: Path to JSONL file
        
        Returns:
            List of ExperimentResult instances
        
        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        results = []
        
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    results.append(cls.from_dict(data))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num}: {e.msg}",
                        e.doc,
                        e.pos
                    )
        
        return results
    
    def save_to_json(self, filepath: str) -> None:
        """
        Save single result to JSON file.
        
        Args:
            filepath: Path to JSON file
        """
        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'ExperimentResult':
        """
        Load single result from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            ExperimentResult instance
        
        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


@dataclass
class ExperimentRecord:
    """
    Complete experiment record including configuration and result.
    Used for storing experiment history with metadata.
    """
    
    config: dict  # ExperimentConfig as dict
    result: ExperimentResult
    timestamp: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'config': self.config,
            'result': self.result.to_dict(),
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentRecord':
        """Create ExperimentRecord from dictionary"""
        return cls(
            config=data['config'],
            result=ExperimentResult.from_dict(data['result']),
            timestamp=data['timestamp']
        )
    
    def save_to_jsonl(self, filepath: str, append: bool = True) -> None:
        """Save record to JSONL file"""
        mode = 'a' if append else 'w'
        
        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, mode) as f:
            f.write(json.dumps(self.to_dict()) + '\n')
    
    @classmethod
    def load_from_jsonl(cls, filepath: str) -> List['ExperimentRecord']:
        """Load records from JSONL file"""
        records = []
        
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    records.append(cls.from_dict(data))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num}: {e.msg}",
                        e.doc,
                        e.pos
                    )
        
        return records
