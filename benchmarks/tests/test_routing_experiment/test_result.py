"""
Tests for ExperimentResult

Property 3: Result Serialization Round-Trip
Validates: Requirements 3.5
"""

import pytest
import json
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routing_experiment.result import ExperimentResult, ExperimentRecord


class TestExperimentResult:
    """Unit tests for ExperimentResult"""
    
    def test_create_result(self):
        """Test creating a valid result"""
        result = ExperimentResult(
            routing_strategy="adapter-aware",
            alpha=0.6,
            num_adapters=100,
            throughput=3.85,
            strip_throughput=3.92,
            avg_latency=2.15,
            avg_first_token_latency=0.45,
            p50_latency=1.89,
            p90_latency=3.21,
            cache_hit_rate=0.72,
            total_requests=480,
            cache_hits=346,
            cache_misses=134,
            worker_request_counts={"0": 158, "1": 162, "2": 160}
        )
        
        assert result.routing_strategy == "adapter-aware"
        assert result.alpha == 0.6
        assert result.throughput == 3.85
        assert result.cache_hit_rate == 0.72
    
    def test_to_dict(self):
        """Test converting result to dictionary"""
        result = ExperimentResult(
            routing_strategy="round-robin",
            alpha=0.3,
            num_adapters=50,
            throughput=3.5,
            strip_throughput=3.6,
            avg_latency=2.0,
            avg_first_token_latency=0.4,
            p50_latency=1.8,
            p90_latency=3.0,
            cache_hit_rate=0.5,
            total_requests=400,
            cache_hits=200,
            cache_misses=200,
            worker_request_counts={"0": 133, "1": 134, "2": 133}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["routing_strategy"] == "round-robin"
        assert result_dict["alpha"] == 0.3
        assert result_dict["throughput"] == 3.5
        assert "worker_request_counts" in result_dict
    
    def test_from_dict(self):
        """Test creating result from dictionary"""
        data = {
            "routing_strategy": "adapter-aware",
            "alpha": 0.6,
            "num_adapters": 100,
            "throughput": 3.85,
            "strip_throughput": 3.92,
            "avg_latency": 2.15,
            "avg_first_token_latency": 0.45,
            "p50_latency": 1.89,
            "p90_latency": 3.21,
            "cache_hit_rate": 0.72,
            "total_requests": 480,
            "cache_hits": 346,
            "cache_misses": 134,
            "worker_request_counts": {"0": 158, "1": 162, "2": 160}
        }
        
        result = ExperimentResult.from_dict(data)
        
        assert result.routing_strategy == "adapter-aware"
        assert result.alpha == 0.6
        assert result.throughput == 3.85
        assert result.worker_request_counts == {"0": 158, "1": 162, "2": 160}
    
    def test_round_trip_dict(self):
        """Test to_dict and from_dict round-trip"""
        original = ExperimentResult(
            routing_strategy="adapter-aware",
            alpha=0.6,
            num_adapters=100,
            throughput=3.85,
            strip_throughput=3.92,
            avg_latency=2.15,
            avg_first_token_latency=0.45,
            p50_latency=1.89,
            p90_latency=3.21,
            cache_hit_rate=0.72,
            total_requests=480,
            cache_hits=346,
            cache_misses=134,
            worker_request_counts={"0": 158, "1": 162, "2": 160}
        )
        
        # Round-trip
        restored = ExperimentResult.from_dict(original.to_dict())
        
        assert restored.routing_strategy == original.routing_strategy
        assert restored.alpha == original.alpha
        assert restored.throughput == original.throughput
        assert restored.cache_hit_rate == original.cache_hit_rate
        assert restored.worker_request_counts == original.worker_request_counts
    
    def test_save_and_load_jsonl(self):
        """Test saving and loading JSONL file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.jsonl"
            
            # Create and save results
            result1 = ExperimentResult(
                routing_strategy="round-robin",
                alpha=0.3,
                num_adapters=50,
                throughput=3.5,
                strip_throughput=3.6,
                avg_latency=2.0,
                avg_first_token_latency=0.4,
                p50_latency=1.8,
                p90_latency=3.0,
                cache_hit_rate=0.5,
                total_requests=400,
                cache_hits=200,
                cache_misses=200,
                worker_request_counts={"0": 133, "1": 134, "2": 133}
            )
            
            result2 = ExperimentResult(
                routing_strategy="adapter-aware",
                alpha=0.6,
                num_adapters=100,
                throughput=3.85,
                strip_throughput=3.92,
                avg_latency=2.15,
                avg_first_token_latency=0.45,
                p50_latency=1.89,
                p90_latency=3.21,
                cache_hit_rate=0.72,
                total_requests=480,
                cache_hits=346,
                cache_misses=134,
                worker_request_counts={"0": 158, "1": 162, "2": 160}
            )
            
            result1.save_to_jsonl(str(filepath))
            result2.save_to_jsonl(str(filepath))
            
            # Load results
            loaded = ExperimentResult.load_from_jsonl(str(filepath))
            
            assert len(loaded) == 2
            assert loaded[0].routing_strategy == "round-robin"
            assert loaded[1].routing_strategy == "adapter-aware"
            assert loaded[0].alpha == 0.3
            assert loaded[1].alpha == 0.6
    
    def test_save_and_load_json(self):
        """Test saving and loading single JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "result.json"
            
            original = ExperimentResult(
                routing_strategy="adapter-aware",
                alpha=0.6,
                num_adapters=100,
                throughput=3.85,
                strip_throughput=3.92,
                avg_latency=2.15,
                avg_first_token_latency=0.45,
                p50_latency=1.89,
                p90_latency=3.21,
                cache_hit_rate=0.72,
                total_requests=480,
                cache_hits=346,
                cache_misses=134,
                worker_request_counts={"0": 158, "1": 162, "2": 160}
            )
            
            original.save_to_json(str(filepath))
            loaded = ExperimentResult.load_from_json(str(filepath))
            
            assert loaded.routing_strategy == original.routing_strategy
            assert loaded.throughput == original.throughput
            assert loaded.cache_hit_rate == original.cache_hit_rate
    
    def test_optional_rank_mismatch(self):
        """Test optional avg_rank_mismatch field"""
        result = ExperimentResult(
            routing_strategy="adapter-aware",
            alpha=0.6,
            num_adapters=100,
            throughput=3.85,
            strip_throughput=3.92,
            avg_latency=2.15,
            avg_first_token_latency=0.45,
            p50_latency=1.89,
            p90_latency=3.21,
            cache_hit_rate=0.72,
            total_requests=480,
            cache_hits=346,
            cache_misses=134,
            worker_request_counts={"0": 158, "1": 162, "2": 160},
            avg_rank_mismatch=0.15
        )
        
        assert result.avg_rank_mismatch == 0.15
        
        # Round-trip should preserve optional field
        restored = ExperimentResult.from_dict(result.to_dict())
        assert restored.avg_rank_mismatch == 0.15


class TestExperimentRecord:
    """Unit tests for ExperimentRecord"""
    
    def test_create_record(self):
        """Test creating experiment record"""
        result = ExperimentResult(
            routing_strategy="adapter-aware",
            alpha=0.6,
            num_adapters=100,
            throughput=3.85,
            strip_throughput=3.92,
            avg_latency=2.15,
            avg_first_token_latency=0.45,
            p50_latency=1.89,
            p90_latency=3.21,
            cache_hit_rate=0.72,
            total_requests=480,
            cache_hits=346,
            cache_misses=134,
            worker_request_counts={"0": 158, "1": 162, "2": 160}
        )
        
        config = {
            "routing_strategy": "adapter-aware",
            "alpha": 0.6,
            "num_adapters": 100
        }
        
        record = ExperimentRecord(
            config=config,
            result=result,
            timestamp=1704067200.0
        )
        
        assert record.config == config
        assert record.result == result
        assert record.timestamp == 1704067200.0
    
    def test_save_and_load_record_jsonl(self):
        """Test saving and loading experiment records"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "records.jsonl"
            
            result = ExperimentResult(
                routing_strategy="adapter-aware",
                alpha=0.6,
                num_adapters=100,
                throughput=3.85,
                strip_throughput=3.92,
                avg_latency=2.15,
                avg_first_token_latency=0.45,
                p50_latency=1.89,
                p90_latency=3.21,
                cache_hit_rate=0.72,
                total_requests=480,
                cache_hits=346,
                cache_misses=134,
                worker_request_counts={"0": 158, "1": 162, "2": 160}
            )
            
            config = {"routing_strategy": "adapter-aware", "alpha": 0.6}
            
            record = ExperimentRecord(
                config=config,
                result=result,
                timestamp=1704067200.0
            )
            
            record.save_to_jsonl(str(filepath))
            
            loaded = ExperimentRecord.load_from_jsonl(str(filepath))
            
            assert len(loaded) == 1
            assert loaded[0].config == config
            assert loaded[0].result.routing_strategy == "adapter-aware"
            assert loaded[0].timestamp == 1704067200.0


class TestResultPropertyBased:
    """Property-based tests for result serialization"""
    
    @given(
        throughput=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        avg_latency=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        cache_hit_rate=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
        worker_counts=st.dictionaries(
            keys=st.integers(min_value=0, max_value=10).map(str),
            values=st.integers(min_value=0, max_value=1000),
            min_size=1, max_size=5
        )
    )
    def test_result_serialization_roundtrip(self, throughput, avg_latency, cache_hit_rate, worker_counts):
        """
        Feature: routing-comparison, Property 3: Result Serialization Round-Trip
        
        For any valid ExperimentResult object, serializing to JSON and then
        deserializing SHALL produce an equivalent object with all fields preserved.
        """
        result = ExperimentResult(
            routing_strategy="adapter-aware",
            alpha=0.6,
            num_adapters=100,
            throughput=throughput,
            strip_throughput=throughput * 0.95,
            avg_latency=avg_latency,
            avg_first_token_latency=avg_latency * 0.2,
            p50_latency=avg_latency * 0.9,
            p90_latency=avg_latency * 1.5,
            cache_hit_rate=cache_hit_rate,
            total_requests=sum(worker_counts.values()),
            cache_hits=int(sum(worker_counts.values()) * cache_hit_rate),
            cache_misses=int(sum(worker_counts.values()) * (1 - cache_hit_rate)),
            worker_request_counts=worker_counts
        )
        
        # Serialize and deserialize
        json_str = json.dumps(result.to_dict())
        restored = ExperimentResult.from_dict(json.loads(json_str))
        
        # Verify equivalence
        assert restored.throughput == result.throughput
        assert restored.avg_latency == result.avg_latency
        assert restored.cache_hit_rate == result.cache_hit_rate
        assert restored.worker_request_counts == result.worker_request_counts
        assert restored.routing_strategy == result.routing_strategy
        assert restored.alpha == result.alpha
        assert restored.num_adapters == result.num_adapters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
