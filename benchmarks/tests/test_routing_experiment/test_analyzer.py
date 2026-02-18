"""
Tests for ResultAnalyzer

Property 5: Statistical Computation Correctness
Property 7: Load Balance CV Calculation
Validates: Requirements 4.1, 4.2, 4.5
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routing_experiment.analyzer import ResultAnalyzer
from routing_experiment.result import ExperimentResult


class TestResultAnalyzer:
    """Unit tests for ResultAnalyzer"""
    
    def create_sample_results(self) -> list:
        """Create sample results for testing"""
        return [
            ExperimentResult(
                routing_strategy="round-robin",
                alpha=0.3,
                num_adapters=100,
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
            ),
            ExperimentResult(
                routing_strategy="adapter-aware",
                alpha=0.3,
                num_adapters=100,
                throughput=4.2,
                strip_throughput=4.3,
                avg_latency=1.5,
                avg_first_token_latency=0.3,
                p50_latency=1.3,
                p90_latency=2.2,
                cache_hit_rate=0.75,
                total_requests=480,
                cache_hits=360,
                cache_misses=120,
                worker_request_counts={"0": 200, "1": 150, "2": 130}
            ),
            ExperimentResult(
                routing_strategy="round-robin",
                alpha=0.6,
                num_adapters=100,
                throughput=3.8,
                strip_throughput=3.9,
                avg_latency=1.8,
                avg_first_token_latency=0.35,
                p50_latency=1.6,
                p90_latency=2.8,
                cache_hit_rate=0.55,
                total_requests=420,
                cache_hits=231,
                cache_misses=189,
                worker_request_counts={"0": 140, "1": 140, "2": 140}
            ),
            ExperimentResult(
                routing_strategy="adapter-aware",
                alpha=0.6,
                num_adapters=100,
                throughput=4.5,
                strip_throughput=4.6,
                avg_latency=1.4,
                avg_first_token_latency=0.28,
                p50_latency=1.2,
                p90_latency=2.0,
                cache_hit_rate=0.80,
                total_requests=500,
                cache_hits=400,
                cache_misses=100,
                worker_request_counts={"0": 180, "1": 170, "2": 150}
            ),
        ]
    
    def test_compute_improvement_ratio_throughput(self):
        """Test throughput improvement ratio calculation"""
        results = self.create_sample_results()
        analyzer = ResultAnalyzer(results)
        
        improvements = analyzer.compute_improvement_ratio("throughput", "alpha")
        
        # For alpha=0.3: adapter-aware=4.2, round-robin=3.5, ratio=4.2/3.5=1.2
        assert 0.3 in improvements
        assert improvements[0.3] == pytest.approx(4.2 / 3.5, rel=1e-6)
        
        # For alpha=0.6: adapter-aware=4.5, round-robin=3.8, ratio=4.5/3.8
        assert 0.6 in improvements
        assert improvements[0.6] == pytest.approx(4.5 / 3.8, rel=1e-6)
    
    def test_compute_improvement_ratio_latency(self):
        """Test latency improvement ratio calculation (lower is better)"""
        results = self.create_sample_results()
        analyzer = ResultAnalyzer(results)
        
        improvements = analyzer.compute_improvement_ratio("avg_latency", "alpha")
        
        # For alpha=0.3: round-robin=2.0, adapter-aware=1.5, ratio=2.0/1.5
        assert 0.3 in improvements
        assert improvements[0.3] == pytest.approx(2.0 / 1.5, rel=1e-6)
        
        # For alpha=0.6: round-robin=1.8, adapter-aware=1.4, ratio=1.8/1.4
        assert 0.6 in improvements
        assert improvements[0.6] == pytest.approx(1.8 / 1.4, rel=1e-6)
    
    def test_compute_statistics(self):
        """Test statistics computation"""
        results = self.create_sample_results()
        analyzer = ResultAnalyzer(results)
        
        stats = analyzer.compute_statistics("throughput", "round-robin")
        
        # Round-robin throughputs: [3.5, 3.8]
        expected_mean = np.mean([3.5, 3.8])
        expected_std = np.std([3.5, 3.8])
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "ci_95_lower" in stats
        assert "ci_95_upper" in stats
        
        assert stats["mean"] == pytest.approx(expected_mean, rel=1e-6)
        assert stats["std"] == pytest.approx(expected_std, rel=1e-6)
        assert stats["min"] == 3.5
        assert stats["max"] == 3.8
    
    def test_compute_load_balance_cv(self):
        """Test coefficient of variation calculation"""
        result = ExperimentResult(
            routing_strategy="round-robin",
            alpha=0.6,
            num_adapters=100,
            throughput=3.8,
            strip_throughput=3.9,
            avg_latency=1.8,
            avg_first_token_latency=0.35,
            p50_latency=1.6,
            p90_latency=2.8,
            cache_hit_rate=0.55,
            total_requests=420,
            cache_hits=231,
            cache_misses=189,
            worker_request_counts={"0": 140, "1": 140, "2": 140}
        )
        
        analyzer = ResultAnalyzer([])
        cv = analyzer.compute_load_balance_cv(result)
        
        # Perfect balance: CV should be 0
        assert cv == pytest.approx(0.0, abs=1e-10)
    
    def test_compute_load_balance_cv_skewed(self):
        """Test CV with skewed distribution"""
        result = ExperimentResult(
            routing_strategy="adapter-aware",
            alpha=0.3,
            num_adapters=100,
            throughput=4.2,
            strip_throughput=4.3,
            avg_latency=1.5,
            avg_first_token_latency=0.3,
            p50_latency=1.3,
            p90_latency=2.2,
            cache_hit_rate=0.75,
            total_requests=480,
            cache_hits=360,
            cache_misses=120,
            worker_request_counts={"0": 200, "1": 150, "2": 130}
        )
        
        analyzer = ResultAnalyzer([])
        cv = analyzer.compute_load_balance_cv(result)
        
        # Skewed distribution: CV > 0
        counts = [200, 150, 130]
        expected_cv = np.std(counts) / np.mean(counts)
        
        assert cv == pytest.approx(expected_cv, rel=1e-6)
        assert cv > 0
    
    def test_generate_summary_table(self):
        """Test summary table generation"""
        results = self.create_sample_results()
        analyzer = ResultAnalyzer(results)
        
        table = analyzer.generate_summary_table()
        
        assert isinstance(table, str)
        assert "Strategy" in table
        assert "Alpha" in table
        assert "Throughput" in table
        assert "round-robin" in table
        assert "adapter-aware" in table
    
    def test_group_results(self):
        """Test result grouping"""
        results = self.create_sample_results()
        analyzer = ResultAnalyzer(results)
        
        grouped = analyzer._group_results("alpha")
        
        assert 0.3 in grouped
        assert 0.6 in grouped
        assert len(grouped[0.3]) == 2  # 2 results with alpha=0.3
        assert len(grouped[0.6]) == 2  # 2 results with alpha=0.6
    
    def test_compute_comparison_summary(self):
        """Test comprehensive comparison summary"""
        results = self.create_sample_results()
        analyzer = ResultAnalyzer(results)
        
        summary = analyzer.compute_comparison_summary("alpha")
        
        assert "group_by" in summary
        assert summary["group_by"] == "alpha"
        assert "throughput_improvement" in summary
        assert "latency_improvement" in summary
        assert "round_robin_stats" in summary
        assert "adapter_aware_stats" in summary
    
    def test_get_best_configuration_throughput(self):
        """Test finding best configuration by throughput"""
        results = self.create_sample_results()
        analyzer = ResultAnalyzer(results)
        
        best = analyzer.get_best_configuration("throughput")
        
        # Best throughput is 4.5 (adapter-aware, alpha=0.6)
        assert best.throughput == 4.5
        assert best.routing_strategy == "adapter-aware"
        assert best.alpha == 0.6
    
    def test_get_best_configuration_latency(self):
        """Test finding best configuration by latency (lower is better)"""
        results = self.create_sample_results()
        analyzer = ResultAnalyzer(results)
        
        best = analyzer.get_best_configuration("avg_latency")
        
        # Best latency is 1.4 (adapter-aware, alpha=0.6)
        assert best.avg_latency == 1.4
        assert best.routing_strategy == "adapter-aware"
        assert best.alpha == 0.6
    
    def test_filter_results(self):
        """Test result filtering"""
        results = self.create_sample_results()
        analyzer = ResultAnalyzer(results)
        
        filtered = analyzer.filter_results(routing_strategy="adapter-aware", alpha=0.3)
        
        assert len(filtered) == 1
        assert filtered[0].routing_strategy == "adapter-aware"
        assert filtered[0].alpha == 0.3
    
    def test_empty_results(self):
        """Test analyzer with empty results"""
        analyzer = ResultAnalyzer([])
        
        stats = analyzer.compute_statistics("throughput", "round-robin")
        assert stats == {}
        
        with pytest.raises(ValueError):
            analyzer.get_best_configuration("throughput")


class TestAnalyzerPropertyBased:
    """Property-based tests for analyzer"""
    
    @given(
        values=st.lists(
            st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False), 
            min_size=2, 
            max_size=100
        )
    )
    def test_statistical_computation_property(self, values):
        """
        Feature: routing-comparison, Property 5: Statistical Computation Correctness
        
        For any list of metric values, the computed mean SHALL equal sum(values)/len(values),
        the std SHALL equal the population standard deviation, and the 95% confidence interval
        bounds SHALL equal the 2.5th and 97.5th percentiles.
        """
        # Create mock results with the given values as throughput
        mock_results = [
            ExperimentResult(
                routing_strategy="adapter-aware",
                alpha=0.6,
                num_adapters=100,
                throughput=v,
                strip_throughput=v,
                avg_latency=0,
                avg_first_token_latency=0,
                p50_latency=0,
                p90_latency=0,
                cache_hit_rate=0,
                total_requests=0,
                cache_hits=0,
                cache_misses=0,
                worker_request_counts={}
            )
            for v in values
        ]
        
        analyzer = ResultAnalyzer(mock_results)
        stats = analyzer.compute_statistics("throughput", "adapter-aware")
        
        expected_mean = np.mean(values)
        expected_std = np.std(values)
        expected_ci_lower = np.percentile(values, 2.5)
        expected_ci_upper = np.percentile(values, 97.5)
        
        assert stats["mean"] == pytest.approx(expected_mean, rel=1e-6)
        assert stats["std"] == pytest.approx(expected_std, rel=1e-6)
        assert stats["ci_95_lower"] == pytest.approx(expected_ci_lower, rel=1e-6)
        assert stats["ci_95_upper"] == pytest.approx(expected_ci_upper, rel=1e-6)
    
    @given(
        counts=st.lists(
            st.integers(min_value=1, max_value=1000), 
            min_size=2, 
            max_size=10
        )
    )
    def test_load_balance_cv_property(self, counts):
        """
        Feature: routing-comparison, Property 7: Load Balance CV Calculation
        
        For any worker_request_counts dictionary, the coefficient of variation
        SHALL equal std(counts) / mean(counts), where std is the population
        standard deviation.
        """
        result = ExperimentResult(
            routing_strategy="adapter-aware",
            alpha=0.6,
            num_adapters=100,
            throughput=0,
            strip_throughput=0,
            avg_latency=0,
            avg_first_token_latency=0,
            p50_latency=0,
            p90_latency=0,
            cache_hit_rate=0,
            total_requests=sum(counts),
            cache_hits=0,
            cache_misses=0,
            worker_request_counts={str(i): c for i, c in enumerate(counts)}
        )
        
        analyzer = ResultAnalyzer([])
        cv = analyzer.compute_load_balance_cv(result)
        
        expected_cv = np.std(counts) / np.mean(counts)
        assert cv == pytest.approx(expected_cv, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
