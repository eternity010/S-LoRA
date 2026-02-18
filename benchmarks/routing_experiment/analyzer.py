"""
Result analysis module for computing statistics and comparisons.
"""

import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

from .result import ExperimentResult


class ResultAnalyzer:
    """Result analyzer for computing statistics and comparisons"""
    
    def __init__(self, results: List[ExperimentResult]):
        """
        Initialize analyzer with experiment results.
        
        Args:
            results: List of ExperimentResult objects
        """
        self.results = results
    
    def compute_improvement_ratio(self, 
                                   metric: str,
                                   group_by: str = "alpha") -> Dict[Any, float]:
        """
        Compute adapter-aware performance improvement ratio over round-robin.
        
        Args:
            metric: Metric name (e.g., "throughput", "avg_latency")
            group_by: Field to group by (e.g., "alpha", "num_adapters")
        
        Returns:
            Dictionary mapping group values to improvement ratios.
            For throughput/cache_hit_rate: ratio > 1 means adapter-aware is better
            For latency: ratio > 1 means adapter-aware is better (lower latency)
        """
        grouped = self._group_results(group_by)
        improvements = {}
        
        for group_val, group_results in grouped.items():
            rr_results = [r for r in group_results if r.routing_strategy == "round-robin"]
            aa_results = [r for r in group_results if r.routing_strategy == "adapter-aware"]
            
            if not rr_results or not aa_results:
                continue
            
            rr_metric = np.mean([getattr(r, metric) for r in rr_results])
            aa_metric = np.mean([getattr(r, metric) for r in aa_results])
            
            # For latency metrics, lower is better, so use rr/aa
            # For throughput and hit rate, higher is better, so use aa/rr
            if "latency" in metric:
                improvements[group_val] = rr_metric / aa_metric if aa_metric > 0 else 0
            else:
                improvements[group_val] = aa_metric / rr_metric if rr_metric > 0 else 0
        
        return improvements
    
    def compute_statistics(self, 
                           metric: str,
                           strategy: str) -> Dict[str, float]:
        """
        Compute statistical measures for a specific metric and strategy.
        
        Args:
            metric: Metric name (e.g., "throughput", "avg_latency")
            strategy: Routing strategy ("round-robin" or "adapter-aware")
        
        Returns:
            Dictionary containing mean, std, min, max, and 95% confidence interval
        """
        values = [
            getattr(r, metric) 
            for r in self.results 
            if r.routing_strategy == strategy
        ]
        
        if not values:
            return {}
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "ci_95_lower": float(np.percentile(values, 2.5)),
            "ci_95_upper": float(np.percentile(values, 97.5)),
        }
    
    def compute_load_balance_cv(self, result: ExperimentResult) -> float:
        """
        Compute coefficient of variation (CV) for worker load distribution.
        CV = std / mean, lower values indicate better load balance.
        
        Args:
            result: ExperimentResult with worker_request_counts
        
        Returns:
            Coefficient of variation (0 means perfect balance)
        """
        counts = list(result.worker_request_counts.values())
        if not counts:
            return 0.0
        
        mean = np.mean(counts)
        std = np.std(counts)
        
        return float(std / mean) if mean > 0 else 0.0
    
    def _group_results(self, group_by: str) -> Dict[Any, List[ExperimentResult]]:
        """
        Group results by a specific field.
        
        Args:
            group_by: Field name to group by
        
        Returns:
            Dictionary mapping field values to lists of results
        """
        grouped = defaultdict(list)
        for r in self.results:
            key = getattr(r, group_by)
            grouped[key].append(r)
        return dict(grouped)
    
    def generate_summary_table(self) -> str:
        """
        Generate a summary table in Markdown format.
        
        Returns:
            Markdown-formatted table string
        """
        lines = [
            "| Strategy | Alpha | Adapters | Throughput | Avg Latency | Cache Hit Rate |",
            "|----------|-------|----------|------------|-------------|----------------|"
        ]
        
        # Sort by strategy, alpha, num_adapters
        sorted_results = sorted(
            self.results, 
            key=lambda x: (x.routing_strategy, x.alpha, x.num_adapters)
        )
        
        for r in sorted_results:
            lines.append(
                f"| {r.routing_strategy} | {r.alpha} | {r.num_adapters} | "
                f"{r.throughput:.2f} | {r.avg_latency:.3f}s | {r.cache_hit_rate:.1%} |"
            )
        
        return "\n".join(lines)
    
    def compute_comparison_summary(self, group_by: str = "alpha") -> Dict[str, Any]:
        """
        Compute comprehensive comparison summary between strategies.
        
        Args:
            group_by: Field to group by for comparison
        
        Returns:
            Dictionary containing improvement ratios and statistics
        """
        summary = {
            "group_by": group_by,
            "throughput_improvement": self.compute_improvement_ratio("throughput", group_by),
            "latency_improvement": self.compute_improvement_ratio("avg_latency", group_by),
            "cache_hit_rate_improvement": self.compute_improvement_ratio("cache_hit_rate", group_by),
            "round_robin_stats": {},
            "adapter_aware_stats": {},
        }
        
        # Compute statistics for each strategy
        for strategy in ["round-robin", "adapter-aware"]:
            strategy_key = strategy.replace("-", "_") + "_stats"
            summary[strategy_key] = {
                "throughput": self.compute_statistics("throughput", strategy),
                "avg_latency": self.compute_statistics("avg_latency", strategy),
                "cache_hit_rate": self.compute_statistics("cache_hit_rate", strategy),
            }
        
        return summary
    
    def get_best_configuration(self, metric: str = "throughput") -> ExperimentResult:
        """
        Find the best configuration based on a specific metric.
        
        Args:
            metric: Metric to optimize (e.g., "throughput", "avg_latency")
        
        Returns:
            ExperimentResult with the best metric value
        
        Raises:
            ValueError: If no results available
        """
        if not self.results:
            raise ValueError("No results available")
        
        # For latency, lower is better; for others, higher is better
        if "latency" in metric:
            return min(self.results, key=lambda r: getattr(r, metric))
        else:
            return max(self.results, key=lambda r: getattr(r, metric))
    
    def filter_results(self, **criteria) -> List[ExperimentResult]:
        """
        Filter results based on criteria.
        
        Args:
            **criteria: Field-value pairs to filter by
        
        Returns:
            List of matching ExperimentResult objects
        
        Example:
            analyzer.filter_results(routing_strategy="adapter-aware", alpha=0.6)
        """
        filtered = self.results
        
        for field, value in criteria.items():
            filtered = [r for r in filtered if getattr(r, field) == value]
        
        return filtered
