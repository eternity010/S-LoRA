"""
Routing Strategy Comparison Experiment Framework

This package provides tools for automated comparison of routing strategies
in the S-LoRA system, including experiment orchestration, result analysis,
and visualization.

Main Components:
    - ExperimentConfig: Configuration for single experiment
    - ExperimentSuite: Predefined experiment suites
    - ExperimentResult: Result data structure
    - ExperimentRecord: Complete experiment record with metadata
    - ResultAnalyzer: Statistical analysis and comparison
    - ChartGenerator: Publication-quality visualization
    - ExperimentRunner: Automated experiment execution

Usage:
    # Run a predefined suite
    from routing_experiment import ExperimentRunner
    runner = ExperimentRunner(output_dir="results")
    runner.run_suite("dp-roundrobin-baseline")
    
    # Analyze results
    from routing_experiment import ResultAnalyzer, ExperimentRecord
    records = ExperimentRecord.load_from_jsonl("results/results.jsonl")
    analyzer = ResultAnalyzer([r.result for r in records])
    print(analyzer.generate_summary_table())
    
    # Generate charts
    from routing_experiment import ChartGenerator
    generator = ChartGenerator(output_dir="charts")
    generator.plot_all_comparisons([r.result for r in records])
"""

__version__ = "1.0.0"
__author__ = "S-LoRA Routing Comparison Team"

from .config import ExperimentConfig
from .suite import ExperimentSuite
from .result import ExperimentResult, ExperimentRecord
from .analyzer import ResultAnalyzer
from .charts import ChartGenerator
from .runner import ExperimentRunner

__all__ = [
    'ExperimentConfig', 
    'ExperimentSuite', 
    'ExperimentResult', 
    'ExperimentRecord',
    'ResultAnalyzer',
    'ChartGenerator',
    'ExperimentRunner',
    '__version__',
]
