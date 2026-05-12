#!/usr/bin/env python3
"""
Routing Strategy Comparison Experiment Tool

This script automates the comparison of routing strategies (round-robin vs adapter-aware)
in the S-LoRA system. It can run experiments, analyze results, and generate charts.

Examples:
    # Run a predefined suite
    python run_routing_comparison.py --suite routing-alpha-comparison
    
    # Resume interrupted experiment
    python run_routing_comparison.py --suite routing-full-comparison --resume
    
    # Analyze existing results
    python run_routing_comparison.py --analyze-only --output-dir results/
    
    # Generate charts from results
    python run_routing_comparison.py --generate-charts --output-dir results/
    
    # List available suites
    python run_routing_comparison.py --list-suites
"""

import argparse
import sys
from pathlib import Path

from routing_experiment import (
    ExperimentRunner,
    ExperimentSuite,
    ResultAnalyzer,
    ChartGenerator,
    ExperimentResult,
    ExperimentRecord
)


def list_suites():
    """List all available experiment suites"""
    print("\nAvailable Experiment Suites:")
    print("=" * 70)
    
    for suite_name in ExperimentSuite.list_suites():
        info = ExperimentSuite.get_suite_info(suite_name)
        print(f"\n{suite_name}")
        print(f"  Configurations: {info['config_count']}")
        print(f"  Parameters:")
        for param, values in info['parameters'].items():
            print(f"    - {param}: {values}")
    
    print("\n" + "=" * 70)


def run_experiments(args):
    """Run experiments and return start timestamp for filtering results."""
    print("\n" + "=" * 70)
    print("Routing Strategy Comparison Experiment")
    print("=" * 70)
    
    import time
    start_ts = time.time()
    
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        model_setting=args.model_setting,
        server_host=f"http://{args.host}:{args.port}",
        benchmarks_dir=args.benchmarks_dir,
        model_dir=args.model_dir,
        adapter_dir=args.adapter_dir,
        debug=args.debug
    )
    
    runner.run_suite(args.suite, resume=args.resume)
    
    print("\n✓ Experiments completed!")
    print(f"  Results saved to: {args.output_dir}/{args.suite}/results.jsonl")
    if args.debug:
        print(f"  Debug log: routing_experiment/debug.log")
    
    return start_ts


def analyze_results(args, since_timestamp: float = None):
    """Analyze experiment results, optionally filtering to current run only."""
    print("\n" + "=" * 70)
    print("Analyzing Results")
    print("=" * 70)
    
    # Use per-suite directory if suite is specified
    suite_name = getattr(args, 'suite', None)
    base_dir = Path(args.output_dir)
    if suite_name:
        suite_dir = base_dir / suite_name
        result_file = suite_dir / "results.jsonl"
    else:
        result_file = base_dir / "results.jsonl"
    
    if not result_file.exists():
        # Fallback: try legacy root-level results.jsonl
        legacy_file = base_dir / "results.jsonl"
        if legacy_file.exists() and result_file != legacy_file:
            print(f"  (Using legacy results file: {legacy_file})")
            result_file = legacy_file
        else:
            print(f"\n✗ Error: No results found at {result_file}")
            print("  Run experiments first or specify correct --output-dir and --suite")
            return
    
    # Load results
    print(f"\nLoading results from {result_file}...")
    records = ExperimentRecord.load_from_jsonl(str(result_file))
    
    # Filter to current run if timestamp provided
    if since_timestamp is not None:
        records = [r for r in records if r.timestamp >= since_timestamp]
        print(f"  Filtered to current run: {len(records)} results (since {since_timestamp:.0f})")
    
    results = [r.result for r in records]
    print(f"  Using {len(results)} experiment results")
    
    # Analyze
    analyzer = ResultAnalyzer(results)
    
    # Generate summary table
    print("\n" + "-" * 70)
    print("Summary Table:")
    print("-" * 70)
    print(analyzer.generate_summary_table())
    
    # Compute improvement ratios
    print("\n" + "-" * 70)
    print("Performance Improvements (Adapter-Aware vs Round-Robin):")
    print("-" * 70)
    
    # Group by alpha if multiple alphas exist
    alphas = set(r.alpha for r in results)
    if len(alphas) > 1:
        throughput_improvements = analyzer.compute_improvement_ratio("throughput", "alpha")
        latency_improvements = analyzer.compute_improvement_ratio("avg_latency", "alpha")
        
        print("\nBy Alpha:")
        for alpha in sorted(throughput_improvements.keys()):
            tput_ratio = throughput_improvements[alpha]
            lat_ratio = latency_improvements[alpha]
            print(f"  α={alpha}:")
            print(f"    Throughput: {tput_ratio:.2%} ({tput_ratio:.3f}x)")
            print(f"    Latency:    {lat_ratio:.2%} ({lat_ratio:.3f}x)")
    
    # Group by num_adapters if multiple exist
    adapters = set(r.num_adapters for r in results)
    if len(adapters) > 1:
        throughput_improvements = analyzer.compute_improvement_ratio("throughput", "num_adapters")
        latency_improvements = analyzer.compute_improvement_ratio("avg_latency", "num_adapters")
        
        print("\nBy Number of Adapters:")
        for num_adapters in sorted(throughput_improvements.keys()):
            tput_ratio = throughput_improvements[num_adapters]
            lat_ratio = latency_improvements[num_adapters]
            print(f"  {num_adapters} adapters:")
            print(f"    Throughput: {tput_ratio:.2%} ({tput_ratio:.3f}x)")
            print(f"    Latency:    {lat_ratio:.2%} ({lat_ratio:.3f}x)")
    
    # Find best configuration
    print("\n" + "-" * 70)
    print("Best Configurations:")
    print("-" * 70)
    
    best_throughput = analyzer.get_best_configuration("throughput")
    print(f"\nHighest Throughput:")
    print(f"  Strategy: {best_throughput.routing_strategy}")
    print(f"  Alpha: {best_throughput.alpha}")
    print(f"  Adapters: {best_throughput.num_adapters}")
    print(f"  Throughput: {best_throughput.throughput:.2f} req/s")
    
    best_latency = analyzer.get_best_configuration("avg_latency")
    print(f"\nLowest Latency:")
    print(f"  Strategy: {best_latency.routing_strategy}")
    print(f"  Alpha: {best_latency.alpha}")
    print(f"  Adapters: {best_latency.num_adapters}")
    print(f"  Latency: {best_latency.avg_latency:.3f}s")
    
    print("\n" + "=" * 70)


def generate_charts(args, since_timestamp: float = None):
    """Generate visualization charts, optionally filtering to current run only."""
    print("\n" + "=" * 70)
    print("Generating Charts")
    print("=" * 70)
    
    # Use per-suite directory if suite is specified
    suite_name = getattr(args, 'suite', None)
    base_dir = Path(args.output_dir)
    if suite_name:
        suite_dir = base_dir / suite_name
        result_file = suite_dir / "results.jsonl"
    else:
        result_file = base_dir / "results.jsonl"
    
    if not result_file.exists():
        # Fallback: try legacy root-level results.jsonl
        legacy_file = base_dir / "results.jsonl"
        if legacy_file.exists() and result_file != legacy_file:
            print(f"  (Using legacy results file: {legacy_file})")
            result_file = legacy_file
        else:
            print(f"\n✗ Error: No results found at {result_file}")
            return
    
    # Load results
    print(f"\nLoading results from {result_file}...")
    records = ExperimentRecord.load_from_jsonl(str(result_file))
    
    # Filter to current run if timestamp provided
    if since_timestamp is not None:
        records = [r for r in records if r.timestamp >= since_timestamp]
        print(f"  Filtered to current run: {len(records)} results (since {since_timestamp:.0f})")
    
    results = [r.result for r in records]
    print(f"  Using {len(results)} experiment results")
    
    # Generate charts (in suite subdirectory if applicable)
    chart_base = suite_dir if suite_name and suite_dir.exists() else base_dir
    chart_dir = chart_base / "charts"
    generator = ChartGenerator(output_dir=str(chart_dir))
    
    print(f"\nGenerating charts in {chart_dir}...")
    generator.plot_all_comparisons(results, records=records)
    
    print("\n✓ Charts generated successfully!")
    print(f"  Location: {chart_dir}/")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Routing Strategy Comparison Experiment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--list-suites',
        action='store_true',
        help='List all available experiment suites'
    )
    mode_group.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze existing results (skip experiments)'
    )
    mode_group.add_argument(
        '--generate-charts',
        action='store_true',
        help='Only generate charts from existing results'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--suite',
        type=str,
        default='routing-alpha-comparison',
        help='Experiment suite to run (default: routing-alpha-comparison)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if interrupted'
    )
    parser.add_argument(
        '--auto-charts',
        action='store_true',
        help='Auto-generate charts after experiments complete (disabled by default)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging to routing_experiment/debug.log'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='routing_comparison_results',
        help='Directory to save results (default: routing_comparison_results)'
    )
    
    # Server configuration
    parser.add_argument(
        '--model-setting',
        type=str,
        default='Real',
        choices=['Real', 'Dummy'],
        help='Model setting: Real or Dummy (default: Real)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Server host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=38000,
        help='Server port (default: 38000)'
    )
    
    # Environment configuration
    parser.add_argument(
        '--gpu-ids',
        type=str,
        default='1,2,3',
        help='GPU IDs for data parallel mode (default: 1,2,3)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=3,
        help='Number of workers for data parallel mode (default: 3)'
    )
    parser.add_argument(
        '--benchmarks-dir',
        type=str,
        default='.',
        help='Path to benchmarks directory (default: current directory)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Path to base model directory (optional)'
    )
    parser.add_argument(
        '--adapter-dir',
        type=str,
        default=None,
        help='Path to adapter directory (optional)'
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.list_suites:
        list_suites()
        return
    
    if args.analyze_only:
        analyze_results(args)
        return
    
    if args.generate_charts:
        generate_charts(args)
        return
    
    # Default: run experiments, then analyze and generate charts
    try:
        start_ts = run_experiments(args)
        
        # Auto-analyze if experiments completed (current run only)
        print("\n" + "=" * 70)
        print("Auto-analyzing results...")
        analyze_results(args, since_timestamp=start_ts)
        
        # Auto-generate charts only if explicitly requested
        if args.auto_charts:
            print("\nAuto-generating charts...")
            generate_charts(args, since_timestamp=start_ts)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved to checkpoint.")
        print("Resume with: python run_routing_comparison.py --suite {} --resume".format(args.suite))
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
