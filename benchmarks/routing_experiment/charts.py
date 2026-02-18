"""
Chart generation module for publication-quality visualizations.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional

from .result import ExperimentResult

# Use non-interactive backend
matplotlib.use('Agg')

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color scheme
COLORS = {
    'round-robin': '#1f77b4',  # Blue
    'adapter-aware': '#ff7f0e',  # Orange
}


class ChartGenerator:
    """Chart generator for publication-quality visualizations"""
    
    def __init__(self, output_dir: str = "charts"):
        """
        Initialize chart generator.
        
        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_throughput_comparison(self,
                                    results: List[ExperimentResult],
                                    group_by: str = "alpha",
                                    title: str = "Throughput Comparison") -> None:
        """
        Generate throughput comparison bar chart.
        
        Args:
            results: List of experiment results
            group_by: Field to group by (e.g., "alpha", "num_adapters")
            title: Chart title
        """
        fig, ax = plt.subplots()
        
        # Group results
        groups = sorted(set(getattr(r, group_by) for r in results))
        x = np.arange(len(groups))
        width = 0.35
        
        rr_values = []
        aa_values = []
        
        for g in groups:
            rr = [r.throughput for r in results 
                  if r.routing_strategy == "round-robin" and getattr(r, group_by) == g]
            aa = [r.throughput for r in results 
                  if r.routing_strategy == "adapter-aware" and getattr(r, group_by) == g]
            rr_values.append(np.mean(rr) if rr else 0)
            aa_values.append(np.mean(aa) if aa else 0)
        
        ax.bar(x - width/2, rr_values, width, label='Round-Robin', color=COLORS['round-robin'])
        ax.bar(x + width/2, aa_values, width, label='Adapter-Aware', color=COLORS['adapter-aware'])
        
        ax.set_xlabel(group_by.replace('_', ' ').title())
        ax.set_ylabel('Throughput (req/s)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        self._save_figure(fig, f"throughput_by_{group_by}")
    
    def plot_latency_comparison(self,
                                 results: List[ExperimentResult],
                                 group_by: str = "alpha") -> None:
        """
        Generate latency comparison charts (Avg, P50, P90, First Token).
        
        Args:
            results: List of experiment results
            group_by: Field to group by
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = [
            ('avg_latency', 'Average Latency'),
            ('p50_latency', 'P50 Latency'),
            ('p90_latency', 'P90 Latency'),
            ('avg_first_token_latency', 'Avg First Token'),
            ('p50_first_token_latency', 'P50 First Token'),
            ('p90_first_token_latency', 'P90 First Token'),
        ]
        
        groups = sorted(set(getattr(r, group_by) for r in results))
        x = np.arange(len(groups))
        width = 0.35
        
        for ax, (metric, label) in zip(axes, metrics):
            rr_values = []
            aa_values = []
            
            for g in groups:
                rr = [getattr(r, metric) for r in results 
                      if r.routing_strategy == "round-robin" and getattr(r, group_by) == g]
                aa = [getattr(r, metric) for r in results 
                      if r.routing_strategy == "adapter-aware" and getattr(r, group_by) == g]
                rr_values.append(np.mean(rr) if rr else 0)
                aa_values.append(np.mean(aa) if aa else 0)
            
            ax.bar(x - width/2, rr_values, width, label='Round-Robin', color=COLORS['round-robin'])
            ax.bar(x + width/2, aa_values, width, label='Adapter-Aware', color=COLORS['adapter-aware'])
            
            ax.set_xlabel(group_by.replace('_', ' ').title())
            ax.set_ylabel(f'{label} (s)')
            ax.set_title(label)
            ax.set_xticks(x)
            ax.set_xticklabels([str(g) for g in groups])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, f"latency_by_{group_by}")
    
    def plot_cache_hit_rate(self,
                            results: List[ExperimentResult],
                            group_by: str = "alpha") -> None:
        """
        Generate cache hit rate comparison chart.
        
        Args:
            results: List of experiment results
            group_by: Field to group by
        """
        fig, ax = plt.subplots()
        
        groups = sorted(set(getattr(r, group_by) for r in results))
        
        # Get cache hit rates for both strategies
        rr_values = []
        aa_values = []
        for g in groups:
            rr = [r.cache_hit_rate for r in results 
                  if r.routing_strategy == "round-robin" and getattr(r, group_by) == g]
            aa = [r.cache_hit_rate for r in results 
                  if r.routing_strategy == "adapter-aware" and getattr(r, group_by) == g]
            rr_values.append(np.mean(rr) * 100 if rr else 0)  # Convert to percentage
            aa_values.append(np.mean(aa) * 100 if aa else 0)
        
        ax.plot(groups, rr_values, 'o-', color=COLORS['round-robin'], 
                linewidth=2, markersize=8, label='Round-Robin')
        ax.plot(groups, aa_values, 's-', color=COLORS['adapter-aware'], 
                linewidth=2, markersize=8, label='Adapter-Aware')
        
        ax.set_xlabel(group_by.replace('_', ' ').title())
        ax.set_ylabel('Cache Hit Rate (%)')
        ax.set_title('Cache Hit Rate vs ' + group_by.replace('_', ' ').title())
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 100)
        
        self._save_figure(fig, f"cache_hit_rate_by_{group_by}")
    
    def plot_scaling_trend(self,
                           results: List[ExperimentResult],
                           metric: str = "throughput") -> None:
        """
        Generate performance scaling trend chart.
        
        Args:
            results: List of experiment results
            metric: Metric to plot (e.g., "throughput", "avg_latency")
        """
        fig, ax = plt.subplots()
        
        adapters = sorted(set(r.num_adapters for r in results))
        
        for strategy in ["round-robin", "adapter-aware"]:
            values = []
            for n in adapters:
                v = [getattr(r, metric) for r in results 
                     if r.routing_strategy == strategy and r.num_adapters == n]
                values.append(np.mean(v) if v else 0)
            
            ax.plot(adapters, values, 'o-', color=COLORS[strategy], 
                    linewidth=2, markersize=8, label=strategy.replace('-', ' ').title())
        
        ax.set_xlabel('Number of Adapters')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Number of Adapters')
        ax.legend()
        ax.grid(alpha=0.3)
        
        self._save_figure(fig, f"{metric}_scaling")
    
    def plot_worker_load_distribution(self,
                                       result: ExperimentResult) -> None:
        """
        Generate worker load distribution bar chart.
        
        Args:
            result: Single experiment result with worker load data
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        workers = sorted(result.worker_request_counts.keys())
        counts = [result.worker_request_counts[w] for w in workers]
        
        color = COLORS[result.routing_strategy]
        ax.bar([f'W{w}' for w in workers], counts, color=color, alpha=0.8)
        
        # Add mean line
        mean_count = np.mean(counts)
        ax.axhline(y=mean_count, color='red', linestyle='--', label=f'Mean: {mean_count:.0f}')
        
        ax.set_xlabel('Worker')
        ax.set_ylabel('Request Count')
        ax.set_title(f'Worker Load Distribution ({result.routing_strategy}, α={result.alpha})')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        filename = f"worker_load_{result.routing_strategy}_alpha{result.alpha}"
        self._save_figure(fig, filename)
    
    def plot_all_comparisons(self, results: List[ExperimentResult]) -> None:
        """
        Generate all standard comparison charts.
        
        Args:
            results: List of experiment results
        """
        # Determine grouping based on available data
        alphas = set(r.alpha for r in results)
        adapters = set(r.num_adapters for r in results)
        
        # Generate charts grouped by alpha if multiple alphas exist
        if len(alphas) > 1:
            self.plot_throughput_comparison(results, group_by="alpha")
            self.plot_latency_comparison(results, group_by="alpha")
            self.plot_cache_hit_rate(results, group_by="alpha")
        
        # Generate charts grouped by num_adapters if multiple adapter counts exist
        if len(adapters) > 1:
            self.plot_throughput_comparison(results, group_by="num_adapters", 
                                           title="Throughput vs Number of Adapters")
            self.plot_latency_comparison(results, group_by="num_adapters")
            self.plot_scaling_trend(results, metric="throughput")
            self.plot_scaling_trend(results, metric="avg_latency")
        
        # Generate worker load distribution for each adapter-aware result
        if results:
            aa_results = [r for r in results if r.routing_strategy == "adapter-aware"]
            for result in aa_results:
                self.plot_worker_load_distribution(result)
    
    def _save_figure(self, fig, name: str) -> None:
        """
        Save figure as PNG and PDF.
        
        Args:
            fig: Matplotlib figure object
            name: Base filename (without extension)
        """
        png_path = self.output_dir / f"{name}.png"
        pdf_path = self.output_dir / f"{name}.pdf"
        
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {png_path}, {pdf_path}")
