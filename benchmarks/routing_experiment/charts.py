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
    
    def plot_throughput_by_w2(self, results: List[ExperimentResult], records: List = None) -> None:
        """
        Generate throughput vs w2 (load penalty weight) chart.
        
        Args:
            results: List of experiment results with varying w2 values
            records: Deprecated, kept for backward compatibility (w2 now on ExperimentResult)
        """
        fig, ax = plt.subplots()
        
        # Filter adapter-aware results
        aa_results = [r for r in results if r.routing_strategy == "adapter-aware"]
        
        # Group by w2 value (now available directly on ExperimentResult)
        w2_throughput = {}
        for r in aa_results:
            w2 = r.routing_w2
            if w2 not in w2_throughput:
                w2_throughput[w2] = []
            w2_throughput[w2].append(r.throughput)
        
        if not w2_throughput:
            print("No w2 data found in results, skipping w2 charts")
            return
        
        w2_values = sorted(w2_throughput.keys())
        throughputs = [np.mean(w2_throughput[w2]) for w2 in w2_values]
        
        ax.plot(w2_values, throughputs, 'o-', color=COLORS['adapter-aware'], 
                linewidth=2, markersize=10)
        
        ax.set_xlabel('w2 (Load Penalty Weight)')
        ax.set_ylabel('Throughput (req/s)')
        ax.set_title('Throughput vs Load Penalty Weight (w2)')
        ax.grid(alpha=0.3)
        
        self._save_figure(fig, "throughput_by_w2")
    
    def plot_latency_by_w2(self, results: List[ExperimentResult], records: List = None) -> None:
        """
        Generate latency vs w2 chart.
        
        Args:
            results: List of experiment results with varying w2 values
            records: Deprecated, kept for backward compatibility
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Group by w2
        w2_avg_latency = {}
        w2_first_token = {}
        
        aa_results = [r for r in results if r.routing_strategy == "adapter-aware"]
        for r in aa_results:
            w2 = r.routing_w2
            if w2 not in w2_avg_latency:
                w2_avg_latency[w2] = []
                w2_first_token[w2] = []
            w2_avg_latency[w2].append(r.avg_latency)
            w2_first_token[w2].append(r.avg_first_token_latency)
        
        if not w2_avg_latency:
            return
        
        w2_values = sorted(w2_avg_latency.keys())
        
        # Average latency
        avg_latencies = [np.mean(w2_avg_latency[w2]) for w2 in w2_values]
        axes[0].plot(w2_values, avg_latencies, 'o-', color=COLORS['adapter-aware'], 
                     linewidth=2, markersize=10)
        axes[0].set_xlabel('w2 (Load Penalty Weight)')
        axes[0].set_ylabel('Average Latency (s)')
        axes[0].set_title('Average Latency vs w2')
        axes[0].grid(alpha=0.3)
        
        # First token latency
        first_tokens = [np.mean(w2_first_token[w2]) for w2 in w2_values]
        axes[1].plot(w2_values, first_tokens, 's-', color=COLORS['adapter-aware'], 
                     linewidth=2, markersize=10)
        axes[1].set_xlabel('w2 (Load Penalty Weight)')
        axes[1].set_ylabel('First Token Latency (s)')
        axes[1].set_title('First Token Latency vs w2')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, "latency_by_w2")
    
    def plot_cache_hit_rate_by_w2(self, results: List[ExperimentResult], records: List = None) -> None:
        """
        Generate cache hit rate vs w2 chart.
        
        Args:
            results: List of experiment results with varying w2 values
            records: Deprecated, kept for backward compatibility
        """
        fig, ax = plt.subplots()
        
        # Group by w2
        w2_cache_hit = {}
        
        aa_results = [r for r in results if r.routing_strategy == "adapter-aware"]
        for r in aa_results:
            w2 = r.routing_w2
            if w2 not in w2_cache_hit:
                w2_cache_hit[w2] = []
            w2_cache_hit[w2].append(r.cache_hit_rate * 100)
        
        if not w2_cache_hit:
            return
        
        w2_values = sorted(w2_cache_hit.keys())
        cache_hits = [np.mean(w2_cache_hit[w2]) for w2 in w2_values]
        
        ax.plot(w2_values, cache_hits, 'o-', color=COLORS['adapter-aware'], 
                linewidth=2, markersize=10)
        
        ax.set_xlabel('w2 (Load Penalty Weight)')
        ax.set_ylabel('Cache Hit Rate (%)')
        ax.set_title('Cache Hit Rate vs Load Penalty Weight (w2)')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 100)
        
        self._save_figure(fig, "cache_hit_rate_by_w2")
    
    def plot_w2_comparison_summary(self, results: List[ExperimentResult], records: List = None) -> None:
        """
        Generate a summary chart comparing all metrics across w2 values.
        
        Args:
            results: List of experiment results with varying w2 values
            records: Deprecated, kept for backward compatibility
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Group by w2
        w2_data = {}
        
        aa_results = [r for r in results if r.routing_strategy == "adapter-aware"]
        for r in aa_results:
            w2 = r.routing_w2
            if w2 not in w2_data:
                w2_data[w2] = {'throughput': [], 'latency': [], 'cache_hit': [], 'first_token': []}
            w2_data[w2]['throughput'].append(r.throughput)
            w2_data[w2]['latency'].append(r.avg_latency)
            w2_data[w2]['cache_hit'].append(r.cache_hit_rate * 100)
            w2_data[w2]['first_token'].append(r.avg_first_token_latency)
        
        if not w2_data:
            return
        
        w2_values = sorted(w2_data.keys())
        
        # Throughput
        axes[0, 0].bar(range(len(w2_values)), 
                       [np.mean(w2_data[w2]['throughput']) for w2 in w2_values],
                       color=COLORS['adapter-aware'], alpha=0.8)
        axes[0, 0].set_xticks(range(len(w2_values)))
        axes[0, 0].set_xticklabels([f'{w2}' for w2 in w2_values])
        axes[0, 0].set_xlabel('w2')
        axes[0, 0].set_ylabel('Throughput (req/s)')
        axes[0, 0].set_title('Throughput')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Average Latency
        axes[0, 1].bar(range(len(w2_values)), 
                       [np.mean(w2_data[w2]['latency']) for w2 in w2_values],
                       color=COLORS['adapter-aware'], alpha=0.8)
        axes[0, 1].set_xticks(range(len(w2_values)))
        axes[0, 1].set_xticklabels([f'{w2}' for w2 in w2_values])
        axes[0, 1].set_xlabel('w2')
        axes[0, 1].set_ylabel('Avg Latency (s)')
        axes[0, 1].set_title('Average Latency')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Cache Hit Rate
        axes[1, 0].bar(range(len(w2_values)), 
                       [np.mean(w2_data[w2]['cache_hit']) for w2 in w2_values],
                       color=COLORS['adapter-aware'], alpha=0.8)
        axes[1, 0].set_xticks(range(len(w2_values)))
        axes[1, 0].set_xticklabels([f'{w2}' for w2 in w2_values])
        axes[1, 0].set_xlabel('w2')
        axes[1, 0].set_ylabel('Cache Hit Rate (%)')
        axes[1, 0].set_title('Cache Hit Rate')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # First Token Latency
        axes[1, 1].bar(range(len(w2_values)), 
                       [np.mean(w2_data[w2]['first_token']) for w2 in w2_values],
                       color=COLORS['adapter-aware'], alpha=0.8)
        axes[1, 1].set_xticks(range(len(w2_values)))
        axes[1, 1].set_xticklabels([f'{w2}' for w2 in w2_values])
        axes[1, 1].set_xlabel('w2')
        axes[1, 1].set_ylabel('First Token Latency (s)')
        axes[1, 1].set_title('First Token Latency')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Performance Metrics vs Load Penalty Weight (w2)', fontsize=14, y=1.02)
        plt.tight_layout()
        self._save_figure(fig, "w2_comparison_summary")

    def plot_w2_sweet_spot(self, results: List[ExperimentResult], records: List = None,
                           p95_ttft_threshold: float = 15.0,
                           cache_hit_floor: float = 50.0) -> Optional[float]:
        """
        Generate dual Y-axis sweet-spot chart for finding optimal w2.

        Left Y-axis:  Cache Hit Rate (%) + Throughput (req/s)
        Right Y-axis: P95 TTFT (s)  (falls back to P90 if P95 unavailable)

        Decision logic:
          1. Exclude w2 where P95 TTFT > p95_ttft_threshold
          2. Exclude w2 where cache_hit_rate < cache_hit_floor (%)
          3. Among survivors, pick w2 with highest throughput

        Args:
            results: List of experiment results
            records: Deprecated, kept for backward compatibility
            p95_ttft_threshold: Max acceptable P95 TTFT in seconds (default 15)
            cache_hit_floor: Min acceptable cache hit rate in % (default 50)

        Returns:
            The sweet-spot w2 value, or None if no valid w2 found
        """
        # ---- collect per-w2 metrics ----
        w2_data: dict = {}

        for r in results:
            if r.routing_strategy != "adapter-aware":
                continue
            w2 = r.routing_w2
            if w2 not in w2_data:
                w2_data[w2] = {'throughput': [], 'cache_hit': [], 'p95_ttft': []}
            w2_data[w2]['throughput'].append(r.throughput)
            w2_data[w2]['cache_hit'].append(r.cache_hit_rate * 100)
            # prefer p95; fall back to p90
            p95 = r.p95_first_token_latency
            if p95 is None or p95 == 0.0:
                p95 = r.p90_first_token_latency
            w2_data[w2]['p95_ttft'].append(p95)

        if not w2_data:
            print("No w2 data found, skipping sweet-spot chart")
            return None

        w2_values = sorted(w2_data.keys())
        throughputs = [np.mean(w2_data[w]['throughput']) for w in w2_values]
        cache_hits  = [np.mean(w2_data[w]['cache_hit'])  for w in w2_values]
        p95_ttfts   = [np.mean(w2_data[w]['p95_ttft'])   for w in w2_values]

        # ---- 3-step decision logic ----
        candidates = list(range(len(w2_values)))
        # step 1: exclude high P95 TTFT
        candidates = [i for i in candidates if p95_ttfts[i] <= p95_ttft_threshold]
        # step 2: exclude low cache hit rate
        candidates = [i for i in candidates if cache_hits[i] >= cache_hit_floor]
        # step 3: pick highest throughput among survivors
        sweet_idx = max(candidates, key=lambda i: throughputs[i]) if candidates else None
        sweet_w2 = w2_values[sweet_idx] if sweet_idx is not None else None

        # ---- plot ----
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # left axis: cache hit rate + throughput
        ln1 = ax1.plot(w2_values, cache_hits, 'o-', color='#2ca02c',
                        linewidth=2, markersize=8, label='Cache Hit Rate (%)')
        ln2 = ax1.plot(w2_values, throughputs, 's-', color='#1f77b4',
                        linewidth=2, markersize=8, label='Throughput (req/s)')
        ax1.set_xlabel('w2 (Load Penalty Weight)', fontsize=13)
        ax1.set_ylabel('Cache Hit Rate (%)  /  Throughput (req/s)', fontsize=12)
        ax1.set_ylim(bottom=0)

        # right axis: P95 TTFT
        ln3 = ax2.plot(w2_values, p95_ttfts, '^--', color='#d62728',
                        linewidth=2, markersize=9, label='P95 TTFT (s)')
        ax2.set_ylabel('P95 TTFT (s)', fontsize=12, color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        ax2.set_ylim(bottom=0)

        # threshold line
        ax2.axhline(y=p95_ttft_threshold, color='#d62728', linestyle=':',
                     alpha=0.5, label=f'P95 threshold ({p95_ttft_threshold}s)')

        # annotate sweet spot
        if sweet_idx is not None:
            ax1.axvline(x=sweet_w2, color='#9467bd', linestyle='--', alpha=0.7)
            ax1.annotate(
                f'Sweet Spot\nw2={sweet_w2}',
                xy=(sweet_w2, throughputs[sweet_idx]),
                xytext=(15, 25), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#9467bd',
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.5),
            )

        # merged legend
        lines = ln1 + ln2 + ln3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, -0.12), fontsize=11)

        ax1.set_title('w2 Sweet Spot Analysis (Dual Y-Axis)', fontsize=14, pad=12)
        ax1.grid(alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, "w2_sweet_spot")

        if sweet_w2 is not None:
            print(f"Sweet-spot w2 = {sweet_w2}  "
                  f"(throughput={throughputs[sweet_idx]:.2f}, "
                  f"cache_hit={cache_hits[sweet_idx]:.1f}%, "
                  f"P95_TTFT={p95_ttfts[sweet_idx]:.2f}s)")
        else:
            print("No valid sweet-spot found within thresholds")

        return sweet_w2

    
    def plot_all_comparisons(self, results: List[ExperimentResult], records: List = None) -> None:
        """
        Generate all standard comparison charts.
        
        Args:
            results: List of experiment results
            records: Deprecated, kept for backward compatibility
        """
        # Determine grouping based on available data
        alphas = set(r.alpha for r in results)
        adapters = set(r.num_adapters for r in results)
        
        # Check for w2 variation directly from results
        w2_values = set(r.routing_w2 for r in results)
        
        # Generate w2-specific charts if multiple w2 values exist
        if len(w2_values) > 1:
            print(f"Detected {len(w2_values)} different w2 values, generating w2 comparison charts...")
            self.plot_throughput_by_w2(results)
            self.plot_latency_by_w2(results)
            self.plot_cache_hit_rate_by_w2(results)
            self.plot_w2_comparison_summary(results)
            self.plot_w2_sweet_spot(results)
        
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
