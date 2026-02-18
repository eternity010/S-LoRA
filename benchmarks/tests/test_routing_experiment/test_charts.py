"""
Tests for ChartGenerator

Property 9: Chart File Format Output
Validates: Requirements 5.7
"""

import pytest
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from routing_experiment.charts import ChartGenerator
from routing_experiment.result import ExperimentResult


class TestChartGenerator:
    """Unit tests for ChartGenerator"""
    
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
                num_adapters=50,
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
                worker_request_counts={"0": "140", "1": "140", "2": "140"}
            ),
            ExperimentResult(
                routing_strategy="adapter-aware",
                alpha=0.6,
                num_adapters=50,
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
    
    def test_chart_generator_init(self):
        """Test chart generator initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            
            assert generator.output_dir == Path(tmpdir)
            assert generator.output_dir.exists()
    
    def test_plot_throughput_comparison(self):
        """Test throughput comparison chart generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            results = self.create_sample_results()
            
            generator.plot_throughput_comparison(results, group_by="alpha")
            
            # Check files exist
            png_file = Path(tmpdir) / "throughput_by_alpha.png"
            pdf_file = Path(tmpdir) / "throughput_by_alpha.pdf"
            
            assert png_file.exists()
            assert pdf_file.exists()
            assert png_file.stat().st_size > 0
            assert pdf_file.stat().st_size > 0
    
    def test_plot_latency_comparison(self):
        """Test latency comparison chart generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            results = self.create_sample_results()
            
            generator.plot_latency_comparison(results, group_by="alpha")
            
            # Check files exist
            png_file = Path(tmpdir) / "latency_by_alpha.png"
            pdf_file = Path(tmpdir) / "latency_by_alpha.pdf"
            
            assert png_file.exists()
            assert pdf_file.exists()
            assert png_file.stat().st_size > 0
            assert pdf_file.stat().st_size > 0
    
    def test_plot_cache_hit_rate(self):
        """Test cache hit rate chart generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            results = self.create_sample_results()
            
            generator.plot_cache_hit_rate(results, group_by="alpha")
            
            # Check files exist
            png_file = Path(tmpdir) / "cache_hit_rate_by_alpha.png"
            pdf_file = Path(tmpdir) / "cache_hit_rate_by_alpha.pdf"
            
            assert png_file.exists()
            assert pdf_file.exists()
            assert png_file.stat().st_size > 0
            assert pdf_file.stat().st_size > 0
    
    def test_plot_scaling_trend(self):
        """Test scaling trend chart generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            results = self.create_sample_results()
            
            generator.plot_scaling_trend(results, metric="throughput")
            
            # Check files exist
            png_file = Path(tmpdir) / "throughput_scaling.png"
            pdf_file = Path(tmpdir) / "throughput_scaling.pdf"
            
            assert png_file.exists()
            assert pdf_file.exists()
            assert png_file.stat().st_size > 0
            assert pdf_file.stat().st_size > 0
    
    def test_plot_worker_load_distribution(self):
        """Test worker load distribution chart generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            result = self.create_sample_results()[1]  # adapter-aware result
            
            generator.plot_worker_load_distribution(result)
            
            # Check files exist
            png_file = Path(tmpdir) / "worker_load_adapter-aware_alpha0.3.png"
            pdf_file = Path(tmpdir) / "worker_load_adapter-aware_alpha0.3.pdf"
            
            assert png_file.exists()
            assert pdf_file.exists()
            assert png_file.stat().st_size > 0
            assert pdf_file.stat().st_size > 0
    
    def test_plot_all_comparisons(self):
        """Test generating all comparison charts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            results = self.create_sample_results()
            
            generator.plot_all_comparisons(results)
            
            # Check that multiple charts were generated
            chart_files = list(Path(tmpdir).glob("*.png"))
            assert len(chart_files) > 0
            
            # Check that both PNG and PDF exist for each chart
            for png_file in chart_files:
                pdf_file = png_file.with_suffix('.pdf')
                assert pdf_file.exists()
    
    def test_empty_results(self):
        """Test chart generation with empty results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            
            # Should not crash with empty results
            generator.plot_all_comparisons([])
    
    def test_group_by_num_adapters(self):
        """Test grouping by num_adapters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            results = self.create_sample_results()
            
            generator.plot_throughput_comparison(results, group_by="num_adapters")
            
            # Check files exist
            png_file = Path(tmpdir) / "throughput_by_num_adapters.png"
            pdf_file = Path(tmpdir) / "throughput_by_num_adapters.pdf"
            
            assert png_file.exists()
            assert pdf_file.exists()


class TestChartPropertyBased:
    """Property-based tests for chart generation"""
    
    @settings(deadline=500)  # Increase deadline for chart generation
    @given(
        chart_name=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'), 
            whitelist_characters='_-'
        ))
    )
    def test_chart_file_format_output_property(self, chart_name):
        """
        Feature: routing-comparison, Property 9: Chart File Format Output
        
        For any chart generation call, the generator SHALL create both a PNG file
        and a PDF file with the same base name in the output directory.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ChartGenerator(output_dir=tmpdir)
            
            # Create a simple figure and save it
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            
            generator._save_figure(fig, chart_name)
            
            # Verify both files exist
            png_file = Path(tmpdir) / f"{chart_name}.png"
            pdf_file = Path(tmpdir) / f"{chart_name}.pdf"
            
            assert png_file.exists(), f"PNG file not found: {png_file}"
            assert pdf_file.exists(), f"PDF file not found: {pdf_file}"
            
            # Verify files have content
            assert png_file.stat().st_size > 0, "PNG file is empty"
            assert pdf_file.stat().st_size > 0, "PDF file is empty"
            
            # Verify base names match
            assert png_file.stem == pdf_file.stem == chart_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
