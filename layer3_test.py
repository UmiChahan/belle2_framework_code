"""
Layer 3: Comprehensive Test Framework with Advanced Visualization
================================================================

A systematic test framework for Layer 3 physics engines with emphasis on:
1. Rigorous unit and integration testing
2. Visual validation through error-aware plotting
3. Performance benchmarking with profiling
4. Realistic Belle II physics scenarios
5. Preparation for visualization layer integration

Architecture:
- Modular test structure with fixtures
- Automated visual regression testing
- Performance tracking across versions
- Mock data generators for realistic scenarios
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import time
import tempfile
from pathlib import Path
import json
import warnings
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
import threading
import gc
import tracemalloc
import cProfile
import pstats
import io
from functools import wraps

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Import Layer 3 components
from layer3_core_framework import (
    PhysicsContext, PhysicsComputeNode, PhysicsEngine,
    PhysicsEngineFactory, WeightedOperation, UncertaintyProvider,
    ResourceManager, ComputationCache, event_bus, Event
)
from layer3_luminosity_processor import (
    LuminosityProcessor, LuminosityDatabase, LuminosityValue,
    ProcessIdentification, BeamEnergyCondition, create_luminosity_processor
)
from layer3_statistical_engine import (
    StatisticalAnalysisEngine, StatisticalMethod, StatisticalResult,
    create_statistical_engine, compute_uncertainty
)

# Mock Layer 2 components for testing
from unittest.mock import Mock as MockDataFrame


# ============================================================================
# TEST CONFIGURATION AND FIXTURES
# ============================================================================

@dataclass
class TestConfiguration:
    """Global test configuration."""
    output_dir: Path = field(default_factory=lambda: Path("test_outputs"))
    enable_visual_tests: bool = True
    enable_performance_tests: bool = True
    save_plots: bool = True
    random_seed: int = 42
    performance_threshold: float = 0.1  # seconds
    memory_threshold: int = 100 * 1024 * 1024  # 100MB


# Global configuration
TEST_CONFIG = TestConfiguration()


@pytest.fixture(scope="session")
def test_output_dir():
    """Create test output directory."""
    TEST_CONFIG.output_dir.mkdir(exist_ok=True)
    return TEST_CONFIG.output_dir


@pytest.fixture(scope="function")
def physics_context():
    """Create physics context for testing."""
    return PhysicsContext(
        beam_energy=10.58,
        integrated_luminosity={'data': 357.3, 'mc': 1000.0},
        detector_configuration={'tracking': 'enabled', 'calorimeter': 'ecl'},
        systematic_variations=['tracking_eff', 'pid_eff'],
        run_period="2019-2021"
    )


@pytest.fixture(scope="function")
def mock_dataframe():
    """Create mock DataFrame for testing."""
    n_events = 10000
    np.random.seed(TEST_CONFIG.random_seed)
    
    df = Mock()
    df.shape = (n_events, 5)
    df.columns = ['energy', 'momentum', 'mass', 'charge', 'weight']
    
    # Mock data
    df.__getitem__ = lambda key: {
        'energy': np.random.exponential(2.0, n_events),
        'momentum': np.random.normal(1.5, 0.5, n_events),
        'mass': np.random.normal(0.1396, 0.001, n_events),  # Pion mass
        'charge': np.random.choice([-1, 0, 1], n_events),
        'weight': np.ones(n_events)
    }[key]
    
    df._get_root_node = Mock(return_value=None)
    df._estimated_rows = n_events
    
    return df


# ============================================================================
# PERFORMANCE PROFILING DECORATORS
# ============================================================================

def profile_performance(func):
    """Decorator for performance profiling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TEST_CONFIG.enable_performance_tests:
            return func(*args, **kwargs)
        
        # CPU profiling
        pr = cProfile.Profile()
        pr.enable()
        
        # Memory tracking
        tracemalloc.start()
        
        # Timing
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Stop profiling
            end_time = time.perf_counter()
            pr.disable()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Report
            elapsed = end_time - start_time
            print(f"\n⏱️  Performance Report for {func.__name__}:")
            print(f"   Time: {elapsed:.3f}s")
            print(f"   Memory: {peak / 1024 / 1024:.1f} MB peak")
            
            # Detailed CPU profile
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions
            
            if elapsed > TEST_CONFIG.performance_threshold:
                warnings.warn(f"{func.__name__} took {elapsed:.3f}s (threshold: {TEST_CONFIG.performance_threshold}s)")
        
        return result
    
    return wrapper


def benchmark(n_iterations=3):
    """Decorator for benchmarking."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                times.append(time.perf_counter() - start)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"\n📊 Benchmark {func.__name__}: {mean_time:.3f} ± {std_time:.3f}s")
            
            return result
        return wrapper
    return decorator


# ============================================================================
# MOCK DATA GENERATORS
# ============================================================================

class BelleIIDataGenerator:
    """Generate realistic Belle II mock data for testing."""
    
    @staticmethod
    def generate_b_decay_events(n_events: int = 10000, seed: int = None) -> pd.DataFrame:
        """Generate B meson decay events."""
        if seed is not None:
            np.random.seed(seed)
        
        # B meson properties
        m_b = 5.279  # GeV
        tau_b = 1.519e-12  # seconds
        
        # Generate kinematic variables
        data = {
            'energy': np.random.normal(m_b/2, 0.05, n_events),
            'momentum': np.random.exponential(0.3, n_events),
            'mass': np.random.normal(m_b, 0.01, n_events),
            'proper_time': np.random.exponential(tau_b, n_events),
            'vertex_x': np.random.normal(0, 0.01, n_events),
            'vertex_y': np.random.normal(0, 0.01, n_events),
            'vertex_z': np.random.normal(0, 0.1, n_events),
            'n_tracks': np.random.poisson(5, n_events),
            'event_weight': np.ones(n_events)
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_continuum_background(n_events: int = 10000, process: str = 'uubar') -> pd.DataFrame:
        """Generate continuum background events."""
        np.random.seed(TEST_CONFIG.random_seed)
        
        # Different distributions for different processes
        params = {
            'uubar': {'energy_scale': 2.0, 'multiplicity': 8},
            'ddbar': {'energy_scale': 2.0, 'multiplicity': 8},
            'ssbar': {'energy_scale': 2.2, 'multiplicity': 7},
            'ccbar': {'energy_scale': 3.5, 'multiplicity': 6},
        }
        
        p = params.get(process, params['uubar'])
        
        data = {
            'energy': np.random.exponential(p['energy_scale'], n_events),
            'momentum': np.random.gamma(2, 0.5, n_events),
            'sphericity': np.random.beta(2, 5, n_events),
            'thrust': np.random.beta(5, 2, n_events),
            'n_tracks': np.random.poisson(p['multiplicity'], n_events),
            'process': process,
            'event_weight': np.random.uniform(0.8, 1.2, n_events)
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_detector_response(true_values: np.ndarray, 
                                 resolution: float = 0.01,
                                 efficiency: float = 0.95) -> Dict[str, np.ndarray]:
        """Simulate detector response with resolution and efficiency."""
        n = len(true_values)
        
        # Efficiency
        detected = np.random.binomial(1, efficiency, n).astype(bool)
        
        # Resolution effects
        measured = true_values.copy()
        measured[detected] += np.random.normal(0, resolution * true_values[detected])
        measured[~detected] = np.nan
        
        # Uncertainties (simplified model)
        uncertainties = np.full_like(measured, resolution * measured)
        uncertainties[~detected] = np.nan
        
        return {
            'measured': measured,
            'uncertainty': uncertainties,
            'detected': detected,
            'efficiency': np.sum(detected) / n
        }


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

class VisualizationTester:
    """Advanced visualization for test validation."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.figure_registry = {}
    
    def plot_histogram_with_errors(self,
                                 data: np.ndarray,
                                 errors: Tuple[np.ndarray, np.ndarray],
                                 bins: int = 50,
                                 label: str = "Data",
                                 title: str = "Histogram with Uncertainties",
                                 save_name: Optional[str] = None) -> plt.Figure:
        """Plot histogram with asymmetric error bars."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Compute histogram
        counts, edges = np.histogram(data, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        
        # Main histogram with errors
        ax1.bar(centers, counts, width=widths, alpha=0.7, 
               label=label, edgecolor='black', linewidth=1)
        
        # Error bars (asymmetric)
        lower_err, upper_err = errors
        if len(lower_err) == len(counts):
            ax1.errorbar(centers, counts, yerr=[lower_err, upper_err],
                        fmt='none', ecolor='red', capsize=3, alpha=0.8,
                        label='Uncertainties')
        
        ax1.set_ylabel('Counts')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pull plot (residuals)
        if len(errors[0]) == len(counts):
            pulls = np.where(counts > 0, 
                           (counts - np.mean(counts)) / np.sqrt(counts),
                           0)
            ax2.scatter(centers, pulls, alpha=0.6, s=20)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.axhline(y=2, color='red', linestyle=':', alpha=0.5)
            ax2.axhline(y=-2, color='red', linestyle=':', alpha=0.5)
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Pull')
            ax2.set_ylim(-3, 3)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and TEST_CONFIG.save_plots:
            filepath = self.output_dir / f"{save_name}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            self.figure_registry[save_name] = filepath
        
        return fig
    
    def plot_efficiency_curve(self,
                            x_values: np.ndarray,
                            efficiencies: np.ndarray,
                            errors: Tuple[np.ndarray, np.ndarray],
                            title: str = "Efficiency vs Variable",
                            xlabel: str = "Variable",
                            save_name: Optional[str] = None) -> plt.Figure:
        """Plot efficiency curve with error band."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Main efficiency curve
        ax.plot(x_values, efficiencies, 'b-', linewidth=2, label='Efficiency')
        
        # Error band
        lower_err, upper_err = errors
        ax.fill_between(x_values,
                       efficiencies - lower_err,
                       efficiencies + upper_err,
                       alpha=0.3, color='blue',
                       label='Uncertainty band')
        
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Efficiency')
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add reference lines
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        
        if save_name and TEST_CONFIG.save_plots:
            filepath = self.output_dir / f"{save_name}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            self.figure_registry[save_name] = filepath
        
        return fig
    
    def plot_2d_distribution(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           weights: Optional[np.ndarray] = None,
                           title: str = "2D Distribution",
                           xlabel: str = "X",
                           ylabel: str = "Y",
                           save_name: Optional[str] = None) -> plt.Figure:
        """Plot 2D distribution with optional weights."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 2D histogram
        if weights is not None:
            h, xedges, yedges = np.histogram2d(x, y, bins=50, weights=weights)
        else:
            h, xedges, yedges = np.histogram2d(x, y, bins=50)
        
        # Plot as image
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(h.T, origin='lower', extent=extent, 
                      aspect='auto', cmap='viridis')
        
        # Contours
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        contours = ax.contour(X, Y, h.T, colors='white', alpha=0.4, linewidths=1)
        
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Counts' if weights is None else 'Weighted counts')
        
        if save_name and TEST_CONFIG.save_plots:
            filepath = self.output_dir / f"{save_name}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            self.figure_registry[save_name] = filepath
        
        return fig
    
    def plot_comparison(self,
                       datasets: Dict[str, np.ndarray],
                       errors: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                       bins: int = 50,
                       title: str = "Data Comparison",
                       xlabel: str = "Value",
                       ylabel: str = "Counts",
                       save_name: Optional[str] = None) -> plt.Figure:
        """Plot multiple datasets for comparison."""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Colors for different datasets
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
        
        # Common binning
        all_data = np.concatenate(list(datasets.values()))
        bin_edges = np.histogram_bin_edges(all_data, bins=bins)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = np.diff(bin_edges)[0]
        
        # Plot each dataset
        for (name, data), color in zip(datasets.items(), colors):
            counts, _ = np.histogram(data, bins=bin_edges)
            
            # Plot bars
            ax.bar(centers, counts, width=width*0.8, alpha=0.6,
                  label=name, color=color, edgecolor='black', linewidth=0.5)
            
            # Add errors if provided
            if errors and name in errors:
                lower_err, upper_err = errors[name]
                if len(lower_err) == len(counts):
                    ax.errorbar(centers, counts, yerr=[lower_err, upper_err],
                              fmt='none', ecolor=color, capsize=2, alpha=0.8)
        
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_name and TEST_CONFIG.save_plots:
            filepath = self.output_dir / f"{save_name}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            self.figure_registry[save_name] = filepath
        
        return fig
    
    def plot_systematic_variations(self,
                                 nominal: np.ndarray,
                                 variations: Dict[str, np.ndarray],
                                 title: str = "Systematic Variations",
                                 save_name: Optional[str] = None) -> plt.Figure:
        """Plot systematic uncertainty variations."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        x = np.arange(len(nominal))
        
        # Main plot - absolute values
        ax1.plot(x, nominal, 'k-', linewidth=2, label='Nominal')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(variations)))
        for (name, varied), color in zip(variations.items(), colors):
            ax1.plot(x, varied, '--', color=color, alpha=0.7, label=name)
        
        ax1.set_ylabel('Value')
        ax1.set_title(title)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Ratio plot
        for (name, varied), color in zip(variations.items(), colors):
            ratio = varied / nominal
            ax2.plot(x, ratio, '-', color=color, alpha=0.7, label=name)
        
        ax2.axhline(y=1, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Ratio to nominal')
        ax2.set_ylim(0.9, 1.1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and TEST_CONFIG.save_plots:
            filepath = self.output_dir / f"{save_name}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            self.figure_registry[save_name] = filepath
        
        return fig
    
    def create_summary_report(self, test_name: str) -> Path:
        """Create HTML summary report with all plots."""
        
        html_content = f"""
        <html>
        <head>
            <title>Layer 3 Test Report: {test_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .plot {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Layer 3 Test Report: {test_name}</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        for name, path in self.figure_registry.items():
            html_content += f"""
            <div class="plot">
                <h2>{name}</h2>
                <img src="{path.name}" alt="{name}">
            </div>
            """
        
        html_content += "</body></html>"
        
        report_path = self.output_dir / f"{test_name}_report.html"
        report_path.write_text(html_content)
        
        return report_path


# ============================================================================
# UNIT TESTS: Core Framework
# ============================================================================

class TestCoreFramework:
    """Test core framework components."""
    
    def test_physics_context_creation(self, physics_context):
        """Test PhysicsContext initialization and validation."""
        assert physics_context.beam_energy == 10.58
        assert 'data' in physics_context.integrated_luminosity
        assert len(physics_context.systematic_variations) == 2
        
        # Test validation
        with pytest.raises(ValueError):
            PhysicsContext(beam_energy=-1.0)
    
    def test_physics_context_thread_safety(self, physics_context):
        """Test thread-safe operations on PhysicsContext."""
        results = []
        
        def modify_context(value):
            with physics_context.temporary_variation(beam_energy=value):
                time.sleep(0.01)  # Simulate work
                results.append(physics_context.beam_energy)
        
        # Launch threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=modify_context, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Check all values were recorded
        assert len(results) == 10
        assert set(results) == set(range(10))
        
        # Original value should be restored
        assert physics_context.beam_energy == 10.58
    
    def test_resource_manager_singleton(self):
        """Test ResourceManager singleton pattern."""
        rm1 = ResourceManager()
        rm2 = ResourceManager()
        assert rm1 is rm2
    
    def test_computation_cache(self):
        """Test computation cache with memory limits."""
        cache = ComputationCache(max_memory=1024)  # 1KB limit
        
        # Add items
        cache.put("key1", np.zeros(100), 800)  # 800 bytes
        cache.put("key2", np.zeros(50), 400)   # 400 bytes - should evict key1
        
        # Check eviction
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        
        # Test LRU
        cache.put("key3", np.zeros(10), 100)
        cache.get("key2")  # Access key2
        cache.put("key4", np.zeros(40), 350)  # Should evict key3, not key2
        
        assert cache.get("key3") is None
        assert cache.get("key2") is not None
    
    @profile_performance
    def test_event_bus(self):
        """Test event bus publish/subscribe."""
        received_events = []
        
        def handler(event):
            received_events.append(event)
        
        # Subscribe
        event_bus.subscribe(Event, handler)
        
        # Publish
        test_event = Event()
        event_bus.publish(test_event)
        
        # Verify
        assert len(received_events) == 1
        assert received_events[0] is test_event


# ============================================================================
# UNIT TESTS: Luminosity Processor
# ============================================================================

class TestLuminosityProcessor:
    """Test luminosity processing engine."""
    
    @pytest.fixture
    def lumi_processor(self, physics_context):
        """Create luminosity processor."""
        return create_luminosity_processor(physics_context)
    
    def test_luminosity_database_singleton(self):
        """Test database singleton pattern."""
        db1 = LuminosityDatabase()
        db2 = LuminosityDatabase()
        assert db1 is db2
    
    def test_process_name_parsing(self, lumi_processor):
        """Test various process name formats."""
        test_cases = [
            ("P16M16rd_mc4S_uubar_p16_v1", "uubar", BeamEnergyCondition.Y4S_ON_RESONANCE),
            ("data_proc_4S_on", "data", BeamEnergyCondition.Y4S_ON_RESONANCE),
            ("mc5S_ccbar", "ccbar", BeamEnergyCondition.Y5S_SCAN),
            ("BBbar", "BBbar", BeamEnergyCondition.Y4S_ON_RESONANCE),  # Alias
        ]
        
        for process_name, expected_physics, expected_energy in test_cases:
            process_id = lumi_processor.database._parse_process_name(process_name)
            assert process_id.physics_process == expected_physics
            assert process_id.energy_condition == expected_energy
    
    @benchmark(n_iterations=5)
    def test_luminosity_lookup_performance(self, lumi_processor):
        """Benchmark luminosity lookups."""
        processes = [f"mc_{i}_process" for i in range(1000)]
        
        for process in processes:
            lumi_processor.database.lookup(process)
    
    def test_weight_calculation(self, lumi_processor):
        """Test luminosity weight calculations."""
        # Known process
        weight = lumi_processor.calculate_weight("uubar", reference_luminosity=357.3)
        assert isinstance(weight, LuminosityValue)
        assert weight.value > 0
        assert weight.total_uncertainty > 0
        
        # Test uncertainty propagation
        lumi1 = LuminosityValue(100.0, stat_uncertainty=5.0, syst_uncertainty=3.0)
        lumi2 = LuminosityValue(50.0, stat_uncertainty=2.0, syst_uncertainty=1.0)
        
        # Division
        ratio = lumi1 / lumi2
        assert abs(ratio.value - 2.0) < 1e-10
        assert ratio.total_uncertainty > 0
    
    def test_batch_processing(self, lumi_processor):
        """Test batch luminosity processing."""
        processes = ["uubar", "ddbar", "ssbar", "ccbar", "mumu"]
        
        batch_results = lumi_processor.database.batch_lookup(processes)
        
        assert len(batch_results) == len(processes)
        for process in processes:
            assert process in batch_results
            assert isinstance(batch_results[process], LuminosityValue)


# ============================================================================
# UNIT TESTS: Statistical Engine
# ============================================================================

class TestStatisticalEngine:
    """Test statistical analysis engine."""
    
    @pytest.fixture
    def stat_engine(self, physics_context):
        """Create statistical engine."""
        return create_statistical_engine(physics_context)
    
    def test_method_selection(self, stat_engine):
        """Test automatic method selection."""
        # Small counts
        small_data = np.array([2, 3, 1, 4, 2])
        method = stat_engine._select_optimal_method(small_data, None, 0.68)
        assert method == StatisticalMethod.CLOPPER_PEARSON
        
        # Large continuous data
        large_data = np.random.normal(100, 10, 10000)
        method = stat_engine._select_optimal_method(large_data, None, 0.68)
        assert method == StatisticalMethod.WILSON_SCORE
        
        # Weighted data
        weighted_data = np.random.exponential(1, 100)
        weights = np.random.uniform(0.5, 2, 100)
        method = stat_engine._select_optimal_method(weighted_data, weights, 0.68)
        assert method == StatisticalMethod.BOOTSTRAP_BCA
    
    def test_uncertainty_methods(self, stat_engine):
        """Test different uncertainty calculation methods."""
        # Test data
        counts = np.array([10, 15, 8, 12, 20])
        
        methods_to_test = [
            StatisticalMethod.CLOPPER_PEARSON,
            StatisticalMethod.WILSON_SCORE,
            StatisticalMethod.JEFFREYS_HPD,
        ]
        
        results = {}
        for method in methods_to_test:
            lower, upper = stat_engine.compute_uncertainty(
                counts, method, confidence_level=0.68
            )
            results[method.method_name] = (lower, upper)
            
            # Basic validation
            assert np.all(lower >= 0)
            assert np.all(upper >= 0)
            assert np.all(upper >= lower)
    
    @profile_performance
    def test_bootstrap_performance(self, stat_engine):
        """Test bootstrap method performance."""
        data = np.random.exponential(2, 1000)
        
        # Time bootstrap
        lower, upper = stat_engine.compute_uncertainty(
            data,
            StatisticalMethod.BOOTSTRAP_BCA,
            confidence_level=0.68,
            n_bootstrap=2000
        )
        
        assert len(lower) == 1
        assert len(upper) == 1
    
    def test_weighted_uncertainties(self, stat_engine):
        """Test weighted uncertainty calculations."""
        data = np.random.normal(100, 10, 1000)
        weights = np.random.uniform(0.5, 2.0, 1000)
        
        # Compute weighted uncertainties
        result = stat_engine._compute_statistics_lazy(
            Mock(__getitem__=lambda x: data),
            column=None,
            weight_column=None,
            method=StatisticalMethod.WILSON_SCORE,
            confidence_level=0.68
        )
        
        assert isinstance(result, StatisticalResult)
        assert result.effective_sample_size < len(data)
    
    def test_systematic_uncertainties(self, stat_engine):
        """Test systematic uncertainty combination."""
        # Nominal result
        nominal = StatisticalResult(
            central_value=100.0,
            lower_uncertainty=5.0,
            upper_uncertainty=5.0,
            confidence_level=0.68,
            method=StatisticalMethod.JEFFREYS_HPD
        )
        
        # Systematic variations
        variations = {
            'tracking_eff': StatisticalResult(102.0, 5.1, 5.1, 0.68, StatisticalMethod.JEFFREYS_HPD),
            'pid_eff': StatisticalResult(98.0, 4.9, 4.9, 0.68, StatisticalMethod.JEFFREYS_HPD),
        }
        
        # Combine
        combined = stat_engine.compute_systematic_uncertainties(nominal, variations)
        
        assert combined.systematic_uncertainties is not None
        assert 'tracking_eff' in combined.systematic_uncertainties
        assert combined.lower_uncertainty > nominal.lower_uncertainty


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple Layer 3 components."""
    
    @pytest.fixture
    def integrated_setup(self, physics_context):
        """Create integrated test setup."""
        return {
            'context': physics_context,
            'lumi_processor': create_luminosity_processor(physics_context),
            'stat_engine': create_statistical_engine(physics_context),
            'data_generator': BelleIIDataGenerator(),
            'visualizer': VisualizationTester(TEST_CONFIG.output_dir)
        }
    
    def test_full_analysis_workflow(self, integrated_setup):
        """Test complete analysis workflow."""
        setup = integrated_setup
        
        # 1. Generate mock data
        signal = setup['data_generator'].generate_b_decay_events(10000)
        background = setup['data_generator'].generate_continuum_background(50000, 'uubar')
        
        # 2. Apply luminosity weights
        signal_weight = setup['lumi_processor'].calculate_weight('BBbar')
        background_weight = setup['lumi_processor'].calculate_weight('uubar')
        
        signal['weight'] *= signal_weight.value
        background['weight'] *= background_weight.value
        
        # 3. Combine datasets
        combined = pd.concat([signal, background], ignore_index=True)
        
        # 4. Compute statistical uncertainties
        mass_data = combined['mass'].values
        weights = combined['weight'].values
        
        lower, upper = setup['stat_engine'].compute_uncertainty(
            mass_data,
            StatisticalMethod.BOOTSTRAP_BCA,
            confidence_level=0.68,
            weights=weights,
            n_bootstrap=1000
        )
        
        # 5. Visualize results
        fig = setup['visualizer'].plot_histogram_with_errors(
            mass_data,
            (lower * np.ones(50), upper * np.ones(50)),
            bins=50,
            title="B Meson Mass Distribution",
            save_name="b_meson_mass"
        )
        
        plt.close(fig)
    
    def test_systematic_variation_study(self, integrated_setup):
        """Test systematic uncertainty evaluation."""
        setup = integrated_setup
        
        # Generate nominal dataset
        nominal_data = setup['data_generator'].generate_b_decay_events(5000)
        
        # Define systematic variations
        systematic_variations = {
            'tracking_up': lambda d: d * 1.02,
            'tracking_down': lambda d: d * 0.98,
            'pid_up': lambda d: d * 1.01,
            'pid_down': lambda d: d * 0.99,
        }
        
        # Compute for each variation
        results = {}
        momentum = nominal_data['momentum'].values
        
        for name, variation in systematic_variations.items():
            varied_momentum = variation(momentum)
            
            lower, upper = setup['stat_engine'].compute_uncertainty(
                varied_momentum,
                StatisticalMethod.JEFFREYS_HPD,
                confidence_level=0.68
            )
            
            results[name] = np.mean(varied_momentum)
        
        # Visualize systematic effects
        nominal_value = np.mean(momentum)
        fig = setup['visualizer'].plot_systematic_variations(
            np.array([nominal_value]),
            {k: np.array([v]) for k, v in results.items()},
            title="Systematic Uncertainty Study",
            save_name="systematics_study"
        )
        
        plt.close(fig)
    
    @benchmark()
    def test_performance_scaling(self, integrated_setup):
        """Test performance scaling with data size."""
        setup = integrated_setup
        
        sizes = [100, 1000, 10000, 100000]
        times = []
        
        for size in sizes:
            data = np.random.poisson(10, size)
            
            start = time.perf_counter()
            setup['stat_engine'].compute_uncertainty(
                data,
                StatisticalMethod.JEFFREYS_HPD,
                confidence_level=0.68
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Check scaling (should be roughly linear for most methods)
        print(f"\nScaling test results:")
        for size, t in zip(sizes, times):
            print(f"  N={size:6d}: {t:.4f}s")


# ============================================================================
# VISUAL VALIDATION TESTS
# ============================================================================

class TestVisualValidation:
    """Visual tests for validating statistical methods."""
    
    @pytest.fixture
    def visual_setup(self, test_output_dir):
        """Setup for visual tests."""
        return {
            'stat_engine': create_statistical_engine(),
            'visualizer': VisualizationTester(test_output_dir),
            'output_dir': test_output_dir
        }
    
    def test_coverage_validation(self, visual_setup):
        """Validate coverage properties of uncertainty methods."""
        if not TEST_CONFIG.enable_visual_tests:
            pytest.skip("Visual tests disabled")
        
        setup = visual_setup
        
        # Parameters
        true_rate = 0.3
        n_trials = 100
        n_experiments = 1000
        confidence_level = 0.68
        
        methods = [
            StatisticalMethod.CLOPPER_PEARSON,
            StatisticalMethod.WILSON_SCORE,
            StatisticalMethod.JEFFREYS_HPD,
        ]
        
        coverage_results = {method.method_name: [] for method in methods}
        
        # Run coverage test
        np.random.seed(TEST_CONFIG.random_seed)
        
        for _ in range(n_experiments):
            # Generate data
            successes = np.random.binomial(n_trials, true_rate)
            
            for method in methods:
                # Compute interval
                lower, upper = setup['stat_engine'].compute_uncertainty(
                    np.array([successes]),
                    method,
                    confidence_level,
                    n_total=n_trials
                )
                
                # Check coverage
                lower_rate = (successes - lower[0]) / n_trials
                upper_rate = (successes + upper[0]) / n_trials
                
                covered = lower_rate <= true_rate <= upper_rate
                coverage_results[method.method_name].append(covered)
        
        # Plot coverage results
        fig, ax = plt.subplots(figsize=(10, 6))
        
        method_names = list(coverage_results.keys())
        coverages = [np.mean(coverage_results[m]) * 100 for m in method_names]
        
        bars = ax.bar(method_names, coverages)
        ax.axhline(y=confidence_level * 100, color='red', linestyle='--',
                  label=f'Target: {confidence_level*100:.0f}%')
        
        # Color bars based on coverage
        for bar, coverage in zip(bars, coverages):
            if abs(coverage - confidence_level * 100) < 2:
                bar.set_color('green')
            elif abs(coverage - confidence_level * 100) < 5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_ylabel('Coverage (%)')
        ax.set_title(f'Coverage Test: {n_experiments} experiments, n={n_trials}, p={true_rate}')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if TEST_CONFIG.save_plots:
            plt.savefig(setup['output_dir'] / 'coverage_validation.png',
                       dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_error_bar_visualization(self, visual_setup):
        """Test various error bar visualizations."""
        if not TEST_CONFIG.enable_visual_tests:
            pytest.skip("Visual tests disabled")
        
        setup = visual_setup
        
        # Generate test data
        x = np.linspace(0, 10, 20)
        y = 100 * np.exp(-x/5) + np.random.normal(0, 5, 20)
        
        # Compute Poisson-like uncertainties
        lower_err = np.sqrt(np.maximum(y, 1))
        upper_err = lower_err * 1.2  # Asymmetric
        
        # Create multiple visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Standard error bars
        ax = axes[0, 0]
        ax.errorbar(x, y, yerr=[lower_err, upper_err], fmt='o',
                   capsize=5, capthick=2, label='Data')
        ax.set_title('Standard Error Bars')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Filled error band
        ax = axes[0, 1]
        ax.plot(x, y, 'o-', label='Data')
        ax.fill_between(x, y - lower_err, y + upper_err,
                       alpha=0.3, label='Error band')
        ax.set_title('Error Band')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Relative uncertainties
        ax = axes[1, 0]
        rel_err = (lower_err + upper_err) / (2 * np.abs(y))
        ax.plot(x, rel_err * 100, 'o-')
        ax.set_title('Relative Uncertainties')
        ax.set_xlabel('X')
        ax.set_ylabel('Relative Error (%)')
        ax.grid(True, alpha=0.3)
        
        # 4. Pull distribution
        ax = axes[1, 1]
        fit = np.polyfit(x, y, 2)
        y_fit = np.polyval(fit, x)
        pulls = (y - y_fit) / lower_err
        
        ax.hist(pulls, bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_title('Pull Distribution')
        ax.set_xlabel('(Data - Fit) / Error')
        ax.set_ylabel('Entries')
        ax.grid(True, alpha=0.3)
        
        # Add Gaussian overlay
        x_gauss = np.linspace(-3, 3, 100)
        y_gauss = len(pulls) * 0.4 * np.exp(-x_gauss**2/2)
        ax.plot(x_gauss, y_gauss, 'r-', linewidth=2, label='Unit Gaussian')
        ax.legend()
        
        plt.tight_layout()
        
        if TEST_CONFIG.save_plots:
            plt.savefig(setup['output_dir'] / 'error_visualizations.png',
                       dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_efficiency_plots(self, visual_setup):
        """Test efficiency curve visualizations."""
        if not TEST_CONFIG.enable_visual_tests:
            pytest.skip("Visual tests disabled")
        
        setup = visual_setup
        
        # Generate efficiency curve
        pt_bins = np.linspace(0, 10, 21)
        pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
        
        # True efficiency curve
        true_eff = 0.9 * (1 - np.exp(-pt_centers/2))
        
        # Simulate measurements
        n_total = 1000 * np.ones_like(pt_centers)
        n_passed = np.random.binomial(n_total.astype(int), true_eff)
        
        # Compute uncertainties
        efficiencies = n_passed / n_total
        
        lower_errs = []
        upper_errs = []
        
        for i in range(len(pt_centers)):
            lower, upper = setup['stat_engine'].compute_uncertainty(
                np.array([n_passed[i]]),
                StatisticalMethod.CLOPPER_PEARSON,
                confidence_level=0.68,
                n_total=int(n_total[i])
            )
            
            lower_errs.append(lower[0] / n_total[i])
            upper_errs.append(upper[0] / n_total[i])
        
        lower_errs = np.array(lower_errs)
        upper_errs = np.array(upper_errs)
        
        # Create plot
        fig = setup['visualizer'].plot_efficiency_curve(
            pt_centers,
            efficiencies,
            (lower_errs, upper_errs),
            title="Trigger Efficiency vs pT",
            xlabel="pT (GeV)",
            save_name="efficiency_curve"
        )
        
        # Add true curve
        ax = fig.gca()
        ax.plot(pt_centers, true_eff, 'r--', linewidth=2, label='True efficiency')
        ax.legend()
        
        plt.close(fig)
    
    def test_2d_weighted_distribution(self, visual_setup):
        """Test 2D distribution plotting with weights."""
        if not TEST_CONFIG.enable_visual_tests:
            pytest.skip("Visual tests disabled")
        
        setup = visual_setup
        
        # Generate correlated 2D data
        mean = [2.0, 3.0]
        cov = [[1.0, 0.7], [0.7, 2.0]]
        n_points = 10000
        
        data = np.random.multivariate_normal(mean, cov, n_points)
        x, y = data[:, 0], data[:, 1]
        
        # Generate weights (e.g., from luminosity)
        weights = np.random.exponential(1.0, n_points)
        weights *= 100 / np.sum(weights)  # Normalize
        
        # Create plots
        fig1 = setup['visualizer'].plot_2d_distribution(
            x, y, None,
            title="Unweighted 2D Distribution",
            xlabel="Variable X",
            ylabel="Variable Y",
            save_name="2d_unweighted"
        )
        plt.close(fig1)
        
        fig2 = setup['visualizer'].plot_2d_distribution(
            x, y, weights,
            title="Weighted 2D Distribution",
            xlabel="Variable X",
            ylabel="Variable Y",
            save_name="2d_weighted"
        )
        plt.close(fig2)


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    def test_scaling_benchmark(self, test_output_dir):
        """Benchmark scaling behavior."""
        if not TEST_CONFIG.enable_performance_tests:
            pytest.skip("Performance tests disabled")
        
        sizes = [100, 1000, 10000, 100000]
        methods = [
            StatisticalMethod.CLOPPER_PEARSON,
            StatisticalMethod.WILSON_SCORE,
            StatisticalMethod.JEFFREYS_HPD,
            StatisticalMethod.BOOTSTRAP_PERCENTILE,
        ]
        
        results = {method.method_name: [] for method in methods}
        
        engine = create_statistical_engine()
        
        for size in sizes:
            data = np.random.poisson(10, size)
            
            for method in methods:
                if method == StatisticalMethod.BOOTSTRAP_PERCENTILE and size > 10000:
                    continue  # Skip slow methods for large data
                
                start = time.perf_counter()
                engine.compute_uncertainty(data, method, 0.68)
                elapsed = time.perf_counter() - start
                
                results[method.method_name].append((size, elapsed))
        
        # Plot scaling
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method_name, timings in results.items():
            if timings:
                sizes_plot = [t[0] for t in timings]
                times_plot = [t[1] for t in timings]
                ax.loglog(sizes_plot, times_plot, 'o-', label=method_name)
        
        ax.set_xlabel('Data size')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Statistical Method Scaling Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(test_output_dir / 'performance_scaling.png',
                   dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests(output_dir: Optional[Path] = None):
    """Run all Layer 3 tests with reporting."""
    
    if output_dir:
        TEST_CONFIG.output_dir = output_dir
    
    # Create output directory
    TEST_CONFIG.output_dir.mkdir(exist_ok=True)
    
    # Run pytest
    pytest_args = [
        __file__,
        "-v",
        f"--tb=short",
        f"--junit-xml={TEST_CONFIG.output_dir}/test_results.xml"
    ]
    
    if TEST_CONFIG.enable_performance_tests:
        pytest_args.append("--durations=10")
    
    return pytest.main(pytest_args)


if __name__ == "__main__":
    # Run tests
    print("🧪 Running Layer 3 Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test visualizer for final summary
    visualizer = VisualizationTester(TEST_CONFIG.output_dir)
    
    # Run tests
    exit_code = run_all_tests()
    
    # Create summary report
    if TEST_CONFIG.enable_visual_tests:
        report_path = visualizer.create_summary_report("layer3_comprehensive")
        print(f"\n📊 Test report generated: {report_path}")
    
    print(f"\n{'✅ Tests passed!' if exit_code == 0 else '❌ Tests failed!'}")
    
    exit(exit_code)