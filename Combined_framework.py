"""
Billion-Capable Belle II Framework - Unified Implementation
==========================================================

This implementation combines the best components from both enhanced and unified
frameworks, leveraging imports and inheritance for maximum efficiency.

Performance Targets:
- 10M rows: <0.5s for histograms, <0.2s for aggregations
- 100M rows: <5s for histograms, <2s for aggregations  
- 1B rows: <50s for histograms, <25s for aggregations

Key Features:
- C++ acceleration integration
- Optimized streaming operations
- Unified query conversion
- Production-ready merge operations
- Comprehensive benchmarking
"""
#modif try
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# ============================================================================
# IMPORTS FROM EXISTING MODULES
# ============================================================================
from unified_framework import (
    BlazingCore,
    SmartEvaluator,
    DataLoader,
    PerformanceCore,
    UnifiedAPI,
    LazySeriesProxy,
    Belle2Extensions,
    PandasCompatLayer,
    StreamingAggregator,
    StreamingFilter,
    PerformanceMonitor,
    StreamingHistogram
)
from enhanced_framework import (
    EnhancedStreamingHistogram,
    UnifiedQueryConverter,
    OptimizedGroupByHandler,
    StreamingMergeEngine,
    EnhancedBlazingCore
)




# ============================================================================
# BILLION-CAPABLE BLAZING CORE - Inherits and enhances
# ============================================================================
class BillionCapableBlazingCore(EnhancedBlazingCore):
    """
    Ultra-optimized BlazingCore with proper C++ integration for billion-row datasets.
    """
    
    def __init__(self, lazy_frames: List[pl.LazyFrame]):
        # Initialize parent with enhanced components
        super().__init__(lazy_frames)
        
        # Additional billion-row optimizations
        self._billion_row_mode = self._estimated_total_rows > 100_000_000
        self._adaptive_chunk_size = self._calculate_optimal_chunk_size()
        
        if self._billion_row_mode:
            print(f"🚀 Billion-row mode activated: ~{self._estimated_total_rows:,} rows")
            print(f"📊 Optimal chunk size: {self._adaptive_chunk_size:,} rows")
            
            # Ensure C++ acceleration is being used for billion-row datasets
            if hasattr(self.streaming_histogram, '_cpp_available'):
                if self.streaming_histogram._cpp_available:
                    print("⚡ C++ acceleration: ACTIVE for billion-row processing")
                else:
                    print("⚠️ C++ acceleration: NOT AVAILABLE - performance may be limited")
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on dataset size and available memory."""
        if self._estimated_total_rows <= 10_000_000:
            return 1_000_000
        elif self._estimated_total_rows <= 100_000_000:
            return 5_000_000
        else:
            # For billion-row datasets, use larger chunks
            return 10_000_000
    
    def hist(self, column: str, bins: int = 50, range: Optional[Tuple[float, float]] = None,
             density: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced histogram with automatic C++ acceleration for billion-row datasets.
        
        This properly delegates to the enhanced streaming histogram which will
        automatically use C++ acceleration when beneficial.
        """
        # Log performance expectations
        if self._billion_row_mode:
            expected_time = self._estimated_total_rows / 20_000_000  # 20M rows/s target
            print(f"📊 Computing histogram on ~{self._estimated_total_rows:,} rows")
            print(f"⏱️ Expected time: ~{expected_time:.1f}s with C++ acceleration")
        
        # Delegate to enhanced parent implementation
        return super().hist(column, bins, range, density, **kwargs)


# ============================================================================
# BILLION-CAPABLE UNIFIED API - Combines all functionality
# ============================================================================

class BillionCapableUnifiedAPI(UnifiedAPI):
    """
    Enhanced UnifiedAPI with billion-row optimizations and merge capabilities.
    """
    
    def __init__(self, blazing_core: BillionCapableBlazingCore, smart_evaluator: SmartEvaluator):
        super().__init__(blazing_core, smart_evaluator)
        
        # Add merge engine for billion-row joins
        self.merge_engine = StreamingMergeEngine()
        
        # Override query converter with unified version
        self.query_converter = UnifiedQueryConverter()
    
    def merge(self, other: 'BillionCapableFramework', on: Union[str, List[str]], 
              how: str = 'inner', validate: str = None) -> 'BillionCapableFramework':
        """
        Billion-row optimized merge using StreamingMergeEngine.
        """
        print(f"🔀 Executing billion-row merge on '{on}' with strategy: auto")
        
        merged_frames = self.merge_engine.merge(
            self.blazing_core.lazy_frames,
            other.blazing_core.lazy_frames,
            on=on, how=how
        )
        
        # Create new framework instance with merged data
        new_framework = BillionCapableFramework(
            memory_budget_gb=self.smart_evaluator.memory_budget_gb
        )
        new_framework.blazing_core = BillionCapableBlazingCore(merged_frames)
        new_framework.smart_evaluator = self.smart_evaluator
        new_framework.unified_api = BillionCapableUnifiedAPI(
            new_framework.blazing_core, 
            new_framework.smart_evaluator
        )
        
        return new_framework
    
    def groupby(self, by: Union[str, List[str]]) -> OptimizedGroupByHandler:
        """
        Use OptimizedGroupByHandler for true streaming GroupBy.
        """
        if isinstance(by, str):
            by = [by]
        return OptimizedGroupByHandler(self.blazing_core, by)


# ============================================================================
# BILLION-CAPABLE DATA LOADER - Enhanced parallel loading
# ============================================================================

class BillionCapableDataLoader(DataLoader):
    """
    Enhanced DataLoader with billion-row specific optimizations.
    """
    
    def __init__(self, max_workers: int = 8):
        # Use more workers for billion-row datasets
        super().__init__(max_workers)
        
        # Additional billion-row settings
        self.billion_row_threshold = 100_000_000
        self.adaptive_partitioning = True
    
    def load_belle2_data(self, paths: Union[str, List[str]], 
                        strategy: str = 'auto') -> Dict[str, List[pl.LazyFrame]]:
        """
        Enhanced loading with billion-row awareness.
        """
        result = super().load_belle2_data(paths, strategy)
        
        # Post-process for billion-row optimization
        for process, frames in result.items():
            estimated_rows = self._estimate_process_size(frames)
            if estimated_rows > self.billion_row_threshold:
                print(f"📈 Process '{process}' detected as billion-row dataset")
                # Could add repartitioning logic here if needed
        
        return result
    
    def _estimate_process_size(self, frames: List[pl.LazyFrame]) -> int:
        """Estimate total rows for a process."""
        try:
            sample_size = min(3, len(frames))
            total = 0
            for frame in frames[:sample_size]:
                count = frame.select(pl.count()).collect(streaming=True)[0, 0]
                total += count
            return (total // sample_size) * len(frames) if sample_size > 0 else 0
        except:
            return 0


# ============================================================================
# MAIN BILLION-CAPABLE FRAMEWORK
# ============================================================================

class BillionCapableFramework:
    """
    The ultimate billion-capable Belle II framework.
    
    Combines all optimizations from both implementations:
    - C++ acceleration from enhanced framework
    - Comprehensive API from unified framework
    - Optimized GroupBy and merge operations
    - Advanced performance monitoring
    
    Usage:
        framework = BillionCapableFramework(memory_budget_gb=32.0)
        framework.load_data("/path/to/data")
        
        # Ultra-fast operations on billions of rows
        hist = framework.hist('M_bc', bins=100)
        filtered = framework.query("M_bc > 5.27 & delta_E < 0.1")
        grouped = framework.groupby(['evt', 'run']).agg({'M_bc': ['mean', 'std']})
    """
    
    def __init__(self, memory_budget_gb: float = 16.0, max_workers: int = 8):
        """
        Initialize billion-capable framework.
        
        Args:
            memory_budget_gb: Memory budget (default 16GB for billion-row datasets)
            max_workers: Parallel workers (default 8 for faster loading)
        """
        self.memory_budget_gb = memory_budget_gb
        self.max_workers = max_workers
        
        # Initialize components
        self.data_loader = BillionCapableDataLoader(max_workers)
        self.performance_core = PerformanceCore(
            max_cache_size_gb=memory_budget_gb * 0.25,
            memory_warning_threshold_gb=memory_budget_gb * 0.8
        )
        
        # Framework state
        self.blazing_core = None
        self.smart_evaluator = None
        self.unified_api = None
        self.current_process = None
        self._lazy_frames_by_process = {}
        
        print(f"🚀 Billion-Capable Belle II Framework initialized")
        print(f"💾 Memory budget: {memory_budget_gb:.1f} GB")
        print(f"⚙️ Max workers: {max_workers}")
        
        # Check for C++ acceleration
        try:
            from cpp_histogram_integrator import cpp_histogram_integrator
            if cpp_histogram_integrator.is_available():
                print("⚡ C++ acceleration: ENABLED")
            else:
                print("⚠️ C++ acceleration: NOT AVAILABLE")
        except:
            print("⚠️ C++ acceleration: NOT INSTALLED")
    
    def load_data(self, paths: Union[str, List[str]], 
                  process: str = 'auto', strategy: str = 'auto') -> 'BillionCapableFramework':
        """
        Load data with billion-row optimization.
        """
        print(f"\n📂 Loading data from: {paths}")
        
        # Load data with enhanced loader
        self._lazy_frames_by_process = self.data_loader.load_belle2_data(paths, strategy)
        
        if not self._lazy_frames_by_process:
            raise ValueError("No data loaded. Check paths and file formats.")
        
        # Select process
        if process == 'auto':
            process = max(self._lazy_frames_by_process.keys(), 
                         key=lambda k: len(self._lazy_frames_by_process[k]))
            print(f"🎯 Auto-selected process: {process}")
        elif process not in self._lazy_frames_by_process:
            available = list(self._lazy_frames_by_process.keys())
            raise ValueError(f"Process '{process}' not found. Available: {available}")
        
        self.current_process = process
        lazy_frames = self._lazy_frames_by_process[process]
        
        # Initialize billion-capable components
        self.blazing_core = BillionCapableBlazingCore(lazy_frames)
        self.smart_evaluator = SmartEvaluator(self.memory_budget_gb)
        self.unified_api = BillionCapableUnifiedAPI(self.blazing_core, self.smart_evaluator)
        
        # Report loading stats
        total_frames = len(lazy_frames)
        estimated_rows = self.blazing_core._estimated_total_rows
        print(f"✅ Loaded {total_frames} frames, ~{estimated_rows:,} rows for process '{process}'")
        
        if estimated_rows > 1_000_000_000:
            print(f"🌟 BILLION-ROW DATASET DETECTED! Optimizations engaged.")
        
        return self
    
    # Delegate all operations to unified API
    def __getattr__(self, name: str):
        """Delegate to unified API for all operations."""
        if self.unified_api and hasattr(self.unified_api, name):
            return getattr(self.unified_api, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def hist(self, column: str, bins: int = 50, range: Optional[Tuple[float, float]] = None,
             density: bool = False, ax: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-fast histogram with C++ acceleration where available."""
        counts, edges = self.performance_core.execute_with_optimization(
            'histogram',
            self.unified_api.hist,
            self.blazing_core._estimated_total_rows,
            column, bins, range, density, **kwargs
        )
        
        # Optional plotting
        if ax is not None:
            ax.stairs(counts, edges, label=kwargs.get('label', column))
            ax.set_xlabel(column)
            ax.set_ylabel('Count' if not density else 'Density')
        
        return counts, edges
    
    def query(self, expr: str) -> 'BillionCapableFramework':
        """Query with unified converter and streaming optimization."""
        filtered_core = self.performance_core.execute_with_optimization(
            'query',
            self.unified_api.query,
            self.blazing_core._estimated_total_rows,
            expr
        )
        
        # Create new framework instance with filtered data
        new_framework = BillionCapableFramework(self.memory_budget_gb, self.max_workers)
        new_framework.blazing_core = filtered_core
        new_framework.smart_evaluator = self.smart_evaluator
        new_framework.unified_api = BillionCapableUnifiedAPI(filtered_core, self.smart_evaluator)
        new_framework.current_process = self.current_process
        
        return new_framework
    
    def groupby(self, by: Union[str, List[str]]) -> OptimizedGroupByHandler:
        """Optimized streaming GroupBy."""
        return self.unified_api.groupby(by)
    
    def merge(self, other: 'BillionCapableFramework', on: Union[str, List[str]], 
              how: str = 'inner', validate: str = None) -> 'BillionCapableFramework':
        """Billion-row optimized merge."""
        return self.unified_api.merge(other, on, how, validate)
    
    def createDeltaColumns(self, base_col: str, target_cols: List[str]) -> 'BillionCapableFramework':
        """Belle II specific: create delta columns."""
        enhanced_core = self.unified_api.belle2_ext.createDeltaColumns(base_col, target_cols)
        
        # Create new framework with enhanced data
        new_framework = BillionCapableFramework(self.memory_budget_gb, self.max_workers)
        new_framework.blazing_core = enhanced_core
        new_framework.smart_evaluator = self.smart_evaluator
        new_framework.unified_api = BillionCapableUnifiedAPI(enhanced_core, self.smart_evaluator)
        new_framework.current_process = self.current_process
        
        return new_framework
    
    def oneCandOnly(self, group_cols: Union[str, List[str]], sort_col: str, 
                    ascending: bool = False, physics_mode: Optional[str] = None) -> 'BillionCapableFramework':
        """Belle II critical: select one best candidate per event."""
        selected_core = self.unified_api.belle2_ext.oneCandOnly(group_cols, sort_col, ascending)
        
        # Create new framework with selected candidates
        new_framework = BillionCapableFramework(self.memory_budget_gb, self.max_workers)
        new_framework.blazing_core = selected_core
        new_framework.smart_evaluator = self.smart_evaluator
        new_framework.unified_api = BillionCapableUnifiedAPI(selected_core, self.smart_evaluator)
        new_framework.current_process = self.current_process
        
        return new_framework
    
    def benchmark_suite(self, column: Optional[str] = None, 
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive benchmark suite with billion-row focus."""
        if not self.blazing_core:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Auto-select column
        if column is None:
            schema = self.blazing_core.lazy_frames[0].schema
            numeric_types = {pl.Float64, pl.Float32, pl.Int64, pl.Int32}
            numeric_cols = [col for col, dtype in schema.items() if dtype in numeric_types]
            if numeric_cols:
                column = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found")
        
        print(f"\n🔥 Running BILLION-ROW benchmark suite")
        print(f"📊 Test column: {column}")
        print(f"📈 Dataset size: ~{self.blazing_core._estimated_total_rows:,} rows")
        print("=" * 60)
        
        results = {
            'metadata': {
                'column': column,
                'estimated_rows': self.blazing_core._estimated_total_rows,
                'billion_row_mode': self.blazing_core._billion_row_mode,
                'memory_budget_gb': self.memory_budget_gb,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'benchmarks': {}
        }
        
        # Test suite optimized for billion-row datasets
        tests = [
            ('histogram_50bins', lambda: self.hist(column, bins=50)),
            ('histogram_100bins', lambda: self.hist(column, bins=100)),
            ('histogram_1000bins', lambda: self.hist(column, bins=1000)),  # Stress test
            ('sum', lambda: self.unified_api.sum()),
            ('mean', lambda: self.unified_api.mean()),
            ('std', lambda: self.unified_api.std()),
            ('count', lambda: self.unified_api.count()),
            ('query_simple', lambda: self.query(f"{column} > 0").unified_api.count()),
            ('query_complex', lambda: self.query(f"{column} > 0 & {column} < 1").unified_api.count()),
            ('groupby_single', lambda: self.groupby(column).agg({column: 'count'})),
        ]
        
        # Add billion-row specific tests
        if self.blazing_core._billion_row_mode:
            tests.extend([
                ('query_multifield', lambda: self.query(f"{column} > 0.5 & {column} < 0.8").unified_api.count()),
                ('histogram_10000bins', lambda: self.hist(column, bins=10000)),  # Extreme stress test
            ])
        
        for test_name, test_func in tests:
            print(f"\n⏱️ {test_name}...", end='', flush=True)
            try:
                start = time.perf_counter()
                _ = test_func()
                duration = time.perf_counter() - start
                
                throughput = self.blazing_core._estimated_total_rows / duration / 1e6
                results['benchmarks'][test_name] = {
                    'duration_s': duration,
                    'throughput_M_rows_s': throughput,
                    'status': 'success'
                }
                print(f" ✅ {duration:.2f}s ({throughput:.1f}M rows/s)")
                
            except Exception as e:
                results['benchmarks'][test_name] = {
                    'duration_s': None,
                    'throughput_M_rows_s': None,
                    'status': 'failed',
                    'error': str(e)
                }
                print(f" ❌ Failed: {e}")
        
        # Summary statistics
        successful = [b for b in results['benchmarks'].values() if b['status'] == 'success']
        if successful:
            avg_throughput = np.mean([b['throughput_M_rows_s'] for b in successful])
            max_throughput = max([b['throughput_M_rows_s'] for b in successful])
            results['summary'] = {
                'avg_throughput_M_rows_s': avg_throughput,
                'max_throughput_M_rows_s': max_throughput,
                'successful_tests': len(successful),
                'failed_tests': len(results['benchmarks']) - len(successful),
                'billion_row_capable': avg_throughput > 20  # 20M rows/s threshold
            }
            
            print(f"\n📊 Summary:")
            print(f"   Average throughput: {avg_throughput:.1f}M rows/s")
            print(f"   Peak throughput: {max_throughput:.1f}M rows/s")
            print(f"   Success rate: {len(successful)}/{len(results['benchmarks'])}")
            print(f"   Billion-row capable: {'✅ YES' if results['summary']['billion_row_capable'] else '❌ NO'}")
        
        # Save results
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Comprehensive performance report."""
        report = {
            'framework': {
                'version': '3.0-billion-capable',
                'process': self.current_process,
                'memory_budget_gb': self.memory_budget_gb,
                'billion_row_mode': self.blazing_core._billion_row_mode if self.blazing_core else False,
                'processes_loaded': list(self._lazy_frames_by_process.keys())
            }
        }
        
        if self.blazing_core:
            report['data'] = {
                'estimated_rows': self.blazing_core._estimated_total_rows,
                'num_frames': len(self.blazing_core.lazy_frames),
                'adaptive_chunk_size': self.blazing_core._adaptive_chunk_size
            }
            report['performance'] = self.blazing_core.get_performance_stats()
        
        if self.smart_evaluator:
            report['evaluation'] = self.smart_evaluator.get_evaluation_stats()
        
        if self.performance_core:
            report['optimization'] = self.performance_core.get_optimization_report()
        
        return report


# ============================================================================
# MAIN: Example usage and testing
# ============================================================================

if __name__ == "__main__":
    print("🌟 Belle II Billion-Capable Framework - Production Test")
    print("=" * 70)
    
    # Initialize framework with generous memory budget for billion-row processing
    framework = BillionCapableFramework(memory_budget_gb=32.0, max_workers=16)
    
    # Test data paths (update with actual paths)
    test_paths = [
        "/gpfs//gpfs/group/belle2/users2022/kyldem/photoneff_updated/parquet_storage/try5"
    ]
    
    try:
        # Load billion-row dataset
        framework.load_data(test_paths, process='auto')
        
        # Run comprehensive benchmarks
        results = framework.benchmark_suite(output_file='billion_row_benchmark.json')
        
        # Example billion-row analysis workflow
        print("\n🔬 Billion-Row Analysis Workflow:")
        
        # 1. Ultra-fast histogram on billions of rows
        hist_counts, hist_edges = framework.hist('M_bc', bins=1000)
        print(f"   ✅ Computed 1000-bin histogram on {framework.blazing_core._estimated_total_rows:,} rows")
        
        # 2. Complex query on billions of rows
        filtered = framework.query("M_bc > 5.27 & M_bc < 5.29 & delta_E.abs() < 0.1")
        print(f"   ✅ Filtered to candidates in signal region")
        
        # 3. Create delta columns
        enhanced = filtered.createDeltaColumns("M_bc", ["M_D", "M_K", "M_pi"])
        print(f"   ✅ Created 3 delta columns")
        
        # 4. GroupBy on billions of rows
        grouped = enhanced.groupby(['evt', 'run']).agg({
            'M_bc': ['mean', 'std', 'count'],
            'delta_M_bc_M_D': ['mean', 'std']
        })
        print(f"   ✅ Grouped by event with 5 aggregations")
        
        # 5. Select best candidates
        best_candidates = enhanced.oneCandOnly(['evt', 'run'], 'chiProb', ascending=False)
        print(f"   ✅ Selected best candidate per event")
        
        # Performance report
        print("\n📈 Performance Report:")
        report = framework.get_performance_report()
        
        if report['data']['estimated_rows'] > 1_000_000_000:
            print(f"   🌟 Successfully processed {report['data']['estimated_rows']/1e9:.1f} BILLION rows!")
        
        print(f"   💾 Memory usage: Optimized for {framework.memory_budget_gb}GB budget")
        print(f"   ⚡ C++ acceleration: {'Enabled' if 'cpp_accelerator' in str(framework.blazing_core.streaming_histogram.__dict__) else 'Not available'}")
        
        print("\n✅ Billion-row production test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()