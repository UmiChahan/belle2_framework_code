"""
Enhanced Belle II Framework - Optimized Integration
==================================================

Key improvements:
1. Integrated C++ acceleration into streaming operations
2. Eliminated query conversion redundancy
3. Optimized GroupBy with true streaming
4. Simplified API layers
5. Production-ready UnifiedBelle2Framework
"""
import ast
import gc
import re
import time
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache, cached_property
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Iterator, 
    Callable, Set, TYPE_CHECKING
)
import threading
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
# First, let's integrate the C++ module properly
from unified_framework import BlazingCore, StreamingHistogram, GroupByHandler, OptimalPandasToPolarsConverter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cpp_histogram_integrator import cpp_histogram_integrator

# Import the existing implementation (assuming it's available)
# We'll selectively override and enhance components

# ============================================================================
# ENHANCEMENT 1: Integrate C++ Acceleration into StreamingHistogram
# ============================================================================

class EnhancedStreamingHistogram:
    """Enhanced histogram computation with proper C++ acceleration."""
    
    def __init__(self):
        # Initialize C++ accelerator
        self.cpp_accelerator = None
        self._cpp_available = False
        try:
            from cpp_histogram_integrator import cpp_histogram_integrator
            if cpp_histogram_integrator.is_available():
                self.cpp_accelerator = cpp_histogram_integrator
                self._cpp_available = True
                print("✅ C++ histogram acceleration enabled")
        except ImportError:
            print("⚠️ C++ acceleration not available, using pure Polars")
    
    def compute_blazing_fast(self, lazy_frames: List[pl.LazyFrame], column: str, 
                           bins: int = 50, range: Optional[Tuple[float, float]] = None,
                           density: bool = False, weights: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blazing fast histogram computation with automatic C++ acceleration.
        
        Intelligently switches between pure Polars and C++ acceleration based on:
        - Dataset size
        - Available memory
        - C++ module availability
        """
        start_time = time.time()
        
        # Get range using efficient sampling if not provided
        if range is None:
            range = self._estimate_range_fast(lazy_frames, column)
        
        min_val, max_val = range
        bin_width = (max_val - min_val) / bins
        
        # Initialize histogram
        hist_counts = np.zeros(bins, dtype=np.int64)
        total_processed = 0
        
        # Determine processing strategy
        use_cpp = self._should_use_cpp_acceleration(lazy_frames, column)
        
        # Process frames with appropriate method
        for lf in lazy_frames:
            if use_cpp:
                chunk_counts, chunk_processed = self._process_frame_with_cpp(
                    lf, column, bins, min_val, bin_width
                )
            else:
                chunk_counts, chunk_processed = self._process_frame_streaming(
                    lf, column, bins, min_val, bin_width
                )
            
            hist_counts += chunk_counts
            total_processed += chunk_processed
        
        # Convert to density if requested
        if density and total_processed > 0:
            hist_counts = hist_counts.astype(float) / (total_processed * bin_width)
        
        # Create bin edges
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        elapsed = time.time() - start_time
        throughput = total_processed / elapsed if elapsed > 0 else 0
        
        acceleration_method = "C++ accelerated" if use_cpp else "Pure Polars"
        print(f"⚡ Histogram ({acceleration_method}): {total_processed:,} rows in {elapsed:.2f}s ({throughput/1e6:.1f}M rows/s)")
        
        return hist_counts, bin_edges
    
    def _should_use_cpp_acceleration(self, lazy_frames: List[pl.LazyFrame], column: str) -> bool:
        """Intelligent decision on whether to use C++ acceleration."""
        if not self._cpp_available:
            return False
        
        # Estimate total dataset size
        try:
            total_rows = sum(
                lf.select(pl.count()).collect(streaming=True)[0, 0] 
                for lf in lazy_frames[:3]
            ) * len(lazy_frames) // 3
            
            # Use C++ for datasets larger than 10M rows
            return total_rows > 10_000_000
        except:
            return False
    
    def _process_frame_with_cpp(self, lf: pl.LazyFrame, column: str, bins: int, 
                               min_val: float, bin_width: float) -> Tuple[np.ndarray, int]:
        """Process frame using C++ acceleration with intelligent chunking."""
        try:
            # Estimate frame size
            frame_size = lf.select(pl.count()).collect(streaming=True)[0, 0]
            
            if frame_size <= 5_000_000:
                # Small enough to process in one go
                data = lf.select(pl.col(column)).collect(streaming=True)
                values = data[column].to_numpy()
                
                # Filter nulls
                values = values[~np.isnan(values)]
                
                # Use C++ accelerator
                hist_counts = self.cpp_accelerator.compute_histogram_parallel(
                    values, min_val, min_val + bins * bin_width, bins
                )
                
                return hist_counts, len(values)
            else:
                # Process in chunks for memory efficiency
                return self._process_large_frame_with_cpp_chunks(
                    lf, column, bins, min_val, bin_width, chunk_size=5_000_000
                )
                
        except Exception as e:
            warnings.warn(f"C++ acceleration failed: {e}, falling back to Polars")
            return self._process_frame_streaming(lf, column, bins, min_val, bin_width)
    
    def _process_large_frame_with_cpp_chunks(self, lf: pl.LazyFrame, column: str, bins: int,
                                           min_val: float, bin_width: float, 
                                           chunk_size: int = 5_000_000) -> Tuple[np.ndarray, int]:
        """Process large frames in chunks with C++ acceleration."""
        hist_counts = np.zeros(bins, dtype=np.int64)
        total_count = 0
        offset = 0
        
        while True:
            # Get chunk
            chunk = (
                lf
                .filter(pl.col(column).is_not_null())
                .slice(offset, chunk_size)
                .select(pl.col(column))
                .collect(streaming=True)
            )
            
            if len(chunk) == 0:
                break
            
            # Process chunk with C++
            chunk_data = chunk[column].to_numpy()
            chunk_hist = self.cpp_accelerator.compute_histogram_parallel(
                chunk_data, min_val, min_val + bins * bin_width, bins
            )
            
            hist_counts += chunk_hist
            total_count += len(chunk_data)
            offset += chunk_size
            
            if len(chunk) < chunk_size:
                break
        
        return hist_counts, total_count
    

# ============================================================================
# ENHANCEMENT 2: Unified Query Converter (Eliminate Redundancy)
# ============================================================================

class UnifiedQueryConverter:
    """Single, optimized pandas-to-polars query converter."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._cache = {}
            self.converter = OptimalPandasToPolarsConverter()
    
    def convert(self, query_str: str) -> List[pl.Expr]:
        """Convert pandas query to Polars expressions with caching."""
        if query_str in self._cache:
            return self._cache[query_str]
        
        try:
            conditions = self.converter.convert_to_filter_conditions(query_str)
            self._cache[query_str] = conditions
            return conditions
        except Exception as e:
            # Simple fallback patterns
            expr = self._simple_pattern_conversion(query_str)
            if expr:
                self._cache[query_str] = [expr]
                return [expr]
            raise ValueError(f"Query conversion failed: {e}")
    
    def _simple_pattern_conversion(self, query_str: str) -> Optional[pl.Expr]:
        """Fallback simple pattern matching."""
        patterns = [
            (r'(\w+)\s*>\s*([\d.]+)', lambda m: pl.col(m.group(1)) > float(m.group(2))),
            (r'(\w+)\s*<\s*([\d.]+)', lambda m: pl.col(m.group(1)) < float(m.group(2))),
            (r'(\w+)\s*>=\s*([\d.]+)', lambda m: pl.col(m.group(1)) >= float(m.group(2))),
            (r'(\w+)\s*<=\s*([\d.]+)', lambda m: pl.col(m.group(1)) <= float(m.group(2))),
            (r'(\w+)\s*==\s*([\d.]+)', lambda m: pl.col(m.group(1)) == float(m.group(2))),
        ]
        
        for pattern, builder in patterns:
            match = re.match(pattern, query_str.strip())
            if match:
                return builder(match)
        return None


# ============================================================================
# ENHANCEMENT 3: Optimized GroupBy with True Streaming
# ============================================================================

class OptimizedGroupByHandler(GroupByHandler):
    """Truly optimized streaming GroupBy that minimizes materialization."""
    
    def agg(self, agg_dict: Dict[str, Union[str, List[str]]]) -> pd.DataFrame:
        """TRUE streaming groupby aggregation with single-pass optimization."""
        
        start_time = time.time()
        
        # Build Polars aggregation expressions
        agg_exprs = self._build_agg_expressions(agg_dict)
        
        # CRITICAL OPTIMIZATION: Single-pass streaming aggregation
        try:
            # Step 1: Create a single concatenated lazy frame
            if len(self.blazing_core.lazy_frames) == 1:
                combined_lazy = self.blazing_core.lazy_frames[0]
            else:
                # Use union instead of concat for better streaming
                combined_lazy = self.blazing_core.lazy_frames[0]
                for lf in self.blazing_core.lazy_frames[1:]:
                    combined_lazy = combined_lazy.union(lf)
            
            # Step 2: Single streaming aggregation
            result_df = (
                combined_lazy
                .group_by(self.group_cols)
                .agg(agg_exprs)
                .collect(streaming=True)
                .to_pandas()
            )
            
            elapsed = time.time() - start_time
            
            # Performance reporting
            rows_estimate = self.blazing_core._estimated_total_rows
            throughput = rows_estimate / elapsed if elapsed > 0 else 0
            print(f"⚡ Optimized GroupBy: {rows_estimate:,} rows → {len(result_df):,} groups "
                  f"in {elapsed:.2f}s ({throughput/1e6:.1f}M rows/s)")
            
            return result_df
            
        except Exception as e:
            warnings.warn(f"Optimized GroupBy failed: {e}, using fallback")
            return super().agg(agg_dict)
    
    def _build_agg_expressions(self, agg_dict: Dict[str, Union[str, List[str]]]) -> List[pl.Expr]:
        """Build efficient aggregation expressions."""
        agg_exprs = []
        
        for col, operations in agg_dict.items():
            if isinstance(operations, str):
                operations = [operations]
            
            for op in operations:
                if op == 'sum':
                    agg_exprs.append(pl.col(col).sum().alias(f'{col}_sum'))
                elif op == 'mean':
                    agg_exprs.append(pl.col(col).mean().alias(f'{col}_mean'))
                elif op == 'count':
                    agg_exprs.append(pl.col(col).count().alias(f'{col}_count'))
                elif op == 'min':
                    agg_exprs.append(pl.col(col).min().alias(f'{col}_min'))
                elif op == 'max':
                    agg_exprs.append(pl.col(col).max().alias(f'{col}_max'))
                elif op == 'std':
                    agg_exprs.append(pl.col(col).std().alias(f'{col}_std'))
                elif op == 'var':
                    agg_exprs.append(pl.col(col).var().alias(f'{col}_var'))
                elif op == 'median':
                    agg_exprs.append(pl.col(col).median().alias(f'{col}_median'))
                
        return agg_exprs


# ============================================================================
# ENHANCEMENT 4: Simplified API Layer
# ============================================================================

from optimized_cpp_integration import OptimizedStreamingHistogram

class EnhancedBlazingCore:
    """Enhanced BlazingCore with properly integrated C++ acceleration."""
    
    def __init__(self, lazy_frames: List[pl.LazyFrame]):
        self.lazy_frames = lazy_frames
        # Use optimized streaming histogram
        self.streaming_histogram = OptimizedStreamingHistogram()
        self._estimated_total_rows = self._estimate_dataset_size()
    
    def _estimate_dataset_size(self) -> int:
        """Quick dataset size estimation."""
        if not self.lazy_frames:
            return 0
        
        sample_size = 0
        for lf in self.lazy_frames[:3]:
            try:
                count = lf.select(pl.len()).collect(engine="streaming")[0, 0]
                sample_size += count
            except Exception:
                continue
        
        if sample_size > 0:
            estimated = (sample_size // 3) * len(self.lazy_frames)
            return estimated
        
        return 0
    
    def hist(self, column: str, bins: int = 50, range: Optional[Tuple[float, float]] = None,
             density: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-fast histogram with automatic C++ acceleration."""
        return self.streaming_histogram.compute_blazing_fast(
            self.lazy_frames, column, bins, range, density
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'estimated_rows': self._estimated_total_rows,
            'num_frames': len(self.lazy_frames),
            'cpp_acceleration': self.streaming_histogram._cpp_available
        }
    
    def hist(self, column: str, bins: int = 50, range: Optional[Tuple[float, float]] = None,
             density: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-fast histogram with automatic C++ acceleration."""
        return self.streaming_histogram.compute_blazing_fast(
            self.lazy_frames, column, bins, range, density
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'estimated_rows': self._estimated_total_rows,
            'num_frames': len(self.lazy_frames),
            'cpp_acceleration': self.streaming_histogram._cpp_available
        }


class StreamingMergeEngine:
    """Cutting-edge merge implementation for billion-row datasets."""
    
    def merge(self, left_frames: List[pl.LazyFrame], right_frames: List[pl.LazyFrame],
              on: Union[str, List[str]], how: str = 'inner', 
              strategy: str = 'auto') -> List[pl.LazyFrame]:
        """
        Core merge logic with adaptive strategy selection.
        
        Strategies:
        - broadcast: Small right side (<100MB) broadcast to all left partitions
        - sort_merge: Both sides sorted, streaming merge
        - hash_partition: Partition both sides by join keys
        - nested_loop: Fallback for complex conditions
        """
        
        # 1. ANALYZE PHASE
        left_stats = self._analyze_join_side(left_frames, on)
        right_stats = self._analyze_join_side(right_frames, on)
        
        # 2. STRATEGY SELECTION
        if strategy == 'auto':
            strategy = self._select_optimal_strategy(left_stats, right_stats, how)
        
        # 3. EXECUTION
        if strategy == 'broadcast':
            return self._broadcast_join(left_frames, right_frames, on, how)
        elif strategy == 'sort_merge':
            return self._sort_merge_join(left_frames, right_frames, on, how)
        elif strategy == 'hash_partition':
            return self._hash_partition_join(left_frames, right_frames, on, how)
        else:
            return self._nested_loop_join(left_frames, right_frames, on, how)
    
    def _analyze_join_side(self, frames, keys):
        """Streaming analysis without materialization."""
        return {
            'estimated_rows': self._estimate_total_rows(frames),
            'estimated_size_gb': self._estimate_memory_size(frames),
            'key_cardinality': self._estimate_key_cardinality(frames, keys),
            'is_sorted': self._check_if_sorted(frames, keys),
            'key_distribution': self._sample_key_distribution(frames, keys)
        }
    
    def _select_optimal_strategy(self, left_stats, right_stats, how):
        """Intelligent strategy selection based on statistics."""
        
        # Broadcast if right side is small
        if right_stats['estimated_size_gb'] < 0.1:  # 100MB threshold
            return 'broadcast'
        
        # Sort-merge if both sides sorted or nearly sorted
        if left_stats['is_sorted'] and right_stats['is_sorted']:
            return 'sort_merge'
        
        # Hash partition for large-scale joins
        if left_stats['estimated_rows'] > 100_000_000:
            return 'hash_partition'
        
        # Adaptive decision based on key distribution
        if self._is_skewed_distribution(left_stats['key_distribution']):
            return 'hash_partition'  # Better for skewed data
        
        return 'sort_merge'  # Default
    
    def _broadcast_join(self, left_frames, right_frames, on, how):
        """Small right side materialized and broadcast."""
        # Materialize small side once
        right_data = self._materialize_small_side(right_frames, on)
        
        # Stream through left side
        result_frames = []
        for lf in left_frames:
            joined = lf.join(right_data, on=on, how=how)
            result_frames.append(joined)
        
        return result_frames
    
    def _sort_merge_join(self, left_frames, right_frames, on, how):
        """Classic sort-merge with streaming."""
        # Pre-sort if needed (lazy operation)
        left_sorted = [lf.sort(on) for lf in left_frames]
        right_sorted = [rf.sort(on) for rf in right_frames]
        
        # Streaming merge using Polars' optimized engine
        return self._streaming_sorted_merge(left_sorted, right_sorted, on, how)
    
    def _hash_partition_join(self, left_frames, right_frames, on, how):
        """Distributed hash join with co-partitioning."""
        # Determine optimal partition count
        n_partitions = self._calculate_optimal_partitions(left_frames, right_frames)
        
        # Hash partition both sides
        left_partitioned = self._hash_partition_frames(left_frames, on, n_partitions)
        right_partitioned = self._hash_partition_frames(right_frames, on, n_partitions)
        
        # Join corresponding partitions
        result_frames = []
        for i in range(n_partitions):
            partition_result = left_partitioned[i].join(
                right_partitioned[i], on=on, how=how
            )
            result_frames.append(partition_result)
        
        return result_frames
    
    def _adaptive_memory_management(self, operation):
        """Monitor and adapt during join execution."""
        memory_pressure = self._get_memory_pressure()
        
        if memory_pressure > 0.8:
            # Switch to more memory-efficient strategy
            self._spill_to_disk()
            self._reduce_partition_size()
        elif memory_pressure < 0.3:
            # Can be more aggressive
            self._increase_buffer_size()
# ============================================================================
# ENHANCEMENT 5: Complete Production-Ready Framework
# ============================================================================

class UnifiedBelle2Framework:
    """
    Production-ready unified Belle II framework with all optimizations.
    
    Features:
    - C++ acceleration integrated into streaming operations
    - Optimized GroupBy with true single-pass streaming
    - Unified query conversion
    - Simplified API layers
    - Comprehensive benchmarking
    """
    
    def __init__(self, memory_budget_gb: float = 8.0, max_workers: int = 4):
        """Initialize enhanced framework."""
        self.memory_budget_gb = memory_budget_gb
        self.max_workers = max_workers
        
        # Core components
        self.data_loader = DataLoader(max_workers)
        self.performance_core = PerformanceCore(
            max_cache_size_gb=memory_budget_gb * 0.25,
            memory_warning_threshold_gb=memory_budget_gb * 0.8
        )
        
        # Framework state
        self.blazing_core = None
        self.smart_evaluator = None
        self.current_process = None
        self._lazy_frames_by_process = {}
        
        print(f"🚀 Enhanced Belle II Framework initialized")
        print(f"💾 Memory budget: {memory_budget_gb:.1f} GB")
        print(f"⚙️ Max workers: {max_workers}")
        
        # Check for C++ acceleration
        try:
            from cpp_histogram_integrator import cpp_histogram_integrator
            if cpp_histogram_integrator.is_available():
                print("⚡ C++ acceleration: ENABLED")
        except:
            print("⚠️ C++ acceleration: NOT AVAILABLE")
    
    def load_data(self, paths: Union[str, List[str]], 
                  process: str = 'auto', strategy: str = 'auto') -> 'UnifiedBelle2Framework':
        """Load data with enhanced optimization."""
        print(f"\n📂 Loading data from: {paths}")
        
        # Load data
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
        
        # Initialize enhanced components
        self.blazing_core = EnhancedBlazingCore(lazy_frames)
        self.smart_evaluator = SmartEvaluator(self.memory_budget_gb)
        
        # Report loading stats
        total_frames = len(lazy_frames)
        estimated_rows = self.blazing_core._estimated_total_rows
        print(f"✅ Loaded {total_frames} frames, ~{estimated_rows:,} rows for process '{process}'")
        
        return self
    
    def merge(self, other: 'UnifiedBelle2Framework', on: Union[str, List[str]], 
              how: str = 'inner', validate: str = None) -> 'UnifiedBelle2Framework':
        """
        Pandas-compatible merge with billion-row optimization.
        
        Features:
        - Automatic strategy selection
        - Memory-safe execution
        - Physics-aware validation
        """
        merge_engine = StreamingMergeEngine()
        
        # Execute streaming merge
        merged_frames = merge_engine.merge(
            self.blazing_core.lazy_frames,
            other.blazing_core.lazy_frames,
            on=on, how=how
        )
        
        # Physics validation if requested
        if validate == 'one_to_one':
            merged_frames = self._validate_one_to_one(merged_frames, on)
        elif validate == 'many_to_one':
            merged_frames = self._validate_many_to_one(merged_frames, on)
        
        # Return new framework instance
        new_framework = self._create_new_instance()
        new_framework.blazing_core = BlazingCore(merged_frames)
        return new_framework
    
    def hist(self, column: str, bins: int = 50, range: Optional[Tuple[float, float]] = None,
             density: bool = False, ax: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-fast histogram with C++ acceleration where beneficial."""
        if not self.blazing_core:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Execute with performance tracking
        counts, edges = self.performance_core.execute_with_optimization(
            'histogram',
            self.blazing_core.hist,
            self.blazing_core._estimated_total_rows,
            column, bins, range, density, **kwargs
        )
        
        # Optional plotting
        if ax is not None:
            ax.stairs(counts, edges, label=kwargs.get('label', column))
            ax.set_xlabel(column)
            ax.set_ylabel('Count' if not density else 'Density')
        
        return counts, edges
    
    def query(self, expr: str) -> 'UnifiedBelle2Framework':
        """Query with unified converter and streaming optimization."""
        if not self.blazing_core:
            raise ValueError("No data loaded. Call load_data() first.")
        
        filtered_core = self.performance_core.execute_with_optimization(
            'query',
            self.blazing_core.query,
            self.blazing_core._estimated_total_rows,
            expr
        )
        
        # Create new framework instance with filtered data
        new_framework = UnifiedBelle2Framework(self.memory_budget_gb, self.max_workers)
        new_framework.blazing_core = filtered_core
        new_framework.smart_evaluator = self.smart_evaluator
        new_framework.current_process = self.current_process
        new_framework._lazy_frames_by_process = {self.current_process: filtered_core.lazy_frames}
        
        return new_framework
    
    def groupby(self, by: Union[str, List[str]]) -> OptimizedGroupByHandler:
        """Optimized streaming GroupBy."""
        if not self.blazing_core:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.blazing_core.groupby(by)
    
    def agg(self, operation: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Streaming aggregation operations."""
        if not self.blazing_core:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.performance_core.execute_with_optimization(
            f'agg_{operation}',
            lambda: self.blazing_core.agg(operation, columns),
            self.blazing_core._estimated_total_rows
        )
    
    def sum(self, numeric_only: bool = True) -> Dict[str, Any]:
        """Pandas-compatible sum."""
        return self.agg('sum')
    
    def mean(self, numeric_only: bool = True) -> Dict[str, Any]:
        """Pandas-compatible mean."""
        return self.agg('mean')
    
    def std(self, numeric_only: bool = True) -> Dict[str, Any]:
        """Pandas-compatible standard deviation."""
        return self.agg('std')
    
    def count(self) -> int:
        """Total row count."""
        result = self.agg('count')
        return result.get('total_count', 0)
    
    def createDeltaColumns(self, base_col: str, target_cols: List[str]) -> 'UnifiedBelle2Framework':
        """Belle II specific: create delta columns."""
        if not self.blazing_core:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Build delta expressions
        delta_exprs = []
        for target_col in target_cols:
            delta_name = f"delta_{base_col}_{target_col}"
            delta_expr = (pl.col(target_col) - pl.col(base_col)).alias(delta_name)
            delta_exprs.append(delta_expr)
        
        # Apply to all frames
        enhanced_frames = []
        for lf in self.blazing_core.lazy_frames:
            enhanced_lf = lf.with_columns(delta_exprs)
            enhanced_frames.append(enhanced_lf)
        
        # Create new framework with enhanced frames
        new_framework = UnifiedBelle2Framework(self.memory_budget_gb, self.max_workers)
        new_framework.blazing_core = EnhancedBlazingCore(enhanced_frames)
        new_framework.smart_evaluator = self.smart_evaluator
        new_framework.current_process = self.current_process
        new_framework._lazy_frames_by_process = {self.current_process: enhanced_frames}
        
        return new_framework
    
    def oneCandOnly(self, group_cols: Union[str, List[str]], sort_col: str, 
                    ascending: bool = False, physics_mode: Optional[str] = None) -> 'UnifiedBelle2Framework':
        """
        Belle II critical function: Select one best candidate per event.
        
        This is essential for physics analysis to avoid double-counting when multiple
        candidate particles are reconstructed in the same event.
        
        Args:
            group_cols: Column(s) to group by (typically event identifiers like 'evt', 'run')
            sort_col: Column to sort by for selection (e.g., 'chiProb', 'isSignal')
            ascending: Sort order (False = descending = higher values selected)
            physics_mode: Optional physics-specific selection strategy
                         ('B_meson', 'D_meson', 'tau', etc.)
        
        Returns:
            New framework with one candidate per group
            
        Example:
            # Select best B meson candidate per event based on chi-squared probability
            best_B = framework.oneCandOnly(['evt', 'run'], 'chiProb', ascending=False)
        """
        if not self.blazing_core:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Ensure group_cols is a list
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        
        # Apply physics-specific selection strategies if specified
        if physics_mode:
            if physics_mode == 'B_meson':
                # B mesons: typically use vertex chi2 probability
                if sort_col is None:
                    sort_col = 'chiProb'
                    ascending = False
            elif physics_mode == 'D_meson':
                # D mesons: often use mass difference from nominal
                if sort_col is None:
                    sort_col = 'M'
                    # Would need special handling for mass window
            elif physics_mode == 'tau':
                # Tau leptons: might use decay mode quality
                if sort_col is None:
                    sort_col = 'decayModeScore'
                    ascending = False
        
        # Streaming candidate selection
        selected_frames = []
        for lf in self.blazing_core.lazy_frames:
            try:
                # CRITICAL: Maintain order for reproducibility
                selected_lf = (
                    lf
                    .with_row_count("__orig_row_id")  # Track original row for debugging
                    .sort(sort_col, descending=not ascending)
                    .group_by(group_cols, maintain_order=True)
                    .first()
                    .drop("__orig_row_id")  # Clean up temporary column
                )
                selected_frames.append(selected_lf)
            except Exception as e:
                warnings.warn(f"oneCandOnly failed for a frame: {e}, keeping original")
                selected_frames.append(lf)
        
        # Create new framework with selected candidates
        new_framework = UnifiedBelle2Framework(self.memory_budget_gb, self.max_workers)
        new_framework.blazing_core = EnhancedBlazingCore(selected_frames)
        new_framework.smart_evaluator = self.smart_evaluator
        new_framework.current_process = self.current_process
        new_framework._lazy_frames_by_process = {self.current_process: selected_frames}
        
        # Update row count estimate (typically reduces by factor of 2-10)
        if hasattr(self.blazing_core, '_estimated_total_rows'):
            # Conservative estimate: assume 3 candidates per event on average
            new_framework.blazing_core._estimated_total_rows = self.blazing_core._estimated_total_rows // 3
        
        print(f"✅ oneCandOnly: Selected best candidates grouped by {group_cols}")
        
        return new_framework
    
    def to_pandas(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Convert to pandas DataFrame with memory safety."""
        if not self.blazing_core:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if max_rows is None and self.blazing_core._estimated_total_rows > 10_000_000:
            warnings.warn(
                f"Dataset has ~{self.blazing_core._estimated_total_rows:,} rows. "
                f"Consider using max_rows parameter to limit memory usage."
            )
        
        combined_frames = []
        total_rows = 0
        
        for lf in self.blazing_core.lazy_frames:
            if max_rows and total_rows >= max_rows:
                break
            
            try:
                if max_rows:
                    remaining = max_rows - total_rows
                    df = lf.head(remaining).collect(streaming=True).to_pandas()
                else:
                    df = lf.collect(streaming=True).to_pandas()
                
                combined_frames.append(df)
                total_rows += len(df)
                
            except Exception as e:
                warnings.warn(f"Frame materialization failed: {e}")
                continue
        
        if combined_frames:
            return pd.concat(combined_frames, ignore_index=True)
        return pd.DataFrame()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Comprehensive performance report."""
        report = {
            'framework': {
                'version': '2.0-optimized',
                'process': self.current_process,
                'memory_budget_gb': self.memory_budget_gb,
                'processes_loaded': list(self._lazy_frames_by_process.keys())
            }
        }
        
        if self.blazing_core:
            report['data'] = {
                'estimated_rows': self.blazing_core._estimated_total_rows,
                'num_frames': len(self.blazing_core.lazy_frames)
            }
            report['performance'] = self.blazing_core.get_performance_stats()
        
        if self.performance_core:
            report['optimization'] = self.performance_core.get_optimization_report()
        
        return report
    
    def benchmark_suite(self, column: Optional[str] = None, 
                       output_file: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive benchmark suite with detailed reporting."""
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
        
        print(f"\n🔥 Running comprehensive benchmark suite")
        print(f"📊 Test column: {column}")
        print(f"📈 Estimated dataset size: {self.blazing_core._estimated_total_rows:,} rows")
        print("=" * 60)
        
        results = {
            'metadata': {
                'column': column,
                'estimated_rows': self.blazing_core._estimated_total_rows,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'benchmarks': {}
        }
        
        # Test suite
        tests = [
            ('histogram_50bins', lambda: self.hist(column, bins=50)),
            ('histogram_100bins', lambda: self.hist(column, bins=100)),
            ('sum', lambda: self.sum()),
            ('mean', lambda: self.mean()),
            ('std', lambda: self.std()),
            ('count', lambda: self.count()),
            ('query_simple', lambda: self.query(f"{column} > 0").count()),
            ('query_complex', lambda: self.query(f"{column} > 0 & {column} < 1").count()),
            ('groupby_single', lambda: self.groupby(column).agg({column: 'count'})),
        ]
        
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
            results['summary'] = {
                'avg_throughput_M_rows_s': avg_throughput,
                'successful_tests': len(successful),
                'failed_tests': len(results['benchmarks']) - len(successful)
            }
            
            print(f"\n📊 Summary:")
            print(f"   Average throughput: {avg_throughput:.1f}M rows/s")
            print(f"   Success rate: {len(successful)}/{len(results['benchmarks'])}")
        
        # Save results
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results


# ============================================================================
# MAIN: Production-Ready Testing
# ============================================================================

if __name__ == "__main__":
    print("🚀 Belle II Unified Framework - Production Test")
    print("=" * 60)
    
    # Initialize framework
    framework = UnifiedBelle2Framework(memory_budget_gb=16.0, max_workers=8)
    
    # Test data loading (example paths)
    test_paths = [
        "/path/to/belle2/data",  # Update with actual paths
        "/path/to/processed/files"
    ]
    
    try:
        # Load data
        framework.load_data(test_paths, process='auto')
        
        # Run comprehensive benchmarks
        results = framework.benchmark_suite(output_file='benchmark_results.json')
        
        # Performance report
        print("\n📈 Performance Report:")
        report = framework.get_performance_report()
        
        import json
        print(json.dumps(report, indent=2))
        
        # Example analysis workflow
        print("\n🔬 Example Analysis Workflow:")
        
        # 1. Filter data
        filtered = framework.query("M_bc > 5.27 & M_bc < 5.29")
        print(f"   Filtered to {filtered.count():,} candidates")
        
        # 2. Create delta columns
        enhanced = filtered.createDeltaColumns("M_bc", ["M_D", "M_K"])
        
        # 3. GroupBy analysis
        grouped = enhanced.groupby(['decay_mode']).agg({
            'M_bc': ['mean', 'std', 'count'],
            'delta_M_bc_M_D': ['mean', 'std']
        })
        print(f"   Grouped into {len(grouped)} decay modes")
        
        # 4. Histogram
        counts, edges = enhanced.hist('M_bc', bins=100)
        print(f"   Histogram computed with {len(counts)} bins")
        
        print("\n✅ Production test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()