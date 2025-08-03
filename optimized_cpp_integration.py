"""
Fixed Optimized C++ Integration with Proper Resource Management
==============================================================

This version addresses:
1. Polars API deprecations
2. OpenMP thread resource exhaustion
3. HPC environment compatibility
"""

import os
import numpy as np
import polars as pl
import time
import warnings
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def configure_openmp_for_hpc():
    """
    Configure OpenMP settings for HPC environments to prevent thread exhaustion.
    
    This function sets environment variables that control OpenMP behavior
    BEFORE any OpenMP-enabled libraries are loaded.
    """
    # Limit OpenMP threads to prevent resource exhaustion
    # This should be called at the very beginning of your script
    if 'OMP_NUM_THREADS' not in os.environ:
        # Use a conservative number of threads
        cpu_count = psutil.cpu_count(logical=False) or 4
        # Limit to 8 threads max for OpenMP to leave room for Python threads
        omp_threads = min(cpu_count, 8)
        os.environ['OMP_NUM_THREADS'] = str(omp_threads)
        print(f"🔧 Set OMP_NUM_THREADS={omp_threads}")
    
    # Disable dynamic thread adjustment to prevent unexpected thread creation
    os.environ['OMP_DYNAMIC'] = 'FALSE'
    
    # Set thread stack size to prevent stack overflow on large computations
    os.environ['OMP_STACKSIZE'] = '64M'
    
    # Use static scheduling for predictable behavior
    os.environ['OMP_SCHEDULE'] = 'static'


# Call this BEFORE importing the C++ integrator
configure_openmp_for_hpc()


class OptimizedStreamingHistogram:
    """
    Production-ready histogram computation with proper resource management.
    
    Key improvements:
    1. Updated Polars API usage
    2. Controlled parallelism to prevent thread exhaustion
    3. HPC-aware resource management
    """
    
    def __init__(self):
        # Import the integrator AFTER configuring OpenMP
        try:
            from cpp_histogram_integrator import cpp_histogram_integrator
            self.cpp_integrator = cpp_histogram_integrator
            self._cpp_available = cpp_histogram_integrator.is_available()
            
            if self._cpp_available:
                print("✅ C++ histogram acceleration: ACTIVE")
                self._analyze_system_capabilities()
            else:
                print("⚠️ C++ acceleration not available, using optimized Polars")
                
        except ImportError:
            print("❌ C++ integrator not found")
            self._cpp_available = False
            self.cpp_integrator = None
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self._cache = {}
        self._lock = threading.Lock()
        
        # Configure thread limits for HPC
        self._configure_thread_limits()
    
    def _configure_thread_limits(self):
        """Configure thread limits based on system resources."""
        # Get current limits
        try:
            import resource
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
            print(f"📊 Process limits: soft={soft_limit}, hard={hard_limit}")
            
            # Calculate safe thread counts
            # Reserve threads: main thread + OpenMP threads + safety margin
            omp_threads = int(os.environ.get('OMP_NUM_THREADS', 8))
            reserved_threads = 1 + omp_threads + 10  # safety margin
            
            # Available for Python thread pool
            available_for_pool = min(soft_limit - reserved_threads, 16)
            self.max_python_threads = max(1, available_for_pool)
            
            print(f"🔧 Max Python threads: {self.max_python_threads}")
            
        except Exception as e:
            print(f"⚠️ Could not determine thread limits: {e}")
            self.max_python_threads = 4  # Conservative default
    
    def _analyze_system_capabilities(self):
        """Analyze system for optimal configuration."""
        # CPU cache sizes (typical modern CPU)
        self.l1_cache_size = 32 * 1024  # 32 KB per core
        self.l2_cache_size = 256 * 1024  # 256 KB per core
        self.l3_cache_size = 8 * 1024 * 1024  # 8 MB shared
        
        # Memory bandwidth estimation
        memory_info = psutil.virtual_memory()
        self.available_memory_gb = memory_info.available / (1024**3)
        
        # Optimal chunk size calculation based on cache hierarchy
        self.optimal_chunk_elements = self._calculate_optimal_chunk_size()
        
        print(f"📊 System analysis:")
        print(f"   Available memory: {self.available_memory_gb:.1f} GB")
        print(f"   Optimal chunk size: {self.optimal_chunk_elements:,} elements")
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on cache hierarchy."""
        histogram_size = 1000 * 8  # Assume max 1000 bins
        overhead = 0.1
        
        # Available L3 for data (leave 25% for other processes)
        available_l3 = self.l3_cache_size * 0.75
        data_space = available_l3 - histogram_size
        
        # Elements that fit in cache
        elements_in_cache = int(data_space / (8 * (1 + overhead)))
        
        # Round to nearest million for predictable performance
        return max(1_000_000, min(10_000_000, 
                                  (elements_in_cache // 1_000_000) * 1_000_000))
    
    def _analyze_dataset(self, lazy_frames: List[pl.LazyFrame], 
                        column: str) -> Dict[str, Any]:
        """Analyze dataset characteristics with updated Polars API."""
        # Sample frames for statistics
        sample_frames = min(3, len(lazy_frames))
        total_rows = 0
        null_ratio = 0.0
        
        for lf in lazy_frames[:sample_frames]:
            try:
                # Updated API: use pl.len() instead of pl.count()
                stats = lf.select([
                    pl.len().alias('count'),
                    pl.col(column).null_count().alias('nulls')
                ]).collect(engine="streaming")  # Updated API
                
                if len(stats) > 0:
                    count = stats[0, 0]
                    nulls = stats[0, 1]
                    total_rows += count
                    null_ratio += nulls / count if count > 0 else 0
                    
            except Exception:
                continue
        
        # Extrapolate to full dataset
        if sample_frames > 0:
            estimated_rows = (total_rows // sample_frames) * len(lazy_frames)
            avg_null_ratio = null_ratio / sample_frames
        else:
            estimated_rows = 0
            avg_null_ratio = 0.0
        
        return {
            'estimated_rows': estimated_rows,
            'null_ratio': avg_null_ratio,
            'num_frames': len(lazy_frames),
            'rows_per_frame': estimated_rows // len(lazy_frames) if len(lazy_frames) > 0 else 0
        }
    
    def _compute_parallel_cpp_safe(self, lazy_frames: List[pl.LazyFrame], column: str,
                                  bins: int, min_val: float, max_val: float) -> Tuple[np.ndarray, int]:
        """
        Parallel processing with controlled resource usage for HPC environments.
        
        This version carefully manages thread creation to avoid resource exhaustion.
        """
        # Use limited thread count
        thread_count = min(self.max_python_threads, len(lazy_frames))
        
        print(f"🔀 Parallel processing with {thread_count} Python threads")
        print(f"   (OpenMP using {os.environ.get('OMP_NUM_THREADS', 'default')} threads per computation)")
        
        hist_counts = np.zeros(bins, dtype=np.uint64)
        total_processed = 0
        
        def process_frame_safe(lf):
            """Process single frame with resource-aware chunking."""
            frame_hist = np.zeros(bins, dtype=np.uint64)
            frame_count = 0
            
            # Process in smaller chunks to reduce memory pressure
            chunk_size = min(self.optimal_chunk_elements, 5_000_000)  # Cap at 5M for safety
            offset = 0
            
            while True:
                try:
                    chunk = (
                        lf
                        .select(pl.col(column))
                        .slice(offset, chunk_size)
                        .collect(engine="streaming")  # Updated API
                    )
                    
                    if len(chunk) == 0:
                        break
                    
                    # Convert to numpy and process
                    chunk_data = chunk.to_numpy().flatten()
                    
                    # Call C++ with data
                    chunk_hist_result = self.cpp_integrator.compute_histogram_parallel(
                        chunk_data, min_val, max_val, bins
                    )
                    
                    frame_hist += chunk_hist_result
                    frame_count += len(chunk_data)
                    offset += chunk_size
                    
                    if len(chunk) < chunk_size:
                        break
                    
                    # Small delay to prevent resource exhaustion
                    time.sleep(0.001)
                        
                except Exception as e:
                    warnings.warn(f"Frame processing error: {e}")
                    break
            
            return frame_hist, frame_count
        
        # Process frames with controlled parallelism
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            # Submit jobs with controlled pacing
            futures = []
            for i, lf in enumerate(lazy_frames):
                future = executor.submit(process_frame_safe, lf)
                futures.append((future, i))
                
                # Pace job submission on large datasets
                if i % 10 == 0 and i > 0:
                    time.sleep(0.01)
            
            # Collect results
            for future, frame_idx in futures:
                try:
                    frame_hist, frame_count = future.result(timeout=300)  # 5 min timeout
                    with self._lock:
                        hist_counts += frame_hist
                        total_processed += frame_count
                except Exception as e:
                    warnings.warn(f"Failed to process frame {frame_idx}: {e}")
        
        return hist_counts, total_processed
    
    def compute_blazing_fast(self, lazy_frames: List[pl.LazyFrame], column: str,
                               bins: int = 50, range: Optional[Tuple[float, float]] = None,
                               density: bool = False, weights: Optional[str] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Enhanced histogram computation with transparent weight support.
            
            Key improvements:
            - Automatic weight detection and routing
            - Zero overhead for unweighted case
            - Seamless C++ acceleration for both paths
            """
            
            # Range detection (existing code)
            if range is None:
                range_min = float('inf')
                range_max = float('-inf')
                
                for lf in lazy_frames:
                    stats = lf.select([
                        pl.col(column).min().alias('min'),
                        pl.col(column).max().alias('max')
                    ]).collect()
                    
                    range_min = min(range_min, stats['min'][0])
                    range_max = max(range_max, stats['max'][0])
                
                epsilon = (range_max - range_min) * 1e-10
                range = (range_min - epsilon, range_max + epsilon)
            
            # Enhanced C++ path selection
            if self._cpp_available and bins <= 10000:
                try:
                    # WEIGHTED PATH
                    if weights:
                        print(f"   Using C++ accelerated weighted histogram")
                        
                        # Efficient data collection
                        all_data = []
                        all_weights = []
                        
                        for lf in lazy_frames:
                            # Collect both columns in single pass
                            chunk = lf.select([column, weights]).collect()
                            
                            # Extract arrays
                            data_array = chunk[column].to_numpy()
                            weight_array = chunk[weights].to_numpy()
                            
                            # Pre-filter NaN values
                            valid_mask = ~(np.isnan(data_array) | np.isnan(weight_array))
                            
                            if np.any(valid_mask):
                                all_data.append(data_array[valid_mask])
                                all_weights.append(weight_array[valid_mask])
                        
                        if not all_data:
                            # No valid data
                            return np.zeros(bins), np.linspace(range[0], range[1], bins + 1)
                        
                        # Concatenate chunks
                        data_concat = np.concatenate(all_data)
                        weights_concat = np.concatenate(all_weights)
                        
                        # Ensure contiguous memory layout
                        if not data_concat.flags['C_CONTIGUOUS']:
                            data_concat = np.ascontiguousarray(data_concat, dtype=np.float64)
                        if not weights_concat.flags['C_CONTIGUOUS']:
                            weights_concat = np.ascontiguousarray(weights_concat, dtype=np.float64)
                        
                        # Allocate output
                        output = np.zeros(bins, dtype=np.float64)
                        
                        # Select best weighted implementation
                        if hasattr(self.cpp_integrator.lib, 'compute_weighted_histogram_avx2') and len(data_concat) > 5000:
                            self.cpp_integrator.lib.compute_weighted_histogram_avx2(
                                data_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                weights_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                len(data_concat),
                                range[0],
                                range[1],
                                bins,
                                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                            )
                        else:
                            self.cpp_integrator.lib.compute_weighted_histogram_scalar(
                                data_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                weights_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                len(data_concat),
                                range[0],
                                range[1],
                                bins,
                                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                            )
                        
                        counts = output
                    
                    # UNWEIGHTED PATH (existing, optimized)
                    else:
                        print(f"   Using C++ accelerated histogram")
                        
                        # Existing unweighted implementation
                        all_data = []
                        for lf in lazy_frames:
                            chunk = lf.select(column).collect()
                            data_array = chunk[column].to_numpy()
                            valid_data = data_array[~np.isnan(data_array)]
                            if len(valid_data) > 0:
                                all_data.append(valid_data)
                        
                        if not all_data:
                            return np.zeros(bins), np.linspace(range[0], range[1], bins + 1)
                        
                        data_concat = np.concatenate(all_data)
                        
                        if not data_concat.flags['C_CONTIGUOUS']:
                            data_concat = np.ascontiguousarray(data_concat, dtype=np.float64)
                        
                        output = np.zeros(bins, dtype=np.uint64)
                        
                        # Use existing functions
                        self.cpp_integrator.lib.compute_histogram_avx2_enhanced(
                            data_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            len(data_concat),
                            range[0],
                            range[1],
                            bins,
                            output.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
                        )
                        
                        counts = output.astype(np.float64)
                    
                    # Generate edges
                    edges = np.linspace(range[0], range[1], bins + 1)
                    
                    # Apply density normalization if requested
                    if density:
                        bin_widths = np.diff(edges)
                        counts = counts / (counts.sum() * bin_widths)
                    
                    return counts, edges
                    
                except Exception as e:
                    print(f"⚠️ C++ histogram failed: {e}, falling back to Polars")
            
            # POLARS FALLBACK (existing implementation)
            print("   Using Polars histogram implementation")
            
            if weights:
                # Weighted Polars path
                all_data = []
                all_weights = []
                
                for lf in lazy_frames:
                    chunk = lf.select([column, weights]).collect()
                    all_data.append(chunk[column])
                    all_weights.append(chunk[weights])
                
                col_data = pl.concat(all_data)
                weight_data = pl.concat(all_weights)
                
                np_data = col_data.to_numpy()
                np_weights = weight_data.to_numpy()
                
                valid_mask = ~(np.isnan(np_data) | np.isnan(np_weights))
                np_data = np_data[valid_mask]
                np_weights = np_weights[valid_mask]
                
            else:
                # Unweighted Polars path
                col_data = pl.concat([
                    lf.select(column).collect()
                    for lf in lazy_frames
                ])[column]
                
                np_data = col_data.to_numpy()
                np_data = np_data[~np.isnan(np_data)]
                np_weights = None
            
            # NumPy histogram (supports weights natively)
            counts, edges = np.histogram(
                np_data,
                bins=bins,
                range=range,
                density=density,
                weights=np_weights
            )
            
            return counts, edges
        
    def _select_algorithm_safe(self, dataset_stats: Dict[str, Any], bins: int) -> str:
        """Select algorithm based on dataset size AND system resources."""
        rows = dataset_stats['estimated_rows']
        
        if not self._cpp_available:
            return 'polars_fallback'
        
        # For HPC environments, be more conservative
        if rows <= 50_000_000:  # 50M
            return 'chunked_cpp'
        else:
            # Always use controlled parallel for large datasets
            return 'parallel_cpp'
    
    def _compute_direct_cpp_safe(self, lazy_frames: List[pl.LazyFrame], column: str,
                                bins: int, min_val: float, max_val: float) -> Tuple[np.ndarray, int]:
        """Direct processing for smaller datasets with updated API."""
        hist_counts = np.zeros(bins, dtype=np.uint64)
        total_processed = 0
        
        for lf in lazy_frames:
            try:
                # Process in chunks even for "direct" approach
                chunk_size = 10_000_000  # 10M at a time
                offset = 0
                
                while True:
                    chunk = (
                        lf
                        .select(pl.col(column))
                        .slice(offset, chunk_size)
                        .collect(engine="streaming")  # Updated API
                    )
                    
                    if len(chunk) == 0:
                        break
                    
                    data = chunk.to_numpy().flatten()
                    
                    # C++ call
                    frame_hist = self.cpp_integrator.compute_histogram_parallel(
                        data, min_val, max_val, bins
                    )
                    
                    hist_counts += frame_hist
                    total_processed += len(data)
                    offset += chunk_size
                    
                    if len(chunk) < chunk_size:
                        break
                
            except Exception as e:
                warnings.warn(f"Direct processing failed: {e}")
                
        return hist_counts, total_processed
    
    def _estimate_range_adaptive(self, lazy_frames: List[pl.LazyFrame], 
                               column: str, dataset_stats: Dict[str, Any]) -> Tuple[float, float]:
        """Estimate range with updated Polars API."""
        # For small datasets, compute exact range
        if dataset_stats['estimated_rows'] <= 10_000_000:
            min_val = float('inf')
            max_val = float('-inf')
            
            for lf in lazy_frames:
                try:
                    stats = lf.select([
                        pl.col(column).min().alias('min_val'),
                        pl.col(column).max().alias('max_val')
                    ]).collect(engine="streaming")  # Updated API
                    
                    if len(stats) > 0 and stats[0, 0] is not None:
                        min_val = min(min_val, stats[0, 0])
                        max_val = max(max_val, stats[0, 1])
                        
                except Exception:
                    continue
            
            if min_val < float('inf') and max_val > float('-inf'):
                return (min_val, max_val)
        
        # For large datasets, sample
        sample_size = min(1_000_000, dataset_stats['estimated_rows'] // 100)
        sample_per_frame = max(1000, sample_size // len(lazy_frames))
        
        min_vals = []
        max_vals = []
        
        for lf in lazy_frames[:min(5, len(lazy_frames))]:
            try:
                sample = (
                    lf
                    .select(pl.col(column))
                    .filter(pl.col(column).is_not_null())
                    .head(sample_per_frame)
                    .collect(engine="streaming")  # Updated API
                )
                
                if len(sample) > 0:
                    data = sample.to_numpy().flatten()
                    min_vals.append(np.min(data))
                    max_vals.append(np.max(data))
                    
            except Exception:
                continue
        
        if min_vals and max_vals:
            # Add 1% margin for safety
            range_span = max(max_vals) - min(min_vals)
            margin = range_span * 0.01
            return (min(min_vals) - margin, max(max_vals) + margin)
        
        return (0.0, 1.0)


@dataclass
class PerformanceMetrics:
    """Track performance metrics."""
    rows_processed: int = 0
    cpp_time: float = 0.0
    polars_time: float = 0.0
    memory_peak_gb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


# Example usage for testing
if __name__ == "__main__":
    print("🚀 Fixed Optimized C++ Integration Test")
    print("=" * 60)
    
    # Check current OpenMP settings
    print(f"\n📊 OpenMP Configuration:")
    print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"   OMP_DYNAMIC: {os.environ.get('OMP_DYNAMIC', 'not set')}")
    
    # Create smaller test dataset for initial testing
    print("\n📊 Creating test dataset...")
    test_frames = []
    
    # Start with smaller dataset to verify it works
    num_frames = 5
    rows_per_frame = 10_000_000  # 10M rows per frame = 50M total
    
    for i in range(num_frames):
        data = {
            'M_bc': np.random.normal(5.279, 0.003, rows_per_frame),
            'delta_E': np.random.normal(0, 0.05, rows_per_frame),
        }
        
        # Add some NaN values
        nan_mask = np.random.rand(rows_per_frame) < 0.001
        data['M_bc'][nan_mask] = np.nan
        
        test_frames.append(pl.DataFrame(data).lazy())
    
    print(f"✅ Created {len(test_frames)} frames with {num_frames * rows_per_frame:,} rows total")
    
    # Test the histogram
    histogram = OptimizedStreamingHistogram()
    
    print("\n🔥 Computing histogram...")
    start = time.time()
    
    counts, edges = histogram.compute_blazing_fast(
        test_frames, 
        'M_bc', 
        bins=1000,
        range=(5.2, 5.3)
    )
    
    total_time = time.time() - start
    
    print(f"\n📈 Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Throughput: {(num_frames * rows_per_frame) / total_time / 1e6:.1f}M rows/s")
    print(f"   Histogram shape: {counts.shape}")
    print(f"   Total counts: {np.sum(counts):,}")
    print(f"   Non-zero bins: {np.sum(counts > 0)}")