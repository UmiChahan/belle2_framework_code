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

# Print-once / verbosity controls
_PRINTED_OMP = False
_PRINTED_PROCESS_LIMITS = False
_PRINTED_CPP_ACTIVE = False
_PRINTED_SYSTEM_ANALYSIS = False

# Set to True to enable detailed runtime prints (kept quiet by default)
PRINT_RUNTIME_INFO = False


def configure_openmp_for_hpc():
    """
    Configure OpenMP settings for HPC environments to prevent thread exhaustion.
    
    This function sets environment variables that control OpenMP behavior
    BEFORE any OpenMP-enabled libraries are loaded.
    """
    # Limit OpenMP threads to prevent resource exhaustion
    # This should be called at the very beginning of your script
    global _PRINTED_OMP
    if 'OMP_NUM_THREADS' not in os.environ:
        # Use a conservative number of threads
        cpu_count = psutil.cpu_count(logical=False) or 4
        # Limit to 8 threads max for OpenMP to leave room for Python threads
        omp_threads = min(cpu_count, 8)
        os.environ['OMP_NUM_THREADS'] = str(omp_threads)
        if not _PRINTED_OMP:
            print(f"🔧 Set OMP_NUM_THREADS={omp_threads}")
            _PRINTED_OMP = True
    
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
            from .cpp_histogram_integrator import cpp_histogram_integrator
            self.cpp_integrator = cpp_histogram_integrator
            self._cpp_available = cpp_histogram_integrator.is_available()
            
            if self._cpp_available:
                global _PRINTED_CPP_ACTIVE
                if not _PRINTED_CPP_ACTIVE:
                    print("✅ C++ histogram acceleration: ACTIVE")
                    _PRINTED_CPP_ACTIVE = True
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
            global _PRINTED_PROCESS_LIMITS
            if not _PRINTED_PROCESS_LIMITS and PRINT_RUNTIME_INFO:
                print(f"📊 Process limits: soft={soft_limit}, hard={hard_limit}")
            
            # Calculate safe thread counts
            # Reserve threads: main thread + OpenMP threads + safety margin
            omp_threads = int(os.environ.get('OMP_NUM_THREADS', 8))
            reserved_threads = 1 + omp_threads + 10  # safety margin
            
            # Available for Python thread pool
            available_for_pool = min(soft_limit - reserved_threads, 16)
            self.max_python_threads = max(1, available_for_pool)
            
            if PRINT_RUNTIME_INFO:
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
        
        global _PRINTED_SYSTEM_ANALYSIS
        if not _PRINTED_SYSTEM_ANALYSIS and PRINT_RUNTIME_INFO:
            print(f"📊 System analysis:")
            print(f"   Available memory: {self.available_memory_gb:.1f} GB")
            print(f"   Optimal chunk size: {self.optimal_chunk_elements:,} elements")
            _PRINTED_SYSTEM_ANALYSIS = True
    
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
        
        if PRINT_RUNTIME_INFO:
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
                           density: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram with automatic algorithm selection and resource management.
        
        This is the main entry point that ensures stable execution on HPC systems.
        """
        start_time = time.time()
        
        # Estimate dataset size and characteristics
        dataset_stats = self._analyze_dataset(lazy_frames, column)
        
        # Get range efficiently
        if range is None:
            range = self._estimate_range_adaptive(lazy_frames, column, dataset_stats)
        
        min_val, max_val = range
        
        # Select optimal algorithm based on size AND available resources
        algorithm = self._select_algorithm_safe(dataset_stats, bins)
        
        print(f"🎯 Algorithm: {algorithm} for {dataset_stats['estimated_rows']:,} rows")
        
        # Execute with selected algorithm
        if algorithm == 'chunked_cpp' or algorithm == 'parallel_cpp':
            # Use our safe parallel implementation
            hist_counts, total_processed = self._compute_parallel_cpp_safe(
                lazy_frames, column, bins, min_val, max_val
            )
        else:
            # For small datasets, use direct approach if C++ available; otherwise, Polars fallback
            if algorithm == 'polars_fallback':
                hist_counts, total_processed = self._compute_polars_fallback(
                    lazy_frames, column, bins, min_val, max_val
                )
            else:
                hist_counts, total_processed = self._compute_direct_cpp_safe(
                    lazy_frames, column, bins, min_val, max_val
                )
        
        # Handle density normalization
        if density and total_processed > 0:
            bin_width = (max_val - min_val) / bins
            hist_counts = hist_counts.astype(float) / (total_processed * bin_width)
        
        # Create edges
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Performance reporting
        elapsed = time.time() - start_time
        throughput = total_processed / elapsed if elapsed > 0 else 0
        
        print(f"⚡ Histogram complete: {total_processed:,} rows in {elapsed:.2f}s "
              f"({throughput/1e6:.1f}M rows/s)")
        # Persist last-run diagnostics for callers
        try:
            self._last_rows_scanned = int(total_processed)
            try:
                self._last_in_range = int(hist_counts.sum())
            except Exception:
                self._last_in_range = None
            self._last_elapsed_s = float(elapsed)
            self._last_throughput_mps = float(throughput / 1e6)
        except Exception:
            pass

        return hist_counts, bin_edges

    def _compute_polars_fallback(self, lazy_frames: List[pl.LazyFrame], column: str,
                                 bins: int, min_val: float, max_val: float) -> Tuple[np.ndarray, int]:
        """Compute histogram using Polars as safe fallback with proper measurement."""
        hist_counts = np.zeros(bins, dtype=np.uint64)
        total_processed = 0

        for lf in lazy_frames:
            try:
                df = (
                    lf.select(pl.col(column))
                      .filter(pl.col(column).is_between(min_val, max_val, closed='both'))
                      .collect(engine="streaming")
                )
                if len(df) == 0:
                    continue
                data = df[column].to_numpy()
                counts, _ = np.histogram(data, bins=bins, range=(min_val, max_val))
                hist_counts += counts.astype(np.uint64)
                total_processed += data.shape[0]
            except Exception as e:
                warnings.warn(f"Polars fallback failed on frame: {e}")
        return hist_counts, total_processed
    
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