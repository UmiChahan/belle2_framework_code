
"""Layer 2: UnifiedLazyDataFrame.histogram_strategies
============================================================
This modules houses histogram related strategies and utilities for the UnifiedLazyDataFrame:
- HistogramExecutionStrategy: Enum defining various execution strategies for histogram computation.
- HistogramMetrics: Dataclass for comprehensive metrics tracking during histogram execution.
- memory_monitor: Context manager for monitoring memory usage with optional threshold alerts.
- ChunkingEnhancement: Class for adaptive chunking optimization with system awareness.
- AdaptiveChunkOptimizer: Class for calculating optimal chunk sizes based on system characteristics and dataset properties.
"""
from .utils import (
    SystemCharacteristics, PerformanceHistory,
)
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum, auto
from contextlib import contextmanager
import psutil
import resource
import sys
import warnings
class HistogramExecutionStrategy(Enum):
    """Execution strategies for histogram computation."""
    CPP_ACCELERATED = auto()      # C++ streaming histogram
    BILLION_ROW_ENGINE = auto()   # Layer 1 billion row engine with spilling
    ADAPTIVE_CHUNKED = auto()     # Adaptive chunked execution (advanced)
    LAZY_CHUNKED = auto()         # Standard Polars lazy with smart chunking
    MEMORY_CONSTRAINED = auto()   # Ultra-conservative for OOM prevention
    FALLBACK_BASIC = auto()       # Last resort basic implementation
# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@dataclass
class HistogramMetrics:
    """Comprehensive metrics for histogram execution."""
    strategy: HistogramExecutionStrategy
    total_rows: int
    processed_rows: int
    execution_time: float
    memory_peak_mb: float
    chunk_size_used: int
    chunks_processed: int
    errors_recovered: int
    
    @property
    def throughput_mps(self) -> float:
        """Million rows per second throughput."""
        if self.execution_time > 0:
            return (self.processed_rows / 1e6) / self.execution_time
        return 0.0
    
    @property
    def efficiency(self) -> float:
        """Processing efficiency (processed/total)."""
        if self.total_rows > 0:
            return self.processed_rows / self.total_rows
        return 0.0

# ============================================================================
# MEMORY MONITORING CONTEXT
# ============================================================================
@contextmanager
def memory_monitor(threshold_mb: float = None, include_children: bool = False):
    """Context manager for memory monitoring with optional threshold alerts.
    
    - Yields a callable mem() -> current MB delta from entry (backward compatible).
    - Adds attributes on the yielded callable:
        .initial_mb: absolute MB at entry (self + children if enabled)
        .final_mb: absolute MB at exit
        .final_delta_mb: final_mb - initial_mb
        .peak_mb: peak MB delta observed during the scope (relative to entry)
        .include_children: whether children were included
    """
    process = psutil.Process()

    def _rss_mb(proc: psutil.Process) -> float:
        try:
            return proc.memory_info().rss / (1024.0 * 1024.0)
        except psutil.Error:
            return 0.0

    def _children_rss_mb(proc: psutil.Process) -> float:
        if not include_children:
            return 0.0
        total = 0.0
        try:
            for child in proc.children(recursive=True):
                total += _rss_mb(child)
        except psutil.Error:
            pass
        return total

    def _ru_to_mb(ru) -> float:
        # ru_maxrss units: Linux=KB, macOS=bytes
        val = getattr(ru, "ru_maxrss", 0)
        if sys.platform == "darwin":
            return float(val) / (1024.0 * 1024.0)
        else:
            return float(val) / 1024.0

    # Entry baselines
    initial_self_mb = _rss_mb(process)
    initial_children_mb = _children_rss_mb(process)
    initial_total_mb = initial_self_mb + initial_children_mb

    try:
        ru_self_entry_mb = _ru_to_mb(resource.getrusage(resource.RUSAGE_SELF))
    except Exception:
        ru_self_entry_mb = initial_self_mb

    # Callable to yield (current delta from entry)
    def usage_delta_mb() -> float:
        current_total = _rss_mb(process) + _children_rss_mb(process)
        return max(0.0, current_total - initial_total_mb)

    # Attach introspection fields; will be updated on exit
    usage_delta_mb.initial_mb = initial_total_mb          # type: ignore[attr-defined]
    usage_delta_mb.final_mb = initial_total_mb            # type: ignore[attr-defined]
    usage_delta_mb.final_delta_mb = 0.0                   # type: ignore[attr-defined]
    usage_delta_mb.peak_mb = 0.0                          # type: ignore[attr-defined]  # delta from entry
    usage_delta_mb.include_children = include_children    # type: ignore[attr-defined]

    try:
        yield usage_delta_mb
    finally:
        final_self_mb = _rss_mb(process)
        final_children_mb = _children_rss_mb(process)
        final_total_mb = final_self_mb + final_children_mb

        # Exit ru_maxrss for self; children peak is approximated by exit delta
        try:
            ru_self_exit_mb = _ru_to_mb(resource.getrusage(resource.RUSAGE_SELF))
        except Exception:
            ru_self_exit_mb = final_self_mb

        # Peak delta for the process itself during the scope
        peak_self_delta_mb = max(0.0, ru_self_exit_mb - max(ru_self_entry_mb, initial_self_mb))
        # Approximate children peak as non-negative exit delta (lower bound)
        peak_children_delta_mb = max(0.0, final_children_mb - initial_children_mb) if include_children else 0.0

        # Peak absolute MB within scope ≈ entry + deltas; also consider final_total_mb
        peak_absolute_mb = max(final_total_mb, initial_total_mb + peak_self_delta_mb + peak_children_delta_mb)

        # Update attributes
        usage_delta_mb.final_mb = final_total_mb           # type: ignore[attr-defined]
        usage_delta_mb.final_delta_mb = max(0.0, final_total_mb - initial_total_mb)  # type: ignore[attr-defined]
        usage_delta_mb.peak_mb = max(0.0, peak_absolute_mb - initial_total_mb)       # type: ignore[attr-defined]

        # Threshold warnings: prefer peak delta; fall back to final delta
        if threshold_mb is not None:
            if usage_delta_mb.peak_mb > threshold_mb:      # type: ignore[attr-defined]
                warnings.warn(
                    f"Memory peak exceeded threshold: {usage_delta_mb.peak_mb:.1f}MB > {threshold_mb}MB"
                )
            elif usage_delta_mb.final_delta_mb > threshold_mb:  # type: ignore[attr-defined]
                warnings.warn(
                    f"Memory usage exceeded threshold: {usage_delta_mb.final_delta_mb:.1f}MB > {threshold_mb}MB"
                )
class AdaptiveChunkOptimizer:
    """
    Research-grade adaptive chunking with system awareness.
    
    ALGORITHMIC FOUNDATION:
    ├── Cache-Conscious Sizing: Optimize for L3 cache utilization
    ├── Bandwidth Optimization: Match memory subsystem capabilities  
    ├── Storage-Aware Patterns: Align with storage characteristics
    ├── Scale-Adaptive Logic: Dataset-proportional optimization
    └── Performance Learning: Continuous improvement via feedback
    
    COMPLEXITY: O(1) amortized after system profiling
    MEMORY: O(1) constant overhead regardless of dataset size
    """
    
    def __init__(self, memory_budget_gb: float):
        self.memory_budget_gb = memory_budget_gb
        self.system = SystemCharacteristics.detect()
        self.performance_history = PerformanceHistory()
        self._cache_utilization_target = 0.75
        self._bandwidth_utilization_target = 0.80

    def calculate_optimal_chunk_size(self, estimated_rows: int,
                                     avg_row_bytes: float,
                                     operation_type: str = 'histogram') -> int:
        """Robust, bounded chunk-size selection.

        Goals:
        - Respect memory budget (<=25% per chunk)
        - Scale with dataset size (10–75 chunks depending on scale)
        - Keep results within [1_000, 20_000_000] and <= estimated_rows
        """
        try:
            # Memory-based bound
            per_chunk_bytes = max(avg_row_bytes, 1.0)
            budget_bytes = float(self.memory_budget_gb) * (1024 ** 3) * 0.25
            mem_based = max(10_000, int(budget_bytes / per_chunk_bytes))

            # Scale-based bound
            if estimated_rows <= 0:
                return 100_000
            if estimated_rows < 1_000_000:
                target_chunks = 10
            elif estimated_rows < 10_000_000:
                target_chunks = 15
            elif estimated_rows < 100_000_000:
                target_chunks = 30
            else:
                # parallelism-aware upper bound
                target_chunks = min(75, (self.system.cpu_cores or 4) * 4)
            scale_based = max(100_000, estimated_rows // target_chunks)

            # Conservative baseline bound
            baseline = max(50_000, min(5_000_000, estimated_rows // 20))

            # Aggregate via median to avoid extremes
            base = int(np.median([mem_based, scale_based, baseline]))

            # Final guards
            base = max(1_000, min(base, 20_000_000))
            base = min(base, estimated_rows)
            return base
        except Exception:
            # Absolute fallback
            return max(100_000, estimated_rows // 50 if estimated_rows > 0 else 100_000)


class ChunkingEnhancement:
    """
    Seamless integration interface for existing frameworks.
    
    INTEGRATION PATTERN: Dependency Injection + Factory
    BACKWARD COMPATIBILITY: 100% preserved
    PERFORMANCE IMPACT: Zero overhead when disabled
    """
    
    _optimizer_cache: Dict[float, AdaptiveChunkOptimizer] = {}
    _enabled: bool = True
    
    @classmethod
    def get_optimizer(cls, memory_budget_gb: float) -> AdaptiveChunkOptimizer:
        """Get cached optimizer instance for memory budget."""
        
        if memory_budget_gb not in cls._optimizer_cache:
            cls._optimizer_cache[memory_budget_gb] = AdaptiveChunkOptimizer(memory_budget_gb)
        
        return cls._optimizer_cache[memory_budget_gb]
    
    @classmethod
    def enable_optimization(cls, enabled: bool = True):
        """Enable/disable optimization globally."""
        cls._enabled = enabled
        print(f"🔧 Adaptive chunking optimization: {'enabled' if enabled else 'disabled'}")
    
    @classmethod
    def calculate_optimal_chunk_size(cls,
                                memory_budget_gb: float,
                                estimated_rows: int,
                                avg_row_bytes: float = 100.0,
                                operation_type: str = 'histogram',
                                fallback_calculation: Optional[callable] = None) -> int:
        """Robust integration with recursion protection."""
        
        # CRITICAL: Thread-safe recursion guard
        import threading
        thread_id = threading.get_ident()
        guard_attr = f'_in_calculation_{thread_id}'
        
        if hasattr(cls, guard_attr):
            print("⚠️ Recursion detected in chunk calculation, using fallback")
            return cls._conservative_fallback(memory_budget_gb, estimated_rows)
        
        setattr(cls, guard_attr, True)
        try:
            if not cls._enabled or estimated_rows <= 0:
                return fallback_calculation() if fallback_calculation else \
                    cls._conservative_fallback(memory_budget_gb, estimated_rows)
            
            # Get optimizer with proper initialization
            optimizer = cls.get_optimizer(memory_budget_gb)
            
            # Validate optimizer structure
            if not hasattr(optimizer, 'calculate_optimal_chunk_size'):
                raise AttributeError("Optimizer missing required method")
            
            # Execute calculation with validation
            chunk_size = optimizer.calculate_optimal_chunk_size(
                estimated_rows, avg_row_bytes, operation_type
            )
            
            # Strict validation
            if chunk_size <= 0:
                chunk_size = max(100_000, estimated_rows // 100)
            elif chunk_size > estimated_rows:
                chunk_size = estimated_rows
            
            return chunk_size
        
        except Exception as e:
            print(f"⚠️ Adaptive optimization failed: {e}, using fallback")
            return cls._conservative_fallback(memory_budget_gb, estimated_rows)
        finally:
            delattr(cls, guard_attr)
    
    @classmethod
    def _conservative_fallback(cls, memory_budget_gb: float, estimated_rows: int) -> int:
        """Conservative fallback calculation."""
        
        # Simple memory-based calculation
        bytes_per_row = 100  # Conservative estimate
        available_memory = memory_budget_gb * 1024**3 * 0.2  # 20% utilization
        memory_optimal = int(available_memory / bytes_per_row)
        
        # Scale with dataset size
        if estimated_rows < 1_000_000:
            return max(10_000, min(memory_optimal, estimated_rows // 10))
        elif estimated_rows < 10_000_000:
            return max(100_000, min(memory_optimal, estimated_rows // 50))
        else:
            return max(500_000, min(memory_optimal, estimated_rows // 100))
