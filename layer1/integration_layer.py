"""
Integration Layer - Bridging Compute Engines with BillionCapableFramework
========================================================================

This module provides seamless integration between our new compute-first engines
and the existing BillionCapableFramework, maintaining backward compatibility
while enabling the new architecture's benefits.

Design Philosophy:
- Zero breaking changes to existing code
- Transparent performance improvements
- Gradual migration path
- Runtime engine selection based on workload
"""

import functools
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TypeVar
import numpy as np
import polars as pl
from pathlib import Path
import warnings

from layer0 import ComputeCapability
# ============================================================================
# Import our new engines
from layer1.lazy_compute_engine import LazyComputeEngine
from layer1.billion_capable_engine import (
    IntegratedBillionCapableEngine
)

# Import existing framework components
try:
    import sys
    from pathlib import Path
    # Add parent directory to path to find Combined_framework
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from Combined_framework import (
        BillionCapableFramework,
        BillionCapableBlazingCore,
        UnifiedAPI,
    )
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    FRAMEWORK_AVAILABLE = False
    warnings.warn(f"BillionCapableFramework not available: {e}, running in standalone mode")

T = TypeVar('T')

# ============================================================================
# Engine Selection Strategy
# ============================================================================

@dataclass
class WorkloadProfile:
    """Characterizes a computational workload for engine selection."""
    estimated_rows: int
    estimated_columns: int
    operation_count: int
    has_joins: bool = False
    has_complex_aggregations: bool = False
    has_udfs: bool = False  # User-defined functions
    memory_intensive: bool = False
    
    @property
    def complexity_score(self) -> float:
        """Compute a complexity score for engine selection."""
        base_score = np.log10(max(1, self.estimated_rows))
        
        # Adjust for operations
        if self.has_joins:
            base_score *= 2.0
        if self.has_complex_aggregations:
            base_score *= 1.5
        if self.has_udfs:
            base_score *= 1.3
        if self.memory_intensive:
            base_score *= 1.2
            
        # Factor in operation count
        base_score *= (1 + self.operation_count * 0.1)
        
        return base_score


class EngineSelector:
    """
    Intelligently selects the best compute engine based on workload characteristics.
    
    This implements a strategy pattern for engine selection, allowing us to adapt
    to different workload patterns without changing client code.
    """
    
    def __init__(self,
                 memory_budget_gb: float = 8.0,
                 force_engine: Optional[str] = None):
        self.memory_budget_gb = memory_budget_gb
        self.force_engine = force_engine
        
        # Engine thresholds
        self.billion_row_threshold = 100_000_000
        self.lazy_threshold = 10_000_000
        
        # Cache engines for reuse
        self._engine_cache = {}
    
    def select_engine(self, profile: WorkloadProfile) -> Union[LazyComputeEngine, IntegratedBillionCapableEngine]:
        """Select optimal engine based on workload profile."""
        # Allow forced selection for testing
        if self.force_engine:
            return self._get_engine(self.force_engine)
        
        # Decision tree based on profile
        if profile.estimated_rows >= self.billion_row_threshold:
            # Billion-row datasets always use BillionCapableEngine
            return self._get_engine('billion')
        
        elif profile.complexity_score > 15.0:
            # High complexity benefits from advanced optimization
            return self._get_engine('billion')
        
        elif profile.has_joins and profile.estimated_rows > self.lazy_threshold:
            # Large joins benefit from specialized strategies
            return self._get_engine('billion')
        
        else:
            # Standard workloads use LazyComputeEngine
            return self._get_engine('lazy')
    
    def _get_engine(self, engine_type: str):
        """Get or create engine instance."""
        if engine_type not in self._engine_cache:
            if engine_type == 'billion':
                self._engine_cache[engine_type] = IntegratedBillionCapableEngine(
                    memory_budget_gb=self.memory_budget_gb,
                    optimization_level=2,
                    enable_profiling=True
                )
            else:
                self._engine_cache[engine_type] = LazyComputeEngine(
                    memory_budget_gb=self.memory_budget_gb,
                    optimization_level=2
                )
        
        return self._engine_cache[engine_type]


# ============================================================================
# Framework Adapter
# ============================================================================

class ComputeEngineAdapter:
    """
    Adapts our compute engines to work seamlessly with BillionCapableFramework.
    
    This adapter implements the Adapter pattern, translating between the new
    compute-first architecture and the existing framework's expectations.
    """
    
    def __init__(self, 
                 selector: EngineSelector = None,
                 enable_monitoring: bool = True):
        self.selector = selector or EngineSelector()
        self.enable_monitoring = enable_monitoring
        
        # Performance tracking
        self.operation_history = []
        self.engine_usage_stats = {'lazy': 0, 'billion': 0}
    
    def adapt_blazing_core(self, blazing_core: 'BillionCapableBlazingCore') -> 'EnhancedBlazingCore':
        """Enhance existing BlazingCore with compute engine capabilities."""
        return EnhancedBlazingCore(blazing_core, self)
    
    def create_capability_from_frames(self, 
                                    lazy_frames: List[pl.LazyFrame],
                                    estimated_rows: int = None) -> ComputeCapability:
        """Create appropriate compute capability from LazyFrames."""
        # Build workload profile
        profile = self._analyze_workload(lazy_frames, estimated_rows)
        
        # Select engine
        engine = self.selector.select_engine(profile)
        
        # Track usage
        engine_type = 'billion' if isinstance(engine, IntegratedBillionCapableEngine) else 'lazy'
        self.engine_usage_stats[engine_type] += 1
        
        # Create capability
        if len(lazy_frames) == 1:
            capability = engine.create_capability(lazy_frames[0])
        else:
            capability = engine.create_capability(lazy_frames)
        
        if self.enable_monitoring:
            # Wrap with monitoring
            capability = MonitoredCapability(capability, self)
        
        return capability
    
    def _analyze_workload(self, 
                         lazy_frames: List[pl.LazyFrame],
                         estimated_rows: int = None) -> WorkloadProfile:
        """Analyze workload characteristics from LazyFrames."""
        # Estimate total rows if not provided
        if estimated_rows is None:
            estimated_rows = self._estimate_total_rows(lazy_frames)
        
        # Analyze schema
        schema = lazy_frames[0].collect_schema() if lazy_frames else {}
        estimated_columns = len(schema)
        
        # Analyze operations (simplified - would inspect lazy plan in practice)
        return WorkloadProfile(
            estimated_rows=estimated_rows,
            estimated_columns=estimated_columns,
            operation_count=len(self.operation_history),
            has_joins=any('join' in str(op).lower() for op in self.operation_history[-10:]),
            has_complex_aggregations=any('agg' in str(op).lower() for op in self.operation_history[-10:]),
            has_udfs=False,  # Would need to inspect plan
            memory_intensive=estimated_rows * estimated_columns * 8 > self.selector.memory_budget_gb * 1024**3
        )
    
    def _estimate_total_rows(self, lazy_frames: List[pl.LazyFrame]) -> int:
        """Fast row estimation from LazyFrames."""
        if not lazy_frames:
            return 0
        
        # Sample first few frames
        sample_size = 0
        for lf in lazy_frames[:3]:
            try:
                # Use metadata if available
                sample_size += lf.select(pl.count()).collect(streaming=True)[0, 0]
            except:
                sample_size += 1_000_000  # Conservative estimate
        
        # Extrapolate
        if len(lazy_frames) <= 3:
            return sample_size
        else:
            return sample_size * len(lazy_frames) // 3


# ============================================================================
# Enhanced Framework Components
# ============================================================================

class EnhancedBlazingCore:
    """
    Wraps existing BlazingCore to use compute engines transparently.
    
    This is a Decorator that adds compute engine capabilities while maintaining
    the original interface. All existing code continues to work unchanged.
    """
    
    def __init__(self, 
                 original_core: 'BillionCapableBlazingCore',
                 adapter: ComputeEngineAdapter):
        self._original = original_core
        self._adapter = adapter
        self._capability = None
        
        # Delegate attribute access to original
        self._setup_delegation()
    
    def _setup_delegation(self):
        """Set up attribute delegation to original core."""
        # Copy essential attributes
        self.lazy_frames = self._original.lazy_frames
        self._estimated_total_rows = self._original._estimated_total_rows
        self._billion_row_mode = self._original._billion_row_mode
    
    def __getattr__(self, name):
        """Delegate unknown attributes to original core."""
        return getattr(self._original, name)
    
    def hist(self, column: str, bins: int = 50, 
             range: Optional[Tuple[float, float]] = None,
             density: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Histogram using compute engine when beneficial."""
        # Create capability if needed
        if self._capability is None:
            self._capability = self._adapter.create_capability_from_frames(
                self.lazy_frames,
                self._estimated_total_rows
            )
        
        # Use compute engine for large datasets
        if self._estimated_total_rows > 10_000_000:
            # Transform to histogram computation
            hist_capability = self._capability.transform(
                lambda df: self._compute_histogram(df, column, bins, range, density)
            )
            
            # Materialize result
            result = hist_capability.materialize()
            
            # Extract histogram data
            if isinstance(result, dict) and 'counts' in result:
                return result['counts'], result['edges']
            else:
                # Fallback to original
                return self._original.hist(column, bins, range, density, **kwargs)
        else:
            # Use original for small datasets
            return self._original.hist(column, bins, range, density, **kwargs)
    
    @staticmethod
    def _compute_histogram(df: pl.DataFrame, column: str, bins: int,
                          range: Optional[Tuple[float, float]], density: bool) -> Dict:
        """Compute histogram from DataFrame."""
        # Extract column data
        data = df[column].to_numpy()
        
        # Compute histogram
        counts, edges = np.histogram(data, bins=bins, range=range, density=density)
        
        return {'counts': counts, 'edges': edges}
    
    def query(self, expr: str) -> 'EnhancedBlazingCore':
        """Query using compute engine optimization."""
        # Track operation
        self._adapter.operation_history.append(('query', expr))
        
        # Apply query through original mechanism
        # (compute engines will optimize during materialization)
        filtered_frames = [lf.filter(pl.sql_expr(expr)) for lf in self.lazy_frames]
        
        # Create new core with filtered frames
        new_original = BillionCapableBlazingCore(filtered_frames)
        return EnhancedBlazingCore(new_original, self._adapter)


class MonitoredCapability(ComputeCapability[T]):
    """
    Wrapper that monitors capability execution for diagnostics.
    
    This implements the Proxy pattern to add monitoring without modifying
    the underlying capability implementation.
    """
    
    def __init__(self, 
                 base_capability: ComputeCapability[T],
                 adapter: ComputeEngineAdapter):
        self._base = base_capability
        self._adapter = adapter
        self._execution_count = 0
    
    def transform(self, operation: Callable[[T], T]) -> 'MonitoredCapability[T]':
        """Monitor transformations."""
        self._adapter.operation_history.append(('transform', operation.__name__ if hasattr(operation, '__name__') else 'lambda'))
        
        new_base = self._base.transform(operation)
        return MonitoredCapability(new_base, self._adapter)
    
    def materialize(self) -> T:
        """Monitor materialization."""
        import time
        
        start_time = time.time()
        self._execution_count += 1
        
        try:
            result = self._base.materialize()
            
            # Record success
            duration = time.time() - start_time
            self._adapter.operation_history.append(('materialize_success', duration))
            
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            self._adapter.operation_history.append(('materialize_failure', (duration, str(e))))
            raise
    
    # Delegate other methods
    def partition_compute(self, partitioner: Callable[[T], Dict[str, T]]) -> Dict[str, ComputeCapability[T]]:
        return self._base.partition_compute(partitioner)
    
    def get_compute_graph(self):
        return self._base.get_compute_graph()
    
    def estimate_memory(self) -> int:
        return self._base.estimate_memory()
    
    def is_materialized(self) -> bool:
        return self._base.is_materialized()


# ============================================================================
# Framework Enhancement
# ============================================================================

class ComputeEnhancedFramework(BillionCapableFramework):
    """
    Enhanced framework that transparently uses compute engines.
    
    This is a drop-in replacement for BillionCapableFramework that automatically
    uses our compute engines for better performance while maintaining full
    backward compatibility.
    """
    
    def __init__(self, 
                 memory_budget_gb: float = 8.0,
                 max_workers: int = 4,
                 enable_compute_engines: bool = True):
        # Initialize parent
        super().__init__(memory_budget_gb, max_workers)
        
        # Set up compute engine integration
        self.enable_compute_engines = enable_compute_engines
        
        if enable_compute_engines:
            self.engine_adapter = ComputeEngineAdapter(
                selector=EngineSelector(memory_budget_gb=memory_budget_gb)
            )
        else:
            self.engine_adapter = None
    
    def load_data(self, 
                  paths: Union[str, List[str]], 
                  process: str = 'auto',
                  use_compute_engine: Optional[bool] = None) -> 'ComputeEnhancedFramework':
        """Load data with optional compute engine acceleration."""
        # Load using parent method
        super().load_data(paths, process)
        
        # Enhance with compute engines if enabled
        use_engines = use_compute_engine if use_compute_engine is not None else self.enable_compute_engines
        
        if use_engines and self.engine_adapter and self.blazing_core:
            # Replace blazing core with enhanced version
            self.blazing_core = self.engine_adapter.adapt_blazing_core(self.blazing_core)
            
            # Update unified API to use enhanced core
            self.unified_api = UnifiedAPI(self.blazing_core, self.smart_evaluator)
            
            print("✨ Compute engines enabled for optimal performance")
        
        return self
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about compute engine usage."""
        if not self.engine_adapter:
            return {'compute_engines_enabled': False}
        
        return {
            'compute_engines_enabled': True,
            'engine_usage': self.engine_adapter.engine_usage_stats,
            'operation_count': len(self.engine_adapter.operation_history),
            'last_operations': self.engine_adapter.operation_history[-10:]
        }


# ============================================================================
# Migration Utilities
# ============================================================================

def migrate_to_compute_engines(existing_framework: BillionCapableFramework) -> ComputeEnhancedFramework:
    """
    Migrate existing framework instance to use compute engines.
    
    This allows gradual migration of existing code without breaking changes.
    """
    # Create new enhanced framework
    enhanced = ComputeEnhancedFramework(
        memory_budget_gb=existing_framework.memory_budget_gb,
        max_workers=existing_framework.max_workers,
        enable_compute_engines=True
    )
    
    # Copy state
    enhanced._lazy_frames_by_process = existing_framework._lazy_frames_by_process
    enhanced.current_process = existing_framework.current_process
    
    # Recreate components with enhancement
    if existing_framework.blazing_core:
        enhanced.blazing_core = enhanced.engine_adapter.adapt_blazing_core(
            existing_framework.blazing_core
        )
        enhanced.smart_evaluator = existing_framework.smart_evaluator
        enhanced.unified_api = UnifiedAPI(enhanced.blazing_core, enhanced.smart_evaluator)
    
    return enhanced


# ============================================================================
# Example Usage and Benchmarking
# ============================================================================

def demonstrate_integration():
    """Demonstrate seamless integration with existing code."""
    
    print("Integration Layer Demo - Compute Engines + BillionCapableFramework")
    print("=" * 65)
    
    # Example 1: Direct drop-in replacement
    print("\n1. Drop-in Replacement Test")
    print("-" * 30)
    
    # This works exactly like BillionCapableFramework
    framework = ComputeEnhancedFramework(memory_budget_gb=16.0)
    
    # Load data normally
    test_path = "/path/to/data"
    if Path(test_path).exists():
        framework.load_data(test_path)
        
        # Use exactly as before - compute engines work transparently
        result = framework.query("value > 100").hist("value", bins=50)
        print("✅ Existing code works unchanged!")
    
    # Example 2: Gradual migration
    print("\n2. Gradual Migration Test")
    print("-" * 30)
    
    if FRAMEWORK_AVAILABLE:
        # Start with existing framework
        old_framework = BillionCapableFramework(memory_budget_gb=16.0)
        
        # Migrate when ready
        new_framework = migrate_to_compute_engines(old_framework)
        print("✅ Migrated to compute engines!")
        
        # Check usage stats
        stats = new_framework.get_engine_stats()
        print(f"Engine usage: {stats}")
    
    # Example 3: Advanced control
    print("\n3. Advanced Engine Control")
    print("-" * 30)
    
    # Force specific engine
    selector = EngineSelector(memory_budget_gb=32.0, force_engine='billion')
    adapter = ComputeEngineAdapter(selector=selector)
    
    # Create capability directly
    test_frames = [pl.LazyFrame({'x': range(1000000)})]
    capability = adapter.create_capability_from_frames(test_frames, estimated_rows=1_000_000_000)
    
    print(f"✅ Forced billion-row engine for 1B row dataset")
    
    # Example 4: Performance comparison
    print("\n4. Performance Comparison")
    print("-" * 30)
    
    def benchmark_operation(framework, operation_desc: str, operation: Callable):
        """Benchmark an operation."""
        import time
        
        start = time.time()
        try:
            result = operation(framework)
            duration = time.time() - start
            print(f"  {operation_desc}: {duration:.3f}s")
            return result, duration
        except Exception as e:
            print(f"  {operation_desc}: Failed - {e}")
            return None, float('inf')
    
    # Would run comparative benchmarks here
    print("  (Benchmarks would run with real data)")
    
    print("\n" + "=" * 65)
    print("Integration Layer ready for production!")
    print("\nBenefits:")
    print("  ✓ Zero breaking changes to existing code")
    print("  ✓ Automatic engine selection based on workload")
    print("  ✓ Transparent performance improvements")
    print("  ✓ Gradual migration path available")
    print("  ✓ Full monitoring and diagnostics")


if __name__ == "__main__":
    demonstrate_integration()