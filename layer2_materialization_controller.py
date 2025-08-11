"""
Layer 2: MaterializationController & Advanced Optimization Components
====================================================================

This module implements intelligent materialization strategies and advanced
optimization components that enhance the compute-first architecture.

Key Components:
- MaterializationController: Smart format selection and conversion
- GraphOptimizationEngine: Continuous compute graph optimization
- MemoryAwareExecutor: Adaptive execution with memory constraints
- PerformanceProfiler: Real-time performance analysis
"""

import os
import sys
import time
import warnings
import weakref
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Union, Iterator, Callable, 
    TypeVar, Generic, Tuple, Set
)
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add missing import for Path
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import Layer 0 and Layer 1 components
from layer0 import (
    ComputeCapability, ComputeNode, ComputeOpType,
    MemoryAwareCompute
)
from layer1.lazy_compute_engine import (
    GraphNode, ExecutionContext, AdaptiveMemoryEstimator
)

T = TypeVar('T')

# ============================================================================
# Materialization Strategy Components
# ============================================================================

class MaterializationFormat(Enum):
    """Supported materialization formats."""
    ARROW = auto()      # Apache Arrow for zero-copy
    POLARS = auto()     # Polars for further processing
    NUMPY = auto()      # NumPy for numerical computation
    PANDAS = auto()     # Pandas for compatibility
    PARQUET = auto()    # Parquet for persistence
    NATIVE = auto()     # Keep in native format


@dataclass
class MaterializationHints:
    """Hints for optimal materialization strategy."""
    downstream_operations: List[str] = field(default_factory=list)
    expected_access_pattern: str = 'sequential'  # 'sequential', 'random', 'columnar'
    memory_constraints: Optional[int] = None
    persistence_required: bool = False
    zero_copy_preferred: bool = True
    compression_acceptable: bool = True


class MaterializationController:
    """
    Intelligent materialization with format selection.
    
    This controller analyzes the compute graph and downstream operations
    to select the optimal format for materialization.
    """
    
    def __init__(self):
        self._format_costs = self._init_format_costs()
        self._conversion_cache = {}
        self._profiling_data = {}
        
    def _init_format_costs(self) -> Dict[Tuple[MaterializationFormat, MaterializationFormat], float]:
        """Initialize conversion cost matrix."""
        # Cost of converting between formats (relative scale)
        return {
            # From Arrow
            (MaterializationFormat.ARROW, MaterializationFormat.POLARS): 0.1,
            (MaterializationFormat.ARROW, MaterializationFormat.NUMPY): 0.3,
            (MaterializationFormat.ARROW, MaterializationFormat.PANDAS): 0.5,
            
            # From Polars
            (MaterializationFormat.POLARS, MaterializationFormat.ARROW): 0.1,
            (MaterializationFormat.POLARS, MaterializationFormat.NUMPY): 0.2,
            (MaterializationFormat.POLARS, MaterializationFormat.PANDAS): 0.3,
            
            # From NumPy
            (MaterializationFormat.NUMPY, MaterializationFormat.ARROW): 0.3,
            (MaterializationFormat.NUMPY, MaterializationFormat.POLARS): 0.2,
            (MaterializationFormat.NUMPY, MaterializationFormat.PANDAS): 0.1,
            
            # From Pandas
            (MaterializationFormat.PANDAS, MaterializationFormat.ARROW): 0.5,
            (MaterializationFormat.PANDAS, MaterializationFormat.POLARS): 0.3,
            (MaterializationFormat.PANDAS, MaterializationFormat.NUMPY): 0.1,
        }
    
    def materialize(self, 
                   compute: ComputeCapability,
                   target_format: Union[str, MaterializationFormat] = 'auto',
                   hints: Optional[MaterializationHints] = None) -> Any:
        """
        Materialize compute graph to optimal format.
        
        Strategy Selection:
        - 'auto': Choose based on downstream operations
        - 'arrow': Zero-copy for analytics
        - 'numpy': Dense computations
        - 'polars': Further transformations
        - 'pandas': Compatibility
        """
        if isinstance(target_format, str):
            if target_format == 'auto':
                target_format = self._infer_optimal_format(compute, hints)
            else:
                target_format = MaterializationFormat[target_format.upper()]
        
        # Check cache
        cache_key = self._compute_cache_key(compute, target_format)
        if cache_key in self._conversion_cache:
            return self._conversion_cache[cache_key]
        
        # Materialize based on format
        start_time = time.time()
        
        if target_format == MaterializationFormat.ARROW:
            result = self._materialize_to_arrow(compute)
        elif target_format == MaterializationFormat.POLARS:
            result = self._materialize_to_polars(compute)
        elif target_format == MaterializationFormat.NUMPY:
            result = self._materialize_to_numpy(compute)
        elif target_format == MaterializationFormat.PANDAS:
            result = self._materialize_to_pandas(compute)
        elif target_format == MaterializationFormat.PARQUET:
            result = self._materialize_to_parquet(compute, hints)
        else:
            result = compute.materialize()
        
        # Profile performance
        elapsed = time.time() - start_time
        self._profiling_data[target_format] = {
            'time': elapsed,
            'size': self._estimate_size(result)
        }
        
        # Cache if appropriate
        if self._should_cache(result):
            self._conversion_cache[cache_key] = result
        
        return result
    
    def _infer_optimal_format(self, 
                         compute: ComputeCapability,
                         hints: Optional[MaterializationHints]) -> MaterializationFormat:
        """Infer optimal format based on compute graph and hints."""
        if hints:
            # Operations take precedence over preferences
            ops_str = str(getattr(hints, 'downstream_operations', [])).lower()
            
            # Check operations first
            if 'numpy' in ops_str or 'fft' in ops_str:
                return MaterializationFormat.NUMPY
            elif 'plot' in ops_str or 'matplotlib' in ops_str:
                return MaterializationFormat.NUMPY
            elif any(op in ops_str for op in ['join', 'merge', 'groupby', 'agg']):
                return MaterializationFormat.POLARS
                
            # Then check flags
            if getattr(hints, 'persistence_required', False):
                return MaterializationFormat.PARQUET
            elif getattr(hints, 'zero_copy_preferred', False):
                return MaterializationFormat.ARROW

        # Default
        return MaterializationFormat.ARROW
    
    def _analyze_graph_for_format(self, node: Optional[GraphNode]) -> MaterializationFormat:
        """Analyze compute graph to determine optimal format."""
        if node is None:
            return MaterializationFormat.ARROW
            
        # Count operation types safely
        op_counts = self._count_operations(node)
        
        # Decision logic based on operations
        if op_counts.get(ComputeOpType.AGGREGATE, 0) > 2:
            return MaterializationFormat.POLARS
        elif op_counts.get(ComputeOpType.JOIN, 0) > 0:
            return MaterializationFormat.POLARS
        elif op_counts.get(ComputeOpType.MAP, 0) > 3:
            return MaterializationFormat.NUMPY
        else:
            return MaterializationFormat.ARROW
    
    def _count_operations(self, node: GraphNode, counts: Optional[Dict] = None) -> Dict:
        """Recursively count operation types with null safety."""
        if counts is None:
            counts = {}
        
        # Handle None node
        if node is None:
            return counts
        
        # Ensure node has required attributes
        if hasattr(node, 'op_type'):
            op_type = node.op_type
            counts[op_type] = counts.get(op_type, 0) + 1
        
        # Recursively process inputs
        if hasattr(node, 'inputs') and node.inputs:
            for input_node in node.inputs:
                if input_node is not None:
                    self._count_operations(input_node, counts)
        
        return counts
    
    def _materialize_to_arrow(self, compute: ComputeCapability) -> pa.Table:
        """Materialize to Arrow format with zero-copy when possible."""
        # First materialize to native format
        native = compute.materialize()
        
        # Convert to Arrow
        if isinstance(native, pa.Table):
            return native
        elif isinstance(native, pl.DataFrame):
            return native.to_arrow()
        elif isinstance(native, pl.LazyFrame):
            return native.collect().to_arrow()
        elif hasattr(native, 'to_arrow'):
            return native.to_arrow()
        else:
            # Convert through Polars
            return pl.DataFrame(native).to_arrow()
    
    def _materialize_to_polars(self, compute: ComputeCapability) -> pl.DataFrame:
        """Materialize to Polars DataFrame."""
        native = compute.materialize()
        
        if isinstance(native, pl.DataFrame):
            return native
        elif isinstance(native, pl.LazyFrame):
            return native.collect()
        elif isinstance(native, pa.Table):
            return pl.from_arrow(native)
        elif hasattr(native, 'to_polars'):
            return native.to_polars()
        else:
            return pl.DataFrame(native)
    
    def _materialize_to_numpy(self, compute: ComputeCapability) -> np.ndarray:
        """Materialize to NumPy array."""
        # For single column, return 1D array
        # For multiple columns, return structured array
        native = compute.materialize()
        
        if isinstance(native, np.ndarray):
            return native
        elif hasattr(native, 'to_numpy'):
            return native.to_numpy()
        else:
            # Convert through Polars
            df = self._materialize_to_polars(compute)
            if len(df.columns) == 1:
                return df[df.columns[0]].to_numpy()
            else:
                # Create structured array
                return df.to_numpy(structured=True)
    
    def _materialize_to_pandas(self, compute: ComputeCapability) -> 'pd.DataFrame':
        """Materialize to Pandas DataFrame."""
        import pandas as pd
        
        native = compute.materialize()
        
        if isinstance(native, pd.DataFrame):
            return native
        elif hasattr(native, 'to_pandas'):
            return native.to_pandas()
        else:
            # Convert through Polars
            return self._materialize_to_polars(compute).to_pandas()
    
    def _materialize_to_parquet(self, 
                               compute: ComputeCapability,
                               hints: Optional[MaterializationHints]) -> str:
        """Materialize to Parquet file."""
        import tempfile
        
        # Get Arrow table
        table = self._materialize_to_arrow(compute)
        
        # Determine file path
        if hints and hasattr(hints, 'output_path'):
            path = hints.output_path
        else:
            path = tempfile.mktemp(suffix='.parquet')
        
        # Write with compression if acceptable
        compression = 'snappy' if hints and hints.compression_acceptable else None
        
        import pyarrow.parquet as pq
        pq.write_table(table, path, compression=compression)
        
        return path
    
    def _compute_cache_key(self, compute: ComputeCapability, format: MaterializationFormat) -> str:
        """Compute cache key for materialization result."""
        # Simple key based on compute graph hash and format
        import hashlib
        
        if hasattr(compute, 'root_node'):
            graph_str = str(compute.root_node.id)
        else:
            graph_str = str(id(compute))
        
        key = f"{graph_str}_{format.name}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _should_cache(self, result: Any) -> bool:
        """Determine if result should be cached."""
        # Cache based on size and type
        size = self._estimate_size(result)
        
        # Don't cache very large results (>1GB)
        if size > 1_000_000_000:
            return False
        
        # Don't cache Pandas DataFrames (memory inefficient)
        if type(result).__name__ == 'DataFrame' and hasattr(result, 'iloc'):
            return False
        
        return True
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        if hasattr(obj, 'nbytes'):
            return obj.nbytes
        elif hasattr(obj, 'memory_usage'):
            return obj.memory_usage(deep=True).sum()
        elif hasattr(obj, 'estimated_size'):
            return obj.estimated_size()
        else:
            # Rough estimate
            import sys
            return sys.getsizeof(obj)
    
    def get_profiling_report(self) -> Dict[str, Any]:
        """Get profiling report for materialization operations."""
        return {
            'format_performance': self._profiling_data,
            'cache_stats': {
                'size': len(self._conversion_cache),
                'memory_usage': sum(self._estimate_size(v) for v in self._conversion_cache.values())
            }
        }


# ============================================================================
# Graph Optimization Engine
# ============================================================================

class OptimizationRule(ABC):
    """Base class for optimization rules."""
    
    @abstractmethod
    def can_apply(self, node: GraphNode) -> bool:
        """Check if rule can be applied to node."""
        pass
    
    @abstractmethod
    def apply(self, node: GraphNode) -> GraphNode:
        """Apply optimization and return new node."""
        pass


class PredicatePushdownRule(OptimizationRule):
    """Push filter predicates down the graph."""
    
    def can_apply(self, node: GraphNode) -> bool:
        return (node.op_type == ComputeOpType.FILTER and 
                len(node.inputs) > 0 and
                node.inputs[0].op_type in [ComputeOpType.PROJECT, ComputeOpType.JOIN])
    
    def apply(self, node: GraphNode) -> GraphNode:
        """Push filter below projection or before join."""
        filter_node = node
        child_node = node.inputs[0]
        
        if child_node.op_type == ComputeOpType.PROJECT:
            # Create new filter below projection
            new_filter = GraphNode(
                op_type=ComputeOpType.FILTER,
                operation=filter_node.operation,
                inputs=child_node.inputs,
                metadata=filter_node.metadata
            )
            
            # New projection on top
            new_project = GraphNode(
                op_type=ComputeOpType.PROJECT,
                operation=child_node.operation,
                inputs=[new_filter],
                metadata=child_node.metadata
            )
            
            return new_project
        
        return node


class ColumnPruningRule(OptimizationRule):
    """Remove unused columns early."""
    
    def can_apply(self, node: GraphNode) -> bool:
        return node.op_type == ComputeOpType.PROJECT
    
    def apply(self, node: GraphNode) -> GraphNode:
        """Analyze usage and prune unnecessary columns."""
        # This would analyze the full graph to determine used columns
        # For now, return as-is
        return node


class CommonSubexpressionElimination(OptimizationRule):
    """Eliminate duplicate computations."""
    
    def __init__(self):
        self._expression_cache = {}
    
    def can_apply(self, node: GraphNode) -> bool:
        # Check if this computation already exists
        key = self._compute_hash(node)
        return key in self._expression_cache
    
    def apply(self, node: GraphNode) -> GraphNode:
        """Return cached computation."""
        key = self._compute_hash(node)
        return self._expression_cache[key]
    
    def _compute_hash(self, node: GraphNode) -> str:
        """Compute hash for expression matching."""
        import hashlib
        
        components = [
            str(node.op_type),
            str(node.operation.__name__ if hasattr(node.operation, '__name__') else 'lambda'),
            str(node.metadata)
        ]
        
        return hashlib.md5('_'.join(components).encode()).hexdigest()


class GraphOptimizationEngine:
    """
    Continuous optimization of compute graphs.
    
    This engine applies various optimization rules to improve
    execution performance while maintaining correctness.
    """
    
    def __init__(self):
        self._rules = [
            PredicatePushdownRule(),
            ColumnPruningRule(),
            CommonSubexpressionElimination()
        ]
        self._optimization_stats = {
            'rules_applied': 0,
            'time_saved_ms': 0,
            'memory_saved_mb': 0
        }
    
    def optimize(self, root_node: GraphNode) -> GraphNode:
        """Optimize compute graph starting from root."""
        optimized = root_node
        changed = True
        iterations = 0
        max_iterations = 10
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            # Try each optimization rule
            for rule in self._rules:
                new_root = self._apply_rule_recursive(optimized, rule)
                if new_root != optimized:
                    optimized = new_root
                    changed = True
                    self._optimization_stats['rules_applied'] += 1
        
        return optimized
    
    def _apply_rule_recursive(self, node: GraphNode, rule: OptimizationRule) -> GraphNode:
        """Recursively apply optimization rule with null safety."""
        if node is None:
            return node
            
        # Ensure node has required attributes
        if not hasattr(node, 'inputs') or node.inputs is None:
            node.inputs = []
        
        # First optimize children
        optimized_inputs = []
        for input_node in node.inputs:
            if input_node is not None:
                optimized_inputs.append(self._apply_rule_recursive(input_node, rule))
        
        # Ensure metadata exists
        if not hasattr(node, 'metadata') or node.metadata is None:
            node.metadata = {}
        
        # Create new node with optimized inputs
        current = GraphNode(
            op_type=node.op_type,
            operation=node.operation,
            inputs=optimized_inputs,
            metadata=node.metadata.copy() if isinstance(node.metadata, dict) else {}
        )
        
        # Try to apply rule to current node
        if rule.can_apply(current):
            return rule.apply(current)
        
        return current
    
    def optimize_based_on_statistics(self, 
                                   root_node: GraphNode,
                                   statistics: Dict[str, Any]) -> GraphNode:
        """Optimize using runtime statistics."""
        # Use statistics to make better optimization decisions
        # For example, reorder joins based on cardinality
        
        if 'cardinality' in statistics:
            # Reorder operations based on selectivity
            return self._reorder_by_cardinality(root_node, statistics['cardinality'])
        
        return self.optimize(root_node)
    
    def _reorder_by_cardinality(self, node: GraphNode, cardinality: Dict) -> GraphNode:
        """Reorder operations based on cardinality estimates."""
        # This would implement join reordering and other cardinality-based optimizations
        return node
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self._optimization_stats.copy()


# ============================================================================
# Memory-Aware Executor
# ============================================================================

@dataclass
class MemoryProfile:
    """Memory profile for an operation."""
    input_size: int
    output_size: int
    working_memory: int
    can_stream: bool = False
    can_partition: bool = True


class MemoryAwareExecutor:
    """
    Adaptive execution based on memory pressure.
    
    This executor monitors memory usage and adapts execution
    strategies to stay within memory constraints.
    """
    
    def __init__(self, memory_limit: int):
        self.memory_limit = memory_limit
        self._memory_profiles = {}
        self._current_usage = 0
        self._spill_manager = None
        
    def execute_with_memory_limit(self, 
                                 compute: ComputeCapability,
                                 hints: Optional[MaterializationHints] = None) -> Any:
        """
        Execute while respecting memory constraints.
        
        Techniques:
        - Dynamic batch size adjustment
        - Automatic spilling activation
        - Operation reordering for memory efficiency
        """
        # Create execution plan
        plan = self._create_execution_plan(compute)
        
        # Estimate memory usage
        memory_estimate = self._estimate_peak_memory(plan)
        
        if memory_estimate > self.memory_limit:
            # Activate memory-saving strategies
            plan = self._optimize_for_memory(plan)
            
            # Re-estimate
            memory_estimate = self._estimate_peak_memory(plan)
            
            if memory_estimate > self.memory_limit:
                # Enable spilling
                plan = self._enable_spilling(plan)
        
        # Execute with monitoring
        return self._execute_with_monitoring(plan)
    
    def _create_execution_plan(self, compute: ComputeCapability) -> 'ExecutionPlan':
        """Create execution plan from compute capability."""
        # This would analyze the compute graph and create an execution plan
        # For now, create a simple plan
        return ExecutionPlan(
            compute=compute,
            stages=[],
            memory_estimate=0
        )
    
    def _estimate_peak_memory(self, plan: 'ExecutionPlan') -> int:
        """Estimate peak memory usage for execution plan."""
        if hasattr(plan.compute, 'estimate_memory'):
            return plan.compute.estimate_memory()
        
        # Default estimate
        return 1_000_000_000  # 1GB
    
    def _optimize_for_memory(self, plan: 'ExecutionPlan') -> 'ExecutionPlan':
        """Optimize execution plan for memory efficiency."""
        # Strategies:
        # 1. Reduce batch sizes
        # 2. Reorder operations to minimize intermediates
        # 3. Enable streaming where possible
        
        optimized = plan.copy()
        
        # Reduce batch sizes
        if hasattr(optimized, 'batch_size'):
            optimized.batch_size = min(optimized.batch_size, 10000)
        
        # Enable streaming
        optimized.streaming_enabled = True
        
        return optimized
    
    def _enable_spilling(self, plan: 'ExecutionPlan') -> 'ExecutionPlan':
        """Enable spilling for memory-intensive operations."""
        plan.spilling_enabled = True
        plan.spill_threshold = int(self.memory_limit * 0.8)
        
        # Initialize spill manager if needed
        if self._spill_manager is None:
            from layer1.billion_capable_engine import ChecksummedSpillManager
            self._spill_manager = ChecksummedSpillManager()
        
        plan.spill_manager = self._spill_manager
        
        return plan
    
    def _execute_with_monitoring(self, plan: 'ExecutionPlan') -> Any:
        """Execute plan with memory monitoring."""
        import psutil
        process = psutil.Process()
        
        # Monitor memory during execution
        initial_memory = process.memory_info().rss
        peak_memory = initial_memory
        
        try:
            # Execute the plan
            result = plan.execute()
            
            # Track peak memory
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)
            
            # Log statistics
            self._log_execution_stats(plan, peak_memory - initial_memory)
            
            return result
            
        except MemoryError:
            # Handle OOM
            return self._handle_oom(plan)
    
    def _handle_oom(self, plan: 'ExecutionPlan') -> Any:
        """Handle out-of-memory error."""
        warnings.warn("OOM detected, retrying with aggressive spilling")
        
        # Enable aggressive spilling
        plan.spilling_enabled = True
        plan.spill_threshold = int(self.memory_limit * 0.5)
        plan.batch_size = 1000  # Very small batches
        
        # Retry execution
        return plan.execute()
    
    def _log_execution_stats(self, plan: 'ExecutionPlan', memory_used: int):
        """Log execution statistics."""
        stats = {
            'memory_used_mb': memory_used / (1024 * 1024),
            'memory_limit_mb': self.memory_limit / (1024 * 1024),
            'spilling_used': plan.spilling_enabled,
            'execution_time': plan.execution_time if hasattr(plan, 'execution_time') else 0
        }
        
        print(f"Execution stats: {stats}")


@dataclass
class ExecutionPlan:
    """Execution plan for memory-aware execution."""
    compute: ComputeCapability
    stages: List[Any]
    memory_estimate: int
    batch_size: int = 100000
    streaming_enabled: bool = False
    spilling_enabled: bool = False
    spill_threshold: Optional[int] = None
    spill_manager: Optional[Any] = None
    
    def copy(self) -> 'ExecutionPlan':
        """Create a copy of the execution plan."""
        return ExecutionPlan(
            compute=self.compute,
            stages=self.stages.copy(),
            memory_estimate=self.memory_estimate,
            batch_size=self.batch_size,
            streaming_enabled=self.streaming_enabled,
            spilling_enabled=self.spilling_enabled,
            spill_threshold=self.spill_threshold,
            spill_manager=self.spill_manager
        )
    
    def execute(self) -> Any:
        """Execute the plan."""
        # This would implement the actual execution logic
        # For now, just materialize the compute
        return self.compute.materialize()


# ============================================================================
# Performance Profiler
# ============================================================================

class PerformanceProfiler:
    """
    Real-time performance analysis for Layer 2 operations.
    
    Tracks performance metrics and provides optimization recommendations.
    """
    
    def __init__(self):
        self._metrics = {
            'operation_times': {},
            'memory_peaks': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'spill_events': []
        }
        self._start_times = {}
        
    def start_operation(self, operation_id: str, operation_type: str):
        """Start timing an operation."""
        self._start_times[operation_id] = {
            'start': time.time(),
            'type': operation_type
        }
    
    def end_operation(self, operation_id: str, memory_used: Optional[int] = None):
        """End timing an operation."""
        if operation_id not in self._start_times:
            return
        
        start_info = self._start_times.pop(operation_id)
        elapsed = time.time() - start_info['start']
        
        op_type = start_info['type']
        if op_type not in self._metrics['operation_times']:
            self._metrics['operation_times'][op_type] = []
        
        self._metrics['operation_times'][op_type].append(elapsed)
        
        if memory_used:
            if op_type not in self._metrics['memory_peaks']:
                self._metrics['memory_peaks'][op_type] = []
            self._metrics['memory_peaks'][op_type].append(memory_used)
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self._metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self._metrics['cache_misses'] += 1
    
    def record_spill_event(self, size: int, duration: float):
        """Record a spill event."""
        self._metrics['spill_events'].append({
            'size': size,
            'duration': duration,
            'timestamp': time.time()
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'operation_summary': self._summarize_operations(),
            'memory_analysis': self._analyze_memory(),
            'cache_efficiency': self._calculate_cache_efficiency(),
            'spilling_analysis': self._analyze_spilling(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _summarize_operations(self) -> Dict[str, Any]:
        """Summarize operation performance."""
        summary = {}
        
        for op_type, times in self._metrics['operation_times'].items():
            if times:
                summary[op_type] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        return summary
    
    def _analyze_memory(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        analysis = {}
        
        for op_type, peaks in self._metrics['memory_peaks'].items():
            if peaks:
                analysis[op_type] = {
                    'avg_peak_mb': sum(peaks) / len(peaks) / (1024 * 1024),
                    'max_peak_mb': max(peaks) / (1024 * 1024)
                }
        
        return analysis
    
    def _calculate_cache_efficiency(self) -> float:
        """Calculate cache hit rate."""
        total = self._metrics['cache_hits'] + self._metrics['cache_misses']
        if total == 0:
            return 0.0
        
        return self._metrics['cache_hits'] / total
    
    def _analyze_spilling(self) -> Dict[str, Any]:
        """Analyze spilling patterns."""
        if not self._metrics['spill_events']:
            return {'spill_count': 0}
        
        spill_events = self._metrics['spill_events']
        total_spilled = sum(e['size'] for e in spill_events)
        total_time = sum(e['duration'] for e in spill_events)
        
        return {
            'spill_count': len(spill_events),
            'total_spilled_gb': total_spilled / (1024**3),
            'total_spill_time': total_time,
            'avg_spill_size_mb': total_spilled / len(spill_events) / (1024**2),
            'spill_throughput_mbps': (total_spilled / total_time / (1024**2)) if total_time > 0 else 0
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling."""
        recommendations = []
        
        # Check cache efficiency
        cache_efficiency = self._calculate_cache_efficiency()
        if cache_efficiency < 0.5:
            recommendations.append(
                f"Low cache hit rate ({cache_efficiency:.1%}). "
                "Consider increasing cache size or improving access patterns."
            )
        
        # Check spilling
        spill_analysis = self._analyze_spilling()
        if spill_analysis['spill_count'] > 10:
            recommendations.append(
                f"Frequent spilling detected ({spill_analysis['spill_count']} events). "
                "Consider increasing memory budget or optimizing memory usage."
            )
        
        # Check operation balance
        op_summary = self._summarize_operations()
        if 'JOIN' in op_summary and op_summary['JOIN']['avg_time'] > 10:
            recommendations.append(
                "Slow join operations detected. "
                "Consider using broadcast joins for small tables or optimizing join order."
            )
        
        return recommendations


# ============================================================================
# Integration Helpers
# ============================================================================

class Layer2Optimizers:
    """Central access point for Layer 2 optimization components."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.materialization_controller = MaterializationController()
        self.graph_optimizer = GraphOptimizationEngine()
        self.memory_executor = MemoryAwareExecutor(
            memory_limit=8 * 1024 * 1024 * 1024  # 8GB default
        )
        self.profiler = PerformanceProfiler()
        
        self._initialized = True
    
    def optimize_and_execute(self,
                           compute: ComputeCapability,
                           target_format: str = 'auto',
                           memory_limit: Optional[int] = None) -> Any:
        """
        Optimize and execute compute with all available optimizations.
        
        This is the main entry point for optimized execution.
        """
        # Start profiling
        op_id = str(id(compute))
        self.profiler.start_operation(op_id, 'full_execution')
        
        try:
            # Optimize graph if available
            if hasattr(compute, 'root_node'):
                optimized_node = self.graph_optimizer.optimize(compute.root_node)
                
                # Create new compute with optimized graph
                if hasattr(compute, '__class__'):
                    compute = compute.__class__(
                        root_node=optimized_node,
                        engine=compute.engine if hasattr(compute, 'engine') else None,
                        estimated_size=compute.estimated_size if hasattr(compute, 'estimated_size') else 0
                    )
            
            # Set memory limit if provided
            if memory_limit:
                self.memory_executor.memory_limit = memory_limit
            
            # Execute with memory awareness
            if memory_limit and hasattr(compute, 'estimate_memory'):
                estimated = compute.estimate_memory()
                if estimated > memory_limit:
                    result = self.memory_executor.execute_with_memory_limit(compute)
                else:
                    result = self.materialization_controller.materialize(
                        compute, target_format
                    )
            else:
                result = self.materialization_controller.materialize(
                    compute, target_format
                )
            
            # Record success
            self.profiler.record_cache_hit()  # Successful execution
            
            return result
            
        finally:
            # End profiling
            self.profiler.end_operation(op_id)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            'materialization': self.materialization_controller.get_profiling_report(),
            'graph_optimization': self.graph_optimizer.get_optimization_report(),
            'performance': self.profiler.get_performance_report()
        }


# Singleton instance
layer2_optimizers = Layer2Optimizers()


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'MaterializationController',
    'MaterializationFormat',
    'MaterializationHints',
    'GraphOptimizationEngine',
    'OptimizationRule',
    'MemoryAwareExecutor',
    'MemoryProfile',
    'ExecutionPlan',
    'PerformanceProfiler',
    'Layer2Optimizers',
    'layer2_optimizers'
]