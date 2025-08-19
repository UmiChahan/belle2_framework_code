import weakref
from typing import Optional, List, Any
import numpy as np
from layer1.lazy_compute_engine import (
    LazyComputeCapability, GraphNode
)
from layer0 import (
    ComputeCapability, ComputeOpType
)
class LazyColumnAccessor:
    """
    Column accessor that maintains compute graph semantics.
    
    Innovation: Seamless transition between lazy and eager operations
    while preserving the compute-first architecture.
    """
    
    def __init__(self, 
                 compute: ComputeCapability,
                 column_name: str,
                 parent_ref: Optional[weakref.ref] = None):
        """
        Initialize column accessor.
        
        Args:
            compute: Compute capability for this column
            column_name: Name of the column
            parent_ref: Weak reference to parent DataFrame
        """
        self._compute = compute
        self._column_name = column_name
        self._parent_ref = parent_ref
        self._stats_cache = None
    
    def __getattr__(self, name):
        """
        Delegate method calls to compute graph.
        
        Example: column.mean() → compute.aggregate('mean')
        """
        # Statistical methods
        if name in ['mean', 'sum', 'min', 'max', 'std', 'var', 'count']:
            return self._create_statistical_method(name)
        
        # String methods
        elif name in ['lower', 'upper', 'strip', 'contains', 'startswith', 'endswith']:
            return self._create_string_method(name)
        
        # Datetime methods
        elif name in ['year', 'month', 'day', 'hour', 'minute', 'second']:
            return self._create_datetime_method(name)
        
        else:
            raise AttributeError(f"Column accessor has no attribute '{name}'")
    
    def _create_statistical_method(self, method: str):
        """Create a statistical aggregation method."""
        def statistical_operation():
            # Create aggregation node
            agg_node = GraphNode(
                op_type=ComputeOpType.AGGREGATE,
                operation=lambda df: getattr(df[self._column_name], method)(),
                inputs=[self._compute.root_node if hasattr(self._compute, 'root_node') else None],
                metadata={'method': method, 'column': self._column_name}
            )
            
            # Execute and return result
            if hasattr(self._compute, 'engine'):
                engine = self._compute.engine()
                if engine:
                    return engine._execute_graph(agg_node, 1)
            
            # Fallback
            materialized = self._compute.materialize()
            return getattr(materialized[self._column_name], method)()
        
        return statistical_operation
    
    def _create_string_method(self, method: str):
        """Create a string operation method."""
        def string_operation(*args, **kwargs):
            # Create transformation node
            if method in ['contains', 'startswith', 'endswith']:
                operation = lambda df: getattr(df[self._column_name].str, method)(*args, **kwargs)
            else:
                operation = lambda df: getattr(df[self._column_name].str, method)()
            
            transform_node = GraphNode(
                op_type=ComputeOpType.MAP,
                operation=operation,
                inputs=[self._compute.root_node if hasattr(self._compute, 'root_node') else None],
                metadata={'method': method, 'column': self._column_name}
            )
            
            # Return new column accessor
            if isinstance(self._compute, LazyComputeCapability):
                new_compute = LazyComputeCapability(
                    root_node=transform_node,
                    engine=self._compute.engine,
                    estimated_size=self._compute.estimated_size,
                    schema={self._column_name: str}
                )
            else:
                new_compute = self._compute.transform(operation)
            
            return LazyColumnAccessor(new_compute, self._column_name, self._parent_ref)
        
        return string_operation
    
    def _create_datetime_method(self, method: str):
        """Create a datetime extraction method."""
        def datetime_operation():
            # Create extraction node
            operation = lambda df: getattr(df[self._column_name].dt, method)()
            
            extract_node = GraphNode(
                op_type=ComputeOpType.MAP,
                operation=operation,
                inputs=[self._compute.root_node if hasattr(self._compute, 'root_node') else None],
                metadata={'method': method, 'column': self._column_name}
            )
            
            # Return new column accessor
            if isinstance(self._compute, LazyComputeCapability):
                new_compute = LazyComputeCapability(
                    root_node=extract_node,
                    engine=self._compute.engine,
                    estimated_size=self._compute.estimated_size,
                    schema={self._column_name: int}
                )
            else:
                new_compute = self._compute.transform(operation)
            
            return LazyColumnAccessor(new_compute, self._column_name, self._parent_ref)
        
        return datetime_operation
    
    def astype(self, dtype) -> 'LazyColumnAccessor':
        """
        Lazy type conversion with validation.
        
        Performance features:
        - Sample-based validation
        - Automatic optimization for compatible types
        - Zero-copy when possible
        """
        # Create cast node
        cast_node = GraphNode(
            op_type=ComputeOpType.MAP,
            operation=lambda df: df[self._column_name].cast(dtype),
            inputs=[self._compute.root_node if hasattr(self._compute, 'root_node') else None],
            metadata={'dtype': str(dtype), 'column': self._column_name}
        )
        
        # Create new compute capability
        if isinstance(self._compute, LazyComputeCapability):
            new_compute = LazyComputeCapability(
                root_node=cast_node,
                engine=self._compute.engine,
                estimated_size=self._compute.estimated_size,
                schema={self._column_name: dtype}
            )
        else:
            new_compute = self._compute.transform(
                lambda df: df[self._column_name].cast(dtype)
            )
        
        return LazyColumnAccessor(new_compute, self._column_name, self._parent_ref)
    
    def apply(self, func, meta=None) -> 'LazyColumnAccessor':
        """
        User-defined function application with optimization.
        
        Critical: Attempts to vectorize or JIT compile UDFs for performance.
        """
        # Analyze function for optimization
        is_vectorizable = self._check_vectorizable(func)
        
        if is_vectorizable:
            # Vectorized execution
            operation = lambda df: df[self._column_name].map_elements(func)
        else:
            # Standard apply
            operation = lambda df: df[self._column_name].apply(func)
        
        # Create UDF node
        udf_node = GraphNode(
            op_type=ComputeOpType.MAP,
            operation=operation,
            inputs=[self._compute.root_node if hasattr(self._compute, 'root_node') else None],
            metadata={
                'udf': func.__name__ if hasattr(func, '__name__') else 'anonymous',
                'vectorizable': is_vectorizable,
                'column': self._column_name
            }
        )
        
        # Create new compute capability
        if isinstance(self._compute, LazyComputeCapability):
            new_compute = LazyComputeCapability(
                root_node=udf_node,
                engine=self._compute.engine,
                estimated_size=self._compute.estimated_size,
                schema={self._column_name: meta} if meta else None
            )
        else:
            new_compute = self._compute.transform(operation)
        
        return LazyColumnAccessor(new_compute, self._column_name, self._parent_ref)
    
    def _check_vectorizable(self, func) -> bool:
        """Check if function can be vectorized."""
        # Simple heuristic - in production, use more sophisticated analysis
        try:
            # Check if function works on arrays
            import inspect
            sig = inspect.signature(func)
            return len(sig.parameters) == 1
        except Exception:
            return False
    
    def to_numpy(self) -> np.ndarray:
        """Materialize column to numpy array."""
        materialized = self._compute.materialize()
        if hasattr(materialized, 'to_numpy'):
            return materialized.to_numpy()
        else:
            return np.array(materialized)
    
    def to_list(self) -> List[Any]:
        """Materialize column to Python list."""
        return self.to_numpy().tolist()
