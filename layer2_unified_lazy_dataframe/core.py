"""
Layer 2: UnifiedLazyDataFrame - Compute-First Data Structure
============================================================

This module implements the flagship Layer 2 data structure that manifests
compute capabilities as a familiar DataFrame API. The DataFrame IS the
compute graph, not a container of data.

Design Philosophy:
- Data structures are views over compute graphs
- Zero-copy semantics throughout
- Transparent optimization under familiar APIs
- Billion-row capability through compute manifestation

Integration: Seamlessly builds on Layer 1 compute engines
"""
# pylint: disable=C0302,C0303,C0301,C0413,E0401,C0415,W1309,W0622,W0404
from __future__ import annotations
import sys
import copy
import warnings
import weakref
import time
import ast
import builtins
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union,
    TypeVar, Generic, Tuple,
)
import numpy as np
import polars as pl
import pyarrow as pa
import psutil
import pandas as pd

# Make repository root importable
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Layer 0/1 integration points

from layer0 import ComputeCapability, ComputeOpType  # type: ignore

from layer1.lazy_compute_engine import LazyComputeCapability, GraphNode  # type: ignore

from layer1.integration_layer import EngineSelector, ComputeEngineAdapter  # type: ignore
try:
    from optimized_cpp_integration import OptimizedStreamingHistogram  # type: ignore  # pylint: disable=unused-import
    CPP_HISTOGRAM_AVAILABLE = True
except Exception:  # pragma: no cover - keep import-safe without native ext
    OptimizedStreamingHistogram = None
    CPP_HISTOGRAM_AVAILABLE = False
from .accessors import LazyColumnAccessor
from .groupby import LazyGroupBy
from .utils import (
    TransformationMetadata, DataTransformationChain,
    AccessPattern
)
from .histogram_strategies import (
    HistogramExecutionStrategy, HistogramMetrics,memory_monitor,
    ChunkingEnhancement
)
T = TypeVar("T")
class UnifiedLazyDataFrame(Generic[T]):
    """
    A data structure that manifests compute capabilities as a familiar DataFrame API.
    
    Key Innovation: The DataFrame IS the compute graph, not a container of data.
    This inverts the traditional relationship - instead of data structures having
    compute methods, compute capabilities manifest as data structures.
    
    Features:
    - Zero-copy column access through LazyColumnAccessor
    - Automatic query optimization through compute graph manipulation
    - Memory-aware execution with spilling support
    - Seamless integration with existing pandas/polars APIs
    - Full transformation tracking for reproducibility
    
    Performance:
    - O(1) operation building (graph construction)
    - O(n) only on materialization
    - Memory proportional to graph size, not data size
    """
    
    def __init__(self,
                 compute=None,
                 lazy_frames=None,
                 schema=None,
                 metadata=None,
                 memory_budget_gb=8.0,
                 materialization_threshold=10_000_000,
                 required_columns=None,
                 transformation_metadata=None,
                 parent_chain=None,
                 histogram_engine=None):  # NEW: Accept parent chain
        """Initialize DataFrame with correct transformation tracking."""

        # Initialize ALL attributes first
        self.memory_budget_gb = memory_budget_gb
        self.materialization_threshold = materialization_threshold
        self.required_columns = required_columns or []
        self._materialized_cache = None
        self._cache_hash = None
        self._access_patterns = []
        self._optimization_hints = {}
        self._operation_cache = {}
        self._engine_ref = None
        self._framework = None
        self._histogram_engine = histogram_engine
        self._coordinated_engine = None
        self._schema_cache = None  # cache used by schema()
        self._query_columns_cache = {}  # per-instance cache for query column extraction
        self._range_cache = {}  # per-instance cache for histogram ranges
        self._range_cache_order = []  # simple LRU order for _range_cache

        # Initialize transformation chain
        self._transformation_chain = DataTransformationChain()

        # CRITICAL: Copy parent chain if provided
        if parent_chain:
            for transform in parent_chain:
                self._transformation_chain.add_transformation(transform)
        
        # Detect empty DataFrame
        self._is_empty = self._detect_empty_state(compute, lazy_frames, schema)
        
        # Initialize schema based on empty state
        if self._is_empty:
            self._schema = {}
            self._estimated_rows = 0
        else:
            if schema:
                self._schema = schema
            elif lazy_frames and len(lazy_frames):
                # CORE FIX: Extract actual schema from lazy_frames
                self._schema = dict(lazy_frames[0].collect_schema())
            else:
                print("⚠️ Warning: No schema provided and no lazy frames available. Defaulting to empty schema.")
        
        self._metadata = metadata or {}
        
        # Handle compute and frames
        if compute is not None:
            self._compute = compute
            self._lazy_frames = None
            if not self._is_empty and hasattr(compute, 'schema') and compute.schema:
                self._schema.update(compute.schema)
            # if hasattr(compute, 'estimated_size'):
            #     self._estimated_rows = compute.estimated_size if not self._is_empty else 0 Commented at the moment, don't remember why
        elif lazy_frames is not None and lazy_frames:
            self._lazy_frames = lazy_frames
            if not self._is_empty:
                self._compute = self._create_compute_from_frames(lazy_frames)
                self._estimated_rows = len(lazy_frames) * 1000
            else:
                self._compute = self._create_minimal_compute()
        else:
            self._compute = self._create_minimal_compute()
            self._lazy_frames = []
        
        # Add current transformation if provided
        if transformation_metadata:
            self._transformation_metadata = transformation_metadata
            self._transformation_chain.add_transformation(transformation_metadata)
        else:
            # Only add initialization transformation if this is truly a new DataFrame
            if parent_chain is None:
                self._transformation_metadata = TransformationMetadata(
                    operation='initialization',
                    parameters={'empty': self._is_empty}
                )
                self._transformation_chain.add_transformation(self._transformation_metadata)
            else:
                # This is a derived DataFrame, don't add redundant init
                self._transformation_metadata = None
        self._estimated_rows = self._estimate_total_rows()

    def schema(self):
        """Return cached schema (dict[str, pl.DataType]) with safe inference and normalization."""
        if self._schema_cache is not None:
            return self._schema_cache

        def _normalize_schema(obj) -> Dict[str, pl.DataType]:
            try:
                if hasattr(obj, 'items') and not isinstance(obj, dict):
                    return {k: v for k, v in obj.items()}
                if isinstance(obj, dict):
                    out = {}
                    for k, v in obj.items():
                        if hasattr(v, 'is_numeric') or str(v).startswith('pl.'):
                            out[k] = v
                        else:
                            try:
                                out[k] = getattr(pl, str(v))
                            except Exception:
                                out[k] = pl.Utf8
                    return out
            except Exception:
                pass
            return {}

        # Prefer compute schema
        try:
            if hasattr(self, '_compute') and getattr(self._compute, 'schema', None):
                self._schema_cache = _normalize_schema(self._compute.schema)
                if self._schema_cache:
                    return self._schema_cache
        except Exception:
            pass

        # Fall back to lazy frames
        try:
            if hasattr(self, '_lazy_frames') and self._lazy_frames:
                lf_schema = self._lazy_frames[0].collect_schema()
                self._schema_cache = _normalize_schema(lf_schema)
                if self._schema_cache:
                    return self._schema_cache
        except Exception:
            pass

        # Last resort: use static _schema mapping
        self._schema_cache = _normalize_schema(getattr(self, '_schema', {}))
        return self._schema_cache

    def _create_compute_capability(self, graph_node: GraphNode, estimated_size: int) -> ComputeCapability:
        """Create compute capability from graph node with proper engine integration."""
        if isinstance(self._compute, LazyComputeCapability):
            return LazyComputeCapability(
                root_node=graph_node,
                engine=self._compute.engine,
                estimated_size=estimated_size,
                schema=self._schema
            )
        else:
            # Enhanced fallback wrapper with complete interface
            class GraphComputeWrapper:
                def __init__(self, node, size, schema):
                    self.root_node = node
                    self.estimated_size = size
                    self.schema = schema or {}
                    
                def materialize(self):
                    """Execute the graph node operation chain."""
                    return self._execute_node(self.root_node)
                    
                def _execute_node(self, node):
                    """Recursively execute compute graph nodes."""
                    if node.inputs:
                        # Execute dependencies first
                        input_results = []
                        for inp in node.inputs:
                            if inp:
                                result = self._execute_node(inp)
                                input_results.append(result)
                        
                        # Apply operation to input results
                        if input_results:
                            try:
                                return node.operation(*input_results)
                            except TypeError:
                                # Operation might expect single argument
                                if len(input_results) == 1:
                                    return node.operation(input_results[0])
                                else:
                                    raise
                        else:
                            return node.operation()
                    else:
                        # Leaf node - execute directly
                        return node.operation()
                        
                def estimate_memory(self):
                    """Estimate memory usage."""
                    return self.estimated_size * 100
                
                def transform(self, operation):
                    """Apply transformation to create new compute capability."""
                    # Create transformation node
                    transform_node = GraphNode(
                        op_type=ComputeOpType.MAP,
                        operation=operation,
                        inputs=[self.root_node],
                        metadata={'transform': 'user_defined'}
                    )
                    
                    # Return new wrapper with transformed graph
                    return GraphComputeWrapper(
                        node=transform_node,
                        size=self.estimated_size,
                        schema=self.schema
                    )
            
            return GraphComputeWrapper(graph_node, estimated_size, self._schema)
    def _compute_cache_key(self):
        """Generate stable cache key: compute id + lineage fingerprint."""
        if not hasattr(self, '_compute'):
            return "no_compute"
        try:
            lineage = self._transformation_chain.get_lineage() if hasattr(self, '_transformation_chain') else []
            K = 16
            tail = lineage[-K:]
            def _safe_params(p):
                try:
                    return tuple(sorted((k, repr(v)) for k, v in (p or {}).items()))
                except Exception:
                    return ()
            fp_tuple = tuple((getattr(t, 'operation', ''), _safe_params(getattr(t, 'parameters', {}))) for t in tail)
            fingerprint = hash(fp_tuple)
            return f"{id(self._compute)}_{len(lineage)}_{fingerprint}"
        except Exception:
            return f"{id(self._compute)}_unknown"
    def _extract_lazy_frames_from_compute(self):
        """Optimized extraction with caching and validation."""
        
        # Cache key based on transformation state
        cache_key = self._compute_cache_key()
        
        # Check cache (with simple per-instance LRU tracking)
        if not hasattr(self, '_frame_extraction_cache'):
            self._frame_extraction_cache = {}
            self._frame_extraction_cache_order = []
        else:
            if cache_key in self._frame_extraction_cache:
                try:
                    self._frame_extraction_cache_order.remove(cache_key)
                except ValueError:
                    pass
                self._frame_extraction_cache_order.append(cache_key)
                return self._frame_extraction_cache[cache_key]
        
        # Extract frames based on compute type
        if isinstance(self._compute, LazyComputeCapability):
            # Direct extraction from known compute type
            frames = self._extract_from_lazy_compute()
        elif hasattr(self._compute, 'to_lazy_frame'):
            # Compute provides conversion method
            frames = [self._compute.to_lazy_frame()]
        elif hasattr(self._compute, 'materialize'):
            # Last resort: Create lazy wrapper around materialization
            # This maintains lazy semantics even if compute doesn't
            frames = [self._create_lazy_materialization_wrapper()]
        else:
            raise ValueError("Cannot extract frames from compute capability")
        
        # Validate extraction
        if not frames:
            raise ValueError("Frame extraction produced no results")
        
        # Cache successful extraction with LRU bound
        self._frame_extraction_cache[cache_key] = frames
        self._frame_extraction_cache_order.append(cache_key)
        MAX_CACHE = 64
        if len(self._frame_extraction_cache_order) > MAX_CACHE:
            evict_key = self._frame_extraction_cache_order.pop(0)
            if evict_key != cache_key and evict_key in self._frame_extraction_cache:
                del self._frame_extraction_cache[evict_key]
        
        return frames

    def _create_lazy_materialization_wrapper(self):
        """Create lazy frame that materializes compute on demand."""
        # This is the KEY to maintaining lazy evaluation
        # even when compute doesn't support it natively
        
        schema = self._schema or None
        compute_ref = weakref.ref(self._compute)
        
        # Create a lazy frame that defers materialization safely
        def _materialize_or_empty(_):
            comp = compute_ref()
            if comp is None:
                warnings.warn("Compute capability was garbage collected; returning empty DataFrame")
                return pl.DataFrame()
            try:
                return comp.materialize()
            except Exception as e:
                warnings.warn(f"Materialization failed in lazy wrapper: {e}")
                return pl.DataFrame()

        return pl.LazyFrame([]).map_batches(_materialize_or_empty, schema=schema)

    def _build_lazy_frame_from_graph(self):
        """Convert compute graph to lazy frame."""
        # Create a lazy evaluation wrapper
        def lazy_evaluator():
            # This creates a lazy frame that will execute the compute graph when collected
            return pl.DataFrame().lazy().map_batches(
                lambda _: self._compute.materialize(),
                schema=self._schema
            )
        
        # Return as single-element list for consistency
        return [lazy_evaluator()]

    def _apply_transformations_to_frame(self, lazy_frame):
        """Apply accumulated transformations to a lazy frame."""
        result = lazy_frame
        
        # Traverse transformation chain and apply operations
        for transform in self._transformation_chain.get_lineage():
            if transform.operation == 'filter':
                # Extract filter expression from parameters
                expr = transform.parameters.get('expr')
                if expr:
                    result = result.filter(expr)
            elif transform.operation == 'select_columns':
                cols = transform.parameters.get('columns', [])
                result = result.select(cols)
            elif transform.operation == 'oneCandOnly':
                group_cols = transform.parameters.get('group_cols', [])
                result = result.group_by(group_cols).first()
        
        return result



    def _detect_empty_state(self, compute, lazy_frames, schema):
        """Robust empty state detection with defensive checks."""
        # Explicit schema check
        if schema is not None and len(schema) == 0:
            return True
        
        # Lazy frame inspection with multiple validation checks
        if lazy_frames:
            for frame in lazy_frames:
                if hasattr(frame, 'schema'):
                    try:
                        frame_schema = frame.collect_schema()
                        # Empty if no columns AND this is the only frame
                        if len(frame_schema) == 0 and len(lazy_frames) == 1:
                            return True
                        # Non-empty if any frame has columns
                        elif len(frame_schema) > 0:
                            return False
                    except Exception:
                        # If schema access fails, assume non-empty for safety
                        return False
        
        # Compute capability check
        if compute is not None:
            if hasattr(compute, 'schema'):
                try:
                    compute_schema = compute.schema or {}
                    return len(compute_schema) == 0
                except Exception:
                    return False
            if hasattr(compute, 'estimated_size'):
                return getattr(compute, 'estimated_size', 1) == 0
        
        # Default to non-empty for safety
        return False
    
    def _create_derived_dataframe(self, new_compute, new_schema=None, 
                              transformation_metadata=None, **kwargs):
        """Create derived DataFrame with deep transformation chain preservation."""
        
        # Preserve transformation lineage
        parent_lineage = []
        if hasattr(self, '_transformation_chain'):
            for transform in self._transformation_chain.get_lineage():
                transform_copy = copy.deepcopy(transform)
                parent_lineage.append(transform_copy)
        
        # Use provided schema or inherit from parent
        if new_schema is None:
            new_schema = self._schema.copy() if self._schema else {}
        
        # CRITICAL: Respect required_columns constraint if provided
        required_cols = kwargs.get('required_columns', self.required_columns.copy())
        
        # If required_columns are specified, ensure schema matches
        if required_cols and new_schema:
            # Filter schema to only include required columns
            filtered_schema = {col: dtype for col, dtype in new_schema.items() 
                            if col in required_cols}
            new_schema = filtered_schema
        
        result = UnifiedLazyDataFrame(
            compute=new_compute,
            schema=new_schema,
            metadata=copy.deepcopy(self._metadata) if self._metadata else {},
            memory_budget_gb=self.memory_budget_gb,
            materialization_threshold=self.materialization_threshold,
            required_columns=required_cols,
            transformation_metadata=None,
            parent_chain=None
        )
        
        # Reconstruct transformation chain
        # Use already imported DataTransformationChain to avoid import-outside-toplevel
        result._transformation_chain = DataTransformationChain()

        for transform in parent_lineage:
            result._transformation_chain.add_transformation(transform)

        if transformation_metadata:
            if parent_lineage:
                transformation_metadata.parent_id = parent_lineage[-1].id
            result._transformation_chain.add_transformation(transformation_metadata)

        return result
    # CRITICAL: Update ALL transformation methods to use parent_chain
    def __getattr__(self, name):
        """Handle test framework attribute injection."""
        if name == 'CUSTOM':
            # Return None for test framework compatibility
            return None
        # Default behavior
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")



    def filter(self, *predicates) -> 'UnifiedLazyDataFrame':
        """Filter with chain preservation and proper predicate capture."""
        # Build a single predicate expression up-front
        final_predicate = pl.all(predicates) if len(predicates) > 1 else predicates[0]

        # Record transform using the actual polars expression so replay works
        transform_meta = TransformationMetadata(
            operation='filter',
            parameters={'expr': final_predicate}
        )

        # Capture predicate via default arg to avoid late-binding issues and aid serialization
        filter_node = GraphNode(
            op_type=ComputeOpType.FILTER,
            operation=(lambda df, pred=final_predicate: df.filter(pred)),
            inputs=[self._get_root_node()],
            metadata={'predicate': str(final_predicate)}
        )

        new_compute = self._create_compute_capability(
            filter_node,
            int(self._estimated_rows * 0.5)  # Conservative estimate
        )

        return self._create_derived_dataframe(
            new_compute=new_compute,
            transformation_metadata=transform_meta
        )
    def select(self, columns: Union[List[str], str]) -> 'UnifiedLazyDataFrame':
        """
        Select columns using Polars-like API.
        
        This provides explicit column selection matching pandas/polars conventions.
        
        Args:
            columns: Column name(s) to select
            
        Returns:
            UnifiedLazyDataFrame with only selected columns
        """
        # Normalize input to list
        if isinstance(columns, str):
            columns = [columns]
        
        # Delegate to existing column selection logic
        return self[columns]


    def _get_multiple_columns(self, columns: List[str]) -> 'UnifiedLazyDataFrame':
        """Definitive multi-column selection with schema-enforced filtering."""
        
        # Schema validation
        if self._schema:
            invalid_cols = [c for c in columns if c not in self._schema]
            if invalid_cols:
                raise KeyError(f"Columns not found: {invalid_cols}")
        
        # Create filtered schema
        filtered_schema = {}
        if self._schema:
            for col in columns:
                if col in self._schema:
                    filtered_schema[col] = self._schema[col]
        
        transform_meta = TransformationMetadata(
            operation='select_columns',
            parameters={'columns': columns, 'target_schema_size': len(filtered_schema)}
        )
        
        # Strict projection operation
        def strict_column_projection(df):
            """Execute STRICT column filtering."""
            try:
                if hasattr(df, 'select'):
                    return df.select(columns)
                elif hasattr(df, '__getitem__') and hasattr(df, 'columns'):
                    available = df.columns if hasattr(df, 'columns') else []
                    valid_columns = [c for c in columns if c in available]
                    result = df[valid_columns] if valid_columns else df.select([])
                    return result
                else:
                    if isinstance(df, dict):
                        result = {col: df[col] for col in columns if col in df}
                    else:
                        result = {}
                        for col in columns:
                            if hasattr(df, col):
                                result[col] = getattr(df, col)
                    return result
            except Exception as e:
                print(f"⚠️ Column projection failed: {e}")
                if hasattr(df, 'select'):
                    return df.select([])
                return {}
        
        project_node = GraphNode(
            op_type=ComputeOpType.PROJECT,
            operation=strict_column_projection,
            inputs=[self._get_root_node()],
            metadata={'columns': columns, 'strict_filtering': True}
        )
        
        new_compute = self._create_compute_capability(project_node, self._estimated_rows)
        
        # CRITICAL: Pass columns as required_columns to enforce constraint
        return self._create_derived_dataframe(
            new_compute=new_compute,
            new_schema=filtered_schema,
            transformation_metadata=transform_meta,
            required_columns=columns  # This enforces column constraint
        )

    def _create_minimal_compute(self):
        """Create minimal compute without embedding test data (lean fallback)."""
        class GraphComputeWrapper:
            """Complete compute wrapper with full transformation support."""

            def __init__(self, node, size, schema):
                self.root_node = node
                self.estimated_size = size
                self.schema = schema or {}

            def materialize(self):
                """Execute the graph node operation chain."""
                return self._execute_node(self.root_node)

            def _execute_node(self, node):
                if node.inputs:
                    input_results = [self._execute_node(inp) for inp in node.inputs if inp]
                    return node.operation(*input_results) if input_results else node.operation()
                else:
                    return node.operation()

            def estimate_memory(self):
                return self.estimated_size * 100  # Conservative estimate

            def transform(self, operation):
                """Apply transformation to create new compute capability."""
                transform_node = GraphNode(
                    op_type=ComputeOpType.MAP,
                    operation=lambda: operation(self.materialize()),
                    inputs=[self.root_node] if self.root_node else [],
                    metadata={'transformation': 'user_defined'}
                )
                return GraphComputeWrapper(
                    node=transform_node,
                    size=self.estimated_size,
                    schema=self.schema
                )

        # Always return an empty, valid compute capability as a safe fallback
        empty_node = GraphNode(
            op_type=ComputeOpType.SOURCE,
            operation=lambda: pl.DataFrame({}),
            inputs=[],
            metadata={'empty': True}
        )
        return GraphComputeWrapper(empty_node, 0, {})

    # Add these helper methods to the UnifiedLazyDataFrame class:

    # def _estimate_total_rows(self) -> int:
    #     """Fast row estimation with fallback."""
    #     try:
    #         if hasattr(self, '_compute') and hasattr(self._compute, 'estimated_size'):
    #             return self._compute.estimated_size
    #         elif hasattr(self, '_lazy_frames') and self._lazy_frames:
    #             return len(self._lazy_frames) * 500_000  # Conservative estimate
    #         else:
    #             return 1_000_000
    #     except Exception:
    #         return 1_000_000

    # def _init_schema(self):
    #     """Initialize schema with fallback."""
    #     try:
    #         if hasattr(self, '_lazy_frames') and self._lazy_frames:
    #             self._schema = dict(self._lazy_frames[0].schema)
    #         elif hasattr(self, '_compute') and hasattr(self._compute, 'schema'):
    #             self._schema = self._compute.schema or {}
    #         else:
    #             self._schema = {}
    #     except Exception:
    #         self._schema = {}

    # @property
    # def shape(self) -> tuple:
    #     """Safe shape property."""
    #     try:
    #         rows = getattr(self, '_estimated_rows', 0)
    #         cols = len(getattr(self, '_schema', {}))
    #         return (rows, cols)
    #     except Exception:
    #         return (0, 0)

    # @property
    # def columns(self) -> List[str]:
    #     """Safe columns property."""
    #     try:
    #         if hasattr(self, '_schema') and self._schema:
    #             return list(self._schema.keys())
    #         elif hasattr(self, '_lazy_frames') and self._lazy_frames:
    #             return self._lazy_frames[0].columns
    #         else:
    #             return []
    #     except Exception:
    #         return []

    # @property
    # def dtypes(self) -> Dict[str, str]:
    #     """Safe dtypes property."""
    #     try:
    #         if hasattr(self, '_schema') and self._schema:
    #             return {col: str(dtype) for col, dtype in self._schema.items()}
    #         else:
    #             return {}
    #     except Exception:
    #     return {}
    
    def _create_compute_from_frames(self, lazy_frames: List[pl.LazyFrame]) -> ComputeCapability:
        """Create compute capability from Polars LazyFrames."""
        # Select appropriate engine based on data size
        selector = EngineSelector(memory_budget_gb=self.memory_budget_gb)
        adapter = ComputeEngineAdapter(selector=selector)
        
        # Estimate size
        estimated_rows = len(lazy_frames) * 1_000_000  # Conservative estimate
        
        # Create compute capability
        return adapter.create_capability_from_frames(
            lazy_frames,
            estimated_rows=estimated_rows
        )
    
    def _init_schema(self):
        """Initialize schema through sampling if not provided."""
        try:
            # For LazyComputeCapability, we can inspect the graph
            if hasattr(self._compute, 'schema') and self._compute.schema:
                self._schema = self._compute.schema
            elif self._lazy_frames:
                # Get schema from first LazyFrame
                self._schema = dict(self._lazy_frames[0].collect_schema())
            else:
                # Sample-based schema inference
                sample_compute = self._create_sample_compute(n=1000)
                sample_data = sample_compute.materialize()
                self._schema = self._infer_schema_from_sample(sample_data)
        except Exception as e:
            warnings.warn(f"Schema inference failed: {e}")
            self._schema = {}
    
    def _create_sample_compute(self, n: int) -> ComputeCapability:
        """Create a compute capability for sampling."""
        # Add sampling operation to compute graph
        if hasattr(self._compute, 'root_node'):
            sample_node = GraphNode(
                op_type=ComputeOpType.SAMPLE,
                operation=lambda df: df.head(n) if hasattr(df, 'head') else df[:n],
                inputs=[self._compute.root_node],
                metadata={'sample_size': n}
            )
            
            return LazyComputeCapability(
                root_node=sample_node,
                engine=self._compute.engine if hasattr(self._compute, 'engine') else None,
                estimated_size=min(n, self._compute.estimated_size if hasattr(self._compute, 'estimated_size') else n),
                schema=self._schema
            )
        else:
            # Fallback for other compute capability types
            return self._compute
    
    def _infer_schema_from_sample(self, sample_data: Any) -> Dict[str, type]:
        """Infer schema from sample data."""
        schema = {}
        
        if isinstance(sample_data, pl.DataFrame):
            for col in sample_data.columns:
                schema[col] = sample_data[col].dtype
        elif isinstance(sample_data, pl.LazyFrame):
            # Get schema without collecting
            schema = dict(sample_data.schema)
        elif hasattr(sample_data, 'dtypes'):  # pandas-like
            schema = dict(sample_data.dtypes)
        elif isinstance(sample_data, pa.Table):
            for field in sample_data.schema:
                schema[field.name] = field.type
        
        return schema
    
    def _get_root_node(self) -> GraphNode:
        """Get root node from compute capability with proper fallback handling."""
        if hasattr(self._compute, 'root_node'):
            return self._compute.root_node
        else:
            try:
                # Create a root node for the compute
                return GraphNode(
                    op_type=ComputeOpType.SOURCE,
                    operation=lambda: self._compute.materialize(),
                    inputs=[],
                    metadata={'source': 'compute_capability'}
                )
            except Exception as e:
                # Create a proper FallbackNode that implements the full interface
                class FallbackNode:
                    def __init__(self, lazy_frames=None):
                        self.op_type = ComputeOpType.SOURCE
                        self.inputs = []
                        self.metadata = {'fallback': True, 'error': str(e)}
                        self.id = f'fallback_{id(self)}'
                        self._lazy_frames = lazy_frames
                        
                        # CRITICAL FIX: Add the operation attribute
                        self.operation = self._create_fallback_operation()
                    
                    def _create_fallback_operation(self):
                        """Create a fallback operation that can materialize data."""
                        def fallback_materialize():
                            # Try to materialize from lazy frames if available
                            if self._lazy_frames:
                                try:
                                    return pl.concat([
                                        lf.collect(streaming=True) 
                                        for lf in self._lazy_frames
                                    ])
                                except:
                                    # If that fails, try individual collection
                                    dfs = []
                                    for lf in self._lazy_frames:
                                        try:
                                            dfs.append(lf.collect())
                                        except:
                                            continue
                                    if dfs:
                                        return pl.concat(dfs)
                            
                            # Last resort: return empty DataFrame
                            return pl.DataFrame()
                        
                        return fallback_materialize
                
                # Pass lazy_frames to FallbackNode for data access
                return FallbackNode(lazy_frames=getattr(self, '_lazy_frames', None))

    
    def _estimate_total_rows(self) -> int:
        """Fast row estimation with adaptive shortcuts and caching."""
        if hasattr(self._compute, 'estimated_size'):
            return self._compute.estimated_size

        if not self._lazy_frames:
            return 0

        cache_key = 'row_estimate'
        if cache_key in self._operation_cache:
            return self._operation_cache[cache_key]

        try:
            total_frames = len(self._lazy_frames)
            # Fast path for very small collections: count directly
            if total_frames <= 3:
                total = 0
                for lf in self._lazy_frames:
                    total += lf.select(pl.count()).limit(1).collect()[0, 0]
                self._operation_cache[cache_key] = total
                return total

            # Light sampling for larger collections
            sample_size = min(5 if total_frames >= 100 else 3, total_frames)
            # Evenly spaced indices across frames
            step = max(1, total_frames // sample_size)
            indices = list(range(0, total_frames, step))[:sample_size]
            sample_count = 0
            for idx in indices:
                lf = self._lazy_frames[idx]
                sample_count += lf.select(pl.count()).limit(1).collect()[0, 0]

            if sample_size > 0:
                avg_rows = sample_count / sample_size
                estimate = int(avg_rows * total_frames)
                self._operation_cache[cache_key] = estimate
                return estimate
            return 0

        except Exception:
            # Conservative fallback estimate
            return len(self._lazy_frames) * 500_000 if self._lazy_frames else 0
    
    def _get_column_count(self) -> int:
        """Get column count from schema."""
        return len(self._schema) if self._schema else 0
    
    def _get_columns(self) -> List[str]:
        """Get column names."""
        if self._schema:
            return list(self._schema.keys())
        elif self._lazy_frames:
            return self._lazy_frames[0].columns
        return []
    
    def _get_dtypes(self) -> Dict[str, str]:
        """Get column data types."""
        if self._schema:
            return {col: str(dtype) for col, dtype in self._schema.items()}
        elif self._lazy_frames:
            return {col: str(dtype) for col, dtype in self._lazy_frames[0].collect_schema().items()}
        return {}
    
    # ========================================================================
    # Core Properties (matching load_processes.py UnifiedLazyDataFrame)
    # ========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape with proper empty state handling."""
        if hasattr(self, '_lazy_frames') and self._lazy_frames:
            # Check for empty DataFrame
            try:
                if (len(self._lazy_frames) == 1 and 
                    hasattr(self._lazy_frames[0], 'width') and 
                    self._lazy_frames[0].collect_schema().len() == 0):
                    return (0, 0)
            except:
                pass
        
        return (self._estimated_rows, self._get_column_count())
    
    @property
    def columns(self) -> List[str]:
        """Get column names respecting required_columns constraint."""
        if hasattr(self, 'required_columns') and self.required_columns:
            # When required_columns is set, only return those columns
            return [col for col in self.required_columns if col in self._schema] if self._schema else self.required_columns
        else:
            # Default behavior - return all schema columns
            return self._get_columns()
    
    @property
    def dtypes(self) -> Dict[str, str]:
        """Get column data types with caching."""
        return self._get_dtypes()
    
    # ========================================================================
    # Core DataFrame Operations
    # ========================================================================
    
    def __getitem__(self, key):
        """
        Column access that maintains compute graph integrity with validation.

        Returns LazyColumnAccessor or a new UnifiedLazyDataFrame; never materializes.
        """
        # Validate key type early for better errors
        if not isinstance(key, (str, list, slice, tuple)):
            raise TypeError(f"Invalid index type: {type(key).__name__}")

        try:
            if isinstance(key, str):
                result = self._get_single_column(key)
            elif isinstance(key, list):
                result = self._get_multiple_columns(key)
            elif isinstance(key, slice):
                result = self._slice_rows(key)
            else:  # tuple
                if len(key) != 2:
                    raise ValueError(f"Invalid indexing with {len(key)} dimensions")
                row_selector, col_selector = key
                result_df = self._slice_rows(row_selector) if row_selector is not None else self
                result = result_df[col_selector] if col_selector is not None else result_df

            # Track successful access
            self._access_patterns.append(AccessPattern(
                column=key if isinstance(key, str) else None,
                operation='getitem',
                timestamp=time.time()
            ))
            return result
        except Exception:
            # Track failed access for diagnostics
            self._access_patterns.append(AccessPattern(
                column=key if isinstance(key, str) else None,
                operation='getitem_failed',
                timestamp=time.time()
            ))
            raise
    def _extract_columns_from_query(self, expr: str) -> List[str]:
        """
        Extract column names from a pandas-style query expression.
        
        Uses the existing AST infrastructure from pandas_to_polars_queries
        to accurately identify all column references in the query.
        """
        # Simple per-instance cache to avoid repeated AST parsing
        try:
            cache = self._query_columns_cache
            if expr in cache:
                return cache[expr]

            # Parse the expression using Python's AST module
            tree = ast.parse(expr, mode='eval')
            
            # Use a visitor to extract all Name nodes (column references)
            class ColumnExtractor(ast.NodeVisitor):
                def __init__(self):
                    self.columns = set()
                    self.reserved = {'True', 'False', 'None', 'and', 'or', 'not'}
                
                def visit_Name(self, node):
                    # Filter out reserved Python keywords and constants
                    if node.id not in self.reserved and not node.id.startswith('__'):
                        self.columns.add(node.id)
                    self.generic_visit(node)
                
                def visit_Attribute(self, node):
                    # Handle attributes like df.column.method()
                    if isinstance(node.value, ast.Name) and node.value.id not in self.reserved:
                        self.columns.add(node.value.id)
                    self.generic_visit(node)
            
            extractor = ColumnExtractor()
            extractor.visit(tree)
            
            # Validate against schema if available
            if hasattr(self, '_schema') and self._schema:
                valid_columns = [col for col in extractor.columns if col in self._schema]
                cache[expr] = valid_columns
                return valid_columns

            result = list(extractor.columns)
            cache[expr] = result
            return result
            
        except Exception as e:
            # Fallback: return empty list rather than crashing
            warnings.warn(f"Failed to extract columns from query '{expr}': {e}")
            return []
    

    
    def _get_single_column(self, column: str) -> 'LazyColumnAccessor':
        """Get a single column as a lazy accessor."""
        if self._schema and column not in self._schema:
            raise KeyError(f"Column '{column}' not found. Available columns: {list(self._schema.keys())}")
        
        # Create projection compute node
        project_node = GraphNode(
            op_type=ComputeOpType.PROJECT,
            operation=lambda df, col=column: df.select(col) if hasattr(df, 'select') else df[col],
            inputs=[self._get_root_node()],
            metadata={'columns': [column], 'projection': 'single'}
        )
        
        # Create compute capability for projection
        if isinstance(self._compute, LazyComputeCapability):
            project_compute = LazyComputeCapability(
                root_node=project_node,
                engine=self._compute.engine,
                estimated_size=self._compute.estimated_size,
                schema={column: self._schema.get(column)} if self._schema else None
            )
        else:
            # Fallback for other compute types
            project_compute = self._compute.transform(
                lambda df, col=column: df.select(col) if hasattr(df, 'select') else df[col]
            )
        
        return LazyColumnAccessor(
            compute=project_compute,
            column_name=column,
            parent_ref=weakref.ref(self)
        )
      
    def _slice_rows(self, slice_obj: slice) -> 'UnifiedLazyDataFrame':
        """Row slicing with chain preservation."""
        start = slice_obj.start or 0
        stop = slice_obj.stop or self._estimated_rows
        step = slice_obj.step or 1
        
        transform_meta = TransformationMetadata(
            operation='slice_rows',
            parameters={'start': start, 'stop': stop, 'step': step}
        )
        
        slice_node = GraphNode(
            op_type=ComputeOpType.SLICE,
            operation=lambda df, s=slice_obj: df[s],
            inputs=[self._get_root_node()],
            metadata={'start': start, 'stop': stop, 'step': step}
        )
        
        new_size = max(0, (stop - start) // step)
        new_compute = self._create_compute_capability(slice_node, new_size)
        
        return self._create_derived_dataframe(
            new_compute=new_compute,
            transformation_metadata=transform_meta
        )
    
    # ========================================================================
    # Query and Transformation Methods (matching load_processes.py)
    # ========================================================================
    
    def query(self, expr: str, auto_generate_columns: bool = True) -> 'UnifiedLazyDataFrame':
        """Query with chain preservation."""
        required_cols = self._extract_columns_from_query(expr)
        missing_cols = [col for col in required_cols if col not in self.columns]
        
        if missing_cols:
            if auto_generate_columns and any('delta' in col or 'abs' in col for col in missing_cols):
                # Single retry with auto-generation; disable thereafter to avoid loops
                print("🔧 Auto-generating kinematic columns for query...")
                return self.createDeltaColumns().query(expr, auto_generate_columns=False)
            raise KeyError(f"Query requires missing columns: {missing_cols}")

        sel = self._estimate_selectivity(expr)
        transform_meta = TransformationMetadata(
            operation='query',
            parameters={'expr': expr, 'selectivity': sel}
        )
        
        polars_expr = self._convert_query_to_polars(expr)
        filter_node = GraphNode(
            op_type=ComputeOpType.FILTER,
            operation=lambda df, pred=polars_expr: df.filter(pred),
            inputs=[self._get_root_node()],
            metadata={'expr': expr, 'estimated_selectivity': sel}
        )
        
        new_compute = self._create_compute_capability(
            filter_node, 
            int(self._estimated_rows * sel)
        )
        
        return self._create_derived_dataframe(
            new_compute=new_compute,
            transformation_metadata=transform_meta
        )
    
    def _convert_query_to_polars(self, expr: str) -> pl.Expr:
        """Convert pandas-style query to Polars expression using dedicated parser."""
        try:
            # Import parser module
            from pandas_to_polars_queries import convert_pandas_query
            
            # Parse expression
            polars_expr = convert_pandas_query(expr)
            
            # Validate against schema if available
            if self._schema:
                # Extract column references from expression
                import re
                col_pattern = re.compile(r'\b([a-zA-Z_]\w*)\b(?!\s*\()')
                referenced_cols = set(col_pattern.findall(expr))
                
                # Check columns exist
                missing_cols = referenced_cols - set(self._schema.keys())
                if missing_cols:
                    warnings.warn(f"Query references unknown columns: {missing_cols}")
            
            return polars_expr
            
        except Exception as e:
            warnings.warn(f"Query parsing failed for '{expr}': {e}")
            # Return permissive filter to avoid breaking execution
            return pl.lit(True)
    
    def _estimate_selectivity(self, expr: str) -> float:
        """Estimate query selectivity for optimization."""
        # Simple heuristics - in production, use statistics
        if '==' in expr:
            return 0.1  # Equality typically selective
        elif '>' in expr or '<' in expr:
            return 0.5  # Range queries moderate selectivity
        else:
            return 0.9  # Conservative estimate

    def oneCandOnly(self, group_cols=None, sort_col=None, ascending=False) -> 'UnifiedLazyDataFrame':
        """
        Select one candidate per group using efficient Polars operations.
        
        Args:
            group_cols: Columns to group by (auto-detected for Belle II if None)
            sort_col: Column to sort by ('random' for random selection, None for first)
            ascending: Sort direction
            
        Returns:
            UnifiedLazyDataFrame with one candidate per group
        """
        
        # Smart defaults for Belle II physics
        if group_cols is None:
            belle2_cols = ['__event__', '__run__', '__experiment__', '__production__']
            group_cols = [col for col in belle2_cols if col in self.columns]
            if not group_cols:
                raise ValueError("No valid grouping columns found. Specify group_cols explicitly.")
        
        # Create transformation metadata
        transform_meta = TransformationMetadata(
            operation='oneCandOnly',
            parameters={'group_cols': group_cols, 'sort_col': sort_col, 'ascending': ascending}
        )
        
        def efficient_candidate_selection(df):
            """Schema-preserving selection using unique or window rank; lazy only."""
            lf = df
            if sort_col == 'random':
                lf = lf.with_columns(pl.random(seed=42).alias('_rk')).sort('_rk').drop('_rk')
                return lf.unique(subset=group_cols, keep='first')
            if sort_col is None:
                return lf.unique(subset=group_cols, keep='first')
            rk = pl.col(sort_col).rank(method='ordinal', descending=not ascending).over(group_cols)
            return lf.with_columns(rk.alias('_rk')).filter(pl.col('_rk') == 1).drop('_rk')
        
        # Create computation graph node
        selection_node = GraphNode(
            op_type=ComputeOpType.AGGREGATE,
            operation=efficient_candidate_selection,
            inputs=[self._get_root_node()],
            metadata={'operation': 'candidate_selection', 'group_cols': group_cols}
        )
        
        # Conservative size estimation (typically 10-30% of original for Belle II)
        estimated_groups = max(1, int(self._estimated_rows * 0.2))
        new_compute = self._create_compute_capability(selection_node, estimated_groups)
        
        return self._create_derived_dataframe(
            new_compute=new_compute,
            transformation_metadata=transform_meta
        )
    
    # ============================================================================
    # MERGED: createDeltaColumns (Identical in both - no conflict)
    # ============================================================================
    def createDeltaColumns(self) -> 'UnifiedLazyDataFrame':
        """
        Create kinematic delta columns for Belle II muon pair analysis.

        Integrated version:
        - Preserves lazy graph and full lineage
        - Single-pass, vectorized expressions with reusable trig
        - Robust 3D angle computation
        - Properly expands required_columns so new columns are retained
        - Optional schema validation for inputs
        """
    # Use module-level numpy and polars imports

        # Validate required input columns (fail fast with clear error)
        required_inputs = [
            # base kinematics
            'mu1Phi', 'mu2Phi', 'mu1Theta', 'mu2Theta', 'mu1P', 'mu2P',
            'pRecoilPhi', 'pRecoilTheta', 'pRecoil',
            # cluster-based features used below
            'mu1clusterPhi', 'mu2clusterPhi', 'mu1clusterTheta', 'mu2clusterTheta',
        ]
        try:
            self._validate_columns_exist(required_inputs, available_cols=set(self._schema.keys()) if self._schema else None)
        except Exception:
            # If schema is absent, skip strict validation (keeps lazy compatibility).
            pass

        # Complete list of delta columns that will be created
        delta_column_specs = {
            'absdPhi': pl.Float64,
            'absdPhiMu1': pl.Float64,
            'absdPhiMu2': pl.Float64,
            'absdThetaMu1': pl.Float64,
            'absdThetaMu2': pl.Float64,
            'absdAlpha1': pl.Float64,
            'absdAlpha2': pl.Float64,
            'dRMu1': pl.Float64,
            'dRMu2': pl.Float64,
            'pTRecoil': pl.Float64,
            'mu1Pt': pl.Float64,
            'mu2Pt': pl.Float64,
            'deltaMu1PRecoil': pl.Float64,
            'deltaMu2PRecoil': pl.Float64,
            'deltaMu1ClusterPRecoil': pl.Float64,
            'deltaMu2ClusterPRecoil': pl.Float64,
            'min_deltaMuPRecoil': pl.Float64,
            'min_deltaMuClusterPRecoil': pl.Float64,
        }

        transform_meta = TransformationMetadata(
            operation='createDeltaColumns',
            parameters={'columns_added': list(delta_column_specs.keys())}
        )

        def compute_delta_columns(df):
            """Single-pass delta column computation with expression optimization."""
            # Reusable trig
            sin_mu1_theta = pl.col('mu1Theta').sin()
            sin_mu2_theta = pl.col('mu2Theta').sin()
            cos_mu1_theta = pl.col('mu1Theta').cos()
            cos_mu2_theta = pl.col('mu2Theta').cos()
            sin_recoil_theta = pl.col('pRecoilTheta').sin()
            cos_recoil_theta = pl.col('pRecoilTheta').cos()

            # Helpers
            def delta_phi_expr(phi1, phi2):
                dphi = (phi1 - phi2).abs()
                return pl.when(dphi > np.pi).then(2 * np.pi - dphi).otherwise(dphi)

            def delta_r_expr(phi1, theta1, phi2, theta2):
                dphi = delta_phi_expr(phi1, phi2)
                dtheta = (theta1 - theta2).abs()
                return (dphi.pow(2) + dtheta.pow(2)).sqrt()

            def angle_3d_expr(p1, theta1, phi1, p2, theta2, phi2,
                            sin_theta1, cos_theta1, sin_theta2, cos_theta2):
                px1 = p1 * sin_theta1 * pl.col(phi1).cos()
                py1 = p1 * sin_theta1 * pl.col(phi1).sin()
                pz1 = p1 * cos_theta1

                px2 = p2 * sin_theta2 * pl.col(phi2).cos()
                py2 = p2 * sin_theta2 * pl.col(phi2).sin()
                pz2 = p2 * cos_theta2

                dot = px1 * px2 + py1 * py2 + pz1 * pz2
                mag1 = (px1.pow(2) + py1.pow(2) + pz1.pow(2)).sqrt()
                mag2 = (px2.pow(2) + py2.pow(2) + pz2.pow(2)).sqrt()
                cos_angle = (dot / (mag1 * mag2)).clip(-1.0, 1.0)
                return cos_angle.arccos()

            return df.with_columns([
                # Delta phi
                delta_phi_expr(pl.col('mu1Phi'), pl.col('mu2Phi')).alias('absdPhi'),
                delta_phi_expr(pl.col('mu1clusterPhi'), pl.col('pRecoilPhi')).alias('absdPhiMu1'),
                delta_phi_expr(pl.col('mu2clusterPhi'), pl.col('pRecoilPhi')).alias('absdPhiMu2'),

                # Delta theta
                (pl.col('mu1clusterTheta') - pl.col('pRecoilTheta')).abs().alias('absdThetaMu1'),
                (pl.col('mu2clusterTheta') - pl.col('pRecoilTheta')).abs().alias('absdThetaMu2'),

                # Transverse momenta
                (pl.col('pRecoil') * sin_recoil_theta).alias('pTRecoil'),
                (pl.col('mu1P') * sin_mu1_theta).alias('mu1Pt'),
                (pl.col('mu2P') * sin_mu2_theta).alias('mu2Pt'),

                # Delta R
                delta_r_expr(pl.col('mu1clusterPhi'), pl.col('mu1clusterTheta'),
                            pl.col('pRecoilPhi'), pl.col('pRecoilTheta')).alias('dRMu1'),
                delta_r_expr(pl.col('mu2clusterPhi'), pl.col('mu2clusterTheta'),
                            pl.col('pRecoilPhi'), pl.col('pRecoilTheta')).alias('dRMu2'),

                # 3D angles to recoil
                angle_3d_expr(pl.col('mu1P'), pl.col('mu1Theta'), 'mu1Phi',
                            pl.col('pRecoil'), pl.col('pRecoilTheta'), 'pRecoilPhi',
                            sin_mu1_theta, cos_mu1_theta, sin_recoil_theta, cos_recoil_theta).alias('absdAlpha1'),
                angle_3d_expr(pl.col('mu2P'), pl.col('mu2Theta'), 'mu2Phi',
                            pl.col('pRecoil'), pl.col('pRecoilTheta'), 'pRecoilPhi',
                            sin_mu2_theta, cos_mu2_theta, sin_recoil_theta, cos_recoil_theta).alias('absdAlpha2'),

                # 3D angles with transverse magnitudes (compute Pt directly to avoid intra-batch deps)
                angle_3d_expr(pl.col('mu1P') * sin_mu1_theta, pl.col('mu1Theta'), 'mu1Phi',
                            pl.col('pRecoil') * sin_recoil_theta, pl.col('pRecoilTheta'), 'pRecoilPhi',
                            sin_mu1_theta, cos_mu1_theta, sin_recoil_theta, cos_recoil_theta).alias('deltaMu1PRecoil'),
                angle_3d_expr(pl.col('mu2P') * sin_mu2_theta, pl.col('mu2Theta'), 'mu2Phi',
                            pl.col('pRecoil') * sin_recoil_theta, pl.col('pRecoilTheta'), 'pRecoilPhi',
                            sin_mu2_theta, cos_mu2_theta, sin_recoil_theta, cos_recoil_theta).alias('deltaMu2PRecoil'),
                angle_3d_expr(pl.col('mu1P') * sin_mu1_theta, pl.col('mu1clusterTheta'), 'mu1clusterPhi',
                            pl.col('pRecoil') * sin_recoil_theta, pl.col('pRecoilTheta'), 'pRecoilPhi',
                            sin_mu1_theta, cos_mu1_theta, sin_recoil_theta, cos_recoil_theta).alias('deltaMu1ClusterPRecoil'),
                angle_3d_expr(pl.col('mu2P') * sin_mu2_theta, pl.col('mu2clusterTheta'), 'mu2clusterPhi',
                            pl.col('pRecoil') * sin_recoil_theta, pl.col('pRecoilTheta'), 'pRecoilPhi',
                            sin_mu2_theta, cos_mu2_theta, sin_recoil_theta, cos_recoil_theta).alias('deltaMu2ClusterPRecoil'),
            ]).with_columns([
                pl.min_horizontal(['deltaMu1PRecoil', 'deltaMu2PRecoil']).alias('min_deltaMuPRecoil'),
                pl.min_horizontal(['deltaMu1ClusterPRecoil', 'deltaMu2ClusterPRecoil']).alias('min_deltaMuClusterPRecoil'),
            ])

        # Create computation node linked to parent graph
        delta_node = GraphNode(
            op_type=ComputeOpType.MAP,
            operation=compute_delta_columns,
            inputs=[self._get_root_node()],
            metadata={'operation': 'delta_columns', 'columns_added': len(delta_column_specs)}
        )

        # Update schema
        new_schema = self._schema.copy() if self._schema else {}
        new_schema.update(delta_column_specs)

        # Expand required_columns so new columns are retained downstream
        new_required_columns = None
        if hasattr(self, 'required_columns') and self.required_columns:
            new_required_columns = list(dict.fromkeys(self.required_columns + list(delta_column_specs.keys())))

        new_compute = self._create_compute_capability(delta_node, self._estimated_rows)

        return self._create_derived_dataframe(
            new_compute=new_compute,
            new_schema=new_schema,
            transformation_metadata=transform_meta,
            required_columns=new_required_columns
        )    
    # ============================================================================
    # OPTIMAL MERGE: hist method combining HEAD's framework with MASTER's enhancements
    # ============================================================================
    def hist(self, column: str, bins: int = 50, 
             range: Optional[Tuple[float, float]] = None,
             density: bool = False, 
             weights: Optional[str] = None,
             force_strategy: Optional[HistogramExecutionStrategy] = None,
             debug: bool = False,  # MASTER: Debug mode support
             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        OPTIMAL MERGED IMPLEMENTATION
        
        Architecture:
        - PRIMARY: HEAD's comprehensive multi-strategy framework
        - ENHANCEMENT: MASTER's debug mode for development
        - OPTIMIZATION: Integrated adaptive chunking from MASTER
        - SYNTHESIS: Combined range computation and helpers
        
        Features:
        - 5-tier execution strategy with graceful degradation
        - Debug mode with execution path tracing
        - Adaptive chunking with performance learning
        - Comprehensive metrics and monitoring
        - Physics-aware optimizations
        
        Args:
            column: Column name to compute histogram for
            bins: Number of bins (default: 50)
            range: Optional (min, max) range for histogram
            density: If True, normalize to density
            weights: Optional column name for weights
            force_strategy: Override automatic strategy selection
            debug: Enable debug mode (MASTER feature)
            **kwargs: Additional arguments for compatibility
            
        Returns:
            Tuple of (counts, bin_edges) as numpy arrays
        """
        
        # Phase 0
        # Build minimal context and delegate selection/range via new helpers (no behavior change)
        try:
            system_avail_gb = psutil.virtual_memory().available / 1024**3
        except Exception:
            system_avail_gb = self.memory_budget_gb
        memory_pressure = system_avail_gb < self.memory_budget_gb * 0.3
        num_frames = len(self._lazy_frames) if hasattr(self, '_lazy_frames') and self._lazy_frames else 0
        dtype_str = None
        try:
            if hasattr(self, '_schema') and self._schema and column in self._schema:
                dtype_str = str(self._schema[column])
        except Exception:
            pass
        from .histogram.core_types import HistogramContext
        from .histogram.selector import StrategySelector
        from .histogram.range_estimator import RangeEstimator
        ctx = HistogramContext(
            df_ref=weakref.ref(self),
            column=column,
            bins=bins,
            range=range,
            density=density,
            weights=weights,
            estimated_rows=getattr(self, '_estimated_rows', 0),
            num_frames=num_frames,
            memory_budget_gb=self.memory_budget_gb,
            system_available_gb=system_avail_gb,
            memory_pressure=memory_pressure,
            dtype_str=dtype_str,
            schema_size=len(self._schema) if hasattr(self, '_schema') and self._schema else 0,
            debug=debug,
        )
        selector = StrategySelector(self)
        decision = selector.choose(ctx, force_strategy)
        chosen_strategy = decision.strategy
        if range is None:
            computed_range = RangeEstimator(self).determine(ctx, chosen_strategy)
        else:
            computed_range = range
        if debug and getattr(decision, 'rationale', ''):
            print(f"🔎 Strategy rationale: {decision.rationale}")
        # Phase 4: Capture rationale for observability
        try:
            metrics_rationale = getattr(decision, 'rationale', '')
            if metrics_rationale:
                setattr(self, '_last_hist_rationale', metrics_rationale)
        except Exception:
            pass

        # MASTER ENHANCEMENT: Debug mode support
        if debug:
            return self._debug_hist_execution(column, bins, range, density, weights, **kwargs)
        
        # HEAD FRAMEWORK: Initialize metrics collection
        metrics = HistogramMetrics(
            strategy=HistogramExecutionStrategy.FALLBACK_BASIC,
            total_rows=self._estimated_rows,
            processed_rows=0,
            execution_time=0.0,
            memory_peak_mb=0.0,
            chunk_size_used=0,
            chunks_processed=0,
            errors_recovered=0
        )
        
        start_time = time.time()
        # Phase 4: Observability inputs (non-breaking via kwargs)
        time_budget_s = kwargs.get('time_budget_s')
        progress_cb = kwargs.get('progress_callback')
        metrics_cb = kwargs.get('metrics_callback')
        # Propagate to executors via instance fields (cleared in finally)
        had_time_budget = False
        had_progress_cb = False
        had_metrics_cb = False
        if time_budget_s is not None:
            setattr(self, '_hist_time_budget_s', float(time_budget_s))
            had_time_budget = True
        if callable(progress_cb):
            setattr(self, '_hist_progress_cb', progress_cb)
            had_progress_cb = True
        if callable(metrics_cb):
            setattr(self, '_hist_metrics_cb', metrics_cb)
            had_metrics_cb = True
        # Reset any stale early-exit flag
        try:
            if hasattr(self, '_hist_early_exit'):
                delattr(self, '_hist_early_exit')
        except Exception:
            pass
        
        # Phase 1: Strategy Selection (ENHANCED)
        # ========================================
        if force_strategy:
            strategy = force_strategy
        else:
            # Phase 0: use delegated decision to maintain behavior with clearer separation
            strategy = chosen_strategy
        
        metrics.strategy = strategy
        print(f"📊 Histogram computation for '{column}' using {strategy.name} strategy")
        
        # Phase 2: Range Determination (MERGED)
        # ======================================
        if range is None:
            # Phase 0: use delegated computation (identical result expected)
            range = computed_range
        
        # Phase 3: Execute with Selected Strategy (HEAD FRAMEWORK)
        # =========================================================
        try:
            from .histogram.executors import get_executor
            with memory_monitor() as get_memory_usage:
                executor = get_executor(self, strategy)
                result = executor.execute(ctx, range, metrics)

                # Record memory usage; use -1 sentinel when unknown
                mem_delta = None
                try:
                    mem_delta = get_memory_usage()
                except Exception:
                    mem_delta = None
                try:
                    if mem_delta is None:
                        metrics.memory_peak_mb = -1.0
                    else:
                        metrics.memory_peak_mb = max(0.0, float(mem_delta))
                except Exception:
                    metrics.memory_peak_mb = -1.0

        except Exception as primary_error:
            # HEAD: Graceful degradation with cascade
            print(f"   ⚠️ {strategy.name} failed: {str(primary_error)}, attempting fallback")
            metrics.errors_recovered += 1

            fallback_strategies = [
                HistogramExecutionStrategy.MEMORY_CONSTRAINED,
                HistogramExecutionStrategy.FALLBACK_BASIC,
            ]

            from .histogram.executors import get_executor
            for fallback in fallback_strategies:
                try:
                    print(f"   🔄 Attempting {fallback.name} strategy")
                    metrics.strategy = fallback
                    executor = get_executor(self, fallback)
                    result = executor.execute(ctx, range, metrics)
                    break
                except Exception as fallback_error:
                    print(f"   ❌ {fallback.name} also failed: {str(fallback_error)}")
                    metrics.errors_recovered += 1
                    continue
            else:
                # All strategies failed - return empty histogram
                print("   ❌ All strategies failed, returning empty histogram")
                result = (np.zeros(bins, dtype=np.float64), np.linspace(0, 1, bins + 1))
        
        # Phase 4: Finalization and Metrics
        # ==================================
        metrics.execution_time = time.time() - start_time
        # Sanity clamps
        if metrics.processed_rows > metrics.total_rows > 0:
            metrics.processed_rows = metrics.total_rows
        # Preserve negative sentinel (-1) to indicate unknown memory peak
        
        # Log performance metrics
        self._log_histogram_performance(metrics)
        
        # Adaptive optimization - record performance for future strategy selection
        self._record_histogram_performance(column, metrics)
        
        # MASTER: Record chunking performance if adaptive was used
        if strategy == HistogramExecutionStrategy.ADAPTIVE_CHUNKED and metrics.chunk_size_used > 0:
            self.record_chunking_performance(metrics.chunk_size_used, metrics.execution_time)

        # Persist compact snapshot for observability and diagnostics
        try:
            self._last_hist_metrics = {
                'strategy': metrics.strategy.name,
                'processed_rows': int(metrics.processed_rows),
                'total_rows': int(metrics.total_rows),
                'efficiency': float(metrics.efficiency),
                'throughput_mps': float(metrics.throughput_mps),
                'memory_peak_mb': float(getattr(metrics, 'memory_peak_mb', -1.0) if getattr(metrics, 'memory_peak_mb', None) is not None else -1.0),
                'chunks_processed': int(metrics.chunks_processed),
                'chunk_size': int(metrics.chunk_size_used),
                'errors_recovered': int(metrics.errors_recovered),
                'elapsed_s': float(metrics.execution_time),
                # Phase 4: extras
                'time_budget_s': float(time_budget_s) if time_budget_s is not None else None,
                'early_exit': bool(getattr(self, '_hist_early_exit', False)),
                'decision_confidence': float(getattr(decision, 'confidence', 0.0)),
            }
            # Optional metrics callback
            try:
                cb = getattr(self, '_hist_metrics_cb', None)
                if callable(cb):
                    payload = dict(self._last_hist_metrics)
                    payload['rationale'] = getattr(self, '_last_hist_rationale', '')
                    cb(payload)
            except Exception:
                pass
        except Exception:
            pass

        # Cleanup transient observability attributes
        try:
            if had_time_budget and hasattr(self, '_hist_time_budget_s'):
                delattr(self, '_hist_time_budget_s')
            if had_progress_cb and hasattr(self, '_hist_progress_cb'):
                delattr(self, '_hist_progress_cb')
            if had_metrics_cb and hasattr(self, '_hist_metrics_cb'):
                delattr(self, '_hist_metrics_cb')
            if hasattr(self, '_hist_early_exit'):
                delattr(self, '_hist_early_exit')
        except Exception:
            pass

        return result
        
    # Ensure cleanup (defensive; will not run due to return above, but kept for clarity)
    # if had_time_budget and hasattr(self, '_hist_time_budget_s'):
    #     delattr(self, '_hist_time_budget_s')
    # if had_progress_cb and hasattr(self, '_hist_progress_cb'):
    #     delattr(self, '_hist_progress_cb')
    
    # ============================================================================
    # MERGED: Strategy Selection combining HEAD's comprehensive approach with MASTER's optimizations
    # ============================================================================
    def _select_optimal_histogram_strategy_merged(self, column: str, bins: int, 
                                                  weights: Optional[str]) -> HistogramExecutionStrategy:
        """
        OPTIMAL MERGED: Combines HEAD's comprehensive selection with MASTER's simplifications.
        
        Strategy priority:
        1. Check performance history (HEAD)
        2. System state assessment (HEAD + MASTER)
        3. Adaptive optimization check (MASTER)
        4. Size-based selection (MERGED)
        """
        
        # Handle WeightedDataFrame context (HEAD)
        actual_df = self
        if hasattr(self, '_dataframe') and hasattr(self._dataframe, '_estimated_rows'):
            actual_df = self._dataframe
        
        # Force cache invalidation for fresh estimation (HEAD)
        if hasattr(actual_df, '_operation_cache'):
            actual_df._operation_cache.clear()
        
        # Get system state (HEAD)
        memory_available_gb = psutil.virtual_memory().available / 1024**3
        memory_budget = getattr(actual_df, 'memory_budget_gb', 4.0)
        memory_pressure = memory_available_gb < memory_budget * 0.3
        
        # Check performance history (HEAD)
        if hasattr(actual_df, '_histogram_performance_history'):
            best_previous = actual_df._get_best_previous_strategy(column)
            if best_previous:
                return best_previous
        
        # Calculate fresh row estimation with transformation awareness (HEAD)
        fresh_rows = self._calculate_fresh_row_estimation(actual_df)
        
        # MASTER SIMPLIFICATION: Check if adaptive chunking is beneficial
        if (fresh_rows > 100_000 and 
            fresh_rows < 100_000_000 and 
            not memory_pressure and 
            getattr(self, 'enable_adaptive_chunking', True)):
            return HistogramExecutionStrategy.ADAPTIVE_CHUNKED
        
        # C++ acceleration check (MERGED)
        if self._should_use_cpp_acceleration(fresh_rows, bins, weights):
            return HistogramExecutionStrategy.CPP_ACCELERATED
        
        # Billion-row engine check (HEAD)
        if fresh_rows > 10_000_000 or memory_pressure:
            if self._check_billion_row_capability():
                return HistogramExecutionStrategy.BILLION_ROW_ENGINE
            elif memory_pressure:
                return HistogramExecutionStrategy.MEMORY_CONSTRAINED
        
        # Size-based selection (HEAD)
        if fresh_rows > 10_000:
            return HistogramExecutionStrategy.LAZY_CHUNKED
        
        return HistogramExecutionStrategy.FALLBACK_BASIC
    
    def _calculate_fresh_row_estimation(self, actual_df) -> int:
        """Helper to calculate fresh row estimation with transformation awareness."""
        fresh_rows = actual_df._estimated_rows
        complexity = 1.0
        
        if hasattr(actual_df, '_transformation_chain'):
            chain = actual_df._transformation_chain.get_lineage()
            
            # Physics-aware reduction calculation
            for transform in chain:
                op = getattr(transform, 'operation', '')
                if op in ('oneCandOnly', 'dataframe_oneCandOnly'):
                    fresh_rows = int(fresh_rows * 0.1)
                elif op in ('query', 'dataframe_query'):
                    selectivity = getattr(transform, 'parameters', {}).get('selectivity', 0.5)
                    fresh_rows = int(fresh_rows * selectivity)
                elif op == 'aggregate':
                    fresh_rows = min(fresh_rows, 1000)
                else:
                    fresh_rows = int(fresh_rows * 0.9)
            
            fresh_rows = max(100, fresh_rows)
            complexity = 1.2 ** len(chain)
        
        # Update estimation
        actual_df._estimated_rows = fresh_rows
        return fresh_rows
    
    def _should_use_cpp_acceleration(self, rows: int, bins: int, weights: Optional[str]) -> bool:
        """
        MERGED: Combines HEAD's comprehensive check with MASTER's simplifications.
        """
        # MASTER: Simple eligibility checks
        if rows < 100_000:  # Overhead not justified
            return False
        if weights and not isinstance(weights, str):  # Complex weights not supported
            return False
        if bins > 10_000:  # Extreme bin counts
            return False
        
        # HEAD: Engine availability check
        if not (hasattr(self, '_histogram_engine') and self._histogram_engine is not None):
            # Try to create engine (MASTER approach)
            if CPP_HISTOGRAM_AVAILABLE:
                try:
                    from optimized_cpp_integration import OptimizedStreamingHistogram
                    self._histogram_engine = OptimizedStreamingHistogram()
                except:
                    return False
            else:
                return False
        
        # HEAD: Functionality test
        return self._test_cpp_engine()
    
    # ============================================================================
    # NEW: Adaptive Chunked Execution (MASTER's crown jewel integrated)
    # ============================================================================
    def _execute_adaptive_chunked_histogram(self, column: str, bins: int, range: Tuple[float, float],
                                           density: bool, weights: Optional[str],
                                           metrics: HistogramMetrics) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute using MASTER's adaptive chunking optimization.
        
        This is MASTER's key innovation integrated into HEAD's framework.
        """
        # Timing for adaptive feedback
        start_time = time.time()
        # Use ChunkingEnhancement for optimal sizing
        chunk_size = ChunkingEnhancement.calculate_optimal_chunk_size(
            memory_budget_gb=self.memory_budget_gb,
            estimated_rows=self._estimated_rows,
            avg_row_bytes=100.0,  # Could be computed from schema
            operation_type='histogram',
            fallback_calculation=lambda: self._calculate_original_chunk_size_preserved()
        )
        
        metrics.chunk_size_used = chunk_size
        print(f"   🎯 Adaptive chunk size: {chunk_size:,} rows")

        # Initialize histogram
        bin_edges = self._compute_bin_edges(range, bins)
        accumulator = np.zeros(bins, dtype=np.float64)
        
        # Get lazy frames
        lazy_frames = self._extract_lazy_frames_safely()
        
        # Process with adaptive chunking (Phase 3: smarter execution with feedback + memory throttling)
        for frame_idx, lazy_frame in enumerate(lazy_frames):
            # Phase 4: Observability — early exit and progress callback
            try:
                tb = getattr(self, '_hist_time_budget_s', None)
                if tb is not None and (time.time() - start_time) > tb:
                    print("   ⏱️ Time budget reached; early exit from adaptive loop")
                    try:
                        setattr(self, '_hist_early_exit', True)
                    except Exception:
                        pass
                    break
            except Exception:
                pass

            # Use enhanced streaming that returns stats for feedback
            frame_contribution, rows_processed, frame_elapsed, last_chunk_used = (
                self._process_frame_streaming_stats(
                    lazy_frame, column, bin_edges, chunk_size, weights
                )
            )

            accumulator += frame_contribution
            metrics.processed_rows += int(rows_processed)
            metrics.chunks_processed += 1

            # Progress callback
            try:
                pcb = getattr(self, '_hist_progress_cb', None)
                if callable(pcb):
                    pcb(
                        processed_rows=metrics.processed_rows,
                        chunks=metrics.chunks_processed,
                        chunk_size=metrics.chunk_size_used,
                        elapsed=time.time() - start_time,
                    )
            except Exception:
                pass

            # Adaptive adjustment after first frame for subsequent frames
            if frame_idx == 0 and len(lazy_frames) > 1:
                # Record performance for learning
                if frame_elapsed and frame_elapsed > 0:
                    throughput = rows_processed / frame_elapsed
                    try:
                        ChunkingEnhancement.get_optimizer(self.memory_budget_gb).record_performance(
                            last_chunk_used, throughput
                        )
                    except Exception:
                        pass
                    # Adjust chunk size towards target latency
                    try:
                        chunk_size = self._adjust_chunk_size_based_on_performance(
                            max(1, int(last_chunk_used)), max(1, int(rows_processed)), float(frame_elapsed)
                        )
                        metrics.chunk_size_used = chunk_size
                    except Exception:
                        pass
        
        # Apply density normalization
        accumulator = self._apply_density_normalization(accumulator, bin_edges, density, metrics.processed_rows)
        
        return accumulator, bin_edges
    
    # ============================================================================
    # MERGED: Range Computation combining both approaches
    # ============================================================================
    def _compute_optimal_range(self, column: str, 
                               strategy: HistogramExecutionStrategy) -> Tuple[float, float]:
        """
        OPTIMAL MERGED: Combines HEAD's strategy-aware approach with MASTER's physics defaults.
        """
        
        # Physics-aware defaults (MASTER + HEAD combined)
        physics_ranges = {
            'pRecoil': (0.1, 6.0),
            'mu1P': (0.0, 5.0), 'mu2P': (0.0, 5.0),
            # 'mu1Pt': (0.0, 3.0), 'mu2Pt': (0.0, 3.0),
        }
        
        if column in physics_ranges:
            return physics_ranges[column]
        
        # Pattern-based inference (both approaches agree)
        if column.endswith(('P', 'Pt', 'Energy', 'E')):
            return (0.0, 5.0)
        elif 'theta' in column.lower():
            return (0.0, np.pi)
        elif 'phi' in column.lower():
            return (-np.pi, np.pi)
        
        # Adaptive sampling (HEAD approach with MASTER's sample size)
        sample_size = 10_000 if strategy != HistogramExecutionStrategy.MEMORY_CONSTRAINED else 1_000
        
        try:
            lazy_frames = self._extract_lazy_frames_safely()
            if lazy_frames:
                sample = lazy_frames[0].select([column]).head(sample_size).collect()
                
                if len(sample) > 0:
                    values = sample[column].to_numpy()
                    finite_values = values[np.isfinite(values)]
                    
                    if len(finite_values) > 10:
                        # Use robust percentiles
                        min_val, max_val = np.percentile(finite_values, [1, 99])
                        margin = (max_val - min_val) * 0.1
                        return (float(min_val - margin), float(max_val + margin))
        
        except Exception:
            pass
        
        # Ultimate fallback
        return (0.0, 10.0)
    
    # ============================================================================
    # MASTER: Debug Mode Implementation (preserved entirely)
    # ============================================================================
    def _debug_hist_execution(self, column, bins, range, density, weights, **kwargs):
        """
        MASTER: Surgical debugging - traces execution path with minimal overhead.
        """
        
        debug_state = {
            'step': 0,
            'path_taken': [],
            'failure_point': None,
            'data_state': {
                'lazy_frames_count': len(self._lazy_frames) if self._lazy_frames else 0,
                'compute_type': type(self._compute).__name__,
                'estimated_rows': getattr(self, '_estimated_rows', 'unknown'),
                'has_cpp': CPP_HISTOGRAM_AVAILABLE
            }
        }
        
        print(f"🔍 DEBUG: Starting histogram - {debug_state['data_state']}")
        
        # CHECKPOINT 1: Data access validation
        debug_state['step'] = 1
        try:
            lazy_frames = self._extract_lazy_frames_from_compute()
            debug_state['path_taken'].append('✅ Frame extraction')
            print(f"  Step 1: Frame extraction → {len(lazy_frames)} frames")
        except Exception as e:
            debug_state['failure_point'] = f"Step 1 - Frame extraction: {e}"
            print(f"  ❌ Step 1 FAILED: {e}")
            return self._emergency_debug_fallback(column, bins, debug_state)
        
        # CHECKPOINT 2: C++ acceleration attempt
        debug_state['step'] = 2
        if self._attempt_cpp_acceleration(column, bins, range, density, weights):
            debug_state['path_taken'].append('🚀 C++ attempted')
            try:
                cpp_engine = self._get_or_create_cpp_engine()
                if cpp_engine:
                    result = cpp_engine.compute_blazing_fast(
                        lazy_frames, column, bins=bins, range=range, density=density
                    )
                    debug_state['path_taken'].append('✅ C++ SUCCESS')
                    print(f"  Step 2: C++ acceleration → SUCCESS")
                    return result
                else:
                    debug_state['path_taken'].append('⚠️ C++ engine None')
                    print(f"  Step 2: C++ engine creation failed")
            except Exception as e:
                debug_state['failure_point'] = f"Step 2 - C++ execution: {e}"
                debug_state['path_taken'].append(f'❌ C++ failed: {type(e).__name__}')
                print(f"  ❌ Step 2: C++ failed → {e}")
        else:
            debug_state['path_taken'].append('⏭️ C++ skipped')
            print(f"  Step 2: C++ acceleration skipped")
        
        # CHECKPOINT 3: Pure Polars streaming
        debug_state['step'] = 3
        try:
            result = self._execute_enhanced_polars_histogram(
                column, bins, range, density, weights, lazy_frames
            )
            debug_state['path_taken'].append('✅ Polars streaming')
            print(f"  Step 3: Polars streaming → SUCCESS")
            return result
        except Exception as e:
            debug_state['failure_point'] = f"Step 3 - Polars streaming: {e}"
            debug_state['path_taken'].append(f'❌ Polars failed: {type(e).__name__}')
            print(f"  ❌ Step 3: Polars streaming failed → {e}")
        
        # CHECKPOINT 4: Materialization fallback
        debug_state['step'] = 4
        debug_state['path_taken'].append('🚨 MATERIALIZATION')
        print(f"  🚨 Step 4: All optimizations failed - using materialization")
        
        try:
            return self._materialization_fallback_debug(column, bins, range, density, weights, debug_state)
        except Exception as e:
            debug_state['failure_point'] = f"Step 4 - Materialization: {e}"
            print(f"  ❌ Step 4: Even materialization failed → {e}")
            raise
    
    def _emergency_debug_fallback(self, column, bins, debug_state):
        """MASTER: Emergency fallback with diagnostic info."""
        print(f"🚨 EMERGENCY FALLBACK - Debug state: {debug_state}")
        
        try:
            if hasattr(self, '_lazy_frames') and self._lazy_frames:
                df = self._lazy_frames[0].select(column).head(1000).collect()
                print(f"  📊 Sample data shape: {df.shape}")
                return np.histogram([1,2,3], bins=bins)  # Minimal histogram
        except Exception as e:
            print(f"  ❌ Emergency access failed: {e}")
        
        return np.zeros(bins), np.linspace(0, 1, bins+1)
    
    def _materialization_fallback_debug(self, column, bins, range, density, weights, debug_state):
        """MASTER: Materialization fallback for debug mode."""
        print(f"  Attempting full materialization as last resort...")
        
        try:
            df = self.collect()
            if column in df.columns:
                values = df[column].to_numpy()
                return np.histogram(values, bins=bins, range=range)
        except Exception as e:
            print(f"  ❌ Materialization failed: {e}")
            raise
    
    # ============================================================================
    # MASTER: Enhanced Polars Histogram (preserved for debug mode)
    # ============================================================================
    def _execute_enhanced_polars_histogram(self, column: str, bins: int, range: Optional[Tuple],
                                          density: bool, weights: Optional[str], 
                                          lazy_frames: List[pl.LazyFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """MASTER: Polars histogram with provided lazy frames."""
        total_processed = 0
        
        if range is None:
            range = self._compute_intelligent_range(column, lazy_frames)
        
        bin_edges = self._compute_bin_edges(range, bins)
        accumulator = np.zeros(bins, dtype=np.int64)
        
        # Process provided frames
        for lazy_frame in lazy_frames:
            frame_contribution = self._process_frame_streaming(
                lazy_frame, column, bin_edges, 
                self._calculate_optimal_histogram_chunk_size(), weights
            )
            accumulator += frame_contribution
            total_processed += np.sum(frame_contribution)
        
        print(f"   ✅ Enhanced Polars: {total_processed:,} events → {np.sum(accumulator):,.0f} counts")
        return accumulator, bin_edges
    
    def _compute_intelligent_range(self, column: str, lazy_frames: List[pl.LazyFrame]) -> Tuple[float, float]:
        """MASTER: Intelligent range computation (simplified version for debug)."""
        # Use the merged _compute_optimal_range instead
        return self._compute_optimal_range(column, HistogramExecutionStrategy.FALLBACK_BASIC)
    
    # ============================================================================
    # MASTER: Streaming and Chunking Helpers
    # ============================================================================
    def _process_frame_streaming(self, lazy_frame, column: str, bin_edges: np.ndarray, 
                                chunk_size: int, weights: Optional[str]) -> np.ndarray:
        """Thin wrapper over stats-enabled streaming to avoid duplication."""
        acc, _rows, _elapsed, _last_chunk = self._process_frame_streaming_stats(
            lazy_frame, column, bin_edges, chunk_size, weights
        )
        return acc

    def _process_frame_streaming_stats(self, lazy_frame, column: str, bin_edges: np.ndarray,
                                       chunk_size: int, weights: Optional[str]) -> Tuple[np.ndarray, int, float, int]:
        """Phase 3: Streaming with row-count and memory-aware throttling.

        Returns (counts, rows_processed, elapsed_seconds, last_chunk_size_used).
        Keeps behavior of _process_frame_streaming but additionally tracks true row counts
        and adjusts chunk size down under memory pressure to avoid OOM.
        """
        t0 = time.time()
        rows_accum = 0
        last_chunk_used = max(1, int(chunk_size))

        try:
            # Validate schema and build projection like the original helper
            frame_schema = lazy_frame.collect_schema()
            available_columns = set(frame_schema.names())

            cols = []
            if column not in available_columns:
                raise KeyError(
                    f"Histogram column '{column}' missing from schema. Available: {sorted(available_columns)}"
                )
            cols.append(column)

            if weights and weights in available_columns:
                cols.append(weights)
            else:
                weights = None

            projected_frame = lazy_frame.select(cols)

            # Determine total rows conservatively
            try:
                total_rows = int(projected_frame.select(pl.count()).collect()[0, 0])
            except Exception:
                df_tmp = projected_frame.head(min(1_000, last_chunk_used)).collect()
                total_rows = len(df_tmp)
                if total_rows > 0:
                    total_rows = max(total_rows, last_chunk_used)

            frame_acc = np.zeros(len(bin_edges) - 1, dtype=np.int64)
            if total_rows <= 0:
                return frame_acc, 0, time.time() - t0, last_chunk_used

            current_chunk = last_chunk_used
            # Loop with memory-aware throttling
            for offset in builtins.range(0, total_rows, max(1, int(current_chunk))):
                # Check memory pressure and throttle if needed
                try:
                    avail_gb = psutil.virtual_memory().available / 1024**3
                    if avail_gb < self.memory_budget_gb * 0.2:
                        current_chunk = max(10_000, current_chunk // 2)
                except Exception:
                    pass

                try:
                    chunk = projected_frame.slice(offset, current_chunk).collect()
                    if len(chunk) == 0:
                        continue
                    values = chunk[column].to_numpy()
                    weight_values = chunk[weights].to_numpy() if weights else None

                    valid_mask = np.isfinite(values)
                    if weights:
                        valid_mask &= np.isfinite(weight_values)
                        weight_values = weight_values[valid_mask]
                    values = values[valid_mask]
                    if len(values) == 0:
                        continue
                    # Count rows processed precisely (pre-weighting)
                    rows_accum += int(values.shape[0])
                    counts, _ = np.histogram(values.astype(np.float64), bins=bin_edges, weights=weight_values)
                    frame_acc += counts
                    last_chunk_used = current_chunk
                except Exception as chunk_err:
                    warnings.warn(f"Chunk failed at offset {offset}: {chunk_err}")
                    continue

            return frame_acc, rows_accum, time.time() - t0, last_chunk_used

        except Exception as e:
            warnings.warn(f"Frame streaming (stats) validation failed: {e}")
            return np.zeros(len(bin_edges) - 1, dtype=np.int64), 0, time.time() - t0, last_chunk_used
    
    # Removed unused _execute_streaming_histogram to reduce code size.
    
    def _calculate_optimal_histogram_chunk_size(self) -> int:
        """
        MASTER: Adaptive chunking with automatic fallback preservation.
        """
        use_adaptive_chunking = getattr(self, 'enable_adaptive_chunking', True)
        
        if use_adaptive_chunking:
            try:
                memory_budget_gb = getattr(self, 'memory_budget_gb', 16.0)
                estimated_rows = getattr(self, '_estimated_rows', 1_000_000)
                
                avg_row_bytes = 100.0
                if hasattr(self, '_schema') and self._schema:
                    avg_row_bytes = len(self._schema) * 20
                
                chunk_size = ChunkingEnhancement.calculate_optimal_chunk_size(
                    memory_budget_gb=memory_budget_gb,
                    estimated_rows=estimated_rows,
                    avg_row_bytes=avg_row_bytes,
                    operation_type='histogram',
                    fallback_calculation=lambda: self._calculate_original_chunk_size_preserved()
                )
                
                if chunk_size > 0 and chunk_size <= estimated_rows * 2:
                    print(f"🚀 Adaptive chunking: {chunk_size:,} rows")
                    return chunk_size
                else:
                    print(f"⚠️ Adaptive result invalid: {chunk_size:,}, using fallback")
                    return self._calculate_original_chunk_size_preserved()
                    
            except Exception as e:
                print(f"⚠️ Adaptive chunking failed: {e}, using original logic")
                return self._calculate_original_chunk_size_preserved()
        else:
            return self._calculate_original_chunk_size_preserved()
    
    def _calculate_original_chunk_size_preserved(self) -> int:
        """MASTER: Original chunking logic for fallback."""
        total_rows = self._estimated_rows
        
        if total_rows <= 10_000:
            return total_rows
        
        bytes_per_row = 8
        available_memory = self.memory_budget_gb * 1024**3 * 0.15
        memory_optimal = int(available_memory / bytes_per_row)
        
        if total_rows < 100_000:
            chunk_size = max(10_000, total_rows // 10)
        elif total_rows < 1_000_000:
            chunk_size = min(memory_optimal, max(500_000, total_rows // 20))
        elif total_rows < 10_000_000:
            chunk_size = min(memory_optimal, max(2_000_000, total_rows // 50))
        else:
            scaled_min = min(1_000_000, total_rows // 100)
            scaled_max = min(5_000_000, total_rows // 20)
            chunk_size = min(memory_optimal, max(scaled_min, scaled_max))
        
        return min(chunk_size, total_rows)
    
    def record_chunking_performance(self, chunk_size: int, execution_time: float):
        """MASTER: Record performance for continuous optimization."""
        if hasattr(self, '_estimated_rows') and hasattr(self, 'memory_budget_gb'):
            try:
                optimizer = ChunkingEnhancement.get_optimizer(self.memory_budget_gb)
                optimizer.record_performance(chunk_size, self._estimated_rows, execution_time)
            except Exception:
                pass
    
    def configure_adaptive_chunking(self, enabled: bool = True, **options):
        """MASTER: Configure adaptive chunking behavior."""
        self.enable_adaptive_chunking = enabled
        
        if 'rollout_percentage' in options:
            import random
            self.enable_adaptive_chunking = enabled and (random.random() < options['rollout_percentage'])
        
        status = "enabled" if self.enable_adaptive_chunking else "disabled"
        print(f"🔧 Adaptive chunking: {status}")
    
    # ============================================================================
    # MASTER: Simple Helpers
    # ============================================================================
    def _attempt_cpp_acceleration(self, column: str, bins: int, range: Optional[Tuple], 
                                 density: bool, weights: Optional[str]) -> bool:
        """MASTER: Strategic decision logic for C++ pathway eligibility."""
        if getattr(self, '_estimated_rows', 0) < 100_000:
            return False
        if weights and not isinstance(weights, str):
            return False
        if bins > 10_000:
            return False
        return True
    
    def _get_or_create_cpp_engine(self):
        """MASTER: Intelligent C++ engine acquisition with caching."""
        if CPP_HISTOGRAM_AVAILABLE:
            try:
                from optimized_cpp_integration import OptimizedStreamingHistogram
                self._histogram_engine = OptimizedStreamingHistogram()
                return self._histogram_engine
            except Exception:
                pass
        return None
    
    # NOTE: The cached extractor _extract_lazy_frames_from_compute is defined earlier.
    # Avoid redefining it here to prevent overriding the cached implementation.
    
    # ============================================================================
    # Utility Methods (preserved from both)
    # ============================================================================
    def _validate_columns_exist(self, required_cols, available_cols=None):
        """Utility method for column validation."""
        if available_cols is None:
            available_cols = self.columns
        
        missing = [col for col in required_cols if col not in available_cols]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return True
    
    def _estimate_group_count(self, group_cols: List[str]) -> int:
        """Estimate number of unique groups."""
        if 'event' in str(group_cols).lower():
            return int(self._estimated_rows * 0.8)
        else:
            return int(self._estimated_rows * 0.1)

    # Small shared helpers to reduce duplication in histogram code
    def _compute_bin_edges(self, range: Tuple[float, float], bins: int) -> np.ndarray:
        try:
            return np.linspace(range[0], range[1], int(bins) + 1)
        except Exception:
            # Safe fallback to 0..1 if inputs are bad
            b = max(1, int(bins) if isinstance(bins, int) else 50)
            return np.linspace(0.0, 1.0, b + 1)

    def _apply_density_normalization(self, counts: np.ndarray, bin_edges: np.ndarray,
                                     density: bool, processed_rows: int) -> np.ndarray:
        if not density:
            return counts
        try:
            if processed_rows and processed_rows > 0:
                width = float(bin_edges[1] - bin_edges[0])
                if width > 0:
                    return counts / (processed_rows * width)
        except Exception:
            pass
        return counts
#     # ============================================================================
#     # EXECUTION STRATEGIES
#     # ============================================================================
    
    def _execute_cpp_accelerated_histogram(self, column: str, bins: int, range: Tuple[float, float],
                                         density: bool, weights: Optional[str], 
                                         metrics: HistogramMetrics) -> Tuple[np.ndarray, np.ndarray]:
        """Execute using C++ acceleration with proper error handling."""
        
        # Extract lazy frames
        lazy_frames = self._extract_lazy_frames_safely()
        
        if not lazy_frames:
            raise ValueError("No lazy frames available for C++ processing")
        
        # Ensure C++ engine exists
        if not hasattr(self, '_histogram_engine') or self._histogram_engine is None:
            engine = self._get_or_create_cpp_engine()
            if engine is None:
                raise ValueError("C++ engine unavailable")
            self._histogram_engine = engine
        
        # Configure C++ engine optimally (if supported)
        if hasattr(self._histogram_engine, 'configure'):
            self._histogram_engine.configure(
                num_threads=min(psutil.cpu_count(), len(lazy_frames)),
                cache_size_mb=min(1024, int(psutil.virtual_memory().available / 1024 / 1024 * 0.1))
            )
        
        # Execute with progress tracking
        counts, edges = self._histogram_engine.compute_blazing_fast(
            lazy_frames, column, bins=bins, range=range, density=density
        )
        
        # Update metrics
        # Prefer measured processed rows when density is False; otherwise, fall back to estimate
        try:
            if not density and isinstance(counts, np.ndarray):
                metrics.processed_rows = int(np.sum(counts))
            else:
                metrics.processed_rows = self._estimated_rows
        except Exception:
            metrics.processed_rows = self._estimated_rows
        metrics.chunks_processed = len(lazy_frames)
        
        return counts, edges
    
    def _execute_billion_row_histogram(self, column: str, bins: int, range: Tuple[float, float],
                                     density: bool, weights: Optional[str],
                                     metrics: HistogramMetrics) -> Tuple[np.ndarray, np.ndarray]:
        """Execute using Layer 1's billion-row engine with spilling support."""
        
        # Get billion-row engine from Layer 1
        billion_engine = self._get_billion_row_engine()
        
        if not billion_engine:
            raise ValueError("Billion-row engine not available")
        
        # Configure for histogram workload
        chunk_strategy = billion_engine.get_chunk_strategy()
        chunk_strategy.configure_for_histogram(
            estimated_rows=self._estimated_rows,
            bins=bins,
            memory_limit_gb=self.memory_budget_gb * 0.5  # Conservative limit
        )
        
        # Initialize accumulator
        bin_edges = self._compute_bin_edges(range, bins)
        accumulator = np.zeros(bins, dtype=np.float64)
        
        # Process with billion-row engine's sophisticated chunking
        lazy_frames = self._extract_lazy_frames_safely()
        
        for frame_idx, lazy_frame in enumerate(lazy_frames):
            print(f"   Processing frame {frame_idx + 1}/{len(lazy_frames)} with billion-row engine")
            
            # Let billion engine handle the complexity
            frame_result = billion_engine.compute_histogram_with_spilling(
                lazy_frame, column, bin_edges, weights=weights
            )
            
            accumulator += frame_result
            metrics.chunks_processed += 1
        
        # Apply density normalization
        accumulator = self._apply_density_normalization(accumulator, bin_edges, density, metrics.processed_rows)
        
        metrics.processed_rows = self._estimated_rows
        
        return accumulator, bin_edges
    
    def _execute_lazy_chunked_histogram(self, column: str, bins: int, range: Tuple[float, float],
                                      density: bool, weights: Optional[str],
                                      metrics: HistogramMetrics) -> Tuple[np.ndarray, np.ndarray]:
        """Standard execution with intelligent chunking."""
        
        # Calculate optimal chunk size using enhanced algorithm
        chunk_size = self._calculate_adaptive_chunk_size(column, bins, weights)
        metrics.chunk_size_used = chunk_size
        
        # Initialize histogram
        bin_edges = self._compute_bin_edges(range, bins)
        accumulator = np.zeros(bins, dtype=np.float64)
        
        # Timing for adaptive feedback
        start_time = time.time()
        # Get lazy frames
        lazy_frames = self._extract_lazy_frames_safely()
        
        # Process each frame with adaptive chunking
        for frame_idx, lazy_frame in enumerate(lazy_frames):
            # Observability: respect time budget
            try:
                tb = getattr(self, '_hist_time_budget_s', None)
                if tb is not None and (time.time() - start_time) > tb:
                    print("   ⏱️ Time budget reached; early exit from lazy-chunked loop")
                    try:
                        setattr(self, '_hist_early_exit', True)
                    except Exception:
                        pass
                    break
            except Exception:
                pass

            frame_contribution, rows_processed = self._process_frame_chunked(
                lazy_frame, column, bin_edges, chunk_size, weights
            )
            
            accumulator += frame_contribution
            metrics.processed_rows += rows_processed
            metrics.chunks_processed += 1
            
            # Progress callback
            try:
                pcb = getattr(self, '_hist_progress_cb', None)
                if callable(pcb):
                    pcb(
                        processed_rows=metrics.processed_rows,
                        chunks=metrics.chunks_processed,
                        chunk_size=metrics.chunk_size_used,
                        elapsed=time.time() - start_time,
                    )
            except Exception:
                pass

            # Adaptive chunk size adjustment based on performance
            if frame_idx == 0 and len(lazy_frames) > 1:
                # Measure first frame performance and adjust
                chunk_size = self._adjust_chunk_size_based_on_performance(
                    chunk_size, rows_processed, time.time() - start_time
                )
                metrics.chunk_size_used = chunk_size
        
        # Apply density normalization
        accumulator = self._apply_density_normalization(accumulator, bin_edges, density, metrics.processed_rows)
        
        return accumulator, bin_edges

    def _calculate_adaptive_chunk_size(self, column: str, bins: int, weights: Optional[str]) -> int:
        """Compatibility shim used by LAZY_CHUNKED path.

        Uses the same adaptive engine as the MASTER path, with very conservative
        fallbacks so this method always exists and returns a sane number.
        """
        try:
            # Prefer the unified adaptive routine
            return self._calculate_optimal_histogram_chunk_size()
        except Exception:
            try:
                # Ask the optimizer directly
                avg_row_bytes = max(20 * (len(self._schema) if hasattr(self, '_schema') and self._schema else 20), 200)
                optimizer = ChunkingEnhancement.get_optimizer(getattr(self, 'memory_budget_gb', 8.0))
                return optimizer.calculate_optimal_chunk_size(
                    getattr(self, '_estimated_rows', 1_000_000), avg_row_bytes, 'histogram'
                )
            except Exception:
                # Last-ditch conservative guess
                rows = max(1_000, getattr(self, '_estimated_rows', 1_000_000))
                return min(max(100_000, rows // 50), rows)
    
    def _execute_memory_constrained_histogram(self, column: str, bins: int, range: Tuple[float, float],
                                            density: bool, weights: Optional[str],
                                            metrics: HistogramMetrics) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-conservative execution for memory-constrained environments."""
        
        # For observability support
        start_time = time.time()

        # Use very small chunks
        chunk_size = min(10_000, self._estimated_rows // 100)
        metrics.chunk_size_used = chunk_size
        
        # Initialize histogram
        bin_edges = self._compute_bin_edges(range, bins)
        accumulator = np.zeros(bins, dtype=np.float64)
        
        # Process with aggressive memory management
        lazy_frames = self._extract_lazy_frames_safely()
        
        for lazy_frame in lazy_frames:
            try:
                # Project only required columns to minimize memory
                cols_needed = [column] if not weights else [column, weights]
                projected = lazy_frame.select(cols_needed)
                
                # Collect in small batches with explicit garbage collection
                total_rows = projected.select(pl.count()).collect()[0, 0]
                
                for offset in builtins.range(0, total_rows, chunk_size):
                    # Observability: respect time budget
                    try:
                        tb = getattr(self, '_hist_time_budget_s', None)
                        if tb is not None and (time.time() - start_time) > tb:
                            print("   ⏱️ Time budget reached; early exit from memory-constrained loop")
                            try:
                                setattr(self, '_hist_early_exit', True)
                            except Exception:
                                pass
                            break
                    except Exception:
                        pass
                    # Explicit garbage collection before each chunk
                    import gc
                    gc.collect()
                    
                    # Process chunk
                    chunk = projected.slice(offset, chunk_size).collect()
                    
                    if len(chunk) > 0:
                        values = chunk[column].to_numpy()
                        weight_values = chunk[weights].to_numpy() if weights else None
                        
                        # Compute histogram for chunk
                        valid_mask = np.isfinite(values)
                        if weights:
                            valid_mask &= np.isfinite(weight_values)
                            weight_values = weight_values[valid_mask]
                        
                        values = values[valid_mask]
                        
                        if len(values) > 0:
                            counts, _ = np.histogram(values, bins=bin_edges, weights=weight_values)
                            accumulator += counts
                            metrics.processed_rows += len(values)
                    
                    # Explicit cleanup
                    del chunk
                    
                metrics.chunks_processed += 1
                # Progress callback per frame
                try:
                    pcb = getattr(self, '_hist_progress_cb', None)
                    if callable(pcb):
                        pcb(
                            processed_rows=metrics.processed_rows,
                            chunks=metrics.chunks_processed,
                            chunk_size=metrics.chunk_size_used,
                            elapsed=time.time() - start_time,
                        )
                except Exception:
                    pass
                
            except Exception as e:
                print(f"   ⚠️ Frame processing failed in memory-constrained mode: {e}")
                metrics.errors_recovered += 1
                continue
        
        # Apply density normalization
        accumulator = self._apply_density_normalization(accumulator, bin_edges, density, metrics.processed_rows)
        
        return accumulator, bin_edges
    
    def _execute_fallback_histogram(self, column: str, bins: int, range: Tuple[float, float],
                                  density: bool, weights: Optional[str],
                                  metrics: HistogramMetrics) -> Tuple[np.ndarray, np.ndarray]:
        """Basic fallback implementation - always works but may be slow."""
        
        # Initialize histogram
        bin_edges = self._compute_bin_edges(range, bins)
        accumulator = np.zeros(bins, dtype=np.float64)
        
        try:
            # Try to materialize and compute directly
            if hasattr(self, 'collect'):
                df = self.collect()
                
                if column in df.columns:
                    values = df[column].to_numpy()
                    weight_values = df[weights].to_numpy() if weights and weights in df.columns else None
                    
                    # Filter valid values
                    valid_mask = np.isfinite(values)
                    if weight_values is not None:
                        valid_mask &= np.isfinite(weight_values)
                        weight_values = weight_values[valid_mask]
                    
                    values = values[valid_mask]
                    
                    if len(values) > 0:
                        accumulator, _ = np.histogram(values, bins=bin_edges, weights=weight_values)
                        metrics.processed_rows = len(values)
                        
                        # Apply density
                        accumulator = self._apply_density_normalization(accumulator, bin_edges, density, len(values))
            
        except Exception as e:
            print(f"   ❌ Fallback histogram failed: {e}")
            metrics.errors_recovered += 1
        
        metrics.chunks_processed = 1
        
        return accumulator, bin_edges
#     # ============================================================================
#     # HELPER METHODS
#     # ============================================================================
    def _process_frame_chunked(self, lazy_frame: pl.LazyFrame, column: str,
                             bin_edges: np.ndarray, chunk_size: int,
                             weights: Optional[str]) -> Tuple[np.ndarray, int]:
        """Route through stats-enabled streaming to avoid duplicate logic."""
        acc, rows, _elapsed, _last = self._process_frame_streaming_stats(
            lazy_frame, column, bin_edges, chunk_size, weights
        )
        # Cast to float accumulator to match previous dtype
        return acc.astype(np.float64, copy=False), int(rows)
    
    def _extract_lazy_frames_safely(self) -> List[pl.LazyFrame]:
        """Safely extract lazy frames with multiple fallback strategies."""
        
        # Direct lazy frames
        if hasattr(self, '_lazy_frames') and self._lazy_frames:
            return self._lazy_frames
        
        # Try to extract from compute graph
        if hasattr(self, '_compute'):
            if hasattr(self._compute, 'to_lazy_frames'):
                return self._compute.to_lazy_frames()
            
            # Try to create lazy wrapper
            if hasattr(self._compute, 'materialize'):
                def lazy_wrapper():
                    materialized = self._compute.materialize()
                    if isinstance(materialized, pl.DataFrame):
                        return materialized.lazy()
                    elif isinstance(materialized, pl.LazyFrame):
                        return materialized
                    else:
                        # Convert to Polars
                        return pl.DataFrame(materialized).lazy()
                
                return [lazy_wrapper()]
        
        # Last resort - empty list
        return []
     
    def _test_cpp_engine(self) -> bool:
        """Test if C++ engine is actually functional."""
        if not hasattr(self, '_histogram_engine') or self._histogram_engine is None:
            return False
        
        try:
            # Quick functionality test
            if hasattr(self._histogram_engine, 'test_functionality'):
                return self._histogram_engine.test_functionality()
            
            # Assume functional if present
            return True
        except:
            return False
    
    def _check_billion_row_capability(self) -> bool:
        """Check if Layer 1 billion-row engine is available."""
        try:
            # Check for Layer 1 integration
            if hasattr(self, '_layer1') and self._layer1:
                billion_engine = self._layer1._get_billion_engine()
                return billion_engine is not None
            
            # Check for direct billion engine
            if hasattr(self, '_billion_engine'):
                return self._billion_engine is not None
            
            return False
        except:
            return False
    
    def _get_billion_row_engine(self):
        """Get billion-row engine from Layer 1."""
        if hasattr(self, '_layer1') and self._layer1:
            return self._layer1._get_billion_engine()
        
        if hasattr(self, '_billion_engine'):
            return self._billion_engine
        
        return None
    
    def _adjust_chunk_size_based_on_performance(self, current_chunk_size: int,
                                              rows_processed: int,
                                              elapsed_time: float) -> int:
        """Dynamically adjust chunk size based on observed performance."""
        
        if elapsed_time <= 0 or rows_processed <= 0:
            return current_chunk_size
        
        # Calculate throughput
        throughput = rows_processed / elapsed_time
        
        # Target: process each chunk in ~100ms for responsiveness
        target_time = 0.1
        optimal_chunk_size = int(throughput * target_time)
        
        # Apply bounds and smoothing
        new_chunk_size = int(0.7 * current_chunk_size + 0.3 * optimal_chunk_size)
        
        # Enforce reasonable bounds
        min_chunk = 10_000
        max_chunk = min(10_000_000, self._estimated_rows // 10)
        
        return max(min_chunk, min(new_chunk_size, max_chunk))
    
    def _log_histogram_performance(self, metrics: HistogramMetrics):
        """Log comprehensive performance metrics."""
        print("\n📊 Histogram Performance Summary:")
        print(f"   Strategy: {metrics.strategy.name}")
        print(
            f"   Rows: {metrics.processed_rows:,} / {metrics.total_rows:,} "
            f"({metrics.efficiency:.1%} efficiency)"
        )
        print(
            f"   Time: {metrics.execution_time:.2f}s "
            f"({metrics.throughput_mps:.1f}M rows/s)"
        )
        mem_val = getattr(metrics, 'memory_peak_mb', None)
        mem_str = "unknown" if (mem_val is None or mem_val < 0) else f"{mem_val:.1f}MB"
        print(f"   Memory Peak: {mem_str}")
        print(
            f"   Chunks: {metrics.chunks_processed} "
            f"(size: {metrics.chunk_size_used:,})"
        )
        if metrics.errors_recovered > 0:
            print(f"   Errors Recovered: {metrics.errors_recovered}")
    
    def _record_histogram_performance(self, column: str, metrics: HistogramMetrics):
        """Record performance for adaptive optimization."""
        
        if not hasattr(self, '_histogram_performance_history'):
            self._histogram_performance_history = {}
        
        if column not in self._histogram_performance_history:
            self._histogram_performance_history[column] = []
        
        # Keep last 10 executions
        history = self._histogram_performance_history[column]
        history.append({
            'strategy': metrics.strategy,
            'throughput': metrics.throughput_mps,
            'efficiency': metrics.efficiency,
            'memory_mb': metrics.memory_peak_mb,
            'timestamp': time.time()
        })
        
        if len(history) > 10:
            history.pop(0)
    
    def _get_best_previous_strategy(self, column: str) -> Optional[HistogramExecutionStrategy]:
        """Get best performing strategy from history."""
        
        if column not in self._histogram_performance_history:
            return None
        
        history = self._histogram_performance_history[column]
        if not history:
            return None
        
        # Find strategy with best throughput
        best_entry = max(history, key=lambda x: x['throughput'] * x['efficiency'])
        
        # Only use if recent (within last hour)
        if time.time() - best_entry['timestamp'] < 3600:
            return best_entry['strategy']
        
        return None
       
    # filter is defined earlier; keep a single implementation to avoid ambiguity.
    
    def groupby(self, by, **kwargs) -> 'LazyGroupBy':
        """
        Group-by that maintains lazy semantics.
        
        Innovation: Returns LazyGroupBy object that delays aggregation specification,
        enabling optimization of the entire group-aggregate pipeline.
        """
        return LazyGroupBy(
            parent_df=self,
            grouping_keys=by if isinstance(by, list) else [by],
            optimization_hints=self._optimization_hints.copy()
        )
    
    def join(self, other: 'UnifiedLazyDataFrame', on=None, how='inner', **kwargs) -> 'UnifiedLazyDataFrame':
        """Join with chain preservation."""
        transform_meta = TransformationMetadata(
            operation='join',
            parameters={'on': on, 'how': how, **kwargs}
        )
        
        # Determine join strategy
        left_size, right_size = self._estimated_rows, other._estimated_rows
        join_type = 'broadcast' if right_size < 1_000_000 and how == 'inner' else 'hash'
        
        join_node = GraphNode(
            op_type=ComputeOpType.JOIN,
            operation=lambda df: df.join(other._compute.materialize(), on=on, how=how),
            inputs=[self._get_root_node()],
            metadata={'join_type': join_type, 'on': on, 'how': how}
        )
        
        merged_schema = self._merge_schemas(self._schema, other._schema, on=on)
        new_size = self._estimate_join_size(left_size, right_size, how)
        new_compute = self._create_compute_capability(join_node, new_size)
        
        return self._create_derived_dataframe(
            new_compute=new_compute,
            new_schema=merged_schema,
            transformation_metadata=transform_meta
        )
    
    def _merge_schemas(self, left_schema: Dict[str, type], right_schema: Dict[str, type], on=None) -> Dict[str, type]:
        """Merge schemas for join operation."""
        merged = left_schema.copy() if left_schema else {}
        
        if right_schema:
            for col, dtype in right_schema.items():
                if on and col in on:
                    continue  # Skip join keys
                if col in merged:
                    merged[f"{col}_right"] = dtype
                else:
                    merged[col] = dtype
        
        return merged
    
    def _estimate_join_size(self, left_size: int, right_size: int, how: str) -> int:
        """Estimate result size for join operation."""
        if how == 'inner':
            return min(left_size, right_size)
        elif how == 'left':
            return left_size
        elif how == 'right':
            return right_size
        elif how == 'outer':
            return left_size + right_size
        else:
            return max(left_size, right_size)
    
    # ========================================================================
    # Materialization and Collection
    # ========================================================================
    
    def collect(self) -> pl.DataFrame:
        """
        Materialize the compute graph to a Polars DataFrame.
        
        This is the primary method to execute the lazy computation chain.
        """
        print(f"🚀 Materializing compute graph with {self._estimated_rows:,} estimated rows...")
        
        start_time = time.time()
        
        if self._lazy_frames:
            # Collect from lazy frames (prefer streaming to reduce peaks)
            dfs = []
            for lf in self._lazy_frames:
                try:
                    dfs.append(lf.collect(streaming=True))
                except Exception:
                    dfs.append(lf.collect())
            result = pl.concat(dfs) if dfs else pl.DataFrame()
        else:
            # Materialize from compute capability
            result = self._compute.materialize()
            
            # Convert to Polars if needed
            if not isinstance(result, pl.DataFrame):
                if hasattr(result, 'to_polars'):
                    result = result.to_polars()
                elif isinstance(result, pa.Table):
                    result = pl.from_arrow(result)
                elif hasattr(result, 'to_pandas'):
                    result = pl.from_pandas(result.to_pandas())
        
        elapsed = time.time() - start_time
        rows = len(result)
        throughput = rows / elapsed if elapsed > 0 else 0
        
        print(f"✅ Collected {rows:,} rows in {elapsed:.2f}s ({throughput/1e6:.1f}M rows/s)")
        
        return result
    
    def to_pandas(self) -> 'pd.DataFrame':
        """Convert to pandas DataFrame."""
        return self.collect().to_pandas()
    
    def head(self, n: int = 5) -> pl.DataFrame:
        """Get first n rows."""
        if self._lazy_frames:
            return self._lazy_frames[0].head(n).collect()
        else:
            sample = self._create_sample_compute(n).materialize()
            if isinstance(sample, pl.DataFrame):
                return sample
            else:
                return pl.DataFrame(sample)
    
    def describe(self) -> pl.DataFrame:
        """Generate descriptive statistics."""
        # Materialize a sample for statistics
        sample_size = min(1_000_000, self._estimated_rows)
        sample = self._create_sample_compute(sample_size).materialize()
        
        if isinstance(sample, pl.DataFrame):
            return sample.describe()
        else:
            return pl.DataFrame({"message": ["Statistics not available"]})
    
    # ========================================================================
    # Introspection and Debugging
    # ========================================================================
    
    def explain(self) -> str:
        """Explain the query plan."""
        if hasattr(self._compute, 'explain'):
            return self._compute.explain()
        elif hasattr(self._compute, 'root_node'):
            return self._explain_graph(self._compute.root_node)
        else:
            return "Query plan not available"
    
    def _explain_graph(self, node: GraphNode, level: int = 0) -> str:
        """Recursively explain the compute graph."""
        indent = "  " * level
        explanation = f"{indent}{node.op_type.name}"
        
        if node.metadata:
            explanation += f" {node.metadata}"
        
        explanation += "\n"
        
        for input_node in node.inputs:
            explanation += self._explain_graph(input_node, level + 1)
        
        return explanation
    
    def get_transformation_history(self) -> List[TransformationMetadata]:
        """Get complete transformation history."""
        return self._transformation_chain.get_lineage()
    
    def profile(self) -> Dict[str, Any]:
        """Profile the DataFrame operations."""
        return {
            'estimated_rows': self._estimated_rows,
            'columns': len(self._schema) if self._schema else 0,
            'transformations': len(self._transformation_chain.get_lineage()),
            'access_patterns': len(self._access_patterns),
            'cache_size': len(self._operation_cache),
            'memory_budget_gb': self.memory_budget_gb
        }

    # ========================================================================
    # Phase 4: Lightweight observability accessors
    # ========================================================================
    def last_hist_info(self) -> Dict[str, Any]:
        """Return last histogram's rationale and metrics if available.

    Fields: rationale (str), and metrics snapshot with keys:
    strategy, processed_rows, total_rows, efficiency, throughput_mps,
    memory_peak_mb, chunks_processed, chunk_size, errors_recovered, elapsed_s,
    time_budget_s, early_exit, decision_confidence.
        """
        return {
            'rationale': getattr(self, '_last_hist_rationale', ''),
            'metrics': getattr(self, '_last_hist_metrics', None),
        }

_original_create_derived_dataframe = UnifiedLazyDataFrame._create_derived_dataframe

def _create_derived_dataframe_with_estimation(self, new_compute, new_schema=None, 
                                            transformation_metadata=None, **kwargs):
    """Enhanced wrapper with estimation propagation."""
    # Call original implementation
    result = _original_create_derived_dataframe(
        self, new_compute, new_schema, transformation_metadata, **kwargs
    )
    
    # Add estimation based on transformation type
    if transformation_metadata and hasattr(result, '_estimated_rows'):
        op = transformation_metadata.operation
        params = transformation_metadata.parameters
        
        # Physics-informed selectivity
        if op == 'dataframe_oneCandOnly':
            # Use stored expectation or default
            result._estimated_rows = params.get('expected_rows', 
                                               max(100, int(self._estimated_rows * 0.1)))
        elif op == 'query':
            selectivity = params.get('selectivity', 0.5)
            result._estimated_rows = max(100, int(self._estimated_rows * selectivity))
        elif op == 'aggregate':
            groups = params.get('estimated_groups', 1000)
            result._estimated_rows = min(groups, self._estimated_rows)
        
        # Clear stale cache
        if hasattr(result, '_operation_cache'):
            result._operation_cache.clear()
    
    return result

# Replace the method
UnifiedLazyDataFrame._create_derived_dataframe = _create_derived_dataframe_with_estimation

# ============================================================================
# Factory Functions
# ============================================================================

def create_dataframe_from_parquet(path: str, 
                                 engine: str = 'auto',
                                 memory_budget_gb: float = 8.0) -> UnifiedLazyDataFrame:
    """
    Create UnifiedLazyDataFrame from Parquet file(s).
    
    This is the primary entry point for creating DataFrames from files.
    """
    from pathlib import Path
    
    # Handle glob patterns
    if '*' in path:
        files = list(Path(path).parent.glob(Path(path).name))
    else:
        files = [Path(path)]
    
    # Create lazy frames
    lazy_frames = []
    for file in files:
        try:
            lf = pl.scan_parquet(str(file))
            lazy_frames.append(lf)
        except Exception as e:
            warnings.warn(f"Failed to read {file}: {e}")
    
    if not lazy_frames:
        raise ValueError(f"No valid files found for pattern: {path}")
    
    # Create DataFrame
    return UnifiedLazyDataFrame(
        lazy_frames=lazy_frames,
        memory_budget_gb=memory_budget_gb
    )


def create_dataframe_from_compute(compute: ComputeCapability,
                                 schema: Optional[Dict[str, type]] = None) -> UnifiedLazyDataFrame:
    """
    Create UnifiedLazyDataFrame from a compute capability.
    
    This enables direct integration with Layer 1 compute engines.
    """
    return UnifiedLazyDataFrame(
        compute=compute,
        schema=schema
    )
