"""Layer 2: UnifiedLazyDataFrame - Lazy GroupBy Operations
============================================================
This module provides a lazy group-by operation that allows for delayed aggregation specification.
Class:
    LazyGroupBy: Represents a lazy group-by operation that delays aggregation specification.
"""
from __future__ import annotations
import polars as pl
from typing import List, Dict, Any, Optional,TYPE_CHECKING
if TYPE_CHECKING:
    from .core import UnifiedLazyDataFrame
from layer0 import  ComputeOpType

from .utils import TransformationMetadata
from layer1.lazy_compute_engine import (
    LazyComputeCapability, GraphNode
)
class LazyGroupBy:
    """
    Lazy group-by operation that delays aggregation specification.
    
    This enables optimization of the entire group-aggregate pipeline
    by seeing all operations before execution.
    """
    
    def __init__(self, 
                 parent_df: UnifiedLazyDataFrame,
                 grouping_keys: List[str],
                 optimization_hints: Optional[Dict[str, Any]] = None):
        """
        Initialize lazy group-by.
        
        Args:
            parent_df: Parent DataFrame
            grouping_keys: Columns to group by
            optimization_hints: Hints for optimization
        """
        self._parent_df = parent_df
        self._grouping_keys = grouping_keys
        self._optimization_hints = optimization_hints or {}
        self._aggregations = []
    
    def agg(self, *aggregations, **named_aggregations) -> UnifiedLazyDataFrame:
        """
        Specify aggregations and execute group-by.
        
        Supports both positional and named aggregations.
        """
        # Collect all aggregations
        agg_specs = list(aggregations)
        
        for name, agg in named_aggregations.items():
            if isinstance(agg, str):
                # Simple aggregation like 'mean', 'sum'
                agg_specs.append(pl.col(name).agg(agg).alias(f"{name}_{agg}"))
            else:
                agg_specs.append(agg.alias(name))
        
        # Create aggregation operation
        def group_aggregate(df):
            return df.group_by(self._grouping_keys).agg(agg_specs)
        
        # Create aggregation node
        agg_node = GraphNode(
            op_type=ComputeOpType.AGGREGATE,
            operation=group_aggregate,
            inputs=[self._parent_df._get_root_node()],
            metadata={
                'group_by': self._grouping_keys,
                'aggregations': len(agg_specs),
                'memory_intensive': True
            }
        )
        
        # Estimate result size
        estimated_groups = self._parent_df._estimate_group_count(self._grouping_keys)
        
        # Create new compute capability
        if isinstance(self._parent_df._compute, LazyComputeCapability):
            new_compute = LazyComputeCapability(
                root_node=agg_node,
                engine=self._parent_df._compute.engine,
                estimated_size=estimated_groups,
                schema=self._infer_agg_schema(agg_specs)
            )
        else:
            new_compute = self._parent_df._compute.transform(group_aggregate)
        
        # Create transformation metadata
        transform_meta = TransformationMetadata(
            operation='group_aggregate',
            parameters={
                'group_by': self._grouping_keys,
                'aggregations': str(agg_specs)
            },
            parent_id=self._parent_df._transformation_metadata.id if self._parent_df._transformation_metadata else None
        )
        
        return UnifiedLazyDataFrame(
            compute=new_compute,
            schema=self._infer_agg_schema(agg_specs),
            metadata=self._parent_df._metadata.copy(),
            memory_budget_gb=self._parent_df.memory_budget_gb,
            materialization_threshold=self._parent_df.materialization_threshold,
            transformation_metadata=transform_meta,
             parent_chain=self._transformation_chain.get_lineage() 
        )
    
    def _infer_agg_schema(self, agg_specs: List[Any]) -> Dict[str, type]:
        """Infer schema for aggregation result."""
        schema = {}
        
        # Include grouping keys
        if self._parent_df._schema:
            for key in self._grouping_keys:
                if key in self._parent_df._schema:
                    schema[key] = self._parent_df._schema[key]
        
        # Add aggregation columns (simplified inference)
        for i, agg in enumerate(agg_specs):
            if hasattr(agg, 'alias'):
                schema[f"agg_{i}"] = float  # Most aggregations produce numeric results
        
        return schema
    
    def count(self) -> UnifiedLazyDataFrame:
        """Shorthand for count aggregation."""
        return self.agg(pl.count().alias('count'))
    
    def mean(self, *columns) -> UnifiedLazyDataFrame:
        """Shorthand for mean aggregation."""
        if columns:
            return self.agg(*[pl.col(c).mean().alias(f"{c}_mean") for c in columns])
        else:
            return self.agg(pl.all().mean())
    
    def sum(self, *columns) -> UnifiedLazyDataFrame:
        """Shorthand for sum aggregation."""
        if columns:
            return self.agg(*[pl.col(c).sum().alias(f"{c}_sum") for c in columns])
        else:
            return self.agg(pl.all().sum())