"""
Layer 0: Core Protocols and Interfaces
=====================================

Foundational protocols for the compute-first architecture. This layer defines
the theoretical foundation where computation is the primary abstraction.

Design Philosophy:
- Computation as first-class citizen, not capability of data structures
- Protocol-based design for maximum flexibility
- Type-safe interfaces with generic support
- Lazy evaluation semantics throughout
- Memory-aware compute contracts

Theoretical Foundation:
- Based on category theory's F-algebras where computations form morphisms
- Follows principle of referential transparency
- Implements lazy evaluation as core semantic
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Union, Iterator, Callable, 
    TypeVar, Generic, Protocol, runtime_checkable, Tuple, Set
)
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

# Type variables for generic programming
T = TypeVar('T')
TResult = TypeVar('TResult')
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')

# ============================================================================
# CORE OPERATION TYPES
# ============================================================================

class ComputeOpType(Enum):
    """Enumeration of fundamental compute operation types."""
    MAP = auto()          # Element-wise transformation
    FILTER = auto()       # Predicate-based selection
    REDUCE = auto()       # Aggregation operation
    JOIN = auto()         # Relational join
    SORT = auto()         # Ordering operation
    AGGREGATE = auto()    # Group-based aggregation
    WINDOW = auto()       # Windowed operation
    TRANSFORM = auto()    # Complex transformation
    MATERIALIZE = auto()  # Force evaluation
    PARTITION = auto()    # Data partitioning
    COLLECT = auto()      # Result collection
    BROADCAST = auto()    # Data broadcasting
    CUSTOM = auto()       # User-defined operation

# ============================================================================
# FOUNDATIONAL COMPUTE PROTOCOLS
# ============================================================================

@runtime_checkable
class ComputeCapability(Protocol, Generic[T]):
    """
    Core protocol defining what it means to be computable.
    
    This is the foundational abstraction - everything that can be computed
    must implement this protocol. Data structures are manifestations of
    compute capabilities, not containers with compute methods.
    """
    
    def transform(self, operation: Callable[[T], TResult]) -> ComputeCapability[TResult]:
        """Apply a transformation to create new compute capability."""
        ...
    
    def materialize(self) -> T:
        """Force evaluation and return concrete result."""
        ...
    
    def estimate_memory(self) -> int:
        """Estimate memory requirements in bytes."""
        ...
    
    def partition_compute(self, strategy: str = 'auto') -> List[ComputeCapability[T]]:
        """Partition computation for parallel execution."""
        ...
    
    def compose(self, other: ComputeCapability) -> ComputeCapability:
        """Compose with another compute capability."""
        ...


@runtime_checkable
class DistributedComputeCapability(ComputeCapability[T], Protocol):
    """Extended capability for distributed computation."""
    
    def distribute_across(self, nodes: int) -> ComputeCapability[T]:
        """Distribute computation across multiple nodes."""
        ...
    
    def gather_results(self) -> ComputeCapability[T]:
        """Gather distributed computation results."""
        ...


@runtime_checkable
class ComputeEngine(Protocol):
    """
    Protocol for compute engines that create and manage capabilities.
    
    Compute engines are factories for compute capabilities and manage
    the execution environment.
    """
    
    def create_capability(self, operation: Callable, 
                         inputs: List[ComputeCapability] = None) -> ComputeCapability:
        """Create a new compute capability from operation and inputs."""
        ...
    
    def optimize_plan(self, capability: ComputeCapability) -> ComputeCapability:
        """Optimize the computation plan."""
        ...
    
    def estimate_resources(self, capability: ComputeCapability) -> Dict[str, Any]:
        """Estimate required resources for computation."""
        ...
    
    def execute_capability(self, capability: ComputeCapability) -> Any:
        """Execute a compute capability and return results."""
        ...


@runtime_checkable
class LazyEvaluationSemantics(Protocol):
    """
    Protocol defining lazy evaluation behavior.
    
    All operations should be lazy by default, building computation graphs
    rather than immediately executing.
    """
    
    def defer_execution(self) -> bool:
        """Check if execution should be deferred."""
        ...
    
    def build_graph(self, operation: Callable) -> Any:
        """Build computation graph node for operation."""
        ...
    
    def should_materialize(self, context: Dict[str, Any]) -> bool:
        """Determine if materialization is needed given context."""
        ...


@runtime_checkable
class OperationComposer(Protocol):
    """
    Protocol for composing operations efficiently.
    
    Enables operation fusion and optimization before execution.
    """
    
    def compose_operations(self, ops: List[Callable]) -> Callable:
        """Compose multiple operations into single operation."""
        ...
    
    def can_fuse(self, op1: Callable, op2: Callable) -> bool:
        """Check if two operations can be fused."""
        ...
    
    def fuse_operations(self, op1: Callable, op2: Callable) -> Callable:
        """Fuse two compatible operations."""
        ...


@runtime_checkable
class ComputeOptimizer(Protocol):
    """
    Protocol for compute graph optimization.
    
    Responsible for optimizing computation graphs before execution.
    """
    
    def optimize_graph(self, graph: Any) -> Any:
        """Optimize computation graph."""
        ...
    
    def eliminate_redundancy(self, graph: Any) -> Any:
        """Eliminate redundant computations."""
        ...
    
    def reorder_operations(self, graph: Any) -> Any:
        """Reorder operations for efficiency."""
        ...


@runtime_checkable
class Materializer(Protocol):
    """
    Protocol for materializing compute capabilities into concrete formats.
    
    Handles conversion from compute abstractions to concrete data structures.
    """
    
    def to_numpy(self, capability: ComputeCapability) -> np.ndarray:
        """Materialize as NumPy array."""
        ...
    
    def to_polars(self, capability: ComputeCapability) -> Any:
        """Materialize as Polars DataFrame."""
        ...
    
    def to_dict(self, capability: ComputeCapability) -> Dict[str, Any]:
        """Materialize as dictionary."""
        ...
    
    def to_format(self, capability: ComputeCapability, format_type: str) -> Any:
        """Materialize to specified format."""
        ...


@runtime_checkable
class ComputeContract(Protocol):
    """
    Protocol defining contracts for compute operations.
    
    Specifies requirements, guarantees, and constraints for computations.
    """
    
    def requires_memory(self) -> int:
        """Memory requirements in bytes."""
        ...
    
    def guarantees_deterministic(self) -> bool:
        """Whether computation is deterministic."""
        ...
    
    def supports_streaming(self) -> bool:
        """Whether computation supports streaming."""
        ...
    
    def max_parallelism(self) -> int:
        """Maximum useful parallelism level."""
        ...


@runtime_checkable
class MemoryAwareCompute(Protocol):
    """
    Protocol for memory-aware computation.
    
    Enables adaptive behavior based on memory constraints.
    """
    
    def estimate_memory_usage(self, input_size: int) -> int:
        """Estimate memory usage for given input size."""
        ...
    
    def supports_spilling(self) -> bool:
        """Whether computation supports memory spilling."""
        ...
    
    def configure_memory_budget(self, budget_bytes: int) -> None:
        """Configure memory budget for computation."""
        ...

# ============================================================================
# CONCRETE COMPUTE NODE IMPLEMENTATION
# ============================================================================

@dataclass(frozen=True)
class ComputeNode:
    """
    Immutable compute graph node representing a single operation.
    
    This is the fundamental building block of computation graphs.
    Forms the nodes in the computation DAG.
    """
    op_type: ComputeOpType
    operation: Callable
    inputs: Tuple[ComputeNode, ...] = tuple()
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    def __hash__(self) -> int:
        """Enable use in sets and as dict keys."""
        return hash((self.op_type, id(self.operation), self.inputs))
    
    def apply_to_inputs(self, *args) -> Any:
        """Apply operation to provided inputs."""
        return self.operation(*args)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no inputs)."""
        return len(self.inputs) == 0
    
    def traverse_depth_first(self) -> Iterator[ComputeNode]:
        """Depth-first traversal of compute graph."""
        yield self
        for input_node in self.inputs:
            yield from input_node.traverse_depth_first()
    
    def get_all_dependencies(self) -> Set[ComputeNode]:
        """Get all nodes this node depends on."""
        dependencies = set()
        for input_node in self.inputs:
            dependencies.add(input_node)
            dependencies.update(input_node.get_all_dependencies())
        return dependencies
    
    def estimate_computational_cost(self) -> float:
        """Estimate relative computational cost."""
        base_costs = {
            ComputeOpType.MAP: 1.0,
            ComputeOpType.FILTER: 1.2,
            ComputeOpType.REDUCE: 2.0,
            ComputeOpType.JOIN: 5.0,
            ComputeOpType.SORT: 3.0,
            ComputeOpType.AGGREGATE: 2.5,
            ComputeOpType.WINDOW: 3.5,
            ComputeOpType.TRANSFORM: 2.0,
            ComputeOpType.MATERIALIZE: 1.0,
            ComputeOpType.PARTITION: 1.5,
            ComputeOpType.COLLECT: 1.0,
            ComputeOpType.BROADCAST: 2.0,
            ComputeOpType.CUSTOM: 3.0
        }
        
        base_cost = base_costs.get(self.op_type, 1.0)
        
        # Add costs from dependencies
        dependency_cost = sum(
            node.estimate_computational_cost() 
            for node in self.inputs
        )
        
        return base_cost + dependency_cost


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_leaf_node(operation: Callable, 
                    op_type: ComputeOpType = ComputeOpType.CUSTOM,
                    metadata: Dict[str, Any] = None) -> ComputeNode:
    """Create a leaf compute node with no dependencies."""
    return ComputeNode(
        op_type=op_type,
        operation=operation,
        inputs=tuple(),
        metadata=metadata or {}
    )


def create_binary_node(left: ComputeNode, 
                      right: ComputeNode,
                      operation: Callable,
                      op_type: ComputeOpType = ComputeOpType.CUSTOM,
                      metadata: Dict[str, Any] = None) -> ComputeNode:
    """Create a compute node with two inputs."""
    return ComputeNode(
        op_type=op_type,
        operation=operation,
        inputs=(left, right),
        metadata=metadata or {}
    )


def create_unary_node(input_node: ComputeNode,
                     operation: Callable,
                     op_type: ComputeOpType = ComputeOpType.CUSTOM,
                     metadata: Dict[str, Any] = None) -> ComputeNode:
    """Create a compute node with single input."""
    return ComputeNode(
        op_type=op_type,
        operation=operation,
        inputs=(input_node,),
        metadata=metadata or {}
    )


def compose_nodes(*nodes: ComputeNode, 
                 combiner: Callable = None,
                 op_type: ComputeOpType = ComputeOpType.CUSTOM) -> ComputeNode:
    """Compose multiple nodes using a combiner function."""
    if not nodes:
        raise ValueError("At least one node required")
    
    if len(nodes) == 1:
        return nodes[0]
    
    if combiner is None:
        # Default composition - apply operations in sequence
        def default_combiner(*results):
            return results[-1]  # Return result of last operation
        combiner = default_combiner
    
    return ComputeNode(
        op_type=op_type,
        operation=combiner,
        inputs=nodes,
        metadata={'composed': True, 'node_count': len(nodes)}
    )


# ============================================================================
# VALIDATION AND TESTING UTILITIES
# ============================================================================

def validate_compute_capability(obj: Any) -> bool:
    """Validate that an object implements ComputeCapability protocol."""
    required_methods = ['transform', 'materialize', 'estimate_memory', 'partition_compute', 'compose']
    return all(hasattr(obj, method) for method in required_methods)


def validate_compute_engine(obj: Any) -> bool:
    """Validate that an object implements ComputeEngine protocol."""
    required_methods = ['create_capability', 'optimize_plan', 'estimate_resources', 'execute_capability']
    return all(hasattr(obj, method) for method in required_methods)


def validate_compute_graph(root: ComputeNode) -> Tuple[bool, List[str]]:
    """Validate a compute graph for correctness."""
    errors = []
    visited = set()
    
    def validate_node(node: ComputeNode) -> None:
        if id(node) in visited:
            return  # Already validated
        visited.add(id(node))
        
        # Check node structure
        if not isinstance(node.op_type, ComputeOpType):
            errors.append(f"Invalid op_type: {node.op_type}")
        
        if not callable(node.operation):
            errors.append(f"Operation is not callable: {node.operation}")
        
        if not isinstance(node.inputs, tuple):
            errors.append(f"Inputs must be tuple: {type(node.inputs)}")
        
        # Validate inputs recursively
        for input_node in node.inputs:
            if not isinstance(input_node, ComputeNode):
                errors.append(f"Input is not ComputeNode: {type(input_node)}")
            else:
                validate_node(input_node)
    
    validate_node(root)
    return len(errors) == 0, errors


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@dataclass
class ComputeMetrics:
    """Metrics for compute operation performance."""
    operation_count: int = 0
    total_execution_time: float = 0.0
    memory_allocated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        return self.total_execution_time / self.operation_count if self.operation_count > 0 else 0.0


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Core protocols
    'ComputeCapability',
    'DistributedComputeCapability', 
    'ComputeEngine',
    'LazyEvaluationSemantics',
    'OperationComposer',
    'ComputeOptimizer',
    'Materializer',
    'ComputeContract',
    'MemoryAwareCompute',
    
    # Core types
    'ComputeOpType',
    'ComputeNode',
    'ComputeMetrics',
    
    # Factory functions
    'create_leaf_node',
    'create_binary_node', 
    'create_unary_node',
    'compose_nodes',
    
    # Validation utilities
    'validate_compute_capability',
    'validate_compute_engine',
    'validate_compute_graph',
]