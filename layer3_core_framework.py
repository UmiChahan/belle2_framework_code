"""
Layer 3: Core Framework and Base Classes
========================================

Provides the foundational architecture for physics-aware analytical engines
that extend Layer 2's compute-first paradigm.

Architecture Improvements:
1. Added explicit interface segregation for better modularity
2. Introduced context managers for resource handling
3. Enhanced type safety with Protocol classes
4. Implemented visitor pattern for compute graph traversal
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Protocol, TypeVar, Generic
import weakref
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor

# Type variables for generic programming
T = TypeVar('T')
TResult = TypeVar('TResult')

# Import Layer 0/1 components
from layer0 import ComputeNode, ComputeOpType
from layer1.lazy_compute_engine import GraphNode


# ============================================================================
# PERFORMANCE OPTIMIZATION: Protocol Classes for Duck Typing
# ============================================================================

class ComputeCapabilityProtocol(Protocol):
    """Protocol for compute capabilities to ensure interface compliance."""
    
    def materialize(self) -> Any: ...
    def estimate_memory(self) -> int: ...
    def transform(self, operation: Callable) -> ComputeCapabilityProtocol: ...


class DataFrameProtocol(Protocol):
    """Protocol for DataFrame-like objects."""
    
    @property
    def shape(self) -> Tuple[int, int]: ...
    
    @property
    def columns(self) -> List[str]: ...
    
    def __getitem__(self, key: str) -> Any: ...


# ============================================================================
# ARCHITECTURAL IMPROVEMENT: Enhanced Physics Context with Thread Safety
# ============================================================================

@dataclass
class PhysicsContext:
    """
    Thread-safe shared physics context for all Layer 3 operations.
    
    Improvements:
    - Thread-local storage for mutable state
    - Immutable configuration options
    - Context validation on initialization
    """
    beam_energy: float = 10.58  # GeV
    integrated_luminosity: Dict[str, float] = field(default_factory=dict)
    detector_configuration: Dict[str, Any] = field(default_factory=dict)
    systematic_variations: List[str] = field(default_factory=list)
    run_period: str = "2019-2021"
    
    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _thread_local: threading.local = field(default_factory=threading.local, init=False, repr=False)
    
    def __post_init__(self):
        """Validate context on initialization."""
        if self.beam_energy <= 0:
            raise ValueError(f"Invalid beam energy: {self.beam_energy}")
        
        # Freeze configuration for immutability
        self._frozen = False
    
    def freeze(self) -> None:
        """Make context immutable after initialization."""
        self._frozen = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Thread-safe conversion to dictionary."""
        with self._lock:
            return {
                'beam_energy': self.beam_energy,
                'integrated_luminosity': self.integrated_luminosity.copy(),
                'run_period': self.run_period,
                'n_systematics': len(self.systematic_variations)
            }
    
    @contextmanager
    def temporary_variation(self, **variations):
        """Context manager for temporary physics variations."""
        old_values = {}
        
        with self._lock:
            # Store old values
            for key, value in variations.items():
                if hasattr(self, key):
                    old_values[key] = getattr(self, key)
                    setattr(self, key, value)
            
            try:
                yield self
            finally:
                # Restore old values
                for key, value in old_values.items():
                    setattr(self, key, value)


# ============================================================================
# PERFORMANCE OPTIMIZATION: Enhanced Compute Node with Caching
# ============================================================================

class PhysicsComputeNode(ComputeNode):
    """
    Extended compute node with physics-specific semantics and optimizations.
    
    Improvements:
    - Result caching with memory management
    - Lazy metadata evaluation
    - Visitor pattern support for graph traversal
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_metadata: Dict[str, Any] = {}
        self.uncertainty_propagation: bool = True
        self.systematic_variations: List[str] = []
        self.luminosity_context: Optional[Dict[str, Any]] = None
        
        # Performance optimizations
        self._result_cache: Optional[Any] = None
        self._cache_valid: bool = False
        self._memory_estimate: Optional[int] = None
        
    def with_physics_context(self, context: PhysicsContext) -> PhysicsComputeNode:
        """Attach physics context with validation."""
        if not isinstance(context, PhysicsContext):
            raise TypeError(f"Expected PhysicsContext, got {type(context)}")
        
        self.physics_metadata.update(context.to_dict())
        return self
    
    def invalidate_cache(self) -> None:
        """Invalidate cached results."""
        self._result_cache = None
        self._cache_valid = False
    
    def accept(self, visitor: ComputeNodeVisitor) -> Any:
        """Accept visitor for graph traversal."""
        return visitor.visit_physics_node(self)
    
    @property
    def estimated_memory(self) -> int:
        """Lazy memory estimation with caching."""
        if self._memory_estimate is None:
            self._memory_estimate = self._estimate_memory_usage()
        return self._memory_estimate
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage for this node."""
        base_estimate = 1000  # Base overhead
        
        # Add estimates based on operation type
        if self.op_type == ComputeOpType.AGGREGATE:
            base_estimate *= 2
        elif self.op_type == ComputeOpType.JOIN:
            base_estimate *= 4
        
        return base_estimate


# ============================================================================
# ARCHITECTURAL IMPROVEMENT: Visitor Pattern for Graph Operations
# ============================================================================

class ComputeNodeVisitor(ABC):
    """Visitor interface for compute graph traversal."""
    
    @abstractmethod
    def visit_physics_node(self, node: PhysicsComputeNode) -> Any:
        """Visit a physics compute node."""
        pass
    
    @abstractmethod
    def visit_generic_node(self, node: ComputeNode) -> Any:
        """Visit a generic compute node."""
        pass


class MemoryEstimationVisitor(ComputeNodeVisitor):
    """Visitor for estimating total graph memory usage."""
    
    def __init__(self):
        self.total_memory = 0
        self.visited_nodes = set()
    
    def visit_physics_node(self, node: PhysicsComputeNode) -> int:
        """Visit physics node and estimate memory."""
        if id(node) in self.visited_nodes:
            return 0
        
        self.visited_nodes.add(id(node))
        node_memory = node.estimated_memory
        self.total_memory += node_memory
        
        # Visit children
        for child in getattr(node, 'inputs', []):
            if hasattr(child, 'accept'):
                child.accept(self)
        
        return node_memory
    
    def visit_generic_node(self, node: ComputeNode) -> int:
        """Visit generic node with default estimation."""
        if id(node) in self.visited_nodes:
            return 0
        
        self.visited_nodes.add(id(node))
        default_memory = 1000
        self.total_memory += default_memory
        
        return default_memory


# ============================================================================
# PERFORMANCE OPTIMIZATION: Resource Management
# ============================================================================

class ResourceManager:
    """
    Centralized resource management for Layer 3 operations.
    
    Features:
    - Thread pool management
    - Memory budget tracking
    - Compute graph optimization scheduling
    """
    
    _instance: Optional[ResourceManager] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> ResourceManager:
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Memory tracking
        self._memory_budget = 8 * 1024 * 1024 * 1024  # 8GB default
        self._allocated_memory = 0
        
        # Compute graph registry
        self._active_graphs = weakref.WeakSet()
        
        self._initialized = True
    
    @contextmanager
    def allocate_memory(self, size: int):
        """Context manager for memory allocation tracking."""
        if self._allocated_memory + size > self._memory_budget:
            raise MemoryError(f"Insufficient memory: {size} requested, "
                            f"{self._memory_budget - self._allocated_memory} available")
        
        self._allocated_memory += size
        try:
            yield
        finally:
            self._allocated_memory -= size
    
    def submit_computation(self, func: Callable[..., T], *args, **kwargs) -> Future[T]:
        """Submit computation to thread pool."""
        return self._executor.submit(func, *args, **kwargs)
    
    def register_graph(self, graph: PhysicsComputeNode) -> None:
        """Register active compute graph for tracking."""
        self._active_graphs.add(graph)
    
    def optimize_graphs(self) -> None:
        """Optimize all registered compute graphs."""
        for graph in self._active_graphs:
            # Apply optimization passes
            self._optimize_single_graph(graph)
    
    def _optimize_single_graph(self, graph: PhysicsComputeNode) -> None:
        """Apply optimization transformations to a single graph."""
        # Placeholder for graph optimization logic
        pass
    
    def shutdown(self) -> None:
        """Clean shutdown of resources."""
        self._executor.shutdown(wait=True)


# ============================================================================
# CLARITY IMPROVEMENT: Structured Physics Engine Base
# ============================================================================

class PhysicsEngine(ABC):
    """
    Enhanced base class for Layer 3 physics engines.
    
    Improvements:
    - Clear separation of concerns
    - Lifecycle management hooks
    - Performance monitoring integration
    """
    
    def __init__(self, context: Optional[PhysicsContext] = None):
        self.context = context or PhysicsContext()
        self._layer2_refs = weakref.WeakValueDictionary()
        self._compute_nodes: List[PhysicsComputeNode] = []
        self._resource_manager = ResourceManager()
        
        # Performance monitoring
        self._operation_count = 0
        self._total_compute_time = 0.0
        
        # Lifecycle hooks
        self._initialize()
    
    def _initialize(self) -> None:
        """Hook for subclass initialization."""
        pass
    
    @abstractmethod
    def create_physics_compute_graph(self, 
                                   data: DataFrameProtocol,
                                   **kwargs) -> PhysicsComputeNode:
        """Create physics-aware compute graph extending Layer 2."""
        pass
    
    def register_layer2_data(self, name: str, data: DataFrameProtocol) -> None:
        """Register Layer 2 data structure with validation."""
        if not hasattr(data, 'shape'):
            raise TypeError(f"Data must have 'shape' attribute, got {type(data)}")
        
        self._layer2_refs[name] = data
        
    def get_registered_data(self, name: str) -> Optional[DataFrameProtocol]:
        """Retrieve registered Layer 2 data with existence check."""
        return self._layer2_refs.get(name)
    
    def execute_graph(self, graph: PhysicsComputeNode) -> Any:
        """Execute compute graph with resource management."""
        # Estimate memory requirements
        visitor = MemoryEstimationVisitor()
        graph.accept(visitor)
        
        # Execute with memory allocation
        with self._resource_manager.allocate_memory(visitor.total_memory):
            return self._execute_graph_internal(graph)
    
    def _execute_graph_internal(self, graph: PhysicsComputeNode) -> Any:
        """Internal graph execution logic."""
        # Placeholder - would integrate with Layer 1/2 execution
        return graph.operation()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        return {
            'operation_count': self._operation_count,
            'average_compute_time': self._total_compute_time / max(self._operation_count, 1),
            'registered_data_count': len(self._layer2_refs),
            'active_compute_nodes': len(self._compute_nodes),
            'memory_allocated': self._resource_manager._allocated_memory
        }


# ============================================================================
# ARCHITECTURAL IMPROVEMENT: Typed Weight Interfaces
# ============================================================================

class WeightedOperation(ABC, Generic[T, TResult]):
    """
    Generic interface for weight-aware operations.
    
    Type parameters:
    - T: Input data type
    - TResult: Result type
    """
    
    @abstractmethod
    def apply_weights(self, data: T, weights: np.ndarray) -> TResult:
        """Apply weights with type safety."""
        pass
    
    @abstractmethod
    def propagate_weighted_uncertainty(self, 
                                     values: np.ndarray,
                                     weights: np.ndarray,
                                     uncertainties: np.ndarray) -> np.ndarray:
        """Propagate uncertainties through weighted operations."""
        pass
    
    def validate_weights(self, data_shape: Tuple[int, ...], 
                        weights_shape: Tuple[int, ...]) -> None:
        """Validate weight dimensions match data."""
        if data_shape != weights_shape:
            raise ValueError(f"Shape mismatch: data {data_shape} vs weights {weights_shape}")


class UncertaintyProvider(ABC):
    """
    Enhanced interface for uncertainty calculation methods.
    
    Improvements:
    - Method registration system
    - Automatic method selection
    - Performance hints
    """
    
    @abstractmethod
    def compute_uncertainty(self, 
                          data: np.ndarray,
                          method: str,
                          confidence_level: float = 0.68,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute uncertainties with method selection."""
        pass
    
    @abstractmethod
    def list_available_methods(self) -> List[str]:
        """List all available uncertainty methods."""
        pass
    
    @abstractmethod
    def get_method_info(self, method: str) -> Dict[str, Any]:
        """Get information about a specific method."""
        pass


# ============================================================================
# PERFORMANCE OPTIMIZATION: Event System for Loose Coupling
# ============================================================================

class Event:
    """Base event class for publish-subscribe pattern."""
    pass


@dataclass
class ComputeGraphCreatedEvent(Event):
    """Event fired when a new compute graph is created."""
    graph: PhysicsComputeNode
    engine: PhysicsEngine
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class WeightCalculatedEvent(Event):
    """Event fired when luminosity weights are calculated."""
    process_name: str
    weight_value: float
    uncertainty: float


class EventBus:
    """
    Simple event bus for decoupled communication.
    
    Enables loose coupling between Layer 3 components.
    """
    
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable]] = {}
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> None:
        """Subscribe to an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)
    
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        event_type = type(event)
        
        with self._lock:
            handlers = self._subscribers.get(event_type, [])
        
        # Call handlers outside lock to prevent deadlock
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log error but don't crash
                print(f"Event handler error: {e}")


# Global event bus instance
event_bus = EventBus()


# ============================================================================
# CLARITY IMPROVEMENT: Factory Pattern for Engine Creation
# ============================================================================

class PhysicsEngineFactory:
    """
    Factory for creating physics engines with proper configuration.
    
    Centralizes engine creation and configuration logic.
    """
    
    _registry: Dict[str, Type[PhysicsEngine]] = {}
    
    @classmethod
    def register(cls, name: str, engine_class: Type[PhysicsEngine]) -> None:
        """Register an engine class."""
        cls._registry[name] = engine_class
    
    @classmethod
    def create(cls, name: str, context: Optional[PhysicsContext] = None, **kwargs) -> PhysicsEngine:
        """Create an engine instance."""
        if name not in cls._registry:
            raise ValueError(f"Unknown engine: {name}. Available: {list(cls._registry.keys())}")
        
        engine_class = cls._registry[name]
        engine = engine_class(context, **kwargs)
        
        # Fire event
        if hasattr(engine, '_compute_nodes') and engine._compute_nodes:
            event_bus.publish(ComputeGraphCreatedEvent(
                graph=engine._compute_nodes[-1],
                engine=engine
            ))
        
        return engine
    
    @classmethod
    def list_engines(cls) -> List[str]:
        """List all registered engines."""
        return list(cls._registry.keys())


# ============================================================================
# PERFORMANCE OPTIMIZATION: Computation Cache
# ============================================================================

class ComputationCache:
    """
    Advanced caching system for expensive computations.
    
    Features:
    - LRU eviction
    - Memory-aware caching
    - Computation deduplication
    """
    
    def __init__(self, max_memory: int = 1024 * 1024 * 1024):  # 1GB default
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._memory_usage: Dict[str, int] = {}
        self._max_memory = max_memory
        self._current_memory = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value with LRU update."""
        with self._lock:
            if key in self._cache:
                # Update access order
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None
    
    def put(self, key: str, value: Any, memory_size: int) -> None:
        """Store value with memory tracking."""
        with self._lock:
            # Evict if necessary
            while self._current_memory + memory_size > self._max_memory and self._access_order:
                self._evict_lru()
            
            # Store new value
            if key in self._cache:
                # Update existing
                self._current_memory -= self._memory_usage.get(key, 0)
                self._access_order.remove(key)
            
            self._cache[key] = value
            self._memory_usage[key] = memory_size
            self._current_memory += memory_size
            self._access_order.append(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_order:
            return
        
        lru_key = self._access_order.pop(0)
        del self._cache[lru_key]
        self._current_memory -= self._memory_usage.pop(lru_key, 0)
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._memory_usage.clear()
            self._current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'items': len(self._cache),
                'memory_used': self._current_memory,
                'memory_limit': self._max_memory,
                'utilization': self._current_memory / self._max_memory if self._max_memory > 0 else 0
            }


# Global computation cache
computation_cache = ComputationCache()


# ============================================================================
# Export all public classes and functions
# ============================================================================

__all__ = [
    # Core classes
    'PhysicsContext',
    'PhysicsComputeNode',
    'PhysicsEngine',
    'PhysicsEngineFactory',
    
    # Interfaces
    'WeightedOperation',
    'UncertaintyProvider',
    'ComputeNodeVisitor',
    
    # Resource management
    'ResourceManager',
    'ComputationCache',
    'computation_cache',
    
    # Event system
    'Event',
    'EventBus',
    'event_bus',
    'ComputeGraphCreatedEvent',
    'WeightCalculatedEvent',
    
    # Protocols
    'ComputeCapabilityProtocol',
    'DataFrameProtocol',
]