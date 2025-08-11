"""
Belle II Analysis Framework - Layer 0: Core Protocols
Theoretical Foundation & Detailed Specification

This module defines the fundamental protocols that establish the contract
for all data structures and operations within the framework.
"""

from typing import Protocol, TypeVar, Generic, List, Dict, Any, Optional, Union, Callable, Tuple, Iterator
from abc import abstractmethod
import numpy as np
from datetime import datetime
from enum import Enum, auto
import numpy.typing as npt
from typing_extensions import Self
from dataclasses import dataclass

# Type variables for generic programming
T = TypeVar('T')
TColumn = TypeVar('TColumn', bound=Union[np.ndarray, List[Any]])
TIndex = TypeVar('TIndex')
TResult = TypeVar('TResult')

class IndexType(Enum):
    """Theoretical foundation: Index taxonomy for data access patterns"""
    INTEGER = "integer"  # O(1) positional access
    LABEL = "label"      # O(1) hash-based access
    SLICE = "slice"      # O(k) range access, k = slice size
    MASK = "mask"        # O(n) boolean array access
    FANCY = "fancy"      # O(k) integer array access

class AccessPattern(Enum):
    """Data access patterns for optimization strategies"""
    SEQUENTIAL = "sequential"      # Optimize for cache locality
    RANDOM = "random"             # Optimize for minimal memory movement
    COLUMNAR = "columnar"         # Optimize for vertical operations
    BROADCAST = "broadcast"       # Optimize for vectorized operations

# ============================================================================
# PROTOCOL 1: DataFrameProtocol - The Data Container Contract
# ============================================================================

class DataFrameProtocol(Protocol[TIndex]):
    """
    Theoretical Foundation:
    - Based on relational algebra with lazy evaluation semantics
    - Implements the concept of a "deferred computation graph"
    - Supports both row-wise and column-wise operations
    
    Design Principles:
    1. Immutability by default (functional programming paradigm)
    2. Lazy evaluation for performance optimization
    3. Type safety through generic protocols
    4. Memory efficiency through view semantics
    """
    
    # Core Data Access - The Fundamental Operations
    @abstractmethod
    def __getitem__(self, key: Union[str, List[str], slice, TIndex]) -> Union['ColumnAccessor', 'DataFrameProtocol']:
        """
        Universal accessor supporting multiple indexing paradigms:
        - Single column: df['energy'] -> ColumnAccessor
        - Multiple columns: df[['energy', 'momentum']] -> DataFrameProtocol
        - Row slicing: df[10:20] -> DataFrameProtocol
        - Boolean masking: df[df['energy'] > 100] -> DataFrameProtocol
        
        Theoretical complexity: O(1) for metadata, O(n) for materialization
        """
        ...
    
    @property
    @abstractmethod
    def columns(self) -> List[str]:
        """Column names - represents the schema"""
        ...
    
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """(n_rows, n_cols) - fundamental dimensionality"""
        ...
    
    @property
    @abstractmethod
    def dtypes(self) -> Dict[str, np.dtype]:
        """Column data types - enables type-safe operations"""
        ...
    
    @property
    @abstractmethod
    def index(self) -> TIndex:
        """Row index - supports various index types for different access patterns"""
        ...
    
    # Lazy Evaluation Interface
    @abstractmethod
    def collect(self) -> 'MaterializedDataFrame':
        """
        Force materialization of the computation graph.
        Transforms lazy -> eager evaluation.
        """
        ...
    
    @abstractmethod
    def is_lazy(self) -> bool:
        """Query if the data is in lazy (deferred) state"""
        ...
    
    # Schema Operations
    @abstractmethod
    def select(self, columns: List[str]) -> 'DataFrameProtocol':
        """Column projection - fundamental relational operation"""
        ...
    
    @abstractmethod
    def drop(self, columns: List[str]) -> 'DataFrameProtocol':
        """Column removal - inverse of select"""
        ...
    
    @abstractmethod
    def rename(self, mapping: Dict[str, str]) -> 'DataFrameProtocol':
        """Schema transformation - preserves data, changes metadata"""
        ...
    
    # Row Operations
    @abstractmethod
    def filter(self, predicate: Union['ColumnAccessor', Callable[[Any], bool]]) -> 'DataFrameProtocol':
        """Row selection based on predicate - fundamental filtering"""
        ...
    
    @abstractmethod
    def head(self, n: int = 5) -> 'DataFrameProtocol':
        """Efficient preview - O(1) for lazy, O(n) for eager"""
        ...
    
    @abstractmethod
    def tail(self, n: int = 5) -> 'DataFrameProtocol':
        """Efficient preview from end"""
        ...
    
    # Memory and Performance Hints
    @abstractmethod
    def memory_usage(self, deep: bool = False) -> Dict[str, int]:
        """Memory profiling for optimization decisions"""
        ...
    
    @abstractmethod
    def access_pattern_hint(self, pattern: AccessPattern) -> None:
        """Hint to optimizer about intended access pattern"""
        ...

# ============================================================================
# PROTOCOL 1.1: ColumnAccessor - Single Column Operations
# ============================================================================

class ColumnAccessor(Protocol[T]):
    """
    Represents a single column with lazy evaluation semantics.
    Supports NumPy-like operations with deferred execution.
    """
    
    @abstractmethod
    def __array__(self) -> np.ndarray:
        """NumPy interoperability - triggers materialization"""
        ...
    
    @abstractmethod
    def __len__(self) -> int:
        """Length without materialization"""
        ...
    
    # Arithmetic operations (return new ColumnAccessor)
    @abstractmethod
    def __add__(self, other: Union[T, 'ColumnAccessor']) -> 'ColumnAccessor':
        ...
    
    @abstractmethod
    def __sub__(self, other: Union[T, 'ColumnAccessor']) -> 'ColumnAccessor':
        ...
    
    @abstractmethod
    def __mul__(self, other: Union[T, 'ColumnAccessor']) -> 'ColumnAccessor':
        ...
    
    @abstractmethod
    def __truediv__(self, other: Union[T, 'ColumnAccessor']) -> 'ColumnAccessor':
        ...
    
    # Comparison operations (return boolean ColumnAccessor)
    @abstractmethod
    def __lt__(self, other: Union[T, 'ColumnAccessor']) -> 'ColumnAccessor[bool]':
        ...
    
    @abstractmethod
    def __gt__(self, other: Union[T, 'ColumnAccessor']) -> 'ColumnAccessor[bool]':
        ...
    
    @abstractmethod
    def __eq__(self, other: Union[T, 'ColumnAccessor']) -> 'ColumnAccessor[bool]':
        ...
    
    # Statistical operations
    @abstractmethod
    def mean(self) -> T:
        """Compute mean - forces partial materialization"""
        ...
    
    @abstractmethod
    def std(self) -> T:
        """Standard deviation"""
        ...
    
    @abstractmethod
    def quantile(self, q: Union[float, List[float]]) -> Union[T, List[T]]:
        """Quantile computation"""
        ...
    
    # Advanced operations
    @abstractmethod
    def rolling(self, window: int) -> 'RollingWindow':
        """Rolling window operations"""
        ...
    
    @abstractmethod
    def apply(self, func: Callable[[T], T], vectorized: bool = True) -> 'ColumnAccessor':
        """Apply arbitrary function"""
        ...

# ============================================================================
# PROTOCOL 2: TransformProtocol - Computation Graph Management
# ============================================================================

class TransformProtocol(Protocol):
    """
    Theoretical Foundation:
    - Based on functional reactive programming principles
    - Implements immutable transformation pipeline
    - Supports both eager and lazy execution strategies
    
    Key Concepts:
    1. Transformations are first-class objects
    2. Composition through monadic chaining
    3. Optimization through graph analysis
    """
    
    @property
    @abstractmethod
    def operation(self) -> str:
        """Operation identifier for optimization and debugging"""
        ...
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Transformation parameters - enables replay and optimization"""
        ...
    
    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Temporal ordering for cache invalidation"""
        ...
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, np.dtype]:
        """Expected input schema for validation"""
        ...
    
    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, np.dtype]:
        """Guaranteed output schema"""
        ...
    
    @abstractmethod
    def validate(self, data: DataFrameProtocol) -> bool:
        """Validate if transformation is applicable"""
        ...
    
    @abstractmethod
    def apply(self, data: DataFrameProtocol) -> DataFrameProtocol:
        """Apply transformation - returns new immutable result"""
        ...
    
    @abstractmethod
    def optimize(self, next_transform: Optional['TransformProtocol']) -> Optional['TransformProtocol']:
        """
        Attempt to fuse with next transformation for optimization.
        Returns optimized transform or None if fusion not possible.
        """
        ...
    
    @abstractmethod
    def cost_estimate(self, input_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate computational cost for query planning.
        Returns: {
            'cpu_ops': float,  # Estimated CPU operations
            'memory_bytes': float,  # Peak memory usage
            'io_bytes': float,  # I/O requirements
        }
        """
        ...

# ============================================================================
# PROTOCOL 3: ComputeProtocol - Execution Engine Contract
# ============================================================================

class StatisticalSummary:
    """
    Comprehensive statistical summary following Tukey's five-number summary
    with extensions for robust statistics and distribution characterization.
    """
    count: int
    mean: float
    std: float
    min: float
    q1: float  # 25th percentile
    median: float  # 50th percentile
    q3: float  # 75th percentile
    max: float
    mode: Optional[float]
    skewness: float
    kurtosis: float
    mad: float  # Median Absolute Deviation
    iqr: float  # Interquartile Range
    cv: float   # Coefficient of Variation
    missing_count: int
    unique_count: int
    dtype: np.dtype

class ComputeProtocol(Protocol):
    """
    Theoretical Foundation:
    - Abstracts the execution backend (NumPy, Polars, GPU, etc.)
    - Implements the Strategy pattern for computation
    - Supports both in-memory and out-of-core processing
    
    Design Goals:
    1. Backend agnostic interface
    2. Performance portability
    3. Resource-aware execution
    
    Mathematical Foundation:
    - Based on relational algebra extended with statistical operations
    - Supports both exact and approximate algorithms
    - Enables streaming computation for large datasets
    """
    
    # ========== Summary Statistics ==========
    
    @abstractmethod
    def describe(self,
                 data: DataFrameProtocol,
                 percentiles: List[float] = [0.25, 0.5, 0.75],
                 include: Optional[List[np.dtype]] = None,
                 datetime_is_numeric: bool = True) -> Dict[str, StatisticalSummary]:
        """
        Comprehensive statistical summary for numerical columns.
        
        Theoretical basis: Combines parametric (mean, std) and non-parametric
        (percentiles, MAD) statistics for robust characterization.
        
        Complexity: O(n·c) where c = number of columns
        """
        ...
    
    @abstractmethod
    def value_counts(self,
                     data: DataFrameProtocol,
                     column: str,
                     normalize: bool = False,
                     sort: bool = True,
                     ascending: bool = False,
                     bins: Optional[int] = None,
                     dropna: bool = True) -> DataFrameProtocol:
        """
        Frequency table computation with optional binning.
        
        Theoretical basis: Empirical distribution function estimation.
        Complexity: O(n log k) where k = number of unique values
        """
        ...
    
    @abstractmethod
    def unique(self,
               data: DataFrameProtocol,
               column: str,
               maintain_order: bool = False) -> np.ndarray:
        """
        Extract unique values with optional order preservation.
        
        Complexity: O(n) expected with hash-based implementation
        """
        ...
    
    @abstractmethod
    def nunique(self,
                data: DataFrameProtocol,
                columns: Optional[List[str]] = None,
                dropna: bool = True,
                approximate: bool = False) -> Union[int, Dict[str, int]]:
        """
        Count unique values with optional HyperLogLog approximation for large data.
        
        Exact complexity: O(n)
        Approximate complexity: O(n) with O(log log n) space
        """
        ...
    
    @abstractmethod
    def mode(self,
             data: DataFrameProtocol,
             columns: Optional[List[str]] = None,
             dropna: bool = True) -> DataFrameProtocol:
        """
        Compute mode(s) - most frequent values.
        Handles multi-modal distributions.
        
        Complexity: O(n log n) worst case
        """
        ...
    
    @abstractmethod
    def quantile(self,
                 data: DataFrameProtocol,
                 q: Union[float, List[float]],
                 columns: Optional[List[str]] = None,
                 interpolation: str = 'linear') -> Union[float, Dict[str, float], DataFrameProtocol]:
        """
        Quantile computation with multiple interpolation methods.
        
        Interpolation methods:
        - 'linear': i + (j - i) * fraction
        - 'lower': i
        - 'higher': j
        - 'midpoint': (i + j) / 2
        - 'nearest': i or j, whichever is nearest
        
        Complexity: O(n) with quickselect, O(n log n) with sorting
        """
        ...
    
    @abstractmethod
    def skew(self,
             data: DataFrameProtocol,
             columns: Optional[List[str]] = None,
             axis: int = 0,
             skipna: bool = True) -> Union[float, Dict[str, float]]:
        """
        Compute skewness - third standardized moment.
        
        Mathematical definition: E[(X - μ)³] / σ³
        Complexity: O(n) single pass with numerical stability
        """
        ...
    
    @abstractmethod
    def kurtosis(self,
                 data: DataFrameProtocol,
                 columns: Optional[List[str]] = None,
                 axis: int = 0,
                 fisher: bool = True,
                 skipna: bool = True) -> Union[float, Dict[str, float]]:
        """
        Compute kurtosis - fourth standardized moment.
        
        Fisher=True: Excess kurtosis (normal = 0)
        Fisher=False: Pearson kurtosis (normal = 3)
        
        Complexity: O(n) single pass
        """
        ...
    
    @abstractmethod
    def covariance(self,
                   data: DataFrameProtocol,
                   columns: Optional[List[str]] = None,
                   min_periods: Optional[int] = None) -> np.ndarray:
        """
        Compute covariance matrix with pairwise deletion for missing values.
        
        Theoretical basis: Unbiased estimator with Bessel's correction
        Complexity: O(n·c²) where c = number of columns
        """
        ...
    
    # ========== Core Computational Primitives ==========
    
    @abstractmethod
    def histogram(self,
                  data: DataFrameProtocol,
                  column: str,
                  bins: Union[int, np.ndarray, str] = 'auto',
                  range: Optional[Tuple[float, float]] = None,
                  weights: Optional[str] = None,
                  density: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram with multiple binning strategies.
        
        Binning strategies:
        - 'auto': Maximum of Sturges and Freedman-Diaconis
        - 'fd': Freedman-Diaconis (IQR-based, robust to outliers)
        - 'sturges': Sturges' formula (log2(n) + 1)
        - 'sqrt': Square-root choice
        - 'rice': Rice rule (2 * n^(1/3))
        - 'scott': Scott's normal reference rule
        
        Complexity: O(n) scan + O(b) bin operations
        """
        ...
    
    @abstractmethod
    def aggregate(self,
                  data: DataFrameProtocol,
                  group_by: List[str],
                  aggregations: Dict[str, Union[str, List[Union[str, Tuple[str, Callable]]]]]) -> DataFrameProtocol:
        """
        Group-by aggregation with support for multiple aggregation functions.
        
        Built-in aggregations: sum, mean, median, min, max, std, var,
        count, nunique, first, last, mode, quantile
        
        Complexity: O(n log g) where g = number of groups
        """
        ...
    
    @abstractmethod
    def pivot_table(self,
                    data: DataFrameProtocol,
                    values: Union[str, List[str]],
                    index: Union[str, List[str]],
                    columns: Union[str, List[str]],
                    aggfunc: Union[str, Callable, Dict[str, Union[str, Callable]]] = 'mean',
                    fill_value: Optional[Any] = None,
                    margins: bool = False) -> DataFrameProtocol:
        """
        Create pivot table - multidimensional aggregation.
        
        Theoretical basis: Generalization of contingency tables
        Complexity: O(n·i·c) where i, c = unique values in index, columns
        """
        ...
    
    @abstractmethod
    def melt(self,
             data: DataFrameProtocol,
             id_vars: Optional[List[str]] = None,
             value_vars: Optional[List[str]] = None,
             var_name: str = 'variable',
             value_name: str = 'value') -> DataFrameProtocol:
        """
        Unpivot from wide to long format - inverse of pivot.
        
        Complexity: O(n·v) where v = number of value variables
        """
        ...
    
    # ========== Sorting and Ranking ==========
    
    @abstractmethod
    def sort_values(self,
                    data: DataFrameProtocol,
                    by: Union[str, List[str]],
                    ascending: Union[bool, List[bool]] = True,
                    inplace: bool = False,
                    kind: str = 'quicksort',
                    na_position: str = 'last') -> DataFrameProtocol:
        """
        Sort by column values with stable sort option.
        
        Algorithms:
        - 'quicksort': O(n log n) average, not stable
        - 'mergesort': O(n log n) guaranteed, stable
        - 'heapsort': O(n log n) guaranteed, not stable
        - 'radixsort': O(n·k) for integers, k = number of digits
        """
        ...
    
    @abstractmethod
    def rank(self,
             data: DataFrameProtocol,
             columns: Optional[List[str]] = None,
             method: str = 'average',
             ascending: bool = True,
             na_option: str = 'keep',
             pct: bool = False) -> DataFrameProtocol:
        """
        Compute numerical ranks with tie-breaking methods.
        
        Methods:
        - 'average': average rank of tied group
        - 'min': lowest rank in tied group
        - 'max': highest rank in tied group
        - 'first': ranks by appearance order
        - 'dense': like 'min' but rank increases by 1
        
        Complexity: O(n log n) due to sorting
        """
        ...
    
    # ========== Join and Merge Operations ==========
    
    @abstractmethod
    def join(self,
             left: DataFrameProtocol,
             right: DataFrameProtocol,
             on: Optional[Union[str, List[str]]] = None,
             left_on: Optional[Union[str, List[str]]] = None,
             right_on: Optional[Union[str, List[str]]] = None,
             how: str = 'inner',
             suffix: Tuple[str, str] = ('_left', '_right'),
             validate: Optional[str] = None) -> DataFrameProtocol:
        """
        Database-style join with validation options.
        
        Join types:
        - 'inner': intersection
        - 'left': all from left
        - 'right': all from right
        - 'outer': union
        - 'cross': cartesian product
        
        Validation: 'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'
        
        Complexity: O(n log n + m log m) for sort-merge join
        """
        ...
    
    @abstractmethod
    def concat(self,
               dataframes: List[DataFrameProtocol],
               axis: int = 0,
               join: str = 'outer',
               ignore_index: bool = False,
               verify_integrity: bool = False) -> DataFrameProtocol:
        """
        Concatenate multiple DataFrames along axis.
        
        Complexity: O(Σn_i) where n_i = rows in each dataframe
        """
        ...
    
    # ========== Window Functions ==========
    
    @abstractmethod
    def rolling(self,
                data: DataFrameProtocol,
                window: Union[int, str],
                columns: Optional[List[str]] = None,
                center: bool = False,
                win_type: Optional[str] = None,
                min_periods: Optional[int] = None) -> 'RollingGroupBy':
        """
        Rolling window calculations with various window types.
        
        Window types: boxcar, triang, blackman, hamming, bartlett,
        parzen, bohman, blackmanharris, nuttall, barthann, kaiser,
        tukey, taylor, exponential
        
        Complexity: O(n·w) where w = window size
        """
        ...
    
    @abstractmethod
    def expanding(self,
                  data: DataFrameProtocol,
                  columns: Optional[List[str]] = None,
                  min_periods: int = 1) -> 'ExpandingGroupBy':
        """
        Expanding window - cumulative calculations.
        
        Complexity: O(n) for single pass algorithms
        """
        ...
    
    @abstractmethod
    def ewm(self,
            data: DataFrameProtocol,
            columns: Optional[List[str]] = None,
            com: Optional[float] = None,
            span: Optional[float] = None,
            halflife: Optional[float] = None,
            alpha: Optional[float] = None,
            adjust: bool = True) -> 'ExponentialMovingWindow':
        """
        Exponentially weighted calculations.
        
        Decay specification (exactly one):
        - com: center of mass
        - span: span
        - halflife: half-life
        - alpha: smoothing factor
        
        Complexity: O(n) single pass
        """
        ...
    
    # ========== Missing Data Handling ==========
    
    @abstractmethod
    def fillna(self,
               data: DataFrameProtocol,
               value: Optional[Any] = None,
               method: Optional[str] = None,
               limit: Optional[int] = None,
               downcast: Optional[Dict] = None) -> DataFrameProtocol:
        """
        Fill missing values with various strategies.
        
        Methods: 'ffill', 'bfill', 'pad', 'backfill'
        
        Complexity: O(n) single pass
        """
        ...
    
    @abstractmethod
    def interpolate(self,
                    data: DataFrameProtocol,
                    method: str = 'linear',
                    axis: int = 0,
                    limit: Optional[int] = None,
                    limit_direction: str = 'forward',
                    limit_area: Optional[str] = None) -> DataFrameProtocol:
        """
        Interpolate missing values with various methods.
        
        Methods:
        - 'linear': linear interpolation
        - 'polynomial': polynomial interpolation
        - 'spline': spline interpolation
        - 'pchip': Piecewise Cubic Hermite Interpolating Polynomial
        - 'akima': Akima interpolator
        - 'nearest': nearest value propagation
        
        Complexity: varies by method, O(n) to O(n²)
        """
        ...
    
    # ========== Query and Expression Evaluation ==========
    
    @abstractmethod
    def query(self,
              data: DataFrameProtocol,
              expression: str,
              parser: str = 'pandas',
              engine: str = 'numexpr',
              local_dict: Optional[Dict] = None,
              global_dict: Optional[Dict] = None) -> DataFrameProtocol:
        """
        Query using expression language with optimized evaluation.
        
        Engines:
        - 'numexpr': JIT-compiled expressions
        - 'python': Pure Python evaluation
        
        Example: "energy > 100 and abs(momentum_z) < 50"
        
        Complexity: O(n) with short-circuit evaluation
        """
        ...
    
    @abstractmethod
    def eval(self,
             data: DataFrameProtocol,
             expression: str,
             parser: str = 'pandas',
             engine: str = 'numexpr',
             inplace: bool = False) -> Union[ColumnAccessor, DataFrameProtocol]:
        """
        Evaluate expression to create new columns.
        
        Example: "new_col = energy / momentum"
        
        Complexity: O(n) per operation
        """
        ...
    
    # ========== Performance and Resource Management ==========
    
    @abstractmethod
    def estimate_memory(self, 
                        operation: str, 
                        data_shape: Tuple[int, int],
                        dtype_map: Dict[str, np.dtype]) -> Dict[str, int]:
        """
        Estimate memory requirements with breakdown.
        
        Returns: {
            'input_memory': int,
            'working_memory': int,
            'output_memory': int,
            'peak_memory': int
        }
        """
        ...
    
    @abstractmethod
    def can_execute_in_memory(self, 
                              data: DataFrameProtocol, 
                              operation: str,
                              safety_factor: float = 1.5) -> bool:
        """
        Determine if operation fits in available memory with safety margin.
        """
        ...
    
    @abstractmethod
    def partition_strategy(self, 
                          data: DataFrameProtocol, 
                          operation: str,
                          available_memory: int) -> List[Tuple[int, int]]:
        """
        Optimal partitioning for out-of-core processing.
        
        Considers:
        - Cache efficiency
        - I/O minimization
        - Load balancing
        
        Returns list of (start_row, end_row) tuples.
        """
        ...
    
    @abstractmethod
    def execution_plan(self,
                       operations: List[TransformProtocol],
                       data_stats: Dict[str, Any]) -> 'ExecutionPlan':
        """
        Generate optimized execution plan with cost-based optimization.
        
        Optimizations:
        - Predicate pushdown
        - Column pruning
        - Operation fusion
        - Partition elimination
        """
        ...
# ============================================================================
# Compute-First Protocol Definitions
# ============================================================================

class ComputeOpType(Enum):
    """Extended compute operation types for Layer 2."""
    # Core operations
    MAP = auto()
    FILTER = auto()
    REDUCE = auto()
    AGGREGATE = auto()
    JOIN = auto()
    SORT = auto()
    
    # Layer 2 specific operations
    PROJECT = auto()      # Column projection
    SOURCE = auto()       # Data source
    SLICE = auto()        # Row slicing
    SAMPLE = auto()       # Data sampling
    PARTITION = auto()    # Data partitioning
    WINDOW = auto()       # Window functions
    PIVOT = auto()        # Pivot operations
    UNION = auto()        # Union operations

T = TypeVar('T')

@dataclass(frozen=True)
class ComputeNode:
    """Immutable node in the compute graph."""
    op_type: ComputeOpType
    operation: Callable[..., Any]
    inputs: Tuple['ComputeNode', ...] = ()
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


class ComputeCapability(Protocol[T]):
    """
    Core protocol for compute capabilities - the fundamental abstraction.
    
    This protocol inverts the traditional model: instead of data structures
    having compute methods, compute capabilities can manifest as data structures.
    """
    
    @abstractmethod
    def transform(self, operation: Callable[[T], T]) -> 'ComputeCapability[T]':
        """Apply a transformation, returning a new compute capability."""
        ...
    
    @abstractmethod
    def materialize(self) -> T:
        """Force evaluation and return the concrete result."""
        ...
    
    @abstractmethod
    def partition_compute(self, partitioner: Callable[[T], Dict[str, T]]) -> Dict[str, 'ComputeCapability[T]']:
        """Partition computation for parallel execution."""
        ...
    
    @abstractmethod
    def get_compute_graph(self) -> ComputeNode:
        """Return the compute graph for optimization."""
        ...
    
    @abstractmethod
    def estimate_memory(self) -> int:
        """Estimate memory requirements in bytes."""
        ...
    
    @abstractmethod
    def is_materialized(self) -> bool:
        """Check if the computation has been materialized."""
        ...

class ComputeEngine(Protocol):
    """
    Protocol for compute engines - the foundational execution layer.
    
    Engines create and optimize compute capabilities, not data structures.
    """
    
    @abstractmethod
    def create_capability(self, source: Any, schema: Optional[Dict[str, type]] = None) -> ComputeCapability:
        """Create a compute capability from a data source."""
        ...
    
    @abstractmethod
    def optimize_plan(self, capability: ComputeCapability) -> ComputeCapability:
        """Optimize the compute plan for efficient execution."""
        ...
    
    @abstractmethod
    def estimate_resource(self, capability: ComputeCapability) -> Dict[str, float]:
        """Estimate resource requirements (memory, compute time, etc.)."""
        ...
    
    @abstractmethod
    def execute_partitioned(self, capabilities: Dict[str, ComputeCapability]) -> Dict[str, Any]:
        """Execute partitioned compute capabilities in parallel."""
        ...
    
    @abstractmethod
    def can_fuse_operations(self, op1: ComputeNode, op2: ComputeNode) -> bool:
        """Determine if two operations can be fused for efficiency."""
        ...


class LazyEvaluationSemantics(Protocol):
    """Protocol defining lazy evaluation contracts."""
    
    @abstractmethod
    def should_materialize(self) -> bool:
        """Determine if materialization is needed based on access patterns."""
        ...
    
    @abstractmethod
    def get_evaluation_strategy(self) -> str:
        """Return the evaluation strategy (eager, lazy, adaptive)."""
        ...
    
    @abstractmethod
    def mark_for_caching(self) -> None:
        """Mark this computation result for caching."""
        ...


class OperationComposer(Protocol):
    """Protocol for composing operations in the compute graph."""
    
    @abstractmethod
    def compose(self, op1: ComputeNode, op2: ComputeNode) -> ComputeNode:
        """Compose two operations into a single optimized operation."""
        ...
    
    @abstractmethod
    def can_compose(self, op1: ComputeNode, op2: ComputeNode) -> bool:
        """Check if two operations can be composed."""
        ...
    
    @abstractmethod
    def decompose(self, op: ComputeNode) -> List[ComputeNode]:
        """Decompose a complex operation into simpler ones."""
        ...


# ============================================================================
# Materialization Protocols
# ============================================================================

class Materializer(Protocol[T]):
    """Protocol for materializing compute results into concrete formats."""
    
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """Materialize to NumPy array."""
        ...
    
    @abstractmethod
    def to_polars(self) -> Any:  # Would be pl.DataFrame
        """Materialize to Polars DataFrame."""
        ...
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Materialize to dictionary representation."""
        ...
    
    @abstractmethod
    def to_iterator(self, chunk_size: int = 10000) -> Iterator[T]:
        """Materialize as a chunked iterator for streaming."""
        ...


# ============================================================================
# Advanced Compute Protocols
# ============================================================================
class DistributedComputeCapability(ComputeCapability[T], Protocol):
    """Protocol for distributed compute operations."""
    
    @abstractmethod
    def distribute(self, cluster_spec: Dict[str, Any]) -> 'DistributedComputeCapability[T]':
        """Distribute computation across cluster."""
        ...
    
    @abstractmethod
    def gather_results(self) -> ComputeCapability[T]:
        """Gather distributed results back to single capability."""
        ...
    
    @abstractmethod
    def get_locality_hints(self) -> Dict[str, str]:
        """Get data locality hints for optimization."""
        ...


# ============================================================================
# Data Structure Protocols (Now Built on Compute)
# ============================================================================

class ComputeBackedDataFrame(Protocol):
    """
    Protocol for DataFrames that are manifestations of compute capabilities.
    
    Key insight: The DataFrame is just a view over a ComputeCapability.
    """
    
    @property
    @abstractmethod
    def compute_capability(self) -> ComputeCapability:
        """Access the underlying compute capability."""
        ...
    
    @abstractmethod
    def __getitem__(self, key: Any) -> Union['ComputeBackedDataFrame', Any]:
        """Column access triggers compute graph modification."""
        ...
    
    @abstractmethod
    def select(self, *columns: str) -> 'ComputeBackedDataFrame':
        """Select columns by modifying compute graph."""
        ...
    
    @abstractmethod
    def filter(self, predicate: Callable[[Any], bool]) -> 'ComputeBackedDataFrame':
        """Filter rows by adding to compute graph."""
        ...
    
    @abstractmethod
    def compute(self) -> Any:
        """Trigger materialization of the compute graph."""
        ...


# ============================================================================
# Compute Graph Optimization Protocols
# ============================================================================

class ComputeOptimizer(Protocol):
    """Protocol for compute graph optimization strategies."""
    
    @abstractmethod
    def optimize(self, graph: ComputeNode) -> ComputeNode:
        """Optimize the compute graph."""
        ...
    
    @abstractmethod
    def cost_model(self, node: ComputeNode) -> float:
        """Estimate the cost of a compute node."""
        ...
    
    @abstractmethod
    def suggest_parallelism(self, graph: ComputeNode) -> int:
        """Suggest optimal parallelism level."""
        ...


class CostBasedOptimizer(ComputeOptimizer, Protocol):
    """Extended optimizer with cost-based decisions."""
    
    @abstractmethod
    def calibrate_costs(self, sample_data: Any) -> None:
        """Calibrate cost model with sample data."""
        ...
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, float]:
        """Get optimization statistics."""
        ...


# ============================================================================
# Memory Management Protocols
# ============================================================================

class MemoryAwareCompute(Protocol):
    """Protocol for memory-aware compute operations."""
    
    @abstractmethod
    def spill_to_disk(self, threshold_bytes: int) -> None:
        """Spill to disk when memory threshold exceeded."""
        ...
    
    @abstractmethod
    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)."""
        ...
    
    @abstractmethod
    def compact_memory(self) -> None:
        """Compact memory representation."""
        ...


# ============================================================================
# Integration Protocols
# ============================================================================

class ComputeDataFrameAdapter(Protocol):
    """Adapter to make compute capabilities look like traditional DataFrames."""
    
    @abstractmethod
    def from_compute_capability(self, capability: ComputeCapability) -> Any:
        """Create DataFrame-like interface from compute capability."""
        ...
    
    @abstractmethod
    def to_compute_capability(self, dataframe: Any) -> ComputeCapability:
        """Convert DataFrame to compute capability."""
        ...


# ============================================================================
# Compute Contracts
# ============================================================================

class ComputeContract(Protocol):
    """Contract specifications for compute operations."""
    
    @abstractmethod
    def verify_input(self, data: Any) -> bool:
        """Verify input data meets contract requirements."""
        ...
    
    @abstractmethod
    def verify_output(self, data: Any) -> bool:
        """Verify output data meets contract requirements."""
        ...
    
    @abstractmethod
    def get_guarantees(self) -> Dict[str, Any]:
        """Get computational guarantees (determinism, ordering, etc.)."""
        ...


# ============================================================================
# Belle II Specific Compute Protocols
# ============================================================================

# class BelleIIComputeCapability(ComputeCapability[T], Protocol):
#     """Belle II specific compute capability extensions."""
    
#     @abstractmethod
#     def apply_belle_cuts(self, cut_config: Dict[str, Any]) -> 'BelleIIComputeCapability[T]':
#         """Apply Belle II specific cuts to the compute graph."""
#         ...
    
#     @abstractmethod
#     def compute_luminosity_weights(self) -> 'BelleIIComputeCapability[T]':
#         """Add luminosity weight computation to graph."""
#         ...
    
#     @abstractmethod
#     def partition_by_run(self) -> Dict[int, 'BelleIIComputeCapability[T]']:
#         """Partition by Belle II run number."""
#         ...


# ============================================================================
# Factory Protocol
# ============================================================================

class ComputeEngineFactory(Protocol):
    """Factory for creating appropriate compute engines."""
    
    @abstractmethod
    def create_engine(self, 
                     data_size: int,
                     compute_pattern: str,
                     resources: Dict[str, Any]) -> ComputeEngine:
        """Create appropriate compute engine based on requirements."""
        ...
    
    @abstractmethod
    def get_available_engines(self) -> List[str]:
        """List available compute engine implementations."""
        ...



class MaterializedDataFrame:
    """Concrete implementation returned by collect()"""
    pass

class RollingWindow:
    """Rolling window operations support"""
    pass

class ExecutionContext:
    """
    Runtime context for optimization decisions.
    Tracks available memory, CPU cores, GPU availability, etc.
    """
    memory_available: int
    cpu_cores: int
    gpu_available: bool
    cache_size: int
    preferred_backend: str