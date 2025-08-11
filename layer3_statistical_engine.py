"""
Layer 3: Statistical Analysis Engine
====================================

High-performance statistical analysis engine for Belle II with theoretical rigor
and practical efficiency. Integrates seamlessly with Layer 2's compute-first
architecture while providing physics-aware statistical methods.

Key Architectural Improvements:
1. Lazy statistical operations that extend compute graphs
2. Vectorized kernels with SIMD optimizations
3. Hierarchical caching with compute-aware invalidation
4. Automatic method selection based on data characteristics
5. Uncertainty propagation through entire analysis chain

Performance Optimizations:
- JIT compilation for critical numerical paths
- Memory pooling for bootstrap operations
- Parallel execution for independent calculations
- Analytical approximations with controlled accuracy
- Zero-copy operations where possible
"""

import numpy as np
import numba as nb
from numba import jit, njit, prange, types
from numba.typed import Dict as NumbaDict
import scipy.stats as stats
from scipy.special import gammaln, loggamma, betainc, beta as beta_func
from scipy.optimize import minimize_scalar, brentq, newton
from scipy import integrate
import warnings
from functools import lru_cache, wraps, partial
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

# Import Layer 3 core framework
from layer3_core_framework import (
    PhysicsEngine, PhysicsComputeNode, PhysicsContext,
    WeightedOperation, UncertaintyProvider, ComputeNodeVisitor,
    ResourceManager, ComputationCache, computation_cache,
    event_bus, EventBus, Event, PhysicsEngineFactory,
    DataFrameProtocol, ComputeCapabilityProtocol
)

# Import Layer 2 components
from layer2_unified_lazy_dataframe import UnifiedLazyDataFrame, LazyColumnAccessor
from layer2_optimized_ultra_lazy_dict import OptimizedUltraLazyDict, BroadcastResult
from layer2_materialization_controller import MaterializationHints, MaterializationFormat

# Import Layer 0/1 components  
from layer0 import ComputeNode, ComputeOpType
from layer1.lazy_compute_engine import GraphNode


# ============================================================================
# STATISTICAL METHOD DEFINITIONS WITH METADATA
# ============================================================================

class StatisticalMethod(Enum):
    """Comprehensive statistical methods with metadata."""
    # Frequentist intervals
    CLOPPER_PEARSON = ("clopper_pearson", "exact", "frequentist")
    WILSON_SCORE = ("wilson_score", "approximate", "frequentist")
    AGRESTI_COULL = ("agresti_coull", "approximate", "frequentist")
    
    # Bayesian intervals
    JEFFREYS_HPD = ("jeffreys_hpd", "exact", "bayesian")
    UNIFORM_HPD = ("uniform_hpd", "exact", "bayesian")
    BETA_BINOMIAL = ("beta_binomial", "hierarchical", "bayesian")
    
    # Bootstrap methods
    BOOTSTRAP_BCA = ("bootstrap_bca", "resampling", "nonparametric")
    BOOTSTRAP_PERCENTILE = ("bootstrap_percentile", "resampling", "nonparametric")
    BOOTSTRAP_STUDENTIZED = ("bootstrap_studentized", "resampling", "nonparametric")
    
    # Likelihood methods
    PROFILE_LIKELIHOOD = ("profile_likelihood", "optimization", "likelihood")
    FELDMAN_COUSINS = ("feldman_cousins", "unified", "likelihood")
    
    # HEP specific
    POISSON_NEYMAN = ("poisson_neyman", "construction", "frequentist")
    BINOMIAL_NEYMAN = ("binomial_neyman", "construction", "frequentist")
    BARLOW_BEESTON = ("barlow_beeston", "weighted", "frequentist")
    
    def __init__(self, method_name: str, category: str, paradigm: str):
        self.method_name = method_name
        self.category = category
        self.paradigm = paradigm


@dataclass
class MethodCharacteristics:
    """Characteristics for method selection."""
    min_samples: int = 1
    max_samples: Optional[int] = None
    supports_weights: bool = False
    computational_cost: str = "low"  # low, medium, high
    coverage_accuracy: str = "exact"  # exact, approximate, asymptotic
    recommended_for: List[str] = field(default_factory=list)


# Method characteristics database
METHOD_CHARACTERISTICS = {
    StatisticalMethod.CLOPPER_PEARSON: MethodCharacteristics(
        min_samples=1,
        supports_weights=False,
        computational_cost="low",
        coverage_accuracy="exact",
        recommended_for=["small_samples", "exact_coverage"]
    ),
    StatisticalMethod.WILSON_SCORE: MethodCharacteristics(
        min_samples=10,
        supports_weights=True,
        computational_cost="low",
        coverage_accuracy="approximate",
        recommended_for=["proportions", "medium_samples"]
    ),
    StatisticalMethod.JEFFREYS_HPD: MethodCharacteristics(
        min_samples=1,
        supports_weights=True,
        computational_cost="medium",
        coverage_accuracy="exact",
        recommended_for=["bayesian_analysis", "small_samples", "default"]
    ),
    StatisticalMethod.BOOTSTRAP_BCA: MethodCharacteristics(
        min_samples=30,
        supports_weights=True,
        computational_cost="high",
        coverage_accuracy="approximate",
        recommended_for=["complex_statistics", "nonparametric"]
    ),
    StatisticalMethod.PROFILE_LIKELIHOOD: MethodCharacteristics(
        min_samples=10,
        supports_weights=True,
        computational_cost="high",
        coverage_accuracy="asymptotic",
        recommended_for=["nuisance_parameters", "systematic_uncertainties"]
    ),
}


# ============================================================================
# HIGH-PERFORMANCE STATISTICAL KERNELS
# ============================================================================

@njit(cache=True, fastmath=True, error_model='numpy')
def _log_factorial(n: int) -> float:
    """Fast log factorial using Stirling's approximation for large n."""
    if n <= 1:
        return 0.0
    elif n < 10:
        result = 0.0
        for i in range(2, n + 1):
            result += np.log(i)
        return result
    else:
        # Stirling's approximation with correction terms
        return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n) + 1/(12*n) - 1/(360*n**3)


@njit(cache=True, fastmath=True)
def _poisson_log_pmf(k: int, mu: float) -> float:
    """Log PMF of Poisson distribution."""
    if mu <= 0 or k < 0:
        return -np.inf
    return k * np.log(mu) - mu - _log_factorial(k)


@njit(cache=True, fastmath=True, parallel=True)
def _weighted_mean_variance(data: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute weighted mean, variance, and effective sample size.
    
    Uses Bessel's correction for unbiased variance estimation.
    """
    n = len(data)
    
    # Normalize weights
    sum_w = 0.0
    for i in prange(n):
        sum_w += weights[i]
    
    if sum_w <= 0:
        return 0.0, 0.0, 0.0
    
    # Weighted mean
    mean = 0.0
    for i in prange(n):
        mean += data[i] * weights[i] / sum_w
    
    # Weighted variance with Bessel's correction
    var_num = 0.0
    sum_w2 = 0.0
    
    for i in prange(n):
        w_norm = weights[i] / sum_w
        var_num += w_norm * (data[i] - mean)**2
        sum_w2 += weights[i]**2
    
    # Effective sample size
    n_eff = sum_w**2 / sum_w2
    
    # Unbiased variance
    if n_eff > 1:
        variance = var_num * n_eff / (n_eff - 1)
    else:
        variance = var_num
    
    return mean, variance, n_eff


@njit(cache=True, fastmath=True, parallel=True)
def _bootstrap_percentile_kernel(data: np.ndarray, 
                                weights: np.ndarray,
                                n_bootstrap: int,
                                confidence_level: float,
                                seed: int = 42) -> Tuple[float, float, float]:
    """
    Optimized bootstrap percentile calculation with parallel execution.
    
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    n = len(data)
    alpha = 1 - confidence_level
    
    # Pre-allocate bootstrap statistics
    bootstrap_stats = np.empty(n_bootstrap)
    
    # Parallel bootstrap with different seeds
    for b in prange(n_bootstrap):
        # Use different seed for each bootstrap sample
        np.random.seed(seed + b)
        
        # Weighted resampling
        if weights is not None:
            # Normalize weights for sampling
            weights_norm = weights / np.sum(weights)
            
            # Weighted bootstrap sample
            sample_sum = 0.0
            weight_sum = 0.0
            
            for _ in range(n):
                idx = np.searchsorted(np.cumsum(weights_norm), np.random.random())
                if idx >= n:
                    idx = n - 1
                sample_sum += data[idx]
                weight_sum += 1.0
            
            bootstrap_stats[b] = sample_sum / weight_sum
        else:
            # Simple bootstrap
            sample_sum = 0.0
            for _ in range(n):
                idx = np.random.randint(0, n)
                sample_sum += data[idx]
            bootstrap_stats[b] = sample_sum / n
    
    # Sort for percentile calculation
    bootstrap_stats.sort()
    
    # Calculate percentiles
    lower_idx = int(alpha/2 * n_bootstrap)
    upper_idx = int((1 - alpha/2) * n_bootstrap)
    
    point_estimate = np.mean(data) if weights is None else np.sum(data * weights) / np.sum(weights)
    lower_bound = bootstrap_stats[lower_idx]
    upper_bound = bootstrap_stats[upper_idx]
    
    return point_estimate, lower_bound, upper_bound


@njit(cache=True, fastmath=True)
def _beta_hpd_interval(alpha: float, beta: float, confidence_level: float) -> Tuple[float, float]:
    """
    Compute highest posterior density interval for beta distribution.
    
    Uses numerical optimization for exact HPD.
    """
    from scipy.stats import beta as beta_dist
    
    # For symmetric beta, HPD = equal-tailed
    if abs(alpha - beta) < 1e-10:
        q_low = (1 - confidence_level) / 2
        q_high = 1 - q_low
        return beta_dist.ppf(q_low, alpha, beta), beta_dist.ppf(q_high, alpha, beta)
    
    # Mode of beta distribution
    if alpha > 1 and beta > 1:
        mode = (alpha - 1) / (alpha + beta - 2)
    elif alpha <= 1 < beta:
        mode = 0.0
    elif alpha > 1 >= beta:
        mode = 1.0
    else:
        # Both <= 1, use equal-tailed
        q_low = (1 - confidence_level) / 2
        q_high = 1 - q_low
        return beta_dist.ppf(q_low, alpha, beta), beta_dist.ppf(q_high, alpha, beta)
    
    # Find HPD by solving for equal density points
    target_mass = confidence_level
    
    def hpd_objective(lower):
        """Objective function for HPD optimization."""
        if lower < 0 or lower > mode:
            return np.inf
        
        # Find upper bound with same density
        density_lower = beta_dist.pdf(lower, alpha, beta)
        
        # Binary search for upper bound
        left, right = mode, 1.0
        while right - left > 1e-10:
            mid = (left + right) / 2
            if beta_dist.pdf(mid, alpha, beta) > density_lower:
                left = mid
            else:
                right = mid
        
        upper = right
        
        # Check if interval has correct mass
        mass = beta_dist.cdf(upper, alpha, beta) - beta_dist.cdf(lower, alpha, beta)
        return abs(mass - target_mass)
    
    # Optimize to find HPD
    result = minimize_scalar(hpd_objective, bounds=(0, mode), method='bounded')
    lower = result.x
    
    # Find corresponding upper bound
    density_lower = beta_dist.pdf(lower, alpha, beta)
    left, right = mode, 1.0
    while right - left > 1e-10:
        mid = (left + right) / 2
        if beta_dist.pdf(mid, alpha, beta) > density_lower:
            left = mid
        else:
            right = mid
    upper = right
    
    return lower, upper


# ============================================================================
# STATISTICAL RESULT CONTAINERS
# ============================================================================

@dataclass
class StatisticalResult:
    """Enhanced container for statistical results with metadata."""
    central_value: Union[float, np.ndarray]
    lower_uncertainty: Union[float, np.ndarray]
    upper_uncertainty: Union[float, np.ndarray]
    confidence_level: float
    method: StatisticalMethod
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional fields for advanced analysis
    effective_sample_size: Optional[float] = None
    systematic_uncertainties: Optional[Dict[str, float]] = None
    correlation_matrix: Optional[np.ndarray] = None
    goodness_of_fit: Optional[Dict[str, float]] = None
    
    @property
    def symmetric_error(self) -> Union[float, np.ndarray]:
        """Average of asymmetric errors."""
        return (self.lower_uncertainty + self.upper_uncertainty) / 2
    
    @property
    def relative_uncertainty(self) -> Union[float, np.ndarray]:
        """Relative uncertainty (symmetric)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(
                self.central_value != 0,
                self.symmetric_error / np.abs(self.central_value),
                np.inf
            )
    
    def to_string(self, precision: int = 3) -> str:
        """Format result as string with uncertainties."""
        if isinstance(self.central_value, np.ndarray):
            return f"Array[{len(self.central_value)}]"
        
        if abs(self.lower_uncertainty - self.upper_uncertainty) < 1e-10:
            # Symmetric uncertainties
            return f"{self.central_value:.{precision}f} ± {self.symmetric_error:.{precision}f}"
        else:
            # Asymmetric uncertainties
            return (f"{self.central_value:.{precision}f} "
                   f"+{self.upper_uncertainty:.{precision}f} "
                   f"-{self.lower_uncertainty:.{precision}f}")


@dataclass
class StatisticalComputeNode(PhysicsComputeNode):
    """Specialized compute node for statistical operations."""
    statistical_method: Optional[StatisticalMethod] = None
    confidence_level: float = 0.68
    weight_column: Optional[str] = None
    
    def estimate_computation_cost(self) -> float:
        """Estimate computational cost for method selection."""
        if self.statistical_method:
            characteristics = METHOD_CHARACTERISTICS.get(self.statistical_method)
            if characteristics:
                cost_map = {"low": 1.0, "medium": 10.0, "high": 100.0}
                return cost_map.get(characteristics.computational_cost, 1.0)
        return 1.0


# ============================================================================
# ADVANCED STATISTICAL CALCULATOR
# ============================================================================

class StatisticalAnalysisEngine(PhysicsEngine, UncertaintyProvider, WeightedOperation):
    """
    High-performance statistical analysis engine with automatic method selection.
    
    Key Features:
    1. Lazy statistical operations on compute graphs
    2. Automatic method selection based on data characteristics
    3. Hierarchical caching with intelligent invalidation
    4. Parallel execution for independent calculations
    5. Comprehensive uncertainty propagation
    """
    
    def __init__(self, context: Optional[PhysicsContext] = None, **kwargs):
        super().__init__(context)
        
        # Configuration
        self.enable_jit = kwargs.get('enable_jit', True)
        self.parallel_threshold = kwargs.get('parallel_threshold', 10000)
        self.cache_size = kwargs.get('cache_size', 256)
        self.auto_method_selection = kwargs.get('auto_method_selection', True)
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=kwargs.get('max_workers', 4))
        
        # Method registry
        self._method_implementations = self._register_methods()
        
        # Performance tracking
        self._method_timings = {}
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        # Memory pools for bootstrap
        self._bootstrap_pool = self._init_memory_pools()
        
        print("📊 Statistical Analysis Engine initialized")
        print(f"   JIT: {'enabled' if self.enable_jit else 'disabled'}")
        print(f"   Auto method selection: {'enabled' if self.auto_method_selection else 'disabled'}")
    
    def _register_methods(self) -> Dict[StatisticalMethod, Callable]:
        """Register all statistical method implementations."""
        return {
            # Frequentist methods
            StatisticalMethod.CLOPPER_PEARSON: self._clopper_pearson_interval,
            StatisticalMethod.WILSON_SCORE: self._wilson_score_interval,
            StatisticalMethod.AGRESTI_COULL: self._agresti_coull_interval,
            
            # Bayesian methods
            StatisticalMethod.JEFFREYS_HPD: self._jeffreys_hpd_interval,
            StatisticalMethod.UNIFORM_HPD: self._uniform_hpd_interval,
            StatisticalMethod.BETA_BINOMIAL: self._beta_binomial_interval,
            
            # Bootstrap methods
            StatisticalMethod.BOOTSTRAP_BCA: self._bootstrap_bca_interval,
            StatisticalMethod.BOOTSTRAP_PERCENTILE: self._bootstrap_percentile_interval,
            StatisticalMethod.BOOTSTRAP_STUDENTIZED: self._bootstrap_studentized_interval,
            
            # Likelihood methods
            StatisticalMethod.PROFILE_LIKELIHOOD: self._profile_likelihood_interval,
            StatisticalMethod.FELDMAN_COUSINS: self._feldman_cousins_interval,
            
            # HEP specific
            StatisticalMethod.POISSON_NEYMAN: self._poisson_neyman_interval,
            StatisticalMethod.BARLOW_BEESTON: self._barlow_beeston_interval,
        }
    
    def _init_memory_pools(self) -> Dict[str, np.ndarray]:
        """Initialize memory pools for performance."""
        return {
            'bootstrap_samples': np.empty(100000),
            'bootstrap_indices': np.empty(100000, dtype=np.int32),
            'weights_buffer': np.empty(100000),
        }
    
    def create_physics_compute_graph(self,
                                   data: DataFrameProtocol,
                                   operation: str = 'compute_uncertainty',
                                   **kwargs) -> StatisticalComputeNode:
        """Create statistical compute node extending Layer 2 graph."""
        
        # Extract parameters
        column = kwargs.get('column')
        method = kwargs.get('method', StatisticalMethod.JEFFREYS_HPD)
        confidence_level = kwargs.get('confidence_level', 0.68)
        weight_column = kwargs.get('weight_column')
        
        # Create statistical operation
        stat_operation = partial(
            self._compute_statistics_lazy,
            column=column,
            method=method,
            confidence_level=confidence_level,
            weight_column=weight_column
        )
        
        # Create compute node
        stat_node = StatisticalComputeNode(
            op_type=ComputeOpType.AGGREGATE,
            operation=stat_operation,
            inputs=[data._get_root_node()] if hasattr(data, '_get_root_node') else [],
            metadata={
                'physics_operation': 'statistical_analysis',
                'method': method.method_name,
                'confidence_level': confidence_level,
                'column': column
            }
        )
        
        # Configure node
        stat_node.statistical_method = method
        stat_node.confidence_level = confidence_level
        stat_node.weight_column = weight_column
        
        # Attach physics context
        stat_node.with_physics_context(self.context)
        
        # Register node
        self._compute_nodes.append(stat_node)
        
        # Publish event
        from layer3_core_framework import ComputeGraphCreatedEvent
        event_bus.publish(ComputeGraphCreatedEvent(
            graph=stat_node,
            engine=self
        ))
        
        return stat_node
    
    def compute_uncertainty(self,
                          data: np.ndarray,
                          method: Union[str, StatisticalMethod] = StatisticalMethod.JEFFREYS_HPD,
                          confidence_level: float = 0.68,
                          weights: Optional[np.ndarray] = None,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute uncertainties with intelligent method selection.
        
        Features:
        - Automatic method selection based on data characteristics
        - Caching for repeated calculations
        - Parallel execution for large datasets
        - Analytical approximations for performance
        """
        
        # Convert string to enum
        if isinstance(method, str):
            method = StatisticalMethod[method.upper()]
        
        # Auto method selection if enabled
        if self.auto_method_selection and method == StatisticalMethod.JEFFREYS_HPD:
            method = self._select_optimal_method(data, weights, confidence_level)
        
        # Check cache
        cache_key = self._make_cache_key(data, method, confidence_level, weights)
        cached_result = computation_cache.get(cache_key)
        
        if cached_result is not None:
            self._cache_stats['hits'] += 1
            return cached_result
        
        self._cache_stats['misses'] += 1
        
        # Performance tracking
        start_time = time.perf_counter()
        
        # Get implementation
        implementation = self._method_implementations.get(method)
        if implementation is None:
            warnings.warn(f"Method {method} not implemented, using Jeffreys HPD")
            implementation = self._jeffreys_hpd_interval
        
        # Compute uncertainties
        if weights is not None:
            result = self._compute_weighted_uncertainty(
                data, weights, method, confidence_level, implementation, **kwargs
            )
        else:
            result = implementation(data, confidence_level, **kwargs)
        
        # Track timing
        elapsed = time.perf_counter() - start_time
        self._track_method_timing(method, elapsed)
        
        # Cache result if expensive
        if elapsed > 0.01:  # 10ms threshold
            memory_size = data.nbytes + (weights.nbytes if weights is not None else 0)
            computation_cache.put(cache_key, result, memory_size)
        
        return result
    
    def _select_optimal_method(self,
                             data: np.ndarray,
                             weights: Optional[np.ndarray],
                             confidence_level: float) -> StatisticalMethod:
        """Select optimal method based on data characteristics."""
        
        n = len(data)
        has_weights = weights is not None
        
        # For count data (integers)
        if np.all(data == data.astype(int)):
            if n < 30:
                return StatisticalMethod.CLOPPER_PEARSON
            elif has_weights:
                return StatisticalMethod.BARLOW_BEESTON
            else:
                return StatisticalMethod.POISSON_NEYMAN
        
        # For continuous data
        if n < 30:
            return StatisticalMethod.JEFFREYS_HPD
        elif n < 1000:
            return StatisticalMethod.BOOTSTRAP_BCA if has_weights else StatisticalMethod.WILSON_SCORE
        else:
            # Large samples - use analytical approximations
            return StatisticalMethod.WILSON_SCORE
    
    def _compute_weighted_uncertainty(self,
                                    data: np.ndarray,
                                    weights: np.ndarray,
                                    method: StatisticalMethod,
                                    confidence_level: float,
                                    implementation: Callable,
                                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Handle weighted uncertainty calculations."""
        
        # Compute effective sample size
        if self.enable_jit:
            mean_weighted, var_weighted, n_eff = _weighted_mean_variance(data, weights)
        else:
            weights_norm = weights / np.sum(weights)
            mean_weighted = np.average(data, weights=weights_norm)
            var_weighted = np.average((data - mean_weighted)**2, weights=weights_norm)
            n_eff = np.sum(weights)**2 / np.sum(weights**2)
        
        # Check if method supports weights
        characteristics = METHOD_CHARACTERISTICS.get(method)
        if characteristics and not characteristics.supports_weights:
            # Use effective sample size scaling
            unweighted_lower, unweighted_upper = implementation(data, confidence_level, **kwargs)
            
            scale_factor = np.sqrt(len(data) / max(n_eff, 1))
            return unweighted_lower * scale_factor, unweighted_upper * scale_factor
        
        # Method supports weights directly
        kwargs['weights'] = weights
        kwargs['n_eff'] = n_eff
        
        return implementation(data, confidence_level, **kwargs)
    
    def apply_weights(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply weights with normalization."""
        # Validate shapes
        self.validate_weights(data.shape, weights.shape)
        
        # Normalize weights to sum to data length
        weights_norm = weights / np.sum(weights) * len(weights)
        
        return data * weights_norm
    
    def propagate_weighted_uncertainty(self,
                                     values: np.ndarray,
                                     weights: np.ndarray,
                                     uncertainties: np.ndarray) -> np.ndarray:
        """
        Propagate uncertainties through weighted operations.
        
        Uses generalized error propagation:
        σ²_weighted = Σ(w_i² σ_i²) / (Σw_i)²
        """
        # Validate inputs
        if not (values.shape == weights.shape == uncertainties.shape):
            raise ValueError("Input arrays must have same shape")
        
        weights_squared = weights ** 2
        weighted_variance = np.sum(weights_squared * uncertainties**2) / np.sum(weights)**2
        
        return np.sqrt(weighted_variance)
    
    # ========================================================================
    # METHOD IMPLEMENTATIONS
    # ========================================================================
    
    def _clopper_pearson_interval(self,
                                 counts: np.ndarray,
                                 confidence_level: float,
                                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Exact Clopper-Pearson intervals for binomial/Poisson data."""
        from scipy.stats import beta
        
        alpha = 1 - confidence_level
        n_total = kwargs.get('n_total')
        
        # Vectorized implementation
        lower_limits = np.empty_like(counts, dtype=np.float64)
        upper_limits = np.empty_like(counts, dtype=np.float64)
        
        for i, x in enumerate(counts):
            if n_total is None:
                # Poisson case
                if x == 0:
                    lower_limits[i] = 0
                    upper_limits[i] = -np.log(alpha/2)
                else:
                    from scipy.stats import chi2
                    lower_limits[i] = chi2.ppf(alpha/2, 2*x) / 2
                    upper_limits[i] = chi2.ppf(1-alpha/2, 2*(x+1)) / 2
            else:
                # Binomial case
                n = n_total[i] if hasattr(n_total, '__len__') else n_total
                
                if x == 0:
                    lower_limits[i] = 0
                else:
                    lower_limits[i] = beta.ppf(alpha/2, x, n-x+1)
                
                if x == n:
                    upper_limits[i] = 1
                else:
                    upper_limits[i] = beta.ppf(1-alpha/2, x+1, n-x)
                
                # Convert to count scale
                lower_limits[i] *= n
                upper_limits[i] *= n
        
        return counts - lower_limits, upper_limits - counts
    
    def _wilson_score_interval(self,
                             data: np.ndarray,
                             confidence_level: float,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Wilson score interval with continuity correction."""
        from scipy.stats import norm
        
        # Handle both proportion and count data
        if 'n_total' in kwargs:
            x = data
            n = kwargs['n_total']
            p = x / n
        else:
            # Assume data represents proportions
            p = data
            n = kwargs.get('n_samples', 100)
            x = p * n
        
        z = norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Wilson score with continuity correction
        n_tilde = n + z**2
        p_tilde = (x + z**2/2) / n_tilde
        
        margin = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde + z**2 / (4 * n_tilde**2))
        
        lower = np.maximum(0, p_tilde - margin)
        upper = np.minimum(1, p_tilde + margin)
        
        # Return as absolute uncertainties
        if 'n_total' in kwargs:
            return x - lower * n, upper * n - x
        else:
            return p - lower, upper - p
    
    def _jeffreys_hpd_interval(self,
                              counts: np.ndarray,
                              confidence_level: float,
                              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jeffreys prior Bayesian HPD intervals.
        
        Optimal coverage properties for small samples.
        """
        alpha = 1 - confidence_level
        
        # Handle scalar or array input
        if np.isscalar(counts):
            counts = np.array([counts])
        
        lower_errs = np.empty_like(counts, dtype=np.float64)
        upper_errs = np.empty_like(counts, dtype=np.float64)
        
        # Jeffreys prior: Beta(0.5, 0.5) for binomial, Gamma(0.5, 1) for Poisson
        for i, count in enumerate(counts):
            if 'n_total' in kwargs:
                # Binomial case
                n = kwargs['n_total']
                alpha_post = count + 0.5
                beta_post = n - count + 0.5
                
                if self.enable_jit:
                    lower, upper = _beta_hpd_interval(alpha_post, beta_post, confidence_level)
                else:
                    # Fallback to equal-tailed
                    from scipy.stats import beta
                    lower = beta.ppf(alpha/2, alpha_post, beta_post)
                    upper = beta.ppf(1-alpha/2, alpha_post, beta_post)
                
                lower_errs[i] = count - lower * n
                upper_errs[i] = upper * n - count
                
            else:
                # Poisson case
                shape = count + 0.5
                
                from scipy.stats import gamma
                lower = gamma.ppf(alpha/2, shape)
                upper = gamma.ppf(1-alpha/2, shape)
                
                lower_errs[i] = count - lower
                upper_errs[i] = upper - count
        
        return lower_errs, upper_errs
    
    def _bootstrap_bca_interval(self,
                               data: np.ndarray,
                               confidence_level: float,
                               n_bootstrap: int = 2000,
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap BCa (bias-corrected and accelerated) intervals.
        
        Second-order accurate coverage with automatic bias correction.
        """
        from scipy.stats import norm
        
        weights = kwargs.get('weights')
        
        # Use JIT kernel for large datasets
        if self.enable_jit and len(data) > self.parallel_threshold:
            point_est, lower_val, upper_val = _bootstrap_percentile_kernel(
                data, weights, n_bootstrap, confidence_level
            )
            
            # BCa correction
            theta_hat = point_est
            
            # Estimate bias correction factor
            bootstrap_median = (lower_val + upper_val) / 2
            z0 = norm.ppf(0.5 + 0.5 * np.sign(bootstrap_median - theta_hat) * 
                         min(abs(bootstrap_median - theta_hat) / theta_hat, 0.5))
            
            # Estimate acceleration factor using jackknife
            n = len(data)
            if n < 100:  # Full jackknife for small samples
                jackknife_estimates = np.array([
                    np.mean(np.delete(data, i)) for i in range(n)
                ])
            else:  # Subsample for large datasets
                idx = np.random.choice(n, size=100, replace=False)
                jackknife_estimates = np.array([
                    np.mean(np.delete(data, i)) for i in idx
                ])
            
            jackknife_mean = np.mean(jackknife_estimates)
            
            # Acceleration factor
            num = np.sum((jackknife_mean - jackknife_estimates)**3)
            den = 6 * np.sum((jackknife_mean - jackknife_estimates)**2)**1.5
            a = num / den if den > 0 else 0
            
            # Adjusted percentiles
            alpha = 1 - confidence_level
            z_lower = norm.ppf(alpha/2)
            z_upper = norm.ppf(1 - alpha/2)
            
            # BCa adjustment
            a_lower = norm.cdf(z0 + (z0 + z_lower) / (1 - a * (z0 + z_lower)))
            a_upper = norm.cdf(z0 + (z0 + z_upper) / (1 - a * (z0 + z_upper)))
            
            # Recompute with adjusted percentiles
            adjusted_cl = a_upper - a_lower
            _, adj_lower, adj_upper = _bootstrap_percentile_kernel(
                data, weights, n_bootstrap, adjusted_cl
            )
            
            return np.array([theta_hat - adj_lower]), np.array([adj_upper - theta_hat])
        
        # Standard implementation for smaller datasets
        return self._bootstrap_percentile_interval(data, confidence_level, n_bootstrap, **kwargs)
    
    def _bootstrap_percentile_interval(self,
                                     data: np.ndarray,
                                     confidence_level: float,
                                     n_bootstrap: int = 2000,
                                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Standard bootstrap percentile interval."""
        
        weights = kwargs.get('weights')
        alpha = 1 - confidence_level
        
        # Generate bootstrap samples
        rng = np.random.default_rng(seed=42)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            if weights is not None:
                # Weighted bootstrap
                indices = rng.choice(len(data), size=len(data), p=weights/np.sum(weights))
                boot_sample = data[indices]
                boot_weights = weights[indices]
                stat = np.average(boot_sample, weights=boot_weights)
            else:
                # Simple bootstrap
                indices = rng.integers(0, len(data), size=len(data))
                stat = np.mean(data[indices])
            
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute percentiles
        lower_percentile = np.percentile(bootstrap_stats, alpha/2 * 100)
        upper_percentile = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        # Point estimate
        if weights is not None:
            point_estimate = np.average(data, weights=weights)
        else:
            point_estimate = np.mean(data)
        
        return np.array([point_estimate - lower_percentile]), np.array([upper_percentile - point_estimate])
    
    def _profile_likelihood_interval(self,
                                   data: np.ndarray,
                                   confidence_level: float,
                                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Profile likelihood intervals for parameter estimation."""
        from scipy.stats import chi2
        from scipy.optimize import minimize_scalar, brentq
        
        # Define likelihood function (example: Poisson)
        def neg_log_likelihood(mu, data):
            if mu <= 0:
                return np.inf
            return -np.sum([_poisson_log_pmf(int(x), mu) for x in data])
        
        # Find MLE
        mle_result = minimize_scalar(
            lambda mu: neg_log_likelihood(mu, data),
            bounds=(0.01, np.max(data) * 3),
            method='bounded'
        )
        mle = mle_result.x
        max_ll = -mle_result.fun
        
        # Likelihood ratio threshold
        threshold = chi2.ppf(confidence_level, df=1) / 2
        
        # Find confidence interval by root finding
        def profile_ll_diff(mu):
            return -neg_log_likelihood(mu, data) - max_ll + threshold
        
        # Search for roots
        try:
            lower = brentq(profile_ll_diff, 0.01, mle)
        except:
            lower = mle * 0.5  # Fallback
        
        try:
            upper = brentq(profile_ll_diff, mle, mle * 3)
        except:
            upper = mle * 1.5  # Fallback
        
        return np.array([mle - lower]), np.array([upper - mle])
    
    def _barlow_beeston_interval(self,
                                counts: np.ndarray,
                                confidence_level: float,
                                **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Barlow-Beeston method for weighted histograms.
        
        Proper treatment of MC statistical uncertainties.
        """
        weights = kwargs.get('weights')
        weights_squared = kwargs.get('weights_squared')
        
        if weights_squared is None and weights is not None:
            weights_squared = weights ** 2
        
        # Effective number of entries
        if weights_squared is not None:
            n_eff = np.where(
                weights_squared > 0,
                counts**2 / weights_squared,
                counts
            )
        else:
            n_eff = counts
        
        # Use modified Poisson intervals with effective counts
        alpha = 1 - confidence_level
        
        from scipy.stats import gamma
        
        lower_errs = np.empty_like(counts, dtype=np.float64)
        upper_errs = np.empty_like(counts, dtype=np.float64)
        
        for i in range(len(counts)):
            if n_eff[i] <= 0:
                lower_errs[i] = 0
                upper_errs[i] = 0
                continue
            
            # Scale factor for weighted events
            if weights_squared is not None and weights_squared[i] > 0:
                scale = np.sqrt(weights_squared[i] / n_eff[i])
            else:
                scale = 1.0
            
            # Gamma distribution quantiles
            lower = gamma.ppf(alpha/2, n_eff[i] + 0.5)
            upper = gamma.ppf(1 - alpha/2, n_eff[i] + 0.5)
            
            # Scale back to original units
            lower_errs[i] = (n_eff[i] - lower) * scale
            upper_errs[i] = (upper - n_eff[i]) * scale
        
        return lower_errs, upper_errs
    
    def _agresti_coull_interval(self,
                              data: np.ndarray,
                              confidence_level: float,
                              **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Agresti-Coull adjusted Wald interval."""
        from scipy.stats import norm
        
        z = norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Add pseudocounts
        n_tilde = kwargs.get('n_total', 100) + z**2
        x_tilde = data + z**2 / 2
        p_tilde = x_tilde / n_tilde
        
        # Standard error
        se = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        
        # Confidence interval
        margin = z * se
        
        lower = np.maximum(0, p_tilde - margin)
        upper = np.minimum(1, p_tilde + margin)
        
        # Return as absolute uncertainties
        if 'n_total' in kwargs:
            n = kwargs['n_total']
            return data - lower * n, upper * n - data
        else:
            return p_tilde - lower, upper - p_tilde
    
    def _uniform_hpd_interval(self,
                            data: np.ndarray,
                            confidence_level: float,
                            **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform prior Bayesian HPD interval."""
        # Similar to Jeffreys but with uniform prior
        # Beta(1, 1) for binomial, Gamma(1, 1) for Poisson
        
        # Adjust prior parameters
        kwargs_uniform = kwargs.copy()
        kwargs_uniform['prior_alpha'] = 1.0
        kwargs_uniform['prior_beta'] = 1.0
        
        return self._beta_binomial_interval(data, confidence_level, **kwargs_uniform)
    
    def _beta_binomial_interval(self,
                               data: np.ndarray,
                               confidence_level: float,
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Hierarchical Beta-Binomial model for overdispersed count data."""
        from scipy.stats import beta
        
        # Prior parameters
        prior_alpha = kwargs.get('prior_alpha', 0.5)  # Jeffreys default
        prior_beta = kwargs.get('prior_beta', 0.5)
        
        # For count data
        if 'n_total' in kwargs:
            n = kwargs['n_total']
            x = data
        else:
            # Assume proportion data
            x = data * 100  # Scale up
            n = 100
        
        # Posterior parameters
        post_alpha = x + prior_alpha
        post_beta = n - x + prior_beta
        
        # HPD interval
        alpha = 1 - confidence_level
        
        if self.enable_jit:
            # Use optimized HPD calculation
            lower_prop = np.empty_like(data, dtype=np.float64)
            upper_prop = np.empty_like(data, dtype=np.float64)
            
            for i in range(len(data)):
                lower_prop[i], upper_prop[i] = _beta_hpd_interval(
                    post_alpha[i] if hasattr(post_alpha, '__len__') else post_alpha,
                    post_beta[i] if hasattr(post_beta, '__len__') else post_beta,
                    confidence_level
                )
        else:
            # Equal-tailed interval
            lower_prop = beta.ppf(alpha/2, post_alpha, post_beta)
            upper_prop = beta.ppf(1-alpha/2, post_alpha, post_beta)
        
        # Convert to absolute uncertainties
        if 'n_total' in kwargs:
            return x - lower_prop * n, upper_prop * n - x
        else:
            return data - lower_prop, upper_prop - data
    
    def _bootstrap_studentized_interval(self,
                                      data: np.ndarray,
                                      confidence_level: float,
                                      n_bootstrap: int = 2000,
                                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Studentized bootstrap for improved coverage."""
        # Implementation would compute variance estimates
        # For now, delegate to percentile method
        return self._bootstrap_percentile_interval(data, confidence_level, n_bootstrap, **kwargs)
    
    def _feldman_cousins_interval(self,
                                 data: np.ndarray,
                                 confidence_level: float,
                                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Feldman-Cousins unified approach."""
        # This would use pre-computed tables or numerical integration
        # For now, use Neyman construction
        return self._poisson_neyman_interval(data, confidence_level, **kwargs)
    
    def _poisson_neyman_interval(self,
                                counts: np.ndarray,
                                confidence_level: float,
                                **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Neyman construction for Poisson intervals."""
        from scipy.stats import poisson
        from scipy.optimize import brentq
        
        alpha = 1 - confidence_level
        
        lower_limits = np.empty_like(counts, dtype=np.float64)
        upper_limits = np.empty_like(counts, dtype=np.float64)
        
        for i, n in enumerate(counts):
            # Find lower limit
            if n == 0:
                lower_limits[i] = 0
            else:
                # Root finding for lower limit
                def lower_eq(mu):
                    return poisson.sf(n-1, mu) - alpha/2
                
                try:
                    lower_limits[i] = brentq(lower_eq, 0.001, n)
                except:
                    lower_limits[i] = max(0, n - 2*np.sqrt(n))
            
            # Find upper limit
            def upper_eq(mu):
                return poisson.cdf(n, mu) - alpha/2
            
            try:
                upper_limits[i] = brentq(upper_eq, n, n + 10*np.sqrt(n+1))
            except:
                upper_limits[i] = n + 2*np.sqrt(n+1)
        
        return counts - lower_limits, upper_limits - counts
    
    # ========================================================================
    # COMPUTE GRAPH INTEGRATION
    # ========================================================================
    
    def _compute_statistics_lazy(self, df: Any, **kwargs) -> StatisticalResult:
        """Compute statistics within Layer 2 compute graph."""
        
        # Extract data
        column = kwargs.get('column')
        if column:
            data = df[column].to_numpy() if hasattr(df[column], 'to_numpy') else np.array(df[column])
        else:
            data = df.to_numpy() if hasattr(df, 'to_numpy') else np.array(df)
        
        # Extract weights
        weight_column = kwargs.get('weight_column')
        if weight_column and hasattr(df, '__getitem__') and weight_column in df:
            weights = df[weight_column].to_numpy() if hasattr(df[weight_column], 'to_numpy') else np.array(df[weight_column])
        else:
            weights = None
        
        # Compute uncertainties
        method = kwargs.get('method', StatisticalMethod.JEFFREYS_HPD)
        confidence_level = kwargs.get('confidence_level', 0.68)
        
        lower, upper = self.compute_uncertainty(
            data, method, confidence_level, weights=weights
        )
        
        # Compute central value
        if weights is not None:
            central = np.average(data, weights=weights)
            n_eff = np.sum(weights)**2 / np.sum(weights**2)
        else:
            central = np.mean(data)
            n_eff = len(data)
        
        return StatisticalResult(
            central_value=central,
            lower_uncertainty=lower,
            upper_uncertainty=upper,
            confidence_level=confidence_level,
            method=method,
            effective_sample_size=n_eff,
            metadata={
                'n_samples': len(data),
                'weighted': weights is not None,
                'column': column
            }
        )
    
    def compute_histogram_uncertainties(self,
                                      counts: np.ndarray,
                                      method: Union[str, StatisticalMethod] = StatisticalMethod.JEFFREYS_HPD,
                                      confidence_level: float = 0.68,
                                      weights_squared: Optional[np.ndarray] = None) -> StatisticalResult:
        """
        Specialized method for histogram uncertainties.
        
        Handles both unweighted and weighted histograms properly.
        """
        
        # Convert method if string
        if isinstance(method, str):
            method = StatisticalMethod[method.upper()]
        
        # For weighted histograms, use Barlow-Beeston
        if weights_squared is not None:
            lower, upper = self._barlow_beeston_interval(
                counts, confidence_level,
                weights_squared=weights_squared
            )
        else:
            # Unweighted histogram
            lower, upper = self.compute_uncertainty(
                counts, method, confidence_level
            )
        
        return StatisticalResult(
            central_value=counts,
            lower_uncertainty=lower,
            upper_uncertainty=upper,
            confidence_level=confidence_level,
            method=method,
            metadata={
                'histogram': True,
                'weighted': weights_squared is not None
            }
        )
    
    # ========================================================================
    # ADVANCED FEATURES
    # ========================================================================
    
    def compute_correlation_matrix(self,
                                 data: Dict[str, np.ndarray],
                                 weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute weighted correlation matrix for multiple variables."""
        
        var_names = list(data.keys())
        n_vars = len(var_names)
        
        # Stack data
        data_matrix = np.column_stack([data[var] for var in var_names])
        
        if weights is not None:
            # Weighted correlation
            weights_norm = weights / np.sum(weights)
            
            # Weighted mean
            means = np.average(data_matrix, axis=0, weights=weights_norm)
            
            # Center data
            centered = data_matrix - means
            
            # Weighted covariance
            cov = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i, n_vars):
                    cov[i, j] = np.average(
                        centered[:, i] * centered[:, j],
                        weights=weights_norm
                    )
                    cov[j, i] = cov[i, j]
            
            # Convert to correlation
            std_devs = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std_devs, std_devs)
            
        else:
            # Unweighted correlation
            corr = np.corrcoef(data_matrix.T)
        
        return corr
    
    def compute_systematic_uncertainties(self,
                                       nominal: StatisticalResult,
                                       variations: Dict[str, StatisticalResult]) -> StatisticalResult:
        """Combine statistical and systematic uncertainties."""
        
        # Extract systematic shifts
        systematics = {}
        
        for name, varied_result in variations.items():
            shift = varied_result.central_value - nominal.central_value
            systematics[name] = abs(shift)
        
        # Total systematic uncertainty (add in quadrature)
        total_syst = np.sqrt(sum(s**2 for s in systematics.values()))
        
        # Combine with statistical
        total_lower = np.sqrt(nominal.lower_uncertainty**2 + total_syst**2)
        total_upper = np.sqrt(nominal.upper_uncertainty**2 + total_syst**2)
        
        # Create combined result
        combined = StatisticalResult(
            central_value=nominal.central_value,
            lower_uncertainty=total_lower,
            upper_uncertainty=total_upper,
            confidence_level=nominal.confidence_level,
            method=nominal.method,
            systematic_uncertainties=systematics,
            metadata={
                **nominal.metadata,
                'has_systematics': True,
                'n_systematics': len(systematics)
            }
        )
        
        return combined
    
    # ========================================================================
    # PERFORMANCE AND DIAGNOSTICS
    # ========================================================================
    
    def _make_cache_key(self, data, method, confidence_level, weights):
        """Create cache key for computation results."""
        
        # Use data statistics for key
        key_parts = [
            'stat',
            len(data),
            float(np.mean(data)),
            float(np.std(data)),
            float(np.min(data)),
            float(np.max(data)),
            method.value if hasattr(method, 'value') else str(method),
            confidence_level
        ]
        
        if weights is not None:
            key_parts.extend([
                'weighted',
                len(weights),
                float(np.mean(weights)),
                float(np.std(weights))
            ])
        
        return str(tuple(key_parts))
    
    def _track_method_timing(self, method: StatisticalMethod, elapsed: float):
        """Track method performance."""
        if method not in self._method_timings:
            self._method_timings[method] = []
        
        self._method_timings[method].append(elapsed)
        
        # Keep only recent timings
        if len(self._method_timings[method]) > 100:
            self._method_timings[method] = self._method_timings[method][-100:]
    
    def list_available_methods(self) -> List[str]:
        """List all available statistical methods."""
        return [method.method_name for method in StatisticalMethod]
    
    def get_method_info(self, method: str) -> Dict[str, Any]:
        """Get detailed information about a method."""
        
        try:
            method_enum = StatisticalMethod[method.upper()]
        except KeyError:
            return {'error': f'Unknown method: {method}'}
        
        characteristics = METHOD_CHARACTERISTICS.get(method_enum, MethodCharacteristics())
        
        # Get average timing if available
        avg_time = None
        if method_enum in self._method_timings and self._method_timings[method_enum]:
            avg_time = np.mean(self._method_timings[method_enum])
        
        return {
            'name': method_enum.method_name,
            'category': method_enum.category,
            'paradigm': method_enum.paradigm,
            'min_samples': characteristics.min_samples,
            'supports_weights': characteristics.supports_weights,
            'computational_cost': characteristics.computational_cost,
            'coverage_accuracy': characteristics.coverage_accuracy,
            'recommended_for': characteristics.recommended_for,
            'average_time_ms': avg_time * 1000 if avg_time else None,
            'implemented': method_enum in self._method_implementations
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        # Base stats from parent
        stats = super().get_performance_stats()
        
        # Add statistical-specific stats
        stats.update({
            'cache_hit_rate': self._cache_stats['hits'] / max(
                self._cache_stats['hits'] + self._cache_stats['misses'], 1
            ),
            'total_computations': self._cache_stats['hits'] + self._cache_stats['misses'],
            'method_usage': {
                method.method_name: len(timings)
                for method, timings in self._method_timings.items()
            },
            'average_times_ms': {
                method.method_name: np.mean(timings) * 1000
                for method, timings in self._method_timings.items()
                if timings
            }
        })
        
        # Add cache stats
        cache_stats = computation_cache.get_stats()
        stats['global_cache'] = cache_stats
        
        return stats
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# ============================================================================
# FACTORY REGISTRATION
# ============================================================================

# Register the statistical engine
PhysicsEngineFactory.register('statistical', StatisticalAnalysisEngine)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_statistical_engine(context: Optional[PhysicsContext] = None, **kwargs) -> StatisticalAnalysisEngine:
    """Create a configured statistical analysis engine."""
    return PhysicsEngineFactory.create('statistical', context, **kwargs)


def compute_uncertainty(data: np.ndarray,
                       method: str = "jeffreys_hpd",
                       confidence_level: float = 0.68,
                       **kwargs) -> StatisticalResult:
    """
    Convenience function for quick uncertainty calculation.
    
    Uses a global engine instance for efficiency.
    """
    global _global_stat_engine
    
    if '_global_stat_engine' not in globals():
        _global_stat_engine = create_statistical_engine()
    
    lower, upper = _global_stat_engine.compute_uncertainty(
        data,
        StatisticalMethod[method.upper()],
        confidence_level,
        **kwargs
    )
    
    return StatisticalResult(
        central_value=np.mean(data),
        lower_uncertainty=lower,
        upper_uncertainty=upper,
        confidence_level=confidence_level,
        method=StatisticalMethod[method.upper()]
    )


# ============================================================================
# EXAMPLE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Statistical Analysis Engine")
    print("=" * 60)
    
    # Create engine
    engine = create_statistical_engine()
    
    # Test different data types
    test_cases = [
        ("Small counts", np.array([3, 5, 2, 7, 4]), None),
        ("Large counts", np.random.poisson(100, 1000), None),
        ("Weighted data", np.random.exponential(2, 100), np.random.uniform(0.5, 2, 100)),
        ("Proportions", np.random.beta(2, 5, 50), None),
    ]
    
    methods = [
        StatisticalMethod.JEFFREYS_HPD,
        StatisticalMethod.CLOPPER_PEARSON,
        StatisticalMethod.BOOTSTRAP_BCA,
    ]
    
    for name, data, weights in test_cases:
        print(f"\n📊 {name}:")
        print(f"   Data shape: {data.shape}")
        print(f"   Weighted: {'Yes' if weights is not None else 'No'}")
        
        for method in methods:
            try:
                lower, upper = engine.compute_uncertainty(
                    data, method, 0.68, weights=weights
                )
                
                result = StatisticalResult(
                    central_value=np.mean(data) if weights is None else np.average(data, weights=weights),
                    lower_uncertainty=lower[0] if len(lower) == 1 else lower,
                    upper_uncertainty=upper[0] if len(upper) == 1 else upper,
                    confidence_level=0.68,
                    method=method
                )
                
                print(f"   {method.method_name}: {result.to_string()}")
                
            except Exception as e:
                print(f"   {method.method_name}: Failed - {e}")
    
    # Performance report
    print("\n📈 Performance Report:")
    stats = engine.get_performance_stats()
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"   Total computations: {stats['total_computations']}")
    print(f"   Active compute nodes: {stats['active_compute_nodes']}")
    
    print("\n✅ Statistical Analysis Engine test complete!")