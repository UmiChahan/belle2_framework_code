from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import psutil
@dataclass
class AccessPattern:
    """Tracks access patterns for optimization."""
    column: Optional[str]
    operation: str
    timestamp: float
    selectivity: Optional[float] = None
    memory_usage: Optional[int] = None


@dataclass
class MaterializationStrategy:
    """Controls how compute graphs materialize to concrete data."""
    format: str = 'auto'  # 'auto', 'arrow', 'polars', 'numpy', 'pandas'
    batch_size: Optional[int] = None
    memory_limit: Optional[int] = None
    spill_enabled: bool = True
    compression: Optional[str] = None

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
@dataclass
class TransformationMetadata:
    """Immutable record of a data transformation with full provenance tracking."""
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_groups: Dict[str, List[str]] = field(default_factory=dict)
    result_processes: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate unique transformation ID."""
        self.id = f"{self.operation}_{self.timestamp.isoformat()}_{id(self)}"


class DataTransformationChain:
    """Manages transformation history with validation and rollback capabilities."""
    
    def __init__(self):
        self._transformations: List[TransformationMetadata] = []
        self._checkpoints: Dict[str, int] = {}
    
    def add_transformation(self, metadata: TransformationMetadata) -> None:
        """Add transformation with validation."""
        if self._transformations:
            metadata.parent_id = self._transformations[-1].id
        self._transformations.append(metadata)
    
    def create_checkpoint(self, name: str) -> None:
        """Create named checkpoint for potential rollback."""
        self._checkpoints[name] = len(self._transformations)
    
    def get_lineage(self) -> List[TransformationMetadata]:
        """Get complete transformation lineage."""
        return self._transformations.copy()
    
    def validate_chain(self) -> Tuple[bool, List[str]]:
        """Validate entire transformation chain for consistency."""
        issues = []
        
        for i, transform in enumerate(self._transformations):
            # Check parent linkage
            if i > 0 and transform.parent_id != self._transformations[i-1].id:
                issues.append(f"Broken chain at transformation {i}: {transform.operation}")
            
            # Validate parameter types
            if 'query' in transform.operation and 'expr' not in transform.parameters:
                issues.append(f"Query operation missing expression at {i}")
        
        return len(issues) == 0, issues

@dataclass
class SystemCharacteristics:
    cpu_cores: int
    memory_gb: float
    cache_mb: int
    storage_type: str

    @staticmethod
    def detect() -> 'SystemCharacteristics':
        try:
            cores = psutil.cpu_count(logical=True) or 4
            mem_gb = (psutil.virtual_memory().total or 8 * 1024**3) / 1024**3
        except Exception:
            cores, mem_gb = 4, 8.0
        # Cache size is not directly available; use a heuristic
        cache_mb = 8 * 1024 if mem_gb >= 64 else 4 * 1024 if mem_gb >= 32 else 2048
        # Storage heuristic: NVMe likely on fast systems, else SSD
        storage = 'nvme' if mem_gb >= 64 else 'ssd'
        return SystemCharacteristics(cpu_cores=cores, memory_gb=mem_gb, cache_mb=cache_mb, storage_type=storage)


class PerformanceHistory:
    def __init__(self):
        self._history: Dict[int, float] = {}

    def record(self, chunk_size: int, throughput: float):
        # Keep max throughput per chunk size
        prev = self._history.get(chunk_size, 0.0)
        if throughput > prev:
            self._history[chunk_size] = throughput

    def get_optimal_chunk_size(self, default_size: int) -> int:
        if not self._history:
            return default_size
        # Return chunk size with best throughput
        return max(self._history.items(), key=lambda kv: kv[1])[0]
        
    def _calculate_optimal_histogram_chunk_size(self) -> int:
        """PRIVATE: Internal calculation implementation."""
        # Use instance attributes set by public interface
        if hasattr(self, '_estimated_rows') and self._estimated_rows > 0:
            # Existing calculation logic...
            cache_optimal = self._calculate_cache_optimal_size(self._avg_row_bytes)
            bandwidth_optimal = self._calculate_bandwidth_optimal_size(
                self._avg_row_bytes, self._operation_type
            )
            storage_optimal = self._calculate_storage_optimal_size(self._avg_row_bytes)
            scale_adaptive = self._calculate_scale_adaptive_bounds(self._estimated_rows)
            
            # Synthesize recommendations
            candidates = [cache_optimal, bandwidth_optimal, storage_optimal, scale_adaptive]
            base_size = int(np.median(candidates))
            
            # Apply constraints and history
            base_size = self._apply_system_constraints(
                base_size, self._estimated_rows, self._avg_row_bytes
            )
            final_size = self._integrate_performance_feedback(base_size)
            
            return final_size
        
        # Fallback
        return min(1_000_000, self._estimated_rows) if self._estimated_rows > 0 else 10_000
    
    def calculate_optimal_chunk_size(self, estimated_rows: int, 
                                   avg_row_bytes: float, 
                                   operation_type: str) -> int:
        """PUBLIC: Primary interface - MUST be a class method, not nested!"""
        # Store parameters as instance attributes
        self._estimated_rows = estimated_rows
        self._avg_row_bytes = avg_row_bytes
        self._operation_type = operation_type
        
        # Delegate to internal implementation
        return self._calculate_optimal_histogram_chunk_size()
    
    def _calculate_cache_optimal_size(self, avg_row_bytes: float) -> int:
        """
        Optimize for L3 cache efficiency.
        
        THEORY: Cache-conscious algorithms (Frigo et al.)
        TARGET: Keep working set within L3 cache for optimal memory access
        """
        l3_cache_bytes = self.system.cache_mb * 1024 * 1024
        usable_cache = l3_cache_bytes * self._cache_utilization_target
        
        # Account for operation overhead (histogram requires ~2x memory)
        working_memory_factor = 2.0
        effective_cache = usable_cache / working_memory_factor
        
        cache_optimal_rows = int(effective_cache / avg_row_bytes)
        return max(cache_optimal_rows, 10_000)  # Minimum viable chunk
    
    def _calculate_bandwidth_optimal_size(self, avg_row_bytes: float, operation_type: str) -> int:
        """
        Optimize for memory bandwidth utilization.
        
        THEORY: Roofline performance model (Williams et al.)
        TARGET: Sustain target percentage of peak memory bandwidth
        """
        # Memory bandwidth estimation (architecture-dependent)
        if self.system.memory_gb < 16:
            bandwidth_gbps = 25.6   # DDR4-3200 single channel
        elif self.system.memory_gb < 64:
            bandwidth_gbps = 51.2   # DDR4-3200 dual channel  
        else:
            bandwidth_gbps = 102.4  # High-end system
        
        target_bandwidth = bandwidth_gbps * self._bandwidth_utilization_target
        
        # Operation complexity factors
        operation_factors = {
            'histogram': 1.5,  # CPU + memory intensive
            'filter': 1.0,     # Memory streaming
            'groupby': 2.0,    # Complex memory patterns
            'aggregation': 1.2 # Moderate complexity
        }
        
        complexity_factor = operation_factors.get(operation_type, 1.0)
        effective_bandwidth = target_bandwidth / complexity_factor
        
        # Chunk duration target: 50ms for responsiveness
        chunk_duration = 0.05  # seconds
        target_bytes_per_chunk = effective_bandwidth * 1e9 * chunk_duration
        
        bandwidth_optimal_rows = int(target_bytes_per_chunk / avg_row_bytes)
        return max(bandwidth_optimal_rows, 50_000)
    
    def _calculate_storage_optimal_size(self, avg_row_bytes: float) -> int:
        """
        Optimize for storage I/O patterns.
        
        THEORY: Storage hierarchy optimization
        TARGET: Align chunk sizes with optimal I/O transfer sizes
        """
        # Storage-specific optimal transfer sizes
        optimal_transfer_mb = {
            'nvme': 16,   # NVMe: Large sequential reads optimal
            'ssd': 8,     # SSD: Moderate transfer sizes  
            'hdd': 32     # HDD: Very large sequential reads critical
        }
        
        transfer_mb = optimal_transfer_mb.get(self.system.storage_type, 8)
        target_bytes = transfer_mb * 1024 * 1024
        
        storage_optimal_rows = int(target_bytes / avg_row_bytes)
        return max(storage_optimal_rows, 25_000)
    
    def _calculate_scale_adaptive_bounds(self, estimated_rows: int) -> int:
        """
        Dataset-proportional optimization bounds.
        
        THEORY: Adaptive algorithms with scale awareness
        TARGET: Chunk count proportional to dataset complexity
        """
        if estimated_rows < 1_000_000:
            # Small datasets: 5-10 chunks for low overhead
            target_chunks = 10
        elif estimated_rows < 10_000_000:
            # Medium datasets: 10-20 chunks for balance
            target_chunks = 15
        elif estimated_rows < 100_000_000:
            # Large datasets: 20-50 chunks for throughput
            target_chunks = 30
        else:
            # Massive datasets: 50-100 chunks for optimal parallelism
            target_chunks = min(75, self.system.cpu_cores * 4)
        
        scale_optimal_rows = estimated_rows // target_chunks
        return max(scale_optimal_rows, 100_000)
    
    def _apply_system_constraints(self, base_size: int, estimated_rows: int, avg_row_bytes: float) -> int:
        """Apply hard system constraints and safety bounds."""
        
        # Constraint 1: Memory budget enforcement
        chunk_memory_gb = (base_size * avg_row_bytes) / (1024**3)
        max_chunk_memory_gb = self.memory_budget_gb * 0.25  # 25% of budget per chunk
        
        if chunk_memory_gb > max_chunk_memory_gb:
            memory_constrained_size = int((max_chunk_memory_gb * 1024**3) / avg_row_bytes)
            base_size = min(base_size, memory_constrained_size)
        
        # Constraint 2: Dataset bounds
        base_size = min(base_size, estimated_rows)
        
        # Constraint 3: Minimum/maximum bounds for stability
        base_size = max(base_size, 1_000)  # Minimum viable
        base_size = min(base_size, 20_000_000)  # Maximum for responsiveness
        
        return base_size
    
    def _integrate_performance_feedback(self, base_size: int) -> int:
        """Integrate performance history for continuous optimization."""
        
        # Get historically optimal size
        history_optimal = self.performance_history.get_optimal_chunk_size(base_size)
        
        if history_optimal == base_size:
            return base_size
        
        # Weighted blend: 70% system optimization, 30% historical performance
        blended_size = int(0.7 * base_size + 0.3 * history_optimal)
        
        # Ensure reasonable bounds
        return max(1_000_000, min(blended_size, 20_000_000))
    
    def record_performance(self, chunk_size: int, rows_processed: int, execution_time: float):
        """Record performance for adaptive learning."""
        
        if execution_time > 0:
            throughput = rows_processed / execution_time
            self.performance_history.record(chunk_size, throughput)
