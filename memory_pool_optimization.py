"""
Memory Pool Optimization for Compute Engines
===========================================

Implements object pooling to reduce allocation overhead in hot paths.
This optimization is crucial for achieving the 20M rows/sec target.
"""

import numpy as np
from collections import deque
from typing import Optional, TypeVar, Generic, Callable
import threading

T = TypeVar('T')

class ObjectPool(Generic[T]):
    """
    Thread-safe object pool for reusable objects.
    
    Based on the Flyweight pattern, this reduces allocation pressure
    by reusing objects instead of creating new ones.
    """
    
    def __init__(self, 
                 factory: Callable[[], T],
                 reset_func: Optional[Callable[[T], None]] = None,
                 max_size: int = 100):
        self._factory = factory
        self._reset_func = reset_func
        self._pool = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._stats = {'created': 0, 'reused': 0}
    
    def acquire(self) -> T:
        """Get object from pool or create new one."""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._stats['reused'] += 1
                return obj
        
        # Create new object outside lock
        obj = self._factory()
        self._stats['created'] += 1
        return obj
    
    def release(self, obj: T) -> None:
        """Return object to pool after resetting."""
        if self._reset_func:
            self._reset_func(obj)
        
        with self._lock:
            if len(self._pool) < self._pool.maxlen:
                self._pool.append(obj)
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        return self._stats.copy()


class MemoryPoolManager:
    """
    Manages memory pools for different object types used in compute engines.
    
    This significantly reduces GC pressure during billion-row processing.
    """
    
    def __init__(self):
        # NumPy array pool for common sizes
        self.array_pools = {}
        
        # DataFrame chunk pool
        self.dataframe_pool = ObjectPool(
            factory=lambda: {'data': None, 'columns': None},
            reset_func=lambda df: df.update({'data': None, 'columns': None}),
            max_size=50
        )
        
        # Result buffer pool
        self.buffer_pool = ObjectPool(
            factory=lambda: bytearray(1024 * 1024),  # 1MB buffers
            reset_func=lambda buf: buf.clear(),
            max_size=20
        )
    
    def get_array(self, shape: tuple, dtype=np.float64) -> np.ndarray:
        """Get array from pool or allocate new one."""
        key = (shape, dtype)
        
        if key not in self.array_pools:
            # Create pool for this shape/dtype combination
            self.array_pools[key] = ObjectPool(
                factory=lambda: np.empty(shape, dtype=dtype),
                reset_func=lambda arr: arr.fill(0),
                max_size=10
            )
        
        return self.array_pools[key].acquire()
    
    def return_array(self, arr: np.ndarray) -> None:
        """Return array to pool."""
        key = (arr.shape, arr.dtype)
        if key in self.array_pools:
            self.array_pools[key].release(arr)


# Global memory pool instance
_memory_pool = MemoryPoolManager()

def get_memory_pool() -> MemoryPoolManager:
    """Get global memory pool instance."""
    return _memory_pool