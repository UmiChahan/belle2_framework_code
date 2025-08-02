# memory_pool_optimization.py - Complete implementation

import numpy as np
import threading
from typing import Dict, Tuple, Optional, Any, Callable
from collections import defaultdict
import weakref

class ArrayPool:
    """Pool for numpy arrays of specific shapes."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.pool = []
        self.stats = {'created': 0, 'reused': 0}
        self._lock = threading.Lock()
    
    def acquire(self, shape: Tuple[int, ...], dtype=np.float64) -> np.ndarray:
        """Get array from pool or create new one."""
        with self._lock:
            for i, arr in enumerate(self.pool):
                if arr.shape == shape and arr.dtype == dtype:
                    self.pool.pop(i)
                    self.stats['reused'] += 1
                    return arr
            
            # Create new array
            self.stats['created'] += 1
            return np.empty(shape, dtype=dtype)
    
    def release(self, arr: np.ndarray):
        """Return array to pool."""
        with self._lock:
            if len(self.pool) < self.max_size:
                self.pool.append(arr)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return self.stats.copy()

class ObjectPool:
    """Generic object pool."""
    
    def __init__(self, factory: Optional[Callable] = None, 
                 reset_func: Optional[Callable] = None,
                 max_size: int = 1000):
        self.factory = factory
        self.reset_func = reset_func
        self.max_size = max_size
        self.pool = []
        self._lock = threading.Lock()
    
    def acquire(self):
        """Get object from pool or create new one."""
        with self._lock:
            if self.pool:
                return self.pool.pop()
            elif self.factory:
                return self.factory()
            return None
    
    def release(self, obj):
        """Return object to pool."""
        with self._lock:
            if len(self.pool) < self.max_size:
                if self.reset_func:
                    self.reset_func(obj)
                self.pool.append(obj)
    
    def clear(self):
        """Clear the pool."""
        with self._lock:
            self.pool.clear()

class MemoryPool:
    """Central memory pool manager."""
    
    def __init__(self):
        self.array_pools = defaultdict(lambda: ArrayPool(max_size=50))
        self.object_pools = {}
        self._lock = threading.Lock()
    
    def get_array(self, shape: Tuple[int, ...], dtype=np.float64) -> np.ndarray:
        """Get numpy array from pool."""
        key = (shape, dtype)
        return self.array_pools[key].acquire(shape, dtype)
    
    def return_array(self, arr: np.ndarray):
        """Return numpy array to pool."""
        key = (arr.shape, arr.dtype)
        self.array_pools[key].release(arr)
    
    def register_object_pool(self, name: str, pool: ObjectPool):
        """Register named object pool."""
        with self._lock:
            self.object_pools[name] = pool
    
    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        """Get named object pool."""
        return self.object_pools.get(name)

# Global memory pool instance
_global_memory_pool = MemoryPool()

def get_memory_pool() -> MemoryPool:
    """Get global memory pool instance."""
    return _global_memory_pool