from __future__ import annotations
from typing import Optional, Tuple, Any
import numpy as np

from .core_types import HistogramContext
from ..histogram_strategies import HistogramExecutionStrategy


class RangeEstimator:
    """
    Phase 1.5: add a small, per-instance LRU cache for computed ranges.

    - Key: (column, lineage_fingerprint, bins_bucket)
    - Capacity: bounded (default 128) with simple FIFO eviction order
    - Fallback: if anything goes wrong, delegate directly without caching
    """

    MAX_CACHE_ENTRIES = 128

    def __init__(self, df_obj: Any):
        self._df = df_obj

    def determine(self, ctx: HistogramContext, strategy: HistogramExecutionStrategy) -> Tuple[float, float]:
        df = self._df
        try:
            # Ensure cache structures exist on the DataFrame instance
            if not hasattr(df, '_range_cache'):
                df._range_cache = {}
            if not hasattr(df, '_range_cache_order'):
                df._range_cache_order = []

            lineage_fp = df._compute_cache_key() if hasattr(df, '_compute_cache_key') else str(id(df))
            bins_bucket = self._bucket_bins(ctx.bins)
            weights_flag = bool(ctx.weights) if hasattr(ctx, 'weights') else False
            cache_key = (ctx.column, lineage_fp, bins_bucket, weights_flag)

            cache = df._range_cache
            order = df._range_cache_order

            # Fast path: cache hit
            if cache_key in cache:
                # Refresh LRU order
                try:
                    order.remove(cache_key)
                except ValueError:
                    pass
                order.append(cache_key)
                return cache[cache_key]

            # Miss: compute via existing logic (no behavior change)
            result = df._compute_optimal_range(ctx.column, strategy)

            # Store in cache and enforce bound
            cache[cache_key] = result
            order.append(cache_key)
            self._evict_if_needed(cache, order)
            return result

        except Exception:
            # Safety net: never break behavior
            return df._compute_optimal_range(ctx.column, strategy)

    def _bucket_bins(self, bins: int) -> int:
        """Bucket bin counts to reduce key cardinality while preserving intent."""
        try:
            if bins <= 50:
                return 64
            if bins <= 100:
                return 128
            if bins <= 200:
                return 256
            # Round up to next power of two, capped reasonably
            power = int(np.ceil(np.log2(max(1, bins))))
            return int(min(2048, 2 ** power))
        except Exception:
            return int(bins)

    def _evict_if_needed(self, cache: dict, order: list) -> None:
        try:
            while len(order) > self.MAX_CACHE_ENTRIES:
                evict_key = order.pop(0)
                cache.pop(evict_key, None)
        except Exception:
            # If eviction fails, reset structures to remain safe
            cache.clear()
            order.clear()
