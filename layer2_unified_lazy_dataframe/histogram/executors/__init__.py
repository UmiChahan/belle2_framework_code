"""
Histogram Executors - Phase 1 modularization
============================================

Thin adapters that delegate to existing UnifiedLazyDataFrame execution methods.
This preserves behavior while decoupling strategy selection from execution.
"""
from __future__ import annotations
from typing import Tuple, Optional

import numpy as np  # type: ignore

from ...histogram_strategies import HistogramExecutionStrategy, HistogramMetrics  # type: ignore


class BaseExecutor:
    def __init__(self, owner):
        self._owner = owner

    def execute(self, ctx, computed_range: Tuple[float, float], metrics: HistogramMetrics) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class CppAcceleratedExecutor(BaseExecutor):
    def execute(self, ctx, computed_range, metrics):
        return self._owner._execute_cpp_accelerated_histogram(
            ctx.column, ctx.bins, computed_range, ctx.density, ctx.weights, metrics
        )


class BillionRowExecutor(BaseExecutor):
    def execute(self, ctx, computed_range, metrics):
        return self._owner._execute_billion_row_histogram(
            ctx.column, ctx.bins, computed_range, ctx.density, ctx.weights, metrics
        )


class AdaptiveChunkedExecutor(BaseExecutor):
    def execute(self, ctx, computed_range, metrics):
        return self._owner._execute_adaptive_chunked_histogram(
            ctx.column, ctx.bins, computed_range, ctx.density, ctx.weights, metrics
        )


class LazyChunkedExecutor(BaseExecutor):
    def execute(self, ctx, computed_range, metrics):
        return self._owner._execute_lazy_chunked_histogram(
            ctx.column, ctx.bins, computed_range, ctx.density, ctx.weights, metrics
        )


class MemoryConstrainedExecutor(BaseExecutor):
    def execute(self, ctx, computed_range, metrics):
        return self._owner._execute_memory_constrained_histogram(
            ctx.column, ctx.bins, computed_range, ctx.density, ctx.weights, metrics
        )


class FallbackBasicExecutor(BaseExecutor):
    def execute(self, ctx, computed_range, metrics):
        return self._owner._execute_fallback_histogram(
            ctx.column, ctx.bins, computed_range, ctx.density, ctx.weights, metrics
        )


def get_executor(owner, strategy: HistogramExecutionStrategy) -> BaseExecutor:
    if strategy == HistogramExecutionStrategy.CPP_ACCELERATED:
        return CppAcceleratedExecutor(owner)
    if strategy == HistogramExecutionStrategy.BILLION_ROW_ENGINE:
        return BillionRowExecutor(owner)
    if strategy == HistogramExecutionStrategy.ADAPTIVE_CHUNKED:
        return AdaptiveChunkedExecutor(owner)
    if strategy == HistogramExecutionStrategy.LAZY_CHUNKED:
        return LazyChunkedExecutor(owner)
    if strategy == HistogramExecutionStrategy.MEMORY_CONSTRAINED:
        return MemoryConstrainedExecutor(owner)
    return FallbackBasicExecutor(owner)
