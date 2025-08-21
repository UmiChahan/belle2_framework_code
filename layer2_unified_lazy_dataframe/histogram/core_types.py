from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import weakref

try:
    import polars as pl
except Exception:  # keep import safe
    pl = None  # type: ignore

from ..histogram_strategies import HistogramExecutionStrategy, HistogramMetrics  # re-use existing types

@dataclass(frozen=True)
class HistogramContext:
    df_ref: weakref.ReferenceType
    column: str
    bins: int
    range: Optional[Tuple[float, float]]
    density: bool
    weights: Optional[str]
    estimated_rows: int
    num_frames: int
    memory_budget_gb: float
    system_available_gb: float
    memory_pressure: bool
    dtype_str: Optional[str]
    schema_size: int
    debug: bool = False

    @property
    def has_weights(self) -> bool:
        return isinstance(self.weights, str) and len(self.weights) > 0

@dataclass(frozen=True)
class Decision:
    strategy: HistogramExecutionStrategy
    rationale: str = ""
    confidence: float = 0.0
