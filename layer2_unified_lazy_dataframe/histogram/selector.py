from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import math
import random
import psutil

from .core_types import HistogramContext, Decision
from ..histogram_strategies import HistogramExecutionStrategy


class StrategySelector:
    """
    Phase 2: Scored selection with history blending.

    - Generate eligible candidates based on context and engine availability
    - Score each strategy using cheap heuristics + historical EWMA
    - Return best strategy with a concise rationale
    - On any error, fall back to legacy selection to preserve behavior
    """

    def __init__(self, df_obj):
        self._df = df_obj

    def choose(self, ctx: HistogramContext, force: Optional[HistogramExecutionStrategy] = None) -> Decision:
        # Force wins
        if force is not None:
            return Decision(strategy=force, rationale="forced")

        try:
            candidates = self._candidate_strategies(ctx)
            scored: List[Tuple[HistogramExecutionStrategy, float]] = []
            for strat in candidates:
                cost = self._score_strategy(ctx, strat)
                hist_cost = self._historical_cost(ctx, strat)
                # Blend heuristic with history; lower is better
                alpha = 0.6
                blended = alpha * cost + (1 - alpha) * hist_cost
                scored.append((strat, blended))

            if not scored:
                # Safety: use legacy
                legacy = self._df._select_optimal_histogram_strategy_merged(ctx.column, ctx.bins, ctx.weights)
                return Decision(strategy=legacy, rationale="legacy_no_candidates")

            # Optional tiny exploration under debug
            if ctx.debug and random.random() < 0.05 and len(scored) > 1:
                random.shuffle(scored)

            best = min(scored, key=lambda x: x[1])
            # Confidence: margin to 2nd-best cost normalized to [0,1]
            try:
                ordered = sorted(scored, key=lambda x: x[1])
                if len(ordered) >= 2:
                    best_cost = float(ordered[0][1])
                    second = float(ordered[1][1])
                    margin = max(0.0, second - best_cost)
                    denom = max(second, 1e-9)
                    confidence = max(0.0, min(1.0, margin / denom))
                else:
                    confidence = 0.8  # only one candidate; fairly confident
            except Exception:
                confidence = 0.5

            rationale = self._build_rationale(ctx, scored, best[0], confidence)
            return Decision(strategy=best[0], rationale=rationale, confidence=confidence)

        except Exception:
            # Fall back to existing selection logic to keep behavior safe
            strat = self._df._select_optimal_histogram_strategy_merged(
                ctx.column, ctx.bins, ctx.weights
            )
            rationale = self._build_basic_rationale(ctx, strat)
            return Decision(strategy=strat, rationale=rationale)

    # ---------------- internal helpers ----------------
    def _candidate_strategies(self, ctx: HistogramContext) -> List[HistogramExecutionStrategy]:
        df = self._df
        rows = max(0, int(ctx.estimated_rows or 0))
        candidates: List[HistogramExecutionStrategy] = []

        # Always available fallbacks
        candidates.append(HistogramExecutionStrategy.FALLBACK_BASIC)
        candidates.append(HistogramExecutionStrategy.MEMORY_CONSTRAINED)
        candidates.append(HistogramExecutionStrategy.LAZY_CHUNKED)

        # Adaptive when enabled and not extreme pressure
        if getattr(df, 'enable_adaptive_chunking', True) and rows > 100_000 and not ctx.memory_pressure:
            candidates.append(HistogramExecutionStrategy.ADAPTIVE_CHUNKED)

        # CPP eligibility (reuse df checks)
        try:
            if df._should_use_cpp_acceleration(rows, ctx.bins, ctx.weights):
                candidates.append(HistogramExecutionStrategy.CPP_ACCELERATED)
        except Exception:
            pass

        # Billion row engine when huge or under pressure
        try:
            if rows > 10_000_000 or ctx.memory_pressure:
                if df._check_billion_row_capability():
                    candidates.append(HistogramExecutionStrategy.BILLION_ROW_ENGINE)
        except Exception:
            pass

        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    def _score_strategy(self, ctx: HistogramContext, strat: HistogramExecutionStrategy) -> float:
        rows = max(1, int(ctx.estimated_rows or 1))
        bins = max(1, int(ctx.bins or 1))
        weight_penalty = 0.2 if ctx.has_weights else 0.0
        pressure_penalty = 0.3 if ctx.memory_pressure else 0.0

        # Base heuristic: ideal throughput by strategy (rows/sec equivalent -> convert to cost)
        # Larger base means faster; cost is inverse.
        if strat == HistogramExecutionStrategy.CPP_ACCELERATED:
            base_rate = 50_000_000.0  # very fast
            base = rows / base_rate
            return base * (1.0 + weight_penalty + 0.1 * math.log2(bins))

        if strat == HistogramExecutionStrategy.BILLION_ROW_ENGINE:
            base_rate = 20_000_000.0
            base = rows / base_rate
            return base * (0.9 if ctx.memory_pressure else 1.1)

        if strat == HistogramExecutionStrategy.ADAPTIVE_CHUNKED:
            base_rate = 10_000_000.0
            base = rows / base_rate
            return base * (1.0 + 0.05 * math.log2(bins))

        if strat == HistogramExecutionStrategy.LAZY_CHUNKED:
            base_rate = 5_000_000.0
            base = rows / base_rate
            return base * (1.0 + 0.08 * math.log2(bins))

        if strat == HistogramExecutionStrategy.MEMORY_CONSTRAINED:
            base_rate = 2_000_000.0
            base = rows / base_rate
            return base * (1.0 - 0.2 if ctx.memory_pressure else 1.4)

        # FALLBACK_BASIC and any others default to worst cost
        base_rate = 1_000_000.0
        base = rows / base_rate
        return base * (1.5 + 0.1 * math.log2(bins))

    def _historical_cost(self, ctx: HistogramContext, strat: HistogramExecutionStrategy) -> float:
        # Convert history (throughput * efficiency) into a cost (lower is better)
        df = self._df
        try:
            history = getattr(df, '_histogram_performance_history', {}).get(ctx.column, [])
            vals = [h for h in history if h.get('strategy') == strat]
            if not vals:
                return 1.0  # neutral cost when no history

            # EWMA on inverse throughput (cost): higher throughput => lower cost
            alpha = 0.5
            cost = None
            for entry in vals[-5:]:
                thr = max(1e-6, float(entry.get('throughput', 0.0)))
                eff = max(0.01, float(entry.get('efficiency', 0.0)))
                obs_cost = 1.0 / (thr * eff)
                cost = obs_cost if cost is None else alpha * obs_cost + (1 - alpha) * cost
            if cost is None or not math.isfinite(cost):
                return 1.0
            # Normalize to a mild range; avoid dwarfing heuristic
            return min(5.0, max(0.2, cost))
        except Exception:
            return 1.0

    def _build_rationale(self, ctx: HistogramContext, scored: List[Tuple[HistogramExecutionStrategy, float]], best: HistogramExecutionStrategy, confidence: float) -> str:
        top = sorted(scored, key=lambda x: x[1])[:3]
        parts = [f"{s.name}:{c:.3f}" for s, c in top]
        return (
            f"rows={ctx.estimated_rows}, bins={ctx.bins}, weights={bool(ctx.weights)}, "
            f"pressure={ctx.memory_pressure} | top={';'.join(parts)} -> {best.name} (conf={confidence:.2f})"
        )

    def _build_basic_rationale(self, ctx: HistogramContext, strat: HistogramExecutionStrategy) -> str:
        return (
            f"rows={ctx.estimated_rows}, bins={ctx.bins}, weights={bool(ctx.weights)}, "
            f"pressure={ctx.memory_pressure} -> {strat.name}"
        )
