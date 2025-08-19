"""
Thin proxy to backend C++-accelerated histogram integration.
This keeps existing imports working (layer2) while the implementation lives in cpp_backend/.
"""
from cpp_backend.optimized_cpp_integration import (
    OptimizedStreamingHistogram,
    configure_openmp_for_hpc,
)

__all__ = [
    "OptimizedStreamingHistogram",
    "configure_openmp_for_hpc",
]
