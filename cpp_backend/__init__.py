"""
cpp_backend package: C++-accelerated histogram utilities for Belle II.

Exports:
- cpp_histogram_integrator: Wrapper for building/loading the C++ shared library.
"""

from .cpp_histogram_integrator import cpp_histogram_integrator, get_cpp_accelerator

__all__ = [
    "cpp_histogram_integrator",
    "get_cpp_accelerator",
]
