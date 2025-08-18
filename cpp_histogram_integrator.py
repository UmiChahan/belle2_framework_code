"""
Enhanced C++ Integration Wrapper with Automatic Compilation and Fallback Systems
===============================================================================

This module provides seamless C++ acceleration with automatic compilation,
intelligent fallback systems, and performance optimization for Belle II analysis.
"""

import ctypes
import numpy as np
from pathlib import Path
import os
import platform
import subprocess
import tempfile
import warnings
from typing import Optional, Tuple, Dict
import threading
import time


class CompilationConfig:
    def __init__(self,
                 optimization_level: str = "O3",
                 enable_openmp: bool = True,
                 enable_avx2: bool = True,
                 enable_native_arch: bool = True,
                 enable_fast_math: bool = True,
                 debug_symbols: bool = False):
        self.optimization_level = optimization_level
        self.enable_openmp = enable_openmp
        self.enable_avx2 = enable_avx2
        self.enable_native_arch = enable_native_arch
        self.enable_fast_math = enable_fast_math
        self.debug_symbols = debug_symbols

    def __repr__(self) -> str:
        return (f"CompilationConfig({self.optimization_level}, omp={self.enable_openmp}, "
                f"avx2={self.enable_avx2}, native={self.enable_native_arch}, "
                f"fast={self.enable_fast_math}, debug={self.debug_symbols})")


class IntelligentCppCompiler:
    def __init__(self, config: Optional[CompilationConfig] = None):
        self.config = config or CompilationConfig()
        self.available_compilers = self._detect_compilers()
        self.cpu_features = self._detect_cpu_features()
        self.compilation_cache: Dict[int, Path] = {}
        self._lock = threading.Lock()

    def _detect_compilers(self) -> Dict[str, Dict]:
        compilers: Dict[str, Dict] = {}
        for cmd, key, prio in (("g++", "g++", 90), ("clang++", "clang", 85)):
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    compilers[key] = {
                        'command': cmd,
                        'supports_openmp': True,
                        'supports_avx2': True,
                        'priority': prio,
                    }
            except Exception:
                pass
        return compilers

    def _detect_cpu_features(self) -> Dict[str, bool]:
        # Best-effort; we compile with -march=native when possible
        return {'avx2': True}

    def get_best_compiler(self) -> Optional[Dict]:
        if not self.available_compilers:
            return None
        return sorted(self.available_compilers.values(), key=lambda c: c['priority'], reverse=True)[0]

    def _build_compile_command(self, compiler: Dict, source_path: Path, output_path: Path) -> list[str]:
        cmd = [compiler['command']]
        cmd.extend([
            f"-{self.config.optimization_level}",
            "-fPIC",
            "-shared",
            "-o", str(output_path),
            str(source_path),
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Wno-sign-compare",
            "-Wno-missing-field-initializers",
        ])
        if self.config.enable_openmp and compiler.get('supports_openmp', False):
            cmd.append("-fopenmp")
        if self.config.enable_native_arch:
            cmd.append("-march=native")
        elif self.config.enable_avx2 and self.cpu_features.get('avx2', False):
            cmd.append("-mavx2")
        if self.config.enable_fast_math:
            cmd.append("-ffast-math")
        if self.config.debug_symbols:
            cmd.append("-g")
        return cmd

    def compile_histogram_accelerator(self, source_code: str, output_path: Optional[Path] = None) -> Optional[Path]:
        cache_key = hash((source_code, repr(self.config)))
        with self._lock:
            cached = self.compilation_cache.get(cache_key)
            if cached and cached.exists():
                return cached
        compiler = self.get_best_compiler()
        if not compiler:
            warnings.warn("No suitable C++ compiler found")
            return None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(source_code)
                source_path = Path(f.name)
            if output_path is None:
                output_path = source_path.with_suffix('.so' if platform.system() != 'Windows' else '.dll')
            cmd = self._build_compile_command(compiler, source_path, output_path)
            print(f"🔧 Compiling C++ accelerator with {compiler['command']}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            try:
                source_path.unlink()
            except Exception:
                pass
            if result.returncode != 0:
                print("❌ Compilation failed:")
                print("stdout:\n" + result.stdout)
                print("stderr:\n" + result.stderr)
                return None
            with self._lock:
                self.compilation_cache[cache_key] = output_path
            return output_path
        except Exception as e:
            print(f"❌ Compilation error: {e}")
            return None


class EnhancedCppAccelerator:
    def __init__(self, auto_compile: bool = True):
        self.auto_compile = auto_compile
        self.lib: Optional[ctypes.CDLL] = None
        self.compiler = IntelligentCppCompiler()
        self._initialization_lock = threading.Lock()
        if auto_compile:
            self._initialize_accelerator()

    def _initialize_accelerator(self):
        with self._initialization_lock:
            if self.lib is not None:
                return
            print("🚀 Initializing enhanced C++ accelerator...")
            existing = self._try_load_existing_library()
            if existing:
                self.lib = existing
                print("✅ Loaded existing C++ accelerator")
                return
            compiled = self._compile_accelerator()
            if compiled:
                self.lib = compiled
                print("✅ Compiled and loaded new C++ accelerator")
                return
            print("⚠️ C++ accelerator not available, using fallback implementations")

    def _try_load_existing_library(self) -> Optional[ctypes.CDLL]:
        potential_names = [
            "histogram_accelerator.so",
            "histogram_accelerator.dll",
            "libhistogram_accelerator.so",
            "libhistogram_accelerator.dll",
        ]
        search_paths = [
            Path.cwd(),
            Path(__file__).parent,
            Path.home() / ".belle2_cache",
            Path("/tmp") if platform.system() != "Windows" else Path(os.environ.get("TEMP", "")),
        ]
        for path in search_paths:
            for name in potential_names:
                lib_path = path / name
                if lib_path.exists():
                    try:
                        lib = ctypes.CDLL(str(lib_path))
                        self._setup_function_signatures(lib)
                        return lib
                    except Exception as e:
                        print(f"   ⚠️ Failed to load {lib_path}: {e}")
        return None

    def _compile_accelerator(self) -> Optional[ctypes.CDLL]:
        cpp_source = r'''
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <atomic>

extern "C" {

// Ultra-optimized AVX2 histogram with prefetching
void compute_histogram_avx2_enhanced(
    const double* __restrict__ data,
    size_t n,
    double min_val,
    double max_val,
    size_t num_bins,
    size_t* __restrict__ output
) {
    const double scale = num_bins / (max_val - min_val);
    const __m256d v_min = _mm256_set1_pd(min_val);
    const __m256d v_scale = _mm256_set1_pd(scale);
    const __m256d v_max_bin = _mm256_set1_pd(num_bins - 1);
    const __m256d v_zero = _mm256_setzero_pd();
    
    std::fill(output, output + num_bins, 0);
    
    #pragma omp parallel
    {
        std::vector<size_t> local_hist(num_bins, 0);
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; i += 8) {
            if (i + 64 < n) {
                _mm_prefetch((const char*)&data[i + 64], _MM_HINT_T0);
            }
            for (int j = 0; j < 2 && i + j*4 < n; ++j) {
                __m256d values = _mm256_loadu_pd(&data[i + j*4]);
                __m256d nan_mask = _mm256_cmp_pd(values, values, _CMP_ORD_Q);
                __m256d normalized = _mm256_sub_pd(values, v_min);
                __m256d bin_indices = _mm256_mul_pd(normalized, v_scale);
                bin_indices = _mm256_max_pd(bin_indices, v_zero);
                bin_indices = _mm256_min_pd(bin_indices, v_max_bin);
                __m128i indices = _mm256_cvtpd_epi32(bin_indices);
                alignas(16) int32_t idx[4];
                _mm_store_si128((__m128i*)idx, indices);
                int mask = _mm256_movemask_pd(nan_mask);
                if (mask & 0x1 && i + j*4 + 0 < n) local_hist[idx[0]]++;
                if (mask & 0x2 && i + j*4 + 1 < n) local_hist[idx[1]]++;
                if (mask & 0x4 && i + j*4 + 2 < n) local_hist[idx[2]]++;
                if (mask & 0x8 && i + j*4 + 3 < n) local_hist[idx[3]]++;
            }
        }
        
        #pragma omp critical
        {
            for (size_t i = 0; i < num_bins; ++i) {
                output[i] += local_hist[i];
            }
        }
    }
}

void compute_histogram_scalar(
    const double* data,
    size_t n,
    double min_val,
    double max_val,
    size_t num_bins,
    size_t* output
) {
    std::fill(output, output + num_bins, 0);
    const double scale = num_bins / (max_val - min_val);
    #pragma omp parallel
    {
        std::vector<size_t> local_hist(num_bins, 0);
        #pragma omp for
        for (size_t i = 0; i < n; ++i) {
            double val = data[i];
            if (!std::isnan(val) && val >= min_val && val <= max_val) {
                size_t bin = static_cast<size_t>((val - min_val) * scale);
                if (bin >= num_bins) bin = num_bins - 1;
                local_hist[bin]++;
            }
        }
        #pragma omp critical
        {
            for (size_t i = 0; i < num_bins; ++i) {
                output[i] += local_hist[i];
            }
        }
    }
}

// ===== WEIGHTED HISTOGRAMS =====

void compute_weighted_histogram_avx2(
    const double* __restrict__ data,
    const double* __restrict__ weights,
    size_t n,
    double min_val,
    double max_val,
    size_t num_bins,
    double* __restrict__ output
) {
    const double scale = num_bins / (max_val - min_val);
    const __m256d v_min = _mm256_set1_pd(min_val);
    const __m256d v_scale = _mm256_set1_pd(scale);
    const __m256d v_max_bin = _mm256_set1_pd(num_bins - 1);
    const __m256d v_zero = _mm256_setzero_pd();

    std::fill(output, output + num_bins, 0.0);

    #pragma omp parallel
    {
        std::vector<double> local_hist(num_bins, 0.0);
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; i += 4) {
            if (i + 64 < n) {
                _mm_prefetch((const char*)&data[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char*)&weights[i + 64], _MM_HINT_T0);
            }
            __m256d values = _mm256_loadu_pd(&data[i]);
            __m256d weight_vals = _mm256_loadu_pd(&weights[i]);
            __m256d nan_mask_data = _mm256_cmp_pd(values, values, _CMP_ORD_Q);
            __m256d nan_mask_weights = _mm256_cmp_pd(weight_vals, weight_vals, _CMP_ORD_Q);
            __m256d valid_mask = _mm256_and_pd(nan_mask_data, nan_mask_weights);
            __m256d normalized = _mm256_sub_pd(values, v_min);
            __m256d bin_indices = _mm256_mul_pd(normalized, v_scale);
            bin_indices = _mm256_max_pd(bin_indices, v_zero);
            bin_indices = _mm256_min_pd(bin_indices, v_max_bin);
            weight_vals = _mm256_and_pd(weight_vals, valid_mask);
            __m128i indices = _mm256_cvtpd_epi32(bin_indices);
            alignas(16) int32_t idx[4];
            _mm_store_si128((__m128i*)idx, indices);
            alignas(32) double w[4];
            _mm256_store_pd(w, weight_vals);
            for (int j = 0; j < 4 && i + j < n; ++j) {
                if (w[j] != 0.0) {
                    local_hist[idx[j]] += w[j];
                }
            }
        }
        #pragma omp critical
        {
            double* local = local_hist.data();
            for (size_t i = 0; i < num_bins; ++i) {
                output[i] += local[i];
            }
        }
    }
}

void compute_weighted_histogram_scalar(
    const double* data,
    const double* weights,
    size_t n,
    double min_val,
    double max_val,
    size_t num_bins,
    double* output
) {
    std::fill(output, output + num_bins, 0.0);
    const double scale = num_bins / (max_val - min_val);
    #pragma omp parallel
    {
        std::vector<double> local_hist(num_bins, 0.0);
        #pragma omp for schedule(guided)
        for (size_t i = 0; i < n; ++i) {
            double val = data[i];
            double weight = weights[i];
            if (!std::isnan(val) && !std::isnan(weight) && val >= min_val && val <= max_val && weight > 0) {
                size_t bin = static_cast<size_t>((val - min_val) * scale);
                if (bin >= num_bins) bin = num_bins - 1;
                local_hist[bin] += weight;
            }
        }
        #pragma omp critical
        {
            for (size_t i = 0; i < num_bins; ++i) {
                output[i] += local_hist[i];
            }
        }
    }
}

} // extern "C"
'''
        try:
            cache_dir = Path.home() / ".belle2_cache"
            cache_dir.mkdir(exist_ok=True)
            output_path = cache_dir / ("histogram_accelerator.dll" if platform.system() == "Windows" else "histogram_accelerator.so")
            compiled_path = self.compiler.compile_histogram_accelerator(cpp_source, output_path)
            if compiled_path and compiled_path.exists():
                lib = ctypes.CDLL(str(compiled_path))
                self._setup_function_signatures(lib)
                return lib
        except Exception as e:
            print(f"❌ Automatic compilation failed: {e}")
        return None

    def _setup_function_signatures(self, lib: ctypes.CDLL) -> None:
        try:
            lib.compute_histogram_avx2_enhanced.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            lib.compute_histogram_avx2_enhanced.restype = None

            lib.compute_histogram_scalar.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            lib.compute_histogram_scalar.restype = None

            if hasattr(lib, 'compute_weighted_histogram_avx2'):
                lib.compute_weighted_histogram_avx2.argtypes = [
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double),
                ]
                lib.compute_weighted_histogram_avx2.restype = None

            if hasattr(lib, 'compute_weighted_histogram_scalar'):
                lib.compute_weighted_histogram_scalar.argtypes = [
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double),
                ]
                lib.compute_weighted_histogram_scalar.restype = None
        except AttributeError as e:
            print(f"⚠️ Some C++ functions not available: {e}")

    def compute_histogram_parallel(self, data: np.ndarray, min_val: float, max_val: float, bins: int) -> np.ndarray:
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data, dtype=np.float64)
        elif data.dtype != np.float64:
            data = data.astype(np.float64)
        output = np.zeros(bins, dtype=np.uint64)
        if self.lib is not None:
            try:
                if hasattr(self.lib, 'compute_histogram_avx2_enhanced'):
                    self.lib.compute_histogram_avx2_enhanced(
                        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        len(data),
                        min_val,
                        max_val,
                        bins,
                        output.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
                    )
                    return output
                elif hasattr(self.lib, 'compute_histogram_scalar'):
                    self.lib.compute_histogram_scalar(
                        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        len(data),
                        min_val,
                        max_val,
                        bins,
                        output.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
                    )
                    return output
            except Exception as e:
                print(f"⚠️ C++ accelerator failed: {e}, falling back to NumPy")
        counts, _ = np.histogram(data, bins=bins, range=(min_val, max_val))
        return counts.astype(np.uint64)

    def benchmark_methods(self, test_data: np.ndarray, bins: int = 100, range_vals: Tuple[float, float] | None = None) -> Dict[str, float]:
        if range_vals is None:
            valid_data = test_data[~np.isnan(test_data)]
            range_vals = (float(np.min(valid_data)), float(np.max(valid_data)))
        methods: Dict[str, float] = {}
        start = time.time()
        np.histogram(test_data, bins=bins, range=range_vals)
        methods['numpy'] = time.time() - start
        if self.lib is not None:
            start = time.time()
            self.compute_histogram_parallel(test_data, range_vals[0], range_vals[1], bins)
            methods['cpp_accelerated'] = time.time() - start
        return methods


_cpp_accelerator: Optional[EnhancedCppAccelerator] = None
_accelerator_lock = threading.Lock()


def get_cpp_accelerator() -> EnhancedCppAccelerator:
    global _cpp_accelerator
    if _cpp_accelerator is None:
        with _accelerator_lock:
            if _cpp_accelerator is None:
                _cpp_accelerator = EnhancedCppAccelerator()
    return _cpp_accelerator


class cpp_histogram_integrator:
    # Expose a class attribute `lib` for compatibility with callers that access it directly
    lib: Optional[ctypes.CDLL] = None

    @classmethod
    def _ensure_loaded(cls):
        if cls.lib is None:
            cls.lib = get_cpp_accelerator().lib

    @staticmethod
    def compute_histogram_parallel(data: np.ndarray, min_val: float, max_val: float, bins: int) -> np.ndarray:
        accelerator = get_cpp_accelerator()
        # Keep class attribute in sync for external callers
        cpp_histogram_integrator.lib = accelerator.lib
        return accelerator.compute_histogram_parallel(data, min_val, max_val, bins)

    @classmethod
    def is_available(cls) -> bool:
        cls._ensure_loaded()
        return cls.lib is not None


if __name__ == "__main__":
    print("Enhanced C++ Integration for Belle II Analysis")
    print("=" * 50)
    accelerator = get_cpp_accelerator()
    print(f"C++ accelerator available: {accelerator.lib is not None}")