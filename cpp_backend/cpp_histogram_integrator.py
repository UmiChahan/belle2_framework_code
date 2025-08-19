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
import sys
import warnings
from typing import Optional, Tuple, Union, Dict, Any, List
import threading
import time
from dataclasses import dataclass

@dataclass
class CompilationConfig:
    """Configuration for C++ compilation."""
    optimization_level: str = "O3"
    enable_openmp: bool = True
    enable_avx2: bool = True
    enable_native_arch: bool = True
    enable_fast_math: bool = True
    debug_symbols: bool = False


class IntelligentCppCompiler:
    """
    Intelligent C++ compiler with automatic optimization detection and fallback.
    
    Features:
    - Automatic compiler detection (GCC, Clang, MSVC)
    - CPU feature detection (AVX2, SSE, etc.)
    - Automatic optimization level selection
    - Graceful fallback for compilation failures
    """
    
    def __init__(self, config: Optional[CompilationConfig] = None):
        self.config = config or CompilationConfig()
        self.available_compilers = self._detect_compilers()
        self.cpu_features = self._detect_cpu_features()
        self.compilation_cache = {}
        self._lock = threading.Lock()
        
    def _detect_compilers(self) -> Dict[str, Dict[str, Any]]:
        """Detect available C++ compilers and their capabilities."""
        compilers = {}
        
        # Test g++
        try:
            result = subprocess.run(['g++', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                version = self._extract_version(version_line)
                # Assume modern g++ supports OpenMP and AVX2
                compilers['g++'] = {
                    'command': 'g++',
                    'version': version,
                    'supports_openmp': True,
                    'supports_avx2': True,
                    'priority': 90
                }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Test Clang
        try:
            result = subprocess.run(['clang++', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                version = self._extract_version(version_line)
                compilers['clang'] = {
                    'command': 'clang++',
                    'version': version,
                    'supports_openmp': version >= (3, 7),
                    'supports_avx2': version >= (3, 3),
                    'priority': 85
                }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Test MSVC (Windows)
        if platform.system() == "Windows":
            try:
                result = subprocess.run(['cl'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or 'Microsoft' in result.stderr:
                    compilers['msvc'] = {
                        'command': 'cl',
                        'version': (19, 0),  # Approximate
                        'supports_openmp': True,
                        'supports_avx2': True,
                        'priority': 80
                    }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return compilers
    
    def _extract_version(self, version_string: str) -> Tuple[int, int]:
        """Extract major.minor version from version string."""
        import re
        match = re.search(r'(\d+)\.(\d+)', version_string)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)
    
    def _detect_cpu_features(self) -> Dict[str, bool]:
        """Detect available CPU features."""
        features = {
            'sse2': False,
            'sse4_1': False,
            'avx': False,
            'avx2': False,
            'fma': False
        }
        
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            
            for feature in features:
                features[feature] = feature in flags or feature.replace('_', '.') in flags
                
        except ImportError:
            # Fallback: assume modern CPU
            features.update({
                'sse2': True,
                'sse4_1': True,
                'avx': True,
                'avx2': True,
                'fma': True
            })
        
        return features
    
    def get_best_compiler(self) -> Optional[Dict[str, Any]]:
        """Get the best available compiler for our needs."""
        if not self.available_compilers:
            return None
        
        # Sort by priority and capabilities
        candidates = []
        for name, compiler in self.available_compilers.items():
            score = compiler['priority']
            
            # Bonus for OpenMP support if needed
            if self.config.enable_openmp and compiler['supports_openmp']:
                score += 10
            
            # Bonus for AVX2 support if needed
            if self.config.enable_avx2 and compiler['supports_avx2']:
                score += 5
            
            candidates.append((score, name, compiler))
        
        if candidates:
            return max(candidates)[2]
        return None
    
    def compile_histogram_accelerator(self, source_code: str, 
                                    output_path: Optional[Path] = None) -> Optional[Path]:
        """Compile histogram accelerator with intelligent optimization."""
        
        # Generate cache key
        cache_key = hash((source_code, str(self.config)))
        
        with self._lock:
            if cache_key in self.compilation_cache:
                cached_path = self.compilation_cache[cache_key]
                if cached_path and cached_path.exists():
                    return cached_path
        
        # Get best compiler
        compiler = self.get_best_compiler()
        if not compiler:
            warnings.warn("No suitable C++ compiler found")
            return None
        
        try:
            # Create temporary source file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(source_code)
                source_path = Path(f.name)
            
            # Determine output path
            if output_path is None:
                if platform.system() == "Windows":
                    output_path = source_path.with_suffix('.dll')
                else:
                    output_path = source_path.with_suffix('.so')
            
            # Build compilation command
            cmd = self._build_compile_command(compiler, source_path, output_path)
            
            # Compile with timeout
            print(f"🔧 Compiling C++ accelerator with {compiler['command']}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ C++ accelerator compiled successfully: {output_path}")
                
                # Cache successful compilation
                with self._lock:
                    self.compilation_cache[cache_key] = output_path
                
                # Clean up source file
                try:
                    source_path.unlink()
                except:
                    pass
                
                return output_path
            else:
                print(f"❌ Compilation failed:")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ Compilation timed out")
            return None
        except Exception as e:
            print(f"❌ Compilation error: {e}")
            return None
        finally:
            # Clean up
            try:
                if 'source_path' in locals():
                    source_path.unlink()
            except:
                pass
    
    def _build_compile_command(self, compiler: Dict[str, Any], 
                             source_path: Path, output_path: Path) -> List[str]:
        """Build optimized compilation command with correct flags."""
        cmd = [compiler['command']]
        
        if compiler['command'] == 'cl':  # MSVC
            cmd.extend([
                f"/O{self.config.optimization_level[-1]}",  # /O2 or /O3
                "/MD",  # Multi-threaded DLL runtime
                "/LD",  # Create DLL
                f"/Fe:{output_path}",
                str(source_path)
            ])
            
            if self.config.enable_openmp:
                cmd.append("/openmp")
            
            if self.config.enable_avx2 and self.cpu_features.get('avx2', False):
                cmd.append("/arch:AVX2")
                
        else:  # GCC or Clang
            cmd.extend([
                f"-{self.config.optimization_level}",
                "-fPIC",
                "-shared",
                "-o", str(output_path),
                str(source_path),
                "-std=c++17",  # Ensure modern C++
                "-Wall",       # Enable all warnings
                "-Wextra",     # Extra warnings
                "-Wno-unused-parameter", # Ignore unused param warnings
                "-Wno-sign-compare",     # Ignore sign compare warnings
                "-Wno-missing-field-initializers", # Ignore missing field warnings
                "-lstdc++"      # Explicitly link C++ stdlib
            ])
            
            if self.config.enable_openmp and compiler['supports_openmp']:
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


class EnhancedCppAccelerator:
    """
    Enhanced C++ accelerator with automatic compilation and intelligent fallbacks.
    
    Features:
    - Automatic C++ compilation with optimization
    - Multiple algorithm implementations
    - Performance benchmarking and selection
    - Graceful fallback to pure Python/NumPy
    """
    
    def __init__(self, auto_compile: bool = True):
        self.auto_compile = auto_compile
        self.lib = None
        self.compiler = IntelligentCppCompiler()
        self.performance_data = {}
        self._initialization_lock = threading.Lock()
        
        if auto_compile:
            self._initialize_accelerator()
    
    def _initialize_accelerator(self):
        """Initialize C++ accelerator with automatic compilation."""
        with self._initialization_lock:
            if self.lib is not None:
                return  # Already initialized
            
            print("🚀 Initializing enhanced C++ accelerator...")
            
            # Try to load existing library
            existing_lib = self._try_load_existing_library()
            if existing_lib:
                self.lib = existing_lib
                print("✅ Loaded existing C++ accelerator")
                return
            
            # Compile new library
            if self.auto_compile:
                compiled_lib = self._compile_accelerator()
                if compiled_lib:
                    self.lib = compiled_lib
                    print("✅ Compiled and loaded new C++ accelerator")
                    return
            
            print("⚠️ C++ accelerator not available, using fallback implementations")
    
    def _try_load_existing_library(self) -> Optional[ctypes.CDLL]:
        """Try to load existing compiled library."""
        potential_names = [
            "histogram_accelerator.so",
            "histogram_accelerator.dll",
            "libhistogram_accelerator.so",
            "libhistogram_accelerator.dll"
        ]
        
        search_paths = [
            Path.cwd(),
            Path(__file__).parent,
            Path.home() / ".belle2_cache",
            Path("/tmp") if platform.system() != "Windows" else Path(os.environ.get("TEMP", ""))
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
        """Compile C++ accelerator automatically."""
        # Enhanced C++ source with multiple algorithms
        cpp_source = '''
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
    
    // Clear output
    std::fill(output, output + num_bins, 0);
    
    #pragma omp parallel
    {
        std::vector<size_t> local_hist(num_bins, 0);
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; i += 8) {
            // Prefetch next cache line
            if (i + 64 < n) {
                _mm_prefetch((const char*)&data[i + 64], _MM_HINT_T0);
            }
            
            // Process 8 elements in two AVX2 operations
            for (int j = 0; j < 2 && i + j*4 < n; ++j) {
                __m256d values = _mm256_loadu_pd(&data[i + j*4]);
                
                // Check for NaN
                __m256d nan_mask = _mm256_cmp_pd(values, values, _CMP_ORD_Q);
                
                // Compute bin indices
                __m256d normalized = _mm256_sub_pd(values, v_min);
                __m256d bin_indices = _mm256_mul_pd(normalized, v_scale);
                bin_indices = _mm256_max_pd(bin_indices, v_zero);
                bin_indices = _mm256_min_pd(bin_indices, v_max_bin);
                
                // Convert to integers
                __m128i indices = _mm256_cvtpd_epi32(bin_indices);
                
                // Extract and update histogram
                alignas(16) int32_t idx[4];
                _mm_store_si128((__m128i*)idx, indices);
                
                int mask = _mm256_movemask_pd(nan_mask);
                if (mask & 0x1 && i + j*4 + 0 < n) local_hist[idx[0]]++;
                if (mask & 0x2 && i + j*4 + 1 < n) local_hist[idx[1]]++;
                if (mask & 0x4 && i + j*4 + 2 < n) local_hist[idx[2]]++;
                if (mask & 0x8 && i + j*4 + 3 < n) local_hist[idx[3]]++;
            }
        }
        
        // Merge local histograms
        #pragma omp critical
        {
            for (size_t i = 0; i < num_bins; ++i) {
                output[i] += local_hist[i];
            }
        }
    }
}

// Fallback scalar implementation
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

// Weighted AVX2 histogram (double precision weights)
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

    // Clear output
    std::fill(output, output + num_bins, 0.0);

    #pragma omp parallel
    {
        std::vector<double> local_hist(num_bins, 0.0);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; i += 4) {
            // Prefetch next cache line
            if (i + 64 < n) {
                _mm_prefetch((const char*)&data[i + 64], _MM_HINT_T0);
                _mm_prefetch((const char*)&weights[i + 64], _MM_HINT_T0);
            }

            __m256d values = _mm256_loadu_pd(&data[i]);
            __m256d w = _mm256_loadu_pd(&weights[i]);

            // Check for NaN
            __m256d nan_mask = _mm256_cmp_pd(values, values, _CMP_ORD_Q);

            __m256d normalized = _mm256_sub_pd(values, v_min);
            __m256d bin_indices = _mm256_mul_pd(normalized, v_scale);
            bin_indices = _mm256_max_pd(bin_indices, v_zero);
            bin_indices = _mm256_min_pd(bin_indices, v_max_bin);

            __m128i indices = _mm256_cvtpd_epi32(bin_indices);

            alignas(16) int32_t idx[4];
            _mm_store_si128((__m128i*)idx, indices);

            alignas(32) double w_arr[4];
            _mm256_store_pd(w_arr, w);

            int mask = _mm256_movemask_pd(nan_mask);
            if (mask & 0x1 && i + 0 < n) local_hist[idx[0]] += w_arr[0];
            if (mask & 0x2 && i + 1 < n) local_hist[idx[1]] += w_arr[1];
            if (mask & 0x4 && i + 2 < n) local_hist[idx[2]] += w_arr[2];
            if (mask & 0x8 && i + 3 < n) local_hist[idx[3]] += w_arr[3];
        }

        // Merge local histograms
        #pragma omp critical
        {
            for (size_t i = 0; i < num_bins; ++i) {
                output[i] += local_hist[i];
            }
        }
    }
}

// Weighted scalar implementation
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

        #pragma omp for
        for (size_t i = 0; i < n; ++i) {
            double val = data[i];
            if (!std::isnan(val) && val >= min_val && val <= max_val) {
                size_t bin = static_cast<size_t>((val - min_val) * scale);
                if (bin >= num_bins) bin = num_bins - 1;
                local_hist[bin] += weights[i];
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
            # Create cache directory
            cache_dir = Path.home() / ".belle2_cache"
            cache_dir.mkdir(exist_ok=True)

            # Compile
            output_path = cache_dir / ("histogram_accelerator.dll" if platform.system() == "Windows" else "histogram_accelerator.so")
            compiled_path = self.compiler.compile_histogram_accelerator(cpp_source, output_path)

            if compiled_path and compiled_path.exists():
                lib = ctypes.CDLL(str(compiled_path))
                self._setup_function_signatures(lib)
                return lib

        except Exception as e:
            print(f"❌ Automatic compilation failed: {e}")

        return None
    
    def _setup_function_signatures(self, lib):
        """Setup function signatures for C library."""
        try:
            # Enhanced AVX2 function
            lib.compute_histogram_avx2_enhanced.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t)
            ]
            lib.compute_histogram_avx2_enhanced.restype = None
            
            # Scalar fallback
            lib.compute_histogram_scalar.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t)
            ]
            lib.compute_histogram_scalar.restype = None
            
            # Weighted AVX2 (optional)
            if hasattr(lib, 'compute_weighted_histogram_avx2'):
                lib.compute_weighted_histogram_avx2.argtypes = [
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double)
                ]
                lib.compute_weighted_histogram_avx2.restype = None

            # Weighted scalar (optional)
            if hasattr(lib, 'compute_weighted_histogram_scalar'):
                lib.compute_weighted_histogram_scalar.argtypes = [
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double)
                ]
                lib.compute_weighted_histogram_scalar.restype = None
            
        except AttributeError as e:
            print(f"⚠️ Some C++ functions not available: {e}")
    
    def compute_histogram_parallel(self, data: np.ndarray, min_val: float, 
                                 max_val: float, bins: int) -> np.ndarray:
        """Compute histogram using best available method."""
        
        # Ensure data is contiguous and double precision
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data, dtype=np.float64)
        elif data.dtype != np.float64:
            data = data.astype(np.float64)
        
        # Prepare output
        output = np.zeros(bins, dtype=np.uint64)
        
        if self.lib is not None:
            try:
                # Try enhanced AVX2 version first
                if hasattr(self.lib, 'compute_histogram_avx2_enhanced'):
                    self.lib.compute_histogram_avx2_enhanced(
                        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        len(data),
                        min_val,
                        max_val,
                        bins,
                        output.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
                    )
                    return output
                
                # Fallback to scalar version
                elif hasattr(self.lib, 'compute_histogram_scalar'):
                    self.lib.compute_histogram_scalar(
                        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        len(data),
                        min_val,
                        max_val,
                        bins,
                        output.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
                    )
                    return output
                    
            except Exception as e:
                print(f"⚠️ C++ accelerator failed: {e}, falling back to NumPy")
        
        # NumPy fallback
        counts, _ = np.histogram(data, bins=bins, range=(min_val, max_val))
        return counts.astype(np.uint64)

    def compute_weighted_histogram_parallel(self, data: np.ndarray, weights: np.ndarray,
                                            min_val: float, max_val: float, bins: int) -> np.ndarray:
        """Compute weighted histogram using best available method."""
        # Ensure data is contiguous and double precision
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data, dtype=np.float64)
        elif data.dtype != np.float64:
            data = data.astype(np.float64)

        if not weights.flags['C_CONTIGUOUS']:
            weights = np.ascontiguousarray(weights, dtype=np.float64)
        elif weights.dtype != np.float64:
            weights = weights.astype(np.float64)

        if data.shape[0] != weights.shape[0]:
            raise ValueError("data and weights must have the same length")

        output = np.zeros(bins, dtype=np.float64)

        if self.lib is not None:
            try:
                if hasattr(self.lib, 'compute_weighted_histogram_avx2'):
                    self.lib.compute_weighted_histogram_avx2(
                        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        len(data),
                        min_val,
                        max_val,
                        bins,
                        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    )
                    return output
                elif hasattr(self.lib, 'compute_weighted_histogram_scalar'):
                    self.lib.compute_weighted_histogram_scalar(
                        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        len(data),
                        min_val,
                        max_val,
                        bins,
                        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    )
                    return output
            except Exception as e:
                print(f"⚠️ C++ weighted accelerator failed: {e}, falling back to NumPy")

        # NumPy fallback
        counts, _ = np.histogram(data, bins=bins, range=(min_val, max_val), weights=weights)
        return counts.astype(np.float64)
    
    def benchmark_methods(self, test_data: np.ndarray, bins: int = 100, 
                         range_vals: Tuple[float, float] = None) -> Dict[str, float]:
        """Benchmark available histogram methods."""
        
        if range_vals is None:
            valid_data = test_data[~np.isnan(test_data)]
            range_vals = (np.min(valid_data), np.max(valid_data))
        
        methods = {}
        
        # Test NumPy baseline
        start = time.time()
        np.histogram(test_data, bins=bins, range=range_vals)
        methods['numpy'] = time.time() - start
        
        # Test C++ if available
        if self.lib is not None:
            start = time.time()
            self.compute_histogram_parallel(test_data, range_vals[0], range_vals[1], bins)
            methods['cpp_accelerated'] = time.time() - start
        
        return methods


# Global accelerator instance
_cpp_accelerator = None
_accelerator_lock = threading.Lock()


def get_cpp_accelerator() -> EnhancedCppAccelerator:
    """Get global C++ accelerator instance (thread-safe singleton)."""
    global _cpp_accelerator
    
    if _cpp_accelerator is None:
        with _accelerator_lock:
            if _cpp_accelerator is None:
                _cpp_accelerator = EnhancedCppAccelerator()
    
    return _cpp_accelerator


# Integration module for easy import
class cpp_histogram_integrator:
    """Drop-in replacement for C++ histogram integrator."""
    @classmethod
    def lib(cls):
        """Delegate to singleton instance."""
        return get_cpp_accelerator().lib
    
    @staticmethod
    def compute_histogram_parallel(data: np.ndarray, min_val: float, 
                                 max_val: float, bins: int) -> np.ndarray:
        """Compute parallel histogram using enhanced C++ accelerator."""
        accelerator = get_cpp_accelerator()
        return accelerator.compute_histogram_parallel(data, min_val, max_val, bins)
    
    @staticmethod
    def compute_weighted_histogram_parallel(data: np.ndarray, weights: np.ndarray,
                                            min_val: float, max_val: float, bins: int) -> np.ndarray:
        """Compute weighted parallel histogram using enhanced C++ accelerator."""
        accelerator = get_cpp_accelerator()
        return accelerator.compute_weighted_histogram_parallel(data, weights, min_val, max_val, bins)
    
    @staticmethod
    def is_available() -> bool:
        """Check if C++ accelerator is available."""
        accelerator = get_cpp_accelerator()
        return accelerator.lib is not None


if __name__ == "__main__":
    print("Enhanced C++ Integration for Belle II Analysis")
    print("=" * 50)
    
    # Test the accelerator
    accelerator = get_cpp_accelerator()
    
    print(f"C++ accelerator available: {accelerator.lib is not None}")
    
    if accelerator.lib is not None:
        # Benchmark with test data
        test_data = np.random.randn(1_000_000_000)
        test_data[np.random.rand(1_000_000_000) < 0.1] = np.nan
        
        print("\nBenchmarking histogram methods...")
        results = accelerator.benchmark_methods(test_data)
        
        for method, time_taken in results.items():
            print(f"  {method}: {time_taken:.4f}s")
        
        if 'cpp_accelerated' in results and 'numpy' in results:
            speedup = results['numpy'] / results['cpp_accelerated']
            print(f"\nSpeedup: {speedup:.2f}x")