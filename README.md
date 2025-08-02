# Belle II Analysis Framework

A high-performance, multi-layered analysis framework for Belle II particle physics data processing, designed to handle billion-row datasets with optimal memory efficiency and computational performance.

## 🏗️ Architecture Overview

The framework employs a **compute-first, inverted architecture** with three distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Belle II Analysis Framework                   │
│                     Compute-First Architecture                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Layer 0: Core Protocols                      │
│  ┌───────────────┐ ┌──────────────────┐ ┌───────────────────┐  │
│  │ ComputeEngine │ │ DataFrameProtocol│ │ MaterializerProtocol │  │
│  │ Protocol      │ │                  │ │                   │  │
│  └───────────────┘ └──────────────────┘ └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   Layer 1: Compute Engines                      │
│  ┌──────────────┐ ┌─────────────────┐ ┌──────────────────────┐ │
│  │ LazyCompute  │ │ BillionCapable  │ │ IntegratedBillion    │ │
│  │ Engine       │ │ Engine          │ │ CapableEngine        │ │
│  └──────────────┘ └─────────────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                 Layer 2: Data Structures                        │
│  ┌──────────────┐ ┌─────────────────┐ ┌──────────────────────┐ │
│  │ UnifiedLazy  │ │ OptimizedUltra  │ │ MaterializationCtrl  │ │
│  │ DataFrame    │ │ LazyDict        │ │                      │ │
│  └──────────────┘ └─────────────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Compute-First Design**: Computational capabilities form the foundation, not data structures
2. **Lazy Evaluation**: Operations are deferred until materialization is required  
3. **Memory Efficiency**: Designed to process billion-row datasets within memory constraints
4. **Physics-Aware Optimizations**: Specialized for Belle II data patterns and analysis workflows
5. **Zero-Breaking-Changes**: Maintains backward compatibility with existing analysis code



### Belle II Physics Optimizations  
- **Process-Aware Loading**: Intelligent discovery and classification of Belle II analysis processes
- **Luminosity Processing**: Specialized handling of luminosity data with process name parsing
- **Physics Query Patterns**: Should support common Belle II analysis expressions and selections
- **Monte Carlo Integration**: Should naturally support MC weights and process classifications


## 📦 Installation

### Requirements

```python
# Core dependencies
polars >= 1.0.0
pandas >= 2.0.0
numpy >= 1.24.0
pyarrow >= 10.0.0
basf2 (recent release preferable, not yet tested)

# Optional performance dependencies
numba >= 0.57.0        # JIT compilation
psutil >= 5.9.0        # Memory monitoring  
matplotlib >= 3.6.0    # Visualization
seaborn >= 0.12.0      # Enhanced plotting

### Setup
```
```bash
# Clone the repository
git clone <repository-url>
cd belle2-analysis-framework

# Install in development mode
pip install -e .

# For HPC environments, configure OpenMP
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OMP_STACKSIZE=64M
```

### Multiple Access Patterns

The framework supports intuitive, pandas-like access patterns:

```python
# 1. Individual process access (standard dictionary)
process = data['P16M16rd_mc5S_mumu_p16_v1']
print(f"Process shape: {process.shape}")

# 2. Group access via dictionary key (intelligent fallback)
mumu_group = data['mumu']  # Returns LazyGroupProxy if 'mumu' is a group

# 3. Group access via attribute (syntactic sugar)
mumu_group = data.mumu
qqbar_group = data.qqbar

# 4. Group access via explicit method
mumu_group = data.group('mumu')

# 5. Broadcasting operations (all processes)
filtered_all = data.query('pRecoil > 2.0')
all_histograms = data.hist('mu1P', bins=50)
```

### Lazy Operation Chaining

Operations are lazy and optimized automatically:

```python
# Operations execute only at terminal method call
result = (data['mumu']
          .query('pRecoil > 2.0')
          .query('mu1P < 5.0') 
          .oneCandOnly()
          .hist('mu1Theta', bins=100))
#         ↑ lazy    ↑ lazy    ↑ lazy    ↑ executes here

# Performance analysis available
performance_report = result.performance_report()
for operation, metrics in performance_report.items():
    print(f"{operation}: {metrics['mean']:.3f}s average")
```

### Advanced Query Patterns

The framework includes specialized query conversion for Belle II physics patterns:

```python
# Complex Belle II physics expressions are automatically optimized
complex_selection = (
    "mu1nCDCHits>4 & mu2nCDCHits>4 & "
    "0.8>mu1clusterEoP & 0.8>mu2clusterEoP & "
    "2.6179938779914944>pRecoilTheta>0.29670597283903605 & "
    "11>totalMuonMomentum & absdPhi>1.5707963267948966"
)

# Automatically converted from pandas syntax to optimized Polars expressions
optimized_data = data.mumu.query(complex_selection)
```

### Performance Monitoring

```python
# Enable comprehensive performance tracking
data = load_belle2_vpho_integrated(
    base_dir="/path/to/data",
    enable_profiling=True,
    performance_target_rows_per_sec=20_000_000
)

# Get detailed performance analytics
profiling_report = data.get_profiling_report()
print(f"Cache hit rate: {profiling_report['cache_stats']['hit_rate']:.2%}")
print(f"Memory efficiency: {profiling_report['memory_stats']['efficiency']:.2%}")
```

## 🧪 Testing

### Running Tests

```python
# Layer 1 engine tests
from testing_suite_layer1 import run_comprehensive_tests
run_comprehensive_tests(memory_budget_gb=16.0)

# Pandas to Polars conversion tests  
from pandas_to_polars_queries import run_test_suite
run_test_suite()

# Integration tests
python3 -m pytest tests/ -v --tb=short
```
More to come ...
## 🏆 Performance Characteristics

### Benchmarked Performance TARGETS

- **Processing Speed**: at least 20M+ rows/second for typical Belle II queries
- **Memory Efficiency**: Process billion-row datasets with 16GB RAM
- **Query Optimization**: 10-100x speedup over pandas for physics expressions
- **Lazy Operations**: O(1) memory overhead for operation chaining
- **Cache Efficiency**: >80% hit rates for repeated operations


## 🔬 Architecture Deep Dive

### Memory Management Strategy

The framework implements a three-tier memory management approach:

1. **L1 Cache**: Metadata and small results in memory
2. **L2 Memory Pool**: Large intermediate results with smart eviction  
3. **L3 Disk Spill**: Compressed spilling for memory pressure situations

```python
# Memory pressure triggers automatic optimization
if memory_pressure > 0.8:
    # Switch to streaming mode
    engine.enable_streaming()
    # Increase compression
    engine.set_compression_level(9)
    # Reduce chunk sizes
    engine.set_chunk_size_gb(1.0)
```

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run full test suite
python3 -m pytest tests/ --cov=belle2_framework
```

### Code Organization

```
belle2-analysis-framework/
├── layer0/                    # Core protocols and interfaces
├── layer1/                    # Compute engines  
├── layer2/                    # Data structures and controllers
├── load_processes.py         # Main loading interface
├── pandas_to_polars_queries.py  # Query optimization
├── tests/                    # Comprehensive test suite
└── docs/                     # Documentation
```

### Performance Optimization Guidelines

1. **Lazy First**: Default to lazy evaluation for all operations
2. **Memory Aware**: Always consider memory implications of operations  
3. **Physics Optimized**: Leverage Belle II data patterns for optimization
4. **Cache Friendly**: Design operations to maximize cache utilization
5. **Profiling Enabled**: Include performance monitoring in all critical paths

## 📚 Additional Resources

### Documentation
- [Architecture Design Document](./docs/architecture.md)
- [Performance Tuning Guide](./docs/performance.md)  
- [Belle II Integration Guide](./docs/belle2_integration.md)
- [API Reference](./docs/api_reference.md)

## 🙏 Acknowledgments

- Belle II Collaboration for physics domain expertise
- Polars development team for high-performance DataFrame library
- Contributors to the HEP software ecosystem

---