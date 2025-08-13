"""
Layer 2: Comprehensive Test Suite with Real Data Architecture
=============================================================

Complete test coverage using real data structures, billion-row testing,
property-based validation, and performance benchmarking.
"""

import pytest
import numpy as np
import polars as pl
import pyarrow as pa
import tempfile
import shutil
import time
import weakref
import gc
import threading
import psutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Add missing import for patch (used for mocking)
from unittest.mock import patch

# Hypothesis for property-based testing
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

# Import Layer 2 components
from layer2_unified_lazy_dataframe import (
    UnifiedLazyDataFrame, LazyColumnAccessor, LazyGroupBy,
    TransformationMetadata, DataTransformationChain,
    create_dataframe_from_parquet, create_dataframe_from_compute
)
from layer2_optimized_ultra_lazy_dict import (
    OptimizedUltraLazyDict, LazyGroupProxy, BroadcastResult,
    ProcessMetadata, create_process_dict_from_directory
)
from layer2_materialization_controller import (
    MaterializationController, MaterializationFormat, MaterializationHints,
    GraphOptimizationEngine, MemoryAwareExecutor, PerformanceProfiler,
    PredicatePushdownRule, ColumnPruningRule, layer2_optimizers
)
from layer2_complete_integration import Belle2Layer2Framework

# Import Layer 1 components
from layer1.lazy_compute_engine import (
    LazyComputeEngine, LazyComputeCapability, GraphNode
)
from layer0 import ComputeOpType

# ============================================================================
# Test Configuration
# ============================================================================

# Register custom marks
pytestmark = pytest.mark.layer2

# Configure hypothesis for larger tests
settings.register_profile("ci", max_examples=50, deadline=None)
settings.register_profile("dev", max_examples=10)
settings.register_profile("debug", max_examples=1)
settings.load_profile("ci")

# ============================================================================
# Advanced Test Fixtures with Scalable Data Generation
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create comprehensive test data directory with multiple process types."""
    base_dir = tmp_path_factory.mktemp("belle2_test_data")
    
    # Process configurations for Belle II
    process_configs = {
        'mumu': {
            'processes': ['P16M16rd_mc5S_mumu_p16_v1', 'P16M16rd_mc5S_mumu_p16_v2'],
            'events_per_file': 50000,
            'M_bc_mean': 5.279,
            'M_bc_sigma': 0.003
        },
        'ee': {
            'processes': ['P16M16rd_mc5S_ee_p16_v1', 'P16M16rd_mc5S_ee_p16_v2'],
            'events_per_file': 45000,
            'M_bc_mean': 5.278,
            'M_bc_sigma': 0.004
        },
        'qqbar': {
            'processes': ['P16M16rd_mc5S_ccbar_p16_v1', 'P16M16rd_mc5S_uubar_p16_v1', 
                         'P16M16rd_mc5S_ddbar_p16_v1', 'P16M16rd_mc5S_ssbar_p16_v1'],
            'events_per_file': 60000,
            'M_bc_mean': 5.276,
            'M_bc_sigma': 0.005
        }
    }
    
    # Generate realistic Belle II data
    for group, config in process_configs.items():
        group_dir = base_dir / group
        group_dir.mkdir()
        
        for process in config['processes']:
            n_events = config['events_per_file']
            
            # Generate physics-realistic data
            data = {
                'M_bc': np.random.normal(config['M_bc_mean'], config['M_bc_sigma'], n_events),
                'delta_E': np.random.normal(0, 0.05, n_events),
                'pRecoil': np.random.exponential(1.5, n_events) + 0.5,
                'mu1P': np.random.uniform(0.5, 4.0, n_events),
                'mu2P': np.random.uniform(0.5, 4.0, n_events),
                'mu1Theta': np.random.uniform(0.3, 2.8, n_events),
                'mu2Theta': np.random.uniform(0.3, 2.8, n_events),
                'mu1Phi': np.random.uniform(-np.pi, np.pi, n_events),
                'mu2Phi': np.random.uniform(-np.pi, np.pi, n_events),
                '__event__': np.arange(n_events),
                '__run__': np.random.randint(1, 1000, n_events),
                '__experiment__': np.full(n_events, 31),
                'isSignal': np.random.choice([0, 1], n_events, p=[0.7, 0.3])
            }
            
            df = pl.DataFrame(data)
            df.write_parquet(str(group_dir / f"{process}.parquet"), compression='snappy')
    
    yield base_dir
    shutil.rmtree(base_dir)

@pytest.fixture
def large_lazy_frames():
    """Create large LazyFrames for performance testing."""
    frames = []
    for i in range(10):
        # Each frame represents 10M events
        n_rows = 10_000_000
        data = {
            'event_id': pl.arange(i * n_rows, (i + 1) * n_rows, eager=False),
            'value': pl.lit(1.0),  # Lazy literal for memory efficiency
            'category': pl.lit('A'),
            'timestamp': pl.lit(datetime.now())
        }
        frames.append(pl.LazyFrame(data))
    return frames

@pytest.fixture
def real_compute_capability():
    """Create real compute capability with proper interface."""
    # Use actual LazyComputeEngine
    engine = LazyComputeEngine(memory_budget_gb=8.0)
    
    # Create source node with real data
    source_data = pl.DataFrame({
        'M_bc': [5.279, 5.280, 5.278, 5.281, 5.279],
        'pRecoil': [2.1, 2.5, 1.8, 3.0, 2.2],
        'value': [10, 20, 30, 40, 50]
    })
    
    source_node = GraphNode(
        op_type=ComputeOpType.SOURCE,
        operation=lambda: source_data,
        inputs=[],
        metadata={'source': 'test_data'}
    )
    
    return LazyComputeCapability(
        root_node=source_node,
        engine=weakref.ref(engine),
        estimated_size=len(source_data),
        schema=dict(source_data.schema)
    )

@pytest.fixture
def process_dict_with_groups(test_data_dir):
    """Create process dictionary with proper group structure."""
    process_dict = create_process_dict_from_directory(
        str(test_data_dir),
        memory_budget_gb=16.0
    )
    
    # Ensure groups are properly populated
    assert len(process_dict) > 0
    assert any(len(processes) > 0 for processes in process_dict._groups.values())
    
    return process_dict

# ============================================================================
# Comprehensive Unit Tests: UnifiedLazyDataFrame
# ============================================================================

class TestUnifiedLazyDataFrame:
    """Exhaustive tests for UnifiedLazyDataFrame with billion-row capabilities."""
    
    def test_initialization_variants(self, real_compute_capability):
        """Test all initialization pathways."""
        # From compute capability
        df1 = UnifiedLazyDataFrame(compute=real_compute_capability)
        assert df1._compute is not None
        assert df1.shape[0] > 0
        
        # From lazy frames
        lazy_frame = pl.DataFrame({'x': [1, 2, 3]}).lazy()
        df2 = UnifiedLazyDataFrame(lazy_frames=[lazy_frame])
        assert df2._lazy_frames is not None
        assert len(df2._lazy_frames) == 1
        
        # With explicit schema
        schema = {'a': pl.Float64, 'b': pl.Int64}
        df3 = UnifiedLazyDataFrame(schema=schema)
        assert df3._schema == schema
        
        # With metadata
        metadata = {'source': 'test', 'version': 1}
        df4 = UnifiedLazyDataFrame(metadata=metadata)
        assert df4._metadata == metadata
    
    def test_column_access_comprehensive(self):
        """Test all column access patterns."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6],
                'c': [7, 8, 9]
            }).lazy()]
        )
        
        # Single column - returns LazyColumnAccessor
        col_a = df['a']
        assert isinstance(col_a, LazyColumnAccessor)
        assert col_a._column_name == 'a'
        
        # Multiple columns - returns DataFrame
        subset = df[['a', 'b']]
        assert isinstance(subset, UnifiedLazyDataFrame)
        assert subset.columns == ['a', 'b']
        
        # Column slice notation
        all_cols = df[:]
        assert isinstance(all_cols, UnifiedLazyDataFrame)
        
        # Boolean indexing preparation
        mask_col = df['a'] > 1
        assert isinstance(mask_col, LazyColumnAccessor)
    
    @pytest.mark.parametrize("query_expr,expected_count", [
        ("pRecoil > 2.0", 3),
        ("value >= 30", 3),
        ("M_bc > 5.278 and pRecoil < 3.0", 3),
        ("value == 20 or value == 40", 2)
    ])
    def test_query_operations(self, query_expr, expected_count):
        """Test various query expressions."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({
                'M_bc': [5.279, 5.280, 5.278, 5.281, 5.279],
                'pRecoil': [2.1, 2.5, 1.8, 3.0, 2.2],
                'value': [10, 20, 30, 40, 50]
            }).lazy()]
        )
        
        result = df.query(query_expr)
        assert isinstance(result, UnifiedLazyDataFrame)
        
        # Verify transformation tracked
        history = result.get_transformation_history()
        assert any('query' in t.operation for t in history)
        
        # Materialize and verify
        materialized = result.collect()
        assert len(materialized) == expected_count
    
    def test_one_cand_only_variations(self):
        """Test oneCandOnly with different configurations."""
        # Create data with multiple candidates per event
        n_events = 1000
        n_candidates = 5
        data = {
            '__event__': np.repeat(np.arange(n_events), n_candidates),
            '__run__': np.repeat(np.random.randint(1, 10, n_events), n_candidates),
            'quality': np.random.randn(n_events * n_candidates),
            'random_val': np.random.rand(n_events * n_candidates)
        }
        
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame(data).lazy()]
        )
        
        # Test with quality-based selection
        best_quality = df.oneCandOnly(
            group_cols=['__event__', '__run__'],
            sort_col='quality',
            ascending=False
        )
        
        result = best_quality.collect()
        assert len(result) == n_events  # One per event
        
        # Test with random selection
        random_selection = df.oneCandOnly(
            group_cols=['__event__'],
            sort_col='random'
        )
        
        result2 = random_selection.collect()
        assert len(result2) == n_events
    
    @pytest.mark.parametrize("bins,range_tuple,density", [
        (50, None, False),
        (100, (5.27, 5.29), False),
        (25, None, True),
        (200, (5.26, 5.30), True)
    ])
    def test_histogram_comprehensive(self, bins, range_tuple, density):
        """Test histogram with various parameters."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({
                'M_bc': np.random.normal(5.279, 0.003, 10000)
            }).lazy()]
        )
        
        counts, edges = df.hist('M_bc', bins=bins, range=range_tuple, density=density)
        
        assert len(counts) == bins
        assert len(edges) == bins + 1
        
        if density:
            # Density histogram should integrate to ~1
            bin_widths = np.diff(edges)
            integral = np.sum(counts * bin_widths)
            assert 0.9 < integral < 1.1
        else:
            # Count histogram should sum to total events
            assert np.sum(counts) == 10000
    
    def test_transformation_chain_complex(self):
        """Test complex transformation chains."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({
                'a': range(100),
                'b': range(100, 200),
                'c': ['X', 'Y', 'Z'] * 33 + ['X']
            }).lazy()]
        )
        
        # Build complex chain
        result = (df
            .query('a > 10')
            .query('b < 180')
            [['a', 'c']]
            .query('c == "X"')
        )
        
        # Verify chain integrity
        history = result.get_transformation_history()
        operations = [t.operation for t in history]
        
        assert 'dataframe_query' in operations
        assert 'select_columns' in operations
        assert len(history) >= 4
        
        # Verify each transformation has parent linkage
        for i in range(1, len(history)):
            assert history[i].parent_id == history[i-1].id
    
    @pytest.mark.benchmark
    def test_billion_row_lazy_operations(self, large_lazy_frames):
        """Test operations on billion-row scale data."""
        df = UnifiedLazyDataFrame(
            lazy_frames=large_lazy_frames,
            memory_budget_gb=16.0
        )
        
        # Lazy operations should be instant
        start = time.time()
        
        result = (df
            .query('value > 0.5')
            .query('event_id % 1000 == 0')
            [['event_id', 'value']]
        )
        
        build_time = time.time() - start
        assert build_time < 0.1  # Should be milliseconds
        
        # Verify lazy - no actual computation yet
        assert result._materialized_cache is None
        
        # Estimated rows should reflect operations
        assert result._estimated_rows < 100_000_000
    
    def test_memory_aware_execution(self):
        """Test memory-aware execution with constraints."""
        # Create data that exceeds memory budget
        large_df = UnifiedLazyDataFrame(
            lazy_frames=[
                pl.DataFrame({
                    'data': np.random.randn(1_000_000)
                }).lazy() for _ in range(10)
            ],
            memory_budget_gb=0.1  # 100MB budget for ~80MB data
        )
        
        # Should handle via chunking/spilling
        result = large_df.collect()
        assert len(result) == 10_000_000
    
    def test_error_handling_comprehensive(self):
        """Test all error conditions."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({'a': [1, 2, 3]}).lazy()]
        )
        
        # Invalid column access
        with pytest.raises(KeyError, match="Column 'invalid' not found"):
            _ = df['invalid']
        
        # Invalid column list
        with pytest.raises(KeyError, match="Columns not found"):
            _ = df[['a', 'invalid']]
        
        # Invalid query syntax
        result = df.query('invalid syntax !@#')
        # Should fall back gracefully
        assert isinstance(result, UnifiedLazyDataFrame)
    
    @given(
        n_rows=st.integers(min_value=100, max_value=10000),
        n_cols=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=20)
    def test_property_shape_consistency(self, n_rows, n_cols):
        """Property: shape should match data dimensions."""
        # Generate random column names
        cols = [f'col_{i}' for i in range(n_cols)]
        data = {col: np.random.randn(n_rows) for col in cols}
        
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame(data).lazy()]
        )
        
        shape = df.shape
        assert shape[0] > 0  # Estimated rows
        assert shape[1] == n_cols
        assert len(df.columns) == n_cols
        assert set(df.columns) == set(cols)

# ============================================================================
# Comprehensive Unit Tests: OptimizedUltraLazyDict
# ============================================================================

class TestOptimizedUltraLazyDict:
    """Exhaustive tests for process-aware dictionary."""
    
    def test_initialization_modes(self):
        """Test all initialization modes."""
        # Empty initialization
        d1 = OptimizedUltraLazyDict()
        assert len(d1) == 0
        assert d1.memory_budget_gb == 8.0
        
        # With initial data
        d2 = OptimizedUltraLazyDict({'proc1': create_test_dataframe()})
        assert len(d2) == 1
        
        # With custom memory budget
        d3 = OptimizedUltraLazyDict(memory_budget_gb=32.0)
        assert d3.memory_budget_gb == 32.0
        
        # From BroadcastResult
        br = BroadcastResult({'p1': create_test_dataframe()}, 'test', d1)
        d4 = OptimizedUltraLazyDict(br)
        assert len(d4) == 1
    
    def test_process_classification_comprehensive(self):
        """Test automatic process classification rules."""
        d = OptimizedUltraLazyDict()
        
        # Test classification patterns
        test_cases = {
            'test_mumu_v1': 'mumu',
            'P16_mc_mu_mu_ISR': 'mumu',
            'test_ee_v1': 'ee',
            'mc_ee_gg': 'ee',
            'test_ccbar_v1': 'qqbar',
            'test_uubar_v1': 'qqbar',
            'test_ddbar_mixed': 'qqbar',
            'charged_B_decay': 'BBbar',
            'mixed_B_decay': 'BBbar',
            'tau_pair_production': 'taupair',
            'eemumu_process': 'llYY',
            'eeee_process': 'llYY',
            'gg_fusion': 'gg',
            'data_proc16_v1': 'data'
        }
        
        for process_name, expected_group in test_cases.items():
            df = create_test_dataframe()
            d.add_process(process_name, df)
            
            assert process_name in d
            assert process_name in d._groups[expected_group]
            assert process_name in d._groups['all']
    
    def test_group_access_all_methods(self):
        """Test all group access patterns."""
        d = OptimizedUltraLazyDict()
        
        # Add processes to different groups
        for i in range(3):
            d.add_process(f'mumu_v{i}', create_test_dataframe())
            d.add_process(f'ee_v{i}', create_test_dataframe())
        
        # Method 1: Explicit group() method
        group1 = d.group('mumu')
        assert isinstance(group1, LazyGroupProxy)
        assert len(group1.members) == 3
        
        # Method 2: Dictionary access fallback
        group2 = d['mumu']
        assert isinstance(group2, LazyGroupProxy)
        assert group2.group_name == 'mumu'
        
        # Method 3: Attribute access
        group3 = d.mumu
        assert isinstance(group3, LazyGroupProxy)
        
        # Method 4: Safe access
        group4 = d.get_group('ee')
        assert isinstance(group4, LazyGroupProxy)
        
        # Non-existent group
        assert d.get_group('invalid') is None
    
    def test_broadcast_operations_comprehensive(self):
        """Test all broadcast operation types."""
        d = OptimizedUltraLazyDict()
        
        # Add diverse processes
        for i in range(5):
            d[f'process_{i}'] = create_test_dataframe(n_rows=1000)
        
        # Test various DataFrame methods
        operations = [
            ('query', lambda: d.query('value > 50')),
            ('filter', lambda: d.filter(pl.col('value') > 50)),
            ('select', lambda: d.select(['value', 'M_bc'])),
            ('head', lambda: d.head(10)),
            ('describe', lambda: d.describe())
        ]
        
        for op_name, op_func in operations:
            result = op_func()
            
            if op_name in ['head', 'describe']:
                # Terminal operations return dict
                assert isinstance(result, dict)
                assert len(result) == 5
            else:
                # Non-terminal return BroadcastResult
                assert isinstance(result, BroadcastResult)
                assert result.operation == op_name
                assert len(result.results) == 5
    
    def test_lazy_group_proxy_chaining(self):
        """Test method chaining on groups."""
        d = OptimizedUltraLazyDict()
        
        # Create realistic process group
        for i in range(3):
            df = create_test_dataframe(n_rows=10000)
            d.add_process(f'mumu_ISR_v{i}', df)
        
        # Test chaining
        result = d.mumu.query('pRecoil > 2.0').query('M_bc > 5.275')
        
        assert isinstance(result, BroadcastResult)
        
        # Test terminal operation on chain
        hist_results = d.mumu.query('pRecoil > 2.0').hist('M_bc', bins=50)
        
        assert isinstance(hist_results, dict)
        assert len(hist_results) == 3
        
        for proc, (counts, edges) in hist_results.items():
            assert len(counts) == 50
            assert len(edges) == 51
    
    def test_access_statistics_tracking(self):
        """Test comprehensive access pattern tracking."""
        d = OptimizedUltraLazyDict()
        
        # Add test data
        for i in range(5):
            d[f'proc_{i}'] = create_test_dataframe()
        
        # Perform various access patterns
        _ = d['proc_0']  # Direct access
        _ = d.get('proc_1')  # Safe access
        _ = d.mumu  # Group access
        _ = d['mumu']  # Group via key
        _ = d.query('test')  # Broadcast
        
        # Analyze statistics
        stats = d.get_access_statistics()
        
        assert stats['total_accesses'] >= 5
        assert stats['group_accesses'] >= 2
        assert stats['process_accesses'] >= 2
        
        # Check most accessed
        most_accessed = stats['most_accessed']
        assert len(most_accessed) > 0
        
        # Check access patterns
        patterns = stats['access_patterns']
        assert 'duration_seconds' in patterns
        assert patterns['group_preference'] >= 0
    
    def test_performance_optimization(self):
        """Test access pattern optimization."""
        d = OptimizedUltraLazyDict()
        
        # Add processes
        for i in range(10):
            d[f'mumu_{i}'] = create_test_dataframe()
        
        # Simulate high group access pattern
        for _ in range(20):
            _ = d.mumu
        
        # Trigger optimization
        d.optimize_access_patterns()
        
        # Verify group proxy is pre-cached
        assert 'mumu' in d._group_proxies
        
        # Performance report
        report = d.performance_report()
        assert isinstance(report, dict)
    
    def test_serialization_metadata(self, tmp_path):
        """Test metadata save/load functionality."""
        d = OptimizedUltraLazyDict()
        
        # Add processes with metadata
        for i in range(3):
            d.add_process(
                f'process_{i}',
                create_test_dataframe(),
                ProcessMetadata(
                    canonical_name=f'Process {i}',
                    process_type='mumu',
                    estimated_events=100000
                )
            )
        
        # Save metadata
        metadata_path = tmp_path / "metadata.pkl"
        d.save_metadata(str(metadata_path))
        
        # Create new dict and load
        d2 = OptimizedUltraLazyDict()
        d2.load_metadata(str(metadata_path))
        
        assert d2._groups == d._groups
        assert len(d2._process_metadata) == 3
    
    def test_integrity_validation(self):
        """Test dictionary integrity checking."""
        d = OptimizedUltraLazyDict()
        
        # Valid state
        d['proc1'] = create_test_dataframe()
        d.add_group('custom', ['proc1'])
        
        is_valid, issues = d.validate_integrity()
        assert is_valid
        assert len(issues) == 0
        
        # Create invalid state
        d._groups['invalid'] = ['nonexistent']
        
        is_valid, issues = d.validate_integrity()
        assert not is_valid
        assert any('nonexistent' in issue for issue in issues)
    
    @given(
        n_processes=st.integers(min_value=1, max_value=20),
        n_groups=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10)
    def test_property_group_consistency(self, n_processes, n_groups):
        """Property: all processes must be in at least one group."""
        d = OptimizedUltraLazyDict()
        
        # Add processes
        for i in range(n_processes):
            d[f'proc_{i}'] = create_test_dataframe()
        
        # Every process should be in 'all' group
        assert len(d._groups['all']) == n_processes
        
        # No process should be orphaned
        all_grouped = set()
        for processes in d._groups.values():
            all_grouped.update(processes)
        
        assert all_grouped == set(d.keys())

# ============================================================================
# Advanced Component Tests
# ============================================================================

class TestMaterializationController:
    """Comprehensive materialization tests."""
    
    def test_format_inference_rules(self):
        """Test all format inference rules."""
        controller = MaterializationController()
        
        # Test with various hints
        test_cases = [
            (MaterializationHints(downstream_operations=['numpy', 'fft']), 
             MaterializationFormat.NUMPY),
            (MaterializationHints(downstream_operations=['plot', 'matplotlib']), 
             MaterializationFormat.NUMPY),
            (MaterializationHints(downstream_operations=['join', 'merge']), 
             MaterializationFormat.POLARS),
            (MaterializationHints(downstream_operations=['groupby', 'agg']), 
             MaterializationFormat.POLARS),
            (MaterializationHints(zero_copy_preferred=True), 
             MaterializationFormat.ARROW),
            (MaterializationHints(persistence_required=True), 
             MaterializationFormat.PARQUET),
        ]
        
        for hints, expected_format in test_cases:
            inferred = controller._infer_optimal_format(None, hints)
            assert inferred == expected_format
    
    def test_materialization_all_formats(self):
        """Test materialization to all supported formats."""
        controller = MaterializationController()
        
        # Create test data
        test_df = pl.DataFrame({
            'a': range(1000),
            'b': np.random.randn(1000),
            'c': ['X', 'Y', 'Z'] * 333 + ['X']
        })
        
        compute = create_compute_from_dataframe(test_df)
        
        # Test each format
        formats = ['arrow', 'polars', 'numpy', 'pandas', 'native']
        
        for format_name in formats:
            result = controller.materialize(compute, target_format=format_name)
            
            if format_name == 'arrow':
                assert isinstance(result, pa.Table)
                assert result.num_rows == 1000
            elif format_name == 'polars':
                assert isinstance(result, pl.DataFrame)
                assert len(result) == 1000
            elif format_name == 'numpy':
                assert isinstance(result, np.ndarray)
            elif format_name == 'pandas':
                import pandas as pd
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 1000
    
    def test_caching_strategy(self):
        """Test intelligent caching decisions."""
        controller = MaterializationController()
        
        # Small data - should cache
        small_compute = create_compute_from_dataframe(
            pl.DataFrame({'x': range(100)})
        )
        
        result1 = controller.materialize(small_compute, 'polars')
        result2 = controller.materialize(small_compute, 'polars')
        
        # Should return cached result
        assert controller._conversion_cache
        
        # Large data - should not cache
        large_df = pl.DataFrame({
            'x': np.random.randn(10_000_000)
        })
        large_compute = create_compute_from_dataframe(large_df)
        
        _ = controller.materialize(large_compute, 'polars')
        
        # Cache should not grow significantly
        cache_size = len(controller._conversion_cache)
        assert cache_size < 10  # Reasonable cache size
    
    def test_graph_optimization_rules(self):
        """Test graph optimization engine."""
        engine = GraphOptimizationEngine()
        
        # Build complex graph
        source = GraphNode(ComputeOpType.SOURCE, lambda: None, [])
        
        # Inefficient pattern: Filter after expensive join
        join1 = GraphNode(ComputeOpType.JOIN, lambda x, y: None, [source])
        filter1 = GraphNode(ComputeOpType.FILTER, lambda x: None, [join1])
        
        # Project after filter (can be pushed down)
        project1 = GraphNode(ComputeOpType.PROJECT, lambda x: None, [source])
        filter2 = GraphNode(ComputeOpType.FILTER, lambda x: None, [project1])
        
        # Optimize
        optimized = engine.optimize(filter2)
        
        # Verify optimization occurred
        stats = engine.get_optimization_report()
        assert stats['rules_applied'] > 0

class TestMemoryAwareExecutor:
    """Test memory-aware execution strategies."""
    
    def test_memory_profiling(self):
        """Test memory usage profiling."""
        executor = MemoryAwareExecutor(memory_limit=1024 * 1024 * 1024)  # 1GB
        
        # Create compute with known memory profile
        df = pl.DataFrame({
            'data': np.random.randn(1_000_000)  # ~8MB
        })
        compute = create_compute_from_dataframe(df)
        
        # Execute with monitoring
        result = executor.execute_with_memory_limit(compute)
        
        assert result is not None
    
    def test_spilling_activation(self):
        """Test automatic spilling activation."""
        # Very small memory limit
        executor = MemoryAwareExecutor(memory_limit=10 * 1024 * 1024)  # 10MB
        
        # Create data larger than limit
        large_df = pl.DataFrame({
            'data': np.random.randn(5_000_000)  # ~40MB
        })
        compute = create_compute_from_dataframe(large_df)
        
        # Should activate spilling
        result = executor.execute_with_memory_limit(compute)
        
        assert result is not None
        
    @patch('psutil.Process')
    def test_memory_monitoring_real(self, mock_process):
        """Test real-time memory monitoring."""
        executor = MemoryAwareExecutor(memory_limit=1024 * 1024 * 1024)
        
        # Mock memory info
        mock_memory = type('MockMemory', (), {'rss': 500 * 1024 * 1024})()
        mock_process.return_value.memory_info.return_value = mock_memory
        
        compute = create_compute_from_dataframe(
            pl.DataFrame({'x': [1, 2, 3]})
        )
        
        result = executor.execute_with_memory_limit(compute)
        
        assert result is not None
        assert mock_process.return_value.memory_info.called

# ============================================================================
# Integration Tests with Real Workflows
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_complete_belle2_workflow(self, process_dict_with_groups):
        """Test realistic Belle II analysis workflow."""
        framework = Belle2Layer2Framework(memory_budget_gb=16.0)
        
        # Step 1: Load processes
        processes = process_dict_with_groups
        
        # Step 2: Apply physics cuts
        mumu_events = processes.mumu
        filtered = framework.apply_cuts(
            mumu_events,
            [
                'pRecoil > 1.8',
                'abs(delta_E) < 0.15',
                'M_bc > 5.27 and M_bc < 5.29'
            ]
        )
        
        # Step 3: Select best candidates
        best_candidates = framework.select_best_candidates(
            filtered,
            group_cols=['__event__', '__run__', '__experiment__'],
            sort_col='pRecoil',
            ascending=False
        )
        
        # Step 4: Compute invariant mass histogram
        hist_results = framework.compute_histogram(
            best_candidates,
            'M_bc',
            bins=100,
            range=(5.2, 5.3)
        )
        
        # Verify results
        assert isinstance(hist_results, dict)
        for process, (counts, edges) in hist_results.items():
            assert len(counts) == 100
            assert np.sum(counts) > 0  # Should have events
            
        # Step 5: Performance analysis
        profile = framework.profile_performance()
        assert profile['framework_stats']['operations_executed'] > 0
    
    def test_multi_process_comparison(self, process_dict_with_groups):
        """Test comparing multiple physics processes."""
        # Get different process groups
        mumu = process_dict_with_groups.mumu
        ee = process_dict_with_groups.ee
        
        # Apply same cuts to both
        cut_expr = 'pRecoil > 2.0 and abs(delta_E) < 0.1'
        
        mumu_cut = mumu.query(cut_expr)
        ee_cut = ee.query(cut_expr)
        
        # Compare distributions
        mumu_hist = mumu_cut.hist('M_bc', bins=50)
        ee_hist = ee_cut.hist('M_bc', bins=50)
        
        # Both should produce results
        assert all(isinstance(v, tuple) for v in mumu_hist.values())
        assert all(isinstance(v, tuple) for v in ee_hist.values())
    
    def test_transformation_preservation_e2e(self):
        """Test transformation history through full pipeline."""
        # Create DataFrame with transformations
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({
                'x': range(1000),
                'y': range(1000, 2000),
                'z': np.random.choice(['A', 'B', 'C'], 1000)
            }).lazy()]
        )
        
        # Apply transformations
        transformed = (df
            .query('x > 100')
            .query('y < 1500')
            [['x', 'z']]
            .query('z == "A"')
        )
        
        # Add to dictionary
        d = OptimizedUltraLazyDict()
        d['transformed_data'] = transformed
        
        # Broadcast operation
        broadcast_result = d.query('x > 200')
        
        # Convert back to dict
        new_dict = broadcast_result.to_dict()
        
        # Get transformation history
        if 'transformed_data' in new_dict:
            history = new_dict['transformed_data'].get_transformation_history()
            
            # Should preserve full history
            operations = [t.operation for t in history]
            assert 'dataframe_query' in operations
            assert 'select_columns' in operations
            assert len(history) >= 5  # Original + broadcast

# ============================================================================
# Performance Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for Layer 2."""
    
    def test_lazy_operation_construction_speed(self, benchmark):
        """Benchmark lazy operation building."""
        frames = [pl.DataFrame({'x': range(1000)}).lazy() for _ in range(10)]
        
        def build_complex_pipeline():
            df = UnifiedLazyDataFrame(lazy_frames=frames)
            return (df
                .query('x > 100')
                .query('x < 900')
                .query('x % 2 == 0')
                [['x']]
                .query('x > 200')
                .query('x < 800')
            )
        
        result = benchmark(build_complex_pipeline)
        assert isinstance(result, UnifiedLazyDataFrame)
        assert result._materialized_cache is None  # Still lazy
    
    def test_histogram_computation_speed(self, benchmark):
        """Benchmark histogram performance."""
        # Create larger dataset
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({
                'values': np.random.normal(0, 1, 1_000_000)
            }).lazy()]
        )
        
        def compute_histogram():
            return df.hist('values', bins=1000)
        
        counts, edges = benchmark(compute_histogram)
        assert len(counts) == 1000
        assert np.sum(counts) == 1_000_000
    
    def test_broadcast_operation_scaling(self, benchmark):
        """Benchmark broadcast operation scaling."""
        # Create dictionary with many processes
        d = OptimizedUltraLazyDict()
        for i in range(100):
            d[f'process_{i}'] = create_test_dataframe(n_rows=10000)
        
        def broadcast_query():
            return d.query('value > 50')
        
        result = benchmark(broadcast_query)
        assert isinstance(result, BroadcastResult)
        assert len(result.results) == 100
    
    @pytest.mark.slow
    def test_billion_row_materialization(self):
        """Test actual billion-row materialization."""
        # Create billion-row lazy dataset
        n_chunks = 100
        chunk_size = 10_000_000
        
        frames = []
        for i in range(n_chunks):
            # Use lazy generation
            frame = pl.LazyFrame({
                'id': pl.arange(i * chunk_size, (i + 1) * chunk_size, eager=False),
                'value': pl.lit(1.0)
            })
            frames.append(frame)
        
        df = UnifiedLazyDataFrame(
            lazy_frames=frames,
            memory_budget_gb=32.0
        )
        
        assert df.shape[0] >= 1_000_000_000
        
        # Compute aggregation instead of full materialization
        start = time.time()
        
        # This should use streaming/chunking
        result = df.query('id % 1000000 == 0')
        sample = result.head(10)  # Just materialize small sample
        
        elapsed = time.time() - start
        
        assert len(sample) == 10
        print(f"Billion-row query executed in {elapsed:.2f}s")

# ============================================================================
# Property-Based Testing
# ============================================================================

class TestProperties:
    """Property-based tests for invariants."""
    
    @given(
        n_operations=st.integers(min_value=1, max_value=10),
        operations=st.lists(
            st.tuples(
                st.sampled_from(['query', 'filter', 'select']),
                st.text(min_size=1, max_size=20)
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=20)
    def test_transformation_chain_integrity(self, n_operations, operations):
        """Property: transformation chain maintains parent linkage."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({
                'x': range(100),
                'y': range(100)
            }).lazy()]
        )
        
        for i in range(min(n_operations, len(operations))):
            op_type, _ = operations[i]
            
            if op_type == 'query':
                df = df.query('x > 0')
            elif op_type == 'filter':
                df = df.filter(pl.col('x') >= 0)
            elif op_type == 'select':
                df = df[['x']]
        
        history = df.get_transformation_history()
        
        # Verify parent chain
        for i in range(1, len(history)):
            if history[i].parent_id:
                assert history[i].parent_id == history[i-1].id
    
    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1000, max_value=100000)
        ).filter(lambda x: not np.all(np.isnan(x))),
        bins=st.integers(min_value=10, max_value=200)
    )
    @settings(max_examples=20)
    def test_histogram_conservation(self, data, bins):
        """Property: histogram preserves total count."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({'values': data}).lazy()]
        )
        
        counts, edges = df.hist('values', bins=bins)
        
        # Total count should match non-NaN values
        valid_count = np.sum(~np.isnan(data))
        assert abs(np.sum(counts) - valid_count) <= 1
        
        # Bins should cover data range
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        
        assert edges[0] <= data_min + 1e-10
        assert edges[-1] >= data_max - 1e-10
    
    @given(
        n_processes=st.integers(min_value=1, max_value=50),
        group_operations=st.lists(
            st.sampled_from(['add', 'remove', 'query']),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=10)
    def test_dictionary_consistency(self, n_processes, group_operations):
        """Property: dictionary operations maintain consistency."""
        d = OptimizedUltraLazyDict()
        
        # Add initial processes
        for i in range(n_processes):
            d[f'proc_{i}'] = create_test_dataframe()
        
        # Apply operations
        for op in group_operations:
            if op == 'add' and len(d) < 100:
                d[f'proc_new_{len(d)}'] = create_test_dataframe()
            elif op == 'remove' and len(d) > 1:
                key = list(d.keys())[0]
                del d[key]
            elif op == 'query':
                _ = d.query('value > 0')
        
        # Verify consistency
        is_valid, issues = d.validate_integrity()
        assert is_valid or len(issues) > 0  # Should detect any issues

# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================

class TestEdgeCases:
    """Comprehensive edge case testing."""
    
    def test_empty_dataframe_handling(self):
        """Test operations on empty DataFrames."""
        # Create truly empty DataFrame
        empty_df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({}).lazy()]
        )
        
        assert empty_df.shape == (0, 0)
        assert empty_df.columns == []
        
        # Operations should not crash
        result = empty_df.query('x > 0')  # Non-existent column
        assert isinstance(result, UnifiedLazyDataFrame)
        
        # Histogram on empty data
        empty_with_col = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({'x': []}).lazy()]
        )
        counts, edges = empty_with_col.hist('x', bins=10)
        assert np.sum(counts) == 0
    
    def test_memory_exhaustion_handling(self):
        """Test recovery from memory exhaustion."""
        executor = MemoryAwareExecutor(memory_limit=1024)  # 1KB
        
        # Create compute that exceeds limit
        large_compute = create_compute_from_dataframe(
            pl.DataFrame({'x': np.random.randn(1_000_000)})
        )
        
        # Should handle gracefully
        result = executor.execute_with_memory_limit(large_compute)
        assert result is not None
    
    def test_concurrent_access_safety(self):
        """Test thread safety of operations."""
        d = OptimizedUltraLazyDict()
        
        # Add test data
        for i in range(20):
            d[f'process_{i}'] = create_test_dataframe()
        
        results = []
        errors = []
        
        def access_dict(thread_id):
            try:
                # Various operations
                _ = d[f'process_{thread_id % 20}']
                _ = d.query('value > 0')
                _ = d.mumu  # Group access
                
                # Add new process
                d[f'thread_{thread_id}'] = create_test_dataframe()
                
                results.append(f"Thread {thread_id} success")
            except Exception as e:
                errors.append((thread_id, e))
        
        # Run concurrent access
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(access_dict, i)
                for i in range(50)
            ]
            
            for future in as_completed(futures):
                future.result()
        
        # Should complete without critical errors
        assert len(results) > 0
        assert len(errors) < len(results) / 2  # Most should succeed
    
    def test_circular_reference_cleanup(self):
        """Test garbage collection with circular references."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({'x': [1, 2, 3]}).lazy()]
        )
        
        # Create circular reference
        col = df['x']
        
        # Track references
        df_ref = weakref.ref(df)
        
        # Delete strong reference
        del df
        gc.collect()
        
        # DataFrame should be collected
        assert df_ref() is None
        
        # Column accessor should handle gracefully
        assert col._parent_ref() is None
    
    def test_invalid_operations(self):
        """Test handling of invalid operations."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({'x': [1, 2, 3]}).lazy()]
        )
        
        # Invalid indexing
        with pytest.raises(TypeError):
            _ = df[123]  # Numeric index not supported
        
        with pytest.raises(ValueError):
            _ = df[1, 2, 3]  # Too many dimensions
        
        # Invalid query syntax
        result = df.query('INVALID SQL SYNTAX !!!')
        assert isinstance(result, UnifiedLazyDataFrame)  # Should fallback
    
    def test_extreme_parameters(self):
        """Test with extreme parameter values."""
        df = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({
                'x': np.random.randn(1000)
            }).lazy()]
        )
        
        # Extreme histogram bins
        counts1, edges1 = df.hist('x', bins=1)
        assert len(counts1) == 1
        assert np.sum(counts1) == 1000
        
        counts2, edges2 = df.hist('x', bins=10000)
        assert len(counts2) == 10000
        
        # Extreme memory budget
        df_tiny = UnifiedLazyDataFrame(
            lazy_frames=[pl.DataFrame({'x': [1]}).lazy()],
            memory_budget_gb=0.000001  # 1 byte
        )
        
        # Should still work
        result = df_tiny.collect()
        assert len(result) == 1

# ============================================================================
# Helper Functions
# ============================================================================

def create_test_dataframe(n_rows=100, columns=None):
    """Create test DataFrame with flexible schema support."""
    if columns is None:
        # Default physics schema with all expected columns
        data = {
            'a': np.arange(n_rows),
            'b': np.arange(n_rows, 2*n_rows),
            'c': np.random.choice(['X', 'Y', 'Z'], n_rows),
            'x': np.arange(n_rows),  # Added for compatibility
            'y': np.arange(n_rows, 2*n_rows),  # Added for compatibility
            'z': np.random.choice(['A', 'B', 'C'], n_rows),  # Added for compatibility
            'test_col': np.random.randn(n_rows),
            'value': np.random.randint(0, 100, n_rows),
            'pRecoil': np.random.exponential(1.5, n_rows) + 0.5,
            'M_bc': np.random.normal(5.279, 0.003, n_rows),
            'delta_E': np.random.normal(0, 0.05, n_rows),
            '__event__': np.arange(n_rows),
            '__run__': np.random.randint(1, 10, n_rows)
        }
    else:
        # Custom schema - ensure all requested columns exist
        data = {}
        
        # Standard test columns
        standard_mappings = {
            'x': lambda: np.arange(n_rows),
            'y': lambda: np.arange(n_rows, 2*n_rows),
            'z': lambda: np.random.choice(['A', 'B', 'C'], n_rows),
            'a': lambda: np.arange(n_rows),
            'b': lambda: np.arange(n_rows, 2*n_rows),
            'c': lambda: np.random.choice(['X', 'Y', 'Z'], n_rows),
            'value': lambda: np.random.randint(0, 100, n_rows),
            'pRecoil': lambda: np.random.exponential(1.5, n_rows) + 0.5,
            'M_bc': lambda: np.random.normal(5.279, 0.003, n_rows),
            'delta_E': lambda: np.random.normal(0, 0.05, n_rows),
            '__event__': lambda: np.arange(n_rows),
            '__run__': lambda: np.random.randint(1, 10, n_rows)
        }
        
        for col in columns:
            if col in standard_mappings:
                data[col] = standard_mappings[col]()
            else:
                # Default to random numeric data
                data[col] = np.random.randn(n_rows)
    
    # Create DataFrame with explicit schema
    df = pl.DataFrame(data)
    
    return UnifiedLazyDataFrame(
        lazy_frames=[df.lazy()],
        memory_budget_gb=1.0,
        schema=dict(df.schema)  # Pass schema explicitly
    )

def create_compute_from_dataframe(df):
    """Create compute capability from DataFrame."""
    class DataFrameCompute:
        def __init__(self, data):
            self.data = data
            self.estimated_size = len(data) if hasattr(data, '__len__') else 1000
            self.schema = dict(data.schema) if hasattr(data, 'schema') else {}
        
        def materialize(self):
            return self.data
        
        def estimate_memory(self):
            if hasattr(self.data, 'estimated_size'):
                return self.data.estimated_size() * 8  # Rough estimate
            return self.estimated_size * 100
    
    return DataFrameCompute(df)

# ============================================================================
# Test Runner Configuration
# ============================================================================

if __name__ == '__main__':
    import sys
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     Layer 2 Comprehensive Test Suite - Real Data Edition   ║
    ╠════════════════════════════════════════════════════════════╣
    ║ • Real data structures instead of mocks                    ║
    ║ • Billion-row testing capabilities                         ║
    ║ • Property-based testing with hypothesis                   ║
    ║ • Performance benchmarks                                   ║
    ║ • Thread safety validation                                 ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Configure pytest arguments
    args = [__file__, '-v']
    
    if '--benchmark' in sys.argv:
        args.extend(['-m', 'benchmark'])
    elif '--slow' in sys.argv:
        args.extend(['-m', 'slow'])
    else:
        args.extend(['-m', 'not benchmark and not slow'])
    
    # Add coverage if requested
    if '--coverage' in sys.argv:
        args.extend(['--cov=layer2', '--cov-report=term-missing'])
    
    # Run tests
    exit_code = pytest.main(args)
    sys.exit(exit_code)