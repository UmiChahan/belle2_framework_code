"""
Belle II Enhanced Production Framework v3.0
===========================================

A refined implementation that addresses the dual-purpose data structure problem
and provides seamless histogram integration with luminosity weighting.

Key Improvements:
- Clear separation between data and weights
- Unified histogram pipeline with consistent C++ acceleration
- Simplified architecture while maintaining backward compatibility
- Better type safety and error handling
- Complete implementation of all critical methods

Author: Enhanced Architecture v3.0
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Callable, TypeVar
from collections import defaultdict
from dataclasses import dataclass, field
import time
import gc
import warnings
import weakref
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import functools
from enum import Enum, auto
import re
import pickle
import hashlib
from datetime import datetime
import inspect
import psutil

# Import Layer2 components (assuming they're available)
from layer2_unified_lazy_dataframe import UnifiedLazyDataFrame, TransformationMetadata, DataTransformationChain
from layer2_optimized_ultra_lazy_dict import OptimizedUltraLazyDict, BroadcastResult, LazyGroupProxy
from layer2_complete_integration import Belle2Layer2Framework
from layer2_materialization_controller import MaterializationHints, MaterializationFormat

# Try to import C++ acceleration
try:
    from optimized_cpp_integration import OptimizedStreamingHistogram, configure_openmp_for_hpc
    CPP_HISTOGRAM_AVAILABLE = True
except ImportError:
    warnings.warn("C++ histogram acceleration not available")
    CPP_HISTOGRAM_AVAILABLE = False
    OptimizedStreamingHistogram = None

T = TypeVar('T')

# ============================================================================
# WEIGHT MANAGEMENT SYSTEM
# ============================================================================

@dataclass
class LuminosityWeight:
    """Encapsulates luminosity weight information for a process."""
    process_name: str
    weight: float
    energy_condition: str
    calculation_method: str
    is_data: bool = False
    mc_luminosity: Optional[float] = None
    data_luminosity: Optional[float] = None
    
    def __post_init__(self):
        """Validate weight values."""
        if not self.is_data and self.weight <= 0:
            warnings.warn(f"Invalid weight {self.weight} for MC process {self.process_name}")
            self.weight = 1.0

class WeightedDataFrame:
    """
    A DataFrame with attached luminosity weight metadata.
    
    This cleanly separates the data (DataFrame) from the weight (metadata),
    solving the dual-purpose data structure problem.
    """
    
    def __init__(self, 
                 dataframe: UnifiedLazyDataFrame,
                 weight: Optional[LuminosityWeight] = None,
                 histogram_engine: Optional[Any] = None):
        self._dataframe = dataframe
        self._weight = weight
        self._histogram_engine = histogram_engine or self._get_global_histogram_engine()
        
        # Ensure histogram engine is propagated to the DataFrame
        if hasattr(self._dataframe, '_histogram_engine'):
            self._dataframe._histogram_engine = self._histogram_engine
    
    @property
    def df(self) -> UnifiedLazyDataFrame:
        """Access the underlying DataFrame."""
        return self._dataframe
    
    @property
    def weight(self) -> float:
        """Get the luminosity weight value."""
        return self._weight.weight if self._weight else 1.0
    
    @property
    def weight_info(self) -> Optional[LuminosityWeight]:
        """Get full weight information."""
        return self._weight
    
    def _get_global_histogram_engine(self):
        """Get or create global histogram engine."""
        if CPP_HISTOGRAM_AVAILABLE:
            if not hasattr(WeightedDataFrame, '_global_histogram_engine'):
                WeightedDataFrame._global_histogram_engine = OptimizedStreamingHistogram()
            return WeightedDataFrame._global_histogram_engine
        return None 
    
    def hist(self, column: str, bins: int = 50, 
             range: Optional[Tuple[float, float]] = None,
             density: bool = False, 
             apply_weight: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram with automatic weight application.
        
        This provides a clean interface where weights are applied transparently.
        """
        # Get raw histogram from DataFrame
        counts, edges = self._dataframe.hist(column, bins=bins, range=range, density=density)
        
        # Apply weight if requested and available
        if apply_weight and self._weight and not self._weight.is_data:
            counts = counts * self.weight
        
        return counts, edges
    
    def query(self, expr: str) -> 'WeightedDataFrame':
        """Query that preserves weight information."""
        new_df = self._dataframe.query(expr)
        return WeightedDataFrame(new_df, self._weight, self._histogram_engine)
    

    def oneCandOnly(self, *args, **kwargs) -> 'WeightedDataFrame':
        """Select one candidate with weight preservation and estimation tracking."""
        # Execute transformation
        new_df = self._dataframe.oneCandOnly(*args, **kwargs)
        
        # Create result with preserved weight and engine
        result = WeightedDataFrame(new_df, self._weight, self._histogram_engine)
        
        # Log estimation change if available (non-invasive)
        if hasattr(self._dataframe, '_estimated_rows') and hasattr(new_df, '_estimated_rows'):
            before = self._dataframe._estimated_rows
            after = new_df._estimated_rows
            if before and after and after != before:
                print(f"   📊 {self._weight.process_name if self._weight else 'Process'}: "
                      f"{before:,} → {after:,} rows (oneCandOnly)")
        
        return result
    
    def __getattr__(self, name):
        """Enhanced delegation for framework properties."""
        # Priority list for direct delegation
        framework_properties = {
            '_estimated_rows', '_transformation_chain', '_schema',
            '_metadata', '_compute_graph', '_operation_cache'
        }
        
        if name in framework_properties:
            return getattr(self._dataframe, name, None)
        
        # Standard delegation
        return getattr(self._dataframe, name)
    
    def __setattr__(self, name, value):
        """Bidirectional property synchronization."""
        if name.startswith('_') and name not in ['_dataframe', '_weight', '_histogram_engine']:
            # Framework properties go to underlying dataframe
            if hasattr(self, '_dataframe') and hasattr(self._dataframe, name):
                setattr(self._dataframe, name, value)
        
        super().__setattr__(name, value)
    def createDeltaColumns(self) -> 'WeightedDataFrame':
        """Create delta columns with weight preservation."""
        new_df = self._dataframe.createDeltaColumns()
        return WeightedDataFrame(new_df, self._weight, self._histogram_engine)
    
    # Delegate all other methods to the DataFrame
    def __getattr__(self, name):
        """Delegate attribute access to the DataFrame."""
        # Check if it's a DataFrame method
        if hasattr(self._dataframe, name):
            attr = getattr(self._dataframe, name)
            
            # If it's a method that returns a DataFrame, wrap the result
            if callable(attr):
                @functools.wraps(attr)
                def wrapped_method(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    
                    # If result is a DataFrame, wrap it with the same weight
                    if isinstance(result, UnifiedLazyDataFrame):
                        return WeightedDataFrame(result, self._weight, self._histogram_engine)
                    return result
                
                return wrapped_method
            return attr
        
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    # Properties that should be delegated
    @property
    def shape(self):
        return self._dataframe.shape
    
    @property
    def columns(self):
        return self._dataframe.columns
    
    @property
    def dtypes(self):
        return self._dataframe.dtypes

# ============================================================================
# ENHANCED PROCESS DICTIONARY
# ============================================================================

class WeightedProcessDict(OptimizedUltraLazyDict):
    """
    Enhanced process dictionary that cleanly manages weights separately from data.
    
    This solves the dual-purpose problem by maintaining weights as metadata
    rather than trying to store them in the same container as DataFrames.
    """
    
    def __init__(self, *args, **kwargs):
        self._energy_condition = kwargs.pop('energy_condition', '5S_scan')
        super().__init__(*args, **kwargs)
        self._weights: Dict[str, LuminosityWeight] = {}
        self._histogram_engine = None
        self._luminosity_manager = None
        
    
    def add_process_with_weight(self, 
                               name: str, 
                               dataframe: UnifiedLazyDataFrame,
                               weight: Optional[LuminosityWeight] = None):
        """Add a process with its DataFrame and weight."""
        # Create WeightedDataFrame
        weighted_df = WeightedDataFrame(dataframe, weight, self._histogram_engine)
        
        # Store in parent dictionary
        self[name] = weighted_df
        
        # Store weight separately for quick access
        if weight:
            self._weights[name] = weight
        
        # Auto-classify into groups
        self._classify_process(name)
        
        # Track in 'all' group
        if name not in self._groups['all']:
            self._groups['all'].append(name)
    
    def get_weights(self) -> Dict[str, float]:
        """Get all weights as a simple dictionary."""
        return {name: info.weight for name, info in self._weights.items()}
    
    def hist(self, column: str, **kwargs) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute histograms for all processes with automatic weight application.
        
        This overrides the parent method to ensure weights are properly applied.
        """
        apply_weights = kwargs.pop('apply_weights', True)
        results = {}
        
        for name, weighted_df in self.items():
            try:
                if isinstance(weighted_df, WeightedDataFrame):
                    results[name] = weighted_df.hist(column, apply_weight=apply_weights, **kwargs)
                else:
                    # Fallback for non-weighted DataFrames
                    results[name] = weighted_df.hist(column, **kwargs)
            except Exception as e:
                warnings.warn(f"Histogram computation failed for {name}: {e}")
                results[name] = None
        
        return results
    
    def query(self, expr: str) -> 'WeightedBroadcastResult':
        """Query all processes with weight preservation."""
        results = {}
        errors = []
        
        for name, weighted_df in self.items():
            try:
                if isinstance(weighted_df, WeightedDataFrame):
                    results[name] = weighted_df.query(expr)
                else:
                    results[name] = weighted_df.query(expr)
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                results[name] = None
        
        # Return enhanced broadcast result
        br = WeightedBroadcastResult(results, f"query('{expr}')", self)
        br._errors = errors
        return br
    
    def oneCandOnly(self, *args, **kwargs) -> 'WeightedBroadcastResult':
        """Select best candidates with weight preservation."""
        results = {}
        errors = []
        
        for name, weighted_df in self.items():
            try:
                if isinstance(weighted_df, WeightedDataFrame):
                    results[name] = weighted_df.oneCandOnly(*args, **kwargs)
                else:
                    results[name] = weighted_df.oneCandOnly(*args, **kwargs)
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                results[name] = None
        
        br = WeightedBroadcastResult(results, "oneCandOnly", self)
        br._errors = errors
        return br

class WeightedBroadcastResult(BroadcastResult):
    """Enhanced broadcast result that preserves weight information."""
    
    def hist(self, column: str, **kwargs):
        """Histogram computation with proper weight handling."""
        apply_weights = kwargs.pop('apply_weights', True)
        hist_results = {}
        successful = 0
        
        for name, weighted_df in self._valid_results.items():
            try:
                if isinstance(weighted_df, WeightedDataFrame):
                    result = weighted_df.hist(column, apply_weight=apply_weights, **kwargs)
                else:
                    result = weighted_df.hist(column, **kwargs)
                hist_results[name] = result
                successful += 1
                print(f"   ✅ {name}: Histogram computed")
            except Exception as e:
                error_msg = f"{name}: {str(e)}"
                self._errors.append(error_msg)
                hist_results[name] = None
                print(f"   ❌ {name}: {str(e)}")
        
        print(f"📈 Histogram completed: {successful}/{len(self._valid_results)} successful")
        return hist_results
    
    def to_dict(self) -> 'WeightedProcessDict':
        """Convert to WeightedProcessDict."""
        source_dict = self.source() or self._source_dict
        
        result = WeightedProcessDict(memory_budget_gb=source_dict.memory_budget_gb)
        result._histogram_engine = getattr(source_dict, '_histogram_engine', None)
        result._luminosity_manager = getattr(source_dict, '_luminosity_manager', None)
        result._energy_condition = getattr(source_dict, '_energy_condition', '5S_scan')
        
        for name, df in self._valid_results.items():
            if isinstance(df, WeightedDataFrame):
                result.add_process_with_weight(name, df.df, df.weight_info)
            else:
                result.add_process(name, df)
        
        return result

# ============================================================================
# UNIFIED HISTOGRAM PIPELINE
# ============================================================================

class HistogramPipeline:
    """
    Unified histogram computation pipeline with consistent C++ acceleration.
    
    This ensures all histogram operations go through the same optimized path.
    """
    
    def __init__(self, histogram_engine: Optional[Any] = None):
        self._engine = histogram_engine or self._create_default_engine()
        self._performance_stats = defaultdict(list)
    
    def _create_default_engine(self):
        """Create default histogram engine with fallback."""
        if CPP_HISTOGRAM_AVAILABLE:
            try:
                return OptimizedStreamingHistogram()
            except Exception as e:
                warnings.warn(f"Failed to create C++ histogram engine: {e}")
        return None
    
    def compute(self,
                data_source: Union[UnifiedLazyDataFrame, List[pl.LazyFrame]],
                column: str,
                bins: int = 50,
                range: Optional[Tuple[float, float]] = None,
                density: bool = False,
                weight: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram through unified pipeline.
        
        Args:
            data_source: DataFrame or lazy frames
            column: Column to histogram
            bins: Number of bins
            range: Histogram range
            density: Normalize to density
            weight: Optional weight to apply
            
        Returns:
            (counts, edges) tuple
        """
        start_time = time.time()
        
        # Extract lazy frames if needed
        if isinstance(data_source, UnifiedLazyDataFrame):
            lazy_frames = data_source._lazy_frames or []
        else:
            lazy_frames = data_source
        
        # Try C++ acceleration first
        if self._engine and lazy_frames:
            try:
                counts, edges = self._engine.compute_blazing_fast(
                    lazy_frames, column, bins=bins, range=range, density=density
                )
                
                # Apply weight if provided
                if weight is not None and weight != 1.0:
                    counts = counts * weight
                
                self._record_performance('cpp', time.time() - start_time)
                return counts, edges
                
            except Exception as e:
                warnings.warn(f"C++ histogram failed, falling back: {e}")
        
        # Fallback to Polars
        return self._polars_fallback(lazy_frames, column, bins, range, density, weight)
    
    def _polars_fallback(self, lazy_frames, column, bins, range, density, weight):
        """Polars-based histogram computation."""
        if range is None:
            range = self._compute_range(lazy_frames, column)
        
        bin_edges = np.linspace(range[0], range[1], bins + 1)
        accumulator = np.zeros(bins, dtype=np.float64)
        
        for lf in lazy_frames:
            try:
                df = lf.select([column]).collect(streaming=True)
                values = df[column].to_numpy()
                
                # Filter NaN values
                valid_values = values[np.isfinite(values)]
                
                if len(valid_values) > 0:
                    counts, _ = np.histogram(valid_values, bins=bin_edges)
                    accumulator += counts
            except Exception as e:
                warnings.warn(f"Frame processing failed: {e}")
        
        # Apply weight
        if weight is not None and weight != 1.0:
            accumulator = accumulator * weight
        
        # Apply density normalization
        if density and np.sum(accumulator) > 0:
            bin_width = bin_edges[1] - bin_edges[0]
            accumulator = accumulator / (np.sum(accumulator) * bin_width)
        
        self._record_performance('polars', time.time() - start_time)
        return accumulator, bin_edges
    
    def _compute_range(self, lazy_frames, column):
        """Compute histogram range from data."""
        # Physics-aware defaults
        physics_ranges = {
            'M_bc': (5.20, 5.30), 'Mbc': (5.20, 5.30),
            'delta_E': (-0.30, 0.30), 'deltaE': (-0.30, 0.30),
            'pRecoil': (0.1, 6.0),
            'mu1P': (0.0, 3.0), 'mu2P': (0.0, 3.0),
            'mu1Pt': (0.0, 3.0), 'mu2Pt': (0.0, 3.0),
        }
        
        if column in physics_ranges:
            return physics_ranges[column]
        
        # Pattern-based inference
        if column.endswith(('P', 'Pt', 'Energy', 'E')):
            return (0.0, 5.0)
        elif 'theta' in column.lower():
            return (0.0, np.pi)
        elif 'phi' in column.lower():
            return (-np.pi, np.pi)
        
        # Sample-based range
        try:
            sample = lazy_frames[0].select([column]).head(10000).collect()
            values = sample[column].to_numpy()
            finite_values = values[np.isfinite(values)]
            
            if len(finite_values) > 10:
                min_val, max_val = np.percentile(finite_values, [1, 99])
                margin = (max_val - min_val) * 0.1
                return (float(min_val - margin), float(max_val + margin))
        except Exception:
            pass
        
        return (0.0, 10.0)
    
    def _record_performance(self, method: str, elapsed: float):
        """Record performance statistics."""
        self._performance_stats[method].append(elapsed)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics."""
        report = {}
        for method, times in self._performance_stats.items():
            if times:
                report[method] = {
                    'count': len(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'total': sum(times)
                }
        return report

# ============================================================================
# ENHANCED LUMINOSITY MANAGER
# ============================================================================

class EnhancedLuminosityManagerV3:
    """
    Refined luminosity manager that produces weight objects instead of raw values.
    
    This provides better integration with the WeightedDataFrame system.
    """
    
    def __init__(self):
        # Reuse the luminosity data from the original implementation
        self.lumi_proc = {
            'data_proc': {
                '4S_on': 357.30651809174,
                '4S_offres': 41.64267813217,
                '5S_scan': 19.63477002990
            },
            'uubar': {'4S_on': 1368.4666, '4S_offres': 158.4068, '5S_scan': 78.5391},
            'ddbar': {'4S_on': 1429.2255, '4S_offres': 168.4762, '5S_scan': 78.5391},
            'ssbar': {'4S_on': 1429.2255, '4S_offres': 168.4762, '5S_scan': 78.5391},
            'ccbar': {'4S_on': 1368.4666, '4S_offres': 158.4068, '5S_scan': 78.5391},
            'taupair': {'4S_on': 1368.4666, '4S_offres': 158.4068, '5S_scan': 78.5391},
            'mumu': {'4S_on': 1368.4666, '4S_offres': 158.4068, '5S_scan': 78.5391},
            'gg': {'4S_on': 684.2325, '4S_offres': 79.2034, '5S_scan': 39.2695},
            'ee': {'4S_on': 34.2116, '4S_offres': 3.9602, '5S_scan': 1.9635},
            'eeee': {'4S_on': 341.6712, '4S_offres': 38.7110, '5S_scan': 19.6348},
            'eemumu': {'4S_on': 342.1162, '4S_offres': 38.7110, '5S_scan': 19.6348},
            'llXX': {'4S_on': 874.7212, '4S_offres': 114.0483, '5S_scan': 19.6348},
            'hhISR': {'4S_on': 357.3065, '4S_offres': 42.1190, '5S_scan': 19.6348}
        }
        
        self.lumi_prompt = {
            'data_prompt': {
                '4S_on': 129.52826011600,
                '4S_offres': 17.67914933099,
                '5S_scan': 0.0
            },
            # MC values same as proc
            'uubar': {'4S_on': 1368.4666, '4S_offres': 158.4068, '5S_scan': 78.5391},
            'ddbar': {'4S_on': 1429.2255, '4S_offres': 168.4762, '5S_scan': 78.5391},
            'ssbar': {'4S_on': 1429.2255, '4S_offres': 168.4762, '5S_scan': 78.5391},
            'ccbar': {'4S_on': 1368.4666, '4S_offres': 158.4068, '5S_scan': 78.5391},
            'taupair': {'4S_on': 1368.4666, '4S_offres': 158.4068, '5S_scan': 78.5391},
            'mumu': {'4S_on': 1368.4666, '4S_offres': 158.4068, '5S_scan': 78.5391},
            'gg': {'4S_on': 684.2325, '4S_offres': 79.2034, '5S_scan': 39.2695},
            'ee': {'4S_on': 34.2116, '4S_offres': 3.9602, '5S_scan': 1.9635},
            'eeee': {'4S_on': 341.6712, '4S_offres': 38.7110, '5S_scan': 19.6348},
            'eemumu': {'4S_on': 342.1162, '4S_offres': 38.7110, '5S_scan': 19.6348},
            'llXX': {'4S_on': 874.7212, '4S_offres': 114.0483, '5S_scan': 19.6348},
            'hhISR': {'4S_on': 357.3065, '4S_offres': 42.1190, '5S_scan': 19.6348}
        }
        
        # Pattern matching from original
        self.process_classification_patterns = {
            'mumu': ['mumu', 'mu+mu-', 'muon'],
            'ee': ['ee', 'e+e-', 'electron', 'bhabha'],
            'taupair': ['taupair', 'tau', 'tautau'],
            'uubar': ['uubar', 'uu'], 
            'ddbar': ['ddbar', 'dd'], 
            'ssbar': ['ssbar', 'ss'],
            'ccbar': ['ccbar', 'cc'], 
            'gg': ['gg', 'gamma', 'photon'],
            'hhISR': ['hhisr', 'hh_isr', 'hadron_isr', 'hh'],
            'eeee': ['eeee', '4e'], 
            'eemumu': ['eemumu', '2e2mu'],
            'llXX': ['llxx', 'llyy', 'four_lepton'],
            'charged': ['charged', 'b+b-', 'bplus'], 
            'mixed': ['mixed', 'b0b0bar', 'bneutral']
        }
    
    def calculate_weight(self, 
                        process_name: str,
                        energy_condition: str) -> LuminosityWeight:
        """
        Calculate luminosity weight for a single process.
        
        Returns a LuminosityWeight object instead of just a float.
        """
        # Determine if data or MC
        is_data = self._is_data_process(process_name)
        
        if is_data:
            return LuminosityWeight(
                process_name=process_name,
                weight=1.0,
                energy_condition=energy_condition,
                calculation_method='data_unity',
                is_data=True
            )
        
        # Determine proc/prompt
        is_proc = self._is_proc_process(process_name)
        
        # Get appropriate luminosity values
        data_lumi = self._get_data_luminosity(energy_condition, is_proc)
        process_type = self._extract_process_type(process_name)
        
        lumi_db = self.lumi_proc if is_proc else self.lumi_prompt
        mc_lumi = lumi_db.get(process_type, {}).get(energy_condition, None)
        
        # Fallback logic
        if mc_lumi is None or mc_lumi <= 0:
            # Try the other database as fallback
            alt_db = self.lumi_prompt if is_proc else self.lumi_proc
            mc_lumi = alt_db.get(process_type, {}).get(energy_condition, 1000.0)
            method = 'cross_database_fallback'
        else:
            method = f'{"proc" if is_proc else "prompt"}_calculation'
        
        # Calculate weight
        if mc_lumi > 0 and np.isfinite(mc_lumi):
            weight = data_lumi / mc_lumi
        else:
            weight = 1.0
            method = 'fallback_unity'
        
        # Sanity check
        if not np.isfinite(weight) or weight <= 0 or weight > 1000:
            weight = 1.0
            method = 'sanity_fallback'
        
        return LuminosityWeight(
            process_name=process_name,
            weight=weight,
            energy_condition=energy_condition,
            calculation_method=method,
            is_data=False,
            mc_luminosity=mc_lumi,
            data_luminosity=data_lumi
        )
    
    def _is_data_process(self, process_name: str) -> bool:
        """Check if process is data."""
        return 'data' in process_name.lower()
    
    def _is_proc_process(self, process_name: str) -> bool:
        return 'prompt' not in process_name
    
    def _get_data_luminosity(self, energy_condition: str, is_proc: bool) -> float:
        """Get appropriate data luminosity."""
        if is_proc:
            return self.lumi_proc.get('data_proc', {}).get(energy_condition, 19.6348)
        else:
            prompt_lumi = self.lumi_prompt.get('data_prompt', {}).get(energy_condition, 0.0)
            return prompt_lumi if prompt_lumi > 0 else self.lumi_proc.get('data_proc', {}).get(energy_condition, 19.6348)
    
    def _extract_process_type(self, process_name: str) -> str:
        """Extract process type from name."""
        name_lower = process_name.lower()
        
        for process_type, patterns in self.process_classification_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return process_type
        
        # Heuristic fallbacks
        fallback_patterns = [
            (['continuum', 'cont'], 'uubar'),
            (['quark', 'qqbar', 'hadron'], 'ccbar'),
            (['lepton', 'lep'], 'mumu'),
            (['photon', 'gamma'], 'gg'),
            (['tau'], 'taupair'),
            (['electron'], 'ee')
        ]
        
        for patterns, process_type in fallback_patterns:
            if any(pattern in name_lower for pattern in patterns):
                return process_type
        
        return 'default'

# ============================================================================
# ENHANCED PARTICLE DATA LOADER
# ============================================================================

class ParticleDataLoaderV3:
    """
    Enhanced loader with proper energy filtering and method injection.
    """
    
    def __init__(self, particle_type: str = 'vpho', config=None):
        self.particle_type = particle_type
        self.config = config
        self.memory_budget_gb = config.memory_budget_gb if config else 8.0
        self._histogram_engine = None
        
        # Default columns by particle type
        self.default_columns = {
            'vpho': [
                '__experiment__', '__run__', '__event__', '__production__', 
                '__candidate__', '__ncandidates__',
                'pRecoilTheta', 'pRecoilPhi', 'eRecoil', 'pRecoil', 'mRecoil', 'm2Recoil',
                'mu1clusterE', 'mu2clusterE', 'mu1clusterEoP', 'mu2clusterEoP',
                'mu1clusterPhi', 'mu2clusterPhi', 'mu1clusterTheta', 'mu2clusterTheta',
                'mu1nCDCHits', 'mu2nCDCHits', 'nGammaROE', 'nTracksROE',
                'nPhotonCands', 'nTracks', 'sumE_offcone', 'sumE_offcone_barrel',
                'totalMuonMomentum', 'theta', 'phi', 'E', 'beamE',
                'vpho_px', 'vpho_py', 'vpho_pz', 'vpho_E', 'psnm_ffo',
                'mu1Theta', 'mu2Theta', 'mu1Phi', 'mu2Phi', 'mu1E', 'mu2E', 'mu1P', 'mu2P'
            ],
            'gamma': [
                '__experiment__', '__run__', '__event__', '__production__', 
                '__candidate__', '__ncandidates__',
                '__eventType__', '__weight__', 'mcMatchWeight', 'mcPDG', 
                'theta', 'phi', 'E',
                'useCMSFrame__bophi__bc', 'useCMSFrame__botheta__bc', 'useCMSFrame__boE__bc',
                'nGammaROE', 'minC2TDist', 'clusterTiming', 'clusterErrorTiming'
            ],
            'photon': None,  # Alias for gamma
            'electron': None,  # Use all columns
            'muon': None,      # Use all columns
        }
    
    def load(self, base_dir: str, 
             pattern: Optional[str] = None,
             columns: Optional[List[str]] = None,
             sample_fraction: Optional[float] = None) -> WeightedProcessDict:
        """
        Load particle data with enhanced functionality.
        """
        print(f"🔍 Loading {self.particle_type} data from {base_dir}")
        
        # Convert to Path
        base_path = Path(base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
        
        # Determine columns
        if columns is None:
            columns = self.default_columns.get(self.particle_type)
            if columns:
                print(f"   Using {len(columns)} default columns for {self.particle_type}")
        
        # Discover process directories
        process_directories = self._discover_process_directories(base_path, pattern)
        
        if not process_directories:
            raise FileNotFoundError(f"No process directories found in {base_dir}")
        
        print(f"   📊 Found {len(process_directories)} process directories")
        
        # Apply energy filtering
        if self.config and hasattr(self.config, 'energy_condition'):
            filtered_directories = self._apply_energy_filtering(process_directories)
            if not filtered_directories:
                raise ValueError(f"No directories passed energy condition filtering")
        else:
            filtered_directories = process_directories
        
        # Load directories
        processes = self._load_directories_as_processes(
            filtered_directories, columns, sample_fraction
        )
        
        # Enhance with particle-specific methods
        self._enhance_processes(processes)
        
        print(f"✅ Successfully loaded {len(processes)} processes")
        
        return processes
    
    def _discover_process_directories(self, base_path: Path, 
                                    pattern: Optional[str]) -> Dict[str, List[Path]]:
        """Discover process directories matching pattern."""
        process_directories = {}
        directory_pattern = pattern or '*'
        
        for entry in base_path.iterdir():
            if entry.is_dir() and entry.match(directory_pattern):
                # Look for parquet files
                parquet_files = list(entry.glob(f'*{self.particle_type}*.parquet'))
                
                # Filter out temp files
                valid_files = [f for f in parquet_files if 'tmp' not in f.name]
                
                if valid_files:
                    process_directories[entry.name] = valid_files
                    print(f"   📁 Found process: {entry.name} ({len(valid_files)} files)")
        
        return process_directories
    
    def _apply_energy_filtering(self, process_directories: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
        """Apply energy condition filtering."""
        energy_patterns = {
            'vpho': {
                '5S_scan': {
                    'include': ['5s', 'mc5s', 'data5s', 'scan5s', 'scan_5s', 'mc-5s', 'data-5s'],
                    'exclude': ['off', '4s', 'off_resonance', 'offres']
                },
                '4S_on': {
                    'include': ['4s', 'mc4s', 'data4s', 'on_resonance', 'onres', 'mc-4s', 'data-4s'],
                    'exclude': ['off', '5s', 'scan', 'off_resonance']
                },
                '4S_offres': {
                    'include': ['off', 'mcoff', 'dataoff', 'off_resonance', 'offres'],
                    'exclude': ['5s', 'on_resonance', 'onres', '4s_on']
                }
            },
            'gamma': {
                '5S_scan': {
                    'include': ['5s', 'scan'],
                    'exclude': ['off', '4s']
                },
                '4S_on': {
                    'include': ['4s', 'on'],
                    'exclude': ['off', '5s']
                },
                '4S_offres': {
                    'include': ['off'],
                    'exclude': ['5s', 'on']
                }
            }
        }
        
        # Get patterns for current particle and energy condition
        particle_patterns = energy_patterns.get(self.particle_type, energy_patterns['vpho'])
        energy_condition = self.config.energy_condition if self.config else '5S_scan'
        
        patterns = particle_patterns.get(energy_condition)
        if not patterns:
            warnings.warn(f"No filtering patterns for {energy_condition}, returning all")
            return process_directories
        
        filtered = {}
        
        print(f"\n   🔍 Applying {energy_condition} energy filtering")
        print(f"      Include patterns: {patterns['include']}")
        print(f"      Exclude patterns: {patterns['exclude']}")
        
        for dir_name, files in process_directories.items():
            name_lower = dir_name.lower()
            
            # Check inclusion
            include_match = any(pattern in name_lower for pattern in patterns['include'])
            
            # Check exclusion
            exclude_match = any(pattern in name_lower for pattern in patterns['exclude'])
            
            if include_match and not exclude_match:
                filtered[dir_name] = files
                print(f"      ✓ Accepted: {dir_name}")
            else:
                reason = "excluded" if exclude_match else "not included"
                print(f"      ✗ Filtered out: {dir_name} ({reason})")
        
        print(f"\n   📊 Filtering result: {len(filtered)}/{len(process_directories)} directories kept")
        
        return filtered
    
    def _load_directories_as_processes(self, 
                                     process_directories: Dict[str, List[Path]], 
                                     columns: Optional[List[str]],
                                     sample_fraction: Optional[float]) -> WeightedProcessDict:
        """Load each directory as a process."""
        from layer2_unified_lazy_dataframe import UnifiedLazyDataFrame
        
        # Create luminosity manager
        luminosity_manager = EnhancedLuminosityManagerV3()
        
        # Create result dictionary
        processes = WeightedProcessDict(
            memory_budget_gb=self.memory_budget_gb,
            energy_condition=self.config.energy_condition if self.config else '5S_scan'
        )
        processes._luminosity_manager = luminosity_manager
        processes._histogram_engine = self._histogram_engine
        
        for process_name, file_list in process_directories.items():
            try:
                print(f"\n   Loading {process_name}...")
                
                # Create lazy frames
                lazy_frames = []
                for file_path in sorted(file_list):
                    try:
                        lf = pl.scan_parquet(str(file_path))
                        
                        # Apply column selection
                        if columns:
                            available_cols = list(lf.collect_schema().keys())
                            valid_cols = [col for col in columns if col in available_cols]
                            if valid_cols:
                                lf = lf.select(valid_cols)
                        
                        # Apply sampling
                        if sample_fraction:
                            lf = lf.filter(pl.col('__event__').hash() % 100 < sample_fraction * 100)
                        
                        lazy_frames.append(lf)
                        
                    except Exception as e:
                        warnings.warn(f"Skipping {file_path.name}: {e}")
                
                if lazy_frames:
                    # Create DataFrame
                    df = UnifiedLazyDataFrame(
                        lazy_frames=lazy_frames,
                        memory_budget_gb=self.memory_budget_gb / len(process_directories),
                        required_columns=columns
                    )
                    
                    # Calculate weight
                    weight = luminosity_manager.calculate_weight(
                        process_name,
                        processes._energy_condition
                    )
                    
                    # Add to processes
                    processes.add_process_with_weight(process_name, df, weight)
                    print(f"      ✅ {process_name}: Loaded with weight={weight.weight:.4f}")
                    
            except Exception as e:
                warnings.warn(f"Failed to load {process_name}: {e}")
        
        return processes
    
    def _enhance_processes(self, processes: WeightedProcessDict):
        """Enhance processes with particle-specific methods."""
        # Method injectors by particle type
        injectors = {
            'gamma': self._inject_photon_methods,
            'photon': self._inject_photon_methods,
            'electron': self._inject_electron_methods,
            'muon': self._inject_muon_methods,
            'vpho': self._inject_vpho_methods,
        }
        
        injector = injectors.get(self.particle_type)
        if injector:
            for name, weighted_df in processes.items():
                if isinstance(weighted_df, WeightedDataFrame):
                    df_class = weighted_df._dataframe.__class__
                    injector(df_class)
    
    def _inject_photon_methods(self, dataframe_class):
        """Inject photon-specific analysis methods."""
        
        def angular_separation(self) -> pl.Expr:
            """Compute angular separation between recoil and photon."""
            required = ['pRecoil', 'pRecoilTheta', 'pRecoilPhi', 'E', 'theta', 'phi']
            
            # Polars expression for angular separation
            p_dot = (
                pl.col('pRecoil') * pl.col('pRecoilTheta').sin() * pl.col('pRecoilPhi').cos() *
                pl.col('E') * pl.col('theta').sin() * pl.col('phi').cos() +
                pl.col('pRecoil') * pl.col('pRecoilTheta').sin() * pl.col('pRecoilPhi').sin() *
                pl.col('E') * pl.col('theta').sin() * pl.col('phi').sin() +
                pl.col('pRecoil') * pl.col('pRecoilTheta').cos() *
                pl.col('E') * pl.col('theta').cos()
            )
            return (p_dot / (pl.col('pRecoil') * pl.col('E'))).arccos()
        
        def select_best_photon(self, nan_fill: float = 4.0) -> 'UnifiedLazyDataFrame':
            """Select best photon per event."""
            # Add angular separation
            df_with_angular = self.with_columns(
                angular_separation(self).alias('angular_separation')
            )
            
            # Fill NaN and select minimum
            df_filled = df_with_angular.with_columns(
                pl.col('angular_separation').fill_nan(nan_fill)
            )
            
            # Group and select
            group_cols = ['__experiment__', '__run__', '__event__', '__production__']
            df_best = df_filled.sort('angular_separation').group_by(group_cols).first()
            
            # Add E/pRecoil and apply energy cut
            return df_best.with_columns(
                (pl.col('E') / pl.col('pRecoil')).fill_nan(0).alias('EoPRecoil')
            ).filter((pl.col('E') > 0.075) | pl.col('E').is_null())
        
        # Add methods
        dataframe_class.angular_separation = angular_separation
        dataframe_class.select_best_photon = select_best_photon
    
    def _inject_electron_methods(self, dataframe_class):
        """Inject electron-specific methods."""
        
        def electron_id_score(self, weights: Optional[Dict[str, float]] = None) -> pl.Expr:
            """Compute electron ID score."""
            if weights is None:
                weights = {
                    'E_over_p': 0.3,
                    'shower_shape': 0.2,
                    'track_match': 0.3,
                    'dE_dx': 0.2
                }
            
            # Build weighted score expression
            score_components = []
            for var, weight in weights.items():
                if var in self.columns:
                    score_components.append(pl.col(var) * weight)
            
            if not score_components:
                raise ValueError("No electron ID variables found")
            
            return sum(score_components).alias('electron_id_score')
        
        dataframe_class.electron_id_score = electron_id_score
    
    def _inject_muon_methods(self, dataframe_class):
        """Inject muon-specific methods."""
        
        def muon_id_score(self) -> pl.Expr:
            """Compute muon ID score."""
            score_components = []
            
            if 'klm_layers_hit' in self.columns:
                score_components.append(pl.col('klm_layers_hit') / 14.0 * 0.5)
            
            if 'mu1P' in self.columns and 'mu2P' in self.columns:
                score_components.append((pl.col('mu1P') + pl.col('mu2P')) / 20.0 * 0.5)
            elif 'momentum' in self.columns:
                score_components.append(pl.col('momentum') / 10.0 * 0.5)
            
            if not score_components:
                raise ValueError("No muon ID variables found")
            
            return sum(score_components).alias('muon_id_score')
        
        dataframe_class.muon_id_score = muon_id_score
    
    def _inject_vpho_methods(self, dataframe_class):
        """Inject vpho-specific methods (virtual photon)."""
        # Vpho inherits photon methods plus additional ones
        self._inject_photon_methods(dataframe_class)
        
        def compute_recoil_properties(self) -> 'UnifiedLazyDataFrame':
            """Compute additional recoil system properties."""
            return self.with_columns([
                (pl.col('pRecoil') * pl.col('pRecoilTheta').sin()).alias('pTRecoil'),
                pl.col('eRecoil').sqrt().alias('sqrtERecoil'),
                (pl.col('m2Recoil') / pl.col('mRecoil')).alias('m2_over_m_recoil')
            ])
        
        dataframe_class.compute_recoil_properties = compute_recoil_properties
    def load_flat_structure(self, base_dir: str,
                       data_type: str = 'selected',  # 'selected' or 'matched'
                       pattern: Optional[str] = None,
                       columns: Optional[List[str]] = None,
                       sample_fraction: Optional[float] = None) -> WeightedProcessDict:
        """
        Load particle data from flat directory structure.
        
        Args:
            base_dir: Directory containing all parquet files
            data_type: Either 'selected' or 'matched' to choose file type
            pattern: Optional pattern to filter files
            columns: Columns to load
            sample_fraction: Fraction of data to sample
        """
        print(f"🔍 Loading {data_type} data from flat structure in {base_dir}")
        
        base_path = Path(base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
        
        # Discover files and group by process
        process_files = self._discover_flat_structure_files(base_path, data_type, pattern)
        
        if not process_files:
            raise FileNotFoundError(f"No {data_type} files found in {base_dir}")
        
        print(f"   📊 Found {len(process_files)} processes")
        
        # Apply energy filtering if needed
        if self.config and hasattr(self.config, 'energy_condition'):
            filtered_files = self._apply_flat_structure_filtering(process_files)
            if not filtered_files:
                raise ValueError(f"No files passed energy condition filtering")
        else:
            filtered_files = process_files
        
        # Load files as processes
        processes = self._load_flat_files_as_processes(
            filtered_files, columns, sample_fraction
        )
        
        # Enhance with particle-specific methods
        self._enhance_processes(processes)
        
        print(f"✅ Successfully loaded {len(processes)} processes")
        
        return processes
    
    def _discover_flat_structure_files(self, base_path: Path, 
                                      data_type: str,
                                      pattern: Optional[str]) -> Dict[str, List[Path]]:
        """Discover and group files from flat structure."""
        process_files = defaultdict(list)
        
        # Get all parquet files ending with the specified type
        suffix = f"_{data_type}.parquet"
        
        for file_path in base_path.glob(f"*{suffix}"):
            if 'tmp' in file_path.name:
                continue
                
            # Parse process name from filename
            filename = file_path.stem  # Remove .parquet
            parts = filename.split('-')
            
            if parts[0] == 'data':
                # Data file: data-{resonance}_{type}
                process_name = f"data-{parts[1].replace(f'_{data_type}', '')}"
            elif parts[0] == 'mc16' and len(parts) >= 3:
                # MC file: mc16-{process}-{resonance}_{type}
                process_name = parts[1]
                resonance = parts[2].replace(f'_{data_type}', '')
                process_name = f"{process_name}-{resonance}"
            else:
                # Skip unrecognized format
                continue
            
            # Apply pattern filter if specified
            if pattern and pattern not in process_name:
                continue
                
            process_files[process_name].append(file_path)
            
        # Report discovered processes
        for process_name, files in sorted(process_files.items()):
            print(f"   📁 Found process: {process_name} ({len(files)} files)")
        
        return dict(process_files)
    
    def _apply_flat_structure_filtering(self, process_files: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
        e = self.config.energy_condition
        return {n:f for n,f in process_files.items() if
                (e=='4S_on' and not any(x in n for x in ['off','5s','scan'])) or
                (e=='4S_offres' and 'off' in n) or
                (e=='5S_scan' and any(x in n for x in ['5s','scan']))} or process_files       
    def _load_flat_files_as_processes(self,
                                     process_files: Dict[str, List[Path]],
                                     columns: Optional[List[str]],
                                     sample_fraction: Optional[float]) -> WeightedProcessDict:
        """Load flat structure files as processes."""
        from layer2_unified_lazy_dataframe import UnifiedLazyDataFrame
        
        # Create luminosity manager
        luminosity_manager = EnhancedLuminosityManagerV3()
        
        # Create result dictionary
        processes = WeightedProcessDict(
            memory_budget_gb=self.memory_budget_gb,
            energy_condition=self.config.energy_condition if self.config else '5S_scan'
        )
        processes._luminosity_manager = luminosity_manager
        processes._histogram_engine = self._histogram_engine
        
        for process_name, file_list in process_files.items():
            try:
                print(f"\n   Loading {process_name}...")
                
                # Create lazy frames
                lazy_frames = []
                for file_path in sorted(file_list):
                    try:
                        lf = pl.scan_parquet(str(file_path))
                        
                        # Apply column selection
                        if columns:
                            available_cols = list(lf.collect_schema().keys())
                            valid_cols = [col for col in columns if col in available_cols]
                            if valid_cols:
                                lf = lf.select(valid_cols)
                        
                        # Apply sampling
                        if sample_fraction:
                            lf = lf.filter(pl.col('__event__').hash() % 100 < sample_fraction * 100)
                        
                        lazy_frames.append(lf)
                        
                    except Exception as e:
                        warnings.warn(f"Skipping {file_path.name}: {e}")
                
                if lazy_frames:
                    # Create DataFrame
                    df = UnifiedLazyDataFrame(
                        lazy_frames=lazy_frames,
                        memory_budget_gb=self.memory_budget_gb / len(process_files),
                        required_columns=columns
                    )
                    
                    # Extract base process name for weight calculation
                    # Remove resonance info to get base process (e.g., "mumu-prompt" -> "mumu")
                    base_process = process_name.split('-')[0]
                    if base_process == 'data':
                        weight_process_name = process_name
                    else:
                        weight_process_name = base_process
                    
                    # Calculate weight
                    weight = luminosity_manager.calculate_weight(
                        weight_process_name,
                        processes._energy_condition
                    )
                    
                    # Add to processes
                    processes.add_process_with_weight(process_name, df, weight)
                    print(f"      ✅ {process_name}: Loaded with weight={weight.weight:.4f}")
                    
            except Exception as e:
                warnings.warn(f"Failed to load {process_name}: {e}")
        
        return processes

# ============================================================================
# PROGRESSIVE ANALYSIS WITH CLEAN WEIGHT INTEGRATION
# ============================================================================

class ProgressiveAnalysisV3:
    """
    Refined progressive analysis that cleanly separates data and weights.
    """
    
    def __init__(self, 
                 base_data: WeightedProcessDict,
                 config: Any,
                 histogram_pipeline: Optional[HistogramPipeline] = None):
        self.base_data = base_data
        self.config = config
        self.histogram_pipeline = histogram_pipeline or HistogramPipeline()
        self._stage_cache = {}
    
    def run_analysis(self, variable: str, 
                    bins: int = 100,
                    range: Optional[Tuple[float, float]] = None,
                    cuts: Optional[List[str]] = None,
                    stages: Optional[List[str]] = None,
                    output_dir: str = './analysis') -> Dict[str, Any]:
        """Run progressive analysis with clean weight handling."""
        print(f"\n🎯 PROGRESSIVE ANALYSIS: {variable}")
        print("=" * 60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if stages is None:
            stages = self._select_optimal_stages(variable, cuts)
        
        results = {
            'variable': variable,
            'configuration': {
                'bins': bins,
                'range': range,
                'cuts': cuts,
                'stages': stages,
                'energy_condition': self.config.energy_condition,
                'luminosity_weighting': self.config.apply_luminosity_weights
            },
            'histograms': {},
            'plots': {},
            'statistics': {}
        }
        
        for stage in stages:
            self._execute_stage(stage, variable, bins, range, cuts, output_path, results)
        
        # Generate efficiency plot if applicable
        if 'efficiency' in stages:
            self._generate_efficiency(results, variable, output_path)
        
        return results
    
    def _select_optimal_stages(self, variable: str, cuts: Optional[List[str]]) -> List[str]:
        """Select stages based on data and variable."""
        stages = ['baseline', 'candidates']
        
        if cuts:
            stages.append('cuts')
        
        # Check for photon capability
        if self._has_photon_data() and variable in ['EoPRecoil', 'angular_separation']:
            stages.append('photon')
            stages.append('efficiency')
        
        return stages
    
    def _has_photon_data(self) -> bool:
        """Check if photon variables are available."""
        if not self.base_data:
            return False
        
        # Check first process
        for name, weighted_df in self.base_data.items():
            if hasattr(weighted_df.df, 'columns'):
                required = ['E', 'theta', 'phi', 'pRecoilTheta', 'pRecoilPhi']
                return all(col in weighted_df.df.columns for col in required)
        
        return False
    
    def _execute_stage(self, stage: str, variable: str, bins: int,
                      range: Optional[Tuple[float, float]], cuts: List[str],
                      output_path: Path, results: Dict):
        """Execute analysis stage with proper weight handling."""
        print(f"\n📊 STAGE: {stage}")
        print("-" * 30)
        
        # Get stage data
        stage_data = self._get_stage_data(stage, cuts)
        
        # Validate columns
        self._validate_stage_columns(stage_data, variable)
        
        # Compute histograms
        hist_results = {}
        
        for name, weighted_df in stage_data.items():
            try:
                counts, edges = weighted_df.hist(
                    variable, 
                    bins=bins, 
                    range=range,
                    apply_weight=self.config.apply_luminosity_weights
                )
                hist_results[name] = (counts, edges)
                
                # Show weight info
                weight_str = f"(w={weighted_df.weight:.4f})" if weighted_df.weight != 1.0 else ""
                print(f"   ✓ {name}: Histogram computed {weight_str}")
                
            except Exception as e:
                print(f"   ❌ {name}: {str(e)}")
        
        results['histograms'][stage] = hist_results
        results['statistics'][stage] = self._compute_statistics(hist_results)
        
        # Create plot
        plot_path = self._create_stage_plot(hist_results, stage, variable, output_path)
        results['plots'][stage] = plot_path
    
    def _get_stage_data(self, stage: str, cuts: Optional[List[str]]) -> Dict[str, WeightedDataFrame]:
        """Get data for specific stage."""
        if stage == 'baseline':
            # Return all processes as-is
            return dict(self.base_data.items())
        
        elif stage == 'candidates':
            # Apply oneCandOnly to each process
            result = {}
            for name, weighted_df in self.base_data.items():
                result[name] = weighted_df.oneCandOnly()
            return result
        
        elif stage == 'cuts' and cuts:
            # Apply cuts sequentially
            result = {}
            for name, weighted_df in self.base_data.items():
                df = weighted_df
                for cut in cuts:
                    df = df.query(cut)
                result[name] = df
            return result
        
        elif stage == 'photon':
            # Apply photon selection
            base_data = self._get_stage_data('cuts', cuts) if cuts else dict(self.base_data.items())
            result = {}
            
            for name, weighted_df in base_data.items():
                if hasattr(weighted_df.df, 'select_best_photon'):
                    result[name] = WeightedDataFrame(
                        weighted_df.df.select_best_photon(),
                        weighted_df.weight_info,
                        weighted_df._histogram_engine
                    )
                else:
                    result[name] = weighted_df
            
            return result
        
        else:
            return dict(self.base_data.items())
    
    def _validate_stage_columns(self, data: Dict[str, WeightedDataFrame], variable: str):
        """Validate required columns exist."""
        if not data:
            return
        
        # Check first process
        first_name, first_wdf = next(iter(data.items()))
        
        if variable not in first_wdf.columns:
            from difflib import get_close_matches
            suggestions = get_close_matches(variable, first_wdf.columns, n=3, cutoff=0.6)
            
            raise KeyError(
                f"Variable '{variable}' not found in {first_name}.\n"
                f"Available columns: {first_wdf.columns[:20]}...\n"
                f"Did you mean: {suggestions}?"
            )
    
    def _create_stage_plot(self, histograms: Dict, stage: str, 
                          variable: str, output_path: Path) -> str:
        """Create publication-quality plot."""
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        
        ax_main = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
        
        # Group and plot histograms
        grouped = self._group_histograms_by_physics(histograms)
        mc_total, data_hist = self._plot_stacked_histograms(grouped, ax_main)
        
        # Ratio panel
        if data_hist is not None and mc_total is not None:
            self._create_ratio_panel(ax_ratio, data_hist, mc_total)
        
        # Styling
        self._apply_belle2_styling(ax_main, ax_ratio, variable, stage)
        
        # Save
        filename = f"belle2_{stage}_{variable}.pdf"
        filepath = output_path / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def _generate_efficiency(self, results: Dict, variable: str, output_path: Path):
        """Generate efficiency plot."""
        baseline = results['histograms'].get('baseline')
        selected = results['histograms'].get('photon', results['histograms'].get('cuts'))
        
        if not baseline or not selected:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Process by physics group
        for group in ['mumu', 'ee', 'qqbar', 'BBbar']:
            # Get processes in group
            group_processes = []
            if hasattr(self.base_data, '_groups') and group in self.base_data._groups:
                group_processes = self.base_data._groups[group]
            
            if not group_processes:
                continue
            
            # Aggregate histograms
            base_counts, edges = self._aggregate_group_histograms(baseline, group_processes)
            sel_counts, _ = self._aggregate_group_histograms(selected, group_processes)
            
            if base_counts is None or sel_counts is None:
                continue
            
            # Calculate efficiency with binomial errors
            eff, err_low, err_high = self._clopper_pearson_efficiency(sel_counts, base_counts)
            
            # Plot
            bin_centers = (edges[:-1] + edges[1:]) / 2
            color = self._get_group_color(group)
            
            ax.errorbar(bin_centers, eff, 
                       yerr=[eff - err_low, err_high - eff],
                       fmt='o-', color=color, 
                       label=self._get_physics_label(group),
                       markersize=6, linewidth=2, capsize=3)
        
        # Styling
        ax.set_xlabel(self._get_variable_label(variable), fontsize=14)
        ax.set_ylabel('Efficiency', fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        filepath = output_path / f"efficiency_{variable}.pdf"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        results['plots']['efficiency'] = str(filepath)
    
    def _group_histograms_by_physics(self, histograms: Dict) -> Dict:
        """Group histograms by physics process type."""
        grouped = {'mc': {}, 'data': None}
        
        # Process to group mapping
        process_to_group = {}
        if hasattr(self.base_data, '_groups'):
            for group, processes in self.base_data._groups.items():
                for process in processes:
                    process_to_group[process] = group
        
        for process, hist_data in histograms.items():
            if hist_data is None:
                continue
                
            counts, edges = hist_data
            
            if 'data' in process.lower():
                # Aggregate data
                if grouped['data'] is None:
                    grouped['data'] = (counts.copy(), edges, np.sqrt(counts))
                else:
                    grouped['data'] = (
                        grouped['data'][0] + counts,
                        edges,
                        np.sqrt(grouped['data'][0] + counts)
                    )
            else:
                # Determine physics group
                group = process_to_group.get(process, self._determine_physics_group(process))
                
                if group not in grouped['mc']:
                    grouped['mc'][group] = (counts.copy(), edges, np.sqrt(counts))
                else:
                    old_counts = grouped['mc'][group][0]
                    grouped['mc'][group] = (
                        old_counts + counts,
                        edges,
                        np.sqrt(old_counts + counts)
                    )
        
        return grouped
    
    def _determine_physics_group(self, process_name: str) -> str:
        """Determine physics group from process name."""
        name_lower = process_name.lower()
        
        group_patterns = {
            'mumu': ['mumu', 'muon'],
            'ee': ['ee', 'electron', 'bhabha'],
            'taupair': ['tau'],
            'qqbar': ['uubar', 'ddbar', 'ssbar', 'ccbar', 'qqbar', 'continuum'],
            'BBbar': ['bbbar', 'bb_', 'charged', 'mixed'],
            'gg': ['gg', 'gamma', 'photon'],
        }
        
        for group, patterns in group_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return group
        
        return 'qqbar'
    
    def _plot_stacked_histograms(self, grouped: Dict, ax):
        """Plot stacked MC histograms."""
        colors = {
            'mumu': '#E31A1C', 'ee': '#1F78B4', 'qqbar': '#33A02C',
            'BBbar': '#8C564B', 'taupair': '#6A3D9A', 'gg': '#FF7F00',
            'hhISR': '#666666', 'llYY': '#FFD92F'
        }
        
        mc_total = None
        mc_errors = None
        bottom = None
        
        # Stack MC components
        for group in ['BBbar', 'qqbar', 'taupair', 'ee', 'mumu', 'gg', 'hhISR', 'llYY']:
            if group not in grouped['mc']:
                continue
            
            counts, edges, errors = grouped['mc'][group]
            
            if mc_total is None:
                mc_total = counts.copy()
                mc_errors = errors.copy()
                bottom = np.zeros_like(counts)
            else:
                mc_total += counts
                mc_errors = np.sqrt(mc_errors**2 + errors**2)
            
            bin_centers = (edges[:-1] + edges[1:]) / 2
            ax.hist(bin_centers, weights=counts, bins=edges,
                   bottom=bottom, color=colors.get(group, '#888'),
                   alpha=0.9, edgecolor='black', linewidth=1,
                   label=self._get_physics_label(group), histtype='stepfilled')
            
            bottom += counts
        
        # Plot data
        data_hist = None
        if grouped.get('data'):
            data_counts, data_edges, data_errors = grouped['data']
            bin_centers = (data_edges[:-1] + data_edges[1:]) / 2
            
            ax.errorbar(bin_centers, data_counts, yerr=data_errors,
                       fmt='o', color='black', markersize=4,
                       label='Data', zorder=100)
            
            data_hist = (data_counts, data_edges, data_errors)
        
        return (mc_total, mc_errors) if mc_total is not None else None, data_hist
    
    def _create_ratio_panel(self, ax_ratio, data_hist, mc_total):
        """Create ratio panel with proper error propagation."""
        data_counts, edges, data_errors = data_hist
        mc_counts, mc_errors = mc_total
        
        bin_centers = (edges[:-1] + edges[1:]) / 2
        bin_width = np.mean(np.diff(edges))
        
        # Calculate ratio with proper error propagation
        ratio = np.ones_like(data_counts)
        ratio_err = np.zeros_like(data_counts)
        
        valid = (mc_counts > 0) & (data_counts >= 0)
        ratio[valid] = data_counts[valid] / mc_counts[valid]
        
        # Error propagation
        rel_data = np.zeros_like(data_counts)
        rel_mc = np.zeros_like(mc_counts)
        
        data_valid = (data_counts > 0) & valid
        rel_data[data_valid] = data_errors[data_valid] / data_counts[data_valid]
        rel_mc[valid] = mc_errors[valid] / mc_counts[valid]
        
        ratio_err[valid] = ratio[valid] * np.sqrt(rel_data[valid]**2 + rel_mc[valid]**2)
        
        # Gray deviation bars
        for i, (center, r) in enumerate(zip(bin_centers[valid], ratio[valid])):
            height = abs(r - 1)
            bottom = min(r, 1)
            ax_ratio.bar(center, height, width=bin_width * 0.8,
                        bottom=bottom, color='gray', alpha=0.5,
                        edgecolor='none', zorder=10)
        
        # MC uncertainty band
        ax_ratio.fill_between(bin_centers, 1 - rel_mc, 1 + rel_mc,
                            alpha=0.3, color='yellow', zorder=5,
                            label='MC uncertainty')
        
        # Data points
        ax_ratio.errorbar(bin_centers[valid], ratio[valid], yerr=ratio_err[valid],
                         fmt='o', color='black', markersize=4, zorder=100)
        
        # Unity line
        ax_ratio.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Smart y-range
        if np.any(valid):
            y_values = ratio[valid]
            y_median = np.median(y_values)
            y_std = np.std(y_values)
            ax_ratio.set_ylim(
                max(0.5, y_median - 2*y_std),
                min(2.0, y_median + 2*y_std)
            )
    
    def _apply_belle2_styling(self, ax_main, ax_ratio, variable: str, stage: str):
        """Apply Belle II styling."""
        ax_main.set_ylabel('Events / bin', fontsize=14)
        ax_main.set_yscale('log')
        ax_main.legend(loc='upper right', fontsize=10, ncol=2)
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(labelbottom=False)
        
        ax_main.text(0.05, 0.95, 'Belle II', transform=ax_main.transAxes,
                    fontsize=16, weight='bold', va='top')
        ax_main.text(0.05, 0.88, f'Stage: {stage}', transform=ax_main.transAxes,
                    fontsize=12, va='top')
        
        ax_ratio.set_xlabel(self._get_variable_label(variable), fontsize=14)
        ax_ratio.set_ylabel('Data/MC', fontsize=14)
        ax_ratio.grid(True, alpha=0.3)
    
    def _get_physics_label(self, group: str) -> str:
        """Get LaTeX label for physics group."""
        labels = {
            'mumu': r'$\mu^+\mu^-$', 
            'ee': r'$e^+e^-$',
            'qqbar': r'$q\bar{q}$',
            'BBbar': r'$B\bar{B}$',
            'taupair': r'$\tau^+\tau^-$',
            'gg': r'$\gamma\gamma$',
            'hhISR': r'$h^+h^-\gamma_{ISR}$',
            'llYY': r'$\ell^+\ell^-\gamma\gamma$'
        }
        return labels.get(group, group)
    
    def _get_variable_label(self, variable: str) -> str:
        """Get LaTeX label for variable."""
        labels = {
            'pRecoil': r'$p_{\mathrm{recoil}}$ [GeV/c]',
            'M_bc': r'$M_{\mathrm{bc}}$ [GeV/c$^2$]',
            'delta_E': r'$\Delta E$ [GeV]',
            'EoPRecoil': r'$E/p_{\mathrm{recoil}}$',
            'angular_separation': r'$\theta_{\gamma,\mathrm{recoil}}$ [rad]',
        }
        return labels.get(variable, variable)
    
    def _get_group_color(self, group: str) -> str:
        """Get consistent color for physics group."""
        colors = {
            'mumu': '#E31A1C', 'ee': '#1F78B4', 'qqbar': '#33A02C',
            'BBbar': '#8C564B', 'taupair': '#6A3D9A', 'gg': '#FF7F00',
            'hhISR': '#666666', 'llYY': '#FFD92F'
        }
        return colors.get(group, '#888888')
    
    def _compute_statistics(self, histograms: Dict) -> Dict:
        """Compute statistics from histograms."""
        total_events = sum(np.sum(h[0]) for h in histograms.values() if h is not None)
        n_processes = len(histograms)
        
        data_events = sum(np.sum(h[0]) for name, h in histograms.items() 
                         if h is not None and 'data' in name.lower())
        mc_events = total_events - data_events
        
        return {
            'total_events': total_events,
            'n_processes': n_processes,
            'data_events': data_events,
            'mc_events': mc_events,
            'avg_events_per_process': total_events / n_processes if n_processes > 0 else 0
        }
    
    def _aggregate_group_histograms(self, histograms: Dict, 
                                   processes: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Aggregate histograms for a group."""
        combined = None
        edges = None
        
        for process in processes:
            if process in histograms and histograms[process] is not None:
                counts, hist_edges = histograms[process]
                if combined is None:
                    combined = counts.copy()
                    edges = hist_edges
                else:
                    combined += counts
        
        return combined, edges
    
    def _clopper_pearson_efficiency(self, k: np.ndarray, n: np.ndarray, 
                                   confidence: float = 0.68) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Clopper-Pearson confidence intervals."""
        from scipy import stats
        
        eff = np.zeros_like(k, dtype=float)
        err_low = np.zeros_like(k, dtype=float)
        err_high = np.zeros_like(k, dtype=float)
        
        alpha = 1 - confidence
        
        for i in range(len(k)):
            if n[i] == 0:
                eff[i] = 0
                err_low[i] = 0
                err_high[i] = 0
            else:
                eff[i] = k[i] / n[i]
                err_low[i] = stats.beta.ppf(alpha/2, k[i], n[i] - k[i] + 1) if k[i] > 0 else 0
                err_high[i] = stats.beta.ppf(1 - alpha/2, k[i] + 1, n[i] - k[i]) if k[i] < n[i] else 1
        
        return eff, err_low, err_high

# ============================================================================
# MAIN PRODUCTION FRAMEWORK V3
# ============================================================================

class Belle2ProductionFrameworkV3(Belle2Layer2Framework):
    """
    Enhanced production framework with clean weight/data separation.
    
    This maintains full backward compatibility while solving the architectural issues.
    """
    
    def __init__(self, config: Optional[Any] = None):
        # Initialize parent with config
        if config:
            super().__init__(
                memory_budget_gb=config.memory_budget_gb,
                enable_cpp_acceleration=config.enable_cpp_acceleration,
                cache_dir=str(config.cache_dir) if hasattr(config, 'cache_dir') else None
            )
        else:
            super().__init__()
        
        self.config = config
        
        # Create unified histogram pipeline
        self.histogram_pipeline = HistogramPipeline(self._cpp_histogram)
        
        # Create luminosity manager
        self.luminosity_manager = EnhancedLuminosityManagerV3()
        
        print(f"""
        ╔════════════════════════════════════════════════════╗
        ║     Belle II Enhanced Production Framework v3      ║
        ║                                                    ║
        ║  ✓ Clean weight/data separation                   ║
        ║  ✓ Unified histogram pipeline                     ║
        ║  ✓ Consistent C++ acceleration                    ║
        ║  ✓ Full backward compatibility                    ║
        ║  ✓ Energy filtering & method injection            ║
        ╚════════════════════════════════════════════════════╝
        """)
    
    def load_particle_data(self, base_dir: str, 
                          particle: str = 'vpho',
                          columns: Optional[List[str]] = None,
                          sample_fraction: Optional[float] = None) -> WeightedProcessDict:
        """
        Load particle data with automatic weight calculation.
        
        Returns WeightedProcessDict with clean separation of data and weights.
        """
        print(f"🔍 Loading {particle} data with weight calculation")
        
        # Create enhanced particle loader
        loader = ParticleDataLoaderV3(particle, self.config)
        loader._histogram_engine = self.histogram_pipeline._engine
        
        # Load processes with all enhancements
        weighted_processes = loader.load(
            base_dir,
            columns=columns,
            sample_fraction=sample_fraction
        )
        
        # Ensure histogram engine is set
        weighted_processes._histogram_engine = self.histogram_pipeline._engine
        
        print(f"✅ Loaded {len(weighted_processes)} processes with weights")
        
        return weighted_processes
    
    def run_progressive_analysis(self, processes: WeightedProcessDict,
                               variable: str,
                               bins: int = 100,
                               range: Optional[Tuple[float, float]] = None,
                               cuts: Optional[List[str]] = None,
                               stages: Optional[List[str]] = None,
                               output_dir: str = './analysis') -> Dict[str, Any]:
        """Run progressive analysis with clean weight handling."""
        # Create analyzer
        analyzer = ProgressiveAnalysisV3(
            processes, 
            self.config,
            self.histogram_pipeline
        )
        
        # Run analysis
        results = analyzer.run_analysis(
            variable=variable,
            bins=bins,
            range=range,
            cuts=cuts,
            stages=stages,
            output_dir=output_dir
        )
        
        # Add performance data
        results['performance'] = {
            'histogram_pipeline': self.histogram_pipeline.get_performance_report(),
            'framework': self.profile_performance()
        }
        
        return results
    
    def quick_photon_analysis(self, data_path: str,
                            cuts: List[str],
                            variables: List[str] = ['EoPRecoil'],
                            output_dir: str = './photon_analysis') -> Dict[str, Any]:
        """Quick photon analysis with clean architecture."""
        # Load data
        processes = self.load_particle_data(data_path, particle='gamma')
        
        # Run analysis for each variable
        all_results = {}
        
        for variable in variables:
            results = self.run_progressive_analysis(
                processes,
                variable=variable,
                bins=100,
                range=(0, 2) if variable == 'EoPRecoil' else None,
                cuts=cuts,
                stages=['baseline', 'candidates', 'cuts', 'photon', 'efficiency'],
                output_dir=f"{output_dir}/{variable}"
            )
            all_results[variable] = results
        
        # Combined report
        all_results['combined_report'] = self._generate_combined_report(all_results)
        
        return all_results
    
    def _generate_combined_report(self, all_results: Dict[str, Dict]) -> str:
        """Generate combined analysis report."""
        report = "# Combined Photon Analysis Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary statistics
        total_events = 0
        total_time = 0
        
        for variable, results in all_results.items():
            if isinstance(results, dict) and 'statistics' in results:
                for stage, stats in results['statistics'].items():
                    total_events += stats.get('total_events', 0)
        
        report += f"## Summary\n"
        report += f"- Total Events Processed: {total_events:,}\n"
        report += f"- Variables Analyzed: {', '.join(v for v in all_results.keys() if v != 'combined_report')}\n\n"
        
        # Individual variable summaries
        for variable, results in all_results.items():
            if variable != 'combined_report' and isinstance(results, dict):
                report += f"\n## Variable: {variable}\n"
                
                # Configuration
                if 'configuration' in results:
                    config = results['configuration']
                    report += f"- Energy Condition: {config.get('energy_condition', 'N/A')}\n"
                    report += f"- Luminosity Weighting: {'Enabled' if config.get('luminosity_weighting') else 'Disabled'}\n"
                    report += f"- Cuts Applied: {len(config.get('cuts', []))}\n"
                
                # Statistics
                if 'statistics' in results:
                    for stage, stats in results['statistics'].items():
                        report += f"\n### Stage: {stage}\n"
                        report += f"- Total Events: {stats.get('total_events', 0):,}\n"
                        report += f"- Data Events: {stats.get('data_events', 0):,}\n"
                        report += f"- MC Events: {stats.get('mc_events', 0):,}\n"
        
        return report

# ============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# ============================================================================
VPHO_KEYS = [
    '__experiment__', '__run__', '__event__', '__production__', '__candidate__', '__ncandidates__',
    'pRecoilTheta', 'pRecoilPhi', 'eRecoil', 'pRecoil', 'mRecoil', 'm2Recoil',
    'mu1clusterE', 'mu2clusterE', 'mu1clusterEoP', 'mu2clusterEoP',
    'mu1clusterPhi', 'mu2clusterPhi', 'mu1clusterTheta', 'mu2clusterTheta',
    'mu1nCDCHits', 'mu2nCDCHits', 'nGammaROE', 'nTracksROE',
    'nPhotonCands', 'nTracks', 'sumE_offcone', 'sumE_offcone_barrel',
    'totalMuonMomentum', 'theta', 'phi', 'E', 'beamE',
    'vpho_px', 'vpho_py', 'vpho_pz', 'vpho_E', 'psnm_ffo',
    'mu1Theta', 'mu2Theta', 'mu1Phi', 'mu2Phi', 'mu1E', 'mu2E', 'mu1P', 'mu2P'
]

GAMMA_KEYS = [
    '__experiment__', '__run__', '__event__', '__production__', '__candidate__', '__ncandidates__',
    '__eventType__', '__weight__', 'mcMatchWeight', 'mcPDG', 'theta', 'phi', 'E',
    'useCMSFrame__bophi__bc', 'useCMSFrame__botheta__bc', 'useCMSFrame__boE__bc',
    'nGammaROE', 'minC2TDist', 'clusterTiming', 'clusterErrorTiming'
]

# Default keys by particle type
DEFAULT_KEYS_BY_PARTICLE = {
    'vpho': VPHO_KEYS,
    'gamma': GAMMA_KEYS,
    'example': None,      # Use all columns
}
@dataclass(frozen=True)
class FrameworkConfig:
    """Immutable configuration with validation."""
    memory_budget_gb: float = 16.0
    enable_cpp_acceleration: bool = True
    enable_adaptive_chunking: bool = True
    cache_dir: Optional[Path] = None
    default_columns: Optional[List[str]] = None
    particle_type: Optional[str] = None
    performance_monitoring: bool = True
    error_resilience_level: str = 'high'  # 'low', 'medium', 'high'
    parallel_stages: bool = True
    energy_condition: str = '5S_scan'  # Default energy condition for weighting
    apply_luminosity_weights: bool = True  # Enable/disable weighting
    
    def __post_init__(self):
        if self.memory_budget_gb <= 0:
            raise ValueError("Memory budget must be positive")
        if self.error_resilience_level not in ['low', 'medium', 'high']:
            raise ValueError(f"Invalid error resilience level: {self.error_resilience_level}")
        if self.cache_dir is None:
            object.__setattr__(self, 'cache_dir', Path.home() / '.belle2_cache')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Set default columns based on particle type if not explicitly provided
        if self.default_columns is None and self.particle_type:
            object.__setattr__(self, 'default_columns', DEFAULT_KEYS_BY_PARTICLE.get(self.particle_type))


def create_belle2_framework(**kwargs) -> Belle2ProductionFrameworkV3:
    """Create framework with backward compatibility."""
    config = FrameworkConfig(**kwargs)
    return Belle2ProductionFrameworkV3(config)

def quick_analysis(data_path: str, 
                  variable: str = 'pRecoil',
                  cuts: Optional[List[str]] = None,
                  particle: str = 'vpho',
                  energy_condition: str = '5S_scan',
                  **kwargs) -> Dict[str, Any]:
    """Quick analysis function with backward compatibility."""
    framework = create_belle2_framework(
        energy_condition=energy_condition, 
        **kwargs
    )
    processes = framework.load_particle_data(data_path, particle=particle)
    
    return framework.run_progressive_analysis(
        processes,
        variable=variable,
        cuts=cuts,
        output_dir=f'./analysis_{variable}_{particle}_{energy_condition}'
    )

class AnalysisChain:
    """Fluent interface for analysis - backward compatibility."""
    
    def __init__(self, framework: Belle2ProductionFrameworkV3):
        self.framework = framework
        self.data = None
        self.cuts = []
        self.results = {}
    
    def load(self, path: str, particle: str = 'vpho') -> 'AnalysisChain':
        self.data = self.framework.load_particle_data(path, particle)
        return self
    
    def cut(self, expression: str) -> 'AnalysisChain':
        self.cuts.append(expression)
        return self
    
    def hist(self, variable: str, **kwargs) -> 'AnalysisChain':
        if self.data:
            data = self.data
            for cut in self.cuts:
                data = data.query(cut)
            self.results[f'hist_{variable}'] = data.hist(variable, **kwargs)
        return self
    
    def analyze(self, variable: str, **kwargs) -> 'AnalysisChain':
        if self.data:
            result = self.framework.run_progressive_analysis(
                self.data, variable, cuts=self.cuts, **kwargs
            )
            self.results[f'analysis_{variable}'] = result
        return self
    
    def get(self) -> Dict[str, Any]:
        return self.results

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main classes
    'Belle2ProductionFrameworkV3',
    'WeightedDataFrame',
    'WeightedProcessDict',
    'WeightedBroadcastResult',
    'LuminosityWeight',
    'HistogramPipeline',
    'EnhancedLuminosityManagerV3',
    'ProgressiveAnalysisV3',
    'ParticleDataLoaderV3',
    
    # Backward compatibility
    'create_belle2_framework',
    'quick_analysis',
    'AnalysisChain',
    
    # Re-exports for compatibility
    'OptimizedUltraLazyDict',
    'UnifiedLazyDataFrame',
    'BroadcastResult',
]