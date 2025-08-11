"""
Belle II MC-Data Comparison Framework - Production Implementation (FIXED)
=======================================================================

Strategic improvements for production deployment:
1. Memory-efficient streaming architecture
2. Comprehensive column discovery and mapping
3. Full weight diagnostics with process-level detail
4. Belle II standard plotting aesthetics
5. C++ acceleration pathway optimization
6. FIXED: Internal consistency and missing methods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass
from enum import Enum
import polars as pl
import warnings
import time
from collections import defaultdict
import re
import gc
import psutil
import sys

# Import proven components
sys.path.append('..')

from belle2_5s_based_test import (
    load_belle2_5s_data_enhanced,
    Belle2DirectoryResolver,
    discover_belle2_files_by_type
)

from layer2_optimized_ultra_lazy_dict import OptimizedUltraLazyDict, BroadcastResult
from layer2_unified_lazy_dataframe import UnifiedLazyDataFrame
from layer2_complete_integration import Belle2Layer2Framework

# # Import Belle II plotting style
# try:
#     from analysis_plots_MCCompare import PlotStyle
#     BELLE2_STYLE_AVAILABLE = True
# except ImportError:
#     BELLE2_STYLE_AVAILABLE = False
#     print("⚠️  Belle II PlotStyle not available, using defaults")
    
# Create dummy PlotStyle for compatibility
class PlotStyle:
    BELLE2_COLORS = {
        'data': '#000000',      # Black
        'mumu': '#E31A1C',      # Vermillion (better than E74C3C)
        'ee': '#1F78B4',        # Blue (more distinct than 3498DB)
        'taupair': '#6A3D9A',   # Purple (better contrast)
        'qqbar': '#33A02C',     # Green (colorblind-safe)
        'gg': '#FF7F00',        # Orange (higher luminance)
        'hhISR': '#666666',     # Gray (neutral)
        'llYY': '#FFD92F',      # Yellow (needs dark edge)
        'BBbar': '#8C564B',     # Brown (earth tone)
    }

# ============================================================================
# MEMORY MONITORING AND MANAGEMENT
# ============================================================================

import gc
import time
import warnings
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class MemoryEvent:
    """Record of memory management events for diagnostics."""
    timestamp: float
    event_type: str  # 'cleanup', 'warning', 'critical'
    memory_usage_gb: float
    objects_collected: int = 0
    trigger_operation: str = ""

class RobustMemoryManager:
    """
    🛡️ PRODUCTION-GRADE: Proactive memory management for Belle II analysis.
    
    CAPABILITIES:
    ✅ Real-time memory monitoring with configurable thresholds
    ✅ Automatic cleanup triggering before critical levels
    ✅ Multi-level cleanup strategies (garbage collection, cache clearing)
    ✅ Comprehensive event logging for performance analysis
    ✅ Integration with Polars and matplotlib resource management
    """
    
    def __init__(self, memory_budget_gb: float = 16.0):
        """
        Initialize memory manager with intelligent defaults.
        
        Args:
            memory_budget_gb: Total memory budget for analysis operations
        """
        self.memory_budget_bytes = int(memory_budget_gb * 1024**3)
        
        # Configurable thresholds for different alert levels
        self.info_threshold = 0.60      # 60% - Informational logging
        self.warning_threshold = 0.75   # 75% - Warning and light cleanup
        self.critical_threshold = 0.85  # 85% - Aggressive cleanup
        self.emergency_threshold = 0.95 # 95% - Emergency measures
        
        # Event tracking for performance analysis
        self.memory_events: List[MemoryEvent] = []
        self.cleanup_count = 0
        self.last_cleanup_time = 0
        
        # Minimum time between cleanups (avoid thrashing)
        self.min_cleanup_interval = 5.0  # seconds
        
        print(f"💾 Memory manager initialized: {memory_budget_gb:.1f}GB budget")
    
    def check_memory_status(self) -> Dict[str, Any]:
        """
        Comprehensive memory status with detailed metrics.
        
        Returns detailed memory information for monitoring and diagnostics.
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Calculate usage metrics
            used_bytes = memory_info.rss
            usage_percent = used_bytes / self.memory_budget_bytes
            available_bytes = self.memory_budget_bytes - used_bytes
            
            # Determine status level
            if usage_percent >= self.emergency_threshold:
                status = 'emergency'
            elif usage_percent >= self.critical_threshold:
                status = 'critical'
            elif usage_percent >= self.warning_threshold:
                status = 'warning'
            elif usage_percent >= self.info_threshold:
                status = 'info'
            else:
                status = 'normal'
            
            return {
                'used_gb': used_bytes / 1024**3,
                'budget_gb': self.memory_budget_bytes / 1024**3,
                'available_gb': available_bytes / 1024**3,
                'usage_percent': usage_percent,
                'status': status,
                'used_bytes': used_bytes,
                'cleanup_count': self.cleanup_count,
                'last_cleanup': self.last_cleanup_time
            }
            
        except ImportError:
            warnings.warn("psutil not available - memory monitoring disabled")
            return {
                'status': 'unknown', 
                'message': 'psutil not available',
                'monitoring_disabled': True
            }
        except Exception as e:
            warnings.warn(f"Memory status check failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'monitoring_failed': True
            }
    
    def should_cleanup(self, operation_name: str = "") -> bool:
        """
        Intelligent cleanup decision based on multiple factors.
        
        Args:
            operation_name: Name of operation requesting cleanup check
            
        Returns:
            True if cleanup should be performed
        """
        status = self.check_memory_status()
        
        # Don't cleanup if monitoring failed
        if status.get('monitoring_disabled') or status.get('monitoring_failed'):
            return False
        
        current_time = time.time()
        
        # Respect minimum cleanup interval to avoid thrashing
        if (current_time - self.last_cleanup_time) < self.min_cleanup_interval:
            return False
        
        # Cleanup decision based on memory status
        memory_status = status.get('status', 'unknown')
        
        if memory_status in ['emergency', 'critical']:
            return True
        elif memory_status == 'warning':
            # For warning level, consider operation type
            high_memory_ops = ['histogram', 'groupby', 'join', 'collect']
            if any(op in operation_name.lower() for op in high_memory_ops):
                return True
        
        return False
    
    def force_cleanup(self, operation_name: str = "", level: str = "standard") -> Dict[str, Any]:
        """
        Execute memory cleanup with configurable intensity.
        
        Args:
            operation_name: Operation that triggered cleanup
            level: Cleanup intensity ('light', 'standard', 'aggressive')
            
        Returns:
            Cleanup results and statistics
        """
        start_time = time.time()
        status_before = self.check_memory_status()
        
        print(f"🧹 Memory cleanup triggered (level: {level})")
        if operation_name:
            print(f"   Triggered by: {operation_name}")
        print(f"   Memory usage: {status_before.get('usage_percent', 0):.1%}")
        
        objects_collected = 0
        cleanup_actions = []
        
        try:
            # LEVEL 1: Basic garbage collection (always performed)
            collected = gc.collect()
            objects_collected += collected
            cleanup_actions.append(f"gc.collect() → {collected} objects")
            
            if level in ['standard', 'aggressive']:
                # LEVEL 2: Clear specific caches
                
                # Clear Polars caches if available
                try:
                    import polars as pl
                    if hasattr(pl, '_clear_cache'):
                        pl._clear_cache()
                        cleanup_actions.append("Polars cache cleared")
                except Exception:
                    pass
                
                # Clear matplotlib figures to free memory
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                    cleanup_actions.append("Matplotlib figures closed")
                except Exception:
                    pass
                
                # Additional garbage collection after cache clearing
                collected = gc.collect()
                objects_collected += collected
                cleanup_actions.append(f"Additional gc.collect() → {collected} objects")
            
            if level == 'aggressive':
                # LEVEL 3: Aggressive cleanup measures
                
                # Force garbage collection with all generations
                for generation in range(3):
                    collected = gc.collect(generation)
                    objects_collected += collected
                
                cleanup_actions.append("Aggressive multi-generation gc")
                
                # Try to clear any remaining caches
                try:
                    # Clear internal Python caches
                    import sys
                    if hasattr(sys, '_clear_type_cache'):
                        sys._clear_type_cache()
                        cleanup_actions.append("Python type cache cleared")
                except Exception:
                    pass
        
        except Exception as e:
            print(f"   ⚠️ Cleanup encountered error: {e}")
            cleanup_actions.append(f"Error: {e}")
        
        # Update tracking
        self.last_cleanup_time = time.time()
        self.cleanup_count += 1
        
        # Get final status
        status_after = self.check_memory_status()
        elapsed_time = time.time() - start_time
        
        # Calculate memory freed
        memory_freed_gb = 0
        if not status_before.get('monitoring_disabled'):
            memory_freed_gb = status_before.get('used_gb', 0) - status_after.get('used_gb', 0)
        
        # Record event
        event = MemoryEvent(
            timestamp=start_time,
            event_type='cleanup',
            memory_usage_gb=status_after.get('used_gb', 0),
            objects_collected=objects_collected,
            trigger_operation=operation_name
        )
        self.memory_events.append(event)
        
        # Report results
        print(f"   ✅ Cleanup completed in {elapsed_time:.2f}s")
        print(f"   Objects collected: {objects_collected}")
        if memory_freed_gb > 0:
            print(f"   Memory freed: {memory_freed_gb:.2f}GB")
        print(f"   Final usage: {status_after.get('usage_percent', 0):.1%}")
        
        return {
            'objects_collected': objects_collected,
            'memory_freed_gb': memory_freed_gb,
            'elapsed_time': elapsed_time,
            'cleanup_actions': cleanup_actions,
            'status_before': status_before,
            'status_after': status_after
        }
    
    def monitor_operation(self, operation_name: str):
        """
        Context manager for monitoring memory during operations.
        
        Usage:
            with memory_manager.monitor_operation("histogram_computation"):
                # Your memory-intensive operation here
                result = compute_histogram(data)
        """
        return MemoryMonitorContext(self, operation_name)
    
    def get_memory_report(self) -> str:
        """Generate comprehensive memory usage report."""
        status = self.check_memory_status()
        
        report_lines = [
            "📊 MEMORY MANAGEMENT REPORT",
            "=" * 40,
            f"Current Usage: {status.get('used_gb', 0):.1f}GB / {status.get('budget_gb', 16):.1f}GB ({status.get('usage_percent', 0):.1%})",
            f"Available: {status.get('available_gb', 0):.1f}GB",
            f"Status: {status.get('status', 'unknown').upper()}",
            f"Cleanup Operations: {self.cleanup_count}",
            "",
            "Recent Memory Events:",
        ]
        
        # Show last 5 memory events
        recent_events = self.memory_events[-5:] if self.memory_events else []
        if recent_events:
            for event in recent_events:
                timestamp_str = time.strftime('%H:%M:%S', time.localtime(event.timestamp))
                report_lines.append(
                    f"  {timestamp_str} - {event.event_type}: {event.memory_usage_gb:.1f}GB"
                )
        else:
            report_lines.append("  No recent events")
        
        return "\n".join(report_lines)


class MemoryMonitorContext:
    """Context manager for monitoring memory during specific operations."""
    
    def __init__(self, memory_manager: RobustMemoryManager, operation_name: str):
        self.memory_manager = memory_manager
        self.operation_name = operation_name
        self.start_status = None
    
    def __enter__(self):
        self.start_status = self.memory_manager.check_memory_status()
        
        # Check if cleanup needed before operation
        if self.memory_manager.should_cleanup(self.operation_name):
            self.memory_manager.force_cleanup(self.operation_name, level="standard")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_status = self.memory_manager.check_memory_status()
        
        # Log operation completion
        if not self.start_status.get('monitoring_disabled'):
            memory_used = end_status.get('used_gb', 0) - self.start_status.get('used_gb', 0)
            if memory_used > 0.1:  # Only log if significant memory was used
                print(f"💾 Operation '{self.operation_name}' used {memory_used:.2f}GB")
        
        # Check if cleanup needed after operation
        if end_status.get('status') in ['critical', 'emergency']:
            self.memory_manager.force_cleanup(
                f"post-{self.operation_name}", 
                level="aggressive"
            )


# ============================================================================
# ENHANCED LUMINOSITY DATABASE WITH PROCESS-LEVEL OUTPUT
# ============================================================================

class BelleLuminosityDB:
    """
    Robust luminosity database with detailed process-level diagnostics.
    """
    
    def __init__(self):
        # Complete luminosity mappings
        self.luminosities = {
            '5S_scan': {
                'data': 19.6348,
                'data_5s': 19.6348,
                'data5s': 19.6348,
                'mumu': 78.5391,
                'uubar': 78.5391,
                'ddbar': 78.5391,
                'ssbar': 78.5391,
                'ccbar': 78.5391,
                'ee': 1.9635,
                'taupair': 78.5391,
                'gg': 39.2695,
                'hhisr': 19.6348,
                'eeee': 19.6348,
                'eemumu': 19.6348,
                'llxx': 19.6348,
                'charged': 1e-6,
                'mixed': 1e-6,
                'default': 78.5391
            },
            '4S_on': {
                'data': 357.3065,
                'uubar': 1368.4666,
                'ddbar': 1429.2255,
                'ssbar': 1429.2255,
                'ccbar': 1368.4666,
                'mumu': 1368.4666,
                'ee': 34.2116,
                'taupair': 1368.4666,
                'gg': 684.2325,
                'hhisr': 357.3065,
                'charged': 340.0,
                'mixed': 340.0,
                'default': 1000.0
            }
        }
        
        self.process_patterns = {
            'mumu': ['mumu', 'mu+mu-', 'muon'],
            'ee': ['ee', 'e+e-', 'electron', 'bhabha'],
            'taupair': ['taupair', 'tau', 'tautau'],
            'uubar': ['uubar', 'uu'],
            'ddbar': ['ddbar', 'dd'],
            'ssbar': ['ssbar', 'ss'],
            'ccbar': ['ccbar', 'cc'],
            'gg': ['gg', 'gamma', 'photon'],
            'hhisr': ['hhisr', 'hh_isr', 'hadron_isr', 'hh'],
            'eeee': ['eeee', '4e', 'four_electron'],
            'eemumu': ['eemumu', '2e2mu'],
            'llxx': ['llxx', 'llyy', 'four_lepton'],
            'charged': ['charged', 'b+b-', 'bplus'],
            'mixed': ['mixed', 'b0b0bar', 'bneutral']
        }
    
    def get_weight_with_diagnostics(self, process_name: str, 
                                  energy_condition: str = '5S_scan') -> Tuple[float, Dict]:
        """
        Calculate weight with full diagnostic information.
        """
        diagnostics = {
            'process_name': process_name,
            'energy_condition': energy_condition,
            'process_type': None,
            'data_luminosity': None,
            'mc_luminosity': None,
            'weight': None,
            'method': None
        }
        
        # Special handling for data
        if any(d in process_name.lower() for d in ['data', 'proc', 'prompt']):
            diagnostics['process_type'] = 'data'
            diagnostics['weight'] = 1.0
            diagnostics['method'] = 'data_process'
            return 1.0, diagnostics
        
        # Extract process type
        process_type = self._extract_process_type(process_name)
        diagnostics['process_type'] = process_type
        
        # Get luminosity database
        lumi_db = self.luminosities.get(energy_condition, self.luminosities['5S_scan'])
        
        # Get luminosities
        data_lumi = lumi_db.get('data', 19.6348)
        mc_lumi = lumi_db.get(process_type, lumi_db.get('default', 78.5391))
        
        diagnostics['data_luminosity'] = data_lumi
        diagnostics['mc_luminosity'] = mc_lumi
        
        # Calculate weight
        if mc_lumi <= 0:
            mc_lumi = lumi_db.get('default', 78.5391)
            diagnostics['method'] = 'fallback_default'
        else:
            diagnostics['method'] = 'standard_lookup'
        
        weight = data_lumi / mc_lumi
        diagnostics['weight'] = weight
        
        return weight, diagnostics
    
    def _extract_process_type(self, process_name: str) -> str:
        """Enhanced process type extraction."""
        name_lower = process_name.lower()
        
        # Direct pattern matching
        for process, patterns in self.process_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                return process
        
        # Heuristic fallbacks
        if 'continuum' in name_lower:
            return 'uubar'
        elif any(q in name_lower for q in ['quark', 'qqbar']):
            return 'ccbar'
        
        return 'default'

# ============================================================================
# COMPREHENSIVE COLUMN DISCOVERY
# ============================================================================

class ColumnDiscovery:
    """Advanced column discovery and validation system."""
    
    def __init__(self):
        self.discovered_columns = {}
        self.column_stats = defaultdict(int)
        
    def discover_all_columns(self, data_dict: OptimizedUltraLazyDict) -> Dict[str, Set[str]]:
        """
        Discover all available columns across all processes.
        """
        print("\n🔍 Discovering available columns across all processes...")
        
        all_columns = set()
        process_columns = {}
        
        for name, df in data_dict.items():
            try:
                # Get columns with multiple strategies
                columns = self._get_dataframe_columns(df)
                if columns:
                    process_columns[name] = columns
                    all_columns.update(columns)
                    
                    # Track column frequency
                    for col in columns:
                        self.column_stats[col] += 1
                        
            except Exception as e:
                print(f"   ⚠️  Failed to get columns from {name}: {e}")
        
        # Print discovery summary
        print(f"\n📊 Column Discovery Summary:")
        print(f"   Total unique columns: {len(all_columns)}")
        print(f"   Processes analyzed: {len(process_columns)}")
        
        # Show most common columns
        common_cols = sorted(self.column_stats.items(), 
                           key=lambda x: x[1], reverse=True)[:20]
        print(f"\n   Most common columns:")
        for col, count in common_cols:
            coverage = count / len(process_columns) * 100
            print(f"      {col:30s} : {coverage:5.1f}% coverage ({count}/{len(process_columns)})")
        
        self.discovered_columns = process_columns
        return process_columns
    
    def _get_dataframe_columns(self, df: Any) -> Set[str]:
        """Get columns from various DataFrame types."""
        columns = set()
        
        # Try multiple access methods
        if hasattr(df, 'columns'):
            columns.update(df.columns)
        elif hasattr(df, 'schema'):
            if hasattr(df.schema, 'names'):
                columns.update(df.schema.names())
            else:
                columns.update(df.schema.keys())
        elif hasattr(df, 'collect_schema'):
            try:
                schema = df.collect_schema()
                columns.update(schema.names())
            except:
                pass
        elif hasattr(df, '_lazy_frames') and df._lazy_frames:
            # UnifiedLazyDataFrame
            try:
                lf = df._lazy_frames[0]
                schema = lf.schema
                columns.update(schema.keys())
            except:
                pass
        
        return columns


from belle2_style_manager import UnifiedBelle2PlotStyle
class MCDataComparisonEngine:
    """Your existing class with critical improvements."""
    EXPECTED_GROUPS = ['data', 'mumu', 'ee', 'gg', 'hhISR', 'llYY', 'qqbar', 'BBbar', 'taupair']
    
    def __init__(self, vpho_data, energy_condition='5S_scan', output_dir:Optional[str] = None,**kwargs):
        """
        🔧 MODIFIED: Initialization with unified styling and memory management.
        """
        # Your existing initialization code here...
        # (keep everything you have, just replace the color/label parts)
        
        # UNIFIED STYLING: Single source of truth for all visual elements
        self.colors = UnifiedBelle2PlotStyle.COLORS
        self.labels = UnifiedBelle2PlotStyle.LABELS
        self.variable_labels = UnifiedBelle2PlotStyle.VARIABLE_LABELS
        self.data_processes = []
        self.mc_processes = {}
        self.weights = {}
        self.weight_diagnostics = {}
        self.column_discovery = ColumnDiscovery()
        self.available_columns = self.column_discovery.discover_all_columns(vpho_data)
        self.lumi_db = BelleLuminosityDB()
        
        # Output directory
        if output_dir is None:
            output_dir = f'./mc_data_{energy_condition}'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        # Apply Belle II standard styling
        UnifiedBelle2PlotStyle.apply_belle2_style()

        
        # MEMORY MANAGEMENT: Add robust memory monitoring
        self.memory_budget_gb = kwargs.get('memory_budget_gb', 16.0)
        self.memory_manager = RobustMemoryManager(self.memory_budget_gb)
        self.framework = Belle2Layer2Framework(memory_budget_gb=self.memory_budget_gb)
        # Initial memory status
        status = self.memory_manager.check_memory_status()
        print(f"💾 Memory initialized: {status.get('used_gb', 0):.1f}GB / {status.get('budget_gb', 16):.1f}GB")
        self.vpho_data = vpho_data
        self.energy_condition = energy_condition
        # Process classification
        self._classify_processes_enhanced()
        # Calculate weights with full diagnostics
        self._calculate_weights_with_diagnostics()
        
      
    
    def _create_ratio_panel_optimized(self, ax_ratio, result, bin_centers, bin_edges):
        """
        🔧 COMPLETE REPLACEMENT: Physics-standard ratio panel implementation
        """
        
        # Extract data with safe fallbacks
        data_hist = result.get('data_hist')
        data_err = result.get('data_err')
        mc_total = result.get('mc_total')
        mc_err = result.get('mc_err')
        
        print("📊 Creating physics-standard Data/MC ratio panel...")
        
        # SAFETY: Validate data availability
        if data_hist is None or mc_total is None or len(data_hist) == 0 or len(mc_total) == 0:
            ax_ratio.text(0.5, 0.5, 'No Data Available for Ratio', 
                         transform=ax_ratio.transAxes,
                         ha='center', va='center', fontsize=12, 
                         style='italic', color='gray')
            ax_ratio.set_ylim(0.5, 1.5)
            ax_ratio.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            ax_ratio.set_ylabel('Data/MC', fontsize=14, fontweight='bold')
            return 0
        
        # INITIALIZATION: Prepare ratio arrays
        ratio = np.ones_like(mc_total)
        ratio_err = np.zeros_like(mc_total)
        
        # Handle missing error arrays
        if data_err is None:
            data_err = np.sqrt(np.maximum(data_hist, 1))  # Poisson errors for data
        if mc_err is None:
            mc_err = np.sqrt(np.maximum(mc_total, 1))     # Poisson errors for MC
        
        # ROBUST: Physics-standard validity mask
        valid_mask = (
            # (mc_total > 0.1) &                      # Sufficient MC statistics
            (data_hist >= 0) &                      # Non-negative data
            # (mc_err < mc_total * 0.8) &            # Reasonable MC errors
            # (data_err < data_hist * 3.0) &         # Reasonable data errors (allow larger for Poisson)
            # np.isfinite(mc_total) &                # No infinities in MC
            # np.isfinite(data_hist) &               # No infinities in data
                       # Avoid division by very small numbers
        )
        
        valid_count = np.sum(valid_mask)
        print(f"   Valid ratio points identified: {valid_count}/{len(valid_mask)}")
        
        if valid_count > 0:
            # PHYSICS: Standard Data/MC ratio calculation
            ratio[valid_mask] = data_hist[valid_mask] / mc_total[valid_mask]
            
            # PHYSICS: Rigorous error propagation
            # Formula: σ(r) = r × √((σ_data/data)² + (σ_mc/mc)²)
            
            # Calculate relative errors safely
            rel_data_err = np.zeros_like(data_hist)
            rel_mc_err = np.zeros_like(mc_total)
            
            # For valid points, calculate relative errors
            data_valid = (data_hist > 0) & valid_mask
            mc_valid = (mc_total > 0) & valid_mask
            
            rel_data_err[data_valid] = data_err[data_valid] / data_hist[data_valid]
            rel_mc_err[mc_valid] = mc_err[mc_valid] / mc_total[mc_valid]
            
            # Combined relative error and propagation to ratio
            rel_total_err = np.sqrt(rel_data_err**2 + rel_mc_err**2)
            ratio_err[valid_mask] = ratio[valid_mask] * rel_total_err[valid_mask]
            
            # VISUALIZATION: Physics-standard presentation
            valid_centers = bin_centers[valid_mask]
            valid_ratios = ratio[valid_mask]
            valid_errors = ratio_err[valid_mask]
            
            # Calculate appropriate horizontal error bar width
            bin_width = np.mean(np.diff(bin_edges)) if len(bin_edges) > 1 else 1.0
            
            # STANDARD: Black data points with error bars
            ax_ratio.errorbar(
                valid_centers, 
                valid_ratios, 
                yerr=valid_errors,
                xerr=bin_width * 0.25,            # Subtle horizontal error bars
                fmt='o',                          # Circle markers (physics standard)
                color='black',                    # Always black for data
                markersize=5,                     # Appropriate visibility
                markeredgewidth=0.8,              # Clean definition
                markeredgecolor='black',          # Consistent edge
                markerfacecolor='black',          # Solid fill
                elinewidth=1.4,                   # Visible error bars
                capsize=2.5,                      # Proportional caps
                capthick=1.2,                     # Substantial caps
                zorder=100,                       # Always on top
                alpha=0.9                         # Slight transparency for overlapping points
            )
            
            # OPTIONAL: MC statistical uncertainty band around unity
            mc_rel_err = np.divide(mc_err, mc_total, 
                                 out=np.zeros_like(mc_err), 
                                 where=mc_total > 0)
            
            # Only show MC uncertainty band if errors are reasonable
            reasonable_mc_err = mc_rel_err < 0.5  # Don't show if MC errors > 50%
            if np.any(reasonable_mc_err):
                ax_ratio.fill_between(
                    bin_centers,
                    1 - mc_rel_err,
                    1 + mc_rel_err,
                    alpha=0.25,
                    color='lightgray',
                    step='mid',
                    label='MC stat. unc.',
                    zorder=1
                )
        
        # PHYSICS: Mandatory unity reference line (red dashed)
        ax_ratio.axhline(y=1, color='red', linestyle='--', 
                        linewidth=2.0, alpha=0.85, zorder=50,
                        label='Unity')
        
        # INTELLIGENT: Data-driven y-axis range
        if valid_count > 2:  # Need at least 3 points for meaningful range
            ratio_values = ratio[valid_mask]
            
            # Use robust statistics (median and MAD)
            ratio_median = np.median(ratio_values)
            ratio_mad = np.median(np.abs(ratio_values - ratio_median))
            
            # Set range based on data spread with intelligent bounds
            y_spread = max(0.1, 4 * ratio_mad)  # At least 0.1, typically 4×MAD
            y_min = max(0.2, ratio_median - y_spread)
            y_max = min(3.0, ratio_median + y_spread)
            
            # Ensure minimum range for visibility
            if y_max - y_min < 0.3:
                y_center = (y_max + y_min) / 2
                y_min = y_center - 0.15
                y_max = y_center + 0.15
            
            ax_ratio.set_ylim(y_min, y_max)
            
        else:
            # Default range when insufficient data
            ax_ratio.set_ylim(0.6, 1.4)
        
        # PROFESSIONAL: Grid and styling
        ax_ratio.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.6)
        ax_ratio.grid(True, which='minor', alpha=0.2, linestyle='-', linewidth=0.4)
        ax_ratio.set_ylabel('Data/MC', fontsize=14, fontweight='bold')
        
        # Enable minor ticks for precision
        ax_ratio.minorticks_on()
        
        # Optional: Add horizontal reference lines at common ratios
        for ref_ratio in [0.5, 0.8, 1.2, 1.5]:
            if ax_ratio.get_ylim()[0] < ref_ratio < ax_ratio.get_ylim()[1]:
                ax_ratio.axhline(y=ref_ratio, color='gray', linestyle=':', 
                               linewidth=0.8, alpha=0.4, zorder=2)
        
        print(f"   ✅ Physics-standard ratio panel completed with {valid_count} data points")
        return valid_count

    
    def _get_component_label(self, component: str) -> str:
        """
        FIXED: Get proper label for component.
        
        This method was missing and causing the error.
        """
        return self.labels.get(component, component)
    
    def _get_component_color(self, component: str) -> str:
        """Get proper color for component."""
        return self.colors.get(component, '#888888')  # Gray fallback
    
    def _get_variable_label(self, variable: str) -> str:
        """Get Belle II standard variable labels."""
        return self.variable_labels.get(variable, variable)
    
    def _classify_processes_enhanced(self):
        """Enhanced process classification."""
        self.data_processes = []
        self.mc_processes = {group: [] for group in self.EXPECTED_GROUPS if group != 'data'}
        
        print("\n📊 Classifying processes...")
        
        for name in self.vpho_data.keys():
            name_lower = name.lower()
            
            # Enhanced data detection
            data_patterns = [
                'data', 'proc', 'prompt',
                'data5s', 'data4s', 'dataoff',
                'data_5s', 'data_4s', 'data_off'
            ]
            
            # Check if it's data
            if any(pattern in name_lower for pattern in data_patterns):
                if not any(mc in name_lower for mc in ['mc', 'sim']):
                    self.data_processes.append(name)
                    print(f"   ✅ Data: {name}")
                    continue
            
            # MC classification
            classified = False
            
            if 'mumu' in name_lower and 'eemumu' not in name_lower:
                self.mc_processes['mumu'].append(name)
                classified = True
            elif 'ee' in name_lower and 'eeee' not in name_lower and 'eemumu' not in name_lower:
                self.mc_processes['ee'].append(name)
                classified = True
            elif any(ll in name_lower for ll in ['eeee', 'eemumu', 'llxx', 'llyy']):
                self.mc_processes['llYY'].append(name)
                classified = True
            elif any(q in name_lower for q in ['uubar', 'ddbar', 'ssbar', 'ccbar']):
                self.mc_processes['qqbar'].append(name)
                classified = True
            elif any(b in name_lower for b in ['charged', 'mixed']):
                self.mc_processes['BBbar'].append(name)
                classified = True
            elif 'taupair' in name_lower or 'tautau' in name_lower:
                self.mc_processes['taupair'].append(name)
                classified = True
            elif 'gg' in name_lower and 'hhisr' not in name_lower:
                self.mc_processes['gg'].append(name)
                classified = True
            elif 'hhisr' in name_lower or 'hh_isr' in name_lower:
                self.mc_processes['hhISR'].append(name)
                classified = True
            
            if not classified:
                print(f"   ⚠️  Unclassified: {name} → qqbar")
                self.mc_processes['qqbar'].append(name)
        
        # Remove empty groups
        self.mc_processes = {k: v for k, v in self.mc_processes.items() if v}
        
        # Summary
        print(f"\n📊 Classification Summary:")
        print(f"   Data: {len(self.data_processes)} datasets")
        for dp in self.data_processes:
            print(f"      - {dp}")
        
        for group, processes in self.mc_processes.items():
            print(f"   {group}: {len(processes)} datasets")
    
    def _calculate_weights_with_diagnostics(self):
        """Calculate weights with comprehensive diagnostics."""
        self.weights = {}
        self.weight_diagnostics = {}
        
        print(f"\n⚖️  Calculating Luminosity Weights for {self.energy_condition}")
        print("=" * 80)
        print(f"{'Process':<40} {'Type':<10} {'MC Lumi':<10} {'Weight':<10} {'Method':<20}")
        print("=" * 80)
        
        # Data processes
        for proc in self.data_processes:
            weight, diag = self.lumi_db.get_weight_with_diagnostics(proc, self.energy_condition)
            self.weight_diagnostics[proc] = diag
            print(f"{proc:<40} {'data':<10} {'N/A':<10} {weight:<10.4f} {diag['method']:<20}")
        
        # MC processes
        for group, processes in self.mc_processes.items():
            self.weights[group] = {}
            
            for proc in processes:
                weight, diag = self.lumi_db.get_weight_with_diagnostics(proc, self.energy_condition)
                self.weights[group][proc] = weight
                self.weight_diagnostics[proc] = diag
                
                print(f"{proc:<40} {diag['process_type']:<10} "
                      f"{diag['mc_luminosity']:<10.2f} {weight:<10.4f} "
                      f"{diag['method']:<20}")
        
        print("=" * 80)
        
        # Summary statistics
        print(f"\n📊 Weight Summary by Group:")
        for group, proc_weights in self.weights.items():
            if proc_weights:
                weights = list(proc_weights.values())
                print(f"   {group:10s}: μ={np.mean(weights):7.4f}, "
                      f"σ={np.std(weights):7.4f}, "
                      f"range=[{min(weights):.4f}, {max(weights):.4f}]")
    
    def _compute_histogram_safe(self, 
                              df: UnifiedLazyDataFrame,
                              variable: str,
                              bins: int,
                              range: Tuple[float, float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute histogram with C++ acceleration preference.
        """
        try:
            # Check memory before computation
            if self.memory_manager.should_cleanup():
                print("   🧹 Triggering memory cleanup...")
                self.memory_manager.force_cleanup()
            
            # Try C++ accelerated path first
            if hasattr(self.framework, '_cpp_histogram') and self.framework._cpp_histogram:
                try:
                    print("   🚀 Attempting C++ accelerated histogram...")
                    counts, edges = self.framework.compute_histogram(
                        df, variable, bins=bins, range=range
                    )
                    print("   ✅ C++ histogram successful")
                    return counts, edges
                except Exception as e:
                    print(f"   ⚠️  C++ failed: {e}, falling back to Polars")
            
            # Fallback to direct histogram computation
            if hasattr(df, 'hist'):
                counts, edges = df.hist(variable, bins=bins, range=range)
                return counts, edges
            
            # Manual computation as last resort
            print("   ⚠️  Using manual histogram computation")
            if hasattr(df, 'collect'):
                # For lazy frames, collect in chunks to manage memory
                chunk_size = 1_000_000
                all_values = []
                
                # Collect in chunks
                collected = df.select(variable).collect()
                values = collected[variable].to_numpy()
                
                # Compute histogram
                counts, edges = np.histogram(values, bins=bins, range=range)
                return counts, edges
                
        except Exception as e:
            print(f"   ❌ Histogram computation failed: {e}")
            return None, None
    
    def _compute_comparison(self, 
                          data: OptimizedUltraLazyDict,
                          variable: str,
                          bins: int,
                          range: Tuple[float, float]) -> Dict[str, Any]:
        """
        Compute MC-data comparison with memory management.
        """
        print(f"   📊 Computing histograms for '{variable}'...")
        
        # Check available columns
        print(f"   🔍 Checking column availability...")
        processes_with_column = 0
        for name in data.keys():
            if name in self.available_columns:
                if variable in self.available_columns[name]:
                    processes_with_column += 1
        
        print(f"   ✅ Column '{variable}' found in {processes_with_column}/{len(data)} processes")
        
        # Initialize histogram storage
        data_hist = np.zeros(bins)
        data_counts = []
        edges = np.linspace(range[0], range[1], bins + 1)
        
        # Process data histograms
        print("   📊 Processing data...")
        for proc in self.data_processes:
            if proc in data:
                try:
                    df = data[proc]
                    counts, edges = self._compute_histogram_safe(
                        df, variable, bins, range
                    )
                    if counts is not None:
                        data_counts.append(counts)
                        print(f"      ✅ {proc}: {np.sum(counts):.0f} events")
                except Exception as e:
                    print(f"      ❌ {proc}: {e}")
        
        if data_counts:
            data_hist = np.sum(data_counts, axis=0)
        
        # Process MC histograms
        print("   📊 Processing MC...")
        mc_hists = {}
        mc_total = np.zeros(bins)
        
        for group, processes in self.mc_processes.items():
            group_hist = np.zeros(bins)
            
            for proc in processes:
                if proc in data:
                    try:
                        df = data[proc]
                        counts, _ = self._compute_histogram_safe(
                            df, variable, bins, range
                        )
                        if counts is not None:
                            weight = self.weights[group].get(proc, 1.0)
                            weighted_counts = counts * weight
                            group_hist += weighted_counts
                    except Exception as e:
                        print(f"      ❌ {proc}: {e}")
            
            if np.sum(group_hist) > 0:
                mc_hists[group] = group_hist
                mc_total += group_hist
                print(f"   ✅ {group}: {np.sum(group_hist):.0f} weighted events")
        
        # Calculate statistics
        data_err = self._calculate_poisson_errors(data_hist)
        mc_err = self._calculate_poisson_errors(mc_total)
        
        ratio, ratio_err = self._calculate_ratio_with_errors(
            data_hist, data_err, mc_total, mc_err
        )
        
        chi2, ndof, p_value = self._calculate_chi2(
            data_hist, mc_total, data_err, mc_err
        )
        
        return {
            'data_hist': data_hist,
            'data_err': data_err,
            'mc_hists': mc_hists,
            'mc_total': mc_total,
            'mc_err': mc_err,
            'edges': edges,
            'ratio': ratio,
            'ratio_err': ratio_err,
            'chi2': chi2,
            'ndof': ndof,
            'p_value': p_value
        }
    
    def _create_comparison_plot(self, result: Dict[str, Any], 
                              variable: str, stage: str):
        """Create Belle II standard comparison plot with stepfilled histograms."""
        
        # Skip if no data
        if np.sum(result['data_hist']) == 0 and np.sum(result['mc_total']) == 0:
            print(f"   ⚠️  No events to plot for {stage}")
            return
        
        # Create figure with Belle II proportions
        fig = plt.figure(figsize=(12, 9))  # Golden ratio
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.04)
        
        ax_main = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
        
        # Calculate bin properties
        bin_edges = result['edges']
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        
        # CRITICAL CHANGE: Stepfilled histograms instead of bar charts
        bottom = np.zeros(len(bin_centers))
        
        # Belle II standard ordering
        plot_order = ['BBbar', 'qqbar', 'taupair', 'ee', 'mumu', 'gg', 'hhISR', 'llYY']
        
        for component in plot_order:
            if component in result['mc_hists']:
                counts = result['mc_hists'][component]
                if np.sum(counts) > 0:
                    # FIXED: Use consistent method calls
                    color = self._get_component_color(component)
                    label = self._get_component_label(component)
                    
                    # STEPFILLED HISTOGRAM using fill_between
                    x_edges = bin_edges
                    y_bottom = np.append(bottom, bottom[-1])
                    y_top = np.append(bottom + counts, (bottom + counts)[-1])
                    
                    ax_main.fill_between(
                        x_edges,
                        y_bottom,
                        y_top,
                        step='post',
                        facecolor=color,
                        edgecolor='black',
                        alpha=0.85,
                        linewidth=0.5,
                        label=label
                    )
                    
                    # Add edge for definition
                    ax_main.step(x_edges, y_top, where='post',
                               color='black', linewidth=0.5, alpha=0.5)
                    
                    bottom += counts
        
        # MC uncertainty band with hatching
        self._add_mc_uncertainty_stepfilled(ax_main, result)
        
        # Plot data points with optimized style
        if np.sum(result['data_hist']) > 0:
            valid = result['data_hist'] > 0
            ax_main.errorbar(
                bin_centers[valid], 
                result['data_hist'][valid],
                yerr=result['data_err'][valid],
                fmt='o',  # Circles
                color='black',
                markersize=4,  # Smaller markers
                markeredgewidth=0,  # No edge
                elinewidth=1.0,
                capsize=0,  # No caps
                label='Data',
                zorder=100
            )
        
        # Create optimized ratio panel
        self._create_ratio_panel_optimized(ax_ratio, result, bin_centers, bin_edges)
        
        # Apply final styling
        self._apply_belle2_styling_enhanced(ax_main, ax_ratio, result, variable, stage)
        
        # Save
        filename = f"{self.energy_condition}_{stage}_{variable}.pdf"
        filepath = self.output_dir / filename
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"   💾 Saved: {filepath}")
        plt.show()
        plt.close(fig)
    
    def _add_mc_uncertainty_stepfilled(self, ax, result):
        """Add MC statistical uncertainty band with stepfilled style."""
        bin_edges = result['edges']
        mc_total = result['mc_total']
        mc_err = result['mc_err']
        
        # Create uncertainty band using fill_between
        x_edges = bin_edges
        y_nominal = np.append(mc_total, mc_total[-1])
        y_up = np.append(mc_total + mc_err, (mc_total + mc_err)[-1])
        y_down = np.append(np.maximum(0, mc_total - mc_err), 
                          np.maximum(0, (mc_total - mc_err)[-1]))
        
        ax.fill_between(
            x_edges,
            y_down,
            y_up,
            step='post',
            facecolor='gray',
            alpha=0.3,
            edgecolor='none',
            hatch='///',  # Hatching for accessibility
            label='MC stat. unc.',
            zorder=50
        )
    
    # def _create_ratio_panel_optimized(self, ax_ratio, result, bin_centers, bin_edges):
    #     """
    #     Create ratio panel with PHYSICS-STANDARD data/MC visualization.
        
    #     STANDARD PARTICLE PHYSICS CONVENTIONS:
    #     ├── Black dots: Data/MC ratio points with error bars
    #     ├── Gray bars: Vertical fill from unity (1) to each ratio point
    #     ├── Unity line: Red dashed horizontal line at ratio = 1
    #     └── MC uncertainty: Optional gray band around unity
    #     """
        
    #     # Extract ratio data
    #     ratio_data = result['ratio']
    #     ratio_errors = result['ratio_err']
        
    #     # Create mask for valid ratio points
    #     valid_mask = (ratio_data > 0) & (ratio_errors < 10) & np.isfinite(ratio_data)
        
    #     if np.any(valid_mask):
    #         valid_centers = bin_centers[valid_mask]
    #         valid_ratios = ratio_data[valid_mask]
    #         valid_errors = ratio_errors[valid_mask]
            
    #         # PHYSICS STANDARD: Vertical bars from 1 to ratio points
    #         bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
    #         bar_width = bin_width * 0.8  # 80% of bin width for visual clarity
            
    #         # Create individual bars from unity to each ratio point
    #         for center, ratio_val in zip(valid_centers, valid_ratios):
    #             if ratio_val > 1:
    #                 # Ratio > 1: Gray bar from 1 upward to ratio
    #                 height = ratio_val - 1
    #                 bottom = 1
    #                 color = 'lightgray'
    #                 alpha = 0.6
    #             else:
    #                 # Ratio < 1: Gray bar from ratio upward to 1  
    #                 height = 1 - ratio_val
    #                 bottom = ratio_val
    #                 color = 'lightgray'
    #                 alpha = 0.6
                
    #             # Draw vertical bar using matplotlib's bar function
    #             ax_ratio.bar(
    #                 center, height, width=bar_width, bottom=bottom,
    #                 color=color, alpha=alpha, edgecolor='none'
    #             )
            
    #         # PHYSICS STANDARD: Black error bar dots on top
    #         ax_ratio.errorbar(
    #             valid_centers, valid_ratios, yerr=valid_errors,
    #             fmt='o',           # Circle markers
    #             color='black',     # Standard physics black
    #             markersize=4,      # Optimized size
    #             markeredgewidth=0, # Clean appearance
    #             elinewidth=1.2,    # Slightly thicker error bars
    #             capsize=2,         # Small caps for precision
    #             zorder=10          # Ensure dots appear on top
    #         )
            
    #         # OPTIONAL: MC statistical uncertainty band around unity
    #         # (Only if explicitly requested - commented out by default)
    #         if False:  # Set to True if MC uncertainty band desired
    #             mc_ratio_err = np.divide(result['mc_err'], result['mc_total'],
    #                                 out=np.zeros_like(result['mc_err']),
    #                                 where=result['mc_total']>0)
                
    #             x_edges = bin_edges
    #             y_up = np.append(1 + mc_ratio_err, (1 + mc_ratio_err)[-1])
    #             y_down = np.append(1 - mc_ratio_err, (1 - mc_ratio_err)[-1])
                
    #             ax_ratio.fill_between(
    #                 x_edges, y_down, y_up, step='post',
    #                 facecolor='yellow', alpha=0.2, edgecolor='none',
    #                 label='MC stat. unc.', zorder=1
    #             )
        
    #     # PHYSICS STANDARD: Unity reference line
    #     ax_ratio.axhline(y=1, color='red', linestyle='--', 
    #                     linewidth=1.5, alpha=0.8, zorder=5)
        
    #     # ENHANCED: Dynamic y-axis range based on actual data
    #     if np.any(valid_mask):
    #         ratio_values = ratio_data[valid_mask]
    #         ratio_range = np.max(ratio_values) - np.min(ratio_values) 
            
    #         if ratio_range > 0:
    #             margin = ratio_range * 0.15  # 15% margin
    #             y_min = max(0.0, np.min(ratio_values) - margin)
    #             y_max = min(3.0, np.max(ratio_values) + margin)
                
    #             # Ensure reasonable bounds
    #             y_min = max(0.5, y_min)
    #             y_max = min(2.0, y_max)
                
    #             ax_ratio.set_ylim(y_min, y_max)
    #         else:
    #             ax_ratio.set_ylim(0.8, 1.2)  # Default tight range
    #     else:
    #         ax_ratio.set_ylim(0.8, 1.2)  # Fallback range
        
    #     # ENHANCED: Professional grid styling
    #     ax_ratio.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.5)
    #     ax_ratio.grid(True, which='minor', alpha=0.15, linestyle='-', linewidth=0.3)
        
    #     # PHYSICS STANDARD: Axis labels
    #     ax_ratio.set_ylabel('Data/MC', fontsize=14, fontweight='bold')
        
    #     print(f"✅ Physics-standard ratio plot: {np.sum(valid_mask)} valid points visualized")
        
    def _apply_belle2_styling_enhanced(self, ax_main, ax_ratio, result, variable, stage):
        """Apply enhanced Belle II standard styling."""
        # Main panel
        ax_main.set_ylabel('Events / bin', fontsize=14)
        ax_main.set_yscale('log')
        ax_main.set_ylim(bottom=0.5, top=ax_main.get_ylim()[1] * 2)
        ax_main.grid(True, alpha=0.25, which='both', linestyle='-', linewidth=0.5)
        
        # Legend with optimized placement
        ax_main.legend(loc='upper right', fontsize=11, frameon=True,
                      fancybox=False, shadow=False, ncol=2,
                      framealpha=0.95, edgecolor='black')
        
        # Ratio panel
        ax_ratio.set_xlabel(self._get_variable_label(variable), fontsize=14)
        ax_ratio.set_ylabel('Data/MC', fontsize=14)
        ax_ratio.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        
        # Title
        title = f'Belle II {self.energy_condition.replace("_", " ")} - {stage.replace("_", " ")}'
        ax_main.set_title(title, fontsize=16, pad=10)
        
        # Statistics box
        self._add_belle2_stats_box(ax_main, result)
        
        # Remove x-labels from main
        ax_main.tick_params(labelbottom=False)
        
        # Minor ticks
        ax_main.minorticks_on()
        ax_ratio.minorticks_on()
        
        # Remove top/right spines for cleaner look
        for ax in [ax_main, ax_ratio]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    def _add_belle2_stats_box(self, ax, result):
        """Add Belle II style statistics box."""
        chi2_ndf = result['chi2'] / max(result['ndof'], 1)
        
        # Belle II standard format
        text_lines = [
            'Belle II',
            f'∫L dt = {self.lumi_db.luminosities[self.energy_condition]["data"]:.1f} fb$^{{-1}}$',
            '',
            f'Data: {int(np.sum(result["data_hist"]))}',
            f'MC: {int(np.sum(result["mc_total"]))}',
        ]
        
        if np.sum(result["mc_total"]) > 0:
            ratio = np.sum(result["data_hist"]) / np.sum(result["mc_total"])
            text_lines.append(f'Data/MC: {ratio:.3f}')
        
        text = '\n'.join(text_lines)
        
        # Belle II box style
        props = dict(boxstyle='round,pad=0.5', 
                    facecolor='white',
                    edgecolor='black', 
                    alpha=0.95,
                    linewidth=1.5)
        
        ax.text(0.95, 0.95, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               horizontalalignment='right', bbox=props,
               fontfamily='monospace')
    
    def _calculate_poisson_errors(self, counts: np.ndarray) -> np.ndarray:
        """Calculate Poisson errors with Garwood intervals."""
        errors = np.zeros_like(counts, dtype=float)
        
        # Low statistics: use Garwood intervals
        low_mask = counts < 25
        if np.any(low_mask):
            try:
                from scipy import stats
                alpha = 0.32  # 68% CL
                
                low_counts = counts[low_mask]
                low_counts = np.maximum(low_counts, 0.1)
                
                lower = stats.chi2.ppf(alpha/2, 2*low_counts) / 2
                upper = stats.chi2.ppf(1-alpha/2, 2*(low_counts+1)) / 2
                
                errors[low_mask] = (upper - lower) / 2
            except ImportError:
                # Fallback if scipy not available
                errors[low_mask] = np.sqrt(np.maximum(counts[low_mask], 0))
        
        # High statistics: use sqrt(N)
        high_mask = ~low_mask
        errors[high_mask] = np.sqrt(np.maximum(counts[high_mask], 0))
        
        return errors
    
    def _calculate_ratio_with_errors(self, num: np.ndarray, num_err: np.ndarray,
                                   den: np.ndarray, den_err: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate ratio with proper error propagation."""
        ratio = np.ones_like(num)
        ratio_err = np.zeros_like(num)
        
        valid = den > 0
        
        if np.any(valid):
            ratio[valid] = num[valid] / den[valid]
            
            rel_num_err = np.zeros_like(num)
            rel_den_err = np.zeros_like(den)
            
            num_valid = (num > 0) & valid
            rel_num_err[num_valid] = num_err[num_valid] / num[num_valid]
            rel_den_err[valid] = den_err[valid] / den[valid]
            
            ratio_err[valid] = ratio[valid] * np.sqrt(
                rel_num_err[valid]**2 + rel_den_err[valid]**2
            )
        
        return ratio, ratio_err
    
    def _calculate_chi2(self, observed: np.ndarray, expected: np.ndarray,
                       obs_err: np.ndarray, exp_err: np.ndarray) -> Tuple[float, int, float]:
        """Calculate chi-square with combined errors."""
        combined_err = np.sqrt(obs_err**2 + exp_err**2)
        
        valid = (combined_err > 0) & (expected >= 0) & (observed >= 0)
        
        if not np.any(valid):
            return 0.0, 0, 1.0
        
        residuals = (observed - expected)[valid]
        errors = combined_err[valid]
        chi2 = np.sum((residuals / errors)**2)
        ndof = np.sum(valid) - 1
        
        try:
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(chi2, ndof) if ndof > 0 else 1.0
        except ImportError:
            p_value = 1.0  # Fallback if scipy not available
        
        return chi2, ndof, p_value
    
    def apply_cut_progression(self, 
                            variable: str,
                            bins: int = 50,
                            range: Tuple[float, float] = (0, 5),
                            user_cuts: List[str] = None) -> Dict[str, Any]:
        """Apply cut progression with memory management."""
        
        # Check if variable exists in any dataset
        if variable not in set().union(*self.available_columns.values()):
            print(f"❌ Variable '{variable}' not found in any dataset!")
            return {}
        
        # Define stages
        from dataclasses import dataclass
        
        @dataclass
        class CutStage:
            name: str
            description: str
            cuts: List[str]
            one_candidate: bool = False
        
        stages = [
            CutStage("No_Cuts", "Baseline distribution", []),
            CutStage("One_Candidate", "Best candidate per event", [], one_candidate=True)
        ]
        
        if user_cuts:
            # Filter cuts to only use available columns
            all_available = set().union(*self.available_columns.values())
            valid_cuts = []
            
            for cut in user_cuts:
                # Extract column names from cut
                cut_cols = re.findall(r'[a-zA-Z_]\w*', cut)
                cut_cols = [c for c in cut_cols if c not in ['abs', 'min', 'max']]
                
                # Check if all columns exist
                if all(col in all_available for col in cut_cols):
                    valid_cuts.append(cut)
                else:
                    missing = [c for c in cut_cols if c not in all_available]
                    print(f"   ⚠️  Skipping cut '{cut}' - missing columns: {missing}")
            
            if valid_cuts:
                stages.append(CutStage(
                    "Full_Cuts", 
                    "Valid user cuts + one candidate", 
                    valid_cuts, 
                    one_candidate=True
                ))
        
        results = {}
        
        for stage in stages:
            print(f"\n🔄 Processing: {stage.name}")
            
            # Check memory before processing
            mem_status = self.memory_manager.check_memory_status()
            print(f"   💾 Memory: {mem_status['used_gb']:.1f}GB used")
            
            # if mem_status['percent'] > 0.8:
            #     print("   🧹 High memory usage, forcing cleanup...")
            #     self.memory_manager.force_cleanup()
            
            try:
                # Apply transformations
                stage_data = self._apply_stage_cuts(stage)
                
                # Compute comparison
                result = self._compute_comparison(stage_data, variable, bins, range)
                results[stage.name] = result
                
                # Generate plot
                self._create_comparison_plot(result, variable, stage.name)
                
                # Report statistics
                if result['ndof'] > 0:
                    chi2_ndf = result['chi2'] / result['ndof']
                    print(f"   ✅ χ²/ndof = {chi2_ndf:.2f}, p-value = {result['p_value']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error in {stage.name}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Cleanup after each stage
            gc.collect()
        
        return results
    
    def _apply_stage_cuts(self, stage) -> OptimizedUltraLazyDict:
        """Apply cuts for a specific stage."""
        data = self.vpho_data
        
        # Apply user cuts if specified
        if stage.cuts:
            try:
                # Use framework with reduced operations
                data = self.framework.apply_cuts(data, stage.cuts)
            except Exception as e:
                print(f"   ⚠️  Failed to apply cuts: {e}")
        
        # Apply one candidate selection if specified
        if stage.one_candidate:
            try:
                data = self.framework.keep_one_candidate(
                    data,
                    group_cols=['__experiment__', '__run__', '__event__', '__production__']
                )
            except Exception as e:
                print(f"   ⚠️  Failed to select best candidates: {e}")
        
        return data

# ============================================================================
# MAIN EXECUTION WITH MEMORY SAFETY
# ============================================================================
def load_belle2_data_filtered(base_dir: str,
                            energy_condition: str,
                            file_type: str = 'vpho',
                            memory_budget_gb: float = 32.0) -> OptimizedUltraLazyDict:
    """
    Load Belle II data with energy condition filtering and validation.
    """
    print(f"\n📂 Loading {energy_condition} data...")
    
    # Load all data first
    all_data = load_belle2_5s_data_enhanced(
        base_dir=base_dir,
        file_type=file_type,
        memory_budget_gb=memory_budget_gb,
        parallel_loading=True
    )
    
    # Create filtered dictionary
    filtered_data = OptimizedUltraLazyDict(memory_budget_gb=memory_budget_gb)
    
    # Enhanced energy patterns
    energy_patterns = {
        '5S_scan': {
            'include': ['5s', 'mc5s', 'data5s'],
            'exclude': ['off']
        },
        '4S_on': {
            'include': ['4s', 'mc4s', 'data4s', 'mc_', 'data_'],
            'exclude': ['off', '5s']
        },
        '4S_offres': {
            'include': ['off', 'mcoff', 'dataoff'],
            'exclude': ['5s']
        }
    }
    
    patterns = energy_patterns.get(energy_condition, {'include': [], 'exclude': []})
    
    for name, df in all_data.items():
        name_lower = name.lower()
        
        # Check inclusion criteria
        include = any(p in name_lower for p in patterns['include']) if patterns['include'] else True
        exclude = any(p in name_lower for p in patterns['exclude']) if patterns['exclude'] else False
        
        if include and not exclude:
            filtered_data[name] = df
    
    # Restore group structure
    for group_name, members in all_data._groups.items():
        filtered_members = [m for m in members if m in filtered_data]
        if filtered_members:
            filtered_data.add_group(group_name, filtered_members)
    
    print(f"✅ Filtered to {len(filtered_data)} {energy_condition} datasets")
    return filtered_data

def run_belle2_analysis(base_dir: str,
                       energy_condition: str = '5S_scan',
                       variables: List[Tuple[str, int, Tuple[float, float]]] = None,
                       user_cuts: List[str] = None,
                       file_type: str = 'vpho',
                       output_dir: str = './mc_data_plots',
                       memory_budget_gb: float = 16.0,
                       return_data: bool = True):
    """
    Production-ready analysis with memory safety and full diagnostics.
    """
    
    print("🚀 Belle II MC-Data Comparison Analysis (Production Version - FIXED)")
    print("=" * 80)
    print(f"Energy condition: {energy_condition}")
    print(f"Memory budget: {memory_budget_gb}GB")
    print(f"File type: {file_type}")
    
    # Load data with reduced memory footprint
    vpho_data = load_belle2_data_filtered(
        base_dir=base_dir,
        energy_condition=energy_condition,
        file_type=file_type,
        memory_budget_gb=memory_budget_gb
    )
    
    if len(vpho_data) == 0:
        print(f"❌ No datasets found for {energy_condition}")
        return None if not return_data else (None, None, None)
    
    # Create engine with memory management
    engine = MCDataComparisonEngine(
        vpho_data,
        energy_condition=energy_condition,
        output_dir=output_dir,
        memory_budget_gb=memory_budget_gb
    )
    
    # Check data availability
    if not engine.data_processes:
        print("\n⚠️  WARNING: No data processes found!")
    
    # Run analysis with memory monitoring
    all_results = {}
    for var_name, bins, var_range in variables:
        print(f"\n{'='*60}")
        print(f"Analyzing variable: {var_name}")
        print(f"{'='*60}")
        
        results = engine.apply_cut_progression(
            variable=var_name,
            bins=bins,
            range=var_range,
            user_cuts=user_cuts
        )
        all_results[var_name] = results
        
        # Force cleanup between variables
        gc.collect()
    
    print("\n✅ Analysis complete!")
    
    # Final memory report
    mem_status = engine.memory_manager.check_memory_status()
    print(f"\n💾 Final memory usage: {mem_status['used_gb']:.1f}GB")
    
    if return_data:
        return all_results, vpho_data, engine
    else:
        return all_results

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    variables = [
        ('pRecoil', 100, (0.1, 6)),
        ('mu1P', 60, (0, 3)),
        ('mu2P', 60, (0, 3)),
    ]
    
    # Simplified cuts for testing
    simple_cuts = ['pRecoil>0.5', 'pRecoil<5.5']
    
    # Run with reduced memory budget
    results, data, engine = run_belle2_analysis(
        base_dir="/gpfs/group/belle2/users2022/kyldem/photoneff_updated/parquet_storage/try5",
        energy_condition='5S_scan',
        variables=variables,
        user_cuts=simple_cuts,
        file_type='vpho',
        output_dir='./belle2_production_plots',
        memory_budget_gb=16.0,  # Reduced for stability
        return_data=True
    )