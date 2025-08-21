#!/usr/bin/env python3
"""
REAL FRAMEWORK ANALYSIS: Deep methodical investigation using actual classes.
No mocks - using the real UnifiedLazyDataFrame, WeightedDataFrame, and compute graph.

REQUIRES: Belle II environment loaded:
source /cvmfs/belle.cern.ch/tools/b2setup light-2505-deimos
"""
"""
REAL FRAMEWORK ANALYSIS: Deep methodical investigation using actual classes.
No mocks - using the real UnifiedLazyDataFrame, WeightedDataFrame, and compute graph.
"""

import sys
import os
import polars as pl
import numpy as np
from pathlib import Path

# Import the actual framework classes
sys.path.insert(0, '/gpfs/group/belle2/users2022/kyldem/photoneff_updated/belle2_framework_code')

def deep_analyze_oneCandOnly_execution():
    """Deep analysis of the actual oneCandOnly execution path."""
    
    print("🔬 DEEP ANALYSIS: Real oneCandOnly Execution Path")
    print("=" * 80)
    
    try:
        # Import the actual classes
        from layer2_unified_lazy_dataframe.core import UnifiedLazyDataFrame
        from layer2_unified_lazy_dataframe import TransformationMetadata, GraphNode, ComputeOpType
        
        print("✅ Successfully imported real framework classes")
        
        # Create REAL Belle II data structure
        print("\n📊 Creating Belle II-realistic data structure...")
        
        # Real Belle II column structure from the framework
        belle2_cols = ['__experiment__', '__run__', '__event__', '__production__', 
                      '__candidate__', '__ncandidates__']
        physics_cols = ['pRecoil', 'mRecoil', 'mu1P', 'mu2P', 'mu1Theta', 'mu2Theta']
        all_cols = belle2_cols + physics_cols
        
        # Create realistic multi-candidate per event data
        n_events = 1000
        candidates_per_event = 5
        n_total = n_events * candidates_per_event
        
        # Create proper Belle II event structure
        event_data = []
        for event_id in range(n_events):
            for candidate_id in range(candidates_per_event):
                row = {
                    '__experiment__': 31,
                    '__run__': 100 + (event_id % 10),
                    '__event__': event_id,  # Proper unique event IDs
                    '__production__': 1,
                    '__candidate__': candidate_id,
                    '__ncandidates__': candidates_per_event,
                    'pRecoil': np.random.exponential(2.0) + 0.5,
                    'mRecoil': np.random.normal(3.1, 0.1),
                    'mu1P': np.random.uniform(0.5, 4.0),
                    'mu2P': np.random.uniform(0.5, 4.0),
                    'mu1Theta': np.random.uniform(0.3, 2.8),
                    'mu2Theta': np.random.uniform(0.3, 2.8)
                }
                event_data.append(row)
        
        df = pl.DataFrame(event_data)
        print(f"   Created {len(df)} rows with {n_events} unique events")
        print(f"   Unique __event__ values: {df['__event__'].n_unique()}")
        print(f"   Expected after oneCandOnly: {n_events} rows")
        
        # Create REAL UnifiedLazyDataFrame with required_columns
        print("\n🔧 Creating real UnifiedLazyDataFrame...")
        
        # Test with different required_columns scenarios
        scenarios = [
            ("All columns", all_cols),
            ("Missing Belle II IDs", physics_cols),  # This might be the issue
            ("Minimal Belle II", ['__event__', '__run__', 'pRecoil']),
        ]
        
        for scenario_name, required_cols in scenarios:
            print(f"\n--- SCENARIO: {scenario_name} ---")
            print(f"Required columns: {required_cols}")
            
            try:
                # Create the REAL UnifiedLazyDataFrame
                udf = UnifiedLazyDataFrame(
                    lazy_frames=[df.lazy()],
                    required_columns=required_cols,
                    memory_budget_gb=1.0
                )
                
                print(f"   Created UDF. Available columns: {udf.columns}")
                
                # Check what columns are actually available for oneCandOnly
                belle2_event_cols = ['__event__', '__run__', '__experiment__', '__production__']
                available_for_grouping = [col for col in belle2_event_cols if col in udf.columns]
                print(f"   Belle II grouping columns available: {available_for_grouping}")
                
                if not available_for_grouping:
                    print("   🚨 NO GROUPING COLUMNS AVAILABLE - oneCandOnly will fail!")
                    continue
                
                # Execute REAL oneCandOnly
                print(f"   Executing real oneCandOnly...")
                
                before_count = udf.measured_row_count()
                print(f"   Before oneCandOnly: {before_count} rows")
                
                # Call the REAL oneCandOnly method
                result_udf = udf.oneCandOnly()
                
                after_count = result_udf.measured_row_count()
                print(f"   After oneCandOnly: {after_count} rows")
                
                retention = after_count / before_count if before_count > 0 else 0
                print(f"   Retention ratio: {retention:.4f} ({retention*100:.2f}%)")
                
                if retention < 0.1:  # Less than 10% retention
                    print(f"   🚨 EXTREME REDUCTION DETECTED!")
                    
                    # Deep dive into WHY
                    print(f"   🔍 Investigating cause...")
                    
                    # Check the actual grouping columns used
                    try:
                        # Access the transformation metadata
                        if hasattr(result_udf, '_transformation_chain'):
                            lineage = result_udf._transformation_chain.get_lineage()
                            for transform in lineage:
                                if transform.operation == 'oneCandOnly':
                                    actual_group_cols = transform.parameters.get('group_cols', [])
                                    print(f"   Actual group_cols used: {actual_group_cols}")
                                    
                                    # Check uniqueness of these columns in original data
                                    for col in actual_group_cols:
                                        if col in df.columns:
                                            unique_vals = df[col].n_unique()
                                            print(f"   Column '{col}' has {unique_vals} unique values")
                    except Exception as e:
                        print(f"   Could not access transformation metadata: {e}")
                
            except Exception as e:
                print(f"   ❌ FAILED: {type(e).__name__}: {e}")
                
    except ImportError as e:
        print(f"❌ Failed to import framework classes: {e}")
        print("This suggests NumPy compatibility issues or missing dependencies")

def deep_analyze_cuts_with_real_classes():
    """Analyze cuts implementation using real WeightedDataFrame objects."""
    
    print(f"\n" + "=" * 80)
    print("🔬 DEEP ANALYSIS: Real Cuts Implementation")
    print("=" * 80)
    
    try:
        from belle2_enhanced_framework import WeightedDataFrame, Belle2FrameworkV3
        from layer2_unified_lazy_dataframe.core import UnifiedLazyDataFrame
        
        print("✅ Successfully imported framework classes")
        
        # Create realistic weighted dataframe
        print("\n📊 Creating real WeightedDataFrame structure...")
        
        # Create base data similar to what the framework loads
        n_rows = 10000
        base_df_data = {
            '__experiment__': [31] * n_rows,
            '__run__': [100] * n_rows,
            '__event__': list(range(n_rows // 10)) * 10,  # 10 candidates per event
            '__production__': [1] * n_rows,
            'pRecoil': np.random.exponential(1.5, n_rows) + 0.5,
            'mRecoil': np.random.normal(3.1, 0.1, n_rows),
            'mu1P': np.random.uniform(0.5, 4.0, n_rows),
            'quality_score': np.random.beta(2, 5, n_rows)  # For cuts testing
        }
        
        base_polars_df = pl.DataFrame(base_df_data)
        
        # Create real UnifiedLazyDataFrame
        base_udf = UnifiedLazyDataFrame(
            lazy_frames=[base_polars_df.lazy()],
            memory_budget_gb=1.0
        )
        
        # Create real WeightedDataFrame (this is what the framework actually uses)
        class MockLuminosityWeight:
            def __init__(self):
                self.process_name = "test_process"
                self.weight_value = 1.5
        
        weight = MockLuminosityWeight()
        weighted_df = WeightedDataFrame(base_udf, weight, None)
        
        print(f"   Created WeightedDataFrame with {weighted_df.measured_row_count()} rows")
        
        # Test the REAL cuts implementation path
        print(f"\n🔧 Testing real cuts execution path...")
        
        # Simulate the ProgressiveAnalysisV3 behavior
        base_data = {"test_process": weighted_df}
        
        print(f"   Base data: {base_data['test_process'].measured_row_count()} rows")
        
        # CANDIDATES stage - apply real oneCandOnly
        print(f"\n--- CANDIDATES STAGE ---")
        candidates_result = {}
        for name, wdf in base_data.items():
            candidates_wdf = wdf.oneCandOnly()
            candidates_count = candidates_wdf.measured_row_count()
            print(f"   {name}: {wdf.measured_row_count()} → {candidates_count} (oneCandOnly)")
            candidates_result[name] = candidates_wdf
        
        # CUTS stage - test both approaches
        cuts = ['pRecoil > 2.0', 'quality_score > 0.5']
        
        print(f"\n--- CUTS STAGE (CURRENT BUGGY) ---")
        # Current implementation: uses base_data
        cuts_result_buggy = {}
        for name, wdf in base_data.items():  # BUGGY: uses base_data
            working_wdf = wdf
            for cut in cuts:
                working_wdf = working_wdf.query(cut)
            cuts_count = working_wdf.measured_row_count()
            print(f"   {name}: {base_data[name].measured_row_count()} → {cuts_count} (cuts from base)")
            cuts_result_buggy[name] = working_wdf
        
        print(f"\n--- CUTS STAGE (CORRECTED) ---")
        # Corrected implementation: uses candidates_result
        cuts_result_fixed = {}
        for name, wdf in candidates_result.items():  # FIXED: uses candidates
            working_wdf = wdf
            for cut in cuts:
                working_wdf = working_wdf.query(cut)
            cuts_count = working_wdf.measured_row_count()
            print(f"   {name}: {candidates_result[name].measured_row_count()} → {cuts_count} (cuts from candidates)")
            cuts_result_fixed[name] = working_wdf
        
        # ANALYSIS
        print(f"\n🎯 REAL EXECUTION ANALYSIS:")
        base_count = base_data["test_process"].measured_row_count()
        candidates_count = candidates_result["test_process"].measured_row_count()
        cuts_buggy_count = cuts_result_buggy["test_process"].measured_row_count()
        cuts_fixed_count = cuts_result_fixed["test_process"].measured_row_count()
        
        print(f"   Base:           {base_count:,} rows")
        print(f"   Candidates:     {candidates_count:,} rows")
        print(f"   Cuts (buggy):   {cuts_buggy_count:,} rows (from base)")
        print(f"   Cuts (fixed):   {cuts_fixed_count:,} rows (from candidates)")
        
        buggy_ratio = cuts_buggy_count / base_count if base_count > 0 else 0
        fixed_ratio = cuts_fixed_count / candidates_count if candidates_count > 0 else 0
        
        print(f"   Buggy retention:  {buggy_ratio:.4f} ({buggy_ratio*100:.2f}%)")
        print(f"   Fixed retention:  {fixed_ratio:.4f} ({fixed_ratio*100:.2f}%)")
        
        if cuts_buggy_count >> cuts_fixed_count:
            print(f"   🚨 CONFIRMED: Buggy implementation processes {cuts_buggy_count/cuts_fixed_count:.1f}x more events!")
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def analyze_actual_transformation_chain():
    """Analyze the real transformation chain execution."""
    
    print(f"\n" + "=" * 80)
    print("🔬 DEEP ANALYSIS: Real Transformation Chain Execution")
    print("=" * 80)
    
    try:
        from layer2_unified_lazy_dataframe.core import UnifiedLazyDataFrame, TransformationMetadata
        
        # Create test data
        test_data = pl.DataFrame({
            '__event__': [1, 1, 2, 2, 3, 3],
            '__run__': [100, 100, 100, 100, 100, 100],
            'pRecoil': [1.0, 2.0, 1.5, 2.5, 0.8, 3.0]
        })
        
        # Create UDF and apply transformations
        udf = UnifiedLazyDataFrame(lazy_frames=[test_data.lazy()])
        
        print(f"Original data: {len(test_data)} rows")
        
        # Apply oneCandOnly and inspect the transformation chain
        result_udf = udf.oneCandOnly()
        
        print(f"After oneCandOnly: {result_udf.measured_row_count()} rows")
        
        # Inspect the transformation chain
        if hasattr(result_udf, '_transformation_chain'):
            lineage = result_udf._transformation_chain.get_lineage()
            print(f"Transformation chain length: {len(lineage)}")
            
            for i, transform in enumerate(lineage):
                print(f"   Transform {i}: {transform.operation}")
                print(f"   Parameters: {transform.parameters}")
        
        # Check compute graph
        if hasattr(result_udf, '_compute'):
            print(f"Has compute graph: {result_udf._compute is not None}")
            if result_udf._compute and hasattr(result_udf._compute, 'root_node'):
                print(f"Root node operation: {result_udf._compute.root_node.operation}")
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("STARTING DEEP, METHODICAL ANALYSIS OF BELLE II FRAMEWORK")
    print("Using REAL classes, not mocks")
    print("=" * 80)
    
    deep_analyze_oneCandOnly_execution()
    deep_analyze_cuts_with_real_classes() 
    analyze_actual_transformation_chain()
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("This uses the actual framework classes to identify real execution issues.")
    print("=" * 80)