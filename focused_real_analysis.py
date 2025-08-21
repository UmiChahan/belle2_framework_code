#!/usr/bin/env python3
"""
FOCUSED REAL ANALYSIS: Deep investigation of the actual execution discrepancy.
Uses real Belle II framework classes to identify root cause.

REQUIRES: source /cvmfs/belle.cern.ch/tools/b2setup light-2505-deimos
"""

import sys
import os
import polars as pl
import numpy as np
from pathlib import Path

# Import the framework
sys.path.insert(0, '/gpfs/group/belle2/users2022/kyldem/photoneff_updated/belle2_framework_code')

def analyze_measurement_vs_histogram_discrepancy():
    """
    CORE ANALYSIS: Why do measured_row_count() and hist() give different results
    on the SAME UnifiedLazyDataFrame after oneCandOnly?
    """
    
    print("🎯 FOCUSED ANALYSIS: Measurement vs Histogram Discrepancy")
    print("=" * 70)
    
    try:
        from layer2_unified_lazy_dataframe.core import UnifiedLazyDataFrame
        
        print("✅ Successfully imported UnifiedLazyDataFrame")
        
        # Create realistic Belle II data that mimics your actual issue
        print("\n📊 Creating Belle II data structure...")
        
        # Scenario 1: Normal multi-candidate data (should work)
        normal_data = []
        for event_id in range(1000):  # 1000 events
            for cand in range(5):     # 5 candidates each
                normal_data.append({
                    '__event__': event_id,
                    '__run__': 100,
                    '__experiment__': 31,
                    '__production__': 1,
                    'pRecoil': np.random.exponential(1.5) + 0.5
                })
        
        # Scenario 2: Event ID collision data (matches your issue)
        collision_data = []
        for i in range(5000):  # 5000 rows
            collision_data.append({
                '__event__': 1,  # ALL SAME EVENT ID
                '__run__': 100,
                '__experiment__': 31,
                '__production__': 1,
                'pRecoil': np.random.exponential(1.5) + 0.5
            })
        
        scenarios = [
            ("Normal Multi-Candidate", normal_data),
            ("Event ID Collision", collision_data)
        ]
        
        for scenario_name, data in scenarios:
            print(f"\n--- SCENARIO: {scenario_name} ---")
            df = pl.DataFrame(data)
            print(f"   Created {len(df)} rows")
            print(f"   Unique events: {df['__event__'].n_unique()}")
            
            # Create UnifiedLazyDataFrame
            udf = UnifiedLazyDataFrame(lazy_frames=[df.lazy()])
            
            # BEFORE oneCandOnly
            before_measured = udf.measured_row_count()
            try:
                before_hist_counts, _ = udf.hist('pRecoil', bins=10, range=(0.5, 5.0))
                before_hist_total = int(np.sum(before_hist_counts))
            except Exception as e:
                before_hist_total = f"ERROR: {e}"
            
            print(f"   BEFORE oneCandOnly:")
            print(f"     measured_row_count(): {before_measured}")
            print(f"     hist() total events:  {before_hist_total}")
            
            # APPLY oneCandOnly
            result_udf = udf.oneCandOnly()
            
            # AFTER oneCandOnly - THIS IS THE CRITICAL COMPARISON
            after_measured = result_udf.measured_row_count()
            try:
                after_hist_counts, _ = result_udf.hist('pRecoil', bins=10, range=(0.5, 5.0))
                after_hist_total = int(np.sum(after_hist_counts))
            except Exception as e:
                after_hist_total = f"ERROR: {e}"
            
            print(f"   AFTER oneCandOnly:")
            print(f"     measured_row_count(): {after_measured}")
            print(f"     hist() total events:  {after_hist_total}")
            
            # ANALYSIS
            if isinstance(after_hist_total, int) and isinstance(before_hist_total, int):
                if after_measured != after_hist_total:
                    print(f"   🚨 DISCREPANCY DETECTED!")
                    print(f"     Measured: {after_measured}")
                    print(f"     Histogram: {after_hist_total}")
                    print(f"     Ratio: {after_hist_total/after_measured if after_measured > 0 else 'inf':.2f}")
                    
                    # This would replicate your observed issue
                    if after_hist_total > after_measured * 1000:  # 1000x difference
                        print(f"   🎯 REPLICATES YOUR ISSUE: Histogram sees {after_hist_total/after_measured:.0f}x more events!")
                else:
                    print(f"   ✅ No discrepancy - both methods agree")
            
            print()
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_weighted_dataframe_delegation():
    """
    Analyze how WeightedDataFrame delegates oneCandOnly and measurement calls.
    """
    
    print("🎯 FOCUSED ANALYSIS: WeightedDataFrame Delegation")
    print("=" * 70)
    
    try:
        from layer2_unified_lazy_dataframe.core import UnifiedLazyDataFrame
        from belle2_enhanced_framework import WeightedDataFrame
        
        print("✅ Successfully imported both classes")
        
        # Create test data
        data = []
        for event_id in range(100):
            for cand in range(10):  # 10 candidates per event
                data.append({
                    '__event__': event_id if event_id < 50 else 1,  # Second half all same event
                    '__run__': 100,
                    'pRecoil': np.random.exponential(1.5) + 0.5
                })
        
        df = pl.DataFrame(data)
        udf = UnifiedLazyDataFrame(lazy_frames=[df.lazy()])
        
        # Create WeightedDataFrame (this is what your framework actually uses)
        class MockWeight:
            process_name = "test_process"
        
        weighted_df = WeightedDataFrame(udf, MockWeight(), None)
        
        print(f"Created WeightedDataFrame with {len(data)} rows")
        print(f"Unique events in data: {df['__event__'].n_unique()}")
        
        # Test delegation
        print(f"\n--- TESTING WeightedDataFrame DELEGATION ---")
        
        # Before oneCandOnly
        before_measured = weighted_df.measured_row_count()
        print(f"Before oneCandOnly - measured: {before_measured}")
        
        # Apply oneCandOnly through WeightedDataFrame
        weighted_candidates = weighted_df.oneCandOnly()
        
        # After oneCandOnly  
        after_measured = weighted_candidates.measured_row_count()
        print(f"After oneCandOnly - measured: {after_measured}")
        
        # Test histogram through WeightedDataFrame
        try:
            hist_counts, _ = weighted_candidates.hist('pRecoil', bins=10, range=(0.5, 5.0))
            hist_total = int(np.sum(hist_counts))
            print(f"After oneCandOnly - histogram: {hist_total}")
            
            if after_measured != hist_total:
                print(f"🚨 DISCREPANCY IN WeightedDataFrame!")
                print(f"   measured_row_count(): {after_measured}")
                print(f"   hist() total:         {hist_total}")
                print(f"   Difference factor:    {hist_total/after_measured if after_measured > 0 else 'inf'}")
            else:
                print(f"✅ WeightedDataFrame delegation is consistent")
                
        except Exception as e:
            print(f"❌ Histogram failed: {e}")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")

def analyze_extract_lazy_frames_paths():
    """
    Deep dive into _extract_lazy_frames_safely() execution paths.
    This is likely where the discrepancy occurs.
    """
    
    print("🎯 FOCUSED ANALYSIS: _extract_lazy_frames_safely Execution Paths")
    print("=" * 70)
    
    try:
        from layer2_unified_lazy_dataframe.core import UnifiedLazyDataFrame
        
        # Create test data
        data = pl.DataFrame({
            '__event__': [1, 1, 2, 2],
            '__run__': [100, 100, 100, 100],
            'pRecoil': [1.0, 2.0, 1.5, 2.5]
        })
        
        udf = UnifiedLazyDataFrame(lazy_frames=[data.lazy()])
        result_udf = udf.oneCandOnly()
        
        print("Created UDF and applied oneCandOnly")
        
        # Test different extraction methods
        print(f"\n--- TESTING FRAME EXTRACTION METHODS ---")
        
        # Method 1: Direct _extract_lazy_frames_safely
        try:
            frames = result_udf._extract_lazy_frames_safely()
            print(f"_extract_lazy_frames_safely returned {len(frames)} frames")
            
            if frames:
                # Count rows in extracted frames
                total_rows = 0
                for i, frame in enumerate(frames):
                    try:
                        count = len(frame.collect())
                        total_rows += count
                        print(f"   Frame {i}: {count} rows")
                    except Exception as e:
                        print(f"   Frame {i}: ERROR {e}")
                
                print(f"   Total rows in extracted frames: {total_rows}")
            
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
        
        # Method 2: Check what measured_row_count uses
        measured = result_udf.measured_row_count()
        print(f"measured_row_count() result: {measured}")
        
        # Method 3: Check transformation chain
        if hasattr(result_udf, '_transformation_chain'):
            lineage = result_udf._transformation_chain.get_lineage()
            print(f"Transformation chain has {len(lineage)} transforms")
            for i, transform in enumerate(lineage):
                print(f"   Transform {i}: {transform.operation}")
        
        # Method 4: Check compute graph
        if hasattr(result_udf, '_compute'):
            print(f"Has compute graph: {result_udf._compute is not None}")
            if hasattr(result_udf._compute, 'to_lazy_frames'):
                print(f"Compute has to_lazy_frames method: {hasattr(result_udf._compute, 'to_lazy_frames')}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    print("FOCUSED REAL FRAMEWORK ANALYSIS")
    print("Using Belle II environment and actual classes")
    print("=" * 70)
    
    analyze_measurement_vs_histogram_discrepancy()
    analyze_weighted_dataframe_delegation()
    analyze_extract_lazy_frames_paths()
    
    print("\n" + "=" * 70)
    print("FOCUSED ANALYSIS COMPLETE")
    print("This should identify the exact execution path differences.")
    print("=" * 70)