# belle2_data_validation.py
"""
Comprehensive Validation Framework for Belle II Synthetic Data

This framework implements multi-layered validation strategies to ensure
synthetic data maintains structural, statistical, and relational integrity
with production Belle II datasets.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from scipy import stats
import json

class DataValidationFramework:
    """
    Hierarchical validation system operating at three levels:
    1. Structural Validation - Schema compliance and data types
    2. Statistical Validation - Distribution characteristics  
    3. Relational Validation - Inter-variable correlations
    """
    
    def __init__(self):
        self.validation_results = {}
        self.expected_columns = set([
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
        ])
        
    def validate_dataset(self, data_path: Path) -> Dict[str, Any]:
        """
        Execute comprehensive validation protocol.
        
        Returns hierarchical validation report with actionable insights.
        """
        print(f"\n🔍 Validating dataset: {data_path}")
        
        # Load data
        df = pl.read_parquet(data_path)
        
        # Execute validation hierarchy
        structural_results = self._validate_structure(df)
        statistical_results = self._validate_statistics(df)
        relational_results = self._validate_relationships(df)
        
        # Synthesize results
        validation_report = {
            'file': str(data_path),
            'structural': structural_results,
            'statistical': statistical_results,
            'relational': relational_results,
            'overall_score': self._calculate_validation_score(
                structural_results, statistical_results, relational_results
            )
        }
        
        return validation_report
    
    def _validate_structure(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Level 1: Structural validation examining schema compliance."""
        
        results = {
            'schema_compliance': True,
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': {},
            'null_analysis': {}
        }
        
        # Column presence analysis
        actual_columns = set(df.columns)
        results['missing_columns'] = list(self.expected_columns - actual_columns)
        results['extra_columns'] = list(actual_columns - self.expected_columns)
        
        if results['missing_columns']:
            results['schema_compliance'] = False
        
        # Type validation
        expected_types = {
            '__experiment__': pl.Int32,
            '__run__': pl.Int32,
            '__event__': pl.Int32,
            'pRecoil': pl.Float64,
            'mu1P': pl.Float64,
            'mu1nCDCHits': pl.Int32
        }
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if actual_type != expected_type:
                    results['type_mismatches'][col] = {
                        'expected': str(expected_type),
                        'actual': str(actual_type)
                    }
        
        # Null value analysis
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                results['null_analysis'][col] = {
                    'count': null_count,
                    'percentage': (null_count / len(df)) * 100
                }
        
        return results
    
    def _validate_statistics(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Level 2: Statistical validation of distribution characteristics."""
        
        results = {
            'distribution_tests': {},
            'range_violations': {},
            'moment_analysis': {}
        }
        
        # Critical variable validation
        critical_variables = {
            'pRecoil': {'range': (0.1, 6.0), 'expected_mean': 2.5, 'expected_std': 0.8},
            'mu1P': {'range': (0.0, 3.0), 'expected_mean': 1.2, 'expected_std': 0.5},
            'mu2P': {'range': (0.0, 3.0), 'expected_mean': 1.2, 'expected_std': 0.5},
            'pRecoilTheta': {'range': (0.0, np.pi), 'expected_mean': np.pi/2, 'expected_std': 0.5},
            'mu1nCDCHits': {'range': (0, 100), 'expected_mean': 40, 'expected_std': 5}
        }
        
        for var, specs in critical_variables.items():
            if var not in df.columns:
                continue
                
            values = df[var].to_numpy()
            
            # Range validation
            violations = np.sum((values < specs['range'][0]) | (values > specs['range'][1]))
            if violations > 0:
                results['range_violations'][var] = {
                    'count': int(violations),
                    'percentage': (violations / len(values)) * 100
                }
            
            # Moment analysis
            results['moment_analysis'][var] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'skewness': float(stats.skew(values)),
                'kurtosis': float(stats.kurtosis(values)),
                'expected_mean': specs['expected_mean'],
                'expected_std': specs['expected_std']
            }
            
            # Distribution test (Kolmogorov-Smirnov)
            if var in ['pRecoil', 'mu1P', 'mu2P']:
                # Test against expected normal distribution
                ks_stat, p_value = stats.kstest(
                    values, 
                    'norm', 
                    args=(specs['expected_mean'], specs['expected_std'])
                )
                results['distribution_tests'][var] = {
                    'test': 'Kolmogorov-Smirnov',
                    'statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
        
        return results
    
    def _validate_relationships(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Level 3: Relational validation of inter-variable correlations."""
        
        results = {
            'correlation_analysis': {},
            'physics_constraints': {},
            'anomaly_detection': {}
        }
        
        # Critical correlations
        correlation_pairs = [
            ('pRecoil', 'totalMuonMomentum'),
            ('mu1P', 'mu2P'),
            ('mu1Theta', 'mu2Theta'),
            ('eRecoil', 'pRecoil')
        ]
        
        for var1, var2 in correlation_pairs:
            if var1 in df.columns and var2 in df.columns:
                corr = np.corrcoef(df[var1].to_numpy(), df[var2].to_numpy())[0, 1]
                results['correlation_analysis'][f'{var1}-{var2}'] = float(corr)
        
        # Physics constraint validation
        if all(col in df.columns for col in ['eRecoil', 'pRecoil', 'mRecoil']):
            # E² = p² + m²
            e_vals = df['eRecoil'].to_numpy()
            p_vals = df['pRecoil'].to_numpy()
            m_vals = df['mRecoil'].to_numpy()
            
            constraint_violation = np.abs(e_vals**2 - p_vals**2 - m_vals**2)
            results['physics_constraints']['energy_momentum'] = {
                'mean_violation': float(np.mean(constraint_violation)),
                'max_violation': float(np.max(constraint_violation)),
                'violations_above_tolerance': int(np.sum(constraint_violation > 0.01))
            }
        
        # Angular constraint validation
        if all(col in df.columns for col in ['mu1Theta', 'mu2Theta', 'pRecoilTheta']):
            # Detector acceptance
            for var in ['mu1Theta', 'mu2Theta', 'pRecoilTheta']:
                values = df[var].to_numpy()
                out_of_acceptance = np.sum((values < 0.3) | (values > 2.84))
                if out_of_acceptance > 0:
                    results['anomaly_detection'][f'{var}_acceptance'] = {
                        'count': int(out_of_acceptance),
                        'percentage': (out_of_acceptance / len(values)) * 100
                    }
        
        return results
    
    def _calculate_validation_score(self, structural: Dict, 
                                   statistical: Dict, 
                                   relational: Dict) -> float:
        """Synthesize multi-level validation into unified score."""
        
        # Weighted scoring system
        score = 100.0
        
        # Structural penalties
        score -= len(structural['missing_columns']) * 5
        score -= len(structural['type_mismatches']) * 2
        
        # Statistical penalties
        for var, violation in statistical['range_violations'].items():
            score -= min(violation['percentage'], 10)
        
        # Relational penalties
        if 'energy_momentum' in relational['physics_constraints']:
            violations = relational['physics_constraints']['energy_momentum']['violations_above_tolerance']
            score -= min(violations / 100, 10)
        
        return max(0, score)
    
    def generate_validation_report(self, results: Dict[str, Any], 
                                  output_path: Optional[Path] = None) -> str:
        """Generate comprehensive validation report with visualizations."""
        
        report = []
        report.append("=" * 60)
        report.append("BELLE II SYNTHETIC DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"\nFile: {results['file']}")
        report.append(f"Overall Score: {results['overall_score']:.1f}/100")
        
        # Structural Analysis
        report.append("\n## STRUCTURAL VALIDATION")
        report.append("-" * 40)
        
        structural = results['structural']
        if structural['schema_compliance']:
            report.append("✓ Schema compliance: PASSED")
        else:
            report.append("✗ Schema compliance: FAILED")
            if structural['missing_columns']:
                report.append(f"  Missing columns: {', '.join(structural['missing_columns'])}")
        
        # Statistical Analysis
        report.append("\n## STATISTICAL VALIDATION")
        report.append("-" * 40)
        
        statistical = results['statistical']
        for var, moments in statistical['moment_analysis'].items():
            deviation = abs(moments['mean'] - moments['expected_mean']) / moments['expected_std']
            status = "✓" if deviation < 2 else "⚠"
            report.append(f"{status} {var}: mean={moments['mean']:.3f} "
                         f"(expected={moments['expected_mean']:.3f})")
        
        # Relational Analysis
        report.append("\n## RELATIONAL VALIDATION")
        report.append("-" * 40)
        
        relational = results['relational']
        for pair, corr in relational['correlation_analysis'].items():
            report.append(f"  Correlation {pair}: {corr:.3f}")
        
        report_text = "\n".join(report)
        
        if output_path:
            output_path.write_text(report_text)
        
        return report_text


# Integration utilities
class DataGenerationIntegration:
    """Seamless integration with Belle II analysis framework."""
    
    @staticmethod
    def create_test_environment(base_dir: str = './integrated_test') -> Dict[str, Any]:
        """Create complete test environment with data and configuration."""
        
        from streamlined_belle2_generator import QuickBelle2Generator
        
        # Create directory structure
        base_path = Path(base_dir)
        data_path = base_path / 'data'
        config_path = base_path / 'config'
        
        for path in [data_path, config_path]:
            path.mkdir(exist_ok=True, parents=True)
        
        # Generate test data
        generator = QuickBelle2Generator(str(data_path))
        files = generator.generate_standard_files(scale=0.01)  # 10k events per process
        
        # Create configuration
        config = {
            'data_path': str(data_path),
            'energy_condition': '4S_offres',
            'memory_budget_gb': 4.0,
            'test_cuts': [
                'pRecoil > 0.5',
                'mu1nCDCHits > 10',
                'mu2nCDCHits > 10'
            ]
        }
        
        config_file = config_path / 'test_config.json'
        config_file.write_text(json.dumps(config, indent=2))
        
        # Validate generated data
        validator = DataValidationFramework()
        validation_results = {}
        
        for name, filepath in files.items():
            if filepath.exists():
                validation_results[name] = validator.validate_dataset(filepath)
        
        return {
            'files': files,
            'config': config,
            'validation': validation_results,
            'paths': {
                'base': base_path,
                'data': data_path,
                'config': config_path
            }
        }


# Example workflow
if __name__ == '__main__':
    # Create integrated test environment
    env = DataGenerationIntegration.create_test_environment()
    
    print("\n📊 Generated Test Environment Summary")
    print("=" * 50)
    print(f"Data files: {len(env['files'])}")
    print(f"Base directory: {env['paths']['base']}")
    
    # Show validation summary
    print("\n📋 Validation Summary:")
    for process, validation in env['validation'].items():
        score = validation['overall_score']
        status = "✅" if score > 90 else "⚠️"
        print(f"  {status} {process}: {score:.1f}/100")