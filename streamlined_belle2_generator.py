# streamlined_belle2_generator.py
"""
Streamlined generator focusing on essential functionality
for rapid development and testing.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional
import time

class QuickBelle2Generator:
    """Minimal generator for rapid prototyping."""
    
    # Essential columns for vpho analysis
    VPHO_COLUMNS = [
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
    ]
    
    def __init__(self, output_dir: str = './quick_synthetic_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_process(self, 
                        process: str, 
                        n_events: int,
                        resonance: str = 'prompt') -> pl.DataFrame:
        """Generate data for a single process with essential correlations."""
        
        # Core event structure
        data = {
            '__experiment__': np.full(n_events, 26),
            '__run__': np.random.randint(1, 1000, n_events),
            '__event__': np.arange(n_events),
            '__production__': np.ones(n_events, dtype=int),
            '__candidate__': np.zeros(n_events, dtype=int),
            '__ncandidates__': np.random.poisson(1.2, n_events) + 1
        }
        
        # Generate correlated physics variables efficiently
        self._add_physics_variables(data, process, n_events)
        
        return pl.DataFrame(data)
    
    def _add_physics_variables(self, data: Dict, process: str, n: int):
        """Add physics variables with minimal but realistic correlations."""
        
        # Primary observables
        data['pRecoil'] = np.clip(np.random.gamma(5, 0.5, n), 0.1, 6.0)  # Positive, skewed
        data['pRecoilTheta'] = np.random.uniform(0.3, 2.84, n)
        data['pRecoilPhi'] = np.random.uniform(-np.pi, np.pi, n)
        
        # Correlated muon variables
        p_scale = data['pRecoil'] * 0.4
        data['mu1P'] = np.abs(np.random.normal(p_scale, 0.3, n))
        data['mu2P'] = np.abs(np.random.normal(p_scale, 0.3, n))
        
        # Angular correlations
        data['mu1Theta'] = np.clip(data['pRecoilTheta'] + 0.1*np.random.randn(n), 0.3, 2.84)
        data['mu2Theta'] = np.clip(data['pRecoilTheta'] + 0.1*np.random.randn(n), 0.3, 2.84)
        data['mu1Phi'] = data['pRecoilPhi'] + np.random.normal(0, 0.3, n)
        data['mu2Phi'] = data['pRecoilPhi'] + np.random.normal(0, 0.3, n)
        
        # Clip angles to physical ranges
        data['mu1Theta'] = np.clip(data['mu1Theta'], 0.3, 2.84)
        data['mu2Theta'] = np.clip(data['mu2Theta'], 0.3, 2.84)
        
        # Energy-momentum relations
        m_mu = 0.105  # GeV
        data['mu1E'] = np.sqrt(data['mu1P']**2 + m_mu**2)
        data['mu2E'] = np.sqrt(data['mu2P']**2 + m_mu**2)
        data['mRecoil'] = np.random.exponential(0.1, n)
        data['eRecoil'] = np.sqrt(data['pRecoil']**2 + data['mRecoil']**2)
        data['m2Recoil'] = data['mRecoil']**2
        
        # Calorimeter variables (anti-correlated with momentum for muons)
        data['mu1clusterE'] = 0.3 * np.exp(-data['mu1P']/2) + np.random.exponential(0.05, n)
        data['mu2clusterE'] = 0.3 * np.exp(-data['mu2P']/2) + np.random.exponential(0.05, n)
        data['mu1clusterEoP'] = data['mu1clusterE'] / (data['mu1P'] + 0.01)
        data['mu2clusterEoP'] = data['mu2clusterE'] / (data['mu2P'] + 0.01)
        
        # Cluster positions
        for mu in ['mu1', 'mu2']:
            data[f'{mu}clusterTheta'] = data[f'{mu}Theta']
            data[f'{mu}clusterPhi'] = data[f'{mu}Phi']
        
        # Quality variables
        data['mu1nCDCHits'] = np.random.poisson(40, n)
        data['mu2nCDCHits'] = np.random.poisson(40, n)
        
        # Event shape
        mult = 5 if 'data' not in process else 4
        data['nTracks'] = np.random.poisson(mult, n) + 2
        data['nTracksROE'] = data['nTracks'] - 2
        data['nGammaROE'] = np.random.poisson(3, n)
        data['nPhotonCands'] = np.random.poisson(2.5, n)
        
        # Energy sums
        data['sumE_offcone'] = np.random.exponential(0.5, n)
        data['sumE_offcone_barrel'] = data['sumE_offcone'] * 0.7
        
        # Virtual photon
        data['vpho_px'] = np.random.normal(0, 0.5, n)
        data['vpho_py'] = np.random.normal(0, 0.5, n)
        data['vpho_pz'] = np.random.normal(0, 1.0, n)
        data['vpho_E'] = np.sqrt(data['vpho_px']**2 + data['vpho_py']**2 + data['vpho_pz']**2)
        
        # Global event variables
        data['theta'] = np.mean([data['mu1Theta'], data['mu2Theta']], axis=0)
        data['phi'] = np.mean([data['mu1Phi'], data['mu2Phi']], axis=0)
        data['E'] = data['mu1E'] + data['mu2E']
        data['beamE'] = np.full(n, 5.29)
        data['totalMuonMomentum'] = data['mu1P'] + data['mu2P']
        data['psnm_ffo'] = np.random.beta(10, 2, n)
    
    def generate_standard_files(self, scale: float = 0.01) -> Dict[str, Path]:
        """Generate standard file set with given scale (0.01 = 10k events)."""
        
        processes = {
            'mumu': int(1_000_000 * scale),
            'ee': int(500_000 * scale),
            'taupair': int(100_000 * scale),
            'uubar': int(200_000 * scale),
            'ddbar': int(200_000 * scale),
            'ssbar': int(200_000 * scale),
            'ccbar': int(200_000 * scale),
            'data': int(500_000 * scale)
        }
        
        files = {}
        
        for resonance in ['prompt', 'offprompt']:
            for process, n_events in processes.items():
                # Generate data
                df = self.generate_process(process, n_events, resonance)
                
                # Determine filename
                if process == 'data':
                    filename = f"data-{resonance}_selected.parquet"
                else:
                    filename = f"mc16-{process}-{resonance}_selected.parquet"
                
                # Write file
                filepath = self.output_dir / filename
                df.write_parquet(filepath)
                files[f"{process}-{resonance}"] = filepath
                
                print(f"✓ Generated {filename}: {n_events:,} events")
        
        return files


# Ultra-fast generation for testing
def generate_minimal_test_data(output_dir: str = './minimal_test'):
    """Generate absolute minimum dataset for framework testing."""
    
    generator = QuickBelle2Generator(output_dir)
    
    # Just 1000 events per file for rapid testing
    test_processes = ['mumu', 'data']
    files = {}
    
    for process in test_processes:
        for resonance in ['prompt', 'offprompt']:
            df = generator.generate_process(process, 1000, resonance)
            
            if process == 'data':
                filename = f"data-{resonance}_selected.parquet"
            else:
                filename = f"mc16-{process}-{resonance}_selected.parquet"
            
            filepath = generator.output_dir / filename
            df.write_parquet(filepath)
            files[f"{process}-{resonance}"] = filepath
    
    return files


# Memory-efficient streaming generator for billion-row datasets
class StreamingBelle2Generator:
    """Memory-efficient generator for massive datasets."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.generator = QuickBelle2Generator(output_dir)
    
    def generate_streaming(self, 
                          process: str,
                          total_events: int,
                          resonance: str,
                          chunk_size: int = 100_000):
        """Generate large dataset in memory-efficient chunks."""
        
        if process == 'data':
            filename = f"data-{resonance}_selected.parquet"
        else:
            filename = f"mc16-{process}-{resonance}_selected.parquet"
        
        filepath = self.output_dir / filename
        
        # Generate first chunk to establish schema
        first_chunk = self.generator.generate_process(process, 
                                                     min(chunk_size, total_events), 
                                                     resonance)
        first_chunk.write_parquet(filepath)
        
        remaining = total_events - chunk_size
        
        # Append remaining chunks
        while remaining > 0:
            current_size = min(chunk_size, remaining)
            chunk = self.generator.generate_process(process, current_size, resonance)
            
            # Append to existing file
            existing = pl.read_parquet(filepath)
            combined = pl.concat([existing, chunk])
            combined.write_parquet(filepath)
            
            remaining -= current_size
            print(f"  Progress: {total_events - remaining:,}/{total_events:,}")
        
        return filepath