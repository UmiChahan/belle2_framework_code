"""
Belle2 Scientific Histogram Visualizer - Production Ready
=========================================================
Comprehensive visualization framework for Belle2 physics analysis
with robust error handling and publication-quality output.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, LogLocator
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
import re

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class Belle2HistogramVisualizer:
    """
    Production-grade histogram visualization with intelligent process discovery.
    Engineered for robustness, clarity, and scientific precision.
    """
    
    def __init__(self, style_preset: str = 'publication', use_latex: bool = True):
        """
        Initialize visualizer with comprehensive configuration.
        
        Args:
            style_preset: Visual style preset ('publication', 'presentation', 'poster')
            use_latex: Enable LaTeX rendering if available
        """
        self.style_preset = style_preset
        self.use_latex = use_latex
        self._setup_color_palette()
        self._configure_matplotlib()
        self.discovered_processes = set()
        
    def _setup_color_palette(self):
        """
        Establish perceptually distinct color palette with optimal visual separation.
        Based on colorblind-safe principles with maximum discriminability.
        """
        self.process_colors = {
            # Data - always black for maximum contrast
            'data': '#000000',
            
            # Primary leptons - high contrast cool tones
            'mumu': '#1E88E5',      # Brilliant blue
            'ee': '#D32F2F',        # Vivid red
            'taupair': '#7B1FA2',   # Deep purple
            
            # Light quarks - warm earth tones
            'uubar': '#F57C00',     # Vibrant orange
            'ddbar': '#388E3C',     # Forest green
            'ssbar': '#FBC02D',     # Golden yellow
            'ccbar': '#8D6E63',     # Warm brown
            
            # Heavy flavor
            'bbbar': '#5D4037',     # Dark chocolate
            'charged': '#6A4C93',   # Royal purple
            'mixed': '#00796B',     # Teal
            
            # Bosons
            'gg': '#FFD600',        # Pure gold
            
            # Rare processes
            'hhISR': '#E91E63',     # Magenta
            'hhisr': '#E91E63',     # Magenta (alias)
            'llXX': '#00ACC1',      # Cyan
            'llxx': '#00ACC1',      # Cyan (alias)
            'eemumu': '#9575CD',    # Lavender
            'eeee': '#4FC3F7',      # Sky blue
            
            # Default
            'default': '#757575'    # Neutral gray
        }
        
        # Define optimal stacking order for visual hierarchy
        self.stacking_order = [
            'mumu', 'ee', 'taupair',
            'uubar', 'ddbar', 'ssbar', 'ccbar',
            'bbbar', 'charged', 'mixed',
            'gg', 'hhisr', 'llxx', 'eemumu', 'eeee'
        ]
        
    def _configure_matplotlib(self):
        """
        Configure matplotlib for publication-quality output with fallback options.
        """
        base_config = {
            'font.family': 'serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'axes.formatter.use_mathtext': True,
            'axes.formatter.limits': (-3, 4),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        }
        
        # Attempt LaTeX configuration if requested
        if self.use_latex:
            try:
                test_fig, test_ax = plt.subplots(figsize=(1, 1))
                test_ax.text(0.5, 0.5, r'$\alpha$', usetex=True)
                plt.close(test_fig)
                
                latex_config = {
                    'text.usetex': True,
                    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
                    'font.serif': ['Computer Modern Roman'],
                }
                base_config.update(latex_config)
                print("✓ LaTeX rendering enabled")
            except Exception:
                print("⚠️ LaTeX not available, using mathtext fallback")
                base_config['text.usetex'] = False
                base_config['font.serif'] = ['DejaVu Serif', 'Times New Roman']
        
        plt.rcParams.update(base_config)
    
    def _extract_physics_process(self, full_name: str) -> str:
        """
        Extract physics process from Belle2 naming conventions with robust pattern matching.
        
        Args:
            full_name: Full process name from data structure
            
        Returns:
            Standardized physics process identifier
        """
        # Remove sample_name= prefix
        name = full_name.replace('sample_name=', '')
        name_lower = name.lower()
        
        # Check for data first
        if 'data' in name_lower:
            return 'data'
        
        # Extract MC physics process using regex
        pattern = r'mc[45]s_([a-z]+)_|mcoff_([a-z]+)_'
        match = re.search(pattern, name_lower)
        if match:
            process = match.group(1) or match.group(2)
            if process in self.process_colors:
                return process
        
        # Fallback: search for known physics processes
        known_processes = list(self.process_colors.keys())
        for proc in known_processes:
            if proc != 'default' and proc in name_lower:
                return proc
        
        return 'default'
    
    def _format_process_label(self, process_key: str) -> str:
        """
        Create formatted label for physics process with LaTeX support.
        
        Args:
            process_key: Standardized process identifier
            
        Returns:
            Formatted label string
        """
        latex_labels = {
            'data': 'Data',
            'mumu': r'$\mu^+\mu^-$',
            'ee': r'$e^+e^-$',
            'taupair': r'$\tau^+\tau^-$',
            'uubar': r'$u\bar{u}$',
            'ddbar': r'$d\bar{d}$',
            'ssbar': r'$s\bar{s}$',
            'ccbar': r'$c\bar{c}$',
            'bbbar': r'$b\bar{b}$',
            'charged': r'$B^+B^-$',
            'mixed': r'$B^0\bar{B}^0$',
            'gg': r'$\gamma\gamma$',
            'hhisr': r'Hadronic ISR',
            'hhISR': r'Hadronic ISR',
            'eemumu': r'$e^+e^-\mu^+\mu^-$',
            'eeee': r'$e^+e^-e^+e^-$',
            'llxx': r'$\ell\ell XX$',
            'llXX': r'$\ell\ell XX$',
            'default': 'Other'
        }
        
        return latex_labels.get(process_key, process_key)
    
    def discover_processes(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Discover and categorize all processes in analysis results.
        
        Args:
            all_results: Complete analysis results dictionary
            
        Returns:
            Mapping of full process names to standardized identifiers
        """
        process_map = {}
        
        # Extract all unique process names
        for var_name, results in all_results.items():
            histograms = results.get('histograms', {})
            for stage, stage_hists in histograms.items():
                for proc_name in stage_hists.keys():
                    if proc_name not in process_map:
                        physics_proc = self._extract_physics_process(proc_name)
                        process_map[proc_name] = physics_proc
        
        # Print discovery summary
        print("\n" + "="*70)
        print("🔍 PROCESS DISCOVERY SUMMARY")
        print("="*70)
        
        # Group by physics process
        grouped = {}
        for full_name, phys_proc in process_map.items():
            if phys_proc not in grouped:
                grouped[phys_proc] = []
            grouped[phys_proc].append(full_name)
        
        # Display grouped processes
        for phys_proc in sorted(grouped.keys()):
            label = self._format_process_label(phys_proc)
            color = self.process_colors.get(phys_proc, self.process_colors['default'])
            print(f"\n{label:20s} [Color: {color}]")
            for full_name in grouped[phys_proc]:
                display_name = full_name.replace('sample_name=', '')
                if len(display_name) > 60:
                    display_name = display_name[:57] + "..."
                print(f"  • {display_name}")
        
        print("\n" + "="*70)
        print(f"Total: {len(process_map)} unique processes")
        print("="*70)
        
        return process_map
    
    def create_histogram(self,
                        results: Dict[str, Any],
                        stage: str = 'cuts',
                        variable_name: str = 'pRecoil',
                        figsize: Tuple[float, float] = (12, 9),
                        save_path: Optional[str] = None,
                        show_ratio: bool = True) -> plt.Figure:
        """
        Create publication-quality histogram with optional ratio plot.
        
        Args:
            results: Analysis results for single variable
            stage: Analysis stage to visualize
            variable_name: Physics variable name
            figsize: Figure dimensions
            save_path: Output file path
            show_ratio: Include ratio plot
            
        Returns:
            Matplotlib figure object
        """
        # Create figure with appropriate layout
        if show_ratio:
            fig, (ax_main, ax_ratio) = plt.subplots(
                2, 1, figsize=figsize,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.02}
            )
        else:
            fig, ax_main = plt.subplots(1, 1, figsize=figsize)
            ax_ratio = None
        
        # Extract histogram data
        histograms = results['histograms'].get(stage, {})
        if not histograms:
            ax_main.text(0.5, 0.5, f'No data for stage: {stage}',
                        transform=ax_main.transAxes, ha='center', va='center')
            return fig
        
        # Process discovery and categorization
        process_map = {}
        for proc_name in histograms.keys():
            process_map[proc_name] = self._extract_physics_process(proc_name)
        
        # Separate data and MC components
        data_info = None
        mc_components = []
        
        print(f"\n📊 Stage: {stage}")
        print("-" * 40)
        
        for proc_name, (counts, edges) in histograms.items():
            phys_proc = process_map[proc_name]
            
            if phys_proc == 'data':
                data_info = {
                    'counts': np.asarray(counts, dtype=np.float64),
                    'edges': edges,
                    'label': 'Data',
                    'total': np.sum(counts)
                }
            else:
                mc_components.append({
                    'counts': np.asarray(counts, dtype=np.float64),
                    'edges': edges,
                    'process': phys_proc,
                    'label': self._format_process_label(phys_proc),
                    'color': self.process_colors.get(phys_proc, self.process_colors['default']),
                    'total': np.sum(counts)
                })
        
        # Sort MC components for optimal stacking
        def get_stack_priority(comp):
            try:
                return (self.stacking_order.index(comp['process']), -comp['total'])
            except ValueError:
                return (999, -comp['total'])
        
        mc_components.sort(key=get_stack_priority, reverse=True)
        
        # Plot stacked MC histograms
        if mc_components:
            bottom = np.zeros(len(mc_components[0]['counts']))
            
            for comp in mc_components:
                # Dynamic alpha based on contribution
                alpha = 0.85 if comp['total'] > 1000 else 0.75
                
                ax_main.fill_between(
                    comp['edges'][:-1],
                    bottom,
                    bottom + comp['counts'],
                    step='post',
                    alpha=alpha,
                    color=comp['color'],
                    edgecolor='#2C2C2C',
                    linewidth=0.6,
                    label=comp['label']
                )
                bottom += comp['counts']
                print(f"  MC: {comp['label']:20s} - {comp['total']:8.0f} events")
        
        # Plot data points
        if data_info:
            counts = data_info['counts']
            edges = data_info['edges']
            bin_centers = (edges[:-1] + edges[1:]) / 2
            bin_width = edges[1] - edges[0]
            
            errors = np.sqrt(np.maximum(counts, 1))
            mask = counts > 0
            
            ax_main.errorbar(
                bin_centers[mask], counts[mask],
                yerr=errors[mask],
                xerr=bin_width/2,
                fmt='o',
                color='black',
                markersize=4.5,
                markeredgewidth=1.0,
                capsize=2,
                capthick=1.0,
                elinewidth=1.0,
                label='Data',
                zorder=10
            )
            print(f"  Data:                      - {data_info['total']:8.0f} events")
        
        # Create ratio plot if requested
        if show_ratio and ax_ratio and data_info and mc_components:
            self._create_ratio_plot(ax_ratio, data_info, mc_components)
        
        # Apply styling
        self._apply_styling(ax_main, ax_ratio, variable_name)
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, facecolor='white', edgecolor='none', dpi=300)
            print(f"\n✓ Saved: {save_path}")
        
        return fig
    
    def _create_ratio_plot(self, ax: plt.Axes, 
                          data_info: Dict, 
                          mc_components: List[Dict]):
        """
        Create data/MC ratio plot with proper error propagation.
        
        Args:
            ax: Matplotlib axes for ratio plot
            data_info: Data histogram information
            mc_components: List of MC component dictionaries
        """
        # Sum MC with explicit float conversion
        mc_sum = np.zeros(len(data_info['counts']), dtype=np.float64)
        for comp in mc_components:
            mc_sum += comp['counts']
        
        data_counts = data_info['counts']
        
        # Calculate ratio with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(data_counts, mc_sum)
            ratio[~np.isfinite(ratio)] = 0
            
            # Error propagation
            data_err = np.sqrt(np.maximum(data_counts, 1))
            mc_err = np.sqrt(np.maximum(mc_sum, 1))
            
            ratio_err = ratio * np.sqrt(
                np.divide(data_err**2, data_counts**2, where=data_counts>0) +
                np.divide(mc_err**2, mc_sum**2, where=mc_sum>0)
            )
            ratio_err[~np.isfinite(ratio_err)] = 0
        
        # Plot ratio points
        edges = data_info['edges']
        bin_centers = (edges[:-1] + edges[1:]) / 2
        bin_width = edges[1] - edges[0]
        
        mask = (data_counts > 0) & (mc_sum > 0)
        
        ax.errorbar(
            bin_centers[mask], ratio[mask],
            yerr=ratio_err[mask],
            xerr=bin_width/2,
            fmt='o',
            color='black',
            markersize=4,
            capsize=2,
            elinewidth=1.0
        )
        
        # Reference lines and bands
        ax.axhline(y=1, color='#424242', linestyle='-', linewidth=1.2, alpha=0.8)
        ax.fill_between(edges[:-1], 0.9, 1.1,
                       step='post', alpha=0.15, color='gray',
                       label='±10% band')
        
        # Enhanced grid
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.2)
        
        # Configure axes
        ax.set_ylabel('Data/MC', fontsize=11)
        ax.set_ylim(0.5, 1.5)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    def _apply_styling(self, ax_main: plt.Axes, 
                      ax_ratio: Optional[plt.Axes],
                      variable_name: str):
        """
        Apply comprehensive styling to histogram plots.
        
        Args:
            ax_main: Main histogram axes
            ax_ratio: Optional ratio plot axes
            variable_name: Physics variable for labeling
        """
        # Configure main plot
        ax_main.set_yscale('log')
        ax_main.set_ylim(bottom=0.1)
        ax_main.set_ylabel('Events / bin', fontsize=12, fontweight='medium')
        
        # Enhanced grid architecture
        ax_main.grid(True, which='major', axis='both', 
                    linestyle='-', linewidth=0.6, alpha=0.25, color='#CCCCCC')
        ax_main.grid(True, which='minor', axis='y',
                    linestyle=':', linewidth=0.3, alpha=0.15, color='#E0E0E0')
        
        # Add subtle magnitude guides
        for power in range(-1, 7):
            y_val = 10**power
            if ax_main.get_ylim()[0] < y_val < ax_main.get_ylim()[1]:
                ax_main.axhline(y=y_val, color='#F0F0F0', linewidth=0.5, 
                              alpha=0.3, zorder=0)
        
        # Variable-specific labels
        var_labels = {
            'pRecoil': r'$p_{\mathrm{recoil}}$ [GeV/$c$]',
            'mu1P': r'$p_{\mu_1}$ [GeV/$c$]',
            'mu2P': r'$p_{\mu_2}$ [GeV/$c$]',
            'EoPRecoil': r'$E/p_{\mathrm{recoil}}$',
            'mRecoil': r'$m_{\mathrm{recoil}}$ [GeV/$c^2$]',
            'm2Recoil': r'$m^2_{\mathrm{recoil}}$ [(GeV/$c^2$)$^2$]',
        }
        
        xlabel = var_labels.get(variable_name, variable_name)
        
        # Configure x-axis
        if ax_ratio:
            ax_ratio.set_xlabel(xlabel, fontsize=12, fontweight='medium')
            ax_main.set_xticklabels([])
            ax_ratio.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax_ratio.grid(True, which='major', axis='both',
                         linestyle='-', linewidth=0.5, alpha=0.25)
        else:
            ax_main.set_xlabel(xlabel, fontsize=12, fontweight='medium')
            ax_main.xaxis.set_minor_locator(AutoMinorLocator(5))
        
        # Configure legend
        handles, labels = ax_main.get_legend_handles_labels()
        if handles:
            legend = ax_main.legend(
                handles[::-1], labels[::-1],
                loc='upper right',
                frameon=True,
                fancybox=False,
                shadow=False,
                framealpha=0.95,
                edgecolor='#666666',
                facecolor='white',
                ncol=2 if len(handles) > 10 else 1,
                columnspacing=1.2,
                handletextpad=0.6,
                borderpad=0.8,
                fontsize=9
            )
            legend.get_frame().set_linewidth(0.8)
        
        # Belle II watermark
        watermark = r'\textbf{Belle II Preliminary}' if self.use_latex else 'Belle II Preliminary'
        ax_main.text(
            0.05, 0.95,
            watermark,
            transform=ax_main.transAxes,
            fontsize=13,
            fontweight='bold',
            verticalalignment='top',
            color='#2C2C2C',
            alpha=0.8
        )
        
        # Frame enhancement
        for spine in ax_main.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor('#333333')
        
        if ax_ratio:
            for spine in ax_ratio.spines.values():
                spine.set_linewidth(1.2)
                spine.set_edgecolor('#333333')


def plot_belle2_analysis(all_results: Dict[str, Any],
                         output_dir: str = './plots',
                         style: str = 'publication',
                         show_ratio: bool = True,
                         use_latex: bool = True) -> Dict[str, plt.Figure]:
    """
    Create complete visualization suite from Belle2 analysis results.
    
    Args:
        all_results: Complete analysis results dictionary
        output_dir: Output directory for plots
        style: Visual style preset
        show_ratio: Include ratio plots
        use_latex: Enable LaTeX rendering
    
    Returns:
        Dictionary of created figure objects
    """
    # Initialize output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    viz = Belle2HistogramVisualizer(style_preset=style, use_latex=use_latex)
    
    print("\n" + "="*70)
    print("🎨 BELLE2 VISUALIZATION SUITE")
    print("="*70)
    
    # Process discovery
    process_map = viz.discover_processes(all_results)
    
    # Create plots
    figures = {}
    
    for var_name, results in all_results.items():
        print(f"\n📈 Variable: {var_name}")
        print("="*40)
        
        for stage in results.get('histograms', {}).keys():
            fig = viz.create_histogram(
                results=results,
                stage=stage,
                variable_name=var_name,
                figsize=(12, 9),
                save_path=f"{output_dir}/{var_name}_{stage}.pdf",
                show_ratio=show_ratio
            )
            figures[f"{var_name}_{stage}"] = fig
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE")
    print(f"   Created {len(figures)} publication-quality figures")
    print(f"   Output: {output_dir}")
    print("="*70 + "\n")
    
    return figures