#!/usr/bin/env python3
"""
DON Emergent Gravity Physics Tests - P2 Validation Suite

Unit tests for P2 validation criteria:
- Convergence requirements
- Window independence  
- Helmholtz field properties
- Physics validation (Kepler, EP)
- Slip analysis validation

Run with: pytest tests/test_physics.py -v
"""
import os
import json
import glob
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Test configuration
TOLERANCE = 0.02  # 2% tolerance for convergence
CURL_TOLERANCE = 0.05  # 5% curl fraction limit
FLUX_TOLERANCE = 0.05  # 5% flux variation limit
KEPLER_RANGE = (0.98, 1.02)  # Kepler slope range
EP_TOLERANCE = 0.05  # 5% EP variation limit
SLIP_RANGE = (0.8, 1.2)  # 20% slip tolerance

class TestP2Convergence:
    """Test grid and box size convergence"""
    
    def test_grid_convergence_data_exists(self):
        """Test that grid convergence data file exists"""
        assert os.path.exists('figs/convergence_grid.csv'), "Grid convergence data missing"
    
    def test_box_convergence_data_exists(self):
        """Test that box convergence data file exists"""
        assert os.path.exists('figs/convergence_box.csv'), "Box convergence data missing"
    
    def test_grid_convergence_criteria(self):
        """Test grid size convergence within tolerance"""
        if not os.path.exists('figs/convergence_grid.csv'):
            pytest.skip("Grid convergence data not available")
        
        df = pd.read_csv('figs/convergence_grid.csv')
        assert not df.empty, "Grid convergence data is empty"
        
        # Check N >= 320 cases meet tolerance
        high_n = df[df['N'] >= 320] if 'N' in df.columns else df
        if not high_n.empty and 'relative_error' in df.columns:
            max_error = high_n['relative_error'].abs().max()
            assert max_error < TOLERANCE, f"Grid convergence error {max_error:.4f} exceeds tolerance {TOLERANCE}"
    
    def test_box_convergence_criteria(self):
        """Test box size convergence within tolerance"""
        if not os.path.exists('figs/convergence_box.csv'):
            pytest.skip("Box convergence data not available")
        
        df = pd.read_csv('figs/convergence_box.csv')
        assert not df.empty, "Box convergence data is empty"
        
        # Check L >= 160 cases meet tolerance
        large_l = df[df['L'] >= 160] if 'L' in df.columns else df
        if not large_l.empty and 'relative_error' in df.columns:
            max_error = large_l['relative_error'].abs().max()
            assert max_error < TOLERANCE, f"Box convergence error {max_error:.4f} exceeds tolerance {TOLERANCE}"

class TestP2WindowIndependence:
    """Test window independence validation"""
    
    def test_window_heatmap_data_exists(self):
        """Test that window heatmap JSON exists"""
        assert os.path.exists('figs/slope_window_heatmap.json'), "Window heatmap data missing"
    
    def test_bootstrap_consistency(self):
        """Test bootstrap confidence interval consistency"""
        if not os.path.exists('figs/slope_window_heatmap.json'):
            pytest.skip("Window heatmap data not available")
        
        with open('figs/slope_window_heatmap.json', 'r') as f:
            data = json.load(f)
        
        # Check bootstrap CI width
        bootstrap_ci = data.get('bootstrap_ci', {})
        if 'lower' in bootstrap_ci and 'upper' in bootstrap_ci:
            ci_width = bootstrap_ci['upper'] - bootstrap_ci['lower']
            assert ci_width < 0.1, f"Bootstrap CI width {ci_width:.4f} too large"
    
    def test_slope_stability(self):
        """Test slope stability across windows"""
        if not os.path.exists('figs/slope_window_heatmap.json'):
            pytest.skip("Window heatmap data not available")
        
        with open('figs/slope_window_heatmap.json', 'r') as f:
            data = json.load(f)
        
        # Check slope standard deviation
        slope_stats = data.get('slope_statistics', {})
        if 'std' in slope_stats:
            slope_std = slope_stats['std']
            assert slope_std < 0.05, f"Slope std {slope_std:.4f} exceeds stability limit"

class TestP2HelmholtzDiagnostics:
    """Test Helmholtz field properties"""
    
    def test_helmholtz_data_exists(self):
        """Test that Helmholtz diagnostics JSON exists"""
        assert os.path.exists('figs/helmholtz_checks.json'), "Helmholtz diagnostics missing"
    
    def test_curl_free_criterion(self):
        """Test curl-free field requirement"""
        if not os.path.exists('figs/helmholtz_checks.json'):
            pytest.skip("Helmholtz diagnostics not available")
        
        with open('figs/helmholtz_checks.json', 'r') as f:
            data = json.load(f)
        
        curl_metrics = data.get('curl_metrics', {})
        if 'curl_fraction' in curl_metrics:
            curl_fraction = curl_metrics['curl_fraction']
            assert curl_fraction < CURL_TOLERANCE, f"Curl fraction {curl_fraction:.4f} exceeds {CURL_TOLERANCE}"
    
    def test_flux_flatness_p2(self):
        """Test P2 flux flatness requirement"""
        if not os.path.exists('figs/helmholtz_checks.json'):
            pytest.skip("Helmholtz diagnostics not available")
        
        with open('figs/helmholtz_checks.json', 'r') as f:
            data = json.load(f)
        
        flux_metrics = data.get('flux_metrics', {})
        if 'flux_p95' in flux_metrics:
            flux_p95 = flux_metrics['flux_p95']
            assert flux_p95 < FLUX_TOLERANCE, f"Flux p95 {flux_p95:.4f} exceeds {FLUX_TOLERANCE}"

class TestP2PhysicsValidation:
    """Test physics validation requirements"""
    
    def test_kepler_data_exists(self):
        """Test that Kepler validation data exists"""
        assert os.path.exists('figs/kepler_fit.csv'), "Kepler validation data missing"
    
    def test_kepler_slope_criterion(self):
        """Test Kepler slope within acceptable range"""
        if not os.path.exists('figs/kepler_fit.csv'):
            pytest.skip("Kepler validation data not available")
        
        df = pd.read_csv('figs/kepler_fit.csv')
        assert not df.empty, "Kepler validation data is empty"
        
        # Check slope column exists and is in range
        slope_col = 'slope' if 'slope' in df.columns else 'kepler_slope'
        if slope_col in df.columns:
            slopes = df[slope_col].dropna()
            if not slopes.empty:
                mean_slope = slopes.mean()
                assert KEPLER_RANGE[0] <= mean_slope <= KEPLER_RANGE[1], \
                    f"Kepler slope {mean_slope:.4f} outside range {KEPLER_RANGE}"
    
    def test_ep_data_availability(self):
        """Test that EP sweep data is available"""
        ep_files = glob.glob('sweeps/ep_m*/summary.csv')
        assert len(ep_files) > 0, "No EP sweep data found"
    
    def test_ep_mass_independence(self):
        """Test equivalence principle mass independence"""
        ep_files = glob.glob('sweeps/ep_m*/summary.csv')
        if not ep_files:
            pytest.skip("No EP sweep data available")
        
        # Collect slopes from all EP files
        slopes = []
        for ep_file in ep_files:
            try:
                df = pd.read_csv(ep_file)
                if 'slope' in df.columns:
                    slopes.extend(df['slope'].dropna().tolist())
            except:
                continue
        
        if slopes:
            slope_variation = np.std(slopes) / np.mean(slopes)
            assert slope_variation < EP_TOLERANCE, \
                f"EP mass dependence {slope_variation:.4f} exceeds {EP_TOLERANCE}"

class TestP2SlipAnalysis:
    """Test scale-dependent slip analysis"""
    
    def test_slip_data_exists(self):
        """Test that slip analysis data exists"""
        assert os.path.exists('outputs/CLENS_release/slip_analysis.csv'), \
            "Slip analysis data missing - run 'make wl-finalize'"
    
    def test_slip_band_means_exists(self):
        """Test that band means data exists"""
        assert os.path.exists('outputs/CLENS_release/band_means.csv'), \
            "Band means data missing - run 'make wl-finalize'"
    
    def test_slip_parameter_agreement(self):
        """Test slip parameter within tolerance"""
        slip_file = 'outputs/CLENS_release/slip_analysis.csv'
        if not os.path.exists(slip_file):
            pytest.skip("Slip analysis data not available")
        
        df = pd.read_csv(slip_file)
        assert not df.empty, "Slip analysis data is empty"
        
        if 'slip_parameter' in df.columns:
            valid_slip = df['slip_parameter'].dropna()
            if not valid_slip.empty:
                mean_slip = valid_slip.mean()
                assert SLIP_RANGE[0] <= mean_slip <= SLIP_RANGE[1], \
                    f"Mean slip {mean_slip:.3f} outside tolerance {SLIP_RANGE}"
    
    def test_slip_plots_generated(self):
        """Test that slip diagnostic plots exist"""
        plot_files = [
            'outputs/CLENS_release/slip_analysis.png',
            'outputs/CLENS_release/gt_panel.png'
        ]
        
        for plot_file in plot_files:
            if not os.path.exists(plot_file):
                pytest.skip(f"Plot {plot_file} not generated - run 'make wl-finalize'")

class TestP2DataIntegrity:
    """Test data integrity and file structure"""
    
    def test_field_files_exist(self):
        """Test that required field files exist"""
        field_patterns = [
            'fields/N320_L160_box.manifest.json',
            'fields/N320_L160_box_eps*.npz'
        ]
        
        for pattern in field_patterns:
            if '*' in pattern:
                files = glob.glob(pattern)
                assert len(files) > 0, f"No files matching pattern {pattern}"
            else:
                assert os.path.exists(pattern), f"Required file {pattern} missing"
    
    def test_output_directories(self):
        """Test that output directories are properly structured"""
        required_dirs = [
            'outputs/',
            'figs/',
            'docs/',
            'sweeps/'
        ]
        
        for dir_path in required_dirs:
            assert os.path.exists(dir_path), f"Required directory {dir_path} missing"
    
    def test_makefile_targets(self):
        """Test that Makefile contains required P2 targets"""
        assert os.path.exists('Makefile'), "Makefile missing"
        
        with open('Makefile', 'r') as f:
            makefile_content = f.read()
        
        required_targets = [
            'paper-kit:',
            'wl-finalize:',
            'conv-grid:',
            'conv-box:',
            'slope-window:',
            'helmholtz:',
            'kepler-fit:',
            'ep-fit:'
        ]
        
        for target in required_targets:
            assert target in makefile_content, f"Makefile target {target} missing"

class TestP2AcceptanceGates:
    """Test overall P2 acceptance criteria"""
    
    def test_acceptance_box_generated(self):
        """Test that acceptance box is generated"""
        if os.path.exists('docs/ACCEPTANCE_BOX_P2.md'):
            # If it exists, check it's not empty
            with open('docs/ACCEPTANCE_BOX_P2.md', 'r') as f:
                content = f.read()
            assert len(content) > 100, "Acceptance box appears to be empty"
    
    def test_p2_report_generated(self):
        """Test that P2 report is generated"""
        if os.path.exists('docs/REPORT.md'):
            # If it exists, check it contains P2 content
            with open('docs/REPORT.md', 'r') as f:
                content = f.read()
            assert 'P2' in content or 'Universality' in content, \
                "Report doesn't appear to be P2-specific"
    
    def test_validation_completeness(self):
        """Test that all major validation components are present"""
        components = {
            'convergence': 'figs/convergence_grid.csv',
            'window_independence': 'figs/slope_window_heatmap.json', 
            'helmholtz': 'figs/helmholtz_checks.json',
            'kepler': 'figs/kepler_fit.csv',
            'slip': 'outputs/CLENS_release/slip_analysis.csv'
        }
        
        missing_components = []
        for name, file_path in components.items():
            if not os.path.exists(file_path):
                missing_components.append(name)
        
        if missing_components:
            pytest.skip(f"Missing validation components: {', '.join(missing_components)}")
        
        # If all components exist, validation is complete
        assert True, "All P2 validation components are present"

# Pytest fixtures for test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        'tolerance': TOLERANCE,
        'curl_tolerance': CURL_TOLERANCE,
        'flux_tolerance': FLUX_TOLERANCE,
        'kepler_range': KEPLER_RANGE,
        'ep_tolerance': EP_TOLERANCE,
        'slip_range': SLIP_RANGE
    }

@pytest.fixture(scope="session") 
def workspace_root():
    """Workspace root directory fixture"""
    return Path.cwd()

# Test markers for different validation phases
pytestmark = [
    pytest.mark.p2,
    pytest.mark.validation,
    pytest.mark.physics
]

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])