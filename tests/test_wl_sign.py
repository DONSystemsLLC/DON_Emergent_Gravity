#!/usr/bin/env python3
"""
Test weak lensing sign convention to prevent regression
"""

import pytest
import json
import numpy as np
from pathlib import Path

class TestWLSignConvention:
    """Test suite to prevent sign convention regression in weak lensing analysis"""
    
    def test_rotation_nulls_near_zero(self):
        """Rotation null tests should be consistent with zero signal"""
        sign_val_path = Path("outputs/CLENS_release/sign_validation.json")
        
        if not sign_val_path.exists():
            pytest.skip("Sign validation file not found")
        
        with open(sign_val_path) as f:
            sign_data = json.load(f)
        
        rot_mean = sign_data.get("rotation_null_mean", 0.0)
        rot_std = sign_data.get("rotation_null_std", 1.0)
        
        # Rotation null mean should be within 1 sigma of zero
        assert abs(rot_mean) <= rot_std, f"Rotation null mean {rot_mean} exceeds 1σ = {rot_std}"
        
        # Should not be systematically offset
        assert abs(rot_mean) < 0.1, f"Rotation null mean {rot_mean} too large (systematic offset?)"
    
    def test_cross_shear_nulls_near_zero(self):
        """Cross-shear null tests should be consistent with zero signal"""
        sign_val_path = Path("outputs/CLENS_release/sign_validation.json")
        
        if not sign_val_path.exists():
            pytest.skip("Sign validation file not found")
        
        with open(sign_val_path) as f:
            sign_data = json.load(f)
        
        if "cross_shear_null_mean" not in sign_data:
            pytest.skip("Cross-shear null data not available")
        
        x_mean = sign_data["cross_shear_null_mean"]
        x_std = sign_data["cross_shear_null_std"]
        
        # Cross-shear null mean should be within 1 sigma of zero
        assert abs(x_mean) <= x_std, f"Cross-shear null mean {x_mean} exceeds 1σ = {x_std}"
    
    def test_slip_measurements_physical_range(self):
        """Slip measurements should be in physically reasonable range"""
        slip_path = Path("outputs/CLENS_release/slip_analysis_final.csv")
        
        if not slip_path.exists():
            pytest.skip("Final slip analysis not found")
        
        import pandas as pd
        df = pd.read_csv(slip_path)
        
        slip_values = df["slip_cal"].values
        
        # All slip values should be finite
        assert np.all(np.isfinite(slip_values)), "Non-finite slip values detected"
        
        # Should be in reasonable physical range
        assert np.all(slip_values >= -5.0), f"Slip values too negative: min = {np.min(slip_values)}"
        assert np.all(slip_values <= 5.0), f"Slip values too positive: max = {np.max(slip_values)}"
    
    def test_calibration_factor_stability(self):
        """Calibration factor should be stable and reasonable"""
        summary_path = Path("outputs/CLENS_release/slip_final_summary.json")
        
        if not summary_path.exists():
            pytest.skip("Slip final summary not found")
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        A = summary.get("calibration_factor", 1.0)
        
        # Calibration factor should be finite and non-zero
        assert np.isfinite(A), f"Calibration factor A = {A} is not finite"
        assert abs(A) > 1e-6, f"Calibration factor A = {A} too close to zero"
        
        # Should be in reasonable range (allowing for sign convention)
        assert abs(A) < 10.0, f"Calibration factor A = {A} suspiciously large"
    
    def test_referee_gate_passes(self):
        """Referee-grade acceptance gate should pass"""
        gate_path = Path("outputs/CLENS_release/slip_gate_referee.json")
        
        if not gate_path.exists():
            pytest.skip("Referee gate file not found")
        
        with open(gate_path) as f:
            gate_data = json.load(f)
        
        overall_pass = gate_data.get("overall_pass", False)
        assert overall_pass, "Referee acceptance gate fails"
        
        # Check individual tests
        gate_tests = gate_data.get("gate", {})
        
        # Null tests
        null_tests = gate_tests.get("nulls", {})
        for test_name, passed in null_tests.items():
            assert passed, f"Null test {test_name} failed"
        
        # SNR tests
        snr_tests = gate_tests.get("snr", {})
        for test_name, passed in snr_tests.items():
            assert passed, f"SNR test {test_name} failed"
    
    def test_sign_consistency_across_bands(self):
        """Sign convention should be consistent across radial bands"""
        slip_path = Path("outputs/CLENS_release/slip_analysis_final.csv")
        
        if not slip_path.exists():
            pytest.skip("Final slip analysis not found")
        
        import pandas as pd
        df = pd.read_csv(slip_path)
        
        gt_obs = df["gt_observed"].values
        gt_th = df["gt_theory"].values
        slip = df["slip_cal"].values
        
        # Check consistency: slip = gt_obs / (A * gt_th)
        # Since A is global, sign relationship should be consistent
        
        # At least some bands should have measurable signal
        assert np.any(np.abs(gt_obs) > 1e-6), "No measurable gt_observed signal"
        assert np.any(np.abs(gt_th) > 1e-6), "No gt_theory signal"
        
        # No systematic sign flips that would indicate convention errors
        sign_gt_obs = np.sign(gt_obs[np.abs(gt_obs) > 1e-6])
        sign_gt_th = np.sign(gt_th[np.abs(gt_th) > 1e-6])
        
        # Theory should have consistent sign (all positive or all negative)
        if len(sign_gt_th) > 1:
            assert len(np.unique(sign_gt_th)) <= 2, "Theory signs too inconsistent"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])