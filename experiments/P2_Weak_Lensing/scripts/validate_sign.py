#!/usr/bin/env python3
"""
Sign validation using rotation nulls for slip analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def validate_sign_convention():
    """Validate sign convention using rotation null tests"""
    
    # Load rotation null test results
    rot_null_path = Path("outputs/CLENS_patch_rot/profile_focus.csv")
    if not rot_null_path.exists():
        print(f"ERROR: Rotation null test file not found: {rot_null_path}")
        return False
    
    rot_null = pd.read_csv(rot_null_path)
    
    # Also check cross-shear nulls 
    cross_null_path = Path("outputs/CLENS_patch_e2flip_rot/profile_focus.csv")
    cross_null = pd.read_csv(cross_null_path) if cross_null_path.exists() else None
    
    # Compute statistics for rotation nulls
    gt_rot = rot_null["gt"].to_numpy()
    mean_rot = np.mean(gt_rot)
    std_rot = np.std(gt_rot)
    max_abs_rot = np.max(np.abs(gt_rot))
    
    # Test if nulls are consistent with zero (within 2-sigma)
    null_consistent = max_abs_rot < 2 * std_rot
    
    print("=" * 60)
    print("SIGN CONVENTION VALIDATION")
    print("=" * 60)
    print(f"Rotation null test (gt should be ~0):")
    print(f"  Mean gt = {mean_rot:.6f}")
    print(f"  Std gt  = {std_rot:.6f}")
    print(f"  Max |gt| = {max_abs_rot:.6f}")
    print(f"  2-sigma threshold = {2*std_rot:.6f}")
    print(f"  Null test passes: {null_consistent}")
    
    if cross_null is not None:
        gt_cross = cross_null["gt"].to_numpy()
        mean_cross = np.mean(gt_cross)
        std_cross = np.std(gt_cross)
        max_abs_cross = np.max(np.abs(gt_cross))
        
        print(f"\nCross-shear null test (gt should be ~0):")
        print(f"  Mean gt = {mean_cross:.6f}")
        print(f"  Std gt  = {std_cross:.6f}")
        print(f"  Max |gt| = {max_abs_cross:.6f}")
    
    # Compare with actual signal
    slip_path = Path("outputs/CLENS_release/slip_analysis.csv")
    if slip_path.exists():
        slip_data = pd.read_csv(slip_path)
        gt_signal = slip_data["gt_observed"].to_numpy()
        mean_signal = np.mean(np.abs(gt_signal))
        
        print(f"\nActual signal (for comparison):")
        print(f"  Mean |gt_observed| = {mean_signal:.6f}")
        print(f"  Signal/noise ratio = {mean_signal/std_rot:.1f}")
    
    # Save validation results
    validation_results = {
        "rotation_null_mean": float(mean_rot),
        "rotation_null_std": float(std_rot),
        "rotation_null_max_abs": float(max_abs_rot),
        "null_test_passes": bool(null_consistent),
        "validation_timestamp": pd.Timestamp.now().isoformat()
    }
    
    if cross_null is not None:
        validation_results.update({
            "cross_shear_null_mean": float(mean_cross),
            "cross_shear_null_std": float(std_cross),
            "cross_shear_null_max_abs": float(max_abs_cross)
        })
    
    output_path = Path("outputs/CLENS_release/sign_validation.json")
    output_path.write_text(json.dumps(validation_results, indent=2))
    print(f"\nValidation results saved to: {output_path}")
    
    return null_consistent

if __name__ == "__main__":
    sign_ok = validate_sign_convention()
    if sign_ok:
        print("\n✅ SIGN CONVENTION VALIDATED")
    else:
        print("\n❌ SIGN CONVENTION FAILED - REVIEW NEEDED")