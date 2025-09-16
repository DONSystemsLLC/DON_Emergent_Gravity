#!/usr/bin/env python3
"""
Jackknife error estimation for slip analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob

def compute_jackknife_errors():
    """Compute jackknife error bars using LOO samples"""
    
    # Find all LOO jackknife samples
    jk_pattern = "outputs/CLENS_patch_jk/LOO_*/profile_focus.csv"
    jk_files = glob.glob(jk_pattern)
    
    if not jk_files:
        print(f"ERROR: No jackknife samples found matching: {jk_pattern}")
        return None
    
    print(f"Found {len(jk_files)} jackknife samples")
    
    # Load all jackknife samples
    jk_samples = []
    for jk_file in sorted(jk_files):
        try:
            df = pd.read_csv(jk_file)
            jk_samples.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {jk_file}: {e}")
    
    if not jk_samples:
        print("ERROR: No valid jackknife samples loaded")
        return None
    
    print(f"Successfully loaded {len(jk_samples)} jackknife samples")
    
    # Extract gt values for each sample (interpolate to common radial grid)
    r_common = np.logspace(np.log10(0.3), np.log10(2.5), 20)  # Common radial grid
    gt_jk_matrix = []
    
    for jk_df in jk_samples:
        # Interpolate to common grid
        gt_interp = np.interp(r_common, jk_df['R_Mpc'], jk_df['gt'])
        gt_jk_matrix.append(gt_interp)
    
    gt_jk_matrix = np.array(gt_jk_matrix)  # Shape: (n_samples, n_radii)
    
    # Compute jackknife statistics
    gt_mean = np.mean(gt_jk_matrix, axis=0)
    gt_std = np.std(gt_jk_matrix, axis=0, ddof=1)  # Unbiased estimator
    gt_stderr = gt_std / np.sqrt(len(jk_samples))
    
    # Create results dataframe
    jk_results = pd.DataFrame({
        'R_Mpc': r_common,
        'gt_mean': gt_mean,
        'gt_std': gt_std,
        'gt_stderr': gt_stderr,
        'n_samples': len(jk_samples)
    })
    
    # Save jackknife results
    jk_output = Path("outputs/CLENS_release/jackknife_errors.csv")
    jk_results.to_csv(jk_output, index=False)
    
    print(f"Jackknife error analysis saved to: {jk_output}")
    print(f"Mean gt stderr = {np.mean(gt_stderr):.6f}")
    print(f"Max gt stderr = {np.max(gt_stderr):.6f}")
    
    return jk_results

def add_errors_to_slip_analysis():
    """Add jackknife errors to the calibrated slip analysis"""
    
    # Compute jackknife errors
    jk_results = compute_jackknife_errors()
    if jk_results is None:
        return False
    
    # Load calibrated slip analysis
    slip_path = Path("outputs/CLENS_release/slip_analysis_calibrated.csv")
    if not slip_path.exists():
        print(f"ERROR: Calibrated slip analysis not found: {slip_path}")
        return False
    
    slip_df = pd.read_csv(slip_path)
    
    # Interpolate jackknife errors to slip analysis radial points
    gt_stderr_interp = np.interp(slip_df['r_center'], jk_results['R_Mpc'], jk_results['gt_stderr'])
    
    # Add error columns
    slip_df['gt_stderr'] = gt_stderr_interp
    slip_df['slip_cal_err'] = np.abs(slip_df['slip_cal']) * np.sqrt(
        (gt_stderr_interp / np.abs(slip_df['gt_observed'] + 1e-12))**2 +
        (0.1)**2  # Assume 10% systematic error on theory prediction
    )
    
    # Save enhanced slip analysis with errors
    output_path = slip_path.with_name("slip_analysis_final.csv")
    slip_df.to_csv(output_path, index=False)
    
    print(f"\nFinal slip analysis with errors saved to: {output_path}")
    
    # Create summary with errors
    final_summary = {
        "slip_measurements": {
            "r_center": slip_df['r_center'].tolist(),
            "slip_cal": slip_df['slip_cal'].tolist(),
            "slip_cal_err": slip_df['slip_cal_err'].tolist()
        },
        "calibration_factor": -0.4451871647659409,  # From previous calibration
        "mean_slip_cal": float(np.mean(slip_df['slip_cal'])),
        "weighted_mean_slip": float(np.average(slip_df['slip_cal'], weights=1/slip_df['slip_cal_err']**2)),
        "jackknife_samples": len(jk_results) if jk_results is not None else 0
    }
    
    summary_path = Path("outputs/CLENS_release/slip_final_summary.json")
    summary_path.write_text(json.dumps(final_summary, indent=2))
    
    print(f"Final summary saved to: {summary_path}")
    print(f"Weighted mean slip (calibrated) = {final_summary['weighted_mean_slip']:.4f}")
    
    return True

if __name__ == "__main__":
    success = add_errors_to_slip_analysis()
    if success:
        print("\n✅ JACKKNIFE ERROR ANALYSIS COMPLETE")
    else:
        print("\n❌ JACKKNIFE ERROR ANALYSIS FAILED")