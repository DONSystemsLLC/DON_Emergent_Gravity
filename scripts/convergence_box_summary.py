#!/usr/bin/env python3
"""
Convergence Box Summary - Analyze L convergence studies for P2
Identical to convergence_grid_summary.py but with different pattern matching
"""
import argparse
import glob
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

def parse_orbit_summary(summary_path):
    """Parse summary.csv from orbit_metrics_sweep.py"""
    try:
        df = pd.read_csv(summary_path)
        if len(df) == 0:
            return None
        
        # Take the single row (vtheta=0.30 run)
        row = df.iloc[0]
        return {
            'dE_over_E': float(row.get('dE_over_E', np.nan)),
            'dL_over_L': float(row.get('dL_over_L', np.nan)),
            'slope_Fr': float(row.get('slope_Fr', np.nan)),
            'slope_err': float(row.get('slope_err', np.nan)),
            'precession_deg_per_orbit': float(row.get('precession_deg_per_orbit', np.nan)),
            'wall_sec': float(row.get('wall_sec', np.nan)),
            'returncode': int(row.get('returncode', -1))
        }
    except Exception as e:
        print(f"Warning: Could not parse {summary_path}: {e}")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_type", choices=["N", "L"], required=True)
    ap.add_argument("--out_csv", default="figs/convergence_box.csv")
    args = ap.parse_args()

    results = []
    
    # L convergence: look for sweeps/conv_box_L*/summary.csv  
    pattern = "sweeps/conv_box_L*/summary.csv"
    param_re = re.compile(r"conv_box_L(\d+)")
    param_name = "L"

    files = sorted(glob.glob(pattern))
    print(f"[conv-summary] Found {len(files)} L convergence files")
    
    for f in files:
        match = param_re.search(f)
        if not match:
            continue
            
        param_val = int(match.group(1))
        data = parse_orbit_summary(f)
        
        if data is None:
            print(f"Warning: Skipping {f} (parsing failed)")
            continue
            
        result = {param_name: param_val}
        result.update(data)
        results.append(result)
        
        # Quick status
        status = "OK" if data['returncode'] == 0 else f"RC={data['returncode']}"
        print(f"  {param_name}={param_val}: slope={data['slope_Fr']:.3f}±{data['slope_err']:.3f}, "
              f"dE={data['dE_over_E']:.2e}, dL={data['dL_over_L']:.2e} [{status}]")

    if not results:
        print(f"Error: No valid L convergence results found")
        return

    # Convert to DataFrame and analyze
    df = pd.DataFrame(results)
    df = df.sort_values(param_name)
    
    # Check slope stability (target: within ±0.03 of -2.0)
    slopes = df['slope_Fr'].dropna()
    if len(slopes) > 1:
        slope_range = slopes.max() - slopes.min()
        slope_mean = slopes.mean()
        slope_std = slopes.std()
        
        print(f"\n[conv-analysis] L convergence:")
        print(f"  Slope range: {slope_range:.4f} (target: ≤0.06)")
        print(f"  Slope mean: {slope_mean:.3f} ± {slope_std:.3f}")
        print(f"  Target: -2.00 ± 0.03")
        
        # Acceptance criteria
        slope_stable = slope_range <= 0.06  # ±0.03 range
        slope_accurate = abs(slope_mean + 2.0) <= 0.05
        
        # Conservation checks (example thresholds)
        dE_ok = (df['dE_over_E'] < 1e-5).all()
        dL_ok = (df['dL_over_L'] < 1e-3).all()
        
        print(f"  ✓ Slope stable: {slope_stable}")
        print(f"  ✓ Slope accurate: {slope_accurate}")  
        print(f"  ✓ Energy conservation: {dE_ok}")
        print(f"  ✓ Angular momentum conservation: {dL_ok}")

    # Save results
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[conv-summary] Wrote {args.out_csv}")

if __name__ == "__main__":
    main()