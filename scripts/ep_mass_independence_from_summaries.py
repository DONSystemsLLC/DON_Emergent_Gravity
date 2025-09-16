#!/usr/bin/env python3
"""
Equivalence Principle Mass Independence Analysis - Enhanced for P2
Analyzes acceleration independence across different test masses
"""
import argparse
import glob
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

def extract_mass_from_path(path):
    """Extract test mass from path like sweeps/ep_m0.5/summary.csv"""
    match = re.search(r'ep_m([\d.]+)', path)
    if match:
        return float(match.group(1))
    return None

def parse_ep_summaries(glob_pattern):
    """Parse EP test summaries and extract acceleration data"""
    paths = sorted(glob.glob(glob_pattern))
    results = []
    
    print(f"[ep] Found {len(paths)} EP summary files")
    
    for p in paths:
        try:
            df = pd.read_csv(p)
            if len(df) == 0:
                continue
                
            # Take first row (single vtheta=0.30 run)
            row = df.iloc[0]
            
            # Try multiple acceleration/slope columns
            accel_val = None
            accel_col = None
            for col in ['slope_Fr', 'a_rad_mean', 'accel_r_mean', 'acc_mean', 'force_slope']:
                if col in df.columns:
                    accel_val = float(row[col])
                    accel_col = col
                    break
            
            if accel_val is None:
                print(f"Warning: No acceleration column found in {p}")
                continue
                
            # Extract mass from columns or path
            mass_val = None
            for col in ['m_test', 'mass_test', 'm4', 'm_last', 'test_mass']:
                if col in df.columns:
                    mass_val = float(row[col])
                    break
            
            if mass_val is None:
                mass_val = extract_mass_from_path(p)
                
            if mass_val is None:
                print(f"Warning: Could not extract mass from {p}")
                continue
                
            # Additional metrics
            result = {
                'path': p,
                'mass': mass_val,
                'acceleration': accel_val,
                'accel_column': accel_col,
                'dE_over_E': float(row.get('dE_over_E', np.nan)),
                'dL_over_L': float(row.get('dL_over_L', np.nan)),
                'returncode': int(row.get('returncode', -1))
            }
            
            results.append(result)
            print(f"  m={mass_val}: a={accel_val:.3f} (from {accel_col})")
            
        except Exception as e:
            print(f"Warning: Could not parse {p}: {e}")
            continue
    
    return results

def analyze_ep_spread(results):
    """Analyze equivalence principle violations"""
    if len(results) < 2:
        return None
        
    masses = np.array([r['mass'] for r in results])
    accels = np.array([r['acceleration'] for r in results])
    
    # Remove any nan/inf values
    valid = np.isfinite(accels) & np.isfinite(masses)
    if np.sum(valid) < 2:
        return None
        
    masses = masses[valid]
    accels = accels[valid]
    
    # EP spread analysis
    accel_mean = np.mean(accels)
    accel_std = np.std(accels)
    accel_range = np.max(accels) - np.min(accels)
    
    # Relative spread (key EP metric)
    rel_spread = (accel_range / abs(accel_mean)) if abs(accel_mean) > 1e-12 else np.inf
    rel_spread_pct = abs(rel_spread) * 100
    
    # Mass range coverage
    mass_range = np.max(masses) - np.min(masses)
    
    analysis = {
        'n_masses': len(masses),
        'mass_range': [float(np.min(masses)), float(np.max(masses))],
        'mass_span': float(mass_range),
        'accel_mean': float(accel_mean),
        'accel_std': float(accel_std),
        'accel_range': float(accel_range),
        'relative_spread': float(rel_spread),
        'relative_spread_percent': float(rel_spread_pct),
        'ep_violation_ok': rel_spread_pct <= 0.5  # Target: <0.5%
    }
    
    return analysis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="sweeps/ep_m*/summary.csv", 
                    help="Glob pattern for EP summary files")
    ap.add_argument("--out_json", default="figs/ep_analysis.json")
    args = ap.parse_args()
    
    results = parse_ep_summaries(args.glob)
    
    if not results:
        print("[ep] No valid EP test data found")
        return
        
    analysis = analyze_ep_spread(results)
    
    if analysis is None:
        print("[ep] Insufficient data for EP analysis")
        return
    
    # Print results
    print(f"\n[ep] Equivalence Principle Analysis:")
    print(f"  Number of masses tested: {analysis['n_masses']}")
    print(f"  Mass range: {analysis['mass_range'][0]:.1f} - {analysis['mass_range'][1]:.1f}")
    print(f"  Mean acceleration: {analysis['accel_mean']:.3f}")
    print(f"  Acceleration spread: {analysis['relative_spread_percent']:.3f}% (target: <0.5%)")
    print(f"  EP violation check: {'✓ PASS' if analysis['ep_violation_ok'] else '✗ FAIL'}")
    
    # Save detailed results
    output = {
        'summary': analysis,
        'individual_results': results,
        'acceptance_criteria': {
            'target_spread_percent': 0.5,
            'minimum_masses': 3,
            'minimum_mass_span': 1.0
        }
    }
    
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"[ep] Saved detailed analysis to {args.out_json}")

if __name__ == "__main__":
    main()