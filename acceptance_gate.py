#!/usr/bin/env python3
"""
Acceptance gate for DON Emergent Gravity validation.
Checks if all physics criteria are met for publication/release.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

# Acceptance criteria thresholds
CRITERIA = {
    "slope": {
        "target": -2.0,
        "tolerance": 0.05,
        "description": "Force law slope (1/r²)"
    },
    "dE_over_E": {
        "max": 1e-5,
        "description": "Energy conservation"
    },
    "dL_over_L": {
        "max": 1e-3,
        "description": "Angular momentum conservation"
    },
    "precession": {
        "max": 0.05,
        "description": "Precession (deg/orbit)"
    },
    "curl_fraction": {
        "max": 1e-6,
        "description": "Curl-free field quality"
    },
    "flux_flatness": {
        "range": [0.9, 1.1],
        "description": "Flux constancy (r²⟨J_r⟩)"
    }
}

def check_slope(profile_file="N320_L160_box_slope_profile.csv"):
    """Check force law slope from profile analysis."""
    if not os.path.exists(profile_file):
        return None, "Profile file not found"

    # Read the slope from the analysis output
    # Looking for lines like "Fitted slope: p = -2.052 ± 0.050"
    slope_file = profile_file.replace("_profile.csv", "_analysis.txt")
    if os.path.exists(slope_file):
        with open(slope_file, 'r') as f:
            for line in f:
                if "Fitted slope" in line:
                    parts = line.split("=")[1].split("±")
                    slope = float(parts[0].strip())
                    error = float(parts[1].strip())
                    passed = abs(slope - CRITERIA["slope"]["target"]) <= CRITERIA["slope"]["tolerance"]
                    return passed, f"p = {slope:.3f} ± {error:.3f}"

    return None, "Slope analysis not found"

def check_orbit_metrics(metrics_file="orbit_metrics.json"):
    """Check orbit conservation metrics."""
    results = {}

    if not os.path.exists(metrics_file):
        # Try to extract from latest log
        log_files = sorted(Path(".").glob("orbit_*.log"), key=os.path.getmtime)
        if log_files:
            with open(log_files[-1], 'r') as f:
                for line in f:
                    if "METRICS|" in line:
                        # Parse METRICS|dE_over_E=X|dL_over_L=Y|...
                        parts = line.split("|")
                        for part in parts[1:]:
                            if "=" in part:
                                key, val = part.split("=")
                                try:
                                    results[key] = float(val)
                                except:
                                    pass
    else:
        with open(metrics_file, 'r') as f:
            results = json.load(f)

    checks = {}
    if "dE_over_E" in results:
        val = abs(results["dE_over_E"])
        checks["dE_over_E"] = (val <= CRITERIA["dE_over_E"]["max"], f"{val:.2e}")

    if "dL_over_L" in results:
        val = abs(results["dL_over_L"])
        checks["dL_over_L"] = (val <= CRITERIA["dL_over_L"]["max"], f"{val:.2e}")

    if "precession_deg_per_orbit" in results:
        val = abs(results["precession_deg_per_orbit"])
        checks["precession"] = (val <= CRITERIA["precession"]["max"], f"{val:.3f}°")

    return checks

def check_curl_free(field_file="fields/N320_L160_box.npz"):
    """Check curl-free nature of field."""
    if not os.path.exists(field_file):
        return None, "Field file not found"

    # Look for curl analysis in logs
    log_files = sorted(Path(".").glob("*curl*.log"), key=os.path.getmtime)
    if log_files:
        with open(log_files[-1], 'r') as f:
            for line in f:
                if "tau_frac" in line or "curl fraction" in line.lower():
                    # Extract the value
                    import re
                    match = re.search(r'[\d.]+e[+-]\d+', line)
                    if match:
                        val = float(match.group())
                        passed = val <= CRITERIA["curl_fraction"]["max"]
                        return passed, f"τ_frac = {val:.2e}"

    return None, "Curl analysis not found"

def check_flux_constancy(profile_file="N320_L160_box_slope_profile.csv"):
    """Check flux constancy from profile."""
    import pandas as pd

    if not os.path.exists(profile_file):
        return None, "Profile file not found"

    df = pd.read_csv(profile_file)
    if "F_rel" in df.columns:
        f_min = df["F_rel"].min()
        f_max = df["F_rel"].max()
        r_min, r_max = CRITERIA["flux_flatness"]["range"]
        passed = (f_min >= r_min) and (f_max <= r_max)
        return passed, f"[{f_min:.3f}, {f_max:.3f}]"

    return None, "Flux data not found"

def main():
    """Run all acceptance checks."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", help="Path to summary CSV from sweep")
    args = parser.parse_args()

    print("=" * 60)
    print("DON EMERGENT GRAVITY - ACCEPTANCE GATE")
    print("=" * 60)
    print()

    all_passed = True
    results = {}

    # 1. Force law slope
    print("1. Force Law Slope Check")
    passed, msg = check_slope()
    if passed is not None:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {msg}")
        print(f"   Target: {CRITERIA['slope']['target']} ± {CRITERIA['slope']['tolerance']}")
        results["slope"] = passed
        all_passed = all_passed and passed
    else:
        print(f"   ⚠ SKIP: {msg}")

    print()

    # 2. Orbit metrics
    print("2. Orbit Conservation Checks")

    # If CSV provided, extract metrics from it
    if args.summary_csv and os.path.exists(args.summary_csv):
        import pandas as pd
        df = pd.read_csv(args.summary_csv)
        # Use the first row if multiple
        if len(df) > 0:
            row = df.iloc[0]
            orbit_checks = {}
            if "dE_over_E" in df.columns:
                val = abs(row["dE_over_E"])
                orbit_checks["dE_over_E"] = (val <= CRITERIA["dE_over_E"]["max"], f"{val:.2e}")
            if "dL_over_L" in df.columns:
                val = abs(row["dL_over_L"])
                orbit_checks["dL_over_L"] = (val <= CRITERIA["dL_over_L"]["max"], f"{val:.2e}")
            if "precession_deg_per_orbit" in df.columns:
                val = abs(row["precession_deg_per_orbit"])
                if not np.isnan(val):
                    orbit_checks["precession"] = (val <= CRITERIA["precession"]["max"], f"{val:.3f}°")
        else:
            orbit_checks = check_orbit_metrics()
    else:
        orbit_checks = check_orbit_metrics()

    for key, (passed, val) in orbit_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        desc = CRITERIA[key]["description"]
        threshold = CRITERIA[key]["max"]
        print(f"   {status}: {desc}: {val} (max: {threshold:.0e})")
        results[key] = passed
        all_passed = all_passed and passed

    if not orbit_checks:
        print("   ⚠ SKIP: No orbit metrics found")

    print()

    # 3. Curl-free check
    print("3. Curl-Free Field Check")
    passed, msg = check_curl_free()
    if passed is not None:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {msg}")
        print(f"   Threshold: < {CRITERIA['curl_fraction']['max']:.0e}")
        results["curl_free"] = passed
        all_passed = all_passed and passed
    else:
        print(f"   ⚠ SKIP: {msg}")

    print()

    # 4. Flux constancy
    print("4. Flux Constancy Check")
    passed, msg = check_flux_constancy()
    if passed is not None:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: Normalized flux range: {msg}")
        print(f"   Target: {CRITERIA['flux_flatness']['range']}")
        results["flux"] = passed
        all_passed = all_passed and passed
    else:
        print(f"   ⚠ SKIP: {msg}")

    print()
    print("=" * 60)

    # Summary
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    if all_passed and total_count >= 4:
        print("GATE STATUS: ✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print(f"Score: {passed_count}/{total_count}")
        print("\nSystem ready for publication/release!")
        sys.exit(0)
    else:
        print(f"GATE STATUS: ✗ FAILED ({passed_count}/{total_count} passed)")
        if total_count < 4:
            print("Note: Some checks were skipped. Run full validation suite.")
        print("\nReview failed criteria and rerun validation.")
        sys.exit(1)

if __name__ == "__main__":
    main()