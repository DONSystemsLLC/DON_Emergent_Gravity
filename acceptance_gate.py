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
import pandas as pd
import argparse

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

def _poly_slope(x, y):
    """Compute power law slope from log-log fit."""
    import numpy as np
    m = np.isfinite(x) & np.isfinite(y) & (y != 0) & (x > 0)
    if m.sum() < 2:
        return None
    x = np.log(x[m])
    y = np.log(np.abs(y[m]))
    return float(np.polyfit(x, y, 1)[0])

def compute_slope_from_csv(path, col_hint=None):
    """Extract slope from CSV with flexible column detection."""
    import pandas as pd
    cand_order = [col_hint, "Fr", "F_r", "a_r", "gradPhi_r", "g_r", "ar", "ar_med"]
    df = pd.read_csv(path)

    # Find r column
    r_col = None
    for rc in ["r", "r_mid"]:
        if rc in df.columns:
            r_col = rc
            break
    if not r_col:
        raise ValueError("No 'r' or 'r_mid' column found in CSV")

    # Find force/acceleration column
    for c in cand_order:
        if c and c in df.columns:
            slope = _poly_slope(df[r_col].to_numpy(), df[c].to_numpy())
            if slope is not None:
                return slope, c

    raise ValueError("No suitable force/accel column found. Provide --slope_column.")

def flux_metric(path, rmin, rmax, use_median=True):
    """Compute flux constancy metrics."""
    import pandas as pd, numpy as np
    d = pd.read_csv(path)

    # Find r column
    if "r" in d.columns:
        r = d["r"].to_numpy()
    elif "r_mid" in d.columns:
        r = d["r_mid"].to_numpy()
    else:
        raise ValueError("No 'r' or 'r_mid' column in flux CSV")

    # Find Jr column (prefer median if available)
    if use_median and "Jr_med" in d.columns:
        Jr = d["Jr_med"].to_numpy()
    elif use_median and "J_med" in d.columns:
        Jr = d["J_med"].to_numpy()
    elif "Jr" in d.columns:
        Jr = d["Jr"].to_numpy()
    elif "J_med" in d.columns:
        Jr = d["J_med"].to_numpy()
    else:
        raise ValueError("No Jr/Jr_med/J_med column in flux CSV")

    m = (r >= rmin) & (r <= rmax) & np.isfinite(Jr) & (Jr != 0)
    if m.sum() == 0:
        raise ValueError(f"No valid flux data in range [{rmin}, {rmax}]")

    F = (r[m]**2) * Jr[m]  # Gauss-law flux
    center = np.median(F) if use_median else np.mean(F)
    dev = np.abs(F/center - 1.0)
    return float(np.percentile(dev, 95)), float(np.sqrt((dev**2).mean()))

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

def main():
    """Run all acceptance checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", help="Path to summary CSV from sweep")

    # Slope arguments
    parser.add_argument("--slope_csv", type=str, default=None,
                        help="CSV with radial profiles (e.g. r, Fr or a_r or gradPhi_r).")
    parser.add_argument("--slope_column", type=str, default=None,
                        help="Column to use for slope (defaults: Fr|F_r|a_r|gradPhi_r|g_r).")
    parser.add_argument("--slope_target", type=float, default=-2.0)
    parser.add_argument("--slope_tol", type=float, default=0.05)

    # Flux arguments
    parser.add_argument("--flux_csv", type=str, default=None,
                        help="CSV with r and Jr (or Jr_med); we test flatness of r^2 ⟨J_r⟩.")
    parser.add_argument("--flux_rmin", type=float, default=10.0)
    parser.add_argument("--flux_rmax", type=float, default=28.0)
    parser.add_argument("--flux_tol", type=float, default=0.05)  # ±5% p95 dev
    parser.add_argument("--flux_use_median", type=int, default=1)

    # Other arguments
    parser.add_argument("--max_prec", type=float, default=0.05,
                        help="Maximum allowed precession (deg/orbit)")

    args = parser.parse_args()

    print("=" * 60)
    print("DON EMERGENT GRAVITY - ACCEPTANCE GATE")
    print("=" * 60)
    print()

    all_passed = True
    results = {}

    # 1. Force law slope
    print("1. Force Law Slope Check")
    if args.slope_csv and os.path.exists(args.slope_csv):
        try:
            p, used_col = compute_slope_from_csv(args.slope_csv, args.slope_column)
            okp = abs(p - args.slope_target) <= args.slope_tol
            status = "✓ PASS" if okp else "✗ FAIL"
            print(f"   {status}: p = {p:.3f} using [{used_col}]  "
                  f"(target {args.slope_target:+.1f} ± {args.slope_tol:.2f})")
            results["slope"] = okp
            all_passed = all_passed and okp

            # Hint for wrong slope type
            if not okp and abs(p) < 0.3:
                print("   hint: slope≈0 suggests you fed a flux (r²⟨J_r⟩) series; "
                      "pass --slope_column Fr|a_r|gradPhi_r to check the force-law (target −2).")
            if not okp and abs(p) > 4:
                print("   hint: |slope|≫2 likely wrong column (e.g., derivative metric). "
                      "Pass --slope_column explicitly.")
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            all_passed = False
    else:
        print("   ⚠ SKIP: Slope analysis not provided")

    print()

    # 2. Orbit metrics
    print("2. Orbit Conservation Checks")

    # If CSV provided, extract metrics from it
    if args.summary_csv and os.path.exists(args.summary_csv):
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
    if args.flux_csv and os.path.exists(args.flux_csv):
        try:
            p95, rms = flux_metric(args.flux_csv, args.flux_rmin, args.flux_rmax,
                                  bool(args.flux_use_median))
            okflux = (p95 <= args.flux_tol)
            status = "✓ PASS" if okflux else "✗ FAIL"
            print(f"   {status}: r²⟨J_r⟩ flatness in [{args.flux_rmin},{args.flux_rmax}]")
            print(f"   p95={p95:.3f}, RMS={rms:.3f} (tol ±{args.flux_tol:.2f})")
            results["flux"] = okflux
            all_passed = all_passed and okflux
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            all_passed = False
    else:
        print("   ⚠ SKIP: Flux CSV not provided")

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