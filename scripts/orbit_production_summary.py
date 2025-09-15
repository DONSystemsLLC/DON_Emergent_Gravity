#!/usr/bin/env python3
"""
Post-run analyzer for production orbit to extract conservation metrics and precession.
Reads trajectory.csv and orbit.out from the production run directory.
"""
import numpy as np
import pandas as pd
import re
from pathlib import Path

def main():
    # Output directory from production run
    OUT = Path("outputs/orbits/N320_L160_box_best")
    traj_file = OUT / "trajectory.csv"
    log_file = OUT / "orbit.out"

    # Create proof directory
    proof_dir = Path("proofs/EMERGENT_GRAVITY_ORBITS_N320_L160")
    proof_dir.mkdir(parents=True, exist_ok=True)

    # 1) Analyze precession from periapse-to-periapse
    precession_deg = np.nan
    precession_std = np.nan

    if traj_file.exists():
        df = pd.read_csv(traj_file)
        if 'x' in df.columns and 'y' in df.columns:
            x = df["x"].values
            y = df["y"].values
            r = np.sqrt(x*x + y*y)
            theta = np.unwrap(np.arctan2(y, x))

            # Find periapse points (local minima in r)
            mins = (r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])
            idx = np.where(mins)[0] + 1

            if len(idx) > 1:
                # Calculate angle change between successive periapses
                dtheta = np.diff(theta[idx])  # radians per orbit
                # Precession is the deviation from 2π
                prec_rad = dtheta - 2*np.pi
                precession_deg = np.degrees(np.mean(prec_rad))
                precession_std = np.degrees(np.std(prec_rad))

    # 2) Parse conservation metrics from log file
    dE_over_E = np.nan
    dL_over_L = np.nan

    if log_file.exists():
        try:
            txt = log_file.read_text(errors="ignore")

            # Look for mean conservation metrics
            mE = re.search(r"mean\s+\|dE/E0\|\s*≈\s*([0-9eE\+\-\.]+)", txt)
            if not mE:
                # Alternative pattern
                mE = re.search(r"mean_abs_dE\s*=\s*([0-9eE\+\-\.]+)", txt)

            mL = re.search(r"mean\s+\|d\|\|L\|\|/\|\|L\|\|0\|\s*≈\s*([0-9eE\+\-\.]+)", txt)
            if not mL:
                # Alternative pattern
                mL = re.search(r"mean_abs_dL\s*=\s*([0-9eE\+\-\.]+)", txt)

            if mE:
                dE_over_E = float(mE.group(1))
            if mL:
                dL_over_L = float(mL.group(1))
        except Exception as e:
            print(f"Warning: Could not parse log file: {e}")

    # 3) Write summary
    summary_file = proof_dir / "ORBIT_PRODUCTION_SUMMARY.md"

    with open(summary_file, "w") as f:
        f.write("# Production Orbit Summary (vθ=0.30)\n\n")
        f.write("## Conservation Metrics\n\n")
        f.write(f"- mean |ΔE/E| = {dE_over_E:.3e}\n")
        f.write(f"- mean |Δ‖L‖/‖L‖₀| = {dL_over_L:.3e}\n")
        f.write("\n## Orbital Dynamics\n\n")
        f.write(f"- Precession = {precession_deg:.3f} ± {precession_std:.3f} deg/orbit (periapse-to-periapse)\n")
        f.write("\n## Configuration\n\n")
        f.write("- Field: N=320, L=160 box\n")
        f.write("- Initial radius: r₀ = 20.0\n")
        f.write("- Tangential velocity: vθ = 0.30\n")
        f.write("- Timestep: dt = 0.003125\n")
        f.write("- Total steps: 120000\n")

    print(f"Wrote: {summary_file}")

    # Also print to console
    print("\n=== Production Orbit Results ===")
    print(f"Energy conservation: |ΔE/E| = {dE_over_E:.3e}")
    print(f"Angular momentum: |Δ‖L‖/‖L‖₀| = {dL_over_L:.3e}")
    print(f"Precession: {precession_deg:.3f} ± {precession_std:.3f} deg/orbit")

if __name__ == "__main__":
    main()