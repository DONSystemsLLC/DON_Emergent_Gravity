#!/usr/bin/env python3
"""
Simple summary that computes mean shear over broad radial ranges
to avoid bin alignment issues.
"""
import glob, numpy as np, pandas as pd
from pathlib import Path

LENS = Path("outputs/CLENS_patch/profile_focus.csv")
RAND_GLOB = "outputs/CLENS_random_strat3_*/profile_focus.csv"

def load_and_average(path, r_min=0.5, r_max=1.5):
    """Load profile and compute weighted mean in radial range"""
    df = pd.read_csv(path)

    # Get R column
    rcol = "R_Mpc" if "R_Mpc" in df.columns else "R"
    if rcol not in df.columns:
        return None, None

    # Filter radial range
    mask = (df[rcol] >= r_min) & (df[rcol] <= r_max)
    df_range = df[mask]

    if len(df_range) == 0:
        return None, None

    # Get shear column
    gcol = "gt" if "gt" in df.columns else "gamma_t"
    if gcol not in df.columns:
        return None, None

    # Get weights (npairs or N_src)
    wcol = None
    for c in ["npairs", "N_src", "W"]:
        if c in df.columns:
            wcol = c
            break

    if wcol:
        weights = df_range[wcol].values
        gt_mean = np.average(df_range[gcol].values, weights=weights)
    else:
        gt_mean = df_range[gcol].mean()

    # Cross shear
    gx_mean = None
    if "gx" in df.columns:
        if wcol:
            gx_mean = np.average(df_range["gx"].values, weights=weights)
        else:
            gx_mean = df_range["gx"].mean()

    return gt_mean, gx_mean

# Check lens
if not LENS.exists():
    print("Lens profile not ready")
    exit(0)

lens_gt, lens_gx = load_and_average(LENS)
print(f"\nLens signal (0.5-1.5 Mpc):")
print(f"  gt = {lens_gt:.4e}")
print(f"  gx = {lens_gx:.4e}" if lens_gx is not None else "  gx = N/A")

# Check randoms
rand_paths = sorted(glob.glob(RAND_GLOB))
if rand_paths:
    rand_gts = []
    for rp in rand_paths:
        rgt, _ = load_and_average(rp)
        if rgt is not None:
            rand_gts.append(rgt)

    if rand_gts:
        mu = np.mean(rand_gts)
        sigma = np.std(rand_gts, ddof=1) if len(rand_gts) > 1 else 0
        z = (lens_gt - mu) / (sigma + 1e-30) if sigma > 0 else 0

        print(f"\nRandom signal (0.5-1.5 Mpc):")
        print(f"  N = {len(rand_gts)}")
        print(f"  gt_mean = {mu:.4e}")
        print(f"  gt_std = {sigma:.4e}")
        print(f"\nWL excess:")
        print(f"  Δ = {lens_gt - mu:.4e}")
        print(f"  z-score = {z:.2f}")
else:
    print("\nNo random profiles available yet")