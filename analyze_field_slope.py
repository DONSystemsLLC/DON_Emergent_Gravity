#!/usr/bin/env python3
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load the warm field snapshot
SNAP = Path("fields/N320_L160_box")
mu = np.load(SNAP.with_suffix(".mu.npy"))
Jx = np.load(SNAP.with_suffix(".Jx.npy"))
Jy = np.load(SNAP.with_suffix(".Jy.npy"))
Jz = np.load(SNAP.with_suffix(".Jz.npy"))

# Field parameters
N = 320  # Grid size
L = 160  # Half-length

# Build radial bins around box center, compute |J| medians per bin
cx = cy = cz = N//2
Jmag = np.sqrt(Jx**2 + Jy**2 + Jz**2)
yy, xx, zz = np.indices(Jmag.shape)
r = np.sqrt((xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2)
r_phys = r * (2*L/N)  # Physical radius

# Match slope window [5, 20]
mask = (r_phys > 5) & (r_phys < 20)
r_bins = np.linspace(5, 20, 61)
rb = np.digitize(r_phys[mask], r_bins)
r_mid = 0.5 * (r_bins[1:] + r_bins[:-1])

# Calculate medians
Jm = []
for k in range(1, len(r_bins)):
    bin_values = Jmag[mask][rb == k]
    if len(bin_values) > 0:
        Jm.append(np.median(bin_values))
    else:
        Jm.append(np.nan)

r_mid = np.array(r_mid)
Jm = np.array(Jm)

# Fit log-log slope
m = (np.isfinite(Jm)) & (Jm > 0)
if np.sum(m) > 2:
    coeffs = np.polyfit(np.log10(r_mid[m]), np.log10(Jm[m]), 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    print(f"log10(|J|) ~ p*log10(r) + C")
    print(f"Slope p = {slope:.3f}")
    print(f"Target: p ≈ -2.00 ± 0.05")
    print(f"Status: {'PASS' if abs(slope + 2.0) < 0.05 else 'MARGINAL' if abs(slope + 2.0) < 0.10 else 'FAIL'}")

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Data points
    ax.loglog(r_mid[m], Jm[m], 'o', label='|J| medians', alpha=0.7)

    # Fit line
    r_fit = np.logspace(np.log10(5), np.log10(20), 100)
    J_fit = 10**(slope * np.log10(r_fit) + intercept)
    ax.loglog(r_fit, J_fit, '-', label=f'Fit: slope={slope:.3f}', linewidth=2)

    # Reference -2 slope
    J_ref = J_fit[0] * (r_fit / r_fit[0])**(-2)
    ax.loglog(r_fit, J_ref, '--', label='1/r² reference', alpha=0.5)

    ax.set_xlabel('r (physical units)')
    ax.set_ylabel('|J| (flux magnitude)')
    ax.set_title(f'Field Law Verification: log|J| ~ {slope:.3f} log(r)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig('field_slope_warm.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: field_slope_warm.png")
else:
    print("ERROR: Not enough valid data points for fit")