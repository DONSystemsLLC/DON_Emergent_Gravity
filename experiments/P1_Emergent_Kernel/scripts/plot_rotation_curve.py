#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os

# Load the slope profile data from the outer window analysis
# Auto-discover the latest profile
cands = [
    "N320_L160_box_eps5e-4_slope_profile.csv",
    "N320_L160_box_slope_profile.csv",
]
cands += sorted(glob.glob("*_slope_profile.csv"), key=os.path.getmtime, reverse=True)
for _f in cands:
    if os.path.exists(_f):
        print("Using slope profile:", _f)
        df = pd.read_csv(_f)
        break
else:
    raise FileNotFoundError("No *_slope_profile.csv found")
r = df["r_mid"].values
Jr = df["J_med"].values

# Calculate rotation curve: v_c = sqrt(r * |J_r|) up to a constant
vc = np.sqrt(r * Jr)

# Match scale at r≈20 to v_theta=0.30
idx_r20 = np.argmin(np.abs(r - 20.0))
kappa = 0.30 / vc[idx_r20] if vc[idx_r20] > 0 else 1.0
vc_scaled = kappa * vc

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Rotation curve
ax1.plot(r, vc_scaled, 'b-', linewidth=2, label='v_c from field')
ax1.axhline(y=0.30, color='r', linestyle='--', alpha=0.5, label='v_θ=0.30 (orbit test)')
ax1.axvline(x=20.0, color='gray', linestyle=':', alpha=0.3)
ax1.set_xlabel('r (physical units)', fontsize=12)
ax1.set_ylabel('v_c (matched at r₀=20)', fontsize=12)
ax1.set_title('Rotation Curve from Emergent Field', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim([r.min(), r.max()])

# Right panel: Verify 1/r² scaling through flux constancy
F_rel = df["F_rel"].values
ax2.plot(r, F_rel, 'g-', linewidth=2)
ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
ax2.fill_between(r, 0.86, 1.09, alpha=0.2, color='green', label='16-84% range')
ax2.set_xlabel('r (physical units)', fontsize=12)
ax2.set_ylabel('F/F_median = J_r × r²', fontsize=12)
ax2.set_title('Flux Constancy (1/r² verification)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim([r.min(), r.max()])
ax2.set_ylim([0.7, 1.3])

plt.suptitle('Emergent Gravity from DEE Collapse: p = -2.052', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("proofs/EMERGENT_GRAVITY_ORBITS_N320_L160/rotation_curve.png", dpi=160, bbox_inches='tight')
print("Saved: proofs/EMERGENT_GRAVITY_ORBITS_N320_L160/rotation_curve.png")

# Print summary statistics
print(f"\nRotation curve at r=20: v_c = {vc_scaled[idx_r20]:.3f}")
print(f"Flux flatness range: [{F_rel.min():.3f}, {F_rel.max():.3f}]")
print(f"Slope window: r ∈ [{r.min():.1f}, {r.max():.1f}]")