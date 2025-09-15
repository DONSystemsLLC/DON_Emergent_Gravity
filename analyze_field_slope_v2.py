#!/usr/bin/env python3
import numpy as np, json
from pathlib import Path
from numpy.linalg import norm

SNAP = Path("fields/N320_L160_box")
# --- NPZ/NPY loader ---
if SNAP.suffix==".npz":
    arr=np.load(SNAP, allow_pickle=True)
    mu, Jx, Jy, Jz = arr["mu"], arr["Jx"], arr["Jy"], arr["Jz"]
    meta=dict(arr["meta"].item()) if "meta" in arr else {"N":mu.shape[0], "L":160.0}
else:
    mu  = np.load(SNAP.with_suffix(".mu.npy"))
    Jx  = np.load(SNAP.with_suffix(".Jx.npy"))
    Jy  = np.load(SNAP.with_suffix(".Jy.npy"))
    Jz  = np.load(SNAP.with_suffix(".Jz.npy"))
    import json
    meta = json.loads(Path(SNAP.with_suffix(".manifest.json")).read_text())
N, L = 320, 160  # Grid size, half-length

# ---- helpers ----
def min_image_shift(dx, N):
    # map integer grid deltas to [-N/2, N/2)
    dx = (dx + N//2) % N - N//2
    return dx

# 1) center = barycenter of mu (periodic-aware)
ix = np.arange(N)
X, Y, Z = np.meshgrid(ix, ix, ix, indexing="ij")
# weights on the torus: choose origin that minimizes spread
# start from box center and refine by local COM (good enough here)
cx0 = cy0 = cz0 = N//2
dx = min_image_shift(X - cx0, N); dy = min_image_shift(Y - cy0, N); dz = min_image_shift(Z - cz0, N)
# continuous COM in grid units
sum_mu = mu.sum() + 1e-30
cx = (cx0 + (dx*mu).sum()/sum_mu) % N
cy = (cy0 + (dy*mu).sum()/sum_mu) % N
cz = (cz0 + (dz*mu).sum()/sum_mu) % N

# recompute integer deltas around refined center
dx = min_image_shift(X - int(round(cx)), N)
dy = min_image_shift(Y - int(round(cy)), N)
dz = min_image_shift(Z - int(round(cz)), N)

# physical spacing (scaling cancels in slope, but useful for radii)
dx_phys = (2*L)/N
rx = dx * dx_phys; ry = dy * dx_phys; rz = dz * dx_phys
r  = np.sqrt(rx*rx + ry*ry + rz*rz)
# avoid r=0
mask_r = r > 0

# 2) radial direction & Jr
Jr = (Jx[mask_r]*rx[mask_r] + Jy[mask_r]*ry[mask_r] + Jz[mask_r]*rz[mask_r]) / r[mask_r]
# use magnitude of radial component (outward flux) for shell stats
Jr_abs = np.abs(Jr)
rvals  = r[mask_r]

# 3) robust shell binning in a trusted window
r_inner, r_outer, nbins = 8.0, 24.0, 60   # adjusted window [8,24]
sel = (rvals >= r_inner) & (rvals <= r_outer)
r_in = rvals[sel]; Jr_in = Jr_abs[sel]
bins = np.linspace(r_inner, r_outer, nbins+1)
which = np.digitize(r_in, bins)
r_mid = 0.5*(bins[1:]+bins[:-1])

def perc(a, q):  # safe percentile
    a = a[np.isfinite(a)]
    return np.nan if a.size==0 else np.percentile(a, q)

J_med = np.array([perc(Jr_in[which==k], 50) for k in range(1, nbins+1)])
J_lo  = np.array([perc(Jr_in[which==k], 16) for k in range(1, nbins+1)])
J_hi  = np.array([perc(Jr_in[which==k], 84) for k in range(1, nbins+1)])

good = np.isfinite(J_med) & (J_med>0)
x = np.log10(r_mid[good]); y = np.log10(J_med[good])

# 4) robust slope: Theil–Sen fallback if available, else polyfit
try:
    from sklearn.linear_model import TheilSenRegressor
    X = x.reshape(-1,1)
    model = TheilSenRegressor().fit(X, y)
    p = model.coef_[0]
except:
    p = np.polyfit(x, y, 1)[0]

print(f"log10(|J_r|) ~ p*log10(r) + C  => p ≈ {p:.3f}")
# 5) flux constancy check: F(r) ∝ <|J_r|>_shell * 4π r^2 (flat if 1/r^2)
F = J_med[good] * (r_mid[good]**2)
F_rel = F / np.median(F)
print("Flux flatness (16-84% range of F/F_med):",
      np.nanpercentile(F_rel, (16,84)))
# Optional: write CSV for plots
import csv
out = SNAP.stem + "_slope_profile.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["r_mid","J_med","J_lo","J_hi","F_rel"])
    for R, Jm, jl, jh, Fr in zip(r_mid[good], J_med[good], J_lo[good], J_hi[good], F_rel):
        w.writerow([R, Jm, jl, jh, Fr])
print("Wrote:", out)