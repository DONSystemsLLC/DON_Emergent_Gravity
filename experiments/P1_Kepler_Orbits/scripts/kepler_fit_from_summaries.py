#!/usr/bin/env python3
import glob, pandas as pd, numpy as np, sys
from pathlib import Path
paths = sorted(glob.glob("sweeps/kepler_*/summary.csv"))
rows = []
for p in paths:
    df = pd.read_csv(p)
    # try common column names; tweak if your parser differs
    T = df.filter(regex=r"(^T$|period)", axis=1).iloc[0,0] if len(df.filter(regex=r"(^T$|period)", axis=1).columns) > 0 else None
    a = df.filter(regex=r"(semi_?major|^a$|r_mean)", axis=1).iloc[0,0] if len(df.filter(regex=r"(semi_?major|^a$|r_mean)", axis=1).columns) > 0 else None
    flat = df.filter(regex=r"(vtheta2r_flat|flatness)", axis=1).iloc[0,0] if len(df.filter(regex=r"(vtheta2r_flat|flatness)", axis=1).columns) > 0 else None
    if T and a:
        rows.append((a, T, flat if flat else 0.0))

if rows:
    arr = np.array(rows)
    logA, logT2 = np.log(arr[:,0]), np.log(arr[:,1]**2)
    slope = np.polyfit(logA, logT2, 1)[0]
    flatness = np.mean(arr[:,2]) if arr[:,2].any() else 0.0
    print(f"[kepler] slope dlog(T^2)/dlog(a) = {slope:.3f}  (target 1.500±0.03)")
    print(f"[kepler] mean vθ²r flatness metric = {flatness:.3f} (target |…|≤0.03)")
    out = Path("figs/kepler_fit.txt"); out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(f"slope={slope:.6f}\nflatness={flatness:.6f}\n")
else:
    print("[kepler] No valid Kepler data found yet")