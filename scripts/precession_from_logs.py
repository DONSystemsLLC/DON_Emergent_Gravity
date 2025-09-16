#!/usr/bin/env python3
import glob, re, numpy as np, pandas as pd, sys
from pathlib import Path

def precession_deg_from_log(csv):
    df = pd.read_csv(csv)
    r = df[["rx","ry"]].to_numpy()
    v = df[["vx","vy"]].to_numpy()
    phi = np.unwrap(np.arctan2(r[:,1], r[:,0]))
    rmag = np.linalg.norm(r,axis=1)
    # find periapses by local minima in r
    idx = np.argwhere((rmag[1:-1] < rmag[:-2]) & (rmag[1:-1] < rmag[2:])).ravel()+1
    if len(idx) < 3: return np.nan
    dphi = np.diff(phi[idx])  # radians per orbit
    return np.degrees(np.median(dphi) - 2*np.pi)  # excess over 2π

rows=[]
for log in sorted(glob.glob("sweeps/precession_e*/logs/*orbit_log.csv")):
    m = re.search(r"precession_e([0-9\.]+)/", log)
    e_hint = m.group(1) if m else "?"
    dp = precession_deg_from_log(log)
    rows.append((log, e_hint, dp))
out = Path("figs/precession_results.csv"); out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["log","e_hint","precession_deg_per_orbit"]).to_csv(out, index=False)
print(f"[precession] wrote {out}")
bad = [r for r in rows if not np.isnan(r[2]) and abs(r[2])>=0.05]
print("[precession] max |Δϖ|/orbit:", np.nanmax([abs(r[2]) for r in rows]))
sys.exit(0 if not bad else 1)