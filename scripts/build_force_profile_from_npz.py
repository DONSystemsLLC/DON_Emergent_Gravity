#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path

CAND_KEYS = [
    ("Fx","Fy","Fz"),
    ("Jx","Jy","Jz"),
    ("ax","ay","az"),
    ("gradPhi_x","gradPhi_y","gradPhi_z"),
    ("gx","gy","gz"),
]

def load_vector(npz):
    z = np.load(npz)
    for kx,ky,kz in CAND_KEYS:
        if kx in z and ky in z and kz in z:
            Vx, Vy, Vz = z[kx].astype(float), z[ky].astype(float), z[kz].astype(float)
            return Vx, Vy, Vz, (kx,ky,kz)
    raise SystemExit("No vector field found in NPZ (tried Fx/Jx/ax/gradPhi/gx variants).")

def shell_profile(r, Fr, rmin, rmax, nbins):
    edges = np.linspace(rmin, rmax, nbins+1)
    idx = np.digitize(r, edges) - 1
    rows=[]
    for i in range(nbins):
        m = idx==i
        if not np.any(m): continue
        rmid = 0.5*(edges[i]+edges[i+1])
        rows.append((rmid, np.median(Fr[m]), np.mean(Fr[m])))
    return pd.DataFrame(rows, columns=["r","Fr_med","Fr_mean"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_npz", required=True)
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--rmin", type=float, default=6.0)
    ap.add_argument("--rmax", type=float, default=36.0)
    ap.add_argument("--nbins", type=int, default=64)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    Vx, Vy, Vz, used = load_vector(args.field_npz)
    N = Vx.shape[0]; dx = args.L / N
    ax = np.arange(N) - N/2
    X,Y,Z = np.meshgrid(ax,ax,ax, indexing="ij")
    X *= dx; Y *= dx; Z *= dx
    r = np.sqrt(X*X + Y*Y + Z*Z) + 1e-12
    rhat_x, rhat_y, rhat_z = X/r, Y/r, Z/r

    # radial component of vector field
    Fr = Vx*rhat_x + Vy*rhat_y + Vz*rhat_z

    prof = shell_profile(r, Fr, args.rmin, args.rmax, args.nbins)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    prof.to_csv(args.out_csv, index=False)
    print(f"[force-profile] wrote {args.out_csv} using keys {used}")

if __name__ == "__main__":
    main()