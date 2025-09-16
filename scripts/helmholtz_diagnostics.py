#!/usr/bin/env python3
# scripts/helmholtz_diagnostics.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def curl_components(Fx, Fy, Fz, dx):
    # central diff with PBC
    roll = lambda a, s, ax: np.roll(a, s, axis=ax)
    dFz_dy = (roll(Fz, -1, 1) - roll(Fz, 1, 1)) / (2*dx)
    dFy_dz = (roll(Fy, -1, 2) - roll(Fy, 1, 2)) / (2*dx)
    dFx_dz = (roll(Fx, -1, 2) - roll(Fx, 1, 2)) / (2*dx)
    dFz_dx = (roll(Fz, -1, 0) - roll(Fz, 1, 0)) / (2*dx)
    dFy_dx = (roll(Fy, -1, 0) - roll(Fy, 1, 0)) / (2*dx)
    dFx_dy = (roll(Fx, -1, 1) - roll(Fx, 1, 1)) / (2*dx)
    cx = dFz_dy - dFy_dz
    cy = dFx_dz - dFz_dx
    cz = dFy_dx - dFx_dy
    return cx, cy, cz

def load_J(npz_path):
    z = np.load(npz_path)
    # Common key aliases
    cand = [
        ("Jx","Jy","Jz"),
        ("Fx","Fy","Fz"),
        ("gradPhi_x","gradPhi_y","gradPhi_z"),
        ("ax","ay","az"),  # acceleration field
        ("gx","gy","gz"),
    ]
    for kx,ky,kz in cand:
        if kx in z and ky in z and kz in z:
            return z[kx].astype(np.float64), z[ky].astype(np.float64), z[kz].astype(np.float64)
    raise KeyError("No vector field found. Expected one of (Jx,Jy,Jz)/(Fx,Fy,Fz)/…")

def shell_average(r, vals, rmin, rmax, nbins=48):
    edges = np.linspace(rmin, rmax, nbins+1)
    idx = np.digitize(r, edges)-1
    out = []
    for i in range(nbins):
        m = idx==i
        if not np.any(m): continue
        rmid = 0.5*(edges[i]+edges[i+1])
        out.append((rmid, np.mean(vals[m]), np.median(vals[m])))
    return np.array(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field_npz", required=True, help="NPZ snapshot with vector field")
    ap.add_argument("--L", type=float, required=True, help="box length")
    ap.add_argument("--rmin", type=float, default=12.0)
    ap.add_argument("--rmax", type=float, default=32.0)
    ap.add_argument("--out_csv", default="figs/helmholtz_checks.csv")
    ap.add_argument("--out_json", default="figs/helmholtz_checks.json")
    args = ap.parse_args()

    Jx, Jy, Jz = load_J(args.field_npz)
    N = Jx.shape[0]
    dx = args.L / N

    # Coordinates centered at box center
    ax = np.arange(N) - N/2
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    X = (X*dx); Y = (Y*dx); Z = (Z*dx)
    r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-12
    rhat = np.stack([X/r, Y/r, Z/r], axis=-1)

    # |J| and tangential fraction
    J = np.stack([Jx,Jy,Jz], axis=-1)
    Jmag = np.linalg.norm(J, axis=-1) + 1e-18
    Jr = (J[...,0]*rhat[...,0] + J[...,1]*rhat[...,1] + J[...,2]*rhat[...,2])
    Jtan_mag = np.sqrt(np.clip(Jmag**2 - Jr**2, 0.0, None))
    tau_frac = (Jtan_mag / Jmag)

    # curl(J)
    cx, cy, cz = curl_components(Jx,Jy,Jz,dx)
    curlJ = np.sqrt(cx*cx + cy*cy + cz*cz)

    # shell averages in [rmin,rmax]
    sel = (r>=args.rmin) & (r<=args.rmax)
    r_sel = r[sel]; Rcurl = (curlJ/Jmag)[sel]; Tau = tau_frac[sel]
    RadErr = 1.0 - np.abs(Jr[sel])/Jmag[sel]

    A1 = shell_average(r_sel, Rcurl, args.rmin, args.rmax)
    A2 = shell_average(r_sel, Tau,   args.rmin, args.rmax)
    A3 = shell_average(r_sel, RadErr,args.rmin, args.rmax)

    # write CSV
    df = pd.DataFrame({
        "r_mid": A1[:,0],
        "Rcurl_mean": A1[:,1], "Rcurl_median": A1[:,2],
        "tau_frac_mean": A2[:,1], "tau_frac_median": A2[:,2],
        "radiality_err_mean": A3[:,1], "radiality_err_median": A3[:,2]
    })
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # quick pass/fail in JSON for the annulus
    summary = {
        "window": [args.rmin, args.rmax],
        "Rcurl_mean": float(np.mean(Rcurl)),
        "tau_frac_mean": float(np.mean(Tau)),
        "radiality_err_mean": float(np.mean(RadErr))
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"[helmholtz] wrote {args.out_csv} and {args.out_json}")
    print(f"[helmholtz] annulus means: Rcurl={summary['Rcurl_mean']:.3e}, "
          f"tau_frac={summary['tau_frac_mean']:.3e}, radiality_err={summary['radiality_err_mean']:.3e}")

if __name__ == "__main__":
    main()