#!/usr/bin/env python3
"""
clens_stack.py — Cluster lensing stacker (amplitude-ready)

Inputs (in --work):
  clusters.parquet  : columns ['ra','dec','z_cl', ...]
  sources.parquet   : columns ['ra','dec','e1','e2','w','z_src', ('m' optional)]

Outputs (in --out):
  profile_focus.csv
  profile_control.csv

Per-bin columns:
  R_Mpc, gt, gx, W, npairs, N_src, z_mean, (K_1p if m available)

Key features:
  - Σcrit weighting per pair: w_tot = w_shape * beta^2, where beta = Σcrit^{-1}(z_l,z_s)
  - background cut: z_src > z_l + zbuf  (default 0.2)
  - optional m-calibration: gt_corr = gt / (1+Kbin), where Kbin = sum(w*m)/sum(w) (if 'm' present and --apply_m)
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as COSMO

# -----------------------------
# Cosmology & lensing helpers
# -----------------------------
def sigma_crit_inv(zl: float, zs: float) -> float:
    """
    Return Σ_crit^{-1} in (Mpc^2 / Msun) units:
      Σ_crit = (c^2 / 4πG) * Ds / (Dl Dls)
    We return its inverse: (4πG/c^2) * (Dl Dls / Ds)

    Using angular diameter distances from Planck18; G/c^2 is folded in a prefactor
    for correct physical units. For stacking weights we only need the RELATIVE
    scaling, but we keep physical norm for completeness.
    """
    if zs <= zl:
        return 0.0
    Dl  = COSMO.angular_diameter_distance(zl).to_value('Mpc')
    Ds  = COSMO.angular_diameter_distance(zs).to_value('Mpc')
    Dls = COSMO.angular_diameter_distance_z1z2(zl, zs).to_value('Mpc')
    if Dls <= 0.0 or Ds <= 0.0:
        return 0.0
    # Physical prefactor: 4πG / c^2  in (Mpc / Msun)
    # G = 4.300917270e-9 (Mpc Msun^-1) (km/s)^2 ; c = 299792.458 km/s
    G_Mpc_Msun = 4.300917270e-9  # (Mpc Msun^-1)(km/s)^2
    c_km_s     = 299792.458
    pref = 4.0 * np.pi * G_Mpc_Msun / (c_km_s**2)  # Mpc / Msun
    return pref * (Dl * Dls / Ds)  # Mpc^2 / Msun

def angular_components(ra_l, dec_l, ra_s, dec_s):
    """
    Return position angle phi and angular separation (deg) of sources vs lens.
    phi = arctan2(Δdec, Δra*cos dec_l) ; θ = sqrt( (Δra cos)^2 + (Δdec)^2 ) [deg]
    """
    dra  = (ra_s - ra_l) * np.cos(np.deg2rad(dec_l))
    ddec = (dec_s - dec_l)
    phi  = np.arctan2(ddec, dra)
    theta_deg = np.hypot(dra, ddec)
    return phi, theta_deg

def tangential_cross(e1, e2, phi):
    """Tangential and cross shear from source ellipticities and position angle."""
    cos2p = np.cos(2.0*phi); sin2p = np.sin(2.0*phi)
    gt = -(e1*cos2p + e2*sin2p)
    gx = -(-e1*sin2p + e2*cos2p)
    return gt, gx

def R_phys_mpc(zl, theta_deg):
    """Physical radius R [Mpc] at lens redshift zl."""
    Da = COSMO.angular_diameter_distance(float(zl)).to_value('Mpc')
    return Da * np.deg2rad(theta_deg)

# -----------------------------
# Stacking core
# -----------------------------
def build_profile(lenses: pd.DataFrame, sources: pd.DataFrame,
                  r_edges: np.ndarray, zbuf: float, apply_m: bool) -> pd.DataFrame:
    """
    Build a Σcrit-weighted tangential shear profile over r_edges.
    Returns DataFrame with: R_Mpc, gt, gx, W, npairs, N_src, z_mean, (K_1p if available)
    """
    if lenses.empty or sources.empty:
        return pd.DataFrame(columns=["R_Mpc","gt","gx","W","npairs","N_src","z_mean","K_1p"])

    rows = []
    has_m = ('m' in sources.columns)

    for _, L in lenses.iterrows():
        zl = float(L["z_cl"])
        # hard background cut
        s = sources.loc[sources["z_src"] > zl + zbuf]
        if s.empty:
            continue

        # geometry
        phi, theta_deg = angular_components(float(L["ra"]), float(L["dec"]),
                                            s["ra"].to_numpy(), s["dec"].to_numpy())
        R   = R_phys_mpc(zl, theta_deg)
        gt, gx = tangential_cross(s["e1"].to_numpy(), s["e2"].to_numpy(), phi)

        # weights: shape × beta^2, where beta = Σcrit^{-1}(zl,zs)
        zs   = s["z_src"].to_numpy()
        beta = np.array([sigma_crit_inv(zl, zsi) for zsi in zs])
        w_sh = s["w"].to_numpy().astype(np.float64)
        w    = w_sh * (beta**2)

        # (optional) m-calibration per pair — we aggregate per bin to get (1+K)
        mvals = s["m"].to_numpy() if has_m else None

        bins = np.digitize(R, r_edges) - 1
        for b in range(len(r_edges)-1):
            m = (bins == b)
            if not np.any(m):
                continue
            w_b = w[m].sum()
            if w_b <= 0:
                continue

            # Weighted sums for gt/gx and representative R; also per-bin K if m is present
            gt_ws = np.sum(gt[m] * w[m])
            gx_ws = np.sum(gx[m] * w[m])
            R_ws  = np.sum(R[m]  * w[m])
            # Kbin: sum(w_shape * m) / sum(w_shape) ; then 1+K used to correct gt at the end if requested
            if has_m:
                wsh_b = w_sh[m].sum()
                K_1p  = 1.0 + (np.sum(w_sh[m] * mvals[m]) / wsh_b) if wsh_b > 0 else 1.0
            else:
                K_1p  = np.nan

            rows.append({
                "bin": b,
                "R_lo": r_edges[b], "R_hi": r_edges[b+1],
                "R_wsum": R_ws,
                "gt_wsum": gt_ws,
                "gx_wsum": gx_ws,
                "W": w_b,
                "npairs": int(np.sum(m)),
                "N_src":  int(np.sum(m)),  # raw source count; used for boost factor
                "z_wsum": float(zl) * w_b,
                "K_1p_sum": K_1p * w_b if np.isfinite(K_1p) else np.nan,
                "K_w": w_b if np.isfinite(K_1p) else np.nan,
            })

    if not rows:
        return pd.DataFrame(columns=["R_Mpc","gt","gx","W","npairs","N_src","z_mean","K_1p"])

    df = pd.DataFrame(rows)

    # aggregate by bin with pure sums (no groupby.apply → no FutureWarning)
    g = df.groupby("bin", sort=True)

    W      = g["W"].sum()
    R_ws   = g["R_wsum"].sum()
    gt_ws  = g["gt_wsum"].sum()
    gx_ws  = g["gx_wsum"].sum()
    npairs = g["npairs"].sum()
    N_src  = g["N_src"].sum()
    z_ws   = g["z_wsum"].sum()

    R_mpc  = R_ws / np.where(W>0, W, np.nan)
    gt     = gt_ws / np.where(W>0, W, np.nan)
    gx     = gx_ws / np.where(W>0, W, np.nan)
    z_mean = z_ws / np.where(W>0, W, np.nan)

    prof = pd.DataFrame({
        "R_Mpc":  R_mpc,
        "gt":     gt,
        "gx":     gx,
        "W":      W,
        "npairs": npairs.astype(int),
        "N_src":  N_src.astype(int),
        "z_mean": z_mean
    }).reset_index(drop=True)

    # Optional m-calibration (divide gt by (1+Kbin))
    if "K_1p_sum" in df.columns and "K_w" in df.columns:
        K_1p = (g["K_1p_sum"].sum()) / np.where(g["K_w"].sum()>0, g["K_w"].sum(), np.nan)
        prof["K_1p"] = K_1p.values
        if apply_m:
            prof["gt"] = prof["gt"] / prof["K_1p"]

    prof = prof.replace([np.inf, -np.inf], np.nan).dropna(subset=["R_Mpc","gt","W"])
    return prof

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="dir with clusters.parquet & sources.parquet")
    ap.add_argument("--out",  required=True, help="output dir")
    ap.add_argument("--z_focus", required=True, help="zmin,zmax for focus (e.g., 0.12,0.16)")
    ap.add_argument("--z_ctrl",  required=True, help="semicolon-separated zmin,zmax bands for control (e.g., 0.06,0.10;0.18,0.30)")
    ap.add_argument("--zbuf", type=float, default=0.2, help="background cut: z_src > z_l + zbuf")
    ap.add_argument("--rmin", type=float, default=0.1, help="min R [Mpc]")
    ap.add_argument("--rmax", type=float, default=2.0, help="max R [Mpc]")
    ap.add_argument("--nbin", type=int,   default=8,   help="number of radial bins")
    ap.add_argument("--apply_m", action="store_true", help="apply per-bin m-calibration if 'm' is present")
    args = ap.parse_args()

    work = pathlib.Path(args.work); out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cl = pd.read_parquet(work/"clusters.parquet")
    s  = pd.read_parquet(work/"sources.parquet")

    zfmin, zfmax = map(float, args.z_focus.split(","))
    zctrl        = [tuple(map(float, seg.split(","))) for seg in args.z_ctrl.split(";")]

    r_edges = np.logspace(np.log10(args.rmin), np.log10(args.rmax), args.nbin+1)

    # Focus
    cl_f = cl[(cl["z_cl"]>=zfmin) & (cl["z_cl"]<=zfmax)].copy()
    pf   = build_profile(cl_f, s, r_edges, zbuf=args.zbuf, apply_m=args.apply_m)
    pf.to_csv(out/"profile_focus.csv", index=False)

    # Control (union of windows)
    cl_c_parts = []
    for (z0, z1) in zctrl:
        part = cl[(cl["z_cl"]>=z0) & (cl["z_cl"]<=z1)]
        if not part.empty: cl_c_parts.append(part)
    cl_c = pd.concat(cl_c_parts, ignore_index=True) if cl_c_parts else pd.DataFrame(columns=cl.columns)
    pc   = build_profile(cl_c, s, r_edges, zbuf=args.zbuf, apply_m=args.apply_m) if not cl_c.empty else pd.DataFrame(columns=["R_Mpc","gt","gx","W","npairs","N_src","z_mean","K_1p"])
    pc.to_csv(out/"profile_control.csv", index=False)

    # Report band-mean
    def band_mean(df, lo=0.5, hi=1.8):
        if df.empty: return np.nan
        m = (df["R_Mpc"]>=lo) & (df["R_Mpc"]<=hi)
        return float(df.loc[m, "gt"].mean()) if np.any(m) else np.nan

    mF = band_mean(pf); mC = band_mean(pc)
    if np.isfinite(mF) and np.isfinite(mC):
        print(f"[effect] focus vs control (0.5–1.8 Mpc): rel = {mF - mC:+.4f}")
    else:
        print("[effect] insufficient data")

if __name__ == "__main__":
    main()

