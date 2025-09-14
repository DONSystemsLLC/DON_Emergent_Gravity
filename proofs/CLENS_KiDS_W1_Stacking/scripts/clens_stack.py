#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
from math import sqrt

# Flat LCDM comoving distance (quick trapezoid), sufficient for % comparison
c = 299792.458; H0 = 70.0; Om = 0.3; Ol = 1-Om
def Ez(z): return sqrt(Om*(1+z)**3 + Ol)
def Dc(z, n=512):
    if z<=0: return 0.0
    zs = np.linspace(0, z, n)
    return (c/H0)*np.trapezoid(1.0/np.vectorize(Ez)(zs), zs)

def SigmaCrit_inv(zl, zs):
    if zs <= zl + 1e-4: return 0.0
    Dl = Dc(zl)/(1+zl); Ds = Dc(zs)/(1+zs); Dls = (Dc(zs)-Dc(zl))/(1+zs)
    return max(Dls/(Dl*Ds), 0.0)  # relative weight ok

def angsep_rad(ra1,dec1,ra2,dec2):
    ra1=np.deg2rad(ra1); dec1=np.deg2rad(dec1)
    ra2=np.deg2rad(ra2); dec2=np.deg2rad(dec2)
    cosd = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    return np.arccos(np.clip(cosd, -1, 1))

def stack_subset(cl, src, r_edges, min_pairs=10, box_expand=1.5):
    r_cent  = 0.5*(r_edges[1:]+r_edges[:-1])
    out = []
    for _,crow in cl.iterrows():
        zl = float(crow["z_cl"])
        Dl = Dc(zl)/(1+zl)  # Mpc
        if Dl<=0: continue

        # prefilter a sky box (approx) for speed
        Rmax = r_edges[-1]
        theta_max = (Rmax / Dl)            # radians
        theta_max_deg = np.rad2deg(theta_max) if np.isfinite(theta_max) else 0.2
        box = src[(np.abs(src["ra"]-crow["ra"])<theta_max_deg*box_expand) &
                  (np.abs(src["dec"]-crow["dec"])<theta_max_deg*box_expand)].copy()
        if box.empty: continue

        th  = angsep_rad(crow["ra"], crow["dec"], box["ra"].to_numpy(), box["dec"].to_numpy())
        R   = Dl * th
        phi = np.arctan2((box["dec"]-crow["dec"]).to_numpy(),
                         (box["ra"] -crow["ra"]).to_numpy() * np.cos(np.deg2rad(crow["dec"])) )
        e1, e2 = box["e1"].to_numpy(), box["e2"].to_numpy()
        et     = -( e1*np.cos(2*phi) + e2*np.sin(2*phi) )

        # lensing weights
        z_s = box["z_src"].to_numpy()
        inv_Sc = np.array([SigmaCrit_inv(zl, zs) for zs in z_s])
        w  = box["w"].to_numpy() * inv_Sc**2
        m  = (R>=r_edges[0]) & (R<r_edges[-1]) & (w>0) & np.isfinite(et)
        if m.sum()<min_pairs: continue

        inds = np.digitize(R[m], r_edges)-1
        for b in range(len(r_cent)):
            pick = inds==b
            if not np.any(pick): continue
            ww = w[m][pick]; yy = et[m][pick]
            out.append([b, np.sum(ww*yy)/np.sum(ww), np.sum(ww), zl])

    if not out:
        return pd.DataFrame(columns=["R_Mpc","gt","W","z_mean"])
    df = pd.DataFrame(out, columns=["bin","gt","W","z_cl"])
    r_cent = 0.5*(r_edges[1:]+r_edges[:-1])
    prof = df.groupby("bin").apply(lambda g: pd.Series({
        "gt": np.average(g["gt"], weights=g["W"]),
        "W":  g["W"].sum(),
        "z_mean": np.average(g["z_cl"], weights=g["W"])
    })).reset_index()
    prof["R_Mpc"] = r_cent[prof["bin"].to_numpy()]
    return prof[["R_Mpc","gt","W","z_mean"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="working/CLENS")
    ap.add_argument("--out",  default="outputs/CLENS")
    ap.add_argument("--r_bins",   default="0.1,0.2,0.3,0.5,0.8,1.2,1.8,2.7,4.0")
    ap.add_argument("--z_focus",  default="0.22,0.32")
    ap.add_argument("--z_ctrl",   default="0.15,0.20;0.35,0.45")
    ap.add_argument("--min_pairs", type=int, default=10)
    ap.add_argument("--box_expand", type=float, default=1.5)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cl  = pd.read_parquet(f"{args.work}/clusters.parquet")
    src = pd.read_parquet(f"{args.work}/sources.parquet")

    r_edges = np.array([float(x) for x in args.r_bins.split(",")])

    zf0,zf1 = [float(x) for x in args.z_focus.split(",")]
    clF = cl[(cl["z_cl"]>=zf0) & (cl["z_cl"]<=zf1)].copy()

    ctrl_ranges = [[float(a) for a in z.split(",")] for z in args.z_ctrl.split(";")]
    clC = pd.concat([ cl[(cl["z_cl"]>=a)&(cl["z_cl"]<=b)] for a,b in ctrl_ranges ], ignore_index=True)

    profF = stack_subset(clF, src, r_edges, min_pairs=args.min_pairs, box_expand=args.box_expand)
    profC = stack_subset(clC, src, r_edges, min_pairs=args.min_pairs, box_expand=args.box_expand)

    profF.to_csv(f"{args.out}/profile_focus.csv", index=False)
    profC.to_csv(f"{args.out}/profile_control.csv", index=False)
    print("[stack] wrote profiles:",
          f"{args.out}/profile_focus.csv ({len(profF)}) and",
          f"{args.out}/profile_control.csv ({len(profC)})")

    # relative excess in 0.5–1.8 Mpc
    def band_mean(p):
        if p.empty: return np.nan
        band = p[(p["R_Mpc"]>=0.5)&(p["R_Mpc"]<=1.8)]
        return np.average(band["gt"], weights=band["W"]) if len(band) else np.nan

    mF, mC = band_mean(profF), band_mean(profC)
    rel = (mF-mC)/mC if np.isfinite(mF) and np.isfinite(mC) and mC!=0 else np.nan
    print(f"[effect] focus vs control (0.5–1.8 Mpc): rel = {rel:.4f}")

if __name__ == "__main__":
    main()
