#!/usr/bin/env python3
import argparse, glob, os
import numpy as np, pandas as pd
from pathlib import Path

def pick(colnames, *cands):
    for c in cands:
        if c in colnames: return c
    return None

def load_profile(path):
    df = pd.read_csv(path)
    return df

def align_bins(dfs, key="R_Mpc", tol=1e-6):
    # Round R to avoid float jitter
    for i,df in enumerate(dfs):
        if key not in df.columns:
            # fallback names
            alt = [c for c in ["R", "R_mpc", "R_Mpc"] if c in df.columns]
            if not alt: raise ValueError(f"No radial column in {i}: {df.columns.tolist()}")
            if key not in df.columns:
                key = alt[0]
        df["_Rkey"] = np.round(df[key].values, 6)
    base = dfs[0][["_Rkey"]].copy()
    for i in range(1,len(dfs)):
        base = base.merge(dfs[i][["_Rkey"]].drop_duplicates(), on="_Rkey", how="inner")
    return base["_Rkey"].values

def summarize(args):
    lens = load_profile(args.lens)
    rnd_paths = sorted(glob.glob(os.path.join(args.random_dir, args.pattern, "profile_focus.csv")))
    if not rnd_paths:
        raise SystemExit(f"No random profiles found under {args.random_dir}/{args.pattern}/profile_focus.csv")
    rnd = [load_profile(p) for p in rnd_paths]

    # Column detection
    Rcol = "R_Mpc" if "R_Mpc" in lens.columns else ( "R" if "R" in lens.columns else None )
    if Rcol is None: raise SystemExit(f"Could not find R column in lens: {lens.columns.tolist()}")
    gcol = "gt" if "gt" in lens.columns else ("gamma_t" if "gamma_t" in lens.columns else None)
    if gcol is None: raise SystemExit(f"Could not find tangential shear in lens: {lens.columns.tolist()}")
    gxcol = "gx" if "gx" in lens.columns else ("g×" if "g×" in lens.columns else None)
    Ncol  = pick(lens.columns, "N_src", "npairs", "pairs", "Npairs")
    Wcol  = "W" if "W" in lens.columns else None

    # Align bins by R
    Rkey = align_bins([lens] + rnd, key=Rcol)
    lsel = lens.set_index(np.round(lens[Rcol].values,6)).loc[Rkey]
    rsel = [r.set_index(np.round(r[Rcol].values,6)).loc[Rkey] for r in rnd]

    # Random μ±σ per bin (on gt)
    rnd_gt = np.stack([r[gcol].values for r in rsel], axis=0)
    mu = np.nanmean(rnd_gt, axis=0)
    sig = np.nanstd(rnd_gt, axis=0, ddof=1) + 1e-30

    # Boost factor via counts; fall back to weights if needed
    def extract_counts(df):
        if Ncol and Ncol in df.columns: return df[Ncol].values.astype(float)
        if "npairs" in df.columns: return df["npairs"].values.astype(float)
        if Wcol and Wcol in df.columns: return df[Wcol].values.astype(float)
        # last ditch: infer from finite entries
        return np.isfinite(df[gcol].values).astype(float)

    N_lens = extract_counts(lsel)
    N_rnds = np.stack([extract_counts(r) for r in rsel], axis=0)
    B = N_lens / (np.nanmean(N_rnds, axis=0) + 1e-30)

    # Apply boost
    gt_lens = lsel[gcol].values.astype(float)
    gt_corr = B * gt_lens

    # Z-score for Δ ≡ gt_lens - mu (unboosted; report both)
    delta = gt_lens - mu
    # Weight by counts (lens) to reflect S/N across bins
    weights = N_lens / (np.nanmedian(N_lens) + 1e-30)
    z_bins = delta / sig
    z_overall = np.nansum(weights * z_bins) / (np.nansum(weights) + 1e-30)

    # Cross-shear null (lens)
    gx_mean = np.nan if gxcol is None else np.nanmean(lsel[gxcol].values)

    # Write outputs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = pd.DataFrame({
        "R_Mpc": Rkey,
        "gt_lens": gt_lens,
        "gt_rand_mu": mu,
        "gt_rand_sigma": sig,
        "B": B,
        "gt_corr": gt_corr,
        "N_lens": N_lens
    })
    prof.to_csv(out/"boosted_profile.csv", index=False)
    with open(out/"Z.txt","w") as f: f.write(f"{z_overall:.3f}\n")
    with open(out/"SUMMARY.txt","w") as f:
        f.write(f"Z_overall (unboosted Δ): {z_overall:.3f}\n")
        f.write(f"Cross-shear mean (lens): {gx_mean:.3e}\n")
        f.write(f"Randoms used: {len(rnd_paths)}\n")
    print("Wrote:", out/"boosted_profile.csv", out/"Z.txt", out/"SUMMARY.txt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True, help="outputs/.../profile_focus.csv")
    ap.add_argument("--random_dir", required=True, help="outputs/")
    ap.add_argument("--pattern", default="CLENS_random_strat3_*", help="glob under random_dir")
    ap.add_argument("--out", required=True, help="proofs/... dir")
    args = ap.parse_args()
    summarize(args)