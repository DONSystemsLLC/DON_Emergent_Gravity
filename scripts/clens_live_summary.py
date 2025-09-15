#!/usr/bin/env python3
import glob, os, numpy as np, pandas as pd
from pathlib import Path

LENS = Path("outputs/CLENS_patch/profile_focus.csv")
RAND_GLOB = "outputs/CLENS_random_strat3_*/profile_focus.csv"
PROOF = Path("proofs/CLENS_KiDS_W1_Stacking"); PROOF.mkdir(parents=True, exist_ok=True)

def load_profile(path):
    df = pd.read_csv(path)
    # Normalize expected columns
    rcol = "R_Mpc" if "R_Mpc" in df.columns else ("R" if "R" in df.columns else None)
    if rcol is None: raise RuntimeError(f"No R column in {path}")
    gt = "gt" if "gt" in df.columns else ("gamma_t" if "gamma_t" in df.columns else None)
    gx = "gx" if "gx" in df.columns else None
    N  = "N_src" if "N_src" in df.columns else ("npairs" if "npairs" in df.columns else None)
    df = df.rename(columns={rcol:"R", gt:"gt"})
    if gx: df = df.rename(columns={gx:"gx"})
    if N:  df = df.rename(columns={N:"N"})
    df["_Rkey"] = np.round(df["R"].values, 6)
    return df

# Check if lens profile exists
if not LENS.exists():
    print(f"Lens profile not yet available: {LENS}")
    exit(0)

lens = load_profile(LENS)
rand_paths = sorted(glob.glob(RAND_GLOB))
done = [load_profile(p) for p in rand_paths]
if not done:
    print("No random profiles found yet.")
    exit(0)

# Align radial bins across all available outputs
# Use outer merge to get all unique R values, then find closest matches
all_R = pd.concat([lens[["_Rkey"]]] + [r[["_Rkey"]] for r in done]).drop_duplicates().sort_values("_Rkey")
Rk = all_R["_Rkey"].values

# For lens and each random, find closest R values
def get_closest_rows(df, target_Rs, tol=0.01):
    result = []
    for tr in target_Rs:
        mask = np.abs(df["_Rkey"] - tr) < tol
        if mask.any():
            result.append(df[mask].iloc[0])
    return pd.DataFrame(result) if result else pd.DataFrame()

L = get_closest_rows(lens, Rk)
R = [get_closest_rows(r, Rk) for r in done]

# Filter to bins present in lens and at least half the randoms
if len(L) == 0 or len(R) == 0:
    print("No matching radial bins found between lens and randoms")
    exit(0)

# Use only bins present in lens
Rk = L["_Rkey"].values
R = [get_closest_rows(r, Rk) for r in done]
R = [r for r in R if len(r) > 0]  # Remove empty dataframes

# Ensure all random profiles have same length as lens
R_valid = []
for ri in R:
    if len(ri) == len(L):
        R_valid.append(ri["gt"].values)

if len(R_valid) == 0:
    print("No random profiles match lens bin structure")
    exit(0)

rnd_gt = np.stack(R_valid, axis=0)
mu  = np.nanmean(rnd_gt, axis=0)
sig = np.nanstd (rnd_gt, axis=0, ddof=1) + 1e-30
delta = L["gt"].values - mu

# weights (counts if present; else uniform)
if "N" in L.columns:
    w = L["N"].values.astype(float); w = w / (np.nanmedian(w) + 1e-30)
else:
    w = np.ones_like(delta)
z_bins = delta / sig
z_overall = np.nansum(w * z_bins) / (np.nansum(w) + 1e-30)

gx_mean = np.nan if "gx" not in L.columns else float(np.nanmean(L["gx"].values))

# Write a compact running summary (idempotent)
summary = PROOF/"RUNNING_SUMMARY.md"
with open(summary, "w") as f:
    f.write("# KiDS W1 — Running Summary\n")
    f.write(f"- Random stacks available: **{len(done)}**\n")
    f.write(f"- Overall Δ z-score (unboosted): **{z_overall:.3f}**\n")
    f.write(f"- Cross-shear mean (lens): **{gx_mean:.3e}**\n")
    f.write("- Radial bins used: {}\n".format(len(Rk)))
print(f"Randoms used: {len(done)}  |  z = {z_overall:.3f}  |  gx_mean = {gx_mean:.3e}")
print("Wrote:", summary)