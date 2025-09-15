#!/usr/bin/env python3
import glob, os, pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "proofs/CLENS_KiDS_W1_Stacking"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

def load_prof(p):
    d = pd.read_csv(p)
    R = 'R_Mpc' if 'R_Mpc' in d.columns else 'R'
    g = 'gt' if 'gt' in d.columns else 'g_t'
    if 'K_1p' in d.columns and not d['K_1p'].isna().all():
        d[g] = d[g] / (1.0 + d['K_1p'])
    d = d.reset_index(drop=True)
    d['bin'] = d.index
    return d, R, g

def wband(d,R,g,lo,hi):
    m=(d[R]>=lo)&(d[R]<=hi)
    y=d.loc[m,g].to_numpy()
    if y.size==0: return np.nan
    if 'W' in d.columns:
        w=d.loc[m,'W'].to_numpy()
        m2=np.isfinite(y)&np.isfinite(w)&(w>0)
        return np.average(y[m2],weights=w[m2]) if m2.any() else np.nan
    return np.nanmean(y)

# Load focus/control
dF, RF, gF = load_prof('outputs/CLENS_patch/profile_focus.csv')
dC, RC, gC = load_prof('outputs/CLENS_patch/profile_control.csv')

# Collect randoms
dirs = sorted({os.path.dirname(p) for p in glob.glob('outputs/CLENS_random_strat3_*/profile_*.csv')})
rF, rC = [], []
for d in dirs:
    pf, pc = os.path.join(d,'profile_focus.csv'), os.path.join(d,'profile_control.csv')
    if os.path.exists(pf) and os.path.exists(pc):
        df, RR, gg = load_prof(pf)
        dc, _, _   = load_prof(pc)
        rF.append(df); rC.append(dc)

# 1) Random-Δ histogram in 0.5–1.8 Mpc
rlo,rhi = 0.5, 1.8
d_rand=[]
for df,dc in zip(rF,rC):
    d_rand.append(wband(df,RF,gF,rlo,rhi)-wband(dc,RC,gC,rlo,rhi))
d_rand = np.asarray([x for x in d_rand if np.isfinite(x)])
plt.figure()
plt.hist(d_rand, bins=20, edgecolor='k')
plt.axvline(np.nanmean(d_rand), lw=2)
plt.title(f"Random Δ g_t (focus-control), {rlo}-{rhi} Mpc (N={len(d_rand)})")
plt.xlabel("Δ g_t"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(f"{OUTDIR}/random_delta_hist.png", dpi=150)

# 2) Boost B(R) = N_lens / <N_rand>
if len(rF)>0 and 'N_src' in dF.columns:
    concat = pd.concat(rF, ignore_index=True)
    rmean  = concat.groupby('bin', as_index=False).mean(numeric_only=True)
    B = dF['N_src'].to_numpy() / np.where(rmean['N_src'].to_numpy()>0, rmean['N_src'].to_numpy(), np.nan)
    plt.figure()
    plt.plot(dF[RF].to_numpy(), B, marker='o')
    plt.xlabel(RF); plt.ylabel("B(R)"); plt.title("Boost factor B(R)")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/boost_B_of_R.png", dpi=150)
print("[wrote] random_delta_hist.png and boost_B_of_R.png")