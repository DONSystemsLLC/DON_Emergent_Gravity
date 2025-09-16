#!/usr/bin/env python3
"""
Finalize KiDS W1 × Cluster stacking:
- Computes Δ in 0.5–1.8 Mpc (W-weighted if W exists; applies K_1p only if finite)
- Aggregates random ensembles, reports μ ± σ and z
- Applies boost correction using mean random N_src per bin
- Writes proofs/CLENS_KiDS_W1_Stacking/RESULTS.md
- Outputs structured band_means.csv for P2 slip analysis
"""
import argparse, glob, os, pathlib
import numpy as np, pandas as pd

def load_prof(p):
    d = pd.read_csv(p)
    R = 'R_Mpc' if 'R_Mpc' in d.columns else 'R'
    g = 'gt' if 'gt' in d.columns else 'g_t'
    if 'K_1p' in d.columns and not d['K_1p'].isna().all():
        d[g] = d[g] / (1.0 + d['K_1p'])
    d = d.reset_index(drop=True)
    d['bin'] = d.index
    return d, R, g

def wbandmean(d, R, g, lo, hi):
    m = (d[R] >= lo) & (d[R] <= hi)
    y = d.loc[m, g].to_numpy()
    if y.size == 0: return float('nan')
    if 'W' in d.columns:
        w = d.loc[m, 'W'].to_numpy()
        m2 = np.isfinite(y) & np.isfinite(w) & (w > 0)
        if m2.any(): return float(np.average(y[m2], weights=w[m2]))
    return float(np.nanmean(y))

def compute_band_means(dF, dC, rF, rC, RF, gF, rlo, rhi):
    """Compute band means for all data sets for P2 analysis"""
    results = []
    
    # Focus (lens sample)
    focus_mean = wbandmean(dF, RF, gF, rlo, rhi)
    focus_err = np.nan  # Single measurement, no error
    focus_nsrc = dF['N_src'].sum() if 'N_src' in dF.columns else np.nan
    results.append({
        'set': 'Focus',
        'gt_mean_0p5_1p8': focus_mean,
        'gt_err': focus_err,
        'N_src_mean': focus_nsrc
    })
    
    # Control (field sample)  
    control_mean = wbandmean(dC, RF, gF, rlo, rhi)
    control_err = np.nan  # Single measurement, no error
    control_nsrc = dC['N_src'].sum() if 'N_src' in dC.columns else np.nan
    results.append({
        'set': 'Control',
        'gt_mean_0p5_1p8': control_mean,
        'gt_err': control_err,
        'N_src_mean': control_nsrc
    })
    
    # Random ensemble (focus)
    if rF:
        rand_focus_vals = [wbandmean(df, RF, gF, rlo, rhi) for df in rF]
        rand_focus_vals = [x for x in rand_focus_vals if np.isfinite(x)]
        rand_focus_mean = np.mean(rand_focus_vals) if rand_focus_vals else np.nan
        rand_focus_err = np.std(rand_focus_vals, ddof=1) if len(rand_focus_vals) > 1 else np.nan
        rand_focus_nsrc = np.mean([df['N_src'].sum() for df in rF if 'N_src' in df.columns]) if rF else np.nan
        results.append({
            'set': 'Random_Focus',
            'gt_mean_0p5_1p8': rand_focus_mean,
            'gt_err': rand_focus_err,
            'N_src_mean': rand_focus_nsrc
        })
    
    # Random ensemble (control)
    if rC:
        rand_control_vals = [wbandmean(dc, RF, gF, rlo, rhi) for dc in rC]
        rand_control_vals = [x for x in rand_control_vals if np.isfinite(x)]
        rand_control_mean = np.mean(rand_control_vals) if rand_control_vals else np.nan
        rand_control_err = np.std(rand_control_vals, ddof=1) if len(rand_control_vals) > 1 else np.nan
        rand_control_nsrc = np.mean([dc['N_src'].sum() for dc in rC if 'N_src' in dc.columns]) if rC else np.nan
        results.append({
            'set': 'Random_Control',
            'gt_mean_0p5_1p8': rand_control_mean,
            'gt_err': rand_control_err,
            'N_src_mean': rand_control_nsrc
        })
    
    # Combined random delta (focus - control)
    if rF and rC and len(rF) == len(rC):
        rand_delta_vals = []
        for df, dc in zip(rF, rC):
            delta = wbandmean(df, RF, gF, rlo, rhi) - wbandmean(dc, RF, gF, rlo, rhi)
            if np.isfinite(delta):
                rand_delta_vals.append(delta)
        
        if rand_delta_vals:
            results.append({
                'set': 'Random_Delta',
                'gt_mean_0p5_1p8': np.mean(rand_delta_vals),
                'gt_err': np.std(rand_delta_vals, ddof=1) if len(rand_delta_vals) > 1 else np.nan,
                'N_src_mean': len(rand_delta_vals)  # Use count as proxy
            })
    
    return pd.DataFrame(results)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--focus',   default='outputs/CLENS_patch/profile_focus.csv')
    ap.add_argument('--control', default='outputs/CLENS_patch/profile_control.csv')
    ap.add_argument('--random_glob', default='outputs/CLENS_random_strat3_*/profile_*.csv')
    ap.add_argument('--r_band',  default='0.5,1.8')
    ap.add_argument('--out',     default='proofs/CLENS_KiDS_W1_Stacking/RESULTS.md')
    ap.add_argument('--out_csv', default='outputs/CLENS_release/band_means.csv')
    args = ap.parse_args()
    rlo, rhi = map(float, args.r_band.split(','))

    dF, RF, gF = load_prof(args.focus)
    dC, RC, gC = load_prof(args.control)
    d_real = wbandmean(dF, RF, gF, rlo, rhi) - wbandmean(dC, RC, gC, rlo, rhi)

    # Pair focus/control profiles per random directory
    dirs = sorted({os.path.dirname(p) for p in glob.glob(args.random_glob)})
    rF, rC = [], []
    for d in dirs:
        pf, pc = os.path.join(d, 'profile_focus.csv'), os.path.join(d, 'profile_control.csv')
        if os.path.exists(pf) and os.path.exists(pc):
            df, RR, gg = load_prof(pf)
            dc, _, _   = load_prof(pc)
            rF.append(df); rC.append(dc)

    N = len(rF)
    d_rand = []
    for df, dc in zip(rF, rC):
        d_rand.append(wbandmean(df, RF, gF, rlo, rhi) - wbandmean(dc, RF, gF, rlo, rhi))
    mu = float(np.nanmean(d_rand))    if N     else float('nan')
    sd = float(np.nanstd(d_rand, ddof=1)) if N>1 else float('nan')
    z  = (d_real - mu)/sd if (isinstance(sd, float) and sd > 0) else float('nan')

    # Boost using mean random N_src per bin
    if N>0 and 'N_src' in dF.columns:
        concat = pd.concat(rF, ignore_index=True)
        rmean  = concat.groupby('bin', as_index=False).mean(numeric_only=True)
        denom  = np.where(rmean['N_src'].to_numpy() > 0, rmean['N_src'].to_numpy(), np.nan)
        B      = dF['N_src'].to_numpy() / denom
        dF_boost      = dF.copy()
        dF_boost[gF]  = dF_boost[gF].to_numpy() * B
        d_real_boost  = wbandmean(dF_boost, RF, gF, rlo, rhi) - wbandmean(dC, RC, gC, rlo, rhi)
    else:
        d_real_boost = float('nan')

    # Generate structured band means for P2 analysis
    band_means_df = compute_band_means(dF, dC, rF, rC, RF, gF, rlo, rhi)
    
    # Write band means CSV
    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    band_means_df.to_csv(args.out_csv, index=False)
    print(f"[wrote] {args.out_csv}")

    # Write RESULTS.md
    out = args.out
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        f.write("# KiDS W1 × Cluster Stacking — Σcrit + Randoms\n\n")
        f.write(f"- Radial band: **{rlo}–{rhi} Mpc**\n")
        f.write(f"- Focus Δ (Σcrit, pre-boost): **{d_real:+.6f}**\n")
        f.write(f"- Random Δ μ±σ (N={N}): **{mu:+.6f} ± {sd:.6f}**\n")
        f.write(f"- z-score: **{z:+.2f}** (target |z|≲1)\n")
        f.write(f"- Boost-corrected Δ: **{d_real_boost:+.6f}**\n\n")
        f.write("## Band Means Summary\n\n")
        f.write(band_means_df.to_string(index=False, float_format='%.6f'))
        f.write(f"\n\nDetailed results: `{args.out_csv}`\n")
    print(f"[wrote] {out}")
    print(f"Δ={d_real:+.6f}  μ±σ={mu:+.6f}±{sd:.6f}  z={z:+.2f}  Δ_boost={d_real_boost:+.6f}  N={N}")

if __name__ == '__main__':
    main()