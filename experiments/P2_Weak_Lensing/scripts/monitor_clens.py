#!/usr/bin/env python3
"""
Live monitor for KiDS W1 × Cluster stacking analysis
Usage: python monitor_clens.py
"""
import numpy as np, pandas as pd, glob, os

def load_prof(p):
    d=pd.read_csv(p)
    R='R_Mpc' if 'R_Mpc' in d else 'R'
    g='gt' if 'gt' in d else 'g_t'
    # Skip K_1p correction if all NaN
    if 'K_1p' in d.columns and not d['K_1p'].isna().all():
        d[g]=d[g]/(1.0+d['K_1p'])
    return d,R,g

def wbandmean(d,R,g,lo,hi):
    m=(d[R]>=lo)&(d[R]<=hi); y=d.loc[m,g].to_numpy()
    if len(y) == 0: return float('nan')
    if 'W' in d.columns:
        w=d.loc[m,'W'].to_numpy()
        m2=np.isfinite(y)&np.isfinite(w)&(w>0)
        if m2.any():
            return float(np.average(y[m2],weights=w[m2]))
    return float(np.nanmean(y))

def main():
    rlo,rhi=0.5,1.8
    dF,RF,gF=load_prof('outputs/CLENS_patch/profile_focus.csv')
    dC,RC,gC=load_prof('outputs/CLENS_patch/profile_control.csv')
    d_real = wbandmean(dF,RF,gF,rlo,rhi)-wbandmean(dC,RC,gC,rlo,rhi)

    vals=[]
    for o in sorted(glob.glob('outputs/CLENS_random_strat3_*')):
        pf=o+'/profile_focus.csv'; pc=o+'/profile_control.csv'
        if os.path.exists(pf) and os.path.exists(pc):
            df,RR,gg=load_prof(pf); dc,_,_=load_prof(pc)
            delta = wbandmean(df,RR,gg,rlo,rhi)-wbandmean(dc,RR,gg,rlo,rhi)
            if not np.isnan(delta):
                vals.append(delta)

    N=len(vals); mu=float(np.nanmean(vals)) if N else float('nan')
    sd=float(np.nanstd(vals,ddof=1)) if N>1 else float('nan')
    z = (d_real-mu)/sd if (isinstance(sd,float) and sd>0) else float('nan')
    print(f"[Δ(real)] {d_real:+.6f}   [μ±σ(rand)] {mu:+.6f} ± {sd:.6f}   z={z:+.2f}   N={N}")

if __name__ == "__main__":
    main()