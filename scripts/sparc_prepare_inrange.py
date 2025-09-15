#!/usr/bin/env python3
import argparse, math, re
import pandas as pd
from pathlib import Path

def load_master(path):
    df = pd.read_csv(path)
    need = ['name','dist_mpc','inc_deg','rd_kpc']
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"master missing columns: {miss}")
    return df

def qc_master(df):
    df = df.copy()
    df['good_i'] = (df['inc_deg'] >= 30) & (df['inc_deg'] <= 85)
    df['good_flags'] = True
    return df[df['good_i'] & df['good_flags']]

def load_rc(p):
    rc = pd.read_csv(p)
    want = ['r_kpc','v_obs','v_err','v_gas','v_disk','v_bulge']
    alt = {'R':'r_kpc','r':'r_kpc','Vobs':'v_obs','VROT':'v_obs','e_Vobs':'v_err','VERR':'v_err',
           'Vgas':'v_gas','Vdisk':'v_disk','Vbul':'v_bulge','Vbulge':'v_bulge'}
    for k,v in alt.items():
        if k in rc.columns and v not in rc.columns: rc.rename(columns={k:v}, inplace=True)
    for c in want:
        if c not in rc.columns: rc[c] = None
    return rc[want].dropna(subset=['r_kpc','v_obs'])

def v_incl_corr(v_obs, inc_deg):
    s = math.sin(math.radians(inc_deg))
    return v_obs/s if s>1e-6 else float('nan')

def nearest_interp(rc, x):
    s = rc.sort_values('r_kpc')
    idx = (s['r_kpc']-x).abs().argmin()
    return float(s.iloc[idx]['v_c'])

def name_candidates(name):
    s = str(name).strip()
    return [s, s.replace(' ','_'), s.replace(' ','-'), s.replace(' ','')]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--rc_dir", required=True)
    ap.add_argument("--out", default="outputs/SPARC/tables/sparc_ready_inrange.csv")
    args = ap.parse_args()

    dfm = qc_master(load_master(args.master))
    rows = []
    rc_dir = Path(args.rc_dir)

    for _,row in dfm.iterrows():
        name = str(row['name'])
        # find one RC file
        found = None
        for cand in name_candidates(name):
            for pat in (f"*{cand}*.csv", f"*{cand}*.dat", f"*{cand}*_rotmod.dat"):
                ms = list(rc_dir.glob(pat))
                if ms: found = ms[0]; break
            if found: break
        if not found:
            rows.append({'name':name,'status':'missing_rc'}); continue

        rc = load_rc(found)
        if len(rc) < 5:
            rows.append({'name':name,'status':'too_few_points'}); continue

        # inclination-corrected v_c(r)
        rc = rc.copy()
        rc['v_c'] = [v_incl_corr(v, row['inc_deg']) for v in rc['v_obs']]

        rd = row['rd_kpc']
        r22 = 2.2*rd
        r30 = 3.0*rd

        rmin = float(rc['r_kpc'].min())
        rmax = float(rc['r_kpc'].max())
        in22 = (r22 >= rmin) and (r22 <= rmax)
        in30 = (r30 >= rmin) and (r30 <= rmax)

        vc22 = nearest_interp(rc, r22) if in22 else float('nan')
        vc30 = nearest_interp(rc, r30) if in30 else float('nan')

        rows.append({
          'name': name,
          'dist_mpc': row['dist_mpc'],
          'inc_deg': row['inc_deg'],
          'rd_kpc': rd,
          f'vc_at_{r22:.2f}_kpc': vc22,
          f'vc_at_{r30:.2f}_kpc': vc30,
          'inrange_2p2Rd': in22,
          'inrange_3p0Rd': in30,
          'rc_rmin_kpc': rmin,
          'rc_rmax_kpc': rmax,
          'status':'ok'
        })

    outdf = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(args.out, index=False)
    ok = int((outdf['status']=="ok").sum())
    miss = len(outdf) - ok
    print(f"[prep-inrange] wrote {args.out}  (ok={ok}, missing={miss})")
    print(outdf[['name','inrange_2p2Rd','inrange_3p0Rd']].value_counts(dropna=False).head())
if __name__ == "__main__":
    main()
