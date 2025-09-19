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
    # map common alternatives if needed
    alt = {'R':'r_kpc','r':'r_kpc','Vobs':'v_obs','VROT':'v_obs','e_Vobs':'v_err','VERR':'v_err',
           'Vgas':'v_gas','Vdisk':'v_disk','Vbul':'v_bulge','Vbulge':'v_bulge'}
    for k,v in alt.items():
        if k in rc.columns and v not in rc.columns: rc.rename(columns={k:v}, inplace=True)
    for c in want:
        if c not in rc.columns: rc[c] = None
    rc = rc[want].dropna(subset=['r_kpc','v_obs'])
    return rc

def v_incl_corr(v_obs, inc_deg):
    s = math.sin(math.radians(inc_deg))
    return v_obs/s if s>1e-6 else float('nan')

def nearest_interp(rc, x):
    s = rc.sort_values('r_kpc')
    idx = (s['r_kpc']-x).abs().argmin()
    return float(s.iloc[idx]['v_c'])

def name_candidates(name):
    # base variants
    s = name.strip()
    cands = [s, s.replace(' ','_'), s.replace(' ','-'), s.replace(' ','')]
    s_up = s.upper()
    cands += [s_up, s_up.replace(' ','_'), s_up.replace(' ','-'), s_up.replace(' ','')]
    # zero-padding for common catalogs
    m = re.match(r'^(NGC|UGC|IC)\s*0*([0-9]+)$', s_up.replace('_',' ').replace('-',' '))
    if m:
        pref, num = m.group(1), int(m.group(2))
        pad = f"{num:05d}" if pref=='UGC' else f"{num:04d}"
        cands += [f"{pref}{pad}", f"{pref}_{pad}", f"{pref}{num}", f"{pref}{num:04d}"]
    return list(dict.fromkeys(cands))  # unique, keep order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--rc_dir", required=True)
    ap.add_argument("--out_summary", default="outputs/SPARC/tables/sparc_summary.csv")
    args = ap.parse_args()

    dfm = qc_master(load_master(args.master))
    rows = []
    rc_dir = Path(args.rc_dir)

    for _,row in dfm.iterrows():
        name = str(row['name'])
        # file search: try csv first, then dat
        found = None
        for cand in name_candidates(name):
            for ext in (".csv", ".dat"):
                # rotmod naming normalization for dat converted case
                pats = [f"*{cand}*{ext}"]
                if ext == ".dat":
                    pats.append(f"*{cand}*_rotmod.dat")
                for pat in pats:
                    matches = list(rc_dir.glob(pat))
                    if matches:
                        found = matches[0]
                        break
                if found: break
            if found: break

        if not found:
            rows.append({'name':name,'status':'missing_rc'}); continue

        # ensure CSV; if dat slipped through, try to parse it directly
        rc = load_rc(found) if found.suffix.lower()==".csv" else pd.read_csv(found, sep=r"\s+", engine="python")
        if found.suffix.lower() != ".csv":
            # try to normalize dat columns
            if rc.shape[1] >= 6:
                rc = rc.iloc[:, :6]
                rc.columns = ["r_kpc","v_obs","v_err","v_gas","v_disk","v_bulge"]
        rc = rc.dropna(subset=['r_kpc','v_obs'])
        if len(rc) < 5:
            rows.append({'name':name,'status':'too_few_points'}); continue

        rc = rc.copy()
        rc['v_c'] = [v_incl_corr(v, row['inc_deg']) for v in rc['v_obs']]

        t22, t30 = 2.2*row['rd_kpc'], 3.0*row['rd_kpc']
        vc22 = nearest_interp(rc, t22)
        vc30 = nearest_interp(rc, t30)

        rows.append({
          'name': name,
          'dist_mpc': row['dist_mpc'],
          'inc_deg': row['inc_deg'],
          'rd_kpc': row['rd_kpc'],
          f'vc_at_{t22:.2f}_kpc': vc22,
          f'vc_at_{t30:.2f}_kpc': vc30,
          'status':'ok'
        })

    outdf = pd.DataFrame(rows)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(args.out_summary, index=False)
    ok = int((outdf['status']=="ok").sum())
    miss = len(outdf) - ok
    print(f"[prep] wrote {args.out_summary}  (ok={ok}, missing={miss})")

if __name__ == "__main__":
    main()
