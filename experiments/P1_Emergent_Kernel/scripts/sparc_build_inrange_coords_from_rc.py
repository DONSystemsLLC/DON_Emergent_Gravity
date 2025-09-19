#!/usr/bin/env python3
import argparse, re, math
import pandas as pd
import numpy as np
from pathlib import Path

def norm_name(s: str) -> str:
    if not isinstance(s,str): return ""
    s0 = s.upper().strip()
    s1 = re.sub(r"[_\-]+"," ",s0)
    s1 = re.sub(r"\s+"," ",s1).strip()
    m = re.match(r"^(NGC)\s*0*(\d+)$", s1)
    if m: return f"{m.group(1)}{int(m.group(2)):04d}"
    m = re.match(r"^(IC)\s*0*(\d+)$", s1)
    if m: return f"{m.group(1)}{int(m.group(2)):04d}"
    m = re.match(r"^(UGC)\s*0*(\d+)$", s1)
    if m: return f"{m.group(1)}{int(m.group(2)):05d}"
    m = re.match(r"^(DDO)\s*0*(\d+)$", s1)
    if m: return f"{m.group(1)}{int(m.group(2)):03d}"
    m = re.match(r"^(PGC)\s*0*(\d+)$", s1)
    if m: return f"{m.group(1)}{int(m.group(2))}"
    if s1.startswith("ESO"):  return re.sub(r"[^\w]","",s1)
    if s1.startswith("F"):    return re.sub(r"[^\w]","",s1)
    if s1.startswith("UGCA"): return s1.replace(" ","")
    return s1.replace(" ","")

def find_rc_file(rc_dir: Path, name: str) -> Path | None:
    # try a few patterns (csv or rotmod.dat)
    patterns = []
    base = name
    patterns += [f"*{base}*.csv", f"*{base}*_rotmod.dat", f"*{base}*.dat"]
    for pat in patterns:
        ms = list(rc_dir.glob(pat))
        if ms:
            return ms[0]
    # also try with spaces removed
    base2 = base.replace(" ","")
    for pat in (f"*{base2}*.csv", f"*{base2}*_rotmod.dat", f"*{base2}*.dat"):
        ms = list(rc_dir.glob(pat))
        if ms:
            return ms[0]
    return None

def load_rc_generic(p: Path) -> pd.DataFrame:
    if p.suffix.lower()==".csv":
        rc = pd.read_csv(p)
    else:
        rc = pd.read_csv(p, sep=r"\s+", engine="python", comment="#", header=None)
        # try to keep the first six columns if many
        if rc.shape[1] >= 6:
            rc = rc.iloc[:, :6]
            rc.columns = ["r_kpc","v_obs","v_err","v_gas","v_disk","v_bulge"]
    # map alt headers
    alt = {'R':'r_kpc','r':'r_kpc','RAD_KPC':'r_kpc',
           'Vobs':'v_obs','VROT':'v_obs','v':'v_obs',
           'e_Vobs':'v_err','VERR':'v_err',
           'Vgas':'v_gas','Vdisk':'v_disk','Vbul':'v_bulge','Vbulge':'v_bulge'}
    for a,b in alt.items():
        if a in rc.columns and b not in rc.columns:
            rc.rename(columns={a:b}, inplace=True)
    keep=['r_kpc','v_obs','v_err','v_gas','v_disk','v_bulge']
    for c in keep:
        if c not in rc.columns: rc[c]=np.nan
    rc = rc[keep].dropna(subset=['r_kpc','v_obs']).copy()
    rc = rc.sort_values('r_kpc')
    return rc

def v_incl_corr(v_obs, inc_deg):
    s = math.sin(math.radians(inc_deg))
    return v_obs/s if s>1e-6 else np.nan

def interp_vc(rc: pd.DataFrame, inc_deg: float, r_target: float) -> float:
    # inclination-correct once
    vc = v_incl_corr(rc['v_obs'].to_numpy(), inc_deg)
    r  = rc['r_kpc'].to_numpy()
    if len(r)<2: return np.nan
    # limit to observed range
    if r_target < r.min() or r_target > r.max(): return np.nan
    return float(np.interp(r_target, r, vc))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inrange", default="outputs/SPARC/tables/sparc_ready_inrange.csv")
    ap.add_argument("--rc_dir",  default="data/SPARC/rotation_curves")
    ap.add_argument("--db_clean", default="data/SPARC/raw/SPARC_database_clean.csv")
    ap.add_argument("--out", default="outputs/SPARC/tables/sparc_ready_inrange_coords.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.inrange)
    df = df[(df['status']=="ok") & df['inrange_2p2Rd'] & df['inrange_3p0Rd']].copy()

    rc_dir = Path(args.rc_dir)
    rows=[]
    for _,r in df.iterrows():
        name = str(r['name'])
        f = find_rc_file(rc_dir, name)
        if not f:
            rows.append(None); continue
        rc = load_rc_generic(f)
        rd = float(r['rd_kpc'])
        vc22 = interp_vc(rc, float(r['inc_deg']), 2.2*rd)
        vc30 = interp_vc(rc, float(r['inc_deg']), 3.0*rd)
        rows.append({
            'name': name,
            'dist_mpc': r['dist_mpc'],
            'inc_deg': r['inc_deg'],
            'rd_kpc': rd,
            'vc_2p2Rd': vc22,
            'vc_3p0Rd': vc30,
        })
    tidy = pd.DataFrame([x for x in rows if x is not None]).dropna(subset=['vc_2p2Rd','vc_3p0Rd']).copy()

    # attach RA/Dec via normalized names
    tidy['norm_name'] = tidy['name'].map(norm_name)
    db = pd.read_csv(args.db_clean)[['name','ra_deg','dec_deg']].copy()
    db['norm_name'] = db['name'].map(norm_name)
    m = tidy.merge(db[['norm_name','ra_deg','dec_deg']], on='norm_name', how='left').dropna(subset=['ra_deg','dec_deg']).copy()
    m.drop(columns=['norm_name'], inplace=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)
    print(f"[build] wrote {args.out}  rows:", len(m))
    print(m.head())
if __name__ == "__main__":
    main()
