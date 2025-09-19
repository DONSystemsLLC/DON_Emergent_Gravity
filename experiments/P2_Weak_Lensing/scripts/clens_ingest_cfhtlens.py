#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from astropy.io import fits
from pathlib import Path

def read_any(path):
    p = Path(path)
    if p.suffix.lower() in (".fits",".fit",".fz"):
        with fits.open(p) as hdul:
            hdu = next((h for h in hdul if h.header.get('XTENSION')=='BINTABLE'), None)
            if hdu is None: raise SystemExit("No table HDU in FITS")
            df = pd.DataFrame(hdu.data.byteswap().newbyteorder())
    else:
        # try CSV/TSV
        try:
            df = pd.read_csv(p)
        except Exception:
            df = pd.read_csv(p, sep=r"\s+", engine="python")
    # normalize column names to simple strings (bytes → str)
    df.columns = [str(c) for c in df.columns]
    return df

def map_cols(df):
    # map likely CFHTLenS headers to our standard
    rename = {}
    for a,b in [
        ("ALPHA_J2000","ra"), ("RA","ra"), ("RAdeg","ra"),
        ("DELTA_J2000","dec"), ("DEC","dec"), ("DEdeg","dec"),

        ("e1","e1"), ("g1","e1"), ("shear1","e1"), ("e1_im","e1"),
        ("e2","e2"), ("g2","e2"), ("shear2","e2"), ("e2_im","e2"),

        ("weight","w"), ("w","w"), ("w_s","w"), ("WEIGHT","w"),

        ("Z_B","z_src"), ("Z_BEST","z_src"), ("PHOTOZ","z_src"),
        ("zphot","z_src"), ("z_mc","z_src"), ("Z","z_src")
    ]:
        if a in df.columns and b not in df.columns:
            rename[a] = b
    if rename:
        df = df.rename(columns=rename)
    keep = [c for c in ["ra","dec","e1","e2","w","z_src"] if c in df.columns]
    return df[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfht", required=True, help="CFHTLenS shear catalog (CSV or FITS)")
    ap.add_argument("--out",  default="data/CLENS/sources.csv")
    ap.add_argument("--abs_e_max", type=float, default=1.5)
    ap.add_argument("--w_min",     type=float, default=0.0)
    ap.add_argument("--zsrc_min_offset", type=float, default=0.05, help="require z_src >= z_cl + offset at stack time (we’ll filter later)")
    args = ap.parse_args()

    df = read_any(args.cfht)
    df = map_cols(df)
    need = {"ra","dec","e1","e2","w","z_src"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[CFHTLenS] missing {need - set(df.columns)}; have {list(df.columns)}")

    # basic quality
    df = df.dropna(subset=list(need)).copy()
    df = df[(np.abs(df["e1"])<=args.abs_e_max) & (np.abs(df["e2"])<=args.abs_e_max) & (df["w"]>args.w_min)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[cfhtlens] wrote {args.out} rows={len(df)}")
    print(df.head())

if __name__ == "__main__":
    main()
