#!/usr/bin/env python3
import argparse
import pandas as pd
from astropy.io import fits
from pathlib import Path

def map_cols(df):
    # try common variants
    mapping = {
        "ra":  ["RA", "RAdeg", "ALPHA_J2000", "RA_X"],
        "dec": ["DEC","DEdeg","DELTA_J2000","DEC_X"],
        "z_cl":["REDSHIFT","Z","z","z_cl"]
    }
    out = {}
    for std, cands in mapping.items():
        for c in cands:
            if c in df.columns:
                out[std] = df[c]; break
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcxc", required=True, help="MCXC FITS table")
    ap.add_argument("--out",  default="data/CLENS/clusters.csv")
    args = ap.parse_args()

    with fits.open(args.mcxc) as hdul:
        # find first binary table HDU
        hdu = next((h for h in hdul if h.header.get('XTENSION')=='BINTABLE'), None)
        if hdu is None: raise SystemExit("No table HDU found in MCXC FITS")
        df = pd.DataFrame(hdu.data.byteswap().newbyteorder())
    df = map_cols(df)
    need = {"ra","dec","z_cl"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[MCXC] missing {need - set(df.columns)}; have {list(df.columns)}")
    df = df.dropna(subset=["ra","dec","z_cl"]).copy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[mcxc] wrote {args.out} rows={len(df)}")

if __name__ == "__main__":
    main()
