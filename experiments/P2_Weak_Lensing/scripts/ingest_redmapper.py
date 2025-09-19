#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except Exception:
    fits = None

def read_any(path: Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".fits", ".fit", ".fz"):
        if fits is None:
            raise SystemExit("[error] astropy not installed; pip install astropy")
        with fits.open(p) as hdul:
            # pick first binary table
            hdu = None
            for h in hdul:
                if h.header.get('XTENSION') == 'BINTABLE':
                    hdu = h; break
            if hdu is None:
                raise SystemExit("[error] no BINTABLE HDU in FITS")
            df = pd.DataFrame(hdu.data.byteswap().newbyteorder())
    else:
        # CSV/TSV; try commas then whitespace
        try:
            df = pd.read_csv(p)
        except Exception:
            df = pd.read_csv(p, sep=r"\s+", engine="python")
    # normalize columns to str
    df.columns = [str(c) for c in df.columns]
    return df

def stdcols(df: pd.DataFrame) -> pd.DataFrame:
    cmap = {c.lower(): c for c in df.columns}
    # common redMaPPer variants
    ra  = cmap.get('ra') or cmap.get('alpha_j2000') or cmap.get('raj2000')
    dec = cmap.get('dec') or cmap.get('delta_j2000') or cmap.get('dej2000')
    z   = (cmap.get('z_lambda') or cmap.get('z') or cmap.get('zspec') or
           cmap.get('z_cl') or cmap.get('zspec_cluster') or cmap.get('zred') or cmap.get('z_phot'))
    need = dict(ra=ra, dec=dec, z_cl=z)
    missing = [k for k,v in need.items() if v is None]
    if missing:
        raise SystemExit(f"[error] missing columns {missing}; found: {list(df.columns)[:20]}")
    out = df.rename(columns={v:k for k,v in need.items() if v is not None})[['ra','dec','z_cl']].copy()
    for c in ['ra','dec','z_cl']:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=['ra','dec','z_cl'])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True, help="path to redMaPPer catalog (FITS/CSV/TSV)")
    ap.add_argument("--out_csv", default="data/CLENS/clusters.csv")
    ap.add_argument("--z", default="", help="optional zmin,zmax to keep (e.g., 0.1,0.35)")
    args = ap.parse_args()

    src_parquet = Path("data/CLENS/sources.parquet")
    if not src_parquet.exists():
        raise SystemExit("[error] data/CLENS/sources.parquet not found. Build KiDS sources first.")

    df = read_any(Path(args.cat))
    cl = stdcols(df)

    # optional z window
    if args.z:
        zmin,zmax = map(float, args.z.split(","))
        cl = cl[(cl['z_cl']>=zmin) & (cl['z_cl']<=zmax)]

    # intersect with sources footprint
    import pyarrow.parquet as pq
    s = pq.read_table(src_parquet, columns=['ra','dec']).to_pandas()
    ra_min, ra_max = s['ra'].min(), s['ra'].max()
    dec_min,dec_max= s['dec'].min(), s['dec'].max()

    in_foot = cl[(cl['ra']>=ra_min)&(cl['ra']<=ra_max)&(cl['dec']>=dec_min)&(cl['dec']<=dec_max)].copy()
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    in_foot.to_csv(args.out_csv, index=False)
    in_foot.to_parquet(Path(args.out_csv).with_suffix(".parquet"), index=False)

    print(f"[redMaPPer] total={len(cl)}   in_footprint={len(in_foot)}   saved -> {args.out_csv}")

if __name__ == "__main__":
    main()
