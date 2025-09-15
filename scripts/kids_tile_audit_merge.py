#!/usr/bin/env python3
import argparse, os, re, sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

def read_tile(path: Path):
    # Try TSV first, then whitespace
    try:
        df = pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)
        if df.shape[1] <= 1:
            raise ValueError("single column -> not TSV")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#", dtype=str, low_memory=False)
    return df

def standardize_columns(df: pd.DataFrame):
    # Lower-case index for robust mapping
    cmap = {c.lower(): c for c in df.columns}
    # Accept common variants
    ra  = cmap.get('raj2000') or cmap.get('ra') or cmap.get('alpha_j2000')
    dec = cmap.get('dej2000') or cmap.get('dec') or cmap.get('delta_j2000')
    e1  = cmap.get('e1') or cmap.get('g1') or cmap.get('shear1')
    e2  = cmap.get('e2') or cmap.get('g2') or cmap.get('shear2')
    w   = cmap.get('weight') or cmap.get('w') or cmap.get('weight_shear')
    zs  = cmap.get('zbest') or cmap.get('z_b') or cmap.get('zb') or cmap.get('photoz') or cmap.get('zphot')
    fit = cmap.get('fitclass') or cmap.get('fit_class')

    need = dict(ra=ra, dec=dec, e1=e1, e2=e2, w=w, z_src=zs, fitclass=fit)
    missing = [k for k,v in need.items() if v is None]
    if missing:
        return None, {"raw_rows": len(df), "missing": ",".join(missing)}
    out = df.rename(columns={v:k for k,v in need.items() if v is not None})[list(need.keys())]
    # Coerce numerics
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out, {"raw_rows": len(df)}

def audit_tile(path: Path, require_fitclass=True):
    if path.stat().st_size < 32:
        return None, {"raw_rows": 0, "note": "empty"}
    # Skip HTML/error pages
    head = path.read_text(errors="ignore")[:256].lower()
    if "<html" in head or "<!doctype" in head:
        return None, {"raw_rows": 0, "note": "html"}
    try:
        df = read_tile(path)
    except Exception as e:
        return None, {"raw_rows": 0, "note": f"read_err:{e}"}

    std, meta = standardize_columns(df)
    if std is None:
        meta["kept_final"] = 0
        return None, meta

    meta["after_numeric"] = std.dropna().shape[0]
    # Weight > 0
    std = std[std["w"] > 0]
    meta["after_wgt"] = len(std)
    # FitClass==0 (galaxies) if required
    if require_fitclass and "fitclass" in std:
        std = std[std["fitclass"] == 0]
        meta["after_fit0"] = len(std)
    else:
        meta["after_fit0"] = len(std)
    # Drop NaNs in essentials
    std = std.dropna(subset=["ra","dec","e1","e2","w","z_src"])
    meta["kept_final"] = len(std)
    return std, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", default="data/CLENS/kids_tiles")
    ap.add_argument("--out_dir",   default="data/CLENS")
    ap.add_argument("--no_fitclass", action="store_true", help="do not enforce fitclass==0")
    args = ap.parse_args()

    tiles = sorted(Path(args.tiles_dir).glob("*.tsv"))
    if not tiles:
        print(f"[error] no tiles in {args.tiles_dir}", file=sys.stderr); sys.exit(1)

    reports = []
    parts = []
    for t in tiles:
        std, meta = audit_tile(t, require_fitclass=not args.no_fitclass)
        meta["tile"] = t.name
        reports.append(meta)
        if std is not None and len(std):
            parts.append(std)

    rep = pd.DataFrame(reports).fillna(0)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rep.to_csv(Path(args.out_dir)/"tile_report.csv", index=False)
    print("[audit] wrote tile_report.csv  (preview):")
    print(rep.head().to_string(index=False))

    if not parts:
        print("[error] no usable rows after QC; try --no_fitclass or widen tiles.", file=sys.stderr)
        sys.exit(2)

    src = pd.concat(parts, ignore_index=True)
    # Dedup exact duplicates
    src.drop_duplicates(subset=["ra","dec","e1","e2","w","z_src"], inplace=True)
    # Save parquet + csv
    pq.write_table(pa.Table.from_pandas(src[['ra','dec','e1','e2','w','z_src']]),
                   str(Path(args.out_dir)/"sources.parquet"), compression="zstd")
    src[['ra','dec','e1','e2','w','z_src']].to_csv(Path(args.out_dir)/"sources.csv", index=False)
    print(f"[merge] sources rows: {len(src)}  RA[{src.ra.min():.3f},{src.ra.max():.3f}]  Dec[{src.dec.min():.3f},{src.dec.max():.3f}]")

    # Subset clusters
    clp = Path(args.out_dir)/"clusters.csv"
    if not clp.exists():
        print(f"[warn] {clp} missing; create it first (MCXC).", file=sys.stderr); sys.exit(0)
    cl = pd.read_csv(clp)
    for c in ['ra','dec','z_cl']:
        cl[c] = pd.to_numeric(cl[c], errors="coerce")
    cl = cl.dropna(subset=['ra','dec','z_cl'])

    ra_min, ra_max = src['ra'].min(), src['ra'].max()
    dec_min,dec_max= src['dec'].min(), src['dec'].max()
    patch = cl[(cl['ra']>=ra_min)&(cl['ra']<=ra_max)&(cl['dec']>=dec_min)&(cl['dec']<=dec_max)].copy()
    patch.to_csv(Path(args.out_dir)/"clusters_patch.csv", index=False)
    patch.to_parquet(Path(args.out_dir)/"clusters.parquet", index=False)
    print(f"[clusters] in footprint: {len(patch)}")
    if len(patch)==0:
        print("[hint] If 0: either tiles do not cover MCXC area yet (extend RA to 230) or filters too strict (--no_fitclass).")

if __name__ == "__main__":
    main()
