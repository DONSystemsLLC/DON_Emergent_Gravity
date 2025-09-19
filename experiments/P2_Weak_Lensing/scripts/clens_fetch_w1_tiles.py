#!/usr/bin/env python3
import os, io, sys, math
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import pyarrow as pa, pyarrow.parquet as pq

# Wide W1 box (this range worked for you before)
RA_MIN, RA_MAX = 208.0, 230.0
DEC_MIN, DEC_MAX = -6.0, 6.0
RA_STEP, DEC_STEP = 2.0, 2.0   # tile sizes (deg)

tiles_dir = Path("data/CLENS/kids_tiles"); tiles_dir.mkdir(parents=True, exist_ok=True)
out_dir   = Path("data/CLENS"); out_dir.mkdir(parents=True, exist_ok=True)

def ra_dec_tiles(ra_min, ra_max, dec_min, dec_max, dra, ddec):
    ra_edges  = np.arange(ra_min, ra_max, dra).tolist() + [ra_max]
    dec_edges = np.arange(dec_min, dec_max, ddec).tolist() + [dec_max]
    for i in range(len(ra_edges)-1):
        for j in range(len(dec_edges)-1):
            yield (float(ra_edges[i]), float(ra_edges[i+1]),
                   float(dec_edges[j]), float(dec_edges[j+1]))

def fetch_tile(ra1,ra2,dec1,dec2):
    q = (
      "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
      "?-source=II/384/kids_shear"
      "&-out=RAJ2000,DEJ2000,e1,e2,Weight,zbest,FitClass"
      "&-oc.form=dec"
      f"&RAJ2000={ra1:.3f}..{ra2:.3f}"
      f"&DEJ2000={dec1:.3f}..{dec2:.3f}"
      "&-out.max=1000000"
    )
    fn = tiles_dir / f"kids_{ra1:.3f}_{ra2:.3f}_{dec1:.3f}_{dec2:.3f}.tsv"
    if fn.exists() and fn.stat().st_size>0:
        return fn, "keep"
    try:
        req = Request(q, headers={"User-Agent":"Mozilla/5.0"})
        with urlopen(req, timeout=60) as r:
            data = r.read()
        # Skip obvious HTML/empty
        head = data[:512].decode("utf-8","ignore").lower()
        if "<html" in head or "doctype html" in head or len(data)<32:
            return fn, "skip"
        fn.write_bytes(data)
        return fn, "ok"
    except (HTTPError, URLError, TimeoutError) as e:
        return fn, f"err:{e}"

# --- fetch all tiles
parts = []
for ra1,ra2,dec1,dec2 in ra_dec_tiles(RA_MIN,RA_MAX,DEC_MIN,DEC_MAX,RA_STEP,DEC_STEP):
    fn,status = fetch_tile(ra1,ra2,dec1,dec2)
    print(f"[{status:4s}] {fn.name}")

# --- merge tiles (skip empties)
need = {'RAJ2000','DEJ2000','e1','e2','Weight','zbest','FitClass'}
for fn in sorted(tiles_dir.glob("*.tsv")):
    try:
        if fn.stat().st_size < 32: 
            continue
        df = pd.read_csv(fn, sep="\t", comment="#", low_memory=False)
        if df.empty or not need.issubset(df.columns):
            continue
        parts.append(df)
    except Exception:
        continue

if not parts:
    sys.exit("[error] No valid KiDS tiles merged. Try expanding the RA/Dec box or check network.")

wide = pd.concat(parts, ignore_index=True)
wide.drop_duplicates(subset=["RAJ2000","DEJ2000","e1","e2","Weight","zbest","FitClass"], inplace=True)
# Standardize & QC (KiDS convention: do NOT flip e2)
wide = wide.rename(columns={'RAJ2000':'ra','DEJ2000':'dec','Weight':'w','zbest':'z_src','FitClass':'fitclass'})
for c in ['ra','dec','e1','e2','w','z_src','fitclass']:
    wide[c] = pd.to_numeric(wide[c], errors='coerce')
wide = wide[(wide['fitclass']==0) & (wide['w']>0)].dropna(subset=['ra','dec','e1','e2','w','z_src'])
print(f"[merge] KiDS sources rows: {len(wide)}")

# Save Parquet + CSV (Parquet for speed)
pq.write_table(pa.Table.from_pandas(wide[['ra','dec','e1','e2','w','z_src']]),
               str(out_dir/"sources.parquet"), compression="zstd")
wide[['ra','dec','e1','e2','w','z_src']].to_csv(out_dir/"sources.csv", index=False)

# Subset MCXC clusters to this footprint
cl_path = out_dir/"clusters.csv"
if not cl_path.exists():
    sys.exit("[warn] data/CLENS/clusters.csv not found. Create it first (we already showed how).")
cl = pd.read_csv(cl_path)
for c in ['ra','dec','z_cl']:
    cl[c] = pd.to_numeric(cl[c], errors='coerce')
cl = cl.dropna(subset=['ra','dec','z_cl'])

ra_min, ra_max = wide['ra'].min(), wide['ra'].max()
dec_min,dec_max= wide['dec'].min(), wide['dec'].max()
cl_patch = cl[(cl['ra']>=ra_min)&(cl['ra']<=ra_max)&(cl['dec']>=dec_min)&(cl['dec']<=dec_max)].copy()
print(f"[clusters] in footprint: {len(cl_patch)}")

cl_patch.to_parquet(out_dir/"clusters.parquet", index=False)
cl_patch.to_csv(out_dir/"clusters_patch.csv", index=False)
print("[done] wrote data/CLENS/sources.parquet, sources.csv, clusters.parquet, clusters_patch.csv")
