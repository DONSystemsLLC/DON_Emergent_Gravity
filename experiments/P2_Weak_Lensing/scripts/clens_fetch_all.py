#!/usr/bin/env python3
import pandas as pd, numpy as np, pathlib
from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1

# KiDS DR4 shear catalog code
CAT = "II/384/kids_shear"
FIELDS = {
    "W1": (211.5, 234.5, -5.5, 5.5),   # RAmin, RAmax, DECmin, DECmax
    "W2": (246.0, 266.0, -6.0, -3.0),
    "W3": (332.0, 357.0, -1.0, 3.0),
    "W4": (35.0,  59.0, -10.0, -4.0),
}

outdir = pathlib.Path("data/CLENS")
outdir.mkdir(parents=True, exist_ok=True)

frames=[]
for field,(ra1,ra2,dec1,dec2) in FIELDS.items():
    print(f"[fetch] KiDS {field} RA={ra1}..{ra2} DEC={dec1}..{dec2}")
    v = Vizier(columns=["RAJ2000","DEJ2000","e1","e2","Weight","zbest","FitClass"])
    try:
        res = v.query_constraints(
            catalog=CAT,
            RAJ2000=f">{ra1}&<{ra2}",
            DEJ2000=f">{dec1}&<{dec2}",
            Weight=">0"
        )
    except Exception as e:
        print("[error] query failed:", e)
        continue
    if len(res)==0: 
        print("[warn] no data for",field)
        continue
    df = res[0].to_pandas()
    df.rename(columns={
        "RAJ2000":"ra",
        "DEJ2000":"dec",
        "Weight":"w",
        "zbest":"z_src",
        "FitClass":"fitclass"
    }, inplace=True)
    for c in ["ra","dec","e1","e2","w","z_src","fitclass"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["fitclass"]==0) & (df["w"]>0)].dropna(subset=["ra","dec","e1","e2","w","z_src"])
    print(f"   rows: {len(df)}")
    df.to_csv(outdir/f"kids_{field}.csv",index=False)
    frames.append(df)

if not frames:
    raise SystemExit("[error] no KiDS sources downloaded.")

all_sources = pd.concat(frames,ignore_index=True)
all_sources.to_csv(outdir/"sources.csv",index=False)
all_sources.to_parquet(outdir/"sources.parquet",index=False)
print("[sources] merged all KiDS fields ->",len(all_sources),"rows")

# Now subset clusters to the combined KiDS footprint
cl = pd.read_csv("data/CLENS/clusters.csv")
ra_min,ra_max = all_sources["ra"].min(), all_sources["ra"].max()
dec_min,dec_max=all_sources["dec"].min(), all_sources["dec"].max()
cl_patch=cl[(cl["ra"]>=ra_min)&(cl["ra"]<=ra_max)&(cl["dec"]>=dec_min)&(cl["dec"]<=dec_max)].copy()
cl_patch.to_csv(outdir/"clusters_patch.csv",index=False)
cl_patch.to_parquet(outdir/"clusters.parquet",index=False)
print("[clusters] patch rows:",len(cl_patch))
