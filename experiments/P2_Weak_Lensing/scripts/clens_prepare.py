#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from pathlib import Path

def map_cols(df, mapping):
    for std, cand in mapping.items():
        for c in cand:
            if c in df.columns:
                df = df.rename(columns={c: std}); break
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--outdir", default="working/CLENS")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # --- clusters ---
    cl = pd.read_csv(args.clusters)
    cl = map_cols(cl, {
        "ra": ["ra","RA","RAdeg"],
        "dec":["dec","DEC","DEdeg"],
        "z_cl":["z","z_cl","zlambda"],
        "r200_mpc":["r200_mpc","R200_Mpc","R200"]
    })
    need = {"ra","dec","z_cl"}
    if not need.issubset(cl.columns): raise SystemExit(f"clusters missing {need - set(cl.columns)}")
    cl = cl[np.isfinite(cl["ra"]) & np.isfinite(cl["dec"]) & np.isfinite(cl["z_cl"])].copy()
    cl.to_parquet(f"{args.outdir}/clusters.parquet", index=False)

    # --- sources ---
    src = pd.read_csv(args.sources)
    src = map_cols(src, {
        "ra":["ra","RA","RAdeg"], "dec":["dec","DEC","DEdeg"],
        "e1":["e1","g1","shear1"], "e2":["e2","g2","shear2"],
        "w":["weight","w","wsys","w_s"], "z_src":["z_src","zbest","z_mc","zphot","photoz"]
    })
    need = {"ra","dec","e1","e2","w","z_src"}
    if not need.issubset(src.columns): raise SystemExit(f"sources missing {need - set(src.columns)}")
    # mild quality cuts
    src = src[np.isfinite(src["ra"]) & np.isfinite(src["dec"]) &
              np.isfinite(src["e1"]) & np.isfinite(src["e2"]) &
              np.isfinite(src["w"])  & np.isfinite(src["z_src"])].copy()
    # keep plausible shapes and positive weights
    src = src[(np.abs(src["e1"])<1.5) & (np.abs(src["e2"])<1.5) & (src["w"]>0)]
    src.to_parquet(f"{args.outdir}/sources.parquet", index=False)

    print("[prepare] wrote",
          f"{args.outdir}/clusters.parquet ({len(cl)}) and",
          f"{args.outdir}/sources.parquet ({len(src)})")

if __name__ == "__main__":
    main()
