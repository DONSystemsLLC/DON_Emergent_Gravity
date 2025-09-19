#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astroquery.vizier import Vizier
import pyarrow as pa, pyarrow.parquet as pq

# ---------------------------------------------
# Helpers to map columns flexibly
# ---------------------------------------------
def pick_col(cands, cols):
    cl = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in cl:
            return cl[c.lower()]
    return None

def std_table(tab):
    """Return (DataFrame[ra,dec,z_cl]) or None if columns missing."""
    df = tab.to_pandas()
    cols = list(df.columns)

    ra  = pick_col(["RAJ2000","RA_ICRS","_RAJ2000","_RA","RAdeg","RA"], cols)
    dec = pick_col(["DEJ2000","DE_ICRS","_DEJ2000","_DE","DEdeg","DEC"], cols)
    z   = pick_col(["z_lambda","zcl","zspec","z","z_phot","zred","zbest","zphoto"], cols)

    if not (ra and dec and z):
        return None

    out = df[[ra,dec,z]].copy()
    out.columns = ["ra","dec","z_cl"]

    # coerce numerics & drop junk
    for c in ["ra","dec","z_cl"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["ra","dec","z_cl"])
    return out

# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    srcp = Path("data/CLENS/sources.parquet")
    if not srcp.exists():
        sys.exit("[error] data/CLENS/sources.parquet not found. Build KiDS sources first.")

    # Footprint & region radius from your actual KiDS sources
    s = pq.read_table(srcp, columns=["ra","dec"]).to_pandas()
    ra_min, ra_max = float(s["ra"].min()), float(s["ra"].max())
    dec_min,dec_max= float(s["dec"].min()), float(s["dec"].max())
    ra_c, dec_c = 0.5*(ra_min+ra_max), 0.5*(dec_min+dec_max)

    # crude half-diagonal radius in deg
    dx = (ra_max - ra_min)*np.cos(np.deg2rad(dec_c))
    dy = (dec_max - dec_min)
    radius_deg = 0.6*np.hypot(dx, dy)  # add buffer
    if radius_deg < 0.5: radius_deg = 0.5

    center = SkyCoord(ra_c*u.deg, dec_c*u.deg, frame='icrs')
    print(f"[footprint] RA[{ra_min:.3f},{ra_max:.3f}] Dec[{dec_min:.3f},{dec_max:.3f}]  center=({ra_c:.3f},{dec_c:.3f})  R≈{radius_deg:.2f}°")

    Vizier.ROW_LIMIT = -1
    # find candidate catalogs (redMaPPer SDSS/DR8/DR9, DES, etc.)
    cand = Vizier.find_catalogs("redMaPPer")
    keys = list(cand.keys())
    if not keys:
        sys.exit("[error] No VizieR catalogs matching 'redMaPPer' were found.")

    print("[vizier] candidates:")
    for k in keys:
        print("  -", k, cand[k].description.splitlines()[0][:80])

    # Try each candidate via query_region; keep those with usable cols
    dfs = []
    for cat in keys:
        try:
            tabs = Vizier(columns=["*"]).query_region(center, radius=Angle(radius_deg, "deg"), catalog=cat)
        except Exception as e:
            print(f"[warn] query failed for {cat}: {e}")
            continue
        if len(tabs)==0: 
            continue
        kept_any = False
        for t in tabs:
            std = std_table(t)
            if std is None or std.empty: 
                continue
            # keep only within the true RA/Dec box (region query is circular)
            std = std[(std["ra"]>=ra_min)&(std["ra"]<=ra_max)&(std["dec"]>=dec_min)&(std["dec"]<=dec_max)]
            if std.empty: 
                continue
            dfs.append(std)
            kept_any = True
        print(f"[vizier] {cat} -> kept={kept_any}")

    if not dfs:
        sys.exit("[error] No clusters found in footprint from VizieR redMaPPer catalogs.")

    cl = pd.concat(dfs, ignore_index=True)
    # de-duplicate exact matches
    cl = cl.drop_duplicates(subset=["ra","dec","z_cl"])

    out_csv = Path("data/CLENS/clusters.csv")
    out_par = out_csv.with_suffix(".parquet")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cl.to_csv(out_csv, index=False)
    pq.write_table(pa.Table.from_pandas(cl), out_par, compression="zstd")
    print(f"[clusters] wrote {out_csv}  rows={len(cl)}")

if __name__ == "__main__":
    main()
