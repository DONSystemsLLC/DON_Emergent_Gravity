#!/usr/bin/env python3
"""
Build and (optionally) stack stratified random cluster ensembles for CLENS.

- Matches the *counts per redshift window* of your focus/control lenses.
- Samples (ra, dec) from the KiDS sources footprint to respect the mask.
- Writes each random's working set into working/CLENS_random_strat3_###
  and stacks it with scripts/clens_stack.py (Σ_crit implied by --zbuf).
- Safe to resume: skips any ensemble that already has both profile CSVs.

Usage (from repo root):
  python -u scripts/clens_random_ensemble.py \
    --patch_dir working/CLENS_patch \
    --out_prefix working/CLENS_random_strat3 \
    --n_ensembles 200 --seed 42 \
    --stack_with scripts/clens_stack.py \
    --zbuf 0.2 \
    --z_focus 0.12,0.16 \
    --z_ctrl "0.06,0.10;0.18,0.30" \
    --rmin 0.2 --rmax 2.0 --nbin 12 \
    --resume
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_dir", required=True,
                    help="working/CLENS_patch with sources.parquet, clusters.parquet")
    ap.add_argument("--out_prefix", required=True,
                    help="prefix for working dirs, e.g., working/CLENS_random_strat3")
    ap.add_argument("--n_ensembles", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stack_with", default=None,
                    help="path to scripts/clens_stack.py (optional)")
    ap.add_argument("--zbuf", type=float, default=None,
                    help="use Σcrit-style background cut: zs > zl + zbuf (implies Σcrit weighting in your stacker)")
    ap.add_argument("--z_focus", default="0.12,0.16",
                    help="focus redshift window 'zmin,zmax'")
    ap.add_argument("--z_ctrl", default="0.06,0.10;0.18,0.30",
                    help="semicolon-separated control windows, e.g. 'a,b;c,d'")
    ap.add_argument("--rmin", type=float, default=None, help="optional pass-through to stacker")
    ap.add_argument("--rmax", type=float, default=None, help="optional pass-through to stacker")
    ap.add_argument("--nbin", type=int, default=None, help="optional pass-through to stacker")
    ap.add_argument("--resume", action="store_true",
                    help="skip ensembles whose outputs already exist")
    return ap.parse_args()


def load_patch(patch_dir: str):
    src = Path(patch_dir) / "sources.parquet"
    cl = Path(patch_dir) / "clusters.parquet"
    if not src.exists() or not cl.exists():
        raise SystemExit(f"[error] missing in {patch_dir} (need sources.parquet & clusters.parquet)")
    sources = pd.read_parquet(src)[["ra", "dec"]]
    clusters = pd.read_parquet(cl)[["ra", "dec", "z_cl"]]
    return sources, clusters


def parse_windows(zstr: str):
    out = []
    for seg in zstr.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        a, b = map(float, seg.split(","))
        out.append((a, b))
    return out


def count_in(df: pd.DataFrame, lo: float, hi: float) -> int:
    return int(((df["z_cl"] >= lo) & (df["z_cl"] <= hi)).sum())


def build_one_random(sources: pd.DataFrame, counts, rng: np.random.Generator) -> pd.DataFrame:
    """Sample sky positions from sources footprint; assign z per window with matched counts."""
    ra = sources["ra"].values
    dec = sources["dec"].values
    n = sum(nwant for (_, _, nwant) in counts)

    idx = rng.integers(0, len(ra), size=n)
    sky = pd.DataFrame({"ra": ra[idx], "dec": dec[idx]})

    rows = []
    start = 0
    for (lo, hi, nwant) in counts:
        if nwant <= 0:
            continue
        sub = sky.iloc[start:start + nwant].copy()
        start += nwant
        sub["z_cl"] = rng.uniform(lo, hi, size=len(sub))
        rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ra", "dec", "z_cl"])


def main():
    args = parse_args()
    sources, clusters = load_patch(args.patch_dir)

    zf = parse_windows(args.z_focus)      # focus windows (usually one)
    zc = parse_windows(args.z_ctrl)       # control windows (one or more)

    # Compute required counts per window from real clusters
    need = []
    for (lo, hi) in zf:
        need.append((lo, hi, count_in(clusters, lo, hi)))
    for (lo, hi) in zc:
        need.append((lo, hi, count_in(clusters, lo, hi)))

    outs = []
    for k in range(1, args.n_ensembles + 1):
        wdir = f"{args.out_prefix}_{k:03d}"
        odir = wdir.replace("working", "outputs")

        # resume: skip if outputs exist
        pf = Path(odir) / "profile_focus.csv"
        pc = Path(odir) / "profile_control.csv"
        if args.resume and pf.exists() and pc.exists():
            print(f"[skip] {odir} (profiles exist)")
            outs.append(odir)
            continue

        os.makedirs("data/CLENS", exist_ok=True)
        Path(wdir).mkdir(parents=True, exist_ok=True)
        Path(odir).mkdir(parents=True, exist_ok=True)

        # reproducible per-ensemble RNG
        rng = np.random.default_rng(args.seed + k)
        rnd = build_one_random(sources, need, rng)
        rnd_csv = f"data/CLENS/random_autostr_{k:03d}.csv"
        rnd.to_csv(rnd_csv, index=False)

        # prepare working set (copies sources + random clusters into wdir)
        subprocess.run([sys.executable, "scripts/clens_prepare.py",
                        "--clusters", rnd_csv,
                        "--sources", "data/CLENS/sources.csv",
                        "--outdir", wdir],
                       check=True)

        # stack with your existing stacker
        if args.stack_with:
            cmd = [sys.executable, args.stack_with,
                   "--work", wdir, "--out", odir,
                   "--z_focus", args.z_focus, "--z_ctrl", args.z_ctrl]
            if args.zbuf is not None:
                cmd += ["--zbuf", str(args.zbuf)]
            if args.rmin is not None:
                cmd += ["--rmin", str(args.rmin)]
            if args.rmax is not None:
                cmd += ["--rmax", str(args.rmax)]
            if args.nbin is not None:
                cmd += ["--nbin", str(args.nbin)]
            print("STACK:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

        outs.append(odir)

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/CLENS_random_autostr_list.json", "w") as f:
        json.dump(outs, f)
    print(f"[done] ensembles={len(outs)}  (resume={args.resume})")


if __name__ == "__main__":
    main()

