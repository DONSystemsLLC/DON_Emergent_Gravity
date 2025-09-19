#!/usr/bin/env python3
"""
Parse orbit log files to extract METRICS and create/append a summary CSV.

Usage:
  python parse_orbit_logs.py                # defaults to sweeps/N320_L160_vtheta_sweep
  python parse_orbit_logs.py path/to/sweep  # custom sweep directory
  python parse_orbit_logs.py --append       # append to existing summary.csv instead of overwriting
"""

import csv, os, re, sys
from pathlib import Path
import argparse
import math

# Accept numbers or nan/inf with optional sign
NUM = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf)"

METRICS_RE = re.compile(
    rf"METRICS\|dE_over_E=(?P<dE>{NUM})\|dL_over_L=(?P<dL>{NUM})\|"
    rf"slope_Fr=(?P<slope>{NUM})\|slope_err=(?P<serr>{NUM})\|"
    rf"precession_deg_per_orbit=(?P<prec>{NUM})",
    re.IGNORECASE
)

def _to_float(s: str) -> float:
    s = s.strip()
    if s.lower() in ("nan",):
        return float("nan")
    if s.lower() in ("inf", "+inf"):
        return float("inf")
    if s.lower() == "-inf":
        return float("-inf")
    return float(s)

def parse_log(log_path: Path):
    """Return dict with metrics for this log, or None if no METRICS line found."""
    # vtheta from filename: log_vtheta_<value>.out
    m = re.search(r"log_vtheta_([0-9.]+)\.out$", log_path.name)
    vtheta = float(m.group(1)) if m else float("nan")

    lines = log_path.read_text(errors="replace").splitlines()
    # take the LAST METRICS line if multiple
    match = None
    for line in reversed(lines):
        if line.startswith("METRICS|"):
            m2 = METRICS_RE.match(line)
            if m2:
                match = m2
                break
    if not match:
        print(f"[warn] No METRICS line in {log_path.name}")
        return None

    d = match.groupdict()
    return dict(
        vtheta=vtheta,
        dE_over_E=_to_float(d["dE"]),
        dL_over_L=_to_float(d["dL"]),
        slope_Fr=_to_float(d["slope"]),
        slope_err=_to_float(d["serr"]),
        precession_deg_per_orbit=_to_float(d["prec"]),
        # We don't have wall time; keep placeholders so schema matches sweep summary
        wall_sec=float("nan"),
        returncode=0,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", nargs="?", default="sweeps/N320_L160_vtheta_sweep")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <sweep_dir>/summary.csv)")
    ap.add_argument("--append", action="store_true", help="Append to existing summary.csv (else overwrite)")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_csv = Path(args.out) if args.out else (sweep_dir / "summary.csv")
    logs = sorted(sweep_dir.glob("log_vtheta_*.out"))
    if not logs:
        print(f"[parse] No log files found in {sweep_dir}")
        return

    rows = []
    for p in logs:
        rec = parse_log(p)
        if rec:
            rows.append(rec)

    if not rows:
        print("[parse] No valid METRICS found.")
        return

    cols = ["vtheta","dE_over_E","dL_over_L","slope_Fr","slope_err",
            "precession_deg_per_orbit","wall_sec","returncode"]

    if args.append and out_csv.exists():
        # naive append; no de-dup
        mode = "a"
        write_header = False
    else:
        mode = "w"
        write_header = True

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[parse] Wrote {out_csv} with {len(rows)} rows ({'append' if args.append else 'overwrite'}).")

if __name__ == "__main__":
    main()