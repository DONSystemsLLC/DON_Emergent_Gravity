#!/usr/bin/env python3
import argparse, csv, math, os, pandas as pd

def score(row):
    # lower is better: energy & L drift + slope dev from -2
    de = abs(row["dE_over_E"])
    dl = abs(row["dL_over_L"])
    ds = abs(row["slope_Fr"] + 2.0)
    # light weights favor physical stability first
    return 5*de + 5*dl + 1*ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="sweeps/N320_L160_vtheta_sweep/summary.csv")
    ap.add_argument("--out", default="sweeps/N320_L160_vtheta_sweep/best.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    df["score"] = df.apply(score, axis=1)
    best = df.sort_values("score", ascending=True).iloc[0]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(
            "Best vtheta={:.3f}  slope={:.3f}±{:.3f}  dE/E={:.3e}  dL/L={:.3e}  precession={:.3f} deg/orbit\n"
            .format(best["vtheta"], best["slope_Fr"], best["slope_err"], best["dE_over_E"], best["dL_over_L"], best["precession_deg_per_orbit"])
        )
    print(open(args.out).read().strip())

if __name__ == "__main__":
    main()