#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd

def shell_average(r, vals, nbins=64, rmin=None, rmax=None):
    """Compute radial shell averages."""
    rmin = rmin if rmin is not None else r.min()
    rmax = rmax if rmax is not None else r.max()
    edges = np.linspace(rmin, rmax, nbins+1)
    idx = np.digitize(r, edges)-1
    rows = []
    for i in range(nbins):
        m = idx == i
        if not np.any(m):
            continue
        r_mid = 0.5*(edges[i]+edges[i+1])
        rows.append((r_mid, np.median(vals[m]), np.mean(vals[m])))
    return pd.DataFrame(rows, columns=["r","Fr_med","Fr_mean"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_csv", required=True,
                    help="orbit field sample CSV with columns r, Fr or gradPhi_r or a_r")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--use_col", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.log_csv)

    # Find r column
    r_col = None
    for rc in ["r", "r_mid"]:
        if rc in df.columns:
            r_col = rc
            break
    if not r_col:
        raise SystemExit("No 'r' or 'r_mid' column found in CSV")

    r = df[r_col].to_numpy()

    # Find force/accel column
    cand = [args.use_col, "Fr", "F_r", "a_r", "gradPhi_r", "g_r"]
    col = None
    for c in cand:
        if c and c in df.columns:
            col = c
            break

    if not col:
        raise SystemExit("No force/accel column found; pass --use_col.")

    prof = shell_average(r, df[col].to_numpy(), nbins=64)
    prof.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} using column {col}")

if __name__ == "__main__":
    main()