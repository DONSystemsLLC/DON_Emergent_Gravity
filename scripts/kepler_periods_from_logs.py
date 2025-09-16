#!/usr/bin/env python3
# Robust Kepler periods & flatness from orbit logs
import argparse, glob, numpy as np, pandas as pd
from pathlib import Path

def unwrap_1d(x, L):
    x = np.asarray(x, float)
    dx = np.diff(x)
    dx -= L * np.round(dx / L)
    return np.concatenate([[x[0]], x[0] + np.cumsum(dx)])

def find_local_extrema(y, kind="min", min_gap=25):
    """
    Find local minima ('min') or maxima ('max') with a minimum index separation (min_gap).
    Simple & robust without SciPy.
    """
    if kind not in ("min", "max"):
        raise ValueError("kind must be 'min' or 'max'")
    if kind == "min":
        m = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
    else:
        m = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    idx = np.where(m)[0] + 1
    if idx.size == 0:
        return idx
    kept = [idx[0]]
    for k in idx[1:]:
        if k - kept[-1] >= min_gap:
            kept.append(k)
    return np.array(kept, dtype=int)

def extract_T_a_flat(log_csv, Lbox, min_gap=25):
    df = pd.read_csv(log_csv)
    t = df["t"].to_numpy()
    r3 = df[["rx", "ry", "rz"]].to_numpy()
    v3 = df[["vx", "vy", "vz"]].to_numpy()

    ru = np.column_stack([unwrap_1d(r3[:, i], Lbox) for i in range(3)])
    r = np.linalg.norm(ru, axis=1)
    eps = 1e-12
    r_safe = np.maximum(r, eps)

    idx_min = find_local_extrema(r, kind="min", min_gap=min_gap)
    idx_max = find_local_extrema(r, kind="max", min_gap=min_gap)
    if idx_min.size < 3:
        return None
    dT = np.diff(t[idx_min])
    dT = dT[dT > 0]
    if dT.size < 2:
        return None
    T = float(np.median(dT))
    r_per = float(np.median(r[idx_min])) if idx_min.size else float(np.median(r))
    r_apo = float(np.median(r[idx_max])) if idx_max.size else float(np.median(r))
    a = 0.5 * (r_per + r_apo)

    Lvec = np.cross(ru, v3)
    L2 = np.einsum("ij,ij->i", Lvec, Lvec)
    vtheta2r = L2 / r_safe
    med = np.median(vtheta2r)
    flat = float(np.median(np.abs(vtheta2r - med)) / (med + eps)) if med > 0 else float("nan")
    return float(T), float(a), flat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="sweeps/kepler_v*/logs/*orbit_log.csv")
    ap.add_argument("--L", type=float, default=160.0)
    ap.add_argument("--min_gap", type=int, default=25, help="min index separation between extrema")
    ap.add_argument("--out", default="figs/kepler_fit.csv")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(args.glob)):
        res = extract_T_a_flat(p, args.L, min_gap=args.min_gap)
        if res is None:
            continue
        T, a, flat = res
        rows.append((p, T, a, flat))

    if not rows:
        print("[kepler] no usable logs found")
        return

    df = pd.DataFrame(rows, columns=["log", "T", "a", "flatness_mad_over_med"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    x = np.log(df["a"].to_numpy())
    y = np.log(np.square(df["T"].to_numpy()))
    slope = np.polyfit(x, y, 1)[0]
    print(f"[kepler] slope dlog(T^2)/dlog(a) = {slope:.3f} (target 1.500±0.03)")
    print(f"[kepler] mean flatness (MAD/med) = {df['flatness_mad_over_med'].mean():.3f} (target ≤ 0.03)")
    print(f"[kepler] wrote {args.out}")

if __name__ == "__main__":
    main()
