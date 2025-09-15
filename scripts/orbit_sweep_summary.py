#!/usr/bin/env python3
import re, argparse, json
from pathlib import Path
import numpy as np, pandas as pd

PAT_EN = re.compile(r"mean \|dE/E0\|\s*≈\s*([0-9eE\+\-\.]+).*mean \|d\|\|L\|\|/\|\|L\|\|0\|\s*≈\s*([0-9eE\+\-\.]+)")
PAT_CENT = re.compile(r"centrality:\s*mean_tau_frac=([0-9eE\+\-\.]+),\s*radiality_err=([0-9eE\+\-\.]+)")
PAT_V = re.compile(r"vtheta=([0-9eE\+\-\.]+)\s+dE=([0-9eE\+\-\.]+)\s+dL=([0-9eE\+\-\.]+)\s+rad=([0-9eE\+\-\.]+)\s+tau_frac=([0-9eE\+\-\.]+)")

def parse_log(path: Path):
    rec = {"log": path.name}
    txt = path.read_text(errors="ignore")
    m = PAT_EN.search(txt)
    if m:
        rec["mean_dE_over_E0"] = float(m.group(1))
        rec["mean_dL_over_L0"] = float(m.group(2))
    m = PAT_CENT.search(txt)
    if m:
        rec["mean_tau_frac"] = float(m.group(1))
        rec["radiality_err"] = float(m.group(2))
    m = PAT_V.search(txt)
    if m:
        rec["vtheta"] = float(m.group(1))
        rec["dE_last"] = float(m.group(2))
        rec["dL_last"] = float(m.group(3))
        rec["rad_last"] = float(m.group(4))
        rec["tau_frac_last"] = float(m.group(5))
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="sweeps/... directory containing orbit_v*.log")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--proof_dir", default="proofs/EMERGENT_GRAVITY_ORBITS_N320_L160")
    args = ap.parse_args()

    out = Path(args.out_dir)
    logs = sorted(out.glob("orbit_v*.log"))
    if not logs:
        print("No orbit_v*.log found in", out)
        return

    rows = [parse_log(p) for p in logs]
    df = pd.DataFrame(rows)
    # Robust pick: minimize (radiality_err) with ties broken by minimal mean_dE/E0 and mean_dL/L0
    df["score"] = df["radiality_err"].fillna(np.inf)
    sidx = df.sort_values(by=["score","mean_dE_over_E0","mean_dL_over_L0"], ascending=[True, True, True]).index
    best = df.loc[sidx[0]]
    print("\n=== ORBIT SWEEP SUMMARY ===")
    print(df[["vtheta","mean_dE_over_E0","mean_dL_over_L0","radiality_err","mean_tau_frac"]].to_string(index=False))
    print("\nBest vtheta:", best.get("vtheta"), "| radiality_err:", best.get("radiality_err"),
          "| mean_dE/E0:", best.get("mean_dE_over_E0"), "| mean_d||L||/||L||0:", best.get("mean_dL_over_L0"))

    # Write CSV and PROOF summary
    out_csv = args.out_csv or (out/"orbit_sweep_metrics.csv")
    df.to_csv(out_csv, index=False)
    Path(args.proof_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.proof_dir)/"ORBIT_SWEEP_SUMMARY.md","w") as f:
        f.write("# Orbit Sweep Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n**Best vθ**: {:.3f} (radiality_err={:.4f}, mean|ΔE/E₀|={:.3e}, mean|Δ‖L‖/‖L‖₀|={:.3e})\n"
                .format(best.get("vtheta"), best.get("radiality_err"),
                        best.get("mean_dE_over_E0", np.nan), best.get("mean_dL_over_L0", np.nan)))

if __name__ == "__main__":
    main()