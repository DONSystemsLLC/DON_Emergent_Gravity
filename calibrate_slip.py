#!/usr/bin/env python3
"""
Amplitude calibration for slip analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    src = Path("outputs/CLENS_release/slip_analysis.csv")
    df = pd.read_csv(src)
    
    # Optional: load per-bin errors; if none, fall back to uniform weights  
    err = df["gt_err"].to_numpy() if "gt_err" in df.columns else np.ones(len(df))
    w = 1.0/np.clip(err, 1e-9, None)**2

    # pick calibration annulus 0.75–2.0 Mpc
    mcal = (df["r_center"] >= 0.75) & (df["r_center"] <= 2.0)
    gt_o = df["gt_observed"].to_numpy()
    gt_t = df["gt_theory"].to_numpy()
    A = float(np.sum(w[mcal]*gt_o[mcal]*gt_t[mcal]) / np.sum(w[mcal]*gt_t[mcal]**2))

    df["slip_cal"] = gt_o / (A*gt_t + 1e-18)

    out_csv = src.with_name("slip_analysis_calibrated.csv")
    df.to_csv(out_csv, index=False)

    summary = {
        "A_fit": A,
        "annulus": [0.75, 2.0],
        "mean_slip_cal": float(np.mean(df.loc[mcal, "slip_cal"])),
        "median_slip_cal": float(np.median(df.loc[mcal, "slip_cal"])),
        "rms_slip_cal": float(np.sqrt(np.mean((df.loc[mcal, "slip_cal"]-1.0)**2)))
    }
    
    Path("outputs/CLENS_release/slip_summary_calibrated.json").write_text(
        json.dumps(summary, indent=2)
    )
    
    print(f"[cal] wrote {out_csv} and slip_summary_calibrated.json with A = {A}")
    print(f"[cal] mean slip (calibrated) = {summary['mean_slip_cal']:.4f}")
    print(f"[cal] median slip (calibrated) = {summary['median_slip_cal']:.4f}")
    print(f"[cal] RMS from unity = {summary['rms_slip_cal']:.4f}")

if __name__ == "__main__":
    main()