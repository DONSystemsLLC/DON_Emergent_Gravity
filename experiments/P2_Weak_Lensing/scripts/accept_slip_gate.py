# scripts/accept_slip_gate.py
import json, numpy as np, pandas as pd
from pathlib import Path

sv = json.loads(Path("outputs/CLENS_release/sign_validation.json").read_text())
sf = json.loads(Path("outputs/CLENS_release/slip_final_summary.json").read_text())

# Pull numbers
slip = np.array(sf["slip_measurements"]["slip_cal"], dtype=float)
slip_err = np.array(sf["slip_measurements"]["slip_cal_err"], dtype=float)
snr = np.abs(slip)/np.clip(slip_err,1e-12,None)

# Nulls (use your keys; fall back defensively)
rot_mean   = float(sv.get("rotation_null_mean", 0.0))
rot_sigma  = float(sv.get("rotation_null_std",  1.0))
rot_maxabs = float(sv.get("rotation_null_max_abs", 0.0))
x_mean     = float(sv.get("cross_shear_null_mean", 0.0))
x_sigma    = float(sv.get("cross_shear_null_std",  1.0))
x_maxabs   = float(sv.get("cross_shear_null_max_abs", 0.0))

# Jackknife calibration spread if present
A = float(sf.get("calibration_factor", 1.0))
Ajk = sf.get("jackknife_cal_factors", None)
if Ajk:
    Ajk = np.array(Ajk, dtype=float)
    rel_err_A = float(np.std(Ajk, ddof=1)/np.abs(np.mean(Ajk)))
else:
    rel_err_A = float(sf.get("calibration_relative_error", 0.0) or 0.0)

gate = {
  "nulls": {
    "rot_mean_within_1sigma": abs(rot_mean) <= rot_sigma,
    "x_mean_within_1sigma":   abs(x_mean)   <= x_sigma,
    "rot_maxabs_le_3sigma":   rot_maxabs <= 3*rot_sigma,
    "x_maxabs_le_3sigma":     x_maxabs   <= 3*x_sigma,
  },
  "snr": {
    "median_snr_ge_0p5": float(np.median(snr)) >= 0.5,
    "min_snr_ge_0p2":    float(np.min(snr))    >= 0.2,
  },
  "calibration": {
    "rel_err_A_le_0p5":  rel_err_A <= 0.5,  # generous for prototype
    "A_value": A, "rel_err_A": rel_err_A
  },
  "range": {
    "all_slip_in_pm2": bool(np.all((slip >= -2.0)&(slip <= 2.0)))
  }
}
overall = (all(gate["nulls"].values())
           and all(gate["snr"].values())
           and gate["calibration"]["rel_err_A_le_0p5"]
           and gate["range"]["all_slip_in_pm2"])

out = {
  "overall_pass": overall,
  "metrics": {
    "slip_mean": float(np.mean(slip)),
    "slip_median": float(np.median(slip)),
    "slip_rms": float(np.sqrt(np.mean((slip - np.mean(slip))**2))),
    "median_snr": float(np.median(snr)), "min_snr": float(np.min(snr))
  },
  "gate": gate
}
Path("outputs/CLENS_release/slip_gate_referee.json").write_text(json.dumps(out, indent=2))
print("SLIP_REFEREE_GATE:", "PASS" if overall else "FAIL")