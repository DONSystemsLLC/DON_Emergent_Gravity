# scripts/plot_slip_panels.py
import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv("outputs/CLENS_release/slip_analysis_final.csv")
r, gt, gt_th = df["r_center"].to_numpy(), df["gt_observed"].to_numpy(), df["gt_theory"].to_numpy()
ge = df["gt_stderr"].to_numpy() if "gt_stderr" in df else np.full_like(gt, np.nan)
slip = df["slip_cal"].to_numpy(); se = df["slip_cal_err"].to_numpy() if "slip_cal_err" in df else np.full_like(gt, np.nan)

# Top: tangential shear
plt.figure(figsize=(6,7))
plt.subplot(2,1,1)
plt.errorbar(r, gt, yerr=ge, fmt='o', label='Observed $g_t$')
plt.plot(r, gt_th, label='Theory $g_t^{\\rm th}$')
plt.xscale('log'); plt.ylabel('$g_t$'); plt.legend(); plt.title('CLENS — W1 Slip Prototype')

# Bottom: slip
plt.subplot(2,1,2)
plt.errorbar(r, slip, yerr=se, fmt='o', label='$s(R)=g_t/g_t^{\\rm th}$')
plt.axhline(1.0, ls='--', label='$s=1$')
plt.xscale('log'); plt.xlabel('R [Mpc]'); plt.ylabel('Slip $s$'); plt.legend()
plt.tight_layout()
plt.savefig("figs/wl_slip_panels.png", dpi=160)