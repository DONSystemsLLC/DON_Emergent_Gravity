#!/usr/bin/env python3
import glob, pandas as pd, numpy as np
paths = sorted(glob.glob("sweeps/ep_m*/summary.csv"))
acc = []
for p in paths:
    df = pd.read_csv(p)
    # pick a robust scalar accel or mean radial accel column
    a_cols = df.filter(regex=r"(a_rad_mean|accel_r_mean|acc_mean|slope_Fr)", axis=1)
    m_cols = df.filter(regex=r"(m_test|mass_test|m4|m_last|masses)", axis=1)

    if len(a_cols.columns) > 0:
        a = a_cols.iloc[0,0]
        # Extract mass from path if not in columns
        if len(m_cols.columns) > 0:
            m = m_cols.iloc[0,0]
        else:
            # Extract from path like sweeps/ep_m0.5/
            import re
            match = re.search(r'ep_m([\d.]+)', p)
            if match:
                m = float(match.group(1))
            else:
                continue
        acc.append((m, a))

if acc:
    acc = sorted(acc)  # by mass
    a_vals = np.array([x[1] for x in acc], float)
    rel_spread = (a_vals.max() - a_vals.min())/np.mean(a_vals) if np.mean(a_vals) != 0 else float('inf')
    print("[ep] accelerations:", acc)
    print(f"[ep] normalized spread = {abs(rel_spread)*100:.3f}%  (target < 0.5%)")
else:
    print("[ep] No EP test data found yet")