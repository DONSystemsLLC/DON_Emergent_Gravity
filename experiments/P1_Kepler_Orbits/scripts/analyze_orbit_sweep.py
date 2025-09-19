#!/usr/bin/env python3
import json, sys, pathlib

if len(sys.argv) < 2:
    print("Usage: python analyze_orbit_sweep.py sweeps/orbit_sweep_xxx.json")
    sys.exit(1)

path = pathlib.Path(sys.argv[1])
with open(path) as f:
    data = json.load(f)

results = data.get("results", [])
print(f"# Results from {path}")
print(f"{'vtheta':>8} {'dE':>12} {'dL':>12} {'rad':>10} {'tau_frac':>10} {'status':>10}")
print("-"*72)

best = None
for r in results:
    p = r.get("params", {})
    m = r.get("metrics", {})
    vtheta = p.get("vtheta")

    # Detect status
    rc = r.get("rc")
    if rc == 124:
        status = "TIMEOUT"
    elif rc == 0:
        status = "OK"
    else:
        status = f"RC={rc}"

    # Values (show "-" if None)
    dE = m.get("dE")
    dL = m.get("dL")
    rad = m.get("rad")
    tau = m.get("tau_frac")

    print(f"{vtheta:>8} {str(dE):>12} {str(dL):>12} {str(rad):>10} {str(tau):>10} {status:>10}")

    # Pick best candidate if valid numbers exist
    if dE is not None and dL is not None and status == "OK":
        score = abs(dE) + abs(dL)
        if best is None or score < best[0]:
            best = (score, vtheta, dE, dL, rad, tau)

if best:
    print("\n=== Best candidate ===")
    print(f"vtheta={best[1]}  dE={best[2]:.3e}  dL={best[3]:.3e}  rad={best[4]:.3e}  tau_frac={best[5]:.3e}")
    print("\nRun this final orbit test for plots:")
    print(f"python -u src/don_emergent_collapse_3d.py --test orbit "
          f"--N 320 --L 160 --sigma 0.4 --tau 0.20 "
          f"--dt 0.003125 --warm 40000 --steps 128000 "
          f"--r0 20.0 --vtheta {best[1]} --masses 80,80,80,1.0 "
          f"--orbit_from_potential --rotation_curve --precession_report")

