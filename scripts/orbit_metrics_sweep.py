#!/usr/bin/env python3
import argparse, csv, os, re, subprocess, sys, time
from pathlib import Path

# Pattern for parsing numbers including nan/inf
METRIC_NUM = r"(?:[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan))"

METRICS_RE = re.compile(
    rf"METRICS\|dE_over_E=(?P<dE>{METRIC_NUM})\|"
    rf"dL_over_L=(?P<dL>{METRIC_NUM})\|"
    rf"slope_Fr=(?P<slope>{METRIC_NUM})\|"
    rf"slope_err=(?P<serr>{METRIC_NUM})\|"
    rf"precession_deg_per_orbit=(?P<prec>{METRIC_NUM})",
    re.IGNORECASE
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtheta", required=True, help="comma list, e.g. 0.25,0.28,0.30")
    ap.add_argument("--out_dir", default="sweeps/N320_L160_vtheta_sweep")
    ap.add_argument("--timeout", type=int, default=7200)
    # Pass-through physics args (match your working run)
    ap.add_argument("--N", type=int, default=320)
    ap.add_argument("--L", type=float, default=160)
    ap.add_argument("--sigma", type=float, default=0.4)
    ap.add_argument("--tau", type=float, default=0.20)
    ap.add_argument("--dt", type=float, default=0.003125)
    ap.add_argument("--steps", type=int, default=128000)
    ap.add_argument("--warm", type=int, default=0, help="0 to skip re-warm")
    ap.add_argument("--r0", type=float, default=20.0)
    ap.add_argument("--masses", default="80,80,80,1.0")
    ap.add_argument("--orbit_from_potential", action="store_true", default=True)
    ap.add_argument("--project_curl_free", action="store_true", default=False)
    ap.add_argument("--save_logs", action="store_true")
    ap.add_argument("--load_field", default=None, help="npz snapshot to reuse (skips warm)")
    return ap.parse_args()

def run_one(v, args):
    from subprocess import Popen, PIPE
    import time, sys
    cmd = [sys.executable, "-u", "src/don_emergent_collapse_3d.py",
           "--test", "orbit",
           "--N", str(args.N), "--L", str(args.L),
           "--sigma", str(args.sigma), "--tau", str(args.tau),
           "--dt", str(args.dt), "--steps", str(args.steps),
           "--r0", str(args.r0), "--vtheta", str(v),
           "--masses", args.masses]
    if args.load_field:
        cmd += ["--load_field", args.load_field]
    elif args.warm is not None:
        cmd += ["--warm", str(args.warm)]
    if args.orbit_from_potential:
        cmd += ["--orbit_from_potential"]
    if args.project_curl_free:
        cmd += ["--project_curl_free"]

    log_path = os.path.join(args.out_dir, f"log_vtheta_{v}.out")
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, bufsize=1,
                 universal_newlines=True, env=env, errors="replace")
    metrics = None
    all_output = []

    # Read stdout line by line
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            all_output.append(line)
            if line.startswith("METRICS|"):
                m = METRICS_RE.search(line)
                if m:
                    metrics = dict(
                        vtheta=float(v),
                        dE_over_E=float(m["dE"]),
                        dL_over_L=float(m["dL"]),
                        slope_Fr=float(m["slope"]),
                        slope_err=float(m["serr"]),
                        precession_deg_per_orbit=float(m["prec"])
                    )
    finally:
        proc.stdout.close()

    # Wait for process to complete
    rc = proc.wait()

    # Fallback: parse the whole buffer if the live loop never saw METRICS
    if not metrics:
        joined = "".join(all_output)
        m = METRICS_RE.search(joined)
        if m:
            metrics = dict(
                vtheta=float(v),
                dE_over_E=float(m["dE"]),
                dL_over_L=float(m["dL"]),
                slope_Fr=float(m["slope"]),
                slope_err=float(m["serr"]),
                precession_deg_per_orbit=float(m["prec"])
            )

    # Write all output to log file
    with open(log_path, "w") as lf:
        lf.writelines(all_output)
        # Add stderr if any
        err = proc.stderr.read()
        if err:
            lf.write("\n[stderr]\n"+err)

    wall = round(time.time()-t0, 3)
    if not metrics:
        raise SystemExit(f"[error] No METRICS line for vtheta={v}; see {log_path}")
    metrics["wall_sec"] = wall
    metrics["returncode"] = rc
    return metrics

def main():
    args = parse_args()

    # Echo args to verify they were passed correctly
    print(f"[args] load_field={args.load_field!r} warm={args.warm} steps={args.steps}", flush=True)

    # Fail fast if neither warm nor load_field
    if not args.load_field and (not args.warm or int(args.warm) <= 0):
        raise SystemExit("Refusing to run cold: pass --load_field or set --warm>0")

    vlist = [x.strip() for x in args.vtheta.split(",") if x.strip()]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for v in vlist:
        print(f"[running] vtheta={v}...")
        rec = run_one(v, args)
        rows.append(rec)

        # Append to CSV immediately
        cols = ["vtheta","dE_over_E","dL_over_L","slope_Fr","slope_err","precession_deg_per_orbit","wall_sec","returncode"]
        csv_path = os.path.join(args.out_dir, "summary.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if write_header: w.writeheader()
            w.writerow(rec)
        print(f"[append] {csv_path}  vtheta={v} completed")

    print(f"[done] {os.path.join(args.out_dir, 'summary.csv')}   runs={len(rows)}")

if __name__ == "__main__":
    main()