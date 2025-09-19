#!/usr/bin/env python3
"""
DON Emergent Gravity (deg) — tiny CLI

One-command helpers for reviewers:
  - simulate: run slope and/or orbit tests with defaults
  - kernel:   analyze field slope from warm snapshot
  - orbits:   analyze orbit sweep/metrics
  - bundle:   write SHA256SUMS.txt for proofs/**
  - all:      simulate → kernel → orbits → bundle

The commands wrap existing scripts so they mirror the README examples.
"""
import argparse, subprocess, sys, shlex

def run(cmd: str) -> int:
    print(f"[deg] $ {cmd}")
    return subprocess.call(shlex.split(cmd))

def cmd_simulate(args) -> int:
    rc = 0
    if args.which in ("slope", "both"):
        rc |= run(
            "python src/don_emergent_collapse_3d.py --test slope "
            f"--N {args.N} --L {args.L} --sigma {args.sigma} --tau {args.tau} "
            f"--dt {args.dt_slope} --warm {args.warm} --rmin {args.rmin} --rmax {args.rmax} "
            f"--masses {args.masses} --project_curl_free"
        )
    if args.which in ("orbit", "both"):
        rc |= run(
            "python src/don_emergent_collapse_3d.py --test orbit "
            f"--N {args.N} --L {args.L} --sigma {args.sigma} --tau {args.tau} "
            f"--dt {args.dt_orbit} --warm {args.warm} --steps {args.steps} "
            f"--r0 {args.r0} --vtheta {args.vtheta} "
            f"--masses {args.masses} --orbit_from_potential"
        )
    return rc

def cmd_kernel(_args) -> int:
    # Uses warm snapshot in fields/ as per experiment script
    return run("python experiments/P1_Emergent_Kernel/scripts/analyze_field_slope.py")

def cmd_orbits(_args) -> int:
    # Summaries/metrics over produced orbit logs
    return run("python experiments/P1_Kepler_Orbits/scripts/orbit_production_summary.py")

def cmd_bundle(_args) -> int:
    return run("python tools/hash_bundle.py")

def cmd_all(args) -> int:
    rc = cmd_simulate(args)
    rc |= cmd_kernel(args)
    rc |= cmd_orbits(args)
    rc |= cmd_bundle(args)
    return rc

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DON Emergent Gravity mini CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # simulate
    ps = sub.add_parser("simulate", help="Run slope/orbit simulator")
    ps.add_argument("--which", choices=["slope","orbit","both"], default="both")
    ps.add_argument("--N", type=int, default=192)
    ps.add_argument("--L", type=float, default=80.0)
    ps.add_argument("--sigma", type=float, default=1.0)
    ps.add_argument("--tau", type=float, default=0.20)
    ps.add_argument("--warm", type=int, default=8000)
    ps.add_argument("--dt-slope", dest="dt_slope", type=float, default=0.05)
    ps.add_argument("--rmin", type=float, default=5.0)
    ps.add_argument("--rmax", type=float, default=20.0)
    ps.add_argument("--dt-orbit", dest="dt_orbit", type=float, default=0.025)
    ps.add_argument("--steps", type=int, default=16000)
    ps.add_argument("--r0", type=float, default=10.0)
    ps.add_argument("--vtheta", type=float, default=0.32)
    ps.add_argument("--masses", default="32,32,32,1.0")
    ps.set_defaults(func=cmd_simulate)

    # kernel
    pk = sub.add_parser("kernel", help="Analyze kernel slope from warm field")
    pk.set_defaults(func=cmd_kernel)

    # orbits
    po = sub.add_parser("orbits", help="Analyze orbit logs/summaries")
    po.set_defaults(func=cmd_orbits)

    # bundle
    pb = sub.add_parser("bundle", help="Write SHA256SUMS for proofs/**")
    pb.set_defaults(func=cmd_bundle)

    # all
    pa = sub.add_parser("all", help="simulate → kernel → orbits → bundle")
    for a in ("--which","--N","--L","--sigma","--tau","--warm","--dt-slope","--rmin","--rmax","--dt-orbit","--steps","--r0","--vtheta","--masses"):
        for act in ps._actions:
            if act.option_strings and act.option_strings[0] == a:
                pa._add_action(act)
                break
    pa.set_defaults(func=cmd_all)

    args = p.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())

