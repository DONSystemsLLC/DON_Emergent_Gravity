#!/usr/bin/env python3
import os, platform, subprocess, sys, yaml, json, time
from pathlib import Path

def sh(cmd): return subprocess.check_output(cmd, shell=True, text=True).strip()

def main():
    out = Path([a for a in sys.argv if a.startswith("--out=")][0].split("=",1)[1]) if any(a.startswith("--out=") for a in sys.argv) else Path("proofs/P2_RELEASE/MANIFEST.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Core metadata
    commit = sh("git rev-parse HEAD")
    branch = sh("git rev-parse --abbrev-ref HEAD")
    py = sh(". .venv/bin/activate && python -V || python -V")
    pip_freeze = sh(". .venv/bin/activate && python -m pip freeze || echo ''")

    # Physics gates
    gate_strict = Path("sweeps/validation_orbit_strict/summary.csv")
    wl_gate = Path("outputs/CLENS_release/slip_gate_referee.json")
    wl_gate_json = json.loads(wl_gate.read_text()) if wl_gate.exists() else {}

    # Key artifacts
    arts = [
        "figs/slope_window_heatmap.png",
        "figs/helmholtz_checks.json",
        "figs/kepler_fit.csv",
        "figs/wl_slip_panels.png",
        "outputs/CLENS_release/slip_analysis_final.csv",
        "outputs/CLENS_release/slip_final_summary.json",
        "outputs/CLENS_release/slip_acceptance_gate.json",
        "outputs/CLENS_release/slip_gate_referee.json",
        "proofs/EMERGENT_GRAVITY_ORBITS_N320_L160/validation_force_profile.csv",
        "docs/REPORT.md",
    ]
    arts = [a for a in arts if Path(a).exists()]

    manifest = {
        "name": "DON Emergent Gravity — P2 Universality & Slip",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"commit": commit, "branch": branch, "remote": sh("git config --get remote.origin.url || echo ''")},
        "system": {"python": py, "platform": platform.platform(), "machine": platform.machine()},
        "env": {"pip_freeze": pip_freeze.splitlines()},
        "acceptance": {
            "physics_gate": "gate-strict",
            "wl_gate_referee": wl_gate_json,
        },
        "artifacts": arts,
        "notes": "P2 bundle contains strict physics acceptance, WL slip with nulls + jackknife, report, and provenance."
    }

    out.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"[manifest] wrote {out}")

if __name__ == "__main__":
    if "--out" not in " ".join(sys.argv):
        Path("proofs/P2_RELEASE").mkdir(parents=True, exist_ok=True)
    main()