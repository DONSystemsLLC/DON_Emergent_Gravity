#!/usr/bin/env python3
import os
import glob
import time
import json
import numpy as np

def monitor_orbits():
    while True:
        print("\n" + "="*60)
        print(f"Orbit Run Monitor - {time.strftime('%H:%M:%S')}")
        print("="*60)

        # Check main run
        main_log = "outputs/orbits/N320_L160_box_v030/orbit.log"
        if os.path.exists(main_log):
            with open(main_log, 'r') as f:
                lines = f.readlines()
                recent = [l for l in lines[-20:] if 'step=' in l or 'dE=' in l]
                if recent:
                    print("\nMain run (v=0.30, 120k steps):")
                    for line in recent[-3:]:
                        print(f"  {line.strip()}")

        # Check sweep runs
        sweep_logs = glob.glob("outputs/orbits/orbit_v*.log")
        if sweep_logs:
            print("\nSweep runs (60k steps each):")
            for log in sorted(sweep_logs):
                v = log.split('_v')[-1].replace('.log', '')
                if os.path.exists(log):
                    with open(log, 'r') as f:
                        lines = f.readlines()
                        status = "running"
                        for line in reversed(lines):
                            if 'dE=' in line:
                                # Extract conservation metrics
                                parts = line.split()
                                dE = next((p.split('=')[1] for p in parts if 'dE=' in p), '?')
                                dL = next((p.split('=')[1] for p in parts if 'dL=' in p), '?')
                                print(f"  v=0.{v}: dE={dE}, dL={dL}")
                                break
                            elif 'completed' in line.lower():
                                status = "completed"
                                print(f"  v=0.{v}: {status}")
                                break

        # Check completed trajectory files
        traj_files = glob.glob("outputs/orbits/*/trajectory.npz")
        if traj_files:
            print("\nCompleted trajectories:")
            for tf in sorted(traj_files):
                dirname = os.path.basename(os.path.dirname(tf))
                print(f"  {dirname}: trajectory.npz saved")

        time.sleep(5)

if __name__ == "__main__":
    try:
        monitor_orbits()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")