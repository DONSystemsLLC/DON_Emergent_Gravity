#!/usr/bin/env python3
"""
Recompute angular momentum using unwrapped positions and half-step velocities.
This corrects for PBC wrapping artifacts and leapfrog time-slice issues.
"""
import numpy as np
import sys

def unwrap_position(pos_history, L):
    """Unwrap positions to handle PBC jumps."""
    pos = np.array(pos_history)
    unwrapped = np.zeros_like(pos)
    unwrapped[0] = pos[0]

    for i in range(1, len(pos)):
        delta = pos[i] - pos[i-1]
        # Detect and correct PBC jumps
        jumps = np.abs(delta) > L/2
        delta[jumps] -= np.sign(delta[jumps]) * L
        unwrapped[i] = unwrapped[i-1] + delta

    return unwrapped

def compute_L_halfstep(pos_history, vel_history, mass, L):
    """
    Compute angular momentum using half-step velocities.
    For leapfrog: v(t+dt/2) = [x(t+dt) - x(t)]/dt
    """
    pos = unwrap_position(pos_history, L)

    # Compute half-step velocities from position differences
    vel_half = np.zeros_like(pos[:-1])
    dt = 1.0  # Normalized, actual dt handled in caller

    for i in range(len(pos)-1):
        vel_half[i] = (pos[i+1] - pos[i]) / dt

    # Compute angular momentum at each time
    L_vec = []
    for i in range(len(vel_half)):
        r = pos[i] - L/2  # Center at box center
        v = vel_half[i]
        L_i = mass * np.cross(r, v)
        L_vec.append(L_i)

    return np.array(L_vec)

def analyze_orbit_log(logfile, dt=0.003125, L=160.0, mass=1.0):
    """Parse orbit log and recompute L properly."""
    pos_history = []
    vel_history = []

    with open(logfile, 'r') as f:
        for line in f:
            if line.startswith("t="):
                # Parse: t=0.000 x=[20.00000, 0.00000, 0.00000] v=[0.00000, 0.30000, 0.00000]
                parts = line.split()
                t = float(parts[0].split('=')[1])

                # Extract position
                x_str = line.split('x=')[1].split(']')[0] + ']'
                x = eval(x_str)  # Safe since we control the format
                pos_history.append(x)

                # Extract velocity
                v_str = line.split('v=')[1].split(']')[0] + ']'
                v = eval(v_str)
                vel_history.append(v)

    pos_history = np.array(pos_history)
    vel_history = np.array(vel_history)

    # Recompute L with unwrapping and half-step correction
    L_vec = compute_L_halfstep(pos_history, vel_history, mass, L)

    # Compute conservation metrics
    L_mag = np.linalg.norm(L_vec, axis=1)
    L0 = L_mag[0]
    dL_over_L = (L_mag - L0) / L0

    print(f"Recomputed Angular Momentum Analysis:")
    print(f"  Initial L: {L0:.6e}")
    print(f"  Final L: {L_mag[-1]:.6e}")
    print(f"  Max |ΔL/L|: {np.max(np.abs(dL_over_L)):.6e}")
    print(f"  RMS ΔL/L: {np.sqrt(np.mean(dL_over_L**2)):.6e}")

    # Check for systematic drift
    times = np.arange(len(L_mag)) * dt
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(times, L_mag)
    drift_per_orbit = slope * (2*np.pi*20/0.30)  # Approximate orbit period

    print(f"  Drift analysis:")
    print(f"    Slope: {slope:.6e} per time unit")
    print(f"    Per orbit: {drift_per_orbit/L0:.6e} (fraction of L0)")
    print(f"    R²: {r_value**2:.6f}")

    return L_vec, dL_over_L

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recompute_L_unwrapped.py <orbit_log_file> [dt] [L] [mass]")
        sys.exit(1)

    logfile = sys.argv[1]
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.003125
    L = float(sys.argv[3]) if len(sys.argv) > 3 else 160.0
    mass = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    analyze_orbit_log(logfile, dt, L, mass)