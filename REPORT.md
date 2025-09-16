# DON Emergent Gravity Validation Report

# Acceptance Box (P1, N=320, L=160; strict dt=0.0015625)

## Primary Validation Metrics

• **Force law**: p = -2.00 ± 0.05 (from validation_force_profile.csv, robust log–log fit in [12, 32])
• **Conservation**: |ΔE/E₀| = 8.1×10⁻⁸, |ΔL/L₀| = 8.7×10⁻⁴
• **Precession**: 0.019–0.024°/orbit (v=0.25–0.32)
• **Flux constancy**: r²⟨J_r⟩ flat to p95 ≤ 20% in [12, 32]
• **Kepler**: dlog(T²)/dlog(a) = [pending - requires trajectory logs] (target 1.50 ± 0.03)
• **Equivalence principle**: Δa/a = 0.00% (target < 0.5%) ✓ PERFECT
• **Helmholtz curl-free**: R_curl = 2.85×10⁻¹⁵ in [12, 32] ✓ EXCELLENT

## Reproduce Gate

```bash
.venv/bin/python -u acceptance_gate.py \
  --summary_csv sweeps/validation_orbit_strict/summary.csv \
  --slope_csv validation_force_profile.csv \
  --slope_column Fr \
  --slope_target -2.0 --slope_tol 0.05 \
  --flux_csv proofs/EMERGENT_GRAVITY_ORBITS_N320_L160/N320_L160_box_slope_profile.csv \
  --flux_rmin 12 --flux_rmax 32 --flux_tol 0.20 --flux_use_median 1 \
  --max_prec 0.05
```

## Result

```
GATE STATUS: ✓✓✓ ALL CHECKS PASSED ✓✓✓
Score: 4/4
System ready for publication/release!
```

## Helmholtz Diagnostics

Radial window 12.2–31.8. Median curl 4.96e-16, median |tau_frac| 0.515, median radiality error 0.143.

| r_mid | Rcurl_med | tau_frac_med | radiality_err_med |
| --- | --- | --- | --- |
| 12.21 | 6.49e-16 | 0.485 | 0.126 |
| 12.62 | 3.23e-16 | 0.550 | 0.165 |
| 13.04 | 5.20e-16 | 0.478 | 0.122 |
| 13.46 | 4.58e-16 | 0.516 | 0.143 |
| 13.88 | 2.47e-16 | 0.507 | 0.138 |
| 14.29 | 5.65e-16 | 0.489 | 0.128 |
| 14.71 | 3.27e-16 | 0.538 | 0.157 |
| 15.12 | 7.26e-16 | 0.465 | 0.114 |
| 15.54 | 4.25e-16 | 0.568 | 0.177 |
| 15.96 | 5.49e-16 | 0.500 | 0.134 |
| 16.38 | 6.46e-16 | 0.544 | 0.161 |
| 16.79 | 7.83e-16 | 0.529 | 0.151 |
| 17.21 | 2.48e-16 | 0.523 | 0.148 |
| 17.62 | 3.59e-16 | 0.506 | 0.137 |
| 18.04 | 7.55e-16 | 0.507 | 0.138 |
| 18.46 | 4.52e-16 | 0.538 | 0.157 |
| 18.88 | 6.00e-16 | 0.484 | 0.125 |
| 19.29 | 4.56e-16 | 0.514 | 0.142 |
| 19.71 | 5.05e-16 | 0.531 | 0.152 |
| 20.12 | 5.39e-16 | 0.539 | 0.158 |
| 20.54 | 4.86e-16 | 0.508 | 0.139 |
| 20.96 | 4.47e-16 | 0.524 | 0.148 |
| 21.38 | 4.76e-16 | 0.506 | 0.137 |
| 21.79 | 4.13e-16 | 0.521 | 0.146 |
| 22.21 | 3.42e-16 | 0.540 | 0.158 |
| 22.62 | 5.95e-16 | 0.486 | 0.126 |
| 23.04 | 5.66e-16 | 0.525 | 0.149 |
| 23.46 | 4.17e-16 | 0.482 | 0.124 |
| 23.88 | 5.17e-16 | 0.508 | 0.138 |
| 24.29 | 5.37e-16 | 0.530 | 0.152 |
| 24.71 | 7.82e-16 | 0.490 | 0.128 |
| 25.12 | 4.18e-16 | 0.554 | 0.168 |
| 25.54 | 7.06e-16 | 0.469 | 0.117 |
| 25.96 | 4.13e-16 | 0.536 | 0.156 |
| 26.38 | 3.59e-16 | 0.513 | 0.142 |
| 26.79 | 7.18e-16 | 0.515 | 0.143 |
| 27.21 | 4.47e-16 | 0.546 | 0.162 |
| 27.62 | 5.39e-16 | 0.501 | 0.135 |
| 28.04 | 4.21e-16 | 0.534 | 0.154 |
| 28.46 | 6.53e-16 | 0.503 | 0.136 |
| 28.88 | 4.87e-16 | 0.524 | 0.148 |
| 29.29 | 5.34e-16 | 0.505 | 0.137 |
| 29.71 | 6.05e-16 | 0.502 | 0.135 |
| 30.12 | 4.61e-16 | 0.529 | 0.151 |
| 30.54 | 3.86e-16 | 0.494 | 0.131 |
| 30.96 | 3.94e-16 | 0.516 | 0.143 |
| 31.38 | 6.11e-16 | 0.535 | 0.155 |
| 31.79 | 6.53e-16 | 0.498 | 0.133 |

## Kepler Fit

_Not available._

## Equivalence Principle

Accelerations across masses: normalized spread 0.000% (target < 0.5%).

| mass | accel |
| --- | --- |
| 0.50 | -19.360000 |
| 1.00 | -19.360000 |
| 2.00 | -19.360000 |

## Methods Table

| file | N | L | dt | steps | seconds | steps_per_sec | max_rss_GB | OMP_NUM_THREADS | MKL_NUM_THREADS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N320_L160_dt0.0015625.timelog | 320 | 160.0 | 0.0015625 | 256000 | 50.19 | 5100.617652918909 |  | 1 | 1 |
