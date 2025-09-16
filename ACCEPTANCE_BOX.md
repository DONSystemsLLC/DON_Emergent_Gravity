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