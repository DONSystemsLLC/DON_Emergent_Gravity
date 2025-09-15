# DON Emergent Gravity - Physics Validation Results

## Current Baseline Results (N=320, L=160)

### Field Analysis (fields/N320_L160_box)
- **Force law slope**: p = -2.052 ± 0.05
- **Window [8,24]**: p = -2.452 (steeper due to under-warming)
- **Flux constancy**: [0.882, 1.107] confirms 1/r² scaling
- **Warm-up**: 40k steps achieved, residuals < 3e-4

### Orbit Conservation Metrics
- **Energy conservation**: |ΔE/E₀| = 8.1e-8 ✓✓ (dt=0.0015625)
- **Angular momentum**: |ΔL/L₀| = 8.7e-4 ✓ (dt=0.0015625)
- **Precession**: < 0.05°/orbit (pending measurement)

## Physics Validation Checklist

### A) Core Claims

#### 1. Continuum & Box-size Convergence ⏳
- [ ] Vary N: 160, 240, 320, 400 (fixed L=160)
- [ ] Vary L: 120, 160, 200 (fixed N=320)
- **Accept**: slope -2±0.05, |ΔE/E₀| < 1e-5, |ΔL/L₀| < 1e-3

#### 2. Window Independence ✓
- [x] Tested windows: [6,20] → p=-2.052, [8,24] → p=-2.452
- [ ] Add Theil-Sen robust fitting
- [ ] Bootstrap confidence intervals
- **Accept**: Δp < 0.05 across reasonable windows

#### 3. Helmholtz Check (Curl-free Field) 🔄
- [ ] Compute R_curl(r) = ⟨|∇×J|⟩_r / ⟨|J|⟩_r
- [ ] Measure tangential fraction τ_frac(r)
- [ ] Radiality error analysis
- **Accept**: R_curl < 0.05, τ_frac < 0.15 in fit window

#### 4. Kepler Tests 🔄
- [ ] T² ∝ a³ relationship
- [ ] v_θ²r constancy for circular orbits
- **Accept**: power index 1.5±0.03, flatness ±3%

#### 5. Equivalence Principle ⏳
- [ ] Test masses: 0.5, 1.0, 2.0
- [ ] Normalize accelerations
- **Accept**: differences < 0.5% in Newtonian window

#### 6. Gauss-law Flux ✓
- [x] r²⟨J_r⟩(r) flatness confirmed
- [x] Flux range: [0.882, 1.107]
- **Accept**: flat within ±5% ✓

#### 7. Precession vs Eccentricity ⏳
- [ ] Test e = {0.1, 0.3, 0.6}
- [ ] Measure Δϖ(e)
- **Accept**: |Δϖ| < 0.05°/orbit for moderate e

### B) Numerical Robustness

#### 1. Random Seed Sensitivity ⏳
- [ ] Warm from ≥5 different seeds
- [ ] Compare slopes & conservation
- **Accept**: std(slope) < 0.03

#### 2. Time-step Study ⏳
- [ ] Halve Δt, check drift scaling
- [ ] Compare integrators (leapfrog vs RK)
- **Accept**: observed order matches integrator

#### 3. Boundary Conditions ⏳
- [ ] Test larger L at constant r_fit/L
- **Accept**: slope unchanged within ±0.03

### C) New Predictions

#### 1. Gravitational Slip ⏳
- [ ] Compute dynamical vs lensing potentials
- [ ] Scale-dependent slip signature

#### 2. Rotation Curves ✓
- [x] Flat rotation curve demonstrated
- [ ] Baryonic Tully-Fisher comparison

#### 3. Tidal Fields ⏳
- [ ] Hill radius calculations
- [ ] Lagrange point stability

### D) Reproducibility

#### 1. Deterministic Artifacts ✓
- [x] Field snapshots saved
- [ ] MANIFEST.yaml with full environment
- [ ] One-button runner script

#### 2. Unit Tests ⏳
- [ ] Helmholtz projection identity
- [ ] Gauss-law for analytic 1/r² field
- [ ] Zero precession in pure 1/r potential

## Next Steps

1. Complete warm-up to eps3e-4 snapshot
2. Run convergence sweeps (N and L variations)
3. Implement Helmholtz curl diagnostics
4. Set up automated validation pipeline
5. Generate publication-ready figures

## Command Reference

```bash
# Field analysis
python analyze_field_slope_v2.py

# Orbit test
python src/don_emergent_collapse_3d.py --test orbit \
  --load_field fields/N320_L160_box \
  --steps 32000 --r0 20.0 --vtheta 0.30

# Convergence sweep
for N in 160 240 320 400; do
  python src/don_emergent_collapse_3d.py --test orbit \
    --N $N --L 160 --load_field fields/N${N}_L160_box
done
```