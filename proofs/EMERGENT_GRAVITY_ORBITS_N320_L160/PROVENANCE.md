# P1 Provenance — Slope & Metrics

**Definitions**
- **PHYSICS slope:** *log–log* fit of shell-averaged radial flux ⟨|J_r|⟩ vs r, with μ-barycenter + minimum-image, interior windows.
- **DIAGNOSTIC slope:** sweep-internal or *log–linear* fits (e.g., log|F_r| vs r); not used for physics claims.
- **Flux flatness:** 16–84% range of \(F(r) ∝ ⟨|J_r|⟩ r^2\) normalized by median (flat ≈ evidence for 1/r²).
- **Conservation metrics:** mean |ΔE/E₀|, |Δ‖L‖/‖L‖₀| from production orbit; precession = periapse-to-periapse.

Only rows marked **PHYSICS (VERIFIED)** support the emergent 1/r² claim.


| ID                       | Snapshot                         | Method                                             | Window        | p_slope      | Flux_flat_16–84% | dE_over_E   | dL_over_L   | Precession_deg/orbit | Class              | Source                          |
| ------------------------ | -------------------------------- | -------------------------------------------------- | ------------- | ------------ | ---------------- | ----------- | ----------- | -------------------- | ------------------ | ------------------------------- |
| P1-FIELD-6-20            | fields/N320_L160_box             | Field kernel, log–log ⟨|J_r|⟩ vs r (COM+min-image) | [6,20]        | -2.085       | [0.86,1.09]      | 4.570e-04   | 5.440e-02   | —                    | PHYSICS (VERIFIED) | analyze_field_slope_v2.py       |
| P1-FIELD-8-24            | fields/N320_L160_box             | Field kernel, log–log ⟨|J_r|⟩ vs r (COM+min-image) | [8,24]        | -1.973       | [0.87,1.09]      | 4.570e-04   | 5.440e-02   | —                    | PHYSICS (VERIFIED) | analyze_field_slope_v2.py       |
| EARLY-NAIVE-2.26         | (earlier warm)                   | |J| at grid center (no COM/min-image)              | —             | -2.26        | —                | —           | —           | —                    | SUPERSEDED         | exploratory                     |
| SWEEP-LOGLIN-~−7…−10     | fields/N320_L160_box             | log–linear |F_r| vs r (sweep diagnostic)           | (scan)        | ~−7..−10     | —                | —           | —           | —                    | DIAGNOSTIC         | orbit_metrics_sweep.py          |
| SWEEP-LOGLIN-−21.238     | production orbit                 | log–linear |F_r| vs r (diagnostic)                 | (scan)        | −21.238      | —                | —           | —           | —                    | DIAGNOSTIC         | orbit.out                       |
| LATEST-UNDERWARMED-−1.56 | (short warm/edge)                | Field kernel with biased window                    | (edge/core)   | −1.56        | —                | —           | —           | —                    | INVALID (bias)     | intermediate                    |
| EPS-SNAPSHOT-PENDING     | fields/N320_L160_box_eps5e-4.npz | Field kernel, log–log ⟨|J_r|⟩ vs r (COM+min-image) | [6,20]/[8,24] | (to compute) | (to compute)     | (after run) | (after run) | (after run)          | PLANNED            | analyze_field_slope_v2.py (NPZ) |
