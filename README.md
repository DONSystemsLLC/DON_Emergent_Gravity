# DON Emergent Gravity — 3D Collapse Proof

**What this shows**
- Collapse-only μ,J transport (no Poisson, no 1/r² coded) yields **|J(r)| ∝ r⁻²** in 3D.
- A test particle under the DON readout orbits with small energy and angular-momentum drift.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
Reproduce the results
1) Slope (paper-quality)
bash
Copy code
python src/don_emergent_collapse_3d.py --test slope \
  --N 192 --L 80 --sigma 1.0 --tau 0.20 \
  --dt 0.05 --warm 8000 --rmin 5 --rmax 20 \
  --masses 32,32,32,1.0 --project_curl_free
Generates: results/proof_emergence_v1/field_slope_3d.png (|J| ∝ r⁻²).

2) Orbit (conservative readout)
bash
Copy code
python src/don_emergent_collapse_3d.py --test orbit \
  --N 192 --L 80 --sigma 1.0 --tau 0.20 \
  --dt 0.025 --warm 8000 --steps 16000 \
  --r0 10.0 --vtheta 0.32 \
  --masses 32,32,32,1.0 --orbit_from_potential
Generates:

results/proof_emergence_v1/orbit_emergent_3d.png (path)

results/proof_emergence_v1/energy_emergent_3d.png (energy drift)

results/proof_emergence_v1/angmom_emergent_3d.png (angular momentum drift)

Exact parameters: results/proof_emergence_v1/run_params.json.

Notes
The solver evolves collapse-only μ,J dynamics; no Poisson / 1/r² is hard-coded.

Optional flags:

--project_curl_free : one-shot Helmholtz projection of J for polished slope/fit.

--orbit_from_potential : use a = −∇(κ c² μ) for a conservative, central force orbit.

