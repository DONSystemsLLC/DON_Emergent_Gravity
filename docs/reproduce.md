# Reproducing the proofs

## P1 — Emergent kernel & orbits
1) Kernel slope (interior window):
```bash
python experiments/P1_Emergent_Kernel/scripts/analyze_field_slope.py \
  --params experiments/P1_Emergent_Kernel/params.yaml
```

2) Orbits & conservation:
```bash
python experiments/P1_Kepler_Orbits/scripts/analyze_orbit_sweep.py \
  --config experiments/P1_Kepler_Orbits/params.yaml
```

3) Hash the bundle (write `SHA256SUMS.txt` next to RESULTS):
```bash
python tools/hash_bundle.py
```

## P2 — CLENS weak-lensing (KiDS W1)
Run the e2e stack in `experiments/P2_Weak_Lensing/scripts/` and compare images and
tables in `proofs/CLENS_KiDS_W1_Stacking/`. Large inputs are referenced in
`experiments/P2_Weak_Lensing/docs/inputs/`.
