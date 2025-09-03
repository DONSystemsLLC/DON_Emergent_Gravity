#!/usr/bin/env bash
set -euo pipefail
python src/don_emergent_collapse_3d.py --test slope \
  --N 192 --L 80 --sigma 1.0 --tau 0.20 \
  --dt 0.05 --warm 8000 --rmin 5 --rmax 20 \
  --masses 32,32,32,1.0 --project_curl_free
python src/don_emergent_collapse_3d.py --test orbit \
  --N 192 --L 80 --sigma 1.0 --tau 0.20 \
  --dt 0.025 --warm 8000 --steps 16000 \
  --r0 10.0 --vtheta 0.32 \
  --masses 32,32,32,1.0 --orbit_from_potential
mkdir -p results/proof_emergence_v1
cp field_slope_3d.png orbit_emergent_3d.png energy_emergent_3d.png angmom_emergent_3d.png results/proof_emergence_v1/ || true
