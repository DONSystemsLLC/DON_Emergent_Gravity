#!/bin/bash
# Precession tests with varying eccentricity
echo "=== Running Precession Sweeps ==="

for v in 0.26 0.28 0.30 0.32; do
  echo "Starting precession scan v=$v..."
  .venv/bin/python -u scripts/orbit_metrics_sweep.py \
    --vtheta $v --out_dir sweeps/precession_scan_v$v \
    --N 320 --L 160 --dt 0.0015625 --steps 256000 \
    --r0 20.0 --masses 80,80,80,1.0 \
    --orbit_from_potential --project_curl_free \
    --load_field fields/N320_L160_box
done

echo "Parsing logs..."
.venv/bin/python -u parse_orbit_logs.py sweeps/precession_scan_v*

echo "Analyzing precession..."
.venv/bin/python -u scripts/precession_from_logs.py

echo "Done!"
