#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-sweeps/N320_L160_vtheta_sweep}"
echo "Monitoring orbit sweep in: $OUT_DIR (ctrl-c to exit)"
echo

while true; do
  clear
  date
  echo "== Active python pids =="
  pgrep -fl "orbit_metrics_sweep.py" || echo "(none)"
  echo
  echo "== Completed logs =="
  ls -1 $OUT_DIR/orbit_v*.log 2>/dev/null | wc -l | awk '{print "count:",$1}'
  echo
  last_log=$(ls -1t $OUT_DIR/orbit_v*.log 2>/dev/null | head -1 || true)
  if [[ -n "${last_log}" ]]; then
    echo "== Tail of ${last_log} =="
    tail -n 15 "${last_log}"
  else
    echo "(no logs yet)"
  fi
  sleep 60
done
