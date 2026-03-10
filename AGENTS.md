# DON Emergent Gravity Repository Guide

## Truthfulness & reproducibility policy
- Do **not** fabricate artifacts, data products, or acceptance outcomes.
- If an artifact cannot be regenerated from files included in this checkout, document that explicitly.
- Keep prototype outputs labeled as prototype; do not describe them as validated release evidence.

## Project layout
- **P1 (emergent kernel + orbits):** `experiments/P1_Emergent_Kernel/`, `experiments/P1_Kepler_Orbits/`
- **P2 (weak lensing):** `experiments/P2_Weak_Lensing/`
- **P3 (BH prediction prototype):** `experiments/BH_Prediction/`

## Environment setup
```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Canonical verification commands
```bash
python -m pytest -q
make wl-finalize
```

### Minimal verification suite (fastest honest check)
```bash
python -m pytest -q tests/test_wl_sign.py tests/test_physics.py::TestP2SlipAnalysis
make wl-finalize
python -m pytest -q tests/test_wl_sign.py
```

## Expected artifact locations after successful `make wl-finalize`
- `outputs/CLENS_release/band_means.csv`
- `outputs/CLENS_release/gt_panel.png`
- `outputs/CLENS_release/slip_analysis.csv`
- `outputs/CLENS_release/slip_analysis_final.csv` (compatibility name)
- `outputs/CLENS_release/slip_summary.json`
- `outputs/CLENS_release/slip_final_summary.json` (compatibility name)
- `outputs/CLENS_release/sign_validation.json`
- `outputs/CLENS_release/slip_gate_referee.json`

## Notes on included data
- WL finalize uses included profiles when available under `outputs/CLENS_patch/`; if absent, scripts fall back to `results/clens_kids_W1_2025-09-14/`.
