# Reproducing repository checks (honest minimal workflow)

## 1) Environment
```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Minimal verification (clean checkout)
```bash
python -m pytest -q
make wl-finalize
python -m pytest -q tests/test_wl_sign.py
```

## 3) Expected WL artifacts from `make wl-finalize`
- `outputs/CLENS_release/band_means.csv`
- `outputs/CLENS_release/gt_panel.png`
- `outputs/CLENS_release/slip_analysis.csv`
- `outputs/CLENS_release/slip_analysis_final.csv` (compatibility)
- `outputs/CLENS_release/slip_summary.json`
- `outputs/CLENS_release/slip_final_summary.json` (compatibility)
- `outputs/CLENS_release/sign_validation.json`
- `outputs/CLENS_release/slip_gate_referee.json`

## Notes
- WL scripts first use `outputs/CLENS_patch/*`; if absent they fall back to included sample profiles in `results/clens_kids_W1_2025-09-14/`.
- Convergence sweeps (`conv-grid`, `conv-box`) require generating `sweeps/*` and can be computationally heavy.
