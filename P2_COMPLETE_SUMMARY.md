# P2 status summary (updated for reproducibility)

P2 weak-lensing tooling is available, but prior claims of complete referee-ready regeneration from any clean checkout were overstated.

## What is now consistent
- `wl-finalize` and `wl-gate` target real script paths under `experiments/P2_Weak_Lensing/scripts/`.
- WL finalize writes both canonical and compatibility output names used by tests/docs:
  - `slip_analysis.csv` and `slip_analysis_final.csv`
  - `slip_summary.json` and `slip_final_summary.json`
- Sign validation and referee gate outputs are produced in `outputs/CLENS_release/`.

## What remains optional/heavy
- Full convergence artifacts (`figs/convergence_grid.csv`, `figs/convergence_box.csv`) require running sweeps.
- Some historical release claims may describe archived/previous runs rather than regenerated outputs.

For reviewer steps, use `docs/reproduce.md`.
