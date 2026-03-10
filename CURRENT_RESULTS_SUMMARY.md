# Current Results Summary (truthful status)

This repository contains a mix of:
- **Generated artifacts** reproducible from included scripts.
- **Reference/proof artifacts** committed from prior runs.
- **Optional heavy workflows** that require additional compute and are not run by default.

## Verified in this cleanup
- Pytest collection/import issues were fixed for `experiments/BH_Prediction/test_pipeline.py`.
- P2 Makefile WL targets now point to existing scripts.
- `make wl-finalize` is wired to generate current WL release outputs and compatibility filenames.

## Not guaranteed from a fast clean run
- Full convergence sweeps and all P1/P2 heavy artifacts (`sweeps/*`, `figs/convergence_*.csv`) unless those jobs are explicitly run.

Use `docs/reproduce.md` and `AGENTS.md` for the canonical verification commands.
