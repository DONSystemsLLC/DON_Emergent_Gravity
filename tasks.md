# Tasks: DON P2 — Universality & Slip

**Input**: Current implementation status and Makefile targets
**Context**: DON Emergent Gravity P2 validation suite with convergence, window-independence, Helmholtz+flux diagnostics, Kepler, EP, and CI automation

## Execution Flow (main)
```
1. Complete remaining CLENS slip prototype implementation
2. Implement report generation and bundle scripts
3. Set up CI/CD pipeline with pytest and gate-strict validation
4. Generate comprehensive P2 acceptance documentation
5. Update MANIFEST with P2 results and checksums
```

## Current Status
- [x] Convergence sweeps (N,L) - Makefile targets implemented
- [x] Window-independence heatmap - Enhanced script created
- [x] Helmholtz + flux diagnostics - Enhanced with P2 criteria
- [x] Kepler + EP validation - Scripts enhanced
- [-] Slip prototype (KiDS W1) - Partial implementation
- [ ] Report bundle generation - Needs completion
- [ ] CI & MANIFEST - Not implemented

## Phase 3.1: Complete Slip Prototype (KiDS W1)
- [ ] T001 [P] Enhance scripts/clens_finalize_results.py to output band_means.csv in outputs/CLENS_release/
- [ ] T002 [P] Update scripts/clens_plots.py to generate gt_panel.png with proper legends and band highlighting
- [ ] T003 Create scripts/clens_slip_analysis.py for lensing–dynamics comparison stub
- [ ] T004 [P] Update Makefile wl-finalize target to call enhanced scripts

## Phase 3.2: Report Bundle Generation
- [ ] T005 Create scripts/build_report_p2.py for comprehensive P2 report with YAML front-matter
- [ ] T006 Create scripts/build_acceptance_box_p2.py for docs/ACCEPTANCE_BOX_P2.md
- [ ] T007 [P] Update paper-kit Makefile target to generate complete P2 documentation bundle
- [ ] T008 [P] Create scripts/update_manifest.py for proofs/EMERGENT_GRAVITY_ORBITS_N320_L160/MANIFEST.yaml

## Phase 3.3: CI & Testing Infrastructure
- [ ] T009 Create tests/test_physics.py with P2 validation unit tests
- [ ] T010 Create .github/workflows/validate.yml for pytest + make gate-strict on PR
- [ ] T011 [P] Add CI badge and status to README.md
- [ ] T012 [P] Create tests/test_makefile_targets.py to validate all make targets work

## Phase 3.4: Integration & Validation
- [ ] T013 Run make conv-grid and validate convergence results in figs/convergence_grid.csv
- [ ] T014 Run make conv-box and validate convergence results in figs/convergence_box.csv  
- [ ] T015 Run make slope-window and validate figs/slope_window_heatmap.json acceptance criteria
- [ ] T016 Run make helmholtz with tightened flux tolerance and validate P2 criteria
- [ ] T017 Run make kepler-fit and validate Kepler slope 1.50±0.03 in figs/kepler_fit.csv
- [ ] T018 Run make ep-fit and validate EP spread ≤0.5% in figs/ep_analysis.json
- [ ] T019 Run make gate-strict with updated P2 criteria and verify all gates pass

## Phase 3.5: Final Documentation & Release
- [ ] T020 [P] Generate docs/REPORT.md with make paper-kit including all P2 results
- [ ] T021 [P] Create comprehensive docs/ACCEPTANCE_BOX_P2.md with all metrics
- [ ] T022 [P] Update proofs/EMERGENT_GRAVITY_ORBITS_N320_L160/MANIFEST.yaml with P2 checksums
- [ ] T023 Run full validation: make paper-kit && make gate-strict
- [ ] T024 [P] Commit all artifacts: PNG/CSV/JSON/MD/TEX files only (ignore heavy dirs)

## Task Specifications

### T001: Enhance CLENS finalize script
**File**: scripts/clens_finalize_results.py
**Command**: Enhance to output structured band_means.csv with columns: set, gt_mean_0p5_1p8, gt_err, N_src_mean
**Output**: outputs/CLENS_release/band_means.csv

### T002: Update CLENS plots
**File**: scripts/clens_plots.py  
**Command**: Add gt_panel.png generation with proper legends, band highlighting [0.5,1.8] Mpc
**Output**: outputs/CLENS_release/gt_panel.png

### T003: Create slip analysis stub
**File**: scripts/clens_slip_analysis.py
**Command**: New script for lensing–dynamics comparison prototype
**Output**: Structured comparison between lensing signal and emergent gravity predictions

### T005: Create P2 report builder
**File**: scripts/build_report_p2.py
**Command**: Generate comprehensive P2 report with YAML front-matter, all validation results
**Output**: docs/REPORT.md with P2 validation summary

### T006: Create P2 acceptance box
**File**: scripts/build_acceptance_box_p2.py
**Command**: Generate P2 acceptance criteria document with pass/fail status
**Output**: docs/ACCEPTANCE_BOX_P2.md

### T009: Create physics tests
**File**: tests/test_physics.py
**Command**: Unit tests for P2 validation criteria: convergence, window-independence, Helmholtz, Kepler, EP
**Validation**: pytest -q tests/test_physics.py

### T010: Create CI workflow
**File**: .github/workflows/validate.yml
**Command**: GitHub Actions workflow running pytest -q && make gate-strict on PR
**Triggers**: Pull request, push to main

### T012: Create Makefile tests
**File**: tests/test_makefile_targets.py
**Command**: Test all make targets return exit code 0
**Validation**: Test conv-grid, conv-box, slope-window, helmholtz, kepler-fit, ep-fit, gate-strict, paper-kit

## Dependencies
- T001, T002, T003 → T004 (Makefile update)
- T005, T006 → T007 (paper-kit target)
- T009 → T010 (CI depends on tests)
- T013-T019 → T020-T022 (validation before docs)
- All tasks → T023, T024 (final validation and commit)

## Parallel Execution Groups

### Group A: CLENS Enhancement [P]
```bash
# T001, T002 can run in parallel (different files)
Task T001: scripts/clens_finalize_results.py
Task T002: scripts/clens_plots.py
```

### Group B: Report Scripts [P]  
```bash
# T005, T006, T008 can run in parallel (different files)
Task T005: scripts/build_report_p2.py
Task T006: scripts/build_acceptance_box_p2.py
Task T008: scripts/update_manifest.py
```

### Group C: Testing Infrastructure [P]
```bash
# T009, T011, T012 can run in parallel (different files)
Task T009: tests/test_physics.py
Task T011: README.md (add CI badge)
Task T012: tests/test_makefile_targets.py
```

### Group D: Final Documentation [P]
```bash
# T020, T021, T022 can run in parallel (different outputs)
Task T020: docs/REPORT.md
Task T021: docs/ACCEPTANCE_BOX_P2.md  
Task T022: proofs/EMERGENT_GRAVITY_ORBITS_N320_L160/MANIFEST.yaml
```

## Makefile Integration
All tasks integrate with existing Makefile targets:
- `make conv-grid` (T013): N convergence sweep
- `make conv-box` (T014): L convergence sweep  
- `make slope-window` (T015): Window-independence heatmap
- `make helmholtz` (T016): Helmholtz + flux diagnostics
- `make kepler-fit` (T017): Kepler T²∝a³ validation
- `make ep-fit` (T018): Equivalence principle validation
- `make gate-strict` (T019): Acceptance gate with P2 criteria
- `make paper-kit` (T020): Complete documentation bundle

## Validation Checklist
- [ ] All P2 acceptance criteria implemented and testable
- [ ] CI pipeline validates physics on every PR
- [ ] Documentation regenerates automatically with paper-kit
- [ ] All artifacts committed (PNG/CSV/JSON/MD/TEX only)
- [ ] MANIFEST.yaml updated with P2 checksums and metadata
- [ ] Gate-strict passes with tightened P2 tolerances

## Success Criteria
1. **One-button validation**: `make paper-kit && make gate-strict` completes successfully
2. **Reproducible results**: All P2 metrics within acceptance tolerances
3. **CI automation**: GitHub Actions validates physics on every PR
4. **Complete documentation**: P2 report with YAML front-matter and acceptance box
5. **Artifact management**: All results committed and checksummed in MANIFEST

## Notes
- Focus on completing the slip prototype (T001-T004) first
- CI setup (T009-T012) can proceed in parallel with report generation
- Final validation (T013-T019) must complete before documentation bundle
- All file paths are absolute from repository root
- [P] tasks operate on different files and can run in parallel
- Sequential tasks share files or have dependencies