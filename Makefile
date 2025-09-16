VENV := .venv/bin/python

.PHONY: wl-run wl-monitor wl-finalize wl-check wl-plot orb-run orb-finalize orb-smoke orb-smoke-hot orb-parse

wl-run:
	$(VENV) -u scripts/clens_random_ensemble.py --patch_dir working/CLENS_patch --out_prefix working/CLENS_random_strat3 --n_ensembles 200 --seed 42 --stack_with scripts/clens_stack.py --zbuf 0.2 --rmin 0.2 --rmax 2.0 --nbin 12 --resume

wl-monitor:
	while true; do date; $(VENV) -u monitor_clens.py; sleep 20; done

wl-finalize:
	$(VENV) -u scripts/clens_finalize_results.py && \
	$(VENV) -u scripts/clens_plots.py && \
	$(VENV) -u scripts/clens_slip_analysis.py
	@echo "[wl-finalize] P2 CLENS slip prototype complete"

wl-check:
	$(VENV) -u monitor_clens.py

wl-plot:
	$(VENV) -u scripts/clens_plots.py && \
	ls -la proofs/CLENS_KiDS_W1_Stacking/*png

orb-run:
	@set -e; \
	LF=""; \
	if [ -n "$$LOAD_FIELD" ]; then LF="--load_field $$LOAD_FIELD"; fi; \
	$(VENV) -u scripts/orbit_metrics_sweep.py \
		--vtheta 0.25,0.28,0.30,0.32,0.34 \
		--out_dir sweeps/N320_L160_vtheta_sweep \
		--N 320 --L 160 --sigma 0.4 --tau 0.20 \
		--dt 0.003125 --steps 128000 --warm 0 \
		--r0 12.0 --masses 80,80,80,1.0 --orbit_from_potential \
		--project_curl_free \
		--timeout 7200 --save_logs $$LF

orb-smoke:
	$(VENV) -u scripts/orbit_metrics_sweep.py \
		--vtheta 0.28,0.30 \
		--out_dir sweeps/smoke_vtheta_sweep \
		--N 64 --L 32 --sigma 0.4 --tau 0.20 \
		--dt 0.003125 --steps 2000 --warm 0 \
		--r0 20.0 --masses 80,80,80,1.0 --orbit_from_potential \
		--timeout 600

orb-smoke-hot:
	$(VENV) -u scripts/orbit_metrics_sweep.py \
		--vtheta 0.30 \
		--out_dir sweeps/smoke_hot \
		--N 320 --L 160 --sigma 0.4 --tau 0.20 \
		--dt 0.003125 --steps 20000 --warm 0 \
		--r0 20.0 --masses 80,80,80,1.0 --orbit_from_potential \
		--timeout 1800 --save_logs \
		--load_field fields/N320_L160_box

orb-parse:
	$(VENV) -u parse_orbit_logs.py sweeps/N320_L160_vtheta_sweep && \
	cat sweeps/N320_L160_vtheta_sweep/summary.csv

orb-finalize:
	$(VENV) -u scripts/orbit_finalize_results.py --summary sweeps/N320_L160_vtheta_sweep/summary.csv

orb-monitor:
	@awk 'NR>1{c++}END{print "[orbit rows in summary] " (c+0)}' sweeps/N320_L160_vtheta_sweep/summary.csv 2>/dev/null || echo "[orbit rows in summary] 0"
	@ls -t sweeps/N320_L160_vtheta_sweep/log_vtheta_*.out 2>/dev/null | head -1 | xargs tail -n 20 || true

# ==== DON Emergent Gravity (P1) Proof Packaging ====

PROOF_DIR := proofs/EMERGENT_GRAVITY_ORBITS_N320_L160
SNAP      := fields/N320_L160_box
ORBIT_OUT := outputs/orbits/N320_L160_box_best

.PHONY: proof-p1 proof-p1-quick proof-verify proof-clean

proof-p1:
	@echo ">> Recomputing field kernel (log–log slope) ..."
	$(VENV) analyze_field_slope_v2.py
	@echo ">> Building rotation curve from slope profile ..."
	$(VENV) plot_rotation_curve.py
	@echo ">> Ensuring proof directory exists ..."
	mkdir -p $(PROOF_DIR)
	@echo ">> Assembling artifacts ..."
	# core figures & data
	cp -f field_slope_warm.png $(PROOF_DIR)/ || true
	cp -f $(SNAP)_slope_profile.csv $(PROOF_DIR)/N320_L160_box_slope_profile.csv || true
	cp -f rotation_curve.png $(PROOF_DIR)/ || true
	# orbit conservation figures
	cp -f energy_emergent_3d.png $(PROOF_DIR)/ || true
	cp -f angmom_emergent_3d.png $(PROOF_DIR)/ || true
	cp -f orbit_emergent_3d.png $(PROOF_DIR)/ || true
	# docs (already authored)
	cp -f $(PROOF_DIR)/FIGURE_CAPTIONS.md $(PROOF_DIR)/ || true
	cp -f $(PROOF_DIR)/METHODS_AND_RESULTS_P1.md $(PROOF_DIR)/ || true
	# lock & checksums
	pip freeze > $(PROOF_DIR)/requirements.lock.txt
	( cd $(PROOF_DIR) && \
	  rm -f SHA256SUMS.txt && \
	  find . -maxdepth 1 -type f ! -name 'SHA256SUMS.txt' -print0 \
	    | xargs -0 shasum -a 256 > SHA256SUMS.txt )
	@echo ">> P1 proof bundle refreshed at: $(PROOF_DIR)"
	@echo ">> Done."

# Quick path: just refresh checksums & lock without recomputing plots
proof-p1-quick:
	mkdir -p $(PROOF_DIR)
	pip freeze > $(PROOF_DIR)/requirements.lock.txt
	( cd $(PROOF_DIR) && \
	  rm -f SHA256SUMS.txt && \
	  find . -maxdepth 1 -type f ! -name 'SHA256SUMS.txt' -print0 \
	    | xargs -0 shasum -a 256 > SHA256SUMS.txt )
	@echo ">> Quick refresh complete."

# Verify recorded checksums
proof-verify:
	@echo ">> Verifying SHA256SUMS.txt ..."
	@cd $(PROOF_DIR) && shasum -a 256 -c SHA256SUMS.txt || \
	 (echo "!! checksum mismatch (expected when images/regenerated)"; exit 1)
	@echo ">> Checksums verified."

# Clean only transient analysis artifacts (keeps proof bundle)
proof-clean:
	@rm -f *_slope_profile.csv || true
	@rm -f field_slope_warm.png field_slope_3d.png || true
	@rm -f rotation_curve.png || true
	@echo ">> Cleaned transient analysis outputs."

# ========================
# P1 Validation Shortcuts
# ========================
.PHONY: helmholtz kepler kepler-runs kepler-fit ep ep-runs ep-fit gate-strict p1-accept

# --- User-tunable defaults ---
PY               ?= .venv/bin/python
L                ?= 160
N                ?= 320
DT_STRICT        ?= 0.0015625
STEPS_STRICT     ?= 256000
R0               ?= 20.0
KEPLER_V         ?= 0.26 0.28 0.30 0.32 0.34
EP_M             ?= 0.5 1.0 2.0
KEPLER_MIN_GAP   ?= 50

# Accept: set to your latest field snapshot
FIELD_NPZ        ?= fields/N320_L160_box_eps3e-4.npz

# Annulus used for slope/flux diagnostics (curl-free window)
ANNULUS          ?= 12,32
space :=
space +=
comma := ,
ANN_RMIN         := $(word 1,$(subst $(comma),$(space),$(ANNULUS)))
ANN_RMAX         := $(word 2,$(subst $(comma),$(space),$(ANNULUS)))

# Profiles & summaries used by the acceptance gate
FORCE_PROFILE    ?= proofs/EMERGENT_GRAVITY_ORBITS_N320_L160/validation_force_profile.csv
SLOPE_COLUMN     ?= Fr_med
FLUX_PROFILE     ?= proofs/EMERGENT_GRAVITY_ORBITS_N320_L160/N320_L160_box_slope_profile.csv
STRICT_SUMMARY   ?= sweeps/validation_orbit_strict/summary.csv

# Flux tolerance (p95). Default 20% for now; tighten to 5% when you're ready.
GATE_FLUX_TOL    ?= 0.05

# -------------------
# P2: Convergence Sweeps
# -------------------
conv-grid:
	@echo "[conv-grid] Testing N convergence (fixed L=$(L))"
	@for n in 160 240 320 400; do \
	  echo "[conv-grid] N=$$n..."; \
	  $(PY) -u scripts/orbit_metrics_sweep.py \
	    --vtheta 0.30 --out_dir sweeps/conv_grid_N$$n \
	    --N $$n --L $(L) --dt $(DT_STRICT) --steps $(STEPS_STRICT) --warm 0 \
	    --r0 $(R0) --masses 80,80,80,1.0 \
	    --orbit_from_potential --project_curl_free --save_logs \
	    --load_field $(LOAD_FIELD_PROFILE) --timeout 3600; \
	done
	$(PY) -u scripts/convergence_grid_summary.py --grid_type N --out_csv figs/convergence_grid.csv

conv-box:
	@echo "[conv-box] Testing L convergence (fixed N=$(N))"
	@for l in 120 160 200; do \
	  echo "[conv-box] L=$$l..."; \
	  $(PY) -u scripts/orbit_metrics_sweep.py \
	    --vtheta 0.30 --out_dir sweeps/conv_box_L$$l \
	    --N $(N) --L $$l --dt $(DT_STRICT) --steps $(STEPS_STRICT) --warm 0 \
	    --r0 $(R0) --masses 80,80,80,1.0 \
	    --orbit_from_potential --project_curl_free --save_logs \
	    --load_field $(LOAD_FIELD_PROFILE) --timeout 3600; \
	done
	$(PY) -u scripts/convergence_box_summary.py --grid_type L --out_csv figs/convergence_box.csv

# -------------------
# P2: Window Independence Heatmap
# -------------------
slope-window:
	$(PY) -u scripts/slope_window_heatmap.py \
	  --field_npz $(FIELD_NPZ) --L $(L) \
	  --out_png figs/slope_window_heatmap.png \
	  --out_json figs/slope_window_heatmap.json

# -------------------
# Helmholtz diagnostics (R_curl, tau_frac, radiality error)
# -------------------
helmholtz:
	$(PY) -u scripts/helmholtz_diagnostics.py \
	  --field_npz $(FIELD_NPZ) \
	  --L $(L) --rmin $(ANN_RMIN) --rmax $(ANN_RMAX) \
	  --out_csv figs/helmholtz_checks.csv \
	  --out_json figs/helmholtz_checks.json

# -------------------
# Kepler: run strict orbits and fit T^2 ~ a^3
# -------------------
kepler-runs:
	@for v in $(KEPLER_V); do \
	  echo "[kepler] vtheta=$$v"; \
	  $(PY) -u scripts/orbit_metrics_sweep.py \
	    --vtheta $$v --out_dir sweeps/kepler_v$$v \
	    --N $(N) --L $(L) --dt $(DT_STRICT) --steps $(STEPS_STRICT) --warm 0 \
	    --r0 $(R0) --masses 80,80,80,1.0 \
	    --orbit_from_potential --project_curl_free --save_logs ; \
	done
	$(PY) -u parse_orbit_logs.py sweeps/kepler_v*

kepler-fit:
	$(PY) -u scripts/kepler_periods_from_logs.py \
	  --glob 'sweeps/kepler_v*/logs/*orbit_log.csv' \
	  --L $(L) --min_gap $(KEPLER_MIN_GAP)

kepler: kepler-runs kepler-fit

# -------------------
# Equivalence Principle (m = 0.5, 1.0, 2.0) + report spread
# -------------------
ep-runs:
	@for m in $(EP_M); do \
	  echo "[ep] m=$$m"; \
	  $(PY) -u scripts/orbit_metrics_sweep.py \
	    --vtheta 0.30 --out_dir sweeps/ep_m$$m \
	    --N $(N) --L $(L) --dt $(DT_STRICT) --steps $(STEPS_STRICT) --warm 0 \
	    --r0 $(R0) --masses 80,80,80,$$m \
	    --orbit_from_potential --project_curl_free --save_logs ; \
	done
	$(PY) -u parse_orbit_logs.py sweeps/ep_m*

ep-fit:
	$(PY) -u scripts/ep_mass_independence_from_summaries.py \
	  --glob 'sweeps/ep_m*/summary.csv'

ep: ep-runs ep-fit

# -------------------
# One-line acceptance gate (strict)
# -------------------
gate-strict:
	$(PY) -u scripts/acceptance_gate.py \
	  --summary_csv $(STRICT_SUMMARY) \
	  --slope_csv $(FORCE_PROFILE) \
	  --slope_column $(SLOPE_COLUMN) \
	  --slope_target -2.0 --slope_tol 0.05 \
	  --flux_csv  $(FLUX_PROFILE) \
	  --flux_rmin $(ANN_RMIN) --flux_rmax $(ANN_RMAX) \
	  --flux_tol $(GATE_FLUX_TOL) --flux_use_median 1 \
	  --max_prec 0.05

# Convenience alias: rebuild proof, run helmholtz, kepler fit, EP fit, then gate
p1-accept: helmholtz kepler-fit ep-fit gate-strict
	@echo ">> P1 acceptance sequence complete."

# ========================
# Methods Table Profiling
# ========================
.PHONY: profile-one profile-grid methods-table appendix-md appendix-tex appendix report report-md

PY                ?= .venv/bin/python
N                 ?= 320
L                 ?= 160
DT_STRICT         ?= 0.0015625
STEPS_STRICT      ?= 256000
R0                ?= 20.0
VTHETA_PROFILE    ?= 0.30
PROFILE_OUTDIR    ?= profile_runs
LOAD_FIELD_PROFILE ?= fields/N320_L160_box

# Cross-platform time command: prefer GNU gtime -v, else BSD /usr/bin/time -l
TIME_CMD ?= $(shell (command -v gtime >/dev/null 2>&1 && echo gtime -v) || echo /usr/bin/time -l)

N_LIST            ?= 160 320
L_LIST            ?= 120 160
DT_LIST           ?= $(DT_STRICT)

export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1

profile-one:
	@mkdir -p $(PROFILE_OUTDIR)
	@echo "[profile] N=$(N) L=$(L) dt=$(DT_STRICT) steps=$(STEPS_STRICT) vtheta=$(VTHETA_PROFILE)"
	@{ $(TIME_CMD) sh -c '\
	  $(PY) -u scripts/orbit_metrics_sweep.py \
	    --vtheta $(VTHETA_PROFILE) --out_dir $(PROFILE_OUTDIR)/N$(N)_L$(L)_dt$(DT_STRICT) \
	    --N $(N) --L $(L) --dt $(DT_STRICT) --steps $(STEPS_STRICT) --warm 0 \
	    --r0 $(R0) --masses 80,80,80,1.0 \
	    --orbit_from_potential --project_curl_free --save_logs --load_field $(LOAD_FIELD_PROFILE) \
	' 2> $(PROFILE_OUTDIR)/N$(N)_L$(L)_dt$(DT_STRICT).timelog; \
	  status=$$?; \
	  if [ $$status -ne 0 ]; then \
	    if [ $$status -eq 1 ] && grep -qi "Operation not permitted" $(PROFILE_OUTDIR)/N$(N)_L$(L)_dt$(DT_STRICT).timelog; then \
	      echo "[profile] warning: resource usage info limited (time exit $$status)"; \
	    else \
	      exit $$status; \
	    fi; \
	  fi; \
	}
	@$(PY) -c "import json, sys; json.dump({'N': int($(N)), 'L': float($(L)), 'dt': float($(DT_STRICT)), 'steps': int($(STEPS_STRICT)), 'vtheta': float($(VTHETA_PROFILE)), 'OMP_NUM_THREADS': int($(OMP_NUM_THREADS)), 'MKL_NUM_THREADS': int($(MKL_NUM_THREADS))}, sys.stdout, indent=2)" > $(PROFILE_OUTDIR)/N$(N)_L$(L)_dt$(DT_STRICT).meta.json
	@echo "[profile] wrote $(PROFILE_OUTDIR)/N$(N)_L$(L)_dt$(DT_STRICT).timelog (+ .meta.json)"

profile-grid:
	@mkdir -p $(PROFILE_OUTDIR)
	@for n in $(N_LIST); do \
	  for l in $(L_LIST); do \
	    for dt in $(DT_LIST); do \
	      echo "[profile] N=$$n L=$$l dt=$$dt steps=$(STEPS_STRICT)"; \
	      N=$$n L=$$l DT_STRICT=$$dt $(MAKE) --no-print-directory profile-one; \
	    done; \
	  done; \
	done

methods-table:
	$(PY) -u scripts/methods_table_from_profiles.py \
	  --glob "$(PROFILE_OUTDIR)/*.timelog" \
	  --out figs/methods_table.csv
	@echo ">> wrote figs/methods_table.csv"

appendix-md:
	$(PY) -u scripts/appendix_methods_table_md.py \
	  --methods_csv figs/methods_table.csv \
	  --out_md docs/Appendix_Methods_Table.md \
	  --title "Appendix A — Numerical Performance and Resources" \
	  --sort --add_summary

appendix-tex:
	$(PY) -u scripts/appendix_methods_table_tex.py \
	  --methods_csv figs/methods_table.csv \
	  --out_tex docs/Appendix_Methods_Table.tex \
	  --title "Appendix A — Numerical Performance and Resources" \
	  --label "tab:methods" --sort --median

appendix: methods-table appendix-md appendix-tex
	@echo ">> Appendix (MD + TeX) generated"

report: methods-table helmholtz kepler-fit ep-fit
	$(PY) -u scripts/build_report.py --out REPORT.md
	@echo ">> Validation report written to REPORT.md"

report-md: methods-table helmholtz kepler-fit ep-fit
	$(PY) -u scripts/build_report_md.py \
	  --title "DON Emergent Gravity — Validation Report (P1)" \
	  --accept ACCEPTANCE_BOX.md \
	  --helm_json figs/helmholtz_checks.json \
	  --kepler_csv figs/kepler_fit.csv \
	  --ep_glob 'sweeps/ep_m*/summary.csv' \
	  --methods_csv figs/methods_table.csv \
	  --out docs/REPORT.md
	@echo ">> wrote docs/REPORT.md"

# -------------------
# P2: Complete Package & Release
# -------------------
paper-kit: conv-grid conv-box slope-window helmholtz kepler-fit ep-fit wl-finalize
	@echo "[paper-kit] Building P2 complete validation package..."
	$(PY) -u scripts/build_report_p2.py \
	  --title "DON Emergent Gravity — P2 Universality & Slip" \
	  --conv_grid_csv figs/convergence_grid.csv \
	  --conv_box_csv figs/convergence_box.csv \
	  --slope_window_json figs/slope_window_heatmap.json \
	  --helm_json figs/helmholtz_checks.json \
	  --kepler_csv figs/kepler_fit.csv \
	  --ep_glob 'sweeps/ep_m*/summary.csv' \
	  --slip_csv outputs/CLENS_release/slip_analysis.csv \
	  --methods_csv figs/methods_table.csv \
	  --out docs/REPORT.md
	$(PY) -u scripts/build_acceptance_box_p2.py \
	  --out docs/ACCEPTANCE_BOX_P2.md
	@echo ">> P2 paper kit complete: docs/REPORT.md + docs/ACCEPTANCE_BOX_P2.md"
