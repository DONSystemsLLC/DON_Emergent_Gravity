VENV := .venv/bin/python

.PHONY: wl-run wl-monitor wl-finalize wl-check wl-plot orb-run orb-finalize orb-smoke orb-smoke-hot orb-parse

wl-run:
	$(VENV) -u scripts/clens_random_ensemble.py --patch_dir working/CLENS_patch --out_prefix working/CLENS_random_strat3 --n_ensembles 200 --seed 42 --stack_with scripts/clens_stack.py --zbuf 0.2 --rmin 0.2 --rmax 2.0 --nbin 12 --resume

wl-monitor:
	while true; do date; $(VENV) -u monitor_clens.py; sleep 20; done

wl-finalize:
	$(VENV) -u scripts/clens_finalize_results.py

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