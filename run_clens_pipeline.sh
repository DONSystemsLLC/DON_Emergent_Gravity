#!/bin/bash
# Complete CLENS × KiDS W1 Pipeline
# Run this in screen/tmux for long-running stacking

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== CLENS × KiDS W1 Stacking Pipeline ==="
echo "Started: $(date)"
echo ""

# Configuration
PATCH_DIR=working/CLENS_patch
OUT_DIR=outputs/CLENS_patch
PROOF_DIR=proofs/CLENS_KiDS_W1_Stacking
RAND_BASE=working/CLENS_random_strat3
RAND_OUT=outputs
N_RANDS=${N_RANDS:-200}  # Adjust as needed

mkdir -p "$PATCH_DIR" "$OUT_DIR" "$PROOF_DIR"

# Step 1: Check if sources are built
if [ ! -f "$PATCH_DIR/sources.parquet" ]; then
    echo "Building sources from KiDS tiles..."
    python scripts/kids_tile_audit_merge.py \
        --tiles data/CLENS/kids_tiles \
        --out_dir "$PATCH_DIR"
else
    echo "✓ Sources already built: $PATCH_DIR/sources.parquet"
fi

# Step 2: Check if clusters are prepared
if [ ! -f "$PATCH_DIR/clusters.parquet" ]; then
    echo "Preparing clusters..."
    python -c "
import pandas as pd
cl = pd.read_csv('data/CLENS/clusters.csv')
cl = cl.rename(columns={'RAdeg': 'ra', 'DEdeg': 'dec', 'z': 'z_cl'})
cl.to_parquet('$PATCH_DIR/clusters.parquet', index=False)
print(f'Saved {len(cl)} clusters')
"
else
    echo "✓ Clusters already prepared"
fi

# Step 3: Stack lens clusters
if [ ! -f "$OUT_DIR/profile_focus.csv" ]; then
    echo "Stacking lens clusters (this may take hours)..."
    python scripts/clens_stack.py \
        --work "$PATCH_DIR" \
        --out "$OUT_DIR" \
        --z_focus 0.20,0.35 \
        --z_ctrl "0.14,0.19;0.36,0.49" \
        --zbuf 0.2 \
        --rmin 0.1 --rmax 2.0 --nbin 8
else
    echo "✓ Lens stack complete"
fi

# Step 4: Generate random ensembles
N_EXISTING=$(ls -d ${RAND_BASE}_* 2>/dev/null | wc -l | tr -d ' ')
if [ $N_EXISTING -lt $N_RANDS ]; then
    echo "Generating random ensembles ($N_EXISTING existing, need $N_RANDS)..."
    python scripts/clens_random_ensemble.py \
        --patch_dir "$PATCH_DIR" \
        --out_prefix "$RAND_BASE" \
        --n_ensembles "$N_RANDS" \
        --seed 42
else
    echo "✓ Random ensembles ready ($N_EXISTING)"
fi

# Step 5: Stack random realizations
echo "Checking random stacks..."
N_TODO=0
for d in ${RAND_BASE}_*; do
    OUT_PATH="$RAND_OUT/$(basename $d)/profile_focus.csv"
    if [ ! -f "$OUT_PATH" ]; then
        ((N_TODO++))
    fi
done

if [ $N_TODO -gt 0 ]; then
    echo "Stacking $N_TODO random realizations (this will take many hours)..."
    echo "Consider running in parallel batches if you have multiple cores."

    # Stack in batches to avoid overwhelming the system
    BATCH_SIZE=4  # Adjust based on your CPU cores
    for d in ${RAND_BASE}_*; do
        OUT_PATH="$RAND_OUT/$(basename $d)/profile_focus.csv"
        if [ ! -f "$OUT_PATH" ]; then
            echo "  Stacking $(basename $d)..."
            python scripts/clens_stack.py \
                --work "$d" \
                --out "$RAND_OUT/$(basename $d)" \
                --z_focus 0.20,0.35 \
                --z_ctrl "0.14,0.19;0.36,0.49" \
                --zbuf 0.2 \
                --rmin 0.1 --rmax 2.0 --nbin 8 &

            # Limit parallel jobs
            while [ $(jobs -r | wc -l) -ge $BATCH_SIZE ]; do
                sleep 10
            done
        fi
    done
    wait  # Wait for all background jobs
else
    echo "✓ All random stacks complete"
fi

# Step 6: Apply boost correction and compute z-score
echo ""
echo "Applying boost correction and computing z-score..."
python scripts/clens_boost_and_z.py \
    --lens "$OUT_DIR/profile_focus.csv" \
    --random_dir "$RAND_OUT" \
    --pattern "CLENS_random_strat3_*" \
    --out "$PROOF_DIR"

# Step 7: Display results
echo ""
echo "=== RESULTS ==="
if [ -f "$PROOF_DIR/Z.txt" ]; then
    echo "WL Δ z-score: $(cat $PROOF_DIR/Z.txt)"
fi
if [ -f "$PROOF_DIR/SUMMARY.txt" ]; then
    cat "$PROOF_DIR/SUMMARY.txt"
fi

echo ""
echo "Pipeline complete: $(date)"
echo "Results saved to: $PROOF_DIR/"