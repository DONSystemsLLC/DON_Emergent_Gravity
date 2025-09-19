#!/bin/bash
# Monitor CLENS stacking progress

echo "=== CLENS Stacking Monitor ==="
echo "Started: $(date)"
echo ""

while true; do
    clear
    echo "=== CLENS Stacking Progress Monitor ==="
    echo "Time: $(date)"
    echo ""

    # Count running processes
    N_RUNNING=$(ps aux | grep 'clens_stack\.py' | grep -v grep | wc -l | tr -d ' ')
    echo "Running stack processes: $N_RUNNING"
    echo ""

    # Check completed lens stack
    if [ -f outputs/CLENS_patch/profile_focus.csv ]; then
        echo "✓ Lens stack complete: outputs/CLENS_patch/profile_focus.csv"
    else
        echo "⏳ Lens stack in progress..."
    fi
    echo ""

    # Count completed random stacks
    N_COMPLETE=$(ls outputs/CLENS_random_strat3_*/profile_focus.csv 2>/dev/null | wc -l | tr -d ' ')
    N_TOTAL=$(ls -d working/CLENS_random_strat3_* 2>/dev/null | wc -l | tr -d ' ')
    echo "Random stacks completed: $N_COMPLETE / $N_TOTAL"

    # Show progress bar
    if [ $N_TOTAL -gt 0 ]; then
        PCT=$((N_COMPLETE * 100 / N_TOTAL))
        printf "Progress: ["
        for i in $(seq 1 20); do
            if [ $((i * 5)) -le $PCT ]; then
                printf "█"
            else
                printf "░"
            fi
        done
        printf "] %d%%\n" $PCT
    fi
    echo ""

    # Check if all done
    if [ $N_RUNNING -eq 0 ] && [ $N_COMPLETE -eq $N_TOTAL ] && [ -f outputs/CLENS_patch/profile_focus.csv ]; then
        echo "🎉 All stacking complete! Ready for boost correction."
        echo ""
        echo "Next step:"
        echo "python scripts/clens_boost_and_z.py \\"
        echo "  --lens outputs/CLENS_patch/profile_focus.csv \\"
        echo "  --random_dir outputs \\"
        echo "  --pattern 'CLENS_random_strat3_*' \\"
        echo "  --out proofs/CLENS_KiDS_W1_Stacking"
        break
    fi

    # Show sample of running processes
    if [ $N_RUNNING -gt 0 ]; then
        echo "Sample of running processes:"
        ps aux | grep 'clens_stack\.py' | grep -v grep | head -3 | awk '{printf "  PID %s: CPU %.1f%% MEM %.1f%% - %s\n", $2, $3, $4, substr($0, index($0,$11))}'
    fi

    sleep 60  # Update every minute
done

echo ""
echo "Monitor complete: $(date)"