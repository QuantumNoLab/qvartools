#!/bin/bash
# Test warm_start bug across N2-20Q PES.
# 7 R points × cold-start verification.
# Each task runs: warm-start v3 → save final basis → cold-start re-diag → compare.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
echo "[ws-pes | $HN] start $(date)"

# (R, FCI_ref) — FCI from PySCF n2_20q_pes runs
declare -a TASKS=(
    "0.8:-106.766308"
    "1.0:-107.549301"
    "1.1:-107.654122"
    "1.5:-107.581635"
    "1.8:-107.483457"
    "2.2:-107.444859"
    "2.6:-107.439785"
)

run_task() {
    local gpu=$1 task=$2
    IFS=":" read -r R fci <<< "$task"
    local Rtag=$(echo "$R" | tr '.' 'p')
    local tag="warmstart_R${Rtag}"
    echo "[GPU=$gpu | $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/warmstart_pes/${tag}.json"
    log="logs/warmstart_pes/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_test_warmstart_bug.py \
        --R=$R --FCI=$fci --out="$out" \
        > "$log" 2>&1 || echo "  [GPU $gpu $tag FAILED]"
    echo "[GPU=$gpu | $tag] done $(date)"
}

pids=()
for i in 0 1 2 3 4 5 6; do
    run_task $i "${TASKS[$i]}" &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    if ! wait $p; then rc=1; fi
done
echo "[ws-pes] done $(date) rc=$rc"
exit $rc
