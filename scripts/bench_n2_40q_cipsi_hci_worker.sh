#!/bin/bash
# 16 tasks: 8 HCI cutoffs + 8 CIPSI budgets.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# backend:param pairs
TASKS=(
    # HCI cutoffs (descending — smaller ε = bigger N_det)
    "hci:1e-2"      "hci:5e-3"      "hci:1e-3"      "hci:5e-4"
    "hci:1e-4"      "hci:5e-5"      "hci:2e-5"      "hci:1e-5"
    # CIPSI budgets
    "cipsi:1000"    "cipsi:5000"    "cipsi:20000"   "cipsi:50000"
    "cipsi:100000"  "cipsi:200000"  "cipsi:500000"  "cipsi:1000000"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r backend param <<< "$t"
    local tag="${backend}_${param}"
    # Dots in cutoffs are fine; replace scientific e notation marker
    tag="${tag//[.]/p}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_40q_cipsi_hci/${tag}.json"
    log="logs/n2_40q_cipsi_hci/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_40q_cipsi_hci_scan.py \
        --backend="$backend" --param="$param" --out="$out" \
        > "$log" 2>&1 || echo "  [task $gidx FAILED (see $log)]"
    echo "[worker $NODE_IDX | $HN | task=$gidx] done $(date)"
}

pids=()
base=$((NODE_IDX * 8))
for local_gpu in 0 1 2 3 4 5 6 7; do
    gidx=$((base + local_gpu))
    run_task $gidx $local_gpu &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    if ! wait $p; then rc=1; fi
done
echo "[worker $NODE_IDX | $HN] done $(date) rc=$rc"
exit $rc
