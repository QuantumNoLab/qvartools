#!/bin/bash
# CIPSI-only rerun on N2-40Q: budgets not yet completed by 25660.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# 14 CIPSI budgets evenly distributed across 16 GPUs (2 idle per node).
# Skip the two that already completed on prior job.
TASKS=(
    "5000"     "20000"    "50000"    "100000"
    "200000"   "500000"   "1000000"  "2000000"
    "5000"     "20000"    "50000"    "100000"
    "200000"   "500000"   "1000000"  "2000000"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local budget="${TASKS[$gidx]}"
    local tag="cipsi_${budget}_g${gidx}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx budget=$budget] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_40q_cipsi_hci/${tag}.json"
    log="logs/n2_40q_cipsi_hci/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_40q_cipsi_hci_scan.py \
        --backend=cipsi --param="$budget" --out="$out" \
        > "$log" 2>&1 || echo "  [task $gidx FAILED]"
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
