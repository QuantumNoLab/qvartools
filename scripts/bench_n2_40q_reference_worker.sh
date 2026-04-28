#!/bin/bash
# Per-node worker. SLURM_PROCID in {0,1} selects node; forks 8 local subprocesses.
# Global task ID = NODE_IDX*8 + local_gpu.
#
# Task 0 (GPU 0): OLD incremental reproduction with n_α×n_β diagnostic.
# Tasks 1-15    : NEW NQS sparse_det, 15 distinct CONFIGS scanning:
#                 A. max_basis scan       (5)
#                 B. n_samples scan       (3)
#                 C. top_k scan           (2)
#                 D. NQS training variants (5)
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"
which python
python -c "import sys; print('python:', sys.executable)"

TASKS=(
    # 0: OLD reproduction (with diagnostic)
    "old_orig_incremental"
    # 1-5: max_basis scan
    "mb_100k"   "mb_200k"   "mb_500k"   "mb_1m"     "mb_1500k"
    # 6-8: n_samples scan
    "s_200k"    "s_500k"    "s_2m"
    # 9-10: top_k scan
    "tk_50k"    "tk_200k"
    # 11-15: NQS training variants
    "var_steps10"  "var_mono"  "var_hot"  "var_cold"  "var_slowlr"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local cfg="${TASKS[$gidx]}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx config=$cfg] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_40q_reference/${cfg}.json"
    log="logs/n2_40q_reference/${cfg}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_40q_reference_run.py \
        --config="$cfg" --seed=42 --out="$out" \
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

echo "[worker $NODE_IDX | $HN] all local subprocesses done $(date) rc=$rc"
exit $rc
