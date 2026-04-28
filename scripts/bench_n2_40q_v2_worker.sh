#!/bin/bash
# Per-node worker. 16 tasks: 4 variants x 4 budgets.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# variant:budget (16 entries; index = global task ID)
TASKS=(
    # ---- budget=100k (4 variants) ----
    "v1:100k"          "v2_seed:100k"     "v2_expand:100k"   "v2_full:100k"
    # ---- budget=200k (4 variants) ----
    "v1:200k"          "v2_seed:200k"     "v2_expand:200k"   "v2_full:200k"
    # ---- budget=500k (4 variants) ----
    "v1:500k"          "v2_seed:500k"     "v2_expand:500k"   "v2_full:500k"
    # ---- budget=1m (4 variants) ----
    "v1:1m"            "v2_seed:1m"       "v2_expand:1m"     "v2_full:1m"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r variant budget <<< "$t"
    tag="${variant}_${budget}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_40q_v2/${tag}.json"
    log="logs/n2_40q_v2/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_40q_v2_run.py \
        --variant="$variant" --budget="$budget" --seed=42 --out="$out" \
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
