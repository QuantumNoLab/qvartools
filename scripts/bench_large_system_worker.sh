#!/bin/bash
# Large-system v4 scan. 16 GPUs, 8 tasks across 2 nodes:
#  - N2-CAS(10,26) 52Q at budgets 50k, 100k, 200k, 500k
#  - 4Fe4S 72Q at budgets 50k, 100k, 200k, 500k
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# 16 tasks: 2 systems × 4 budgets × 2 replicas (budget replicas to increase chance
# some finish within wall time)
TASKS=(
    "N2-CAS(10,26):50k"    "N2-CAS(10,26):100k"  "N2-CAS(10,26):200k"  "N2-CAS(10,26):500k"
    "N2-CAS(10,26):50k"    "N2-CAS(10,26):100k"  "N2-CAS(10,26):200k"  "N2-CAS(10,26):500k"
    "4Fe4S:50k"            "4Fe4S:100k"          "4Fe4S:200k"          "4Fe4S:500k"
    "4Fe4S:50k"            "4Fe4S:100k"          "4Fe4S:200k"          "4Fe4S:500k"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r sys budget <<< "$t"
    # Use gidx-based seed to avoid redundant replicas
    local seed=$((42 + gidx))
    local tag="${sys//[()]/_}_${budget}_s${seed}"
    tag="${tag//,/_}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx sys=$sys bud=$budget seed=$seed] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/large_system/${tag}.json"
    log="logs/large_system/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_large_system_run.py \
        --system="$sys" --budget="$budget" --seed=$seed --out="$out" \
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
