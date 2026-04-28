#!/bin/bash
# v4 GPU Davidson vs scipy eigsh benchmark.
# Each (variant, budget) runs twice: once with scipy eigsh, once with GPU Davidson.
# Direct A/B comparison on same config.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# 8 pairs (variant, budget) — each run twice (scipy vs davidson) => 16 tasks
TASKS=(
    # node 0: scipy eigsh (baseline)
    "scipy:default:100k"   "scipy:default:200k"
    "scipy:default:500k"   "scipy:default:1m"
    "scipy:big:100k"       "scipy:big:200k"
    "scipy:big:500k"       "scipy:big:1m"
    # node 1: GPU Davidson (comparison)
    "davidson:default:100k"   "davidson:default:200k"
    "davidson:default:500k"   "davidson:default:1m"
    "davidson:big:100k"       "davidson:big:200k"
    "davidson:big:500k"       "davidson:big:1m"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r mode variant budget <<< "$t"
    tag="${mode}_${variant}_${budget}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu

    out="results/n2_40q_v4_davidson/${tag}.json"
    log="logs/n2_40q_v4_davidson/${tag}_${SLURM_JOB_ID}.log"

    flag=""
    if [ "$mode" = "davidson" ]; then
        flag="--gpu_davidson"
    fi

    python -u scratch_n2_40q_v4_davidson_run.py \
        --variant="$variant" --budget="$budget" --seed=42 --out="$out" $flag \
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
