#!/bin/bash
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# 16 tasks: push small budgets (see precision floor) + match v3 budgets for
# direct speedup comparison.
TASKS=(
    # default variant (expand_top_n=1000): 6 budgets
    "default:30k"    "default:50k"    "default:100k"
    "default:200k"   "default:500k"   "default:1m"
    # small variant (expand_top_n=200): 6 budgets
    "small:30k"      "small:50k"      "small:100k"
    "small:200k"     "small:500k"     "small:1m"
    # big variant (expand_top_n=2000): 4 budgets
    "big:100k"       "big:500k"       "big:1m"         "big:2m"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r variant budget <<< "$t"
    tag="${variant}_${budget}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_40q_v4/${tag}.json"
    log="logs/n2_40q_v4/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_40q_v4_run.py \
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
