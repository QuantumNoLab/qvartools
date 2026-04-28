#!/bin/bash
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

TASKS=(
    # default: expand_top_n=1000, PT2 on
    "default:100k"   "default:200k"   "default:500k"   "default:1m"
    # nopt2: same but PT2 off (ablate PT2)
    "nopt2:100k"     "nopt2:200k"     "nopt2:500k"     "nopt2:1m"
    # small: expand_top_n=200, PT2 on (ablate top_n)
    "small:100k"     "small:200k"     "small:500k"     "small:1m"
    # big: expand_top_n=2000, PT2 on (saturate top_n)
    "big:100k"       "big:200k"       "big:500k"       "big:1m"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r variant budget <<< "$t"
    tag="${variant}_${budget}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_40q_v3/${tag}.json"
    log="logs/n2_40q_v3/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_40q_v3_run.py \
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
