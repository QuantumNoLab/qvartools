#!/bin/bash
# 52Q n_samples × seed scan: 4 n_samples × 4 seeds = 16 tasks
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# n_samples : seed
TASKS=(
    # n_samples=1M (baseline)
    "1000000:42"   "1000000:2024"  "1000000:777"  "1000000:123"
    # n_samples=2M
    "2000000:42"   "2000000:2024"  "2000000:777"  "2000000:123"
    # n_samples=5M
    "5000000:42"   "5000000:2024"  "5000000:777"  "5000000:123"
    # n_samples=10M
    "10000000:42"  "10000000:2024" "10000000:777" "10000000:123"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r ns sd <<< "$t"
    local nsk=$((ns / 1000000))M
    local tag="ns${nsk}_s${sd}"
    echo "[worker $NODE_IDX | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_52q_sample_scan/${tag}.json"
    log="logs/n2_52q_sample_scan/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_52q_sample_scan.py \
        --n_samples=$ns --top_k=50000 --budget=500000 \
        --top_n=2000 --pt2_top_n=20000 \
        --seed=$sd --max_iterations=20 --out="$out" \
        > "$log" 2>&1 || echo "  [task $gidx FAILED]"
    echo "[worker $NODE_IDX | task=$gidx] done $(date)"
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
echo "[worker $NODE_IDX] done $(date) rc=$rc"
exit $rc
