#!/bin/bash
# 52Q top_k scan: 4 top_k × 4 seeds = 16 tasks. Fixed budget=500k.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# top_k : seed
TASKS=(
    # top_k=50k baseline (10 iter to fill 500k basis)
    "50000:42"     "50000:2024"   "50000:777"    "50000:123"
    # top_k=100k (5 iter to fill basis)
    "100000:42"    "100000:2024"  "100000:777"   "100000:123"
    # top_k=200k (2.5 iter to fill basis)
    "200000:42"    "200000:2024"  "200000:777"   "200000:123"
    # top_k=500k (= budget, fills at iter 1)
    "500000:42"    "500000:2024"  "500000:777"   "500000:123"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r tk sd <<< "$t"
    local tkk=$((tk / 1000))k
    local tag="tk${tkk}_s${sd}"
    echo "[worker $NODE_IDX | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_52q_topk_scan/${tag}.json"
    log="logs/n2_52q_topk_scan/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_52q_topk_run.py \
        --top_k=$tk --n_samples=1000000 --budget=500000 \
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
