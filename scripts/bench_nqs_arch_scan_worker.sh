#!/bin/bash
# 8 NQS sizes × 2 budgets = 16 tasks
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# embed:heads:layers:budget
TASKS=(
    # 200k budget
    "32:2:2:200000"     "64:4:3:200000"     "96:4:3:200000"     "128:4:4:200000"
    "192:8:6:200000"    "256:8:10:200000"   "384:12:16:200000"  "512:16:20:200000"
    # 500k budget
    "32:2:2:500000"     "64:4:3:500000"     "96:4:3:500000"     "128:4:4:500000"
    "192:8:6:500000"    "256:8:10:500000"   "384:12:16:500000"  "512:16:20:500000"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r embed heads layers bud <<< "$t"
    local bk=$((bud / 1000))k
    local tag="e${embed}_h${heads}_l${layers}_${bk}"
    echo "[worker $NODE_IDX | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/nqs_arch_scan/${tag}.json"
    log="logs/nqs_arch_scan/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_nqs_arch_scan.py \
        --embed_dim=$embed --n_heads=$heads --n_layers=$layers \
        --budget=$bud --top_n=2000 --pt2_top_n=20000 \
        --seed=42 --max_iterations=20 --out="$out" \
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
