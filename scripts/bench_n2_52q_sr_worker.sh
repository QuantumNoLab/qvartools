#!/bin/bash
# 52Q SR optimizer test: 16 tasks scanning SR hyperparams + 2 Adam baselines
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# 16 configurations: SR with various (lr, damping, K) + Adam baselines
TASKS=(
    # Adam baselines (use_sr=0)
    "0:1e-3:1e-4:64"   "0:1e-3:1e-4:64"
    # SR sweep (lr × damping)
    "1:1e-3:1e-4:64"   "1:5e-3:1e-4:64"   "1:1e-2:1e-4:64"   "1:5e-2:1e-4:64"
    "1:1e-3:1e-3:64"   "1:5e-3:1e-3:64"   "1:1e-2:1e-3:64"
    "1:1e-3:1e-2:64"   "1:5e-3:1e-2:64"   "1:1e-2:1e-2:64"
    # SR with bigger Fisher batch
    "1:5e-3:1e-4:128"  "1:5e-3:1e-4:32"
    "1:1e-2:1e-3:128"  "1:5e-3:1e-3:32"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r use_sr lr damp K <<< "$t"
    local mode="adam"
    [ "$use_sr" = "1" ] && mode="sr"
    local tag="${mode}_lr${lr}_d${damp}_K${K}_g${gidx}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_52q_sr/${tag}.json"
    log="logs/n2_52q_sr/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_52q_sr_run.py \
        --use_sr=$use_sr --sr_lr=$lr --sr_damping=$damp --sr_fisher_K=$K \
        --budget=500000 --top_n=2000 --pt2_top_n=20000 \
        --seed=$((42 + gidx)) --max_iterations=25 --out="$out" \
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
