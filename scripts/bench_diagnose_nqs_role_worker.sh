#!/bin/bash
# 4 ablation modes × 2 seeds = 8 tasks on 8 GPUs (1 node).
# Each runs N2-CAS(10,26), top_k=20k, n_samples=500k, 5 iters (~30-45 min).
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
echo "[diag-worker | $HN] start $(date)"

# (mode, seed) pairs: 4 modes × 2 seeds
TASKS=(
    "baseline:42"      "baseline:777"
    "no_nqs:42"        "no_nqs:777"
    "nqs_only:42"      "nqs_only:777"
    "random_sample:42" "random_sample:777"
)

run_task() {
    local gpu=$1
    local t="${TASKS[$gpu]}"
    IFS=":" read -r mode seed <<< "$t"
    local tag="${mode}_s${seed}"
    echo "[diag | GPU=$gpu | $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/diagnose_nqs_role/${tag}.json"
    log="logs/diagnose_nqs_role/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_diagnose_nqs_role.py \
        --mode="$mode" --seed=$seed \
        --molecule="N2-CAS(10,26)" \
        --n_samples=500000 --top_k=20000 --budget=100000 \
        --max_iterations=5 --pt2_top_n=10000 \
        --out="$out" \
        > "$log" 2>&1 || echo "  [GPU $gpu $tag FAILED]"
    echo "[diag | GPU=$gpu | $tag] done $(date)"
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
    run_task $gpu &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    if ! wait $p; then rc=1; fi
done
echo "[diag-worker] done $(date) rc=$rc"
exit $rc
