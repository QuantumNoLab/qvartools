#!/bin/bash
# β-reshape ablation: does conditional temperature reshape fix mode collapse?
# 8 tasks: 2 betas × 2 molecules × 2 seeds.
# Compare to existing nqs_only_*_s{42,777}.json from job 32036.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
echo "[nqs-beta | $HN] start $(date)"

# (mode, molecule, seed) — 8 tasks on 8 GPUs
TASKS=(
    "nqs_only_beta04:N2-CAS(10,20):42"        "nqs_only_beta04:N2-CAS(10,20):777"
    "nqs_only_beta02:N2-CAS(10,20):42"        "nqs_only_beta02:N2-CAS(10,20):777"
    "nqs_only_beta04:N2-CAS(10,26):42"        "nqs_only_beta04:N2-CAS(10,26):777"
    "nqs_only_beta02:N2-CAS(10,26):42"        "nqs_only_beta02:N2-CAS(10,26):777"
)

run_task() {
    local gpu=$1
    local t="${TASKS[$gpu]}"
    IFS=":" read -r mode mol seed <<< "$t"
    local mtag=$(echo "$mol" | tr '(),-' '_')
    local tag="${mode}_${mtag}_s${seed}"
    echo "[GPU=$gpu | $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/nqs_beta_reshape/${tag}.json"
    log="logs/nqs_beta_reshape/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_diagnose_nqs_role.py \
        --mode="$mode" --seed=$seed \
        --molecule="$mol" \
        --n_samples=500000 --top_k=20000 --budget=100000 \
        --max_iterations=8 --pt2_top_n=10000 \
        --out="$out" \
        > "$log" 2>&1 || echo "  [GPU $gpu $tag FAILED]"
    echo "[GPU=$gpu | $tag] done $(date)"
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
echo "[nqs-beta] done $(date) rc=$rc"
exit $rc
