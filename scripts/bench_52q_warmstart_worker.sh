#!/bin/bash
# 52Q warm-start vs cold-start: same NQS, same expansion, same basis growth.
# Only difference is whether Davidson uses prev eigenvector as ci0.
# 4 tasks: 2 modes × 2 seeds.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
echo "[52q-ws | $HN] start $(date)"

run_task() {
    local gpu=$1 ws=$2 seed=$3
    local tag="warm${ws}_s${seed}"
    echo "[GPU=$gpu | $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/52q_warmstart/${tag}.json"
    log="logs/52q_warmstart/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_52q_warmstart_test.py \
        --warm_start=$ws --seed=$seed --out="$out" \
        > "$log" 2>&1 || echo "  [GPU $gpu $tag FAILED]"
    echo "[GPU=$gpu | $tag] done $(date)"
}

pids=()
run_task 0 0 42  & pids+=($!)
run_task 1 0 777 & pids+=($!)
run_task 2 1 42  & pids+=($!)
run_task 3 1 777 & pids+=($!)

rc=0
for p in "${pids[@]}"; do
    if ! wait $p; then rc=1; fi
done
echo "[52q-ws] done $(date) rc=$rc"
exit $rc
