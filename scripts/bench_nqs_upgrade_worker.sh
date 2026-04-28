#!/bin/bash
# 40Q verify + 52Q deploy of NQS upgrades (SR, MCMC, combined).
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# system:budget:use_sr:use_mcmc:max_iter
# 40Q verification: 4 variants × 1 budget = 4 tasks (small budget for fast)
# 52Q deployment: 4 variants × 2 budgets × 1 seed = 8 tasks
TASKS=(
    # 40Q verify @ budget=100k (Adam/SR/MCMC/Both)
    "N2-CAS(10,20):100000:0:0:25"   "N2-CAS(10,20):100000:1:0:25"
    "N2-CAS(10,20):100000:0:1:25"   "N2-CAS(10,20):100000:1:1:25"
    # 52Q deploy @ budget=200k (faster than 500k for testing)
    "N2-CAS(10,26):200000:0:0:25"   "N2-CAS(10,26):200000:1:0:25"
    "N2-CAS(10,26):200000:0:1:25"   "N2-CAS(10,26):200000:1:1:25"
    # 52Q deploy @ budget=500k (full scale)
    "N2-CAS(10,26):500000:0:0:25"   "N2-CAS(10,26):500000:1:0:25"
    "N2-CAS(10,26):500000:0:1:25"   "N2-CAS(10,26):500000:1:1:25"
    # 40Q SR+MCMC second seed for stat
    "N2-CAS(10,20):100000:1:1:25"   "N2-CAS(10,20):500000:1:1:25"
    # 52Q smaller budget for trend
    "N2-CAS(10,26):100000:1:1:25"   "N2-CAS(10,26):100000:0:0:25"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r sys bud sr mcmc miter <<< "$t"
    local mode="adam"; [ "$sr" = "1" ] && mode="sr"
    local samp="fwd"; [ "$mcmc" = "1" ] && samp="mcmc"
    local bk=$((bud / 1000))k
    local sysshort="${sys//[(),]/_}"
    local tag="${sysshort}_${bk}_${mode}_${samp}_g${gidx}"
    echo "[worker $NODE_IDX | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/nqs_upgrade/${tag}.json"
    log="logs/nqs_upgrade/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_nqs_upgrade_test.py \
        --system="$sys" --budget=$bud --use_sr=$sr --use_mcmc=$mcmc \
        --top_n=2000 --pt2_top_n=20000 \
        --seed=$((42 + gidx)) --max_iterations=$miter --out="$out" \
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
