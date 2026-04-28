#!/bin/bash
# 52Q config scan: 4×4 grid (top_n × pt2_top_n) at budget=500k.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# 16 configs: top_n ∈ {1000, 2000, 3000, 5000} × pt2_top_n ∈ {10k, 20k, 30k, 50k}
TASKS=(
    # top_n=1000
    "1000:10000"    "1000:20000"    "1000:30000"    "1000:50000"
    # top_n=2000
    "2000:10000"    "2000:20000"    "2000:30000"    "2000:50000"
    # top_n=3000
    "3000:10000"    "3000:20000"    "3000:30000"    "3000:50000"
    # top_n=5000
    "5000:10000"    "5000:20000"    "5000:30000"    "5000:50000"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r tn ptn <<< "$t"
    local tag="tn${tn}_pt${ptn}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx top_n=$tn pt2=$ptn] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_52q_config_scan/${tag}.json"
    log="logs/n2_52q_config_scan/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_52q_config_scan_run.py \
        --top_n=$tn --pt2_top_n=$ptn --budget=500000 \
        --seed=42 --max_iterations=40 --out="$out" \
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
