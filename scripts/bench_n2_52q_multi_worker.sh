#!/bin/bash
# 52Q top_n × multi-T scan: 4 × 4 = 16 tasks, fixed budget=500k, n_samples=2M.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# top_n : temps (comma-sep, "" = single annealed T)
TASKS=(
    # top_n=2000 (baseline)
    "2000:"             "2000:0.5,1.5"      "2000:0.3,1.0,2.0"   "2000:0.2,0.7,1.3,2.5"
    # top_n=5000
    "5000:"             "5000:0.5,1.5"      "5000:0.3,1.0,2.0"   "5000:0.2,0.7,1.3,2.5"
    # top_n=10000
    "10000:"            "10000:0.5,1.5"     "10000:0.3,1.0,2.0"  "10000:0.2,0.7,1.3,2.5"
    # top_n=20000
    "20000:"            "20000:0.5,1.5"     "20000:0.3,1.0,2.0"  "20000:0.2,0.7,1.3,2.5"
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r tn temps <<< "$t"
    local tnk=$((tn / 1000))k
    local ttag=$(echo "$temps" | tr -d ',.' | sed 's/^$/single/')
    local tag="tn${tnk}_T${ttag}"
    echo "[worker $NODE_IDX | GPU=$gpu | task=$gidx $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_52q_multi/${tag}.json"
    log="logs/n2_52q_multi/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_52q_multi_run.py \
        --top_n=$tn --temps="$temps" \
        --n_samples=2000000 --top_k=50000 --budget=500000 \
        --pt2_top_n=20000 --seed=42 --max_iterations=20 \
        --out="$out" \
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
