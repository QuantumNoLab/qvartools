#!/bin/bash
# One task per node, each task gets all 8 GPUs on that node.
# Demonstrates per-iter PT2 + final PT2 multi-GPU scaling.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# All 8 local GPUs visible to this task (1 task per node)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Each node picks 1 (variant, budget) config to test multi-GPU scaling
TASKS=(
    "default:500k"   # node 0: mid-size, should show strong scaling
    "big:1m"         # node 1: largest — most PT2 work to parallelize
)

idx=$NODE_IDX
t="${TASKS[$idx]}"
IFS=":" read -r variant budget <<< "$t"
tag="${variant}_${budget}_mg8"
echo "[worker $NODE_IDX | $HN | GPUs=0..7 | task=$idx $tag] $(date)"

out="results/n2_40q_v4_multi/${tag}.json"
log="logs/n2_40q_v4_multi/${tag}_${SLURM_JOB_ID}.log"

python -u scratch_n2_40q_v4_multi_run.py \
    --variant="$variant" --budget="$budget" --seed=42 --out="$out" \
    > "$log" 2>&1 || echo "  [task $idx FAILED (see $log)]"

echo "[worker $NODE_IDX | $HN | task=$idx] done $(date)"
