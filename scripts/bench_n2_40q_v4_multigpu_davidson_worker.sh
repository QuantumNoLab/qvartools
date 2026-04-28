#!/bin/bash
# One task per node, 8 GPUs per task. Each task tests full multi-GPU stack.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2 nodes x (variant, budget, mode):
# node 0: default@500k, multi-GPU Davidson only
# node 1: default@500k, Davidson + PT2 multi-GPU (full)
TASKS=(
    "default:500k:mgd_only"
    "default:500k:mgd_and_mgpt2"
)

idx=$NODE_IDX
t="${TASKS[$idx]}"
IFS=":" read -r variant budget mode <<< "$t"
tag="${variant}_${budget}_${mode}"
echo "[worker $NODE_IDX | $HN | GPUs=0..7 | task=$idx $tag] $(date)"

out="results/n2_40q_v4_multigpu_davidson/${tag}.json"
log="logs/n2_40q_v4_multigpu_davidson/${tag}_${SLURM_JOB_ID}.log"

python -u scratch_n2_40q_v4_multigpu_davidson_run.py \
    --variant="$variant" --budget="$budget" --mode="$mode" --seed=42 --out="$out" \
    > "$log" 2>&1 || echo "  [task $idx FAILED (see $log)]"

echo "[worker $NODE_IDX | $HN | task=$idx] done $(date)"
