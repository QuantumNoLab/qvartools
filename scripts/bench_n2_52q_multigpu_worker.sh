#!/bin/bash
# 1 task per node × 8 GPUs = each task uses 8 GPUs for multi-GPU on-the-fly.
# 2 nodes = 2 budgets (1M and 2M) tested simultaneously.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2 tasks: (budget) per node
TASKS=(
    "1000000"
    "2000000"
)

idx=$NODE_IDX
budget="${TASKS[$idx]}"
tag="bud${budget}"
echo "[worker $NODE_IDX | $HN | GPUs=0..7 | budget=$budget] $(date)"

out="results/n2_52q_multigpu/${tag}.json"
log="logs/n2_52q_multigpu/${tag}_${SLURM_JOB_ID}.log"

python -u scratch_n2_52q_multigpu_run.py \
    --budget=$budget --top_n=2000 --pt2_top_n=20000 \
    --max_iterations=30 --seed=42 --out="$out" \
    > "$log" 2>&1 || echo "  [task $idx FAILED]"

echo "[worker $NODE_IDX | $HN | task=$idx] done $(date)"
