#!/bin/bash
# N2 20Q PES scan: 10 bond lengths evenly split across 2 nodes × 5 GPUs each.
# Avoids previous srun load-imbalance termination.
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# 10 bond lengths, evenly distributed (5 per node)
BOND_LENGTHS=(0.8 1.0 1.1 1.3 1.5 1.8 2.2 2.6 3.0 4.0)

run_task() {
    local gidx=$1
    local gpu=$2
    if [ $gidx -ge ${#BOND_LENGTHS[@]} ]; then
        echo "[worker $NODE_IDX | GPU=$gpu | task=$gidx] no task assigned (only 10 total)"
        return
    fi
    local R=${BOND_LENGTHS[$gidx]}
    local tag="R_${R//./p}"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx R=$R] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_20q_pes/${tag}.json"
    log="logs/n2_20q_pes/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_20q_pes_scan.py \
        --bond_length="$R" --methods="fci,ccsd,hci,cipsi,nqs" --out="$out" \
        > "$log" 2>&1 || echo "  [task $gidx FAILED (see $log)]"
    echo "[worker $NODE_IDX | $HN | task=$gidx] done $(date)"
}

pids=()
# Evenly split: node 0 gets tasks 0..4 (on GPUs 0..4), node 1 gets 5..9
# Each node uses only 5 of its 8 GPUs. Idle 3 GPUs per node.
base=$((NODE_IDX * 5))
for local_gpu in 0 1 2 3 4; do
    gidx=$((base + local_gpu))
    run_task $gidx $local_gpu &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    if ! wait $p; then rc=1; fi
done
echo "[worker $NODE_IDX | $HN] all done $(date) rc=$rc"
exit $rc
