#!/bin/bash
# Resubmit missing strongly correlated PES bonds (killed by srun on 26583).
set -u
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date)"

# 5 missing bonds: R = 1.8, 2.2, 2.6, 3.0, 4.0 Å (strongly correlated regime)
# Both nodes run the same 5 tasks (seeded same, so identical — use one node's
# output) — safer than asymmetric node split.
BOND_LENGTHS=(1.8 2.2 2.6 3.0 4.0)

run_task() {
    local gidx=$1
    local gpu=$2
    if [ $gidx -ge ${#BOND_LENGTHS[@]} ]; then
        return
    fi
    local R=${BOND_LENGTHS[$gidx]}
    local tag="R_${R//./p}"
    # Only node 0 produces results (node 1 acts as load balancer so srun doesn't kill)
    if [ "$NODE_IDX" != "0" ]; then
        sleep 600  # make node 1 wait long enough that srun doesn't kill node 0 early
        return
    fi
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx R=$R] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/n2_20q_pes/${tag}.json"
    log="logs/n2_20q_pes/${tag}_${SLURM_JOB_ID}.log"
    python -u scratch_n2_20q_pes_scan.py \
        --bond_length="$R" --methods="fci,ccsd,hci,cipsi,nqs" --out="$out" \
        > "$log" 2>&1 || echo "  [task $gidx FAILED]"
    echo "[worker $NODE_IDX | $HN | task=$gidx] done $(date)"
}

pids=()
for local_gpu in 0 1 2 3 4; do
    run_task $local_gpu $local_gpu &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    if ! wait $p; then rc=1; fi
done
echo "[worker $NODE_IDX | $HN] done $(date) rc=$rc"
exit $rc
