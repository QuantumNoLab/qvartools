#!/bin/bash
# Per-node worker launched by srun.
# SLURM_PROCID in {0, 1} identifies which node-worker.
# Each worker forks 8 subprocesses, one per local GPU.

set -u

# System python has all deps; no conda needed.
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

HN=$(hostname)
NODE_IDX=${SLURM_PROCID:-0}
echo "[worker $NODE_IDX | $HN] start $(date) PATH=$PATH"
which python
python -c "import sys; print('python:', sys.executable)"

# Global task layout (16 entries, index = global task ID)
TASKS=(
    "classical::0"                                   # 0
    "nqs:H2O:42"     "nqs:H2O:2024"   "nqs:H2O:777"  # 1-3
    "nqs:BeH2:42"    "nqs:BeH2:2024"  "nqs:BeH2:777" # 4-6
    "nqs:NH3:42"     "nqs:NH3:2024"                  # 7-8
    "nqs:NH3:777"    "nqs:CH4:42"                    # 9-10
    "nqs:CH4:2024"   "nqs:CH4:777"                   # 11-12
    "nqs:N2:42"      "nqs:N2:2024"    "nqs:N2:777"   # 13-15
)

run_task() {
    local gidx=$1
    local gpu=$2
    local t="${TASKS[$gidx]}"
    IFS=":" read -r mode mol seed <<< "$t"
    echo "[worker $NODE_IDX | $HN | GPU=$gpu | task=$gidx $mode $mol $seed]"
    if [ "$mode" = "classical" ]; then
        export CUDA_VISIBLE_DEVICES=""
        for m in H2O BeH2 NH3 CH4 N2; do
            out="results/small_mol_ndet_scan/${m}_classical.json"
            log="logs/small_mol_ndet_scan/${m}_classical_${SLURM_JOB_ID}.log"
            python -u scratch_small_mol_ndet_scan.py \
                --molecule="$m" --mode=classical --out="$out" \
                > "$log" 2>&1 || echo "  [classical $m FAILED]"
        done
    else
        export CUDA_VISIBLE_DEVICES=$gpu
        out="results/small_mol_ndet_scan/${mol}_nqs_s${seed}.json"
        log="logs/small_mol_ndet_scan/${mol}_nqs_s${seed}_${SLURM_JOB_ID}.log"
        python -u scratch_small_mol_ndet_scan.py \
            --molecule="$mol" --mode=nqs --seed="$seed" --out="$out" \
            > "$log" 2>&1 || echo "  [task $gidx FAILED]"
    fi
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
    if ! wait $p; then
        rc=1
    fi
done

echo "[worker $NODE_IDX | $HN] all local subprocesses done $(date) rc=$rc"
exit $rc
