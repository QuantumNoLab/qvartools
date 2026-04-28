#!/bin/bash
#SBATCH --job-name=vmc-smoke
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_vmc_smoke_%j.log

echo "=== Job $SLURM_JOB_ID ==="; date
cd /home/leo07010/HI-VQE
mkdir -p results/vmc_smoke logs/vmc_smoke
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh

# 4 ablations × 2 molecules: BeH2 + N2-CAS(10,20)
# (mode, molecule, gpu)
declare -a TASKS=(
    "baseline:BeH2:0"
    "vmc:BeH2:1"
    "baseline:N2-CAS(10,20):2"
    "vmc:N2-CAS(10,20):3"
)

run_task() {
    local t=$1
    IFS=":" read -r mode mol gpu <<< "$t"
    local mtag=$(echo "$mol" | tr '(),-' '_')
    local tag="${mode}_${mtag}"
    echo "[smoke | GPU=$gpu | $tag] $(date)"
    export CUDA_VISIBLE_DEVICES=$gpu
    out="results/vmc_smoke/${tag}.json"
    log="logs/vmc_smoke/${tag}_${SLURM_JOB_ID}.log"
    if [ "$mol" = "BeH2" ]; then
        topk=2000; budget=5000; ns=50000; iters=10
    else
        topk=10000; budget=50000; ns=300000; iters=8
    fi
    python -u scratch_vmc_smoke.py \
        --mode="$mode" --molecule="$mol" --seed=42 \
        --max_iterations=$iters --top_k=$topk \
        --n_samples=$ns --budget=$budget \
        --out="$out" \
        > "$log" 2>&1 || echo "  [GPU $gpu $tag FAILED]"
    echo "[smoke | GPU=$gpu | $tag] done $(date)"
}

pids=()
for t in "${TASKS[@]}"; do
    run_task "$t" &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    if ! wait $p; then rc=1; fi
done
echo "=== smoke done $(date) rc=$rc ==="
ls -l results/vmc_smoke/
exit $rc
