#!/bin/bash
#SBATCH --job-name=nqs-bench
#SBATCH --account=gov114009
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:H100:3
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/leo07010/HI-NQS/nqs-sqd/bench_all_parallel_%j.log

echo "=== Job $SLURM_JOB_ID | $(hostname) ==="
echo "Start: $(date)"
echo "3 GPUs parallel: N₂ PES | Cr₂ PES | Fe clusters"
nvidia-smi --query-gpu=index,name --format=csv,noheader

source /home/leo07010/.bashrc 2>/dev/null || true
conda activate cudaq-env 2>/dev/null || source activate cudaq-env 2>/dev/null || true

cd /home/leo07010/HI-NQS/nqs-sqd

# GPU 0: N₂ PES (8 bond lengths)
CUDA_VISIBLE_DEVICES=0 python scripts/bench_n2_pes.py \
    > bench_n2_pes_${SLURM_JOB_ID}.log 2>&1 &
PID_N2=$!
echo "GPU 0: N₂ PES (PID=$PID_N2)"

# GPU 1: Cr₂ PES (7 bond lengths)
CUDA_VISIBLE_DEVICES=1 python scripts/bench_cr2_pes.py \
    > bench_cr2_pes_${SLURM_JOB_ID}.log 2>&1 &
PID_CR2=$!
echo "GPU 1: Cr₂ PES (PID=$PID_CR2)"

# GPU 2: Fe clusters ([2Fe-2S] + [4Fe-4S])
CUDA_VISIBLE_DEVICES=2 python scripts/bench_fe_clusters.py \
    > bench_fe_clusters_${SLURM_JOB_ID}.log 2>&1 &
PID_FE=$!
echo "GPU 2: Fe clusters (PID=$PID_FE)"

echo "All 3 tasks launched, waiting..."

# Wait and report
wait $PID_N2
echo "N₂ PES done (exit=$?): $(date)"

wait $PID_CR2
echo "Cr₂ PES done (exit=$?): $(date)"

wait $PID_FE
echo "Fe clusters done (exit=$?): $(date)"

echo "=== All done: $(date) ==="
