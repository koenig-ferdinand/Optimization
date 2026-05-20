#!/bin/bash
# =============================================================================
# run_muon_baseline.sh  —  Train a Muon baseline with a custom iteration count
# =============================================================================
#
# Trains Muon with a fully-decayed LR schedule and saves the log to:
#   V6 - Train/logs/muon_{NUM_ITER}/log.txt
#
# sweep_summary.py will automatically pick this up as the Muon baseline
# when SWEEP_END matches NUM_ITER.
#
# Submit:
#   sbatch "V6 - Train/slurm/run_muon_baseline.sh"
#
# Monitor:
#   squeue -u $USER
#   tail -f "V6 - Train/logs/muon_3000/log.txt"
# =============================================================================

#SBATCH --job-name=muon_baseline
#SBATCH --account=es_he
#SBATCH --output=/cluster/scratch/leochen/Muon/V4/slurm_logs/muon_baseline_%j.out
#SBATCH --error=/cluster/scratch/leochen/Muon/V4/slurm_logs/muon_baseline_%j.err
#SBATCH --time=02:00:00          # ~50 min for 3000 iter + buffer
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:4

# =============================================================================
# Config  —  adjust these to change which baseline you train
# =============================================================================

NUM_ITER=3000
WARMDOWN=870        # proportional to 6200-step run: round(1800/6200 * NUM_ITER)
SAVE_EVERY=100

# =============================================================================
# Environment
# =============================================================================

eval "$(/cluster/scratch/leochen/miniconda3/bin/conda shell.bash hook)"
conda activate muon

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT="/cluster/scratch/leochen/Muon/V4"
V6_DIR="$PROJECT_ROOT/V6 - Train"

echo "Project root  : $PROJECT_ROOT"
echo "Job ID        : $SLURM_JOB_ID"
echo "num_iterations: $NUM_ITER"
echo "warmdown_iters: $WARMDOWN"
echo "exp_name      : muon_${NUM_ITER}"
echo "Started       : $(date)"
echo "============================================================"

cd "$PROJECT_ROOT"

torchrun \
    --standalone \
    --nproc_per_node=4 \
    "$V6_DIR/train_muon_free.py" \
    --exp_name       "muon_${NUM_ITER}" \
    --num_iterations "$NUM_ITER" \
    --warmdown_iters "$WARMDOWN" \
    --save_every     "$SAVE_EVERY"

echo "============================================================"
echo "Done: $(date)"
echo "Log saved to: $V6_DIR/logs/muon_${NUM_ITER}/log.txt"
