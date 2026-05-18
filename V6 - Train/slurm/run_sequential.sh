#!/bin/bash
# =============================================================================
# run_sequential.sh  —  Run all experiments from sweep_config.py sequentially
# =============================================================================
#
# Each experiment gets all 4 GPUs and runs to completion before the next starts.
# Experiments are defined in:  V6 - Train/sweep_config.py  (edit that file)
#
# Time estimate per experiment:
#   3000 iter sweep  → ~50 min  with 4 GPUs
#   6200 iter full   → ~110 min with 4 GPUs
#   18 sweep jobs    → ~15 h total  (submit before sleeping)
#
# Submit:
#   sbatch "V6 - Train/slurm/run_sequential.sh"
#
# Monitor:
#   squeue -u $USER
#   tail -f "V6 - Train/logs/<exp_name>/log.txt"
#
# After sweep finishes, run:
#   python "V6 - Train/slurm/sweep_summary.py"
# =============================================================================

#SBATCH --job-name=reg_sequential
#SBATCH --account=es_he
#SBATCH --output=/cluster/scratch/leochen/Muon/V4/slurm_logs/sequential_%j.out
#SBATCH --error=/cluster/scratch/leochen/Muon/V4/slurm_logs/sequential_%j.err
#SBATCH --time=20:00:00          # 20 h wall time  (18 sweeps × ~50 min + buffer)
                                 # Change to 07:00:00 for 3 full runs
#SBATCH --ntasks=1               # 4 MPI ranks = 4 GPU processes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        # CPU threads per GPU
#SBATCH --mem-per-cpu=8G         # 32 GB per GPU process

# ── Adjust the GPU type for your Euler partition, e.g.: ──────────────────────
#SBATCH --gpus=rtx_4090:4

# Check available types:  sinfo -o "%P %G" | grep gpu

# =============================================================================
# Environment  —  ADJUST THESE TWO LINES for your Euler setup
# =============================================================================

eval "$(/cluster/scratch/leochen/miniconda3/bin/conda shell.bash hook)"
conda activate muon
# — OR if using conda —
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your_env>

# =============================================================================
# Paths  —  adjust PROJECT_ROOT if your folder layout differs
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export V6_DIR="$PROJECT_ROOT/V6 - Train"   # exported so Python heredoc can read it

echo "Project root : $PROJECT_ROOT"
echo "V6 dir       : $V6_DIR"
echo "Job ID       : $SLURM_JOB_ID"
echo "Started      : $(date)"
echo "============================================================"

cd "$PROJECT_ROOT"    # torchrun data paths (data/fineweb10B/) are relative to V4/

# =============================================================================
# Read experiments from sweep_config.py and run them one by one
# =============================================================================

python3 - <<'PYEOF'
import subprocess, sys, os, time

# Load experiment list from sweep_config.py
sys.path.insert(0, os.environ.get('V6_DIR', 'V6 - Train'))
from sweep_config import EXPERIMENTS

v6_dir    = os.environ.get('V6_DIR', 'V6 - Train')
train_script = os.path.join(v6_dir, 'train_adamw_prewarm_fine.py')

total = len(EXPERIMENTS)
for idx, exp in enumerate(EXPERIMENTS, 1):
    name = exp['exp_name']
    print(f'\n{"="*60}', flush=True)
    print(f'[{idx}/{total}] Starting: {name}', flush=True)
    print(f'  reg={exp["reg_name"]}  λ={exp["reg_lambda"]}  '
          f'matrices={exp["reg_matrices"]}  iters={exp["num_iterations"]}', flush=True)
    print(f'  {time.strftime("%H:%M:%S")}', flush=True)
    print(f'{"="*60}', flush=True)

    cmd = [
        'torchrun',
        '--standalone',
        '--nproc_per_node=4',
        train_script,
        '--exp_name',       exp['exp_name'],
        '--reg_name',       exp['reg_name'],
        '--reg_lambda',     str(exp['reg_lambda']),
        '--reg_matrices',   exp['reg_matrices'],
        '--reg_layers',     exp['reg_layers'],
        '--num_iterations', str(exp['num_iterations']),
        '--save_every',     str(exp['save_every']),
    ]
    if exp.get('early_stop', False):
        cmd.append('--early_stop')

    t0  = time.time()
    ret = subprocess.run(cmd)
    elapsed = time.time() - t0

    status = 'OK' if ret.returncode == 0 else f'FAILED (code {ret.returncode})'
    print(f'\n[{idx}/{total}] {name} → {status}  ({elapsed/60:.1f} min)', flush=True)

print(f'\nAll {total} experiments finished.  {time.strftime("%H:%M:%S")}')
PYEOF

echo "============================================================"
echo "Batch job done: $(date)"
