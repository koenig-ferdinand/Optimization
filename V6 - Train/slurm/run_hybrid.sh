#!/bin/bash
# =============================================================================
# run_hybrid.sh  —  Layer-specific Muon substitution experiments (10 runs)
# =============================================================================
#
# Splits parameters: selected layers/matrices → Muon, everything else → AdamW.
# Tests WHICH layers and WHICH matrix types drive Muon's convergence advantage.
#
# Job 1 (experiments 1–5): Layer depth sweep — vary which layers get Muon
#   hybrid_first1  Muon layer 0 only
#   hybrid_first3  Muon layers 0-2 (first quarter)
#   hybrid_first6  Muon layers 0-5 (first half)
#   hybrid_last6   Muon layers 6-11 (second half)
#   hybrid_last3   Muon layers 9-11 (last quarter)
#
# Job 2 (experiments 6–10): Matrix type sweep — all layers, vary which matrices
#   hybrid_mlp      Muon mlp.c_fc + mlp.c_proj
#   hybrid_attn     Muon attn.c_attn + attn.c_proj
#   hybrid_cfc      Muon mlp.c_fc only
#   hybrid_cproj    Muon mlp.c_proj only
#   hybrid_attn_out Muon attn.c_proj only
#
# Submit job 1:  SLICE_START=1 SLICE_END=5 sbatch "V6 - Train/slurm/run_hybrid.sh"
# Submit job 2:  SLICE_START=6 SLICE_END=10 sbatch "V6 - Train/slurm/run_hybrid.sh"
# Submit all:    sbatch "V6 - Train/slurm/run_hybrid.sh"   (no env vars needed)
#
# Monitor:
#   squeue -u $USER
#   tail -f "V6 - Train/logs/hybrid_first3/log.txt"
#
# Compare results after both jobs finish:
#   python "V6 - Train/slurm/sweep_summary.py"
# =============================================================================

#SBATCH --job-name=hybrid_muon
#SBATCH --account=es_he
#SBATCH --output=/cluster/scratch/leochen/Muon/V4/slurm_logs/hybrid_%j.out
#SBATCH --error=/cluster/scratch/leochen/Muon/V4/slurm_logs/hybrid_%j.err
#SBATCH --time=05:00:00          # 5 runs × ~50 min + buffer (for half the set)
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:4

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
TRAIN_SCRIPT="$V6_DIR/train_hybrid.py"

# ── Slice (1-based, inclusive). Leave at 0/0 to run all 10. ──────────────────
SLICE_START=${SLICE_START:-0}
SLICE_END=${SLICE_END:-0}

echo "Project root : $PROJECT_ROOT"
echo "Job ID       : $SLURM_JOB_ID"
echo "Slice        : ${SLICE_START}-${SLICE_END}  (0/0 = all)"
echo "Started      : $(date)"
echo "============================================================"

cd "$PROJECT_ROOT"

# =============================================================================
# Shared settings (match adamw_3000 / muon_3000 baselines)
# =============================================================================

NUM_ITER=3000
WARMDOWN=870      # round(1800 / 6200 * 3000)
SAVE_EVERY=100

# =============================================================================
# Experiment list  (order = 1-based index for SLICE_START/SLICE_END)
# =============================================================================
#
# Format: "exp_name|muon_layers|muon_matrices"
#
# ── Job 1: Layer depth sweep (all matrix types → Muon) ───────────────────────
EXPS=(
    "hybrid_first1|0|all"
    "hybrid_first3|0,1,2|all"
    "hybrid_first6|0,1,2,3,4,5|all"
    "hybrid_last6|6,7,8,9,10,11|all"
    "hybrid_last3|9,10,11|all"
# ── Job 2: Matrix type sweep (all layers, vary matrix) ───────────────────────
    "hybrid_mlp|all|mlp.c_fc,mlp.c_proj"
    "hybrid_attn|all|attn.c_attn,attn.c_proj"
    "hybrid_cfc|all|mlp.c_fc"
    "hybrid_cproj|all|mlp.c_proj"
    "hybrid_attn_out|all|attn.c_proj"
)

TOTAL=${#EXPS[@]}

# Apply slice (1-based)
if [ "$SLICE_START" -gt 0 ]; then
    START_IDX=$(( SLICE_START - 1 ))
    if [ "$SLICE_END" -gt 0 ]; then
        END_IDX=$(( SLICE_END - 1 ))
    else
        END_IDX=$(( TOTAL - 1 ))
    fi
    EXPS=("${EXPS[@]:$START_IDX:$(( END_IDX - START_IDX + 1 ))}")
    echo "[INFO] Running experiments ${SLICE_START}–${SLICE_END} (${#EXPS[@]} total)"
fi

# =============================================================================
# Run experiments
# =============================================================================

RUN_TOTAL=${#EXPS[@]}
RUN_IDX=0

for entry in "${EXPS[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    IFS='|' read -r EXP_NAME MUON_LAYERS MUON_MATRICES <<< "$entry"

    echo ""
    echo "============================================================"
    echo "[${RUN_IDX}/${RUN_TOTAL}] Starting : $EXP_NAME"
    echo "  muon_layers   : $MUON_LAYERS"
    echo "  muon_matrices : $MUON_MATRICES"
    echo "  num_iterations: $NUM_ITER"
    echo "  $(date)"
    echo "============================================================"

    torchrun \
        --standalone \
        --nproc_per_node=4 \
        "$TRAIN_SCRIPT" \
        --exp_name       "$EXP_NAME" \
        --muon_layers    "$MUON_LAYERS" \
        --muon_matrices  "$MUON_MATRICES" \
        --num_iterations "$NUM_ITER" \
        --warmdown_iters "$WARMDOWN" \
        --save_every     "$SAVE_EVERY"

    RC=$?
    if [ $RC -eq 0 ]; then
        echo "[OK] $EXP_NAME finished  ($(date))"
    else
        echo "[FAILED] $EXP_NAME exited with code $RC  ($(date))"
    fi
done

echo ""
echo "============================================================"
echo "Hybrid job done: $(date)"
echo "Run sweep_summary.py to compare results:"
echo "  python \"$V6_DIR/slurm/sweep_summary.py\""
echo "============================================================"
