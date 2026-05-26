#!/bin/bash
# =============================================================================
# run_hybrid2.sh  —  Phase 2 hybrid experiments: LR diagnosis + paper replication
# =============================================================================
#
# Builds on run_hybrid.sh Phase 1 results. Addresses two issues:
#   1. LR imbalance: MLP matrices (3072×768) get Muon steps 2× larger than attn,
#      causing GELU instability when c_fc goes under Muon.
#   2. Paper replication: split Q/K/V to put V under Muon while K,Q stay AdamW.
#
# ── Phase 1: LR diagnosis (submit all 3 jobs simultaneously for max speed) ───
#
#   Phase 1 / Job A  (experiments 1–2, ~1.7h):  Muon Wout LR sweep
#     hybrid_cproj_lr05   Muon mlp.c_proj, lr=0.05×
#     hybrid_cproj_lr02   Muon mlp.c_proj, lr=0.02×
#
#   Phase 1 / Job B  (experiments 3–4, ~1.7h):  Muon Win LR + Q2 layer fill
#     hybrid_cfc_lr02     Muon mlp.c_fc,   lr=0.02×
#     hybrid_q2           Muon layers 3-5 (Q2), all matrices
#
#   Phase 1 / Job C  (experiment 5, ~0.9h):  Q3 layer fill
#     hybrid_q3           Muon layers 6-8 (Q3), all matrices
#
# ── Phase 2: Coupled MLP + paper replication (submit after Phase 1 results) ──
#
#   Phase 2 / Job D  (experiments 6–10, ~4.5h):
#     hybrid_mlp_lr02     Muon Win+Wout (mlp.c_fc,mlp.c_proj), lr=0.02×
#     hybrid_mlp_lr01     Muon Win+Wout (mlp.c_fc,mlp.c_proj), lr=0.01×
#     hybrid_out          Muon O+Wout   (attn.c_proj,mlp.c_proj), lr=0.02× MLP
#     hybrid_vo           Muon V+O      (attn.c_v,attn.c_proj), split_qkv
#     hybrid_vo_ffn       Muon V+O+FFN  (attn.c_v,c_proj,c_fc,mlp.c_proj), split_qkv, lr=0.02× MLP
#
# ── Submit commands ───────────────────────────────────────────────────────────
#
#   Phase 1 (submit all three at once):
#     SLICE_START=1 SLICE_END=2 sbatch --time=02:00:00 "V6 - Train/slurm/run_hybrid2.sh"
#     SLICE_START=3 SLICE_END=4 sbatch --time=02:00:00 "V6 - Train/slurm/run_hybrid2.sh"
#     SLICE_START=5 SLICE_END=5 sbatch --time=01:30:00 "V6 - Train/slurm/run_hybrid2.sh"
#
#   Phase 2 (after reviewing Phase 1 results):
#     SLICE_START=6 SLICE_END=10 sbatch --time=05:00:00 "V6 - Train/slurm/run_hybrid2.sh"
#
# Monitor:
#   squeue -u $USER
#   tail -f "V6 - Train/logs/hybrid_cproj_lr02/log.txt"
#
# Compare results:
#   python "V6 - Train/slurm/sweep_summary.py"
# =============================================================================

#SBATCH --job-name=hybrid2_muon
#SBATCH --account=es_he
#SBATCH --output=/cluster/scratch/leochen/Muon/V4/slurm_logs/hybrid2_%j.out
#SBATCH --error=/cluster/scratch/leochen/Muon/V4/slurm_logs/hybrid2_%j.err
#SBATCH --time=05:00:00          # default covers worst case; override per-job with --time=
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
# Format: "exp_name|muon_layers|muon_matrices|muon_mlp_lr_ratio|split_qkv"
#   muon_mlp_lr_ratio : 0 = don't pass flag (use default attn lr)
#   split_qkv         : 0 = standard fused c_attn, 1 = pass --split_qkv
#
# ── Phase 1 / Job A (1-2): Muon Wout LR sweep ────────────────────────────────
EXPS=(
    "hybrid_cproj_lr05|all|mlp.c_proj|0.05|0"
    "hybrid_cproj_lr02|all|mlp.c_proj|0.02|0"
# ── Phase 1 / Job B (3-4): Muon Win LR + Q2 layer fill ───────────────────────
    "hybrid_cfc_lr02|all|mlp.c_fc|0.02|0"
    "hybrid_q2|3,4,5|all|0|0"
# ── Phase 1 / Job C (5): Q3 layer fill ───────────────────────────────────────
    "hybrid_q3|6,7,8|all|0|0"
# ── Phase 2 / Job D (6-10): Coupled MLP + paper replication ──────────────────
    "hybrid_mlp_lr02|all|mlp.c_fc,mlp.c_proj|0.02|0"
    "hybrid_mlp_lr01|all|mlp.c_fc,mlp.c_proj|0.01|0"
    "hybrid_out|all|attn.c_proj,mlp.c_proj|0.02|0"
    "hybrid_vo|all|attn.c_v,attn.c_proj|0|1"
    "hybrid_vo_ffn|all|attn.c_v,attn.c_proj,mlp.c_fc,mlp.c_proj|0.02|1"
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
    IFS='|' read -r EXP_NAME MUON_LAYERS MUON_MATRICES MUON_MLP_LR_RATIO SPLIT_QKV <<< "$entry"

    echo ""
    echo "============================================================"
    echo "[${RUN_IDX}/${RUN_TOTAL}] Starting : $EXP_NAME"
    echo "  muon_layers         : $MUON_LAYERS"
    echo "  muon_matrices       : $MUON_MATRICES"
    echo "  muon_mlp_lr_ratio   : $MUON_MLP_LR_RATIO"
    echo "  split_qkv           : $SPLIT_QKV"
    echo "  num_iterations      : $NUM_ITER"
    echo "  $(date)"
    echo "============================================================"

    # Build optional flags
    EXTRA_FLAGS=""
    if [ "$MUON_MLP_LR_RATIO" != "0" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --muon_mlp_lr_ratio $MUON_MLP_LR_RATIO"
    fi
    if [ "$SPLIT_QKV" = "1" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --split_qkv"
    fi

    torchrun \
        --standalone \
        --nproc_per_node=4 \
        "$TRAIN_SCRIPT" \
        --exp_name       "$EXP_NAME" \
        --muon_layers    "$MUON_LAYERS" \
        --muon_matrices  "$MUON_MATRICES" \
        --num_iterations "$NUM_ITER" \
        --warmdown_iters "$WARMDOWN" \
        --save_every     "$SAVE_EVERY" \
        $EXTRA_FLAGS

    RC=$?
    if [ $RC -eq 0 ]; then
        echo "[OK] $EXP_NAME finished  ($(date))"
    else
        echo "[FAILED] $EXP_NAME exited with code $RC  ($(date))"
    fi
done

echo ""
echo "============================================================"
echo "Hybrid2 job done: $(date)"
echo "Run sweep_summary.py to compare results:"
echo "  python \"$V6_DIR/slurm/sweep_summary.py\""
echo "============================================================"
