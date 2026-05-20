# =============================================================================
# sweep_config.py  —  EDIT THIS FILE to control which experiments run
# =============================================================================
#
# Each entry in EXPERIMENTS is one training run.
# The SLURM script (slurm/run_sequential.sh) reads this file and executes
# all entries one after the other, each using 4 GPUs.
#
# Fields
# ------
# exp_name       Unique name → logs/{exp_name}/ will be created
# reg_name       Regularizer to apply. Options:
#                  'none'           — plain AdamW (baseline)
#                  'sv_variance'    — flatten spectrum (lower power-law α)
#                  'orthogonal'     — push WᵀW toward identity (mimics Muon)
#                  'isometry'       — scale-free orthogonality
#                  'effective_rank' — maximise entropy of σ distribution
#                  'stable_rank'    — maximise Σσᵢ²/σ₁²
#                  'dead_sv'        — penalise near-zero singular values
# reg_lambda     Regularization strength. Start small; see recommended ranges below.
# reg_matrices   Which weight matrices to penalise (comma-separated).
#                  Options: 'mlp.c_proj', 'mlp.c_fc', 'attn.c_proj'
#                  Recommendation: 'mlp.c_proj,mlp.c_fc' (strongest α signal)
# reg_layers     'all'  or  comma-separated layer indices, e.g. '6,7,8,9,10,11'
# num_iterations 3000 for sweep (fast), 6200 for full run (matches V5 baseline)
# save_every     How often to save checkpoints.
#                  500  for sweep (saves disk space)
#                  100  for full run (needed for evolution.py analysis)
# early_stop     True = kill run if val_loss exceeds V6 AdamW baseline + tolerance
#                  Use True for sweep, False for full runs
#
# Recommended λ ranges (start here, go up/down based on sweep results):
#   sv_variance    : 1e-5 … 5e-4
#   orthogonal     : 1e-6 … 5e-5   (strong effect, keep small)
#   isometry       : 1e-6 … 1e-4
#   effective_rank : 1e-5 … 1e-4
#   stable_rank    : 1e-5 … 1e-4
#   dead_sv        : 1e-5 … 1e-3
# =============================================================================


EXPERIMENTS = [

    # ── Phase 1: λ sweep (3000 iter, early stopping on) ─── DONE, do not re-run
    # dict(exp_name='sv_var_lam1e-5',  reg_name='sv_variance',    reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='sv_var_lam5e-5',  reg_name='sv_variance',    reg_lambda=5e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='sv_var_lam1e-4',  reg_name='sv_variance',    reg_lambda=1e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='sv_var_lam5e-4',  reg_name='sv_variance',    reg_lambda=5e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='orth_lam1e-6',    reg_name='orthogonal',     reg_lambda=1e-6,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='orth_lam5e-6',    reg_name='orthogonal',     reg_lambda=5e-6,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='orth_lam1e-5',    reg_name='orthogonal',     reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='orth_lam5e-5',    reg_name='orthogonal',     reg_lambda=5e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='effrank_lam1e-5', reg_name='effective_rank', reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='effrank_lam5e-5', reg_name='effective_rank', reg_lambda=5e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='effrank_lam1e-4', reg_name='effective_rank', reg_lambda=1e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='strank_lam1e-5',  reg_name='stable_rank',    reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='strank_lam5e-5',  reg_name='stable_rank',    reg_lambda=5e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='strank_lam1e-4',  reg_name='stable_rank',    reg_lambda=1e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='iso_lam1e-6',     reg_name='isometry',       reg_lambda=1e-6,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='iso_lam1e-5',     reg_name='isometry',       reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='deadsv_lam1e-5',  reg_name='dead_sv',        reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='deadsv_lam1e-4',  reg_name='dead_sv',        reg_lambda=1e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),

    # ── Phase 2: full runs with best λ per regularizer (6200 iter) ───────────
    # Best λ determined from Phase 1 sweep results (sweep_summary.py).
    # Ranked by gap closed at step 3000:
    #   sv_variance  λ=5e-4  → 103.8%   orthogonal  λ=1e-5  → 103.2%
    #   isometry     λ=1e-5  → 102.3%   dead_sv     λ=1e-5  → 101.2%
    #   stable_rank  λ=1e-5  → 100.5%   eff_rank    λ=5e-5  →  99.3%
    # Time estimate: 6 runs × ~200 min = ~20 h  (fits within 30 h wall time)

    dict(exp_name='full_sv_var',    reg_name='sv_variance',    reg_lambda=5e-4, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=6200, save_every=100, early_stop=False),
    dict(exp_name='full_orth',      reg_name='orthogonal',     reg_lambda=1e-5, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=6200, save_every=100, early_stop=False),

]
