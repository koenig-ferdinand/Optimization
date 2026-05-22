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

    # ── Phase 1 baseline — AdamW with same 3000-step schedule ────────────────
    # Run this once to get a fair comparison baseline for all Phase 1 reg runs.
    # After this finishes, sweep_summary.py will automatically use it instead
    # of log_adamw.txt (which used a 6200-step schedule).

    dict(exp_name='adamw_3000', reg_name='none', reg_lambda=0.0, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100,   early_stop=False),

    # ── Phase 3: gradient spectrum flattening sweep (3000 iter) ──────────────
    # Tests whether equalising gradient singular values (like Muon does) closes
    # the early-convergence gap. strength=0 → AdamW, strength=1 → Muon-like.
    # Baseline: adamw_3000 (strength=0). Target: muon_3000.

    dict(exp_name='gflat_25',  reg_name='none', reg_lambda=0.0, grad_flatten_strength=0.25, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),
    dict(exp_name='gflat_50',  reg_name='none', reg_lambda=0.0, grad_flatten_strength=0.50, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),
    dict(exp_name='gflat_75',  reg_name='none', reg_lambda=0.0, grad_flatten_strength=0.75, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),
    dict(exp_name='gflat_100', reg_name='none', reg_lambda=0.0, grad_flatten_strength=1.00, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),

    # ── Phase 3b: high-λ sv_variance (Phase 1 trend was monotone, push further) ─
    # Phase 1 best was λ=5e-4. Try higher to see if trend continues or diverges.
    # early_stop=True catches divergence automatically.
    dict(exp_name='sv_var_lam1e-3', reg_name='sv_variance', reg_lambda=1e-3, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='sv_var_lam3e-3', reg_name='sv_variance', reg_lambda=3e-3, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='sv_var_lam1e-2', reg_name='sv_variance', reg_lambda=1e-2, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),

    # ── Phase 4: four new methods (3000 iter) ────────────────────────────────
    # 4a. Dynamic sv_variance — self-amplifying lambda, gradient stays large
    #     even when variance collapses. Same λ range as Phase 1 for comparison.
    # dict(exp_name='dyn_sv_lam1e-4', reg_name='dynamic_sv_variance', reg_lambda=1e-4, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='dyn_sv_lam1e-3', reg_name='dynamic_sv_variance', reg_lambda=1e-3, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),

    # 4b. Log effective rank — log-barrier keeps gradient large near optimum.
    # dict(exp_name='log_effrank_lam1e-4', reg_name='log_effective_rank', reg_lambda=1e-4, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='log_effrank_lam1e-3', reg_name='log_effective_rank', reg_lambda=1e-3, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),

    # 4c. Gradient balancing — reg gradient scaled to fixed % of task gradient.
    #     ratio=0.05 means reg contributes exactly 5% of update magnitude.
    # dict(exp_name='gbal_5pct',  reg_name='sv_variance', reg_lambda=1.0, grad_balance_ratio=0.05, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    # dict(exp_name='gbal_10pct', reg_name='sv_variance', reg_lambda=1.0, grad_balance_ratio=0.10, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),

    # 4d. Post-update weight projection — directly flattens SV spectrum of
    #     weights after each AdamW step, bypassing v_t entirely.
    # dict(exp_name='wproj_01', reg_name='none', reg_lambda=0.0, weight_proj_strength=0.01, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),
    # dict(exp_name='wproj_05', reg_name='none', reg_lambda=0.0, weight_proj_strength=0.05, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),
    # dict(exp_name='wproj_10', reg_name='none', reg_lambda=0.0, weight_proj_strength=0.10, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),

    # ── Phase 2: full runs with best λ per regularizer (6200 iter) ───────────
    # sweep_summary.py will auto-use log_adamw_{SWEEP_END}.txt as the baseline.
    # Before running Phase 2: copy logs/adamw_3000/log.txt → log_adamw_3000.txt,
    # then set SWEEP_END=6200 in sweep_summary.py — it will fall back to log_adamw.txt.

    # dict(exp_name='full_sv_var',    reg_name='sv_variance',    reg_lambda=5e-4, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=6200, save_every=100, early_stop=False),
    # dict(exp_name='full_orth',      reg_name='orthogonal',     reg_lambda=1e-5, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=6200, save_every=100, early_stop=False),

]
