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
    dict(exp_name='sv_var_lam1e-5',  reg_name='sv_variance',    reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='sv_var_lam5e-5',  reg_name='sv_variance',    reg_lambda=5e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='sv_var_lam1e-4',  reg_name='sv_variance',    reg_lambda=1e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='sv_var_lam5e-4',  reg_name='sv_variance',    reg_lambda=5e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='orth_lam1e-6',    reg_name='orthogonal',     reg_lambda=1e-6,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='orth_lam5e-6',    reg_name='orthogonal',     reg_lambda=5e-6,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='orth_lam1e-5',    reg_name='orthogonal',     reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='orth_lam5e-5',    reg_name='orthogonal',     reg_lambda=5e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='effrank_lam1e-5', reg_name='effective_rank', reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='effrank_lam5e-5', reg_name='effective_rank', reg_lambda=5e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='effrank_lam1e-4', reg_name='effective_rank', reg_lambda=1e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='strank_lam1e-5',  reg_name='stable_rank',    reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='strank_lam5e-5',  reg_name='stable_rank',    reg_lambda=5e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='strank_lam1e-4',  reg_name='stable_rank',    reg_lambda=1e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='iso_lam1e-6',     reg_name='isometry',       reg_lambda=1e-6,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='iso_lam1e-5',     reg_name='isometry',       reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='deadsv_lam1e-5',  reg_name='dead_sv',        reg_lambda=1e-5,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='deadsv_lam1e-4',  reg_name='dead_sv',        reg_lambda=1e-4,  reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),

    # ── Phase 1 baseline — AdamW with same 3000-step schedule ────────────────
    # Run this once to get a fair comparison baseline for all Phase 1 reg runs.
    # After this finishes, sweep_summary.py will automatically use it instead
    # of log_adamw.txt (which used a 6200-step schedule).

    dict(exp_name='adamw_3000', reg_name='none', reg_lambda=0.0, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100,   early_stop=False),

    # ── Corrected AdamW baseline — same 3000-step schedule but with warmdown=871 ─
    # Previous adamw_3000 used the default warmdown=1800 (designed for 6200 steps),
    # meaning 60% of training ran at a reduced LR. This re-run uses the correct
    # proportional warmdown: round(1800/6200 * 3000) = 871 — matching muon_3000
    # and all hybrid experiments. Use this as the true baseline for hybrid gap %.
    dict(exp_name='adamw_3000_correct_warmdown', reg_name='none', reg_lambda=0.0, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),

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
    dict(exp_name='dyn_sv_lam1e-4', reg_name='dynamic_sv_variance', reg_lambda=1e-4, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='dyn_sv_lam1e-3', reg_name='dynamic_sv_variance', reg_lambda=1e-3, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),

    # 4b. Log effective rank — log-barrier keeps gradient large near optimum.
    dict(exp_name='log_effrank_lam1e-4', reg_name='log_effective_rank', reg_lambda=1e-4, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='log_effrank_lam1e-3', reg_name='log_effective_rank', reg_lambda=1e-3, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),

    # 4c. Gradient balancing — reg gradient scaled to fixed % of task gradient.
    #     ratio=0.05 means reg contributes exactly 5% of update magnitude.
    dict(exp_name='gbal_5pct',  reg_name='sv_variance', reg_lambda=1.0, grad_balance_ratio=0.05, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='gbal_10pct', reg_name='sv_variance', reg_lambda=1.0, grad_balance_ratio=0.10, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=True),

    # 4d. Post-update weight projection — directly flattens SV spectrum of
    #     weights after each AdamW step, bypassing v_t entirely.
    dict(exp_name='wproj_01', reg_name='none', reg_lambda=0.0, weight_proj_strength=0.01, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),
    dict(exp_name='wproj_05', reg_name='none', reg_lambda=0.0, weight_proj_strength=0.05, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),
    dict(exp_name='wproj_10', reg_name='none', reg_lambda=0.0, weight_proj_strength=0.10, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=3000, save_every=100, early_stop=False),

    # ── Phase 5a: g_flat annealing ────────────────────────────────────────────
    # Idea: high flatten strength during early exploration, then decay to 0 so
    # late-stage gradients are precise (pure AdamW).  Decoupled from LR schedule.
    # Baseline gflat_25/50/75 use constant strength throughout 3000 steps.
    # Here we test whether annealing the strength to 0 before warmdown helps.
    #
    #   gflat_ann_25_e1300 : 0.25→0 over steps 0–1300  (before warmdown at ~2130)
    #   gflat_ann_50_e1500 : 0.50→0 over steps 0–1500
    #   gflat_ann_75_e1500 : 0.75→0 over steps 0–1500
    #
    # reg_matrices kept as MLP (same as baseline gflat_*), so only changes
    # the schedule — not the target matrices.
    dict(exp_name='gflat_ann_25_e1300', reg_name='none', reg_lambda=0.0,
         grad_flatten_strength=0.25, grad_flatten_end_iter=1300,
         reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all',
         num_iterations=3000, save_every=100, early_stop=False),
    dict(exp_name='gflat_ann_50_e1500', reg_name='none', reg_lambda=0.0,
         grad_flatten_strength=0.50, grad_flatten_end_iter=1500,
         reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all',
         num_iterations=3000, save_every=100, early_stop=False),
    dict(exp_name='gflat_ann_75_e1500', reg_name='none', reg_lambda=0.0,
         grad_flatten_strength=0.75, grad_flatten_end_iter=1500,
         reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all',
         num_iterations=3000, save_every=100, early_stop=False),

    # ── Phase 5b: dynamic_sv_variance on attention matrices ───────────────────
    # Prior dyn_sv_lam* experiments targeted MLP (mlp.c_proj,mlp.c_fc).
    # Hybrid results show attention drives most of Muon's advantage (+2.9%).
    # Here we test whether keeping attention SV spectrum wide improves AdamW.
    #
    # Naming convention: "dyn_sv_attn_*" = both c_attn+c_proj,
    #                    "dyn_sv_cattn_*" = c_attn only (Q/K/V fused weight).
    #
    # Variants:
    #   all layers, λ=1e-4 and 1e-3  (λ range from Phase 4 MLP experiments)
    #   layers 0-5 only, λ=1e-4 and 1e-3  (early layers drive synergy per hybrid)
    #   c_attn only (no c_proj), all layers, λ=1e-4  (isolate Q/K/V vs O matrix)
    dict(exp_name='dyn_sv_attn_lam1e-4',       reg_name='dynamic_sv_variance', reg_lambda=1e-4,
         reg_matrices='attn.c_attn,attn.c_proj', reg_layers='all',
         num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='dyn_sv_attn_lam1e-3',       reg_name='dynamic_sv_variance', reg_lambda=1e-3,
         reg_matrices='attn.c_attn,attn.c_proj', reg_layers='all',
         num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='dyn_sv_attn_early_lam1e-4', reg_name='dynamic_sv_variance', reg_lambda=1e-4,
         reg_matrices='attn.c_attn,attn.c_proj', reg_layers='0,1,2,3,4,5',
         num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='dyn_sv_attn_early_lam1e-3', reg_name='dynamic_sv_variance', reg_lambda=1e-3,
         reg_matrices='attn.c_attn,attn.c_proj', reg_layers='0,1,2,3,4,5',
         num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='dyn_sv_cattn_lam1e-4',      reg_name='dynamic_sv_variance', reg_lambda=1e-4,
         reg_matrices='attn.c_attn', reg_layers='all',
         num_iterations=3000, save_every=100, early_stop=True),

    # ── Phase 5c: full 6200-iter run with best Phase 5b config ───────────────
    # dyn_sv_attn_lam1e-4 was the best sweep result (+4.3% gap closed vs AdamW).
    # This full run confirms whether the advantage holds to convergence and
    # produces a complete checkpoint sequence for evolution analysis.
    dict(exp_name='dyn_sv_attn_full_lam1e-4', reg_name='dynamic_sv_variance', reg_lambda=1e-4,
         reg_matrices='attn.c_attn,attn.c_proj', reg_layers='all',
         num_iterations=6200, save_every=100, early_stop=False),

    # ── Phase 5d: dynamic_sv_variance on ALL weight matrices ─────────────────
    # Phase 5b showed attn matrices drive most of the benefit.  Here we ask:
    # does adding MLP (c_fc + c_proj) on top of attn further help, hurt, or do nothing?
    #
    # reg_matrices covers all 4 weight matrix types across 12 layers = 48 pairs.
    # compute_reg_loss normalises by pair count, so the per-matrix gradient at a
    # given λ is halved compared to attn-only (24 pairs).  Lambda choices are
    # therefore shifted up relative to Phase 5b to keep per-matrix signal comparable:
    #
    #   lam5e-5  → per-matrix ≈ lam2.5e-5 on attn-only  (conservative; MLP can be fragile)
    #   lam1e-4  → per-matrix ≈ lam5e-5   on attn-only
    #   lam3e-4  → per-matrix ≈ lam1.5e-4 on attn-only  (slightly above best attn regime)
    dict(exp_name='dyn_sv_all_lam5e-5', reg_name='dynamic_sv_variance', reg_lambda=5e-5,
         reg_matrices='attn.c_attn,attn.c_proj,mlp.c_proj,mlp.c_fc', reg_layers='all',
         num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='dyn_sv_all_lam1e-4', reg_name='dynamic_sv_variance', reg_lambda=1e-4,
         reg_matrices='attn.c_attn,attn.c_proj,mlp.c_proj,mlp.c_fc', reg_layers='all',
         num_iterations=3000, save_every=100, early_stop=True),
    dict(exp_name='dyn_sv_all_lam3e-4', reg_name='dynamic_sv_variance', reg_lambda=3e-4,
         reg_matrices='attn.c_attn,attn.c_proj,mlp.c_proj,mlp.c_fc', reg_layers='all',
         num_iterations=3000, save_every=100, early_stop=True),

    # ── Phase 2: full runs with best λ per regularizer (6200 iter) ───────────
    # sweep_summary.py will auto-use log_adamw_{SWEEP_END}.txt as the baseline.
    # Before running Phase 2: copy logs/adamw_3000/log.txt → log_adamw_3000.txt,
    # then set SWEEP_END=6200 in sweep_summary.py — it will fall back to log_adamw.txt.

    # dict(exp_name='full_sv_var',    reg_name='sv_variance',    reg_lambda=5e-4, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=6200, save_every=100, early_stop=False),
    # dict(exp_name='full_orth',      reg_name='orthogonal',     reg_lambda=1e-5, reg_matrices='mlp.c_proj,mlp.c_fc', reg_layers='all', num_iterations=6200, save_every=100, early_stop=False),

]


# =============================================================================
# HYBRID_EXPERIMENTS  —  Layer-specific Muon substitution (train_hybrid.py)
# =============================================================================
#
# These experiments split parameters: selected layers/matrices → Muon,
# everything else → AdamW.  They are run via slurm/run_hybrid.sh, NOT via
# run_sequential.sh (which uses train_adamw_prewarm_fine.py).
#
# sweep_summary.py reads this list separately and adds the results to the table.
#
# Fields
# ------
# exp_name       Log directory: V6 - Train/logs/{exp_name}/log.txt
# muon_layers    'all'  or  comma-separated layer indices, e.g. '0,1,2'
# muon_matrices  'all'  or  comma-separated matrix names, e.g. 'mlp.c_fc'
# num_iterations Match the sweep baselines (3000)
# description    Short label shown in sweep_summary table
# =============================================================================

HYBRID_EXPERIMENTS = [

    # ── Job 1: Layer depth sweep (all 4 matrix types → Muon) ─────────────────
    # Question: do early layers, late layers, or all layers drive Muon's advantage?
    # GPT-2 has 12 transformer layers (0–11).

    # 1 — First layer only
    dict(exp_name='hybrid_first1', muon_layers='0',         muon_matrices='all', num_iterations=3000,
         description='Muon layer 0'),

    # 2 — First 3 layers (first quarter)
    dict(exp_name='hybrid_first3', muon_layers='0,1,2',     muon_matrices='all', num_iterations=3000,
         description='Muon layers 0-2'),

    # 3 — First half (layers 0–5)
    dict(exp_name='hybrid_first6', muon_layers='0,1,2,3,4,5', muon_matrices='all', num_iterations=3000,
         description='Muon layers 0-5 (first half)'),

    # 4 — Second half (layers 6–11)
    dict(exp_name='hybrid_last6',  muon_layers='6,7,8,9,10,11', muon_matrices='all', num_iterations=3000,
         description='Muon layers 6-11 (second half)'),

    # 5 — Last 3 layers (last quarter)
    dict(exp_name='hybrid_last3',  muon_layers='9,10,11',   muon_matrices='all', num_iterations=3000,
         description='Muon layers 9-11'),

    # ── Job 2: Matrix type sweep (all 12 layers, vary which matrices get Muon) ─
    # Question: which weight matrix type drives Muon's advantage?
    # Per block: mlp.c_fc (up-proj), mlp.c_proj (down-proj),
    #            attn.c_attn (Q/K/V), attn.c_proj (out-proj)

    # 6 — Both MLP projections, all layers
    dict(exp_name='hybrid_mlp',      muon_layers='all', muon_matrices='mlp.c_fc,mlp.c_proj',       num_iterations=3000,
         description='Muon MLP (c_fc+c_proj)'),

    # 7 — Both attention projections, all layers
    dict(exp_name='hybrid_attn',     muon_layers='all', muon_matrices='attn.c_attn,attn.c_proj',   num_iterations=3000,
         description='Muon Attn (c_attn+c_proj)'),

    # 8 — MLP up-projection only (largest MLP matrix, most effect on hidden state)
    dict(exp_name='hybrid_cfc',      muon_layers='all', muon_matrices='mlp.c_fc',                  num_iterations=3000,
         description='Muon mlp.c_fc only'),

    # 9 — MLP down-projection only (projects back to residual stream)
    dict(exp_name='hybrid_cproj',    muon_layers='all', muon_matrices='mlp.c_proj',                num_iterations=3000,
         description='Muon mlp.c_proj only'),

    # 10 — Attention output projection only (writes back to residual stream)
    dict(exp_name='hybrid_attn_out', muon_layers='all', muon_matrices='attn.c_proj',               num_iterations=3000,
         description='Muon attn.c_proj only'),

    # =========================================================================
    # Phase 2 experiments  —  run via slurm/run_hybrid2.sh
    # =========================================================================

    # ── Job 1 (1-5): LR diagnosis + quarter-layer sweep ──────────────────────
    # Phase 1a: was the Wout (mlp.c_proj) failure a pure LR-scale issue?
    # hybrid_cproj (0.10×, above) already exists — compare at 0.05× and 0.02×.
    dict(exp_name='hybrid_cproj_lr05', muon_layers='all', muon_matrices='mlp.c_proj',
         muon_mlp_lr_ratio=0.05, num_iterations=3000,
         description='Muon Wout lr=0.05x'),
    dict(exp_name='hybrid_cproj_lr02', muon_layers='all', muon_matrices='mlp.c_proj',
         muon_mlp_lr_ratio=0.02, num_iterations=3000,
         description='Muon Wout lr=0.02x'),

    # Phase 1a: same diagnostic for Win (mlp.c_fc) — deepest failure case
    dict(exp_name='hybrid_cfc_lr02',   muon_layers='all', muon_matrices='mlp.c_fc',
         muon_mlp_lr_ratio=0.02, num_iterations=3000,
         description='Muon Win lr=0.02x'),

    # Phase 2: fill in the missing middle quarters
    # (Q1≈first3 and Q4≈last3 already exist from Job 1 above)
    dict(exp_name='hybrid_q2', muon_layers='3,4,5', muon_matrices='all',
         num_iterations=3000, description='Muon layers 3-5 (Q2)'),
    dict(exp_name='hybrid_q3', muon_layers='6,7,8', muon_matrices='all',
         num_iterations=3000, description='Muon layers 6-8 (Q3)'),

    # ── Job 2 (6-10): Coupled MLP + paper replication ────────────────────────
    # Phase 1b: both MLP matrices under Muon with reduced LR — does coupling help?
    dict(exp_name='hybrid_mlp_lr02', muon_layers='all', muon_matrices='mlp.c_fc,mlp.c_proj',
         muon_mlp_lr_ratio=0.02, num_iterations=3000,
         description='Muon MLP lr=0.02x'),
    dict(exp_name='hybrid_mlp_lr01', muon_layers='all', muon_matrices='mlp.c_fc,mlp.c_proj',
         muon_mlp_lr_ratio=0.01, num_iterations=3000,
         description='Muon MLP lr=0.01x'),

    # Phase 3a: O + Wout combined (no model change needed)
    # attn.c_proj (O) uses standard 0.1× LR; mlp.c_proj (Wout) uses 0.02×
    dict(exp_name='hybrid_out', muon_layers='all', muon_matrices='attn.c_proj,mlp.c_proj',
         muon_mlp_lr_ratio=0.02, num_iterations=3000,
         description='Muon O+Wout'),

    # Phase 3b: paper's VO setting — V and O only, split_qkv=True required
    # Q and K stay under AdamW; only V and O go to Muon.
    dict(exp_name='hybrid_vo',     muon_layers='all', muon_matrices='attn.c_v,attn.c_proj',
         split_qkv=True, num_iterations=3000,
         description='Muon V+O (paper VO)'),

    # Phase 3b: paper's VO+FFN — V, O, Win, Wout all Muon; Q, K AdamW
    # MLP matrices use 0.02× LR to prevent GELU instability.
    dict(exp_name='hybrid_vo_ffn', muon_layers='all',
         muon_matrices='attn.c_v,attn.c_proj,mlp.c_fc,mlp.c_proj',
         split_qkv=True, muon_mlp_lr_ratio=0.02, num_iterations=3000,
         description='Muon V+O+FFN (paper)'),
]
