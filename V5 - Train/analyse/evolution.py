# PACKAGES
import sys
import os

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # V5 - Train/analyse/
_V5_DIR       = os.path.dirname(_SCRIPT_DIR)                  # V5 - Train/
_PROJECT_ROOT = os.path.dirname(_V5_DIR)                      # project root
sys.path.insert(0, _V5_DIR)

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from multiprocessing import Pool, cpu_count
import time

# FILES
import functions
# -------------------------------------------------------------------------------------------------


# SETUP:
# Produces one tall stacked figure per matrix type (Q, K, V, attn.c_proj, mlp.c_fc, mlp.c_proj).
# Each figure contains (all sharing the same iteration x-axis):
#   Heatmap rows (Muon left | AdamW right, shared colorbar per metric):
#     effective_rank, stable_rank, rank_utilization, spectral_norm, nuclear_norm,
#     power-law alpha, MP signal fraction, update magnitude, effective step size,
#     subspace drift from init, cumulative energy (energy_k)
#   Full-width heatmap:
#     cross-optimizer subspace overlap (principal angles Muon vs AdamW)
#   Two identical line charts (one per column, for visual alignment):
#     validation loss
#
# New vs original:
#   + update_magnitude    : ||W_t - W_{t-100}||_F  (how much weights change per step)
#   + effective_step_size : ||ΔW||_F / ||W||_F      (normalised update size)
#   + subspace_drift      : mean principal angle between W_t and W_0 row-spaces
#   + mp_signal_fraction  : fraction of SVs above Marchenko-Pastur bulk edge
#   + cross_optimizer     : mean principal angle between Muon and AdamW at same step
#   + cumulative energy turned into heatmap (was line chart)
#   + validation loss duplicated into two columns for visual alignment
#
# Computation is parallelised across (optimizer, step) pairs via multiprocessing.Pool.
# W_0 singular vectors are pre-computed and injected into workers via pool initialiser.
# Run from any directory: python "V5 - Train/analyse/evolution.py"
# -------------------------------------------------------------------------------------------------


# CONFIG
ITERATIONS   = list(range(0, 6201, 100))   # 63 checkpoints: 0, 100, …, 6200
N_LAYERS     = 12
MATRIX_TYPES = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
OPTIMIZERS   = ['muon', 'adamw']
OPT_COLORS   = {'muon': '#4C72B0', 'adamw': '#C44E52'}
K_SUBSPACE   = 50   # top singular vectors used for subspace comparisons

DATA_PATH  = os.path.join(_PROJECT_ROOT, 'data')
LOG_PATHS  = {
    'muon':  os.path.join(_V5_DIR, 'log_muon.txt'),
    'adamw': os.path.join(_V5_DIR, 'log_adamw.txt'),
}
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'plots')

METRICS = [
    'effective_rank',
    'stable_rank',
    'rank_utilization',
    'spectral_norm',
    'nuclear_norm',
    'alpha',
    'mp_signal_fraction',
    'update_magnitude',
    'effective_step_size',
    'subspace_drift',
]
METRIC_LABELS = {
    'effective_rank':     'Effective Rank',
    'stable_rank':        'Stable Rank',
    'rank_utilization':   'Rank Utilization',
    'spectral_norm':      'Spectral Norm (σ₁)',
    'nuclear_norm':       'Nuclear Norm (Σσ)',
    'alpha':              'Power-law α',
    'mp_signal_fraction': 'MP Signal Fraction',
    'update_magnitude':   'Update Magnitude ‖ΔW‖_F',
    'effective_step_size':'Effective Step Size ‖ΔW‖/‖W‖',
    'subspace_drift':     'Subspace Drift from Init (°)',
}
METRIC_CMAPS = {
    'effective_rank':     'viridis',
    'stable_rank':        'viridis',
    'rank_utilization':   'plasma',
    'spectral_norm':      'inferno',
    'nuclear_norm':       'inferno',
    'alpha':              'RdYlGn',
    'mp_signal_fraction': 'viridis',
    'update_magnitude':   'hot_r',
    'effective_step_size':'hot_r',
    'subspace_drift':     'coolwarm',
}

# Worker-process global — set by pool initialiser before any tasks run
_INIT_VH = {}


def _init_worker(init_vh):
    global _INIT_VH
    _INIT_VH = init_vh


def _log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# -------------------------------------------------------------------------------------------------


def load_val_loss(path):
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            if 'val_loss' not in line or not line.startswith('step:'):
                continue
            parts = line.split()
            step  = int(parts[0].split(':')[1].split('/')[0])
            loss  = float(parts[1].split(':')[1])
            steps.append(step)
            losses.append(loss)
    return np.array(steps), np.array(losses)


def _mean_principal_angle(Vh1, Vh2):
    """Mean principal angle (degrees) between two k×n subspaces."""
    M        = Vh1 @ Vh2.T
    cos_vals = np.linalg.svd(M, compute_uv=False)
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_vals)).mean())


def _process_matrix(mat_curr, mat_prev, opt, mat_key, layer_i):
    """Compute all metrics for one weight matrix at one checkpoint."""
    # Single SVD — reused for all metrics
    _, S, Vh = torch.linalg.svd(mat_curr.float(), full_matrices=False)
    Vh_top   = Vh[:K_SUBSPACE].numpy()

    alpha, _ = functions.fit_power_law_tail(S)
    ek       = functions.energy_k(S, 0.9)

    # Subspace drift from initialisation (W_0 right singular vectors)
    vh_init = _INIT_VH.get((opt, mat_key, layer_i))
    drift   = _mean_principal_angle(Vh_top, vh_init) if vh_init is not None else float('nan')

    # Delta metrics vs previous checkpoint
    if mat_prev is not None:
        delta_frob = (mat_curr.float() - mat_prev.float()).norm().item()
        w_frob     = mat_curr.float().norm().item()
        upd_mag    = delta_frob
        eff_step   = delta_frob / w_frob if w_frob > 0 else float('nan')
    else:
        upd_mag  = float('nan')
        eff_step = float('nan')

    return {
        'effective_rank':     functions.effective_rank(S).item(),
        'stable_rank':        functions.stable_rank(S).item(),
        'rank_utilization':   functions.rank_utilization(S, mat_curr.shape),
        'spectral_norm':      functions.spectral_norm(S),
        'nuclear_norm':       functions.nuclear_norm(S),
        'energy_k':           ek if ek is not None else int(mat_curr.shape[0]),
        'alpha':              alpha,
        'mp_signal_fraction': functions.mp_signal_fraction(S, mat_curr.shape),
        'update_magnitude':   upd_mag,
        'effective_step_size':eff_step,
        'subspace_drift':     drift,
    }, Vh_top


def compute_step(args):
    """Worker: load current + previous checkpoints and return per-layer metrics."""
    opt, step = args
    torch.set_num_threads(1)

    model_curr = torch.load(
        os.path.join(DATA_PATH, opt, f'state_step{step:06d}.pt'), map_location='cpu'
    )['model']

    model_prev = None
    if step >= 100:
        model_prev = torch.load(
            os.path.join(DATA_PATH, opt, f'state_step{step-100:06d}.pt'), map_location='cpu'
        )['model']

    result   = {}   # (mat_type, layer) -> {metric: float}
    vh_store = {}   # (mat_type, layer) -> top-K right singular vectors (numpy)

    for i in range(N_LAYERS):

        # QKV — split into Q, K, V
        qkv_c       = model_curr[f'_orig_mod.transformer.h.{i}.attn.c_attn.weight']
        Qc, Kc, Vc  = qkv_c.split(768, dim=0)

        if model_prev is not None:
            qkv_p      = model_prev[f'_orig_mod.transformer.h.{i}.attn.c_attn.weight']
            Qp, Kp, Vp = qkv_p.split(768, dim=0)
        else:
            Qp = Kp = Vp = None

        for mat_key, mc, mp in zip(['Q', 'K', 'V'], [Qc, Kc, Vc], [Qp, Kp, Vp]):
            m, Vh                    = _process_matrix(mc, mp, opt, mat_key, i)
            result[(mat_key, i)]     = m
            vh_store[(mat_key, i)]   = Vh

        # Other projection matrices
        for appendix in ['attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
            key                          = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
            mc                           = model_curr[key]
            mp                           = model_prev[key] if model_prev is not None else None
            m, Vh                        = _process_matrix(mc, mp, opt, appendix, i)
            result[(appendix, i)]        = m
            vh_store[(appendix, i)]      = Vh

    return opt, step, result, vh_store


def build_arrays(raw_results):
    """Reorganise worker output; compute cross-optimizer subspace overlap."""
    iter_idx    = {step: j for j, step in enumerate(ITERATIONS)}
    n_iters     = len(ITERATIONS)
    scalar_keys = METRICS + ['energy_k']

    arrays = {
        opt: {mat: {m: np.full((N_LAYERS, n_iters), np.nan) for m in scalar_keys}
              for mat in MATRIX_TYPES}
        for opt in OPTIMIZERS
    }

    # Temporarily hold all Vh for cross-optimizer comparison
    vh_all = {}

    for opt, step, result, vh_store in raw_results:
        j = iter_idx[step]
        for (mat_type, layer), metrics in result.items():
            if mat_type not in arrays[opt]:
                continue
            for metric, value in metrics.items():
                if metric in arrays[opt][mat_type]:
                    arrays[opt][mat_type][metric][layer, j] = value
        for (mat_type, layer), Vh_top in vh_store.items():
            vh_all[(opt, step, mat_type, layer)] = Vh_top

    # Cross-optimizer subspace overlap: mean principal angle Muon vs AdamW
    cross_overlap = {mat: np.full((N_LAYERS, n_iters), np.nan) for mat in MATRIX_TYPES}
    n_steps = len(ITERATIONS)
    for idx, step in enumerate(ITERATIONS):
        j = iter_idx[step]
        for mat_type in MATRIX_TYPES:
            for layer in range(N_LAYERS):
                Vm = vh_all.get(('muon',  step, mat_type, layer))
                Va = vh_all.get(('adamw', step, mat_type, layer))
                if Vm is not None and Va is not None:
                    cross_overlap[mat_type][layer, j] = _mean_principal_angle(Vm, Va)
        if (idx + 1) % 10 == 0 or idx == n_steps - 1:
            _log(f'  cross-optimizer overlap: {idx + 1}/{n_steps} steps')

    return arrays, cross_overlap


def _add_heatmap(fig, gs, gs_row, col, data, title, cmap, vmin, vmax,
                 x_tick_pos, x_tick_labels, show_xticks, colspan=False):
    ax = fig.add_subplot(gs[gs_row, :] if colspan else gs[gs_row, col])
    im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower',
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_ylabel('Layer', fontsize=7)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels(range(N_LAYERS), fontsize=5)
    ax.set_xticks(x_tick_pos)
    if show_xticks:
        ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=6)
        ax.set_xlabel('Iteration', fontsize=7)
    else:
        ax.set_xticklabels([])
    fig.colorbar(im, ax=ax, fraction=0.023 if colspan else 0.046, pad=0.04)
    return ax


def plot_matrix_type(mat_type, arrays, cross_overlap, val_loss_data):
    n_iters   = len(ITERATIONS)
    n_hm_rows = len(METRICS) + 1   # 10 metrics + energy_k heatmap

    x_tick_pos    = list(range(0, n_iters, 10))
    x_tick_labels = [str(ITERATIONS[i]) for i in x_tick_pos]

    # GridSpec layout:
    #   rows 0 .. n_hm_rows-1   : per-optimizer heatmaps (2 cols: Muon | AdamW)
    #   row  n_hm_rows           : spacer
    #   row  n_hm_rows+1         : cross-optimizer heatmap (full width)
    #   row  n_hm_rows+2         : spacer
    #   row  n_hm_rows+3         : validation loss (2 cols, same data)
    height_ratios = [2.5] * n_hm_rows + [0.35, 3.0, 0.35, 2.5]
    n_rows = n_hm_rows + 4

    fig_h = n_hm_rows * 2.5 + 0.35 + 3.0 + 0.35 + 2.5 + 2.5   # +2.5 for suptitle padding
    fig = plt.figure(figsize=(22, fig_h))
    gs  = gridspec.GridSpec(n_rows, 2, figure=fig,
                            height_ratios=height_ratios, hspace=0.6, wspace=0.08)

    safe = mat_type.replace('.', '_')
    fig.suptitle(f'Weight Matrix Evolution — {mat_type}', fontsize=15, y=1.002)

    # ── Per-optimizer heatmap rows ────────────────────────────────────
    # Build list: (metric_key, muon_data, adamw_data, label, cmap)
    hm_rows = [(m, arrays['muon'][mat_type][m], arrays['adamw'][mat_type][m],
                METRIC_LABELS[m], METRIC_CMAPS[m]) for m in METRICS]
    # Cumulative energy as final heatmap row
    hm_rows.append(('energy_k',
                     arrays['muon'][mat_type]['energy_k'],
                     arrays['adamw'][mat_type]['energy_k'],
                     'Cumulative Energy (SVs for 90%)', 'magma'))

    for row, (_, d_m, d_a, label, cmap) in enumerate(hm_rows):
        vmin = np.nanmin([d_m, d_a])
        vmax = np.nanmax([d_m, d_a])
        _add_heatmap(fig, gs, row, 0, d_m, f'{label}  [MUON]',
                     cmap, vmin, vmax, x_tick_pos, x_tick_labels, True)
        _add_heatmap(fig, gs, row, 1, d_a, f'{label}  [ADAMW]',
                     cmap, vmin, vmax, x_tick_pos, x_tick_labels, True)

    # ── Cross-optimizer subspace overlap (full width) ─────────────────
    _add_heatmap(fig, gs, n_hm_rows + 1, 0, cross_overlap[mat_type],
                 'Cross-optimizer Subspace Overlap — mean principal angle  '
                 '(0° = identical subspace,  90° = orthogonal)',
                 'coolwarm_r', 0, 90, x_tick_pos, x_tick_labels,
                 show_xticks=True, colspan=True)

    # ── Validation loss — duplicated into two columns ─────────────────
    for col, opt in enumerate(OPTIMIZERS):
        ax_vl = fig.add_subplot(gs[n_hm_rows + 3, col])
        for o, (steps, losses) in val_loss_data.items():
            ax_vl.plot(steps, losses, label=o.upper(),
                       color=OPT_COLORS[o], linewidth=1.8)
        ax_vl.set_ylabel('Val Loss', fontsize=8)
        ax_vl.set_xlabel('Iteration', fontsize=8)
        ax_vl.set_title('Validation Loss', fontsize=9)
        ax_vl.legend(fontsize=8)
        ax_vl.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, f'evolution_{safe}.png')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved {out}')


if __name__ == '__main__':
    t_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _log('Loading validation losses...')
    val_loss_data = {opt: load_val_loss(LOG_PATHS[opt]) for opt in OPTIMIZERS}

    # Pre-compute W_0 right singular vectors (used for subspace_drift metric)
    _log('Pre-computing initialisation singular vectors...')
    init_vh = {}
    for opt in OPTIMIZERS:
        _log(f'  {opt}...')
        model0 = torch.load(
            os.path.join(DATA_PATH, opt, 'state_step000000.pt'), map_location='cpu'
        )['model']
        for i in range(N_LAYERS):
            qkv0        = model0[f'_orig_mod.transformer.h.{i}.attn.c_attn.weight']
            Q0, K0, V0  = qkv0.split(768, dim=0)
            for mat_key, mat0 in zip(['Q', 'K', 'V'], [Q0, K0, V0]):
                _, _, Vh0 = torch.linalg.svd(mat0.float(), full_matrices=False)
                init_vh[(opt, mat_key, i)] = Vh0[:K_SUBSPACE].numpy()
            for appendix in ['attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
                mat0 = model0[f'_orig_mod.transformer.h.{i}.{appendix}.weight']
                _, _, Vh0 = torch.linalg.svd(mat0.float(), full_matrices=False)
                init_vh[(opt, appendix, i)] = Vh0[:K_SUBSPACE].numpy()

    # Parallel metric computation
    jobs      = [(opt, step) for opt in OPTIMIZERS for step in ITERATIONS]
    n_jobs    = len(jobs)
    n_workers = min(cpu_count(), 8)
    _log(f'Computing {n_jobs} checkpoints using {n_workers} workers...')

    raw_results = []
    t_start = time.time()
    with Pool(n_workers, initializer=_init_worker, initargs=(init_vh,)) as pool:
        for result in pool.imap_unordered(compute_step, jobs):
            raw_results.append(result)
            done     = len(raw_results)
            elapsed  = time.time() - t_start
            rate     = done / elapsed
            eta      = (n_jobs - done) / rate if rate > 0 else 0
            pct      = done / n_jobs * 100
            _log(f'  [{done:3d}/{n_jobs}] ({pct:5.1f}%)  '
                 f'{result[0]:5s} step {result[1]:5d}  |  '
                 f'elapsed {elapsed:5.0f}s  eta ~{eta:4.0f}s')

    _log('Building arrays and computing cross-optimizer overlap...')
    arrays, cross_overlap = build_arrays(raw_results)
    del raw_results   # free ~1.4 GB of stored Vh data

    # ── Save cache for correlation.py ────────────────────────────────
    _log('Saving cache for correlation analysis...')
    _MAT_KEY = {
        'Q': 'Q', 'K': 'K', 'V': 'V',
        'attn.c_proj': 'attn_c_proj',
        'mlp.c_fc':    'mlp_c_fc',
        'mlp.c_proj':  'mlp_c_proj',
    }
    cache_dir = os.path.join(_SCRIPT_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    save = {'iterations': np.array(ITERATIONS)}
    for opt in OPTIMIZERS:
        s, l = val_loss_data[opt]
        save[f'valloss__{opt}__steps']  = np.array(s)
        save[f'valloss__{opt}__losses'] = np.array(l)
        for mat in MATRIX_TYPES:
            mk = _MAT_KEY[mat]
            for metric, arr in arrays[opt][mat].items():
                save[f'{opt}__{mk}__{metric}'] = arr
    for mat in MATRIX_TYPES:
        save[f'cross__{_MAT_KEY[mat]}'] = cross_overlap[mat]
    np.savez_compressed(os.path.join(cache_dir, 'arrays.npz'), **save)
    _log('  → Saved cache/arrays.npz')

    _log('Plotting...')
    for mat_type in MATRIX_TYPES:
        _log(f'  {mat_type}...')
        plot_matrix_type(mat_type, arrays, cross_overlap, val_loss_data)

    _log(f'Done — total time {time.time() - t_total:.0f}s')
