# PACKAGES
import sys
import os

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # V5 - Train/analyse/
_V5_DIR       = os.path.dirname(_SCRIPT_DIR)                  # V5 - Train/
_PROJECT_ROOT = os.path.dirname(_V5_DIR)                      # project root
sys.path.insert(0, _V5_DIR)

import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time

import functions

# -------------------------------------------------------------------------------------------------

# SETUP:
# Tracks how cross-matrix subspace similarity (DOCS) evolves across training iterations.
# For each checkpoint (0, 100, …, 6200) and each optimizer, computes:
#
#   DOCS(V_matA_layer_l, V_matB_layer_l)  — same-layer, cross-matrix comparison
#
# Matrix pairs analysed (all right-singular vectors live in ℝ^768, keeping dims compatible):
#   Q   vs K
#   Q   vs V
#   K   vs V
#   mlp.c_fc (V, in ℝ^768) vs mlp.c_proj (U, in ℝ^768)
#
# Output: one tall figure  cross_matrix_evolution.png
#   4 rows (pairs) × 2 cols (Muon | AdamW)
#   Each panel: heatmap x=iteration, y=layer, heat=DOCS value
#   + one summary line-chart row (mean DOCS across layers per pair)
#
# Parallelised across (optimizer, step) pairs via multiprocessing.Pool.
# Run:  python "V5 - Train/analyse/cross_matrix_evolution.py"
# -------------------------------------------------------------------------------------------------


# CONFIG
ITERATIONS   = list(range(0, 6201, 100))
N_LAYERS     = 12
OPTIMIZERS   = ['muon', 'adamw']
OPT_COLORS   = {'muon': '#4C72B0', 'adamw': '#C44E52'}
K_SUBSPACE   = 50

DATA_PATH  = os.path.join(_PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'plots')

# (name, mat_a, mat_b, use_left_for_b)
# use_left_for_b: mlp.c_proj is (768×3072) → left SVs are in ℝ^768
PAIRS = [
    ('Q vs K',              'Q',         'K',          False),
    ('Q vs V',              'Q',         'V',          False),
    ('K vs V',              'K',         'V',          False),
    ('mlp.c_fc vs c_proj',  'mlp.c_fc',  'mlp.c_proj', True),
]
def _log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# -------------------------------------------------------------------------------------------------
# WORKER
# -------------------------------------------------------------------------------------------------

def compute_step(args):
    """
    Worker: load one checkpoint, compute DOCS for each (pair, layer).
    Returns (opt, step, results) where results[pair_idx][layer] = DOCS scalar.
    """
    opt, step = args
    torch.set_num_threads(1)

    model = torch.load(
        os.path.join(DATA_PATH, opt, f'state_step{step:06d}.pt'), map_location='cpu'
    )['model']

    # Collect right/left singular vectors per matrix per layer
    Vh = {mat: [] for mat in ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']}
    U  = {mat: [] for mat in ['mlp.c_proj']}   # only needed for c_proj (left SVs in ℝ^768)

    for i in range(N_LAYERS):
        qkv = model[f'_orig_mod.transformer.h.{i}.attn.c_attn.weight'].float()
        Q, K, V = qkv.split(768, dim=0)
        for mat_key, mat in zip(['Q', 'K', 'V'], [Q, K, V]):
            _, _, vh = torch.linalg.svd(mat, full_matrices=False)
            Vh[mat_key].append(vh[:K_SUBSPACE].numpy().T)   # (768, k)

        for app in ['attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
            w = model[f'_orig_mod.transformer.h.{i}.{app}.weight'].float()
            u, _, vh = torch.linalg.svd(w, full_matrices=False)
            Vh[app].append(vh[:K_SUBSPACE].numpy().T)        # (n, k)
            if app == 'mlp.c_proj':
                U[app].append(u[:, :K_SUBSPACE].numpy())     # (768, k)

    # Compute DOCS for each pair and layer
    results = {}
    for pi, (label, mat_a, mat_b, use_left_b) in enumerate(PAIRS):
        layer_vals = []
        for i in range(N_LAYERS):
            A = Vh[mat_a][i]                                  # (d, k)
            B = U[mat_b][i] if use_left_b else Vh[mat_b][i]  # (d, k)
            layer_vals.append(functions.DOCS(A, B))
        results[pi] = layer_vals

    return opt, step, results


# -------------------------------------------------------------------------------------------------
# BUILD ARRAYS
# -------------------------------------------------------------------------------------------------

def build_arrays(raw_results):
    """
    raw_results: list of (opt, step, results)
    Returns arrays[pair_idx][opt] = (N_LAYERS, N_ITERS) ndarray.
    """
    iter_idx = {step: j for j, step in enumerate(ITERATIONS)}
    n_iters  = len(ITERATIONS)
    n_pairs  = len(PAIRS)

    arrays = {
        pi: {opt: np.full((N_LAYERS, n_iters), np.nan) for opt in OPTIMIZERS}
        for pi in range(n_pairs)
    }

    for opt, step, results in raw_results:
        j = iter_idx[step]
        for pi, layer_vals in results.items():
            for layer, val in enumerate(layer_vals):
                arrays[pi][opt][layer, j] = val

    return arrays


# -------------------------------------------------------------------------------------------------
# PLOT
# -------------------------------------------------------------------------------------------------

def plot(arrays):
    n_pairs = len(PAIRS)
    n_opts  = len(OPTIMIZERS)

    # Layout: n_pairs heatmap rows + 1 summary line row, × n_opts cols
    # Heights: heatmap rows are taller, summary row is shorter
    row_heights = [3.0] * n_pairs + [2.5]
    fig_height  = sum(row_heights) * 1.15

    fig, axes = plt.subplots(
        n_pairs + 1, n_opts,
        figsize=(14, fig_height),
        dpi=150,
        gridspec_kw={'height_ratios': row_heights},
    )
    fig.suptitle('Cross-matrix DOCS evolution over training\n'
                 'DOCS(V_matA_layer, V_matB_layer)  —  same layer, across training',
                 fontsize=12, y=1.01)

    iters_arr = np.array(ITERATIONS)

    for pi, (label, *_) in enumerate(PAIRS):
        for oi, opt in enumerate(OPTIMIZERS):
            ax  = axes[pi][oi]
            mat = arrays[pi][opt]          # (N_LAYERS, N_ITERS)

            vmin, vmax = np.nanmin(mat), np.nanmax(mat)
            im = ax.imshow(
                mat, aspect='auto', cmap='inferno',
                vmin=vmin, vmax=vmax,
                origin='upper',
                extent=[ITERATIONS[0], ITERATIONS[-1], N_LAYERS - 0.5, -0.5],
            )
            ax.set_yticks(range(N_LAYERS))
            ax.set_yticklabels([str(l) for l in range(N_LAYERS)], fontsize=6)
            ax.set_ylabel('Layer', fontsize=7)
            if pi == n_pairs - 1:
                ax.set_xlabel('Iteration', fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
            if pi == 0:
                ax.set_title(opt.upper(), fontsize=10)
            if oi == 0:
                ax.text(-0.18, 0.5, label, transform=ax.transAxes,
                        fontsize=8, va='center', ha='right', rotation=90, fontweight='bold')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='DOCS')

    # Summary row: mean DOCS across layers
    for oi, opt in enumerate(OPTIMIZERS):
        ax = axes[n_pairs][oi]
        for pi, (label, *_) in enumerate(PAIRS):
            mean_docs = np.nanmean(arrays[pi][opt], axis=0)   # (N_ITERS,)
            ax.plot(iters_arr, mean_docs, linewidth=1.4, label=label)
        ax.set_xlabel('Iteration', fontsize=8)
        ax.set_ylabel('Mean DOCS\n(across layers)', fontsize=7)
        ax.set_title(f'Summary — {opt.upper()}', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'cross_matrix_evolution.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    _log(f'  → Saved {out}')


# -------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jobs      = [(opt, step) for opt in OPTIMIZERS for step in ITERATIONS]
    n_jobs    = len(jobs)
    n_workers = min(cpu_count(), 8)
    _log(f'Computing {n_jobs} checkpoints using {n_workers} workers...')

    raw_results = []
    t_start = time.time()
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(compute_step, jobs):
            raw_results.append(result)
            done    = len(raw_results)
            elapsed = time.time() - t_start
            rate    = done / elapsed
            eta     = (n_jobs - done) / rate if rate > 0 else 0
            pct     = done / n_jobs * 100
            _log(f'  [{done:3d}/{n_jobs}] ({pct:5.1f}%)  '
                 f'{result[0]:5s} step {result[1]:5d}  |  '
                 f'elapsed {elapsed:4.0f}s  eta ~{eta:4.0f}s')

    _log('Building arrays...')
    arrays = build_arrays(raw_results)

    _log('Plotting...')
    plot(arrays)

    _log(f'Done — total time {time.time() - t_start:.0f}s')
