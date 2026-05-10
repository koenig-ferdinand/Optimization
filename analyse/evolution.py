# PACKAGES
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from multiprocessing import Pool, cpu_count

# FILES
import functions
# -------------------------------------------------------------------------------------------------


# SETUP:
# Produces one tall stacked figure per matrix type (Q, K, V, attn.c_proj, mlp.c_fc, mlp.c_proj).
# Each figure contains (all sharing the same iteration x-axis):
#   - 6 heatmap rows (effective_rank, stable_rank, rank_utilization,
#                     spectral_norm, nuclear_norm, power-law alpha)
#     Each row: Muon heatmap (left) | AdamW heatmap (right), shared colorbar scale.
#     x = iteration, y = layer, color = metric value.
#   - spacer
#   - cumulative energy row: dims needed for 90% energy, mean across layers (line chart)
#   - spacer
#   - validation loss row (line chart, both models)
#
# Metric computation is parallelized across (optimizer, step) pairs via multiprocessing.
# Run from project root: python analyse/evolution.py
# -------------------------------------------------------------------------------------------------


# CONFIG
ITERATIONS   = list(range(0, 6201, 100))   # 63 checkpoints: 0, 100, …, 6200
N_LAYERS     = 12
MATRIX_TYPES = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
OPTIMIZERS   = ['muon', 'adamw']
OPT_COLORS   = {'muon': '#4C72B0', 'adamw': '#C44E52'}

DATA_PATH = 'data'
LOG_PATHS = {
    'muon':  'V5 - Train/log_muon.txt',
    'adamw': 'V5 - Train/log_adamw.txt',
}
OUTPUT_DIR = 'analyse/plots'

METRICS = [
    'effective_rank',
    'stable_rank',
    'rank_utilization',
    'spectral_norm',
    'nuclear_norm',
    'alpha',
]
METRIC_LABELS = {
    'effective_rank':   'Effective Rank',
    'stable_rank':      'Stable Rank',
    'rank_utilization': 'Rank Utilization',
    'spectral_norm':    'Spectral Norm (σ₁)',
    'nuclear_norm':     'Nuclear Norm (Σσ)',
    'alpha':            'Power-law α',
}
METRIC_CMAPS = {
    'effective_rank':   'viridis',
    'stable_rank':      'viridis',
    'rank_utilization': 'plasma',
    'spectral_norm':    'inferno',
    'nuclear_norm':     'inferno',
    'alpha':            'RdYlGn',
}
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


def compute_step(args):
    """Worker: load one checkpoint and return per-layer metrics for all matrix types."""
    opt, step = args
    torch.set_num_threads(1)

    path  = f'{DATA_PATH}/{opt}/state_step{step:06d}.pt'
    data  = torch.load(path, map_location='cpu')
    model = data['model']

    result = {}  # (mat_type, layer_idx) -> {metric: float}

    for i in range(N_LAYERS):

        # QKV — split into Q, K, V
        qkv   = model[f'_orig_mod.transformer.h.{i}.attn.c_attn.weight']
        Q, K, V = qkv.split(768, dim=0)

        for mat_name, matrix in zip(['Q', 'K', 'V'], [Q, K, V]):
            S         = functions.svd(matrix)
            alpha, _  = functions.fit_power_law_tail(S)
            ek        = functions.energy_k(S, 0.9)
            result[(mat_name, i)] = {
                'effective_rank':   functions.effective_rank(S).item(),
                'stable_rank':      functions.stable_rank(S).item(),
                'rank_utilization': functions.rank_utilization(S, matrix.shape),
                'spectral_norm':    functions.spectral_norm(S),
                'nuclear_norm':     functions.nuclear_norm(S),
                'energy_k':         ek if ek is not None else int(matrix.shape[0]),
                'alpha':            alpha,
            }

        # Other projection matrices
        for appendix in ['attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
            matrix    = model[f'_orig_mod.transformer.h.{i}.{appendix}.weight']
            S         = functions.svd(matrix)
            alpha, _  = functions.fit_power_law_tail(S)
            ek        = functions.energy_k(S, 0.9)
            result[(appendix, i)] = {
                'effective_rank':   functions.effective_rank(S).item(),
                'stable_rank':      functions.stable_rank(S).item(),
                'rank_utilization': functions.rank_utilization(S, matrix.shape),
                'spectral_norm':    functions.spectral_norm(S),
                'nuclear_norm':     functions.nuclear_norm(S),
                'energy_k':         ek if ek is not None else int(matrix.shape[0]),
                'alpha':            alpha,
            }

    return opt, step, result


def build_arrays(raw_results):
    """Reshape worker output into arrays[opt][mat][metric] of shape (N_LAYERS, N_ITERS)."""
    iter_idx = {step: j for j, step in enumerate(ITERATIONS)}
    n_iters  = len(ITERATIONS)
    all_keys = METRICS + ['energy_k']

    arrays = {
        opt: {
            mat: {m: np.full((N_LAYERS, n_iters), np.nan) for m in all_keys}
            for mat in MATRIX_TYPES
        }
        for opt in OPTIMIZERS
    }

    for opt, step, result in raw_results:
        j = iter_idx[step]
        for (mat_type, layer), metrics in result.items():
            if mat_type not in arrays[opt]:
                continue
            for metric, value in metrics.items():
                if metric in arrays[opt][mat_type]:
                    arrays[opt][mat_type][metric][layer, j] = value

    return arrays


def plot_matrix_type(mat_type, arrays, val_loss_data):
    n_iters   = len(ITERATIONS)
    n_metrics = len(METRICS)

    # x-ticks: every 10 checkpoints = every 1000 iterations
    x_tick_pos    = list(range(0, n_iters, 10))
    x_tick_labels = [str(ITERATIONS[i]) for i in x_tick_pos]

    # GridSpec layout:
    #   rows 0 .. n_metrics-1   : heatmap rows (2 cols each: Muon | AdamW)
    #   row  n_metrics           : spacer
    #   row  n_metrics+1         : cumulative energy line chart (colspan 2)
    #   row  n_metrics+2         : spacer
    #   row  n_metrics+3         : validation loss line chart (colspan 2)
    height_ratios = [4] * n_metrics + [0.35, 2.8, 0.35, 2.8]
    n_rows = n_metrics + 4

    fig = plt.figure(figsize=(22, n_metrics * 4 + 14))
    gs  = gridspec.GridSpec(
        n_rows, 2,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.55,
        wspace=0.08,
    )

    safe_name = mat_type.replace('.', '_')
    fig.suptitle(f'Weight Matrix Evolution — {mat_type}', fontsize=15, y=1.002)

    # ── Heatmap rows ──────────────────────────────────────────────────
    for row, metric in enumerate(METRICS):
        d_muon  = arrays['muon'][mat_type][metric]    # (N_LAYERS, N_ITERS)
        d_adamw = arrays['adamw'][mat_type][metric]

        vmin = np.nanmin([d_muon, d_adamw])
        vmax = np.nanmax([d_muon, d_adamw])

        is_last_heatmap = (row == n_metrics - 1)

        for col, (opt, data) in enumerate([('muon', d_muon), ('adamw', d_adamw)]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(
                data,
                aspect='auto',
                cmap=METRIC_CMAPS[metric],
                origin='lower',
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest',
            )
            ax.set_title(f'{METRIC_LABELS[metric]}  [{opt.upper()}]', fontsize=9, pad=3)
            ax.set_ylabel('Layer', fontsize=7)
            ax.set_yticks(range(N_LAYERS))
            ax.set_yticklabels(range(N_LAYERS), fontsize=5)
            ax.set_xticks(x_tick_pos)

            if is_last_heatmap:
                ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=6)
                ax.set_xlabel('Iteration', fontsize=7)
            else:
                ax.set_xticklabels([])

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Cumulative energy row (n_metrics + 1) ────────────────────────
    ax_en = fig.add_subplot(gs[n_metrics + 1, :])
    for opt in OPTIMIZERS:
        ek_arr  = arrays[opt][mat_type]['energy_k']   # (N_LAYERS, N_ITERS)
        ek_mean = np.nanmean(ek_arr, axis=0)          # (N_ITERS,)
        ax_en.plot(ITERATIONS, ek_mean, label=opt.upper(),
                   color=OPT_COLORS[opt], linewidth=1.8)
    ax_en.set_ylabel('SVs for 90% Energy\n(mean over layers)', fontsize=8)
    ax_en.set_xlabel('Iteration', fontsize=8)
    ax_en.set_title('Cumulative Energy — dims needed for 90% energy (mean across layers)', fontsize=9)
    ax_en.legend(fontsize=8)
    ax_en.grid(True, alpha=0.3)

    # ── Validation loss row (n_metrics + 3) ──────────────────────────
    ax_vl = fig.add_subplot(gs[n_metrics + 3, :])
    for opt, (steps, losses) in val_loss_data.items():
        ax_vl.plot(steps, losses, label=opt.upper(),
                   color=OPT_COLORS[opt], linewidth=1.8)
    ax_vl.set_ylabel('Val Loss', fontsize=8)
    ax_vl.set_xlabel('Iteration', fontsize=8)
    ax_vl.set_title('Validation Loss', fontsize=9)
    ax_vl.legend(fontsize=8)
    ax_vl.grid(True, alpha=0.3)

    out = os.path.join(OUTPUT_DIR, f'evolution_{safe_name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved {out}')


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load validation losses
    print('Loading validation losses...')
    val_loss_data = {opt: load_val_loss(LOG_PATHS[opt]) for opt in OPTIMIZERS}

    # Parallel metric computation across all (optimizer, step) pairs
    jobs      = [(opt, step) for opt in OPTIMIZERS for step in ITERATIONS]
    n_workers = min(cpu_count(), 8)
    print(f'Computing {len(jobs)} checkpoints using {n_workers} workers...')

    with Pool(n_workers) as pool:
        raw_results = pool.map(compute_step, jobs)

    print('Building data arrays...')
    arrays = build_arrays(raw_results)

    print('Plotting...')
    for mat_type in MATRIX_TYPES:
        print(f'  {mat_type}...')
        plot_matrix_type(mat_type, arrays, val_loss_data)

    print('Done.')
