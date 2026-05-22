# PACKAGES
import sys
import os

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # V5 - Train/analyse/
_V5_DIR       = os.path.dirname(_SCRIPT_DIR)                  # V5 - Train/
_PROJECT_ROOT = os.path.dirname(_V5_DIR)                      # project root
sys.path.insert(0, _V5_DIR)

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------------------------------------------------------------------------------

# SETUP:
# Reads the cache produced by evolution.py (cache/arrays.npz) and computes three types of
# correlation between each metric (per layer) and the validation loss:
#
#   1. Pearson r          — raw correlation (may be inflated by shared monotone trend)
#   2. Partial r          — Pearson r controlling for iteration index (removes monotone confound)
#   3. First-diff r       — correlation of Δmetric vs Δloss (robust to level effects)
#
# Output per matrix type (Q, K, V, attn.c_proj, mlp.c_fc, mlp.c_proj):
#   • Correlation heatmap : 3 corr types × 2 optimizers, rows = metrics, cols = layers
#   • Top-16 scatter plots: sorted by |partial r|, both optimizers shown
#   • Summary bar chart   : mean |partial r| per metric, averaged over layers and matrix types
#
# Run:  python "V5 - Train/analyse/correlation.py"
# Pre-requisite: run evolution.py first to generate cache/arrays.npz
# -------------------------------------------------------------------------------------------------


CACHE_PATH = os.path.join(_SCRIPT_DIR, 'cache', 'arrays.npz')
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'plots')

OPTIMIZERS   = ['muon', 'adamw']
OPT_COLORS   = {'muon': '#4C72B0', 'adamw': '#C44E52'}
MATRIX_TYPES = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
START_ITER = 400   # skip warmup iterations with flat loss and noisy metrics

KEY_TO_MAT = {
    'Q':           'Q',
    'K':           'K',
    'V':           'V',
    'attn_c_proj': 'attn.c_proj',
    'mlp_c_fc':    'mlp.c_fc',
    'mlp_c_proj':  'mlp.c_proj',
}

ALL_METRICS = [
    'effective_rank',
    'stable_rank',
    'rank_utilization',
    'spectral_norm',
    'nuclear_norm',
    'alpha',
    'alpha_clauset',
    'energy_k',
]

METRIC_LABELS = {
    'effective_rank':     'Effective Rank',
    'stable_rank':        'Stable Rank',
    'rank_utilization':   'Rank Utilization',
    'spectral_norm':      'Spectral Norm',
    'nuclear_norm':       'Nuclear Norm',
    'alpha':              'Power-law α (regression)',
    'alpha_clauset':      'Power-law α (Clauset MLE)',
    'energy_k':           'Cumulative Energy (k)',
}

CORR_TYPES  = ['pearson', 'partial', 'firstdiff']
CORR_LABELS = {
    'pearson':   'Pearson r',
    'partial':   'Partial r (ctrl iter)',
    'firstdiff': 'First-diff r (Δmetric vs Δloss)',
}

N_LAYERS = 12


# -------------------------------------------------------------------------------------------------


def load_cache(path):
    """Load arrays.npz and reconstruct nested dicts."""
    raw = np.load(path)
    iterations = raw['iterations']

    val_loss = {}
    for opt in OPTIMIZERS:
        val_loss[opt] = (raw[f'valloss__{opt}__steps'], raw[f'valloss__{opt}__losses'])

    arrays = {opt: {mat: {} for mat in MATRIX_TYPES} for opt in OPTIMIZERS}
    cross_overlap = {}

    for key in raw.files:
        if key.startswith('valloss__') or key == 'iterations':
            continue
        parts = key.split('__')
        if parts[0] == 'cross':
            cross_overlap[KEY_TO_MAT[parts[1]]] = raw[key]
        elif len(parts) == 3:
            opt, mat_key, metric = parts
            mat = KEY_TO_MAT.get(mat_key, mat_key)
            if opt in OPTIMIZERS and mat in MATRIX_TYPES:
                arrays[opt][mat][metric] = raw[key]

    # cross_overlap[mat] = (N_LAYERS, N_ITERS) mean principal angle Muon vs AdamW
    # loaded for completeness; not used in correlation analysis (no single-optimizer ground truth)
    return iterations, val_loss, arrays, cross_overlap


def _align_loss(val_steps, val_losses, iterations):
    """Interpolate val_loss values onto the given iteration indices."""
    return np.interp(iterations, val_steps, val_losses)


# ----- correlation primitives -----

def _pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float('nan')
    r, _ = stats.pearsonr(x[mask], y[mask])
    return r


def _partial_r(x, y, z):
    """Partial correlation of x and y controlling for z (iteration index)."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 5:
        return float('nan')
    xm, ym, zm = x[mask], y[mask], z[mask]
    # residualise both x and y on z
    def _resid(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    rx = _resid(xm, zm)
    ry = _resid(ym, zm)
    r, _ = stats.pearsonr(rx, ry)
    return r


def _firstdiff_r(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 6:
        return float('nan')
    dx = np.diff(x)
    dy = np.diff(y)
    m2 = np.isfinite(dx) & np.isfinite(dy)
    if m2.sum() < 5:
        return float('nan')
    r, _ = stats.pearsonr(dx[m2], dy[m2])
    return r


# -------------------------------------------------------------------------------------------------


def compute_correlations(iterations, val_loss, arrays):
    """
    Returns corr[opt][mat][corr_type] = (N_METRICS, N_LAYERS) array of correlation values.
    """
    iter_arr = np.array(iterations, dtype=float)
    keep = iter_arr >= START_ITER
    iter_arr = iter_arr[keep]

    corr = {
        opt: {mat: {ct: np.full((len(ALL_METRICS), N_LAYERS), np.nan)
                    for ct in CORR_TYPES}
              for mat in MATRIX_TYPES}
        for opt in OPTIMIZERS
    }

    for opt in OPTIMIZERS:
        loss_aligned = _align_loss(*val_loss[opt], iterations)[keep]

        for mat in MATRIX_TYPES:
            for mi, metric in enumerate(ALL_METRICS):
                arr = arrays[opt][mat].get(metric)  # (N_LAYERS, N_ITERS)
                if arr is None:
                    continue
                for layer in range(N_LAYERS):
                    x = arr[layer][keep].astype(float)
                    y = loss_aligned

                    corr[opt][mat]['pearson'][mi, layer]   = _pearson(x, y)
                    corr[opt][mat]['partial'][mi, layer]   = _partial_r(x, y, iter_arr)
                    corr[opt][mat]['firstdiff'][mi, layer] = _firstdiff_r(x, y)

    return corr


# -------------------------------------------------------------------------------------------------


def plot_correlation_heatmaps(corr, mat_type):
    """
    One figure per matrix type: 3 corr types × 2 optimizers grid of heatmaps.
    Rows = metrics, columns = layers, colour = correlation.
    """
    n_metrics = len(ALL_METRICS)
    n_ct      = len(CORR_TYPES)
    n_opt     = len(OPTIMIZERS)

    fig, axes = plt.subplots(
        n_ct, n_opt,
        figsize=(12, 2.2 * n_ct),
        dpi=150,
        squeeze=False,
    )
    fig.suptitle(f'Metric ↔ Val-Loss Correlations — {mat_type}', fontsize=13, y=1.01)

    layer_ticks = np.arange(N_LAYERS)

    for ci, ct in enumerate(CORR_TYPES):
        for oi, opt in enumerate(OPTIMIZERS):
            ax  = axes[ci][oi]
            mat = corr[opt][mat_type][ct]   # (N_METRICS, N_LAYERS)

            im = ax.imshow(
                mat, aspect='auto', cmap='RdBu_r',
                vmin=-1, vmax=1,
                origin='upper',
            )
            ax.set_xticks(layer_ticks)
            ax.set_xticklabels([str(l) for l in layer_ticks], fontsize=6)
            ax.set_yticks(range(n_metrics))
            ax.set_yticklabels([METRIC_LABELS[m] for m in ALL_METRICS], fontsize=7)
            ax.set_xlabel('Layer', fontsize=8)
            ax.set_title(f'{CORR_LABELS[ct]} — {opt.upper()}', fontsize=9)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    safe = mat_type.replace('.', '_').replace(' ', '_')
    out  = os.path.join(OUTPUT_DIR, f'correlation_heatmap_without_prewarm_{safe}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


# -------------------------------------------------------------------------------------------------


def plot_scatter_top_pairs(corr, iterations, val_loss, arrays, n_top=16):
    """
    Scatter plots for the top-n_top (metric, mat, layer) pairs by mean |partial r|
    averaged across optimizers.  Both optimizers shown on each panel.
    """
    # Rank all (metric, mat, layer) triples by mean |partial r| across optimizers
    EXCLUDE = {'subspace_drift','update_magnitude','effective_step_size','rank_utilization'}  # noisy metrics with many nans, which dominate top ranks by |r|
    scores = []
    for mat in MATRIX_TYPES:
        for mi, metric in enumerate(ALL_METRICS):
            if metric in EXCLUDE:
                continue
            for layer in range(N_LAYERS):
                vals = [abs(corr[opt][mat]['partial'][mi, layer]) for opt in OPTIMIZERS]
                vals = [v for v in vals if np.isfinite(v)]
                if not vals:
                    continue
                scores.append((np.mean(vals), metric, mat, layer, mi))

    scores.sort(reverse=True)
    top = scores[:n_top]

    ncols = 4
    nrows = int(np.ceil(n_top / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), dpi=150)
    axes = axes.flatten()
    fig.suptitle(f'Top-{n_top} Metric ↔ Val-Loss Scatter Plots (by |partial r|)', fontsize=13)

    iter_arr = np.array(iterations, dtype=float)
    keep = iter_arr >= START_ITER
    for idx, (score, metric, mat, layer, mi) in enumerate(top):
        ax = axes[idx]
        for opt in OPTIMIZERS:
            arr  = arrays[opt][mat].get(metric)
            if arr is None:
                continue
            
            x    = arr[layer][keep].astype(float)
            y    = _align_loss(*val_loss[opt], iterations)[keep]
            mask = np.isfinite(x) & np.isfinite(y)
            pr   = corr[opt][mat]['partial'][mi, layer]
            ax.scatter(x[mask], y[mask], s=10, alpha=0.6,
                       color=OPT_COLORS[opt], label=f'{opt.upper()} (r={pr:.2f})')
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=7)
        ax.set_ylabel('Val Loss', fontsize=7)
        ax.set_title(f'{mat} L{layer}\n{METRIC_LABELS[metric]}', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    for idx in range(len(top), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'correlation_top_scatter_without_prewarm.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


# -------------------------------------------------------------------------------------------------


def plot_metric_summary(corr):
    """
    Bar chart: mean |partial r| per metric, averaged over layers, matrix types, and optimizers.
    Helps identify which metrics are most informative about val loss.
    """
    means = {}
    for mi, metric in enumerate(ALL_METRICS):
        vals = []
        for opt in OPTIMIZERS:
            for mat in MATRIX_TYPES:
                row = corr[opt][mat]['partial'][mi, :]   # (N_LAYERS,)
                vals.extend(np.abs(row[np.isfinite(row)]).tolist())
        means[metric] = np.mean(vals) if vals else 0.0

    labels = [METRIC_LABELS[m] for m in ALL_METRICS]
    values = [means[m] for m in ALL_METRICS]
    order  = np.argsort(values)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.barh(
        [labels[i] for i in order],
        [values[i] for i in order],
        color='steelblue', edgecolor='white', linewidth=0.5,
    )
    ax.set_xlabel('Mean |Partial r| with Val Loss', fontsize=10)
    ax.set_title('Metric–ValLoss Correlation Summary\n(partial r, averaged over layers & matrix types)',
                 fontsize=11)
    ax.set_xlim(0, 1)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'correlation_summary_without_prewarm2.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


# -------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # pearsonr returns nan when one input is constant — expected for flat metrics
    warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
    if not os.path.exists(CACHE_PATH):
        print(f'Cache not found: {CACHE_PATH}')
        print('Run evolution.py first to generate the cache.')
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading cache...')
    iterations, val_loss, arrays, cross_overlap = load_cache(CACHE_PATH)

    print('Computing correlations...')
    corr = compute_correlations(iterations, val_loss, arrays)

    print('Plotting correlation heatmaps...')
    for mat in MATRIX_TYPES:
        print(f'  {mat}...')
        plot_correlation_heatmaps(corr, mat)

    print('Plotting top scatter pairs...')
    plot_scatter_top_pairs(corr, iterations, val_loss, arrays)

    print('Plotting metric summary...')
    plot_metric_summary(corr)

    print('Done.')
