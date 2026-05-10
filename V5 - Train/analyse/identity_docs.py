# PACKAGES
import sys
import os

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # V5 - Train/analyse/
_V5_DIR       = os.path.dirname(_SCRIPT_DIR)                  # V5 - Train/
_PROJECT_ROOT = os.path.dirname(_V5_DIR)                      # project root
sys.path.insert(0, _V5_DIR)

import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import functions

# -------------------------------------------------------------------------------------------------

# SETUP:
# Analyses two orthogonal questions at the final training checkpoint (step 6200):
#
# 1. IDENTITY PROXIMITY — how close is each weight matrix to an identity transform?
#    Four complementary metrics per (optimizer, matrix type, layer):
#      sv_std        : std of singular values   → 0 = flat spectrum (identity-like)
#      sv_dev_from_1 : mean |σᵢ − 1|            → 0 = all SVs = 1 (exact identity)
#      uv_angle      : mean principal angle U↔V  → 0° = input/output subspaces aligned
#      frob_from_I   : ‖W − I‖_F / √n           → 0 = exact identity
#    uv_angle and frob_from_I are defined only for square matrices (Q,K,V, attn.c_proj).
#    Output: identity_proximity.png
#
# 2. EXTENDED DOCS — new cross-comparisons using inferno colormap (matching existing DOCS style)
#    a. U vs V within-matrix (per layer, square matrices only)
#       DOCS(U_k, V_k) — high value = input/output subspaces similar = identity-like behaviour
#       Output: docs_uv.png
#    b. Cross-model (Muon vs AdamW) — do they learn the same subspaces?
#       DOCS(V_muon_i, V_adamw_j) → 12×12 heatmap per matrix type
#       Output: docs_cross_model.png
#    c. Cross-matrix within attention / MLP
#       Per layer: DOCS(V_Q, V_K), DOCS(V_Q, V_V), DOCS(V_K, V_V) and DOCS(V_fc, V_proj)
#       (Right SVs for Q/K/V/attn.c_proj/mlp.c_fc all live in ℝ^768, making comparison valid.)
#       Output: docs_cross_matrix.png
#
# Run:  python "V5 - Train/analyse/identity_docs.py"
# -------------------------------------------------------------------------------------------------


# CONFIG
N_LAYERS     = 12
MATRIX_TYPES = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
SQUARE_MATS  = ['Q', 'K', 'V', 'attn.c_proj']   # 768×768 — identity metrics fully defined
OPTIMIZERS   = ['muon', 'adamw']
OPT_COLORS   = {'muon': '#4C72B0', 'adamw': '#C44E52'}
K_SUBSPACE   = 50

FINAL_STEP = 6200
DATA_PATH  = os.path.join(_PROJECT_ROOT, 'data')
CKPT       = {opt: os.path.join(DATA_PATH, opt, f'state_step{FINAL_STEP:06d}.pt')
              for opt in OPTIMIZERS}
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, 'plots')

DOCS_CMAP = plt.get_cmap('inferno').copy()   # matches existing DOCS style


# -------------------------------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------------------------------

def load_weights(opt):
    """Return dict mat_type → list[np.ndarray] (one per layer, shape d_out × d_in)."""
    print(f'  Loading {opt} checkpoint...')
    model = torch.load(CKPT[opt], map_location='cpu')['model']
    W = {mat: [] for mat in MATRIX_TYPES}
    for i in range(N_LAYERS):
        QKV = model[f'_orig_mod.transformer.h.{i}.attn.c_attn.weight']
        Q, K, V = QKV.split(768, dim=0)
        W['Q'].append(Q.float().numpy())
        W['K'].append(K.float().numpy())
        W['V'].append(V.float().numpy())
        for app in ['attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
            mat = model[f'_orig_mod.transformer.h.{i}.{app}.weight']
            W[app].append(mat.float().numpy())
    return W


def decompose(weights):
    """
    Compute SVD for each (mat_type, layer).
    Returns dicts keyed by mat_type, each a list of (U, S, Vh) per layer.
    """
    svd = {mat: [] for mat in MATRIX_TYPES}
    for mat in MATRIX_TYPES:
        for W in weights[mat]:
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            svd[mat].append((U, S, Vh))
    return svd


# -------------------------------------------------------------------------------------------------
# IDENTITY PROXIMITY METRICS
# -------------------------------------------------------------------------------------------------

def sv_std(S):
    return float(S.std())

def sv_dev_from_1(S):
    return float(np.abs(S - 1).mean())

def uv_angle(U, Vh, k=K_SUBSPACE):
    """Mean principal angle (degrees) between top-k left and right singular subspaces."""
    U_k  = U[:, :k]
    V_k  = Vh[:k].T          # Vh is (k×n) → V columns are rows of Vh
    # Both U_k and V_k must have the same number of rows (only valid for square matrices)
    if U_k.shape[0] != V_k.shape[0]:
        return float('nan')
    M        = U_k.T @ V_k   # (k×k)
    cos_vals = np.linalg.svd(M, compute_uv=False)
    cos_vals = np.clip(cos_vals, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_vals)).mean())

def frob_from_I(W):
    """‖W − I‖_F / √n; NaN for non-square."""
    m, n = W.shape
    if m != n:
        return float('nan')
    return float(np.linalg.norm(W - np.eye(n)) / np.sqrt(n))


IDENTITY_METRICS = ['sv_std', 'sv_dev_from_1', 'uv_angle', 'frob_from_I']
IDENTITY_LABELS  = {
    'sv_std':        'SV std (flat=0)',
    'sv_dev_from_1': 'Mean |σᵢ−1| (identity=0)',
    'uv_angle':      'U–V angle (°)',
    'frob_from_I':   '‖W−I‖_F/√n',
}


def compute_identity_proximity(svd_data, weights):
    """Returns result[mat][metric] = (N_LAYERS,) array."""
    result = {mat: {m: np.full(N_LAYERS, np.nan) for m in IDENTITY_METRICS}
              for mat in MATRIX_TYPES}
    for mat in MATRIX_TYPES:
        for layer in range(N_LAYERS):
            U, S, Vh = svd_data[mat][layer]
            W        = weights[mat][layer]
            result[mat]['sv_std'][layer]        = sv_std(S)
            result[mat]['sv_dev_from_1'][layer] = sv_dev_from_1(S)
            result[mat]['uv_angle'][layer]      = uv_angle(U, Vh)
            result[mat]['frob_from_I'][layer]   = frob_from_I(W)
    return result


# -------------------------------------------------------------------------------------------------
# IDENTITY PROXIMITY PLOT
# -------------------------------------------------------------------------------------------------

def plot_identity_proximity(prox):
    """
    One figure: 2 rows (optimizer) × 6 cols (matrix type).
    Each panel: heatmap rows=4 metrics, cols=12 layers.
    """
    n_metrics = len(IDENTITY_METRICS)
    fig, axes = plt.subplots(
        len(OPTIMIZERS), len(MATRIX_TYPES),
        figsize=(22, 6), dpi=150,
    )
    fig.suptitle('Identity Proximity — final checkpoint', fontsize=13, y=1.01)

    layer_ticks = list(range(N_LAYERS))

    for oi, opt in enumerate(OPTIMIZERS):
        for mi, mat in enumerate(MATRIX_TYPES):
            ax = axes[oi][mi]
            data = np.array([prox[opt][mat][m] for m in IDENTITY_METRICS])  # (4, 12)

            # Per-row normalisation so each metric uses its own scale
            # (all-NaN rows = uv_angle/frob_from_I for non-square matrices — silenced)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                row_min = np.nanmin(data, axis=1, keepdims=True)
                row_max = np.nanmax(data, axis=1, keepdims=True)
            normed  = (data - row_min) / np.where(row_max - row_min > 0,
                                                   row_max - row_min, 1)

            im = ax.imshow(normed, aspect='auto', cmap='plasma',
                           vmin=0, vmax=1, origin='upper')
            ax.set_xticks(layer_ticks)
            ax.set_xticklabels([str(l) for l in layer_ticks], fontsize=6)
            ax.set_yticks(range(n_metrics))
            ax.set_yticklabels([IDENTITY_LABELS[m] for m in IDENTITY_METRICS], fontsize=7)
            ax.set_xlabel('Layer', fontsize=7)
            if mi == 0:
                ax.set_ylabel(opt.upper(), fontsize=8)
            if oi == 0:
                ax.set_title(mat, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='(row-normed)')

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'identity_proximity.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


# -------------------------------------------------------------------------------------------------
# IDENTITY PROXIMITY — ABSOLUTE VALUES
# -------------------------------------------------------------------------------------------------

def plot_identity_proximity_absolute(prox):
    """
    One figure per metric (4 total).
    Each figure: 2 rows (optimizer) × 6 cols (matrix type).
    Each panel: line chart — x=layer (0-11), one line per matrix type.
    Both optimizers overlaid on the same axes using different line styles.

    Since showing all 6 matrix types × 2 optimizers on one axes would be crowded,
    we use 2 rows (one per optimizer) and plot all 6 matrix type lines per row.
    """
    layer_x = np.arange(N_LAYERS)
    mat_colors = plt.cm.tab10(np.linspace(0, 1, len(MATRIX_TYPES)))

    for metric in IDENTITY_METRICS:
        fig, axes = plt.subplots(
            1, len(OPTIMIZERS),
            figsize=(14, 4), dpi=150,
            sharey=False,
        )
        fig.suptitle(f'Identity Proximity (absolute) — {IDENTITY_LABELS[metric]}', fontsize=12)

        for oi, opt in enumerate(OPTIMIZERS):
            ax = axes[oi]
            for mi, mat in enumerate(MATRIX_TYPES):
                vals = prox[opt][mat][metric]   # (N_LAYERS,) — may contain NaN for non-square
                if np.all(np.isnan(vals)):
                    continue
                ax.plot(layer_x, vals,
                        marker='o', markersize=4,
                        linewidth=1.6,
                        color=mat_colors[mi],
                        label=mat)
            ax.set_xlabel('Layer', fontsize=9)
            ax.set_ylabel(IDENTITY_LABELS[metric], fontsize=9)
            ax.set_title(opt.upper(), fontsize=10)
            ax.set_xticks(layer_x)
            ax.grid(True, alpha=0.3)
            if oi == len(OPTIMIZERS) - 1:
                ax.legend(fontsize=8, loc='best')

        fig.tight_layout()
        safe = metric.replace(' ', '_')
        out  = os.path.join(OUTPUT_DIR, f'identity_proximity_abs_{safe}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  → {out}')


# -------------------------------------------------------------------------------------------------
# EXTENDED DOCS HELPERS
# -------------------------------------------------------------------------------------------------

def _right_svecs(Vh, k=K_SUBSPACE):
    """Top-k right singular vectors as columns: shape (n, k)."""
    return Vh[:k].T

def _left_svecs(U, k=K_SUBSPACE):
    """Top-k left singular vectors as columns: shape (m, k)."""
    return U[:, :k]


def _docs(A, B):
    """DOCS scalar between two column matrices A (d,a) and B (d,b). Requires same d."""
    return functions.DOCS(A, B)


def _docs_matrix_12x12(layers_A, layers_B, mask_diagonal=False):
    """
    12×12 DOCS matrix.
    mask_diagonal=True only for same-matrix comparisons (trivially 1 on diagonal).
    For cross-matrix comparisons the diagonal is the most informative cell and must be kept.
    """
    M = np.zeros((N_LAYERS, N_LAYERS))
    for i in range(N_LAYERS):
        for j in range(N_LAYERS):
            M[i, j] = _docs(layers_A[i], layers_B[j])
    if mask_diagonal:
        np.fill_diagonal(M, np.nan)
    return M


def _plot_12x12_grid(title, matrix_dict, out_name, row_labels=None, col_labels=None):
    """
    Generic 2×3 inferno heatmap grid for 12×12 matrices, one panel per matrix type.
    matrix_dict: mat_type → (12×12) np.ndarray with NaN on diagonal.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    fig.suptitle(title, fontsize=13)

    for idx, mat in enumerate(MATRIX_TYPES):
        ax  = axes[idx // 3][idx % 3]
        M   = matrix_dict[mat]
        vmin, vmax = np.nanmin(M), np.nanmax(M)

        im = ax.imshow(M, cmap=DOCS_CMAP, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks(range(N_LAYERS))
        ax.set_yticks(range(N_LAYERS))
        ax.set_xlabel(col_labels or 'Layer')
        ax.set_ylabel(row_labels or 'Layer')
        ax.set_title(mat)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


# -------------------------------------------------------------------------------------------------
# DOCS: U vs V (identity check)
# -------------------------------------------------------------------------------------------------

def plot_docs_uv(svd_all):
    """
    DOCS(U_k, V_k) per (optimizer, matrix type, layer).
    Square matrices only; non-square → skipped.
    One figure: bar charts, 2 rows (optimizers) × 4 cols (square mat types).
    Uses inferno coloring per optimizer for consistency.
    """
    sq = SQUARE_MATS
    fig, axes = plt.subplots(len(OPTIMIZERS), len(sq),
                             figsize=(14, 6), dpi=150, sharey=False)
    fig.suptitle('DOCS(U, V) — input/output subspace similarity\n'
                 '(high = identity-like; square matrices only)', fontsize=12)

    layer_x = np.arange(N_LAYERS)
    for oi, opt in enumerate(OPTIMIZERS):
        for mi, mat in enumerate(sq):
            ax = axes[oi][mi]
            vals = np.array([
                _docs(_left_svecs(svd_all[opt][mat][l][0]),
                      _right_svecs(svd_all[opt][mat][l][2]))
                for l in range(N_LAYERS)
            ])
            bars = ax.bar(layer_x, vals, color=OPT_COLORS[opt], edgecolor='white', linewidth=0.4)
            ax.set_xticks(layer_x)
            ax.set_xticklabels([str(l) for l in layer_x], fontsize=7)
            ax.set_xlabel('Layer', fontsize=8)
            ax.set_ylabel('DOCS(U,V)', fontsize=8)
            if oi == 0:
                ax.set_title(mat, fontsize=9)
            if mi == 0:
                ax.text(-0.25, 0.5, opt.upper(), transform=ax.transAxes,
                        fontsize=9, va='center', ha='right', rotation=90, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'docs_uv.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out}')


# -------------------------------------------------------------------------------------------------
# DOCS: Cross-model
# -------------------------------------------------------------------------------------------------

def plot_docs_cross_model(svd_all):
    """
    DOCS(V_muon_i, V_adamw_j) → 12×12 per matrix type.
    Rows = muon layer, cols = adamw layer.
    Diagonal (i==j) = same-layer cross-model similarity — shown, not masked.
    """
    print('  Computing cross-model DOCS 12×12...')
    mats = {}
    for mat in MATRIX_TYPES:
        muon_Vs  = [_right_svecs(svd_all['muon'][mat][l][2])  for l in range(N_LAYERS)]
        adamw_Vs = [_right_svecs(svd_all['adamw'][mat][l][2]) for l in range(N_LAYERS)]
        mats[mat] = _docs_matrix_12x12(muon_Vs, adamw_Vs, mask_diagonal=False)

    _plot_12x12_grid(
        'DOCS cross-model: V_muon (row) vs V_adamw (col)',
        mats,
        'docs_cross_model.png',
        row_labels='Muon layer',
        col_labels='AdamW layer',
    )


# -------------------------------------------------------------------------------------------------
# DOCS: Cross-matrix type
# -------------------------------------------------------------------------------------------------

def plot_docs_cross_matrix(svd_all):
    """
    Per optimizer: cross-matrix DOCS pairs at the same layer.
    Attention triples: (Q,K), (Q,V), (K,V).
    MLP pair: (mlp.c_fc, mlp.c_proj) — different ambient dims, uses left SVs for fc
    and right SVs for proj (both in ℝ^768) to keep dims compatible.

    Layout: one figure per optimizer, 4 subplots (one per pair), 12×12 heatmaps.
    """
    for opt in OPTIMIZERS:
        print(f'  Cross-matrix DOCS — {opt}...')
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=150)
        fig.suptitle(f'DOCS cross-matrix type — {opt.upper()}', fontsize=12)

        pair_defs = [
            ('Q',        'K',         'Q vs K'),
            ('Q',        'V',         'Q vs V'),
            ('K',        'V',         'K vs V'),
            ('mlp.c_fc', 'mlp.c_proj','mlp.c_fc (V) vs mlp.c_proj (U)'),
        ]

        for pi, (mat_a, mat_b, label) in enumerate(pair_defs):
            ax = axes[pi]

            # Choose SVecs that live in ℝ^768 for both matrices
            if mat_a == 'mlp.c_fc':
                # c_fc: (3072,768) → right SVs ∈ ℝ^768; c_proj: (768,3072) → left SVs ∈ ℝ^768
                A_vecs = [_right_svecs(svd_all[opt][mat_a][l][2]) for l in range(N_LAYERS)]
                B_vecs = [_left_svecs(svd_all[opt][mat_b][l][0])  for l in range(N_LAYERS)]
            else:
                A_vecs = [_right_svecs(svd_all[opt][mat_a][l][2]) for l in range(N_LAYERS)]
                B_vecs = [_right_svecs(svd_all[opt][mat_b][l][2]) for l in range(N_LAYERS)]

            M = _docs_matrix_12x12(A_vecs, B_vecs, mask_diagonal=False)
            vmin, vmax = M.min(), M.max()

            im = ax.imshow(M, cmap=DOCS_CMAP, vmin=vmin, vmax=vmax, origin='lower')
            ax.set_xticks(range(N_LAYERS))
            ax.set_yticks(range(N_LAYERS))
            ax.set_xlabel(f'{mat_b} layer', fontsize=8)
            ax.set_ylabel(f'{mat_a} layer', fontsize=8)
            ax.set_title(label, fontsize=9)
            fig.colorbar(im, ax=ax)

        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, f'docs_cross_matrix_{opt}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    → {out}')


# -------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading weights...')
    weights_all = {opt: load_weights(opt) for opt in OPTIMIZERS}

    print('Computing SVDs...')
    svd_all = {opt: decompose(weights_all[opt]) for opt in OPTIMIZERS}

    print('Computing identity proximity metrics...')
    prox = {opt: compute_identity_proximity(svd_all[opt], weights_all[opt])
            for opt in OPTIMIZERS}

    print('Plotting identity proximity (row-normalised)...')
    plot_identity_proximity(prox)

    print('Plotting identity proximity (absolute values)...')
    plot_identity_proximity_absolute(prox)

    print('Plotting DOCS U vs V...')
    plot_docs_uv(svd_all)

    print('Plotting DOCS cross-model...')
    plot_docs_cross_model(svd_all)

    print('Plotting DOCS cross-matrix...')
    plot_docs_cross_matrix(svd_all)

    print('Done.')
