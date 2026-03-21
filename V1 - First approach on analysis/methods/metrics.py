"""
Shared metric functions and plotting helpers.

Metric functions take a singular value tensor S (1D, sorted descending)
and return a scalar value.

Plotting helpers handle MoE vs dense model differences.
"""

import torch
import numpy as np
from scipy import stats


# ── Weight type classification ────────────────────────────────────────

ATTENTION_TYPES = {"Q", "K", "V", "O", "QKV_fused", "Q_a", "Q_b", "KV_a", "KV_b"}
MLP_TYPES = {"Gate", "Up", "Down"}
SHARED_TYPES = {"Shared_Gate", "Shared_Up", "Shared_Down"}
SKIP_TYPES = {"Router"}  # too small for meaningful spectral analysis


def group_weight_types(weight_types):
    """
    Split weight types into plot-friendly groups.
    Returns dict: {"Attention": [...], "MLP": [...], "Shared Expert": [...]}
    Filters out Router (too small).
    """
    groups = {}
    attn = [wt for wt in weight_types if wt in ATTENTION_TYPES]
    mlp = [wt for wt in weight_types if wt in MLP_TYPES]
    shared = [wt for wt in weight_types if wt in SHARED_TYPES]

    if attn:
        groups["Attention"] = attn
    if mlp:
        groups["MLP"] = mlp
    if shared:
        groups["Shared Expert"] = shared
    return groups


def filter_weight_types(weight_types):
    """Remove Router and other tiny weight types from analysis."""
    return [wt for wt in weight_types if wt not in SKIP_TYPES]


def get_valid_layers(svd_data, wtype, n_layers):
    """
    Get layer indices where a weight type has consistent shape.
    For MoE models, layer 0 often has a dense MLP with different shape
    than the expert representative in layers 1+. This filters to only
    layers with matching shapes.
    """
    shapes = {}
    for l in range(n_layers):
        if wtype in svd_data.get(l, {}):
            shape = svd_data[l][wtype]["shape"]
            shapes[l] = shape

    if not shapes:
        return []

    # Find the most common shape (skip outlier layers)
    from collections import Counter
    shape_counts = Counter(shapes.values())
    dominant_shape = shape_counts.most_common(1)[0][0]

    return [l for l, s in shapes.items() if s == dominant_shape]


def spectral_norm(S):
    """Largest singular value σ₁. Maximum amplification factor."""
    return S[0].item()


def stable_rank(S):
    """||W||²_F / ||W||²₂. Effective dimensionality (energy-based)."""
    return (S ** 2).sum().item() / (S[0] ** 2).item()


def effective_rank(S):
    """Shannon entropy-based rank (Roy & Vetterli, 2007)."""
    p = S / S.sum()
    p = p[p > 1e-10]
    entropy = -(p * torch.log(p)).sum().item()
    return np.exp(entropy)


def condition_number(S):
    """σ_max / σ_min. Numerical conditioning."""
    s_nonzero = S[S > 1e-10]
    if len(s_nonzero) == 0:
        return float("inf")
    return S[0].item() / s_nonzero[-1].item()


def nuclear_norm(S):
    """Sum of singular values (trace norm)."""
    return S.sum().item()


def frobenius_norm(S):
    """sqrt(sum of squared singular values)."""
    return torch.sqrt((S ** 2).sum()).item()


def cumulative_energy(S):
    """Cumulative fraction of total energy (σ²) as function of index."""
    energy = (S ** 2).numpy()
    cumsum = np.cumsum(energy)
    return cumsum / cumsum[-1]


def fit_power_law_tail(S, tail_fraction=0.9):
    """
    Martin & Mahoney (2021) Heavy-Tailed Self-Regularization.
    Fit a power law to the empirical spectral density.

    Returns:
        alpha: power-law tail exponent
        r_squared: R² of the fit
    """
    s = S.numpy()
    eigenvalues = s ** 2
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    n_tail = max(int(len(eigenvalues) * tail_fraction), 10)
    tail = np.sort(eigenvalues)[-n_tail:][::-1]  # descending

    log_vals = np.log10(tail)
    log_rank = np.log10(np.arange(1, len(tail) + 1))

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_vals, log_rank)
    alpha = -slope
    return alpha, r_value ** 2


def principal_angles(Vh1, Vh2, k=None):
    """
    Compute principal angles between two subspaces spanned by
    the top-k right singular vectors.

    Args:
        Vh1, Vh2: right singular vector matrices (rows = vectors)
        k: number of top vectors to compare

    Returns:
        angles in degrees (numpy array)
    """
    if k is None:
        k = min(50, Vh1.shape[0], Vh2.shape[0])

    V1_k = Vh1[:k, :]
    V2_k = Vh2[:k, :]

    M = V1_k @ V2_k.T
    _, cos_angles, _ = torch.linalg.svd(M)
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    angles = torch.acos(cos_angles) * 180 / np.pi
    return angles[:k].numpy()
