"""
reg_functions.py — Spectral Regularization Functions for GPT-2 Weight Matrices
================================================================================
Six regularizers, all operating on a single weight matrix W.
Each returns a scalar tensor (with grad) to be added to the task loss.

Dispatcher:  compute_reg_loss(model, reg_name, lam, target_matrices, target_layers)

Regularizers
------------
sv_variance    : Var(σᵢ / Σσᵢ)            — flatten spectrum → lower power-law α
orthogonal     : ||WᵀW - I||²_F / n       — push singular values toward 1
isometry       : ||WᵀW - σ̄²·I||²_F / n   — scale-free orthogonality
effective_rank : −H(σᵢ / Σσᵢ)             — maximise entropy of σ distribution
stable_rank    : −(Σσᵢ²) / σ₁²            — maximise stable rank
dead_sv        : Σ max(0, τ − σᵢ/σ₁)²    — penalise near-zero singular values
"""

import torch


# ── Individual regularizers ──────────────────────────────────────────────────

def sv_variance_penalty(W: torch.Tensor) -> torch.Tensor:
    """
    Penalise variance of the normalised singular-value distribution.
    Flattening the spectrum reduces the power-law tail exponent α toward 2–3.
    """
    S = torch.linalg.svdvals(W.float())
    S_norm = S / (S.sum() + 1e-8)
    return S_norm.var()


def orthogonal_penalty(W: torch.Tensor) -> torch.Tensor:
    """
    Penalise deviation of WᵀW (or WWᵀ) from the identity.
    Directly mimics Muon's orthogonality enforcement.
    Divided by n to keep scale independent of matrix dimension.
    """
    if W.shape[0] >= W.shape[1]:
        G = W.T @ W                                          # (n, n)
        I = torch.eye(G.shape[0], device=W.device, dtype=G.dtype)
    else:
        G = W @ W.T                                          # (m, m)
        I = torch.eye(G.shape[0], device=W.device, dtype=G.dtype)
    return (G - I).pow(2).mean()                             # mean = scale-invariant


def isometry_penalty(W: torch.Tensor) -> torch.Tensor:
    """
    Scale-free orthogonality: WᵀW ≈ σ̄²·I (allow uniform scaling, penalise distortion).
    Softer than orthogonal_penalty — does not fix the scale of singular values.
    """
    if W.shape[0] >= W.shape[1]:
        G = W.T @ W
    else:
        G = W @ W.T
    scale = G.diagonal().mean().detach()                     # σ̄² — no gradient through scale
    I = torch.eye(G.shape[0], device=W.device, dtype=G.dtype)
    return (G - scale * I).pow(2).mean()


def effective_rank_penalty(W: torch.Tensor) -> torch.Tensor:
    """
    Maximise entropy H(σᵢ/Σσᵢ) = effective rank (Roy & Vetterli 2007).
    Returns −H so that minimising this loss maximises effective rank.
    """
    S = torch.linalg.svdvals(W.float())
    p = S / (S.sum() + 1e-8)
    p = p.clamp(min=1e-10)
    H = -(p * p.log()).sum()
    return -H                                                # negative: we minimise


def stable_rank_penalty(W: torch.Tensor) -> torch.Tensor:
    """
    Maximise stable rank = (Σσᵢ²) / σ₁².
    Returns −stable_rank so that minimising this maximises stable rank.
    """
    S = torch.linalg.svdvals(W.float())
    return -(S.pow(2).sum() / (S[0].pow(2) + 1e-8))


def dead_sv_penalty(W: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    Penalise singular values below `threshold` (relative to σ₁).
    Hinge loss: penalty = Σ max(0, τ − σᵢ/σ₁)²
    Prevents rank collapse without forcing a specific spectral shape.
    """
    S = torch.linalg.svdvals(W.float())
    S_norm = S / (S[0].detach() + 1e-8)                     # normalise by σ₁, no grad
    deficit = torch.clamp(threshold - S_norm, min=0.0)
    return deficit.pow(2).sum()


# ── Registry ─────────────────────────────────────────────────────────────────

REGULARIZERS = {
    'sv_variance':    sv_variance_penalty,
    'orthogonal':     orthogonal_penalty,
    'isometry':       isometry_penalty,
    'effective_rank': effective_rank_penalty,
    'stable_rank':    stable_rank_penalty,
    'dead_sv':        dead_sv_penalty,
}


# ── Dispatcher ───────────────────────────────────────────────────────────────

def compute_reg_loss(
    model,
    reg_name: str,
    lam: float,
    target_matrices: list,          # e.g. ['mlp.c_proj', 'mlp.c_fc']
    target_layers,                  # 'all'  or  list[int]
) -> torch.Tensor:
    """
    Sum regularization loss over selected weight matrices and transformer layers.

    Normalised by the number of (layer, matrix) pairs so that λ is comparable
    across different numbers of target matrices.

    Args:
        model           : raw (unwrapped) GPT model  — model.module if using DDP
        reg_name        : key in REGULARIZERS dict
        lam             : scalar multiplier
        target_matrices : list of dotted attribute paths relative to a Block,
                          e.g. ['mlp.c_proj', 'mlp.c_fc', 'attn.c_proj']
        target_layers   : 'all'  or  list of layer indices

    Returns:
        scalar tensor with gradient
    """
    fn = REGULARIZERS[reg_name]
    layers = range(len(model.transformer.h)) if target_layers == 'all' else target_layers

    device = next(model.parameters()).device
    reg = torch.zeros(1, device=device, dtype=torch.float32)
    count = 0

    for i in layers:
        block = model.transformer.h[i]
        for mat_path in target_matrices:
            obj = block
            for attr in mat_path.split('.'):
                obj = getattr(obj, attr)
            W = obj.weight.float()                           # always compute in float32
            reg = reg + fn(W)
            count += 1

    if count == 0:
        return reg
    return lam * reg / count                                 # mean — scale-invariant w.r.t. count
