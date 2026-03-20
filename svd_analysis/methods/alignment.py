"""
Cross-Layer Singular Vector Alignment — heatmap showing which layers
share structural similarity.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import get_valid_layers, filter_weight_types

DESCRIPTION = "Cross-layer alignment heatmap (left & right singular vectors)"

PREFERRED = ["Q", "QKV_fused", "Down", "Up", "KV_b"]


def run(models, output_dir, k_align=10):
    wt0 = filter_weight_types(models[0]["weight_types"])
    align_type = next((t for t in PREFERRED if t in wt0), wt0[0])

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2,
                             figsize=(14, 6 * n_models),
                             squeeze=False)

    for row, md in enumerate(models):
        svd_data = md["svd_data"]
        n_layers = md["n_layers"]
        label = md["label"]

        valid = get_valid_layers(svd_data, align_type, n_layers)
        n_valid = len(valid)

        align_right = np.zeros((n_valid, n_valid))
        align_left = np.zeros((n_valid, n_valid))

        for i, li in enumerate(valid):
            for j, lj in enumerate(valid):
                Vh_i = svd_data[li][align_type]["Vh"][:k_align, :]
                Vh_j = svd_data[lj][align_type]["Vh"][:k_align, :]
                if Vh_i.shape[1] == Vh_j.shape[1]:
                    cos = torch.abs(Vh_i @ Vh_j.T)
                    align_right[i, j] = cos.mean().item()

                U_i = svd_data[li][align_type]["U"][:, :k_align]
                U_j = svd_data[lj][align_type]["U"][:, :k_align]
                if U_i.shape[0] == U_j.shape[0]:
                    cos = torch.abs(U_i.T @ U_j)
                    align_left[i, j] = cos.mean().item()

        tick_labels = [str(l) for l in valid]
        tick_step = max(1, n_valid // 10)

        sns.heatmap(align_right, ax=axes[row][0], cmap="magma",
                    vmin=0, vmax=1, square=True,
                    xticklabels=tick_step, yticklabels=tick_step)
        axes[row][0].set_title(f"{label} — Right SVs ({align_type})\n(input directions)")
        axes[row][0].set_xlabel("Layer")
        axes[row][0].set_ylabel("Layer")

        sns.heatmap(align_left, ax=axes[row][1], cmap="magma",
                    vmin=0, vmax=1, square=True,
                    xticklabels=tick_step, yticklabels=tick_step)
        axes[row][1].set_title(f"{label} — Left SVs ({align_type})\n(output directions)")
        axes[row][1].set_xlabel("Layer")
        axes[row][1].set_ylabel("Layer")

    fig.suptitle(f"Cross-Layer Singular Vector Alignment (top-{k_align}, {align_type})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "alignment.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")
