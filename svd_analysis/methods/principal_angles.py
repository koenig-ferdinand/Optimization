"""
Principal Angles Between Adjacent Layers — measures subspace drift.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .metrics import principal_angles, get_valid_layers, filter_weight_types

DESCRIPTION = "Principal angles between adjacent layers (subspace drift)"

PREFERRED = ["Q", "KV_b", "Down", "Up", "K", "QKV_fused", "O"]
COLORS = ["steelblue", "coral", "seagreen", "goldenrod"]


def run(models, output_dir, k_subspace=20):
    wt0 = filter_weight_types(models[0]["weight_types"])
    angle_types = [t for t in PREFERRED if t in wt0][:4]
    if not angle_types:
        angle_types = wt0[:4]

    n_models = len(models)

    fig, axes = plt.subplots(len(angle_types), n_models,
                             figsize=(7 * n_models, 4 * len(angle_types)),
                             squeeze=False)

    for col, md in enumerate(models):
        svd_data = md["svd_data"]
        n_layers = md["n_layers"]
        label = md["label"]
        color = COLORS[col % len(COLORS)]

        for row, wtype in enumerate(angle_types):
            ax = axes[row][col]
            if wtype not in md["weight_types"]:
                ax.text(0.5, 0.5, f"{wtype} not in {label}",
                        ha="center", transform=ax.transAxes)
                continue

            valid = get_valid_layers(svd_data, wtype, n_layers)
            mean_angles, min_angles, max_angles, layer_pairs = [], [], [], []

            for i in range(len(valid) - 1):
                l1, l2 = valid[i], valid[i + 1]
                Vh1 = svd_data[l1][wtype]["Vh"]
                Vh2 = svd_data[l2][wtype]["Vh"]
                if Vh1.shape[1] == Vh2.shape[1]:
                    angles = principal_angles(Vh1, Vh2, k=k_subspace)
                    mean_angles.append(np.mean(angles))
                    min_angles.append(np.min(angles))
                    max_angles.append(np.max(angles))
                    layer_pairs.append(l1)

            if layer_pairs:
                ax.fill_between(layer_pairs, min_angles, max_angles,
                                alpha=0.2, color=color)
                ax.plot(layer_pairs, mean_angles, "o-", color=color,
                        markersize=4, label="Mean angle")
                ax.axhline(y=90, color="red", linestyle="--", alpha=0.3)
                ax.axhline(y=0, color="green", linestyle="--", alpha=0.3)

            ax.set_xlabel("Layer pair (L → L+1)")
            ax.set_ylabel("Angle (°)")
            ax.set_title(f"{label} — {wtype}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-5, 100)

    fig.suptitle(f"Principal Angles Between Adjacent Layers (top-{k_subspace} SVs)\n"
                 f"(0° = same subspace, 90° = orthogonal)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "principal_angles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")
