"""
Singular Value Spectra — plot all singular values per layer on log scale.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .metrics import get_valid_layers

DESCRIPTION = "Singular value spectra (log scale, all layers overlaid)"

# Prefer one attention type + one MLP type
PREFERRED_ATTN = ["Q", "KV_b", "K", "O", "QKV_fused", "KV_a"]
PREFERRED_MLP = ["Down", "Gate", "Up", "Shared_Down"]


def run(models, output_dir):
    wt0 = models[0]["weight_types"]
    pick_a = next((t for t in PREFERRED_ATTN if t in wt0), None)
    pick_m = next((t for t in PREFERRED_MLP if t in wt0), None)
    focus_types = [t for t in [pick_a, pick_m] if t is not None]
    if not focus_types:
        focus_types = wt0[:2]

    n_models = len(models)
    fig, axes = plt.subplots(n_models, len(focus_types),
                             figsize=(7 * len(focus_types), 5 * n_models),
                             squeeze=False)

    for row, md in enumerate(models):
        svd_data = md["svd_data"]
        n_layers = md["n_layers"]
        label = md["label"]

        for col, wtype in enumerate(focus_types):
            ax = axes[row][col]
            if wtype not in md["weight_types"]:
                ax.text(0.5, 0.5, f"{wtype} not in {label}",
                        ha="center", transform=ax.transAxes)
                continue

            valid_layers = get_valid_layers(svd_data, wtype, n_layers)
            if not valid_layers:
                continue

            cmap = plt.cm.viridis(np.linspace(0, 1, len(valid_layers)))
            for i, li in enumerate(valid_layers):
                S = svd_data[li][wtype]["S"].numpy()
                ax.semilogy(S, color=cmap[i], alpha=0.6, linewidth=0.8)

            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Singular value (log scale)")
            ax.set_title(f"{label} — {wtype}")
            ax.grid(True, alpha=0.3)

            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                       norm=plt.Normalize(min(valid_layers),
                                                          max(valid_layers)))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Layer")

    fig.suptitle("Singular Value Spectra", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "spectra.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")
