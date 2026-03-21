"""
Singular Value Distribution Shapes — histograms at selected layers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .metrics import stable_rank, get_valid_layers

DESCRIPTION = "SV distribution histograms (first, middle, last layer)"

PREFERRED_ATTN = ["Q", "KV_b", "K", "O", "QKV_fused"]
PREFERRED_MLP = ["Down", "Gate", "Up", "Shared_Down"]
COLORS = ["steelblue", "coral", "seagreen", "goldenrod"]


def run(models, output_dir):
    wt0 = models[0]["weight_types"]
    pick_a = next((t for t in PREFERRED_ATTN if t in wt0), None)
    pick_m = next((t for t in PREFERRED_MLP if t in wt0), None)
    show_types = [t for t in [pick_a, pick_m] if t is not None]
    if not show_types:
        show_types = wt0[:2]

    n_models = len(models)

    # 3 layers × len(show_types) columns × n_models rows
    fig, axes = plt.subplots(n_models * len(show_types), 3,
                             figsize=(15, 4 * n_models * len(show_types)),
                             squeeze=False)

    plot_row = 0
    for m_idx, md in enumerate(models):
        svd_data = md["svd_data"]
        n_layers = md["n_layers"]
        label = md["label"]

        for wtype in show_types:
            if wtype not in md["weight_types"]:
                plot_row += 1
                continue

            valid = get_valid_layers(svd_data, wtype, n_layers)
            if len(valid) < 3:
                plot_row += 1
                continue

            show_layers = [valid[0], valid[len(valid) // 2], valid[-1]]
            color = COLORS[m_idx % len(COLORS)]

            for col, li in enumerate(show_layers):
                ax = axes[plot_row][col]
                S = svd_data[li][wtype]["S"]
                s_np = S.numpy()
                sr = stable_rank(S)

                ax.hist(s_np, bins=50, density=True, alpha=0.7,
                        color=color, edgecolor="white")
                ax.axvline(s_np[0], color="red", linestyle="--",
                           label=f"σ₁ = {s_np[0]:.2f}")
                ax.axvline(np.median(s_np), color="orange", linestyle="--",
                           label=f"median = {np.median(s_np):.2f}")
                ax.set_title(f"{label} — {wtype} L{li}  (sr={sr:.1f})", fontsize=10)
                ax.set_xlabel("Singular value")
                ax.set_ylabel("Density")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            plot_row += 1

    fig.suptitle("Singular Value Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")
