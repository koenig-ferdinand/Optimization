"""
Weight Norm Landscape — Frobenius, spectral, and nuclear norms across layers.
"""

import os
import matplotlib.pyplot as plt
from .metrics import (frobenius_norm, spectral_norm, nuclear_norm,
                      group_weight_types, get_valid_layers)

DESCRIPTION = "Frobenius, spectral, and nuclear norms per layer"

LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]


def run(models, output_dir):
    groups = group_weight_types(models[0]["weight_types"])
    n_groups = len(groups)
    norm_funcs = [
        (frobenius_norm, "Frobenius ||W||_F"),
        (spectral_norm,  "Spectral ||W||₂"),
        (nuclear_norm,   "Nuclear ||W||_*"),
    ]

    fig, axes = plt.subplots(n_groups, 3, figsize=(18, 4.5 * n_groups), squeeze=False)

    for g_idx, (group_name, group_wtypes) in enumerate(groups.items()):
        for col, (func, title) in enumerate(norm_funcs):
            ax = axes[g_idx][col]

            for m_idx, md in enumerate(models):
                svd_data = md["svd_data"]
                n_layers = md["n_layers"]
                label = md["label"]
                ls = LINESTYLES[m_idx % len(LINESTYLES)]
                marker = MARKERS[m_idx % len(MARKERS)]

                for wtype in group_wtypes:
                    if wtype not in md["weight_types"]:
                        continue
                    valid = get_valid_layers(svd_data, wtype, n_layers)
                    vals = [func(svd_data[l][wtype]["S"]) for l in valid]

                    suffix = f" ({label})" if len(models) > 1 else ""
                    ax.plot(valid, vals, ls, marker=marker,
                            label=f"{wtype}{suffix}", markersize=3)

            ax.set_xlabel("Layer")
            ax.set_title(f"{group_name} — {title}")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

    if len(models) > 1:
        fig.suptitle(f"Solid = {models[0]['label']}  |  Dashed = {models[1]['label']}",
                     fontsize=12, y=1.01)
    else:
        fig.suptitle("Weight Norm Landscape", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "norms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")
