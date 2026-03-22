"""
Martin & Mahoney Alpha — power-law tail exponent per layer.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .metrics import fit_power_law_tail, group_weight_types, get_valid_layers

DESCRIPTION = "Martin & Mahoney power-law tail exponent (α) per layer"

LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]


def run(models, output_dir):
    groups = group_weight_types(models[0]["weight_types"])
    n_groups = len(groups)

    fig, axes = plt.subplots(n_groups, 2, figsize=(14, 4.5 * n_groups), squeeze=False)

    for g_idx, (group_name, group_wtypes) in enumerate(groups.items()):
        ax1, ax2 = axes[g_idx]

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
                alphas, r2s = [], []
                for l in valid:
                    S = svd_data[l][wtype]["S"]
                    a, r2 = fit_power_law_tail(S)
                    alphas.append(a)
                    r2s.append(r2)

                suffix = f" ({label})" if len(models) > 1 else ""
                ax1.plot(valid, alphas, ls, marker=marker,
                         label=f"{wtype}{suffix}", markersize=4, linewidth=1.5)
                ax2.plot(valid, r2s, ls, marker=marker,
                         label=f"{wtype}{suffix}", markersize=4, linewidth=1.5)

                # Print summary
                if alphas:
                    mean_a = np.mean(alphas)
                    zone = ("well-trained" if 2 <= mean_a <= 4
                            else "under-trained" if mean_a < 2
                            else "over-regularized")
                    print(f"  [{label}] {wtype:>12s}: α = {mean_a:.2f} ± "
                          f"{np.std(alphas):.2f}  ({zone})")

        ax1.axhspan(2, 4, alpha=0.1, color="green", label="Well-trained (α ∈ [2,4])")
        ax1.axhline(y=2, color="green", linestyle="--", alpha=0.3)
        ax1.axhline(y=4, color="green", linestyle="--", alpha=0.3)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("α")
        ax1.set_title(f"{group_name} — Power-law exponent α")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Layer")
        ax2.set_ylabel("R²")
        ax2.set_title(f"{group_name} — Fit Quality")
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

    if len(models) > 1:
        fig.suptitle(f"Solid = {models[0]['label']}  |  Dashed = {models[1]['label']}",
                     fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "alpha.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")
