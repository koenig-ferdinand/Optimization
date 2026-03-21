"""
Spectral Norm, Stable Rank, and Condition Number across layers.
Grouped by Attention / MLP / Shared Expert for readability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .metrics import (spectral_norm, stable_rank, condition_number,
                      group_weight_types, get_valid_layers)

DESCRIPTION = "Spectral norm, stable rank, and condition number per layer"

LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]


def run(models, output_dir):
    # Determine groups from first model
    groups = group_weight_types(models[0]["weight_types"])
    n_groups = len(groups)

    fig, axes = plt.subplots(n_groups, 3, figsize=(18, 4.5 * n_groups), squeeze=False)

    for g_idx, (group_name, group_wtypes) in enumerate(groups.items()):
        ax1, ax2, ax3 = axes[g_idx]

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
                spec = [spectral_norm(svd_data[l][wtype]["S"]) for l in valid]
                stab = [stable_rank(svd_data[l][wtype]["S"]) for l in valid]
                cond = [condition_number(svd_data[l][wtype]["S"]) for l in valid]

                suffix = f" ({label})" if len(models) > 1 else ""
                ax1.plot(valid, spec, ls, marker=marker,
                         label=f"{wtype}{suffix}", markersize=3)
                ax2.plot(valid, stab, ls, marker=marker,
                         label=f"{wtype}{suffix}", markersize=3)
                ax3.semilogy(valid, cond, ls, marker=marker,
                             label=f"{wtype}{suffix}", markersize=3)

        ax1.set_title(f"{group_name} — Spectral Norm (σ₁)")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("σ₁")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        ax2.set_title(f"{group_name} — Stable Rank")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Stable rank")
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)

        ax3.set_title(f"{group_name} — Condition Number")
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Condition (log)")
        ax3.legend(fontsize=7, ncol=2)
        ax3.grid(True, alpha=0.3)

    if len(models) > 1:
        fig.suptitle(f"Solid = {models[0]['label']}  |  Dashed = {models[1]['label']}",
                     fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "stable_rank.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")

    # Print summary
    for md in models:
        svd_data = md["svd_data"]
        n_layers = md["n_layers"]
        print(f"\n  [{md['label']}]")
        print(f"  {'Type':>12s} | {'σ₁':>8s} | {'Stable Rank':>12s} | {'Condition':>12s}")
        print(f"  {'─'*12}─┼─{'─'*8}─┼─{'─'*12}─┼─{'─'*12}")
        for wtype in md["weight_types"]:
            valid = get_valid_layers(svd_data, wtype, n_layers)
            if not valid:
                continue
            specs = [spectral_norm(svd_data[l][wtype]["S"]) for l in valid]
            stabs = [stable_rank(svd_data[l][wtype]["S"]) for l in valid]
            conds = [condition_number(svd_data[l][wtype]["S"]) for l in valid]
            print(f"  {wtype:>12s} | {np.mean(specs):8.3f} | {np.mean(stabs):12.1f} | {np.mean(conds):12.1f}")
