"""
Effective Rank (Shannon entropy-based, Roy & Vetterli 2007).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .metrics import effective_rank, group_weight_types, get_valid_layers

DESCRIPTION = "Effective rank and rank utilization per layer"

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
                eff = [effective_rank(svd_data[l][wtype]["S"]) for l in valid]
                fracs = [effective_rank(svd_data[l][wtype]["S"]) /
                         min(svd_data[l][wtype]["shape"]) for l in valid]

                suffix = f" ({label})" if len(models) > 1 else ""
                ax1.plot(valid, eff, ls, marker=marker,
                         label=f"{wtype}{suffix}", markersize=3)
                ax2.plot(valid, fracs, ls, marker=marker,
                         label=f"{wtype}{suffix}", markersize=3)

        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Effective rank")
        ax1.set_title(f"{group_name} — Effective Rank")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Effective rank / Full rank")
        ax2.set_title(f"{group_name} — Rank Utilization")
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

    if len(models) > 1:
        fig.suptitle(f"Solid = {models[0]['label']}  |  Dashed = {models[1]['label']}",
                     fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "effective_rank.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")
