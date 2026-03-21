"""
Cumulative Explained Variance — how many singular values capture X% of energy.
THE key plot for comparing AdamW vs Muon.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .metrics import cumulative_energy, group_weight_types, get_valid_layers

DESCRIPTION = "Cumulative explained variance (energy concentration)"

LINESTYLES = ["-", "--", ":", "-."]


def run(models, output_dir):
    groups = group_weight_types(models[0]["weight_types"])
    n_groups = len(groups)

    fig, axes = plt.subplots(n_groups, 2, figsize=(14, 5 * n_groups), squeeze=False)

    for g_idx, (group_name, group_wtypes) in enumerate(groups.items()):
        ax1, ax2 = axes[g_idx]

        for m_idx, md in enumerate(models):
            svd_data = md["svd_data"]
            n_layers = md["n_layers"]
            label = md["label"]
            ls = LINESTYLES[m_idx % len(LINESTYLES)]

            for wtype in group_wtypes:
                if wtype not in md["weight_types"]:
                    continue
                valid = get_valid_layers(svd_data, wtype, n_layers)
                if not valid:
                    continue

                # Left: average cumulative energy curve
                curves = [cumulative_energy(svd_data[l][wtype]["S"]) for l in valid]
                min_len = min(len(c) for c in curves)
                aligned = np.array([c[:min_len] for c in curves])
                mean_curve = aligned.mean(axis=0)

                suffix = f" ({label})" if len(models) > 1 else ""
                ax1.plot(np.arange(min_len), mean_curve, linestyle=ls,
                         label=f"{wtype}{suffix}", linewidth=1.5)

                # Right: dimensions for 90% energy per layer
                dims_90 = []
                for l in valid:
                    ce = cumulative_energy(svd_data[l][wtype]["S"])
                    dims_90.append(np.searchsorted(ce, 0.9) + 1)

                ax2.plot(valid, dims_90, linestyle=ls,
                         marker="o" if m_idx == 0 else "s",
                         label=f"{wtype}{suffix}", markersize=3)

        ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax1.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
        ax1.axhline(y=0.99, color="gray", linestyle="-.", alpha=0.5)
        ax1.set_xlabel("Singular values kept")
        ax1.set_ylabel("Fraction of energy")
        ax1.set_title(f"{group_name} — Cumulative Energy")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)

        ax2.set_xlabel("Layer")
        ax2.set_ylabel("SVs needed")
        ax2.set_title(f"{group_name} — Dims for 90% Energy")
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)

    if len(models) > 1:
        fig.suptitle(f"Solid = {models[0]['label']}  |  Dashed = {models[1]['label']}",
                     fontsize=12, y=1.02)
    else:
        fig.suptitle("Cumulative Energy Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "cumulative_energy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")

    # Print summary
    thresholds = [0.5, 0.9, 0.99]
    for md in models:
        svd_data = md["svd_data"]
        n_layers = md["n_layers"]
        print(f"\n  [{md['label']}] Dims for X% energy (avg):")
        print(f"  {'Type':>12s} | {'50%':>6s} | {'90%':>6s} | {'99%':>6s}")
        print(f"  {'─'*12}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")
        for wtype in md["weight_types"]:
            if wtype in {"Router"}:
                continue
            valid = get_valid_layers(svd_data, wtype, n_layers)
            if not valid:
                continue
            vals = []
            for t in thresholds:
                dims = [np.searchsorted(
                    cumulative_energy(svd_data[l][wtype]["S"]), t) + 1
                    for l in valid]
                vals.append(f"{np.mean(dims):6.0f}")
            print(f"  {wtype:>12s} | {' | '.join(vals)}")
