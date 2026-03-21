"""
MoE Expert Analysis — how similar or different are the experts?

Shows:
  1. Expert spectral diversity: overlay all experts' SV spectra per layer
  2. Expert stable rank distribution: box plot across experts per layer
  3. Inter-expert alignment: do experts share subspaces or are they independent?
  4. Expert specialization score: variance of singular values across experts

Only runs if the model has MoE layers. Skips gracefully for dense models.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import spectral_norm, stable_rank, effective_rank

DESCRIPTION = "MoE expert diversity analysis (specialization, alignment, spectral spread)"


def run(models, output_dir):
    for md in models:
        moe_info = md.get("moe_info", None)
        if moe_info is None:
            print(f"  [{md['label']}] Dense model — skipping MoE analysis.")
            continue

        expert_svd = moe_info.get("expert_svd", {})
        n_experts = moe_info["n_experts"]

        if not expert_svd or n_experts == 0:
            print(f"  [{md['label']}] No expert SVD data found — skipping.")
            print(f"  (This can happen if expert weights are stored in an "
                  f"unsupported format.)")
            print(f"  Debug: run this to see parameter names:")
            print(f"    for n,p in model.named_parameters():")
            print(f"        if p.dim() >= 2: print(n, tuple(p.shape))")
            continue

        label = md["label"]
        n_layers = md["n_layers"]
        moe_layers = sorted(moe_info["moe_layers"])

        # Find a layer that actually has expert SVD data
        sample_layer = None
        for li in moe_layers:
            if li in expert_svd and expert_svd[li]:
                sample_layer = li
                break

        if sample_layer is None:
            print(f"  [{md['label']}] Expert SVDs are empty — skipping.")
            continue

        print(f"  [{label}] Analyzing {n_experts} experts across {len(moe_layers)} MoE layers")

        sample_expert = list(expert_svd[sample_layer].keys())[0]
        expert_wtypes = sorted(expert_svd[sample_layer][sample_expert].keys())
        print(f"  Expert weight types: {expert_wtypes}")

        # ══════════════════════════════════════════════════════════════
        # FIGURE 1: Expert SV Spectra Overlay
        # All experts' singular value curves overlaid for selected layers
        # ══════════════════════════════════════════════════════════════
        show_layers = [moe_layers[0], moe_layers[len(moe_layers) // 2], moe_layers[-1]]
        show_wtype = expert_wtypes[0] if expert_wtypes else "Gate"

        fig1, axes = plt.subplots(1, len(show_layers),
                                  figsize=(6 * len(show_layers), 5))
        if len(show_layers) == 1:
            axes = [axes]

        cmap = plt.cm.tab20(np.linspace(0, 1, min(n_experts, 20)))

        for ax, li in zip(axes, show_layers):
            if li not in expert_svd:
                continue
            for e_idx in sorted(expert_svd[li].keys()):
                if show_wtype in expert_svd[li][e_idx]:
                    S = expert_svd[li][e_idx][show_wtype]["S"].numpy()
                    color = cmap[e_idx % len(cmap)]
                    ax.semilogy(S, color=color, alpha=0.5, linewidth=0.7)

            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Singular value (log)")
            ax.set_title(f"Layer {li} — {show_wtype}\n({n_experts} experts overlaid)")
            ax.grid(True, alpha=0.3)

        fig1.suptitle(f"{label} — Expert SV Spectra\n"
                      f"(tight bundle = similar experts, spread = specialized experts)",
                      fontsize=13, y=1.03)
        fig1.tight_layout()
        path1 = os.path.join(output_dir, "moe_spectra.png")
        fig1.savefig(path1, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"  → Saved {path1}")

        # ══════════════════════════════════════════════════════════════
        # FIGURE 2: Stable Rank & Spectral Norm Distribution per Layer
        # Box plot showing spread across experts
        # ══════════════════════════════════════════════════════════════
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for wtype in expert_wtypes:
            sr_per_layer = []
            sn_per_layer = []
            layer_labels = []

            for li in moe_layers:
                if li not in expert_svd:
                    continue
                srs = []
                sns_vals = []
                for e_idx in sorted(expert_svd[li].keys()):
                    if wtype in expert_svd[li][e_idx]:
                        S = expert_svd[li][e_idx][wtype]["S"]
                        srs.append(stable_rank(S))
                        sns_vals.append(spectral_norm(S))
                if srs:
                    sr_per_layer.append(srs)
                    sn_per_layer.append(sns_vals)
                    layer_labels.append(li)

            if not sr_per_layer:
                continue

            # Plot as box plots
            positions = range(len(layer_labels))
            bp1 = ax1.boxplot(sr_per_layer, positions=positions, widths=0.6,
                              patch_artist=True, showfliers=False)
            bp2 = ax2.boxplot(sn_per_layer, positions=positions, widths=0.6,
                              patch_artist=True, showfliers=False)

            color = "#4C72B0" if wtype == expert_wtypes[0] else "#C44E52"
            for bp in [bp1, bp2]:
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

            if wtype == expert_wtypes[0]:
                ax1.set_xticks(positions)
                ax1.set_xticklabels(layer_labels, fontsize=7, rotation=45)
                ax2.set_xticks(positions)
                ax2.set_xticklabels(layer_labels, fontsize=7, rotation=45)
            break  # plot only first wtype for readability

        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Stable rank")
        ax1.set_title(f"Stable Rank Across Experts — {expert_wtypes[0]}\n"
                      f"(tall boxes = diverse experts)")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Spectral norm (σ₁)")
        ax2.set_title(f"Spectral Norm Across Experts — {expert_wtypes[0]}\n"
                      f"(tall boxes = diverse experts)")
        ax2.grid(True, alpha=0.3, axis="y")

        fig2.suptitle(f"{label} — Expert Metric Distributions", fontsize=13, y=1.02)
        fig2.tight_layout()
        path2 = os.path.join(output_dir, "moe_metrics.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  → Saved {path2}")

        # ══════════════════════════════════════════════════════════════
        # FIGURE 3: Inter-Expert Alignment
        # For a selected layer: NxN heatmap of cosine similarity
        # between experts' top-k right singular vectors
        # ══════════════════════════════════════════════════════════════
        k_align = 10
        # Pick middle MoE layer
        align_layer = moe_layers[len(moe_layers) // 2]
        align_wtype = expert_wtypes[0]

        experts_in_layer = sorted(expert_svd.get(align_layer, {}).keys())
        ne = len(experts_in_layer)

        if ne > 1 and align_wtype in expert_svd.get(align_layer, {}).get(experts_in_layer[0], {}):
            align_matrix = np.zeros((ne, ne))

            for i, ei in enumerate(experts_in_layer):
                for j, ej in enumerate(experts_in_layer):
                    if (align_wtype in expert_svd[align_layer].get(ei, {}) and
                            align_wtype in expert_svd[align_layer].get(ej, {})):
                        Vh_i = expert_svd[align_layer][ei][align_wtype]["Vh"][:k_align, :]
                        Vh_j = expert_svd[align_layer][ej][align_wtype]["Vh"][:k_align, :]
                        if Vh_i.shape[1] == Vh_j.shape[1]:
                            cos = torch.abs(Vh_i @ Vh_j.T).mean().item()
                            align_matrix[i, j] = cos

            fig3, ax = plt.subplots(figsize=(min(12, ne * 0.3 + 2), min(10, ne * 0.3 + 2)))
            sns.heatmap(align_matrix, ax=ax, cmap="magma", vmin=0, vmax=1,
                        square=True,
                        xticklabels=[f"E{e}" for e in experts_in_layer],
                        yticklabels=[f"E{e}" for e in experts_in_layer])
            ax.set_title(f"{label} — Inter-Expert Alignment\n"
                         f"Layer {align_layer}, {align_wtype}, top-{k_align} SVs\n"
                         f"(bright = shared subspace, dark = independent)")
            ax.tick_params(labelsize=max(4, min(8, 200 // ne)))

            fig3.tight_layout()
            path3 = os.path.join(output_dir, "moe_alignment.png")
            fig3.savefig(path3, dpi=150, bbox_inches="tight")
            plt.close(fig3)
            print(f"  → Saved {path3}")

        # ══════════════════════════════════════════════════════════════
        # FIGURE 4: Expert Specialization Score Across Layers
        # CV (coefficient of variation) of singular values across experts
        # High CV = experts learned different things
        # ══════════════════════════════════════════════════════════════
        fig4, ax = plt.subplots(figsize=(12, 5))

        for wtype in expert_wtypes:
            cv_per_layer = []
            layers_valid = []

            for li in moe_layers:
                if li not in expert_svd:
                    continue
                # Collect spectral norms from all experts
                norms = []
                for e_idx in sorted(expert_svd[li].keys()):
                    if wtype in expert_svd[li][e_idx]:
                        S = expert_svd[li][e_idx][wtype]["S"]
                        norms.append(spectral_norm(S))

                if len(norms) > 1:
                    mean_n = np.mean(norms)
                    std_n = np.std(norms)
                    cv = std_n / mean_n if mean_n > 0 else 0
                    cv_per_layer.append(cv)
                    layers_valid.append(li)

            if cv_per_layer:
                ax.plot(layers_valid, cv_per_layer, "o-", label=wtype,
                        markersize=4, linewidth=1.5)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Coefficient of Variation (σ/μ) of σ₁")
        ax.set_title(f"{label} — Expert Specialization Score\n"
                     f"(higher = experts are more different from each other)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig4.tight_layout()
        path4 = os.path.join(output_dir, "moe_specialization.png")
        fig4.savefig(path4, dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"  → Saved {path4}")

        # ── Print summary ─────────────────────────────────────────────
        print(f"\n  [{label}] MoE Summary:")
        print(f"  {'─' * 60}")
        print(f"  Experts: {n_experts}")
        print(f"  MoE layers: {len(moe_layers)} / {n_layers}")
        print(f"  Expert weight types: {expert_wtypes}")
        print(f"  Shared experts: {'Yes' if moe_info.get('has_shared_experts') else 'No'}")

        for wtype in expert_wtypes:
            all_srs = []
            all_sns = []
            for li in moe_layers:
                for e_idx in expert_svd.get(li, {}).keys():
                    if wtype in expert_svd[li][e_idx]:
                        S = expert_svd[li][e_idx][wtype]["S"]
                        all_srs.append(stable_rank(S))
                        all_sns.append(spectral_norm(S))
            if all_srs:
                print(f"\n  {wtype} (across all experts, all layers):")
                print(f"    Stable rank:  {np.mean(all_srs):.1f} ± {np.std(all_srs):.1f}")
                print(f"    Spectral norm: {np.mean(all_sns):.3f} ± {np.std(all_sns):.3f}")
