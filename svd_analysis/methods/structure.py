"""
Model Structure Overview — visualize the architecture before analyzing weights.

Shows:
  - Matrix dimensions per weight type
  - Parameter budget breakdown
  - Capacity allocation: Attention vs MLP (dense) or Attention vs Experts vs Shared vs Router (MoE)

For MoE models, additionally shows:
  - Number of experts, MoE layers, expert shapes
  - True total parameter count including all experts
  - Active vs total parameter ratio
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DESCRIPTION = "Model architecture overview (shapes, parameter counts, capacity allocation)"

# ── Type classification ───────────────────────────────────────────────
ATTENTION_TYPES = {"Q", "K", "V", "O", "QKV_fused", "Q_a", "Q_b", "KV_a", "KV_b"}
MLP_TYPES = {"Gate", "Up", "Down"}
SHARED_EXPERT_TYPES = {"Shared_Gate", "Shared_Up", "Shared_Down"}
ROUTER_TYPES = {"Router"}

COLORS = {
    "Q": "#4C72B0", "K": "#55A868", "V": "#C44E52", "O": "#8172B2",
    "Gate": "#CCB974", "Up": "#64B5CD", "Down": "#D68B5A",
    "QKV_fused": "#4C72B0", "Q_a": "#4C72B0", "Q_b": "#6C92D0",
    "KV_a": "#55A868", "KV_b": "#75C888",
    "Shared_Gate": "#E8D994", "Shared_Up": "#84D5ED", "Shared_Down": "#F6AB7A",
    "Router": "#E07B7B",
}

# Colors for the capacity stacked bar
CAT_COLORS = {
    "Attention": "#4C72B0",
    "MLP": "#CCB974",
    "Expert MLP": "#CCB974",
    "Shared Expert": "#E8D994",
    "Router": "#E07B7B",
}


def _classify_category(wtype):
    """Classify a weight type into a high-level category."""
    if wtype in ATTENTION_TYPES:
        return "Attention"
    elif wtype in SHARED_EXPERT_TYPES:
        return "Shared Expert"
    elif wtype in ROUTER_TYPES:
        return "Router"
    elif wtype in MLP_TYPES:
        return "MLP"
    else:
        return "Other"


def run(models, output_dir):
    n_models = len(models)
    fig = plt.figure(figsize=(18, 6 * n_models))

    for m_idx, md in enumerate(models):
        svd_data = md["svd_data"]
        n_layers = md["n_layers"]
        weight_types = md["weight_types"]
        label = md["label"]
        moe_info = md.get("moe_info", None)
        is_moe = moe_info is not None

        # ── Gather data ───────────────────────────────────────────────
        param_counts = {}
        shapes = {}

        for l in range(n_layers):
            param_counts[l] = {}
            for wtype in weight_types:
                if wtype in svd_data.get(l, {}):
                    shape = svd_data[l][wtype]["shape"]
                    param_counts[l][wtype] = shape[0] * shape[1]
                    if wtype not in shapes:
                        shapes[wtype] = shape

        # For MoE: get expert shapes from expert_svd
        expert_shapes = {}
        if is_moe and "expert_svd" in moe_info:
            esvd = moe_info["expert_svd"]
            for li in sorted(esvd.keys()):
                for ei in sorted(esvd[li].keys()):
                    for wt, data in esvd[li][ei].items():
                        if wt not in expert_shapes:
                            expert_shapes[wt] = data["shape"]
                    break  # one expert is enough
                break

        # ── Subplot 1: Matrix shapes ──────────────────────────────────
        ax1 = fig.add_subplot(n_models, 3, m_idx * 3 + 1)

        # Combine dense shapes + expert shapes
        all_shapes = dict(shapes)
        for wt, sh in expert_shapes.items():
            all_shapes[f"Expert_{wt}"] = sh

        wtype_list = [wt for wt in
                      ["Q", "Q_a", "Q_b", "K", "KV_a", "KV_b", "V", "O",
                       "Gate", "Up", "Down",
                       "Expert_Gate", "Expert_Up", "Expert_Down",
                       "Shared_Gate", "Shared_Up", "Shared_Down",
                       "Router", "QKV_fused"]
                      if wt in all_shapes]
        n_types = len(wtype_list)

        if n_types > 0:
            max_dim = max(max(s) for s in all_shapes.values())
            bar_height = 0.8

            for i, wtype in enumerate(wtype_list):
                rows, cols = all_shapes[wtype]
                base_wt = wtype.replace("Expert_", "")
                color = COLORS.get(base_wt, COLORS.get(wtype, "#999999"))

                w_normalized = cols / max_dim
                rect = mpatches.FancyBboxPatch(
                    (0, i - bar_height / 2), w_normalized, bar_height,
                    boxstyle="round,pad=0.02",
                    facecolor=color, alpha=0.7, edgecolor="black", linewidth=0.5)
                ax1.add_patch(rect)

                # Label
                extra = ""
                if wtype.startswith("Expert_") and is_moe:
                    extra = f" ×{moe_info['n_experts']}"
                ax1.text(w_normalized + 0.02, i,
                         f"{rows}×{cols}{extra}\n({rows * cols:,})",
                         va="center", fontsize=7, fontfamily="monospace")

            ax1.set_yticks(range(n_types))
            ax1.set_yticklabels(wtype_list, fontsize=9, fontweight="bold")
            ax1.set_xlim(-0.05, 1.8)
            ax1.set_ylim(-0.8, n_types - 0.2)
            ax1.set_xlabel("Relative width (cols / max)")
            ax1.set_title(f"{label}\nMatrix Shapes" +
                          (f" (MoE: {moe_info['n_experts']} experts)" if is_moe else ""),
                          fontsize=11)
            ax1.invert_yaxis()
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)

        # ── Subplot 2: Parameter budget breakdown ─────────────────────
        ax2 = fig.add_subplot(n_models, 3, m_idx * 3 + 2)

        # Calculate true totals including all experts
        budget = {}
        for wtype in weight_types:
            total = sum(param_counts[l].get(wtype, 0) for l in range(n_layers))
            budget[wtype] = total

        # Add expert parameters (not in svd_data, but in moe_info)
        if is_moe and expert_shapes:
            n_exp = moe_info["n_experts"]
            n_moe_layers = len(moe_info.get("moe_layers", []))
            for wt, sh in expert_shapes.items():
                key = f"Expert_{wt}"
                budget[key] = sh[0] * sh[1] * n_exp * n_moe_layers

        total_all = sum(budget.values())
        types_sorted = sorted(budget.keys(), key=lambda w: budget[w], reverse=True)
        values_sorted = [budget[wt] for wt in types_sorted]
        colors_sorted = [COLORS.get(wt.replace("Expert_", ""),
                         COLORS.get(wt, "#999999")) for wt in types_sorted]

        bars = ax2.barh(range(len(types_sorted)), values_sorted,
                        color=colors_sorted, alpha=0.8, edgecolor="white")
        ax2.set_yticks(range(len(types_sorted)))
        ax2.set_yticklabels(types_sorted, fontsize=9, fontweight="bold")
        ax2.set_xlabel("Total parameters")
        ax2.set_title(f"{label}\nParameter Budget ({total_all:,.0f} total)", fontsize=11)
        ax2.invert_yaxis()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        for i, (bar, wt) in enumerate(zip(bars, types_sorted)):
            pct = budget[wt] / total_all * 100 if total_all > 0 else 0
            ax2.text(bar.get_width() + total_all * 0.005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{pct:.1f}%", va="center", fontsize=7, fontfamily="monospace")

        # ── Subplot 3: Capacity allocation ────────────────────────────
        ax3 = fig.add_subplot(n_models, 3, m_idx * 3 + 3)

        if is_moe and expert_shapes:
            # MoE: show Attention / Expert MLP / Shared Expert / Router
            categories = ["Attention", "Expert MLP", "Shared Expert", "Router"]

            cat_totals = {c: 0 for c in categories}
            for wt, val in budget.items():
                if wt.startswith("Expert_"):
                    cat_totals["Expert MLP"] += val
                elif wt in ATTENTION_TYPES:
                    cat_totals["Attention"] += val
                elif wt in SHARED_EXPERT_TYPES:
                    cat_totals["Shared Expert"] += val
                elif wt in ROUTER_TYPES:
                    cat_totals["Router"] += val
                elif wt in MLP_TYPES:
                    # Dense MLP in non-MoE layers (layer 0 in DeepSeek)
                    cat_totals["Attention"] += val  # lump with attention for simplicity

            # Pie chart
            pie_vals = [cat_totals[c] for c in categories if cat_totals[c] > 0]
            pie_labels = [c for c in categories if cat_totals[c] > 0]
            pie_colors = [CAT_COLORS.get(c, "#999") for c in pie_labels]

            wedges, texts, autotexts = ax3.pie(
                pie_vals, labels=pie_labels, colors=pie_colors,
                autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
                startangle=90, textprops={"fontsize": 9})
            for t in autotexts:
                t.set_fontsize(8)
                t.set_fontweight("bold")

            # Summary in title
            n_exp = moe_info["n_experts"]
            n_moe_layers = len(moe_info.get("moe_layers", []))
            active_params = cat_totals["Attention"] + cat_totals["Shared Expert"] + cat_totals["Router"]
            # Active expert params = 1 expert (approx, depends on top-k)
            if expert_shapes and n_exp > 0:
                per_expert = sum(s[0] * s[1] for s in expert_shapes.values()) * n_moe_layers
                active_params += per_expert * 2  # typical top-2 routing

            ax3.set_title(f"{label}\n{n_exp} experts × {n_moe_layers} MoE layers\n"
                          f"Total: {total_all:,.0f} | "
                          f"Active: ~{active_params:,.0f}",
                          fontsize=10)
        else:
            # Dense model: Attention vs MLP bar chart
            categories = ["Attention", "MLP"]
            cat_per_layer = {c: [] for c in categories}

            for l in range(n_layers):
                attn = sum(param_counts[l].get(wt, 0)
                           for wt in weight_types if wt in ATTENTION_TYPES)
                mlp = sum(param_counts[l].get(wt, 0)
                          for wt in weight_types if wt in MLP_TYPES)
                cat_per_layer["Attention"].append(attn)
                cat_per_layer["MLP"].append(mlp)

            layers = range(n_layers)
            ax3.bar(layers, cat_per_layer["Attention"],
                    label="Attention", color=CAT_COLORS["Attention"], alpha=0.8)
            ax3.bar(layers, cat_per_layer["MLP"],
                    bottom=cat_per_layer["Attention"],
                    label="MLP", color=CAT_COLORS["MLP"], alpha=0.8)

            attn_total = sum(cat_per_layer["Attention"])
            mlp_total = sum(cat_per_layer["MLP"])
            all_total = attn_total + mlp_total
            attn_pct = attn_total / all_total * 100 if all_total > 0 else 0
            mlp_pct = mlp_total / all_total * 100 if all_total > 0 else 0

            ax3.set_xlabel("Layer")
            ax3.set_ylabel("Parameters")
            ax3.set_title(f"{label}\nAttn {attn_pct:.0f}% | MLP {mlp_pct:.0f}%\n"
                          f"Total: {all_total:,.0f}", fontsize=11)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "structure.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {path}")

    # ── Print summary ─────────────────────────────────────────────────
    for md in models:
        svd_data = md["svd_data"]
        n_layers = md["n_layers"]
        weight_types = md["weight_types"]
        label = md["label"]
        moe_info = md.get("moe_info", None)

        print(f"\n  [{label}] Architecture Summary")
        print(f"  {'─' * 60}")
        print(f"  Layers: {n_layers}")

        if moe_info:
            n_exp = moe_info["n_experts"]
            n_moe = len(moe_info.get("moe_layers", []))
            has_shared = moe_info.get("has_shared_experts", False)
            print(f"  Type: MoE ({n_exp} experts, {n_moe} MoE layers, "
                  f"{'with' if has_shared else 'no'} shared experts)")
        else:
            print(f"  Type: Dense")

        print(f"  Weight types: {', '.join(weight_types)}")
        print()

        # Dense weights table
        print(f"  {'Type':>12s} | {'Shape':>15s} | {'Params/Layer':>14s} | {'Total':>14s}")
        print(f"  {'─'*12}─┼─{'─'*15}─┼─{'─'*14}─┼─{'─'*14}")

        grand_total = 0
        for wtype in weight_types:
            shape = None
            for l in range(n_layers):
                if wtype in svd_data.get(l, {}):
                    shape = svd_data[l][wtype]["shape"]
                    break
            if shape is None:
                continue

            per_layer = shape[0] * shape[1]
            total = per_layer * n_layers
            grand_total += total
            print(f"  {wtype:>12s} | {str(shape[0])+'×'+str(shape[1]):>15s} | "
                  f"{per_layer:>14,d} | {total:>14,d}")

        # Expert weights table (if MoE)
        if moe_info and "expert_svd" in moe_info:
            esvd = moe_info["expert_svd"]
            exp_shapes = {}
            for li in sorted(esvd.keys()):
                for ei in sorted(esvd[li].keys()):
                    for wt, data in esvd[li][ei].items():
                        if wt not in exp_shapes:
                            exp_shapes[wt] = data["shape"]
                    break
                break

            if exp_shapes:
                n_exp = moe_info["n_experts"]
                n_moe = len(moe_info.get("moe_layers", []))
                print(f"  {'─'*12}─┼─{'─'*15}─┼─{'─'*14}─┼─{'─'*14}")
                print(f"  Expert weights ({n_exp} experts × {n_moe} layers):")

                for wt in sorted(exp_shapes.keys()):
                    sh = exp_shapes[wt]
                    per_expert = sh[0] * sh[1]
                    total = per_expert * n_exp * n_moe
                    grand_total += total
                    print(f"  {'Exp_'+wt:>12s} | {str(sh[0])+'×'+str(sh[1]):>15s} | "
                          f"{per_expert:>14,d} | {total:>14,d}")

        print(f"  {'─'*12}─┼─{'─'*15}─┼─{'─'*14}─┼─{'─'*14}")
        print(f"  {'TOTAL':>12s} | {'':>15s} | {'':>14s} | {grand_total:>14,d}")

    # ── Comparison delta ──────────────────────────────────────────────
    if len(models) == 2:
        print(f"\n  Structural Comparison")
        print(f"  {'─' * 60}")
        a, b = models[0], models[1]
        print(f"  {'':>12s} | {a['label']:>20s} | {b['label']:>20s}")
        print(f"  {'─'*12}─┼─{'─'*20}─┼─{'─'*20}")
        print(f"  {'Layers':>12s} | {a['n_layers']:>20d} | {b['n_layers']:>20d}")

        a_moe = a.get("moe_info")
        b_moe = b.get("moe_info")
        a_type = f"MoE ({a_moe['n_experts']})" if a_moe else "Dense"
        b_type = f"MoE ({b_moe['n_experts']})" if b_moe else "Dense"
        print(f"  {'Type':>12s} | {a_type:>20s} | {b_type:>20s}")

        shared = set(a["weight_types"]) & set(b["weight_types"])
        for wtype in sorted(shared):
            shape_a, shape_b = None, None
            for l in range(a["n_layers"]):
                if wtype in a["svd_data"].get(l, {}):
                    shape_a = a["svd_data"][l][wtype]["shape"]
                    break
            for l in range(b["n_layers"]):
                if wtype in b["svd_data"].get(l, {}):
                    shape_b = b["svd_data"][l][wtype]["shape"]
                    break
            sa = f"{shape_a[0]}×{shape_a[1]}" if shape_a else "N/A"
            sb = f"{shape_b[0]}×{shape_b[1]}" if shape_b else "N/A"
            match = "✓" if shape_a == shape_b else "≠"
            print(f"  {wtype:>12s} | {sa:>20s} | {sb:>20s}  {match}")
