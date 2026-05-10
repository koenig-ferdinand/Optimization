# PACKAGES
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from joblib import Parallel, delayed
torch.set_num_threads(1)  # prevent thread oversubscription with joblib
# -------------------------------------------------------------------------------------------------

# FILES
import functions
# -------------------------------------------------------------------------------------------------

# DEFINE results dict
iterations = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200]

results = {}
for opt in ['muon', 'adamw']:
    results[opt] = {}
    for mat in ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
        results[opt][mat] = []


def process_step(step):
    data = torch.load(f'logs/Muon/state_step{step:06d}.pt', map_location='cpu', weights_only=False)
    model_muon = data['model']
    data = torch.load(f'logs/AdamW/state_step{step:06d}.pt', map_location='cpu', weights_only=False)
    model_adamw = data['model']

    step_data = {}
    for name, model in [['muon', model_muon], ['adamw', model_adamw]]:
        step_results = {mat: [] for mat in ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']}
        for appendix in ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
            if appendix == 'attn.c_attn':
                for i in range(12):
                    layer = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
                    QKV = model[layer]
                    Q, K, V = QKV.split(768, dim=0)
                    for mat_name, matrix in zip(['Q', 'K', 'V'], [Q, K, V]):
                        S = functions.svd(matrix)
                        step_results[mat_name].append(functions.effective_rank(S))
            else:
                for i in range(12):
                    layer = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
                    S = functions.svd(model[layer])
                    step_results[appendix].append(functions.effective_rank(S))
        step_data[name] = step_results
    print(f"STEP {step} done")
    return step_data


# run in parallel
all_step_data = Parallel(n_jobs=32)(delayed(process_step)(step) for step in iterations)

# collect results in order
for step_data in all_step_data:
    for name in ['muon', 'adamw']:
        for mat in step_data[name]:
            results[name][mat].append(step_data[name][mat])

# -------------------------------------------------------------------------------------------------
# COMPUTE diff arrays and convergence curves
# -------------------------------------------------------------------------------------------------

matrix_types = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
tick_every = 10

# diff_arrays[mat]: shape (62, 12) — |muon - adamw| per step per layer
diff_arrays = {}
# conv_curves[mat]: shape (62,) — sum over layers of (diff[step] - diff[-1])
conv_curves = {}

for mat in matrix_types:
    muon_arr = np.array(results['muon'][mat])   # (62, 12)
    adamw_arr = np.array(results['adamw'][mat]) # (62, 12)
    diff = np.abs(muon_arr - adamw_arr)          # (62, 12)
    diff_arrays[mat] = diff

    # for each layer, compute distance from its final value
    final_vals = diff[-1, :]                     # (12,) — last step value per layer
    dist_from_final = diff - final_vals          # (62, 12)
    # sum over all layers → one curve per matrix
    conv_curves[mat] = dist_from_final.sum(axis=1)  # (62,)

# combined curve: sum all 6 matrix curves
combined_curve = sum(conv_curves[mat] for mat in matrix_types)

# -------------------------------------------------------------------------------------------------
# PLOT 1: Heatmaps + convergence curves
# -------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(20, 14))
fig.suptitle('Effective Rank Difference (Muon - AdamW) + Convergence Curves', fontsize=14)

outer = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

for i, mat in enumerate(matrix_types):
    row = i // 3
    col = i % 3

    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col],
                                              height_ratios=[3, 1], hspace=0.1)

    # --- Heatmap ---
    ax_heat = fig.add_subplot(inner[0])
    diff = diff_arrays[mat].T  # (12, 62)
    im = ax_heat.imshow(diff, aspect='auto', cmap='inferno_r', origin='lower', vmin=0, vmax=200)
    ax_heat.set_xticks(range(0, len(iterations), tick_every))
    ax_heat.set_xticklabels([])
    ax_heat.set_yticks(range(12))
    ax_heat.set_ylabel('Layer', fontsize=8)
    ax_heat.set_title(mat, fontsize=10)
    fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)

    # --- Convergence curve ---
    ax_curve = fig.add_subplot(inner[1])
    ax_curve.plot(range(len(iterations)), conv_curves[mat], color='steelblue', linewidth=1.2)
    ax_curve.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_curve.set_xticks(range(0, len(iterations), tick_every))
    ax_curve.set_xticklabels(iterations[::tick_every], rotation=45, fontsize=6)
    ax_curve.set_ylabel('Σ dist\nfrom final', fontsize=7)
    ax_curve.set_xlabel('Iteration', fontsize=8)
    ax_curve.tick_params(axis='y', labelsize=7)

plt.savefig('analyse/plots/heatmap_with_convergence.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot 1 saved: analyse/plots/heatmap_with_convergence.png")

# -------------------------------------------------------------------------------------------------
# PLOT 2: Combined convergence curve (all 6 matrices summed) + individual curves
# -------------------------------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(12, 5))

# individual matrix curves
colors = ['#e07b54', '#e0c454', '#54e07b', '#5492e0', '#a054e0', '#e054c4']
for mat, color in zip(matrix_types, colors):
    ax2.plot(iterations, conv_curves[mat], linewidth=0.9, alpha=0.6,
             linestyle='--', color=color, label=mat)

# combined curve on top
ax2.plot(iterations, combined_curve, color='darkgreen', linewidth=2.0,
         label='Combined (all matrices)', zorder=5)

ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Σ distance from final effective rank diff')
ax2.set_title('Convergence of Effective Rank Difference (Muon vs AdamW)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analyse/plots/combined_convergence_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot 2 saved: analyse/plots/combined_convergence_curve.png")
print("All done!")