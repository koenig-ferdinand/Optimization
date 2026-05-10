# PACKAGES
import torch
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
torch.set_num_threads(1)  # prevent thread oversubscription with joblib
# -------------------------------------------------------------------------------------------------


# FILES
import functions
# -------------------------------------------------------------------------------------------------


# SETUP:
# functions file contains modular functions (svd, stable_rank, effective_rank,...)
# configure new modular functions in main/ separate files
# -------------------------------------------------------------------------------------------------


# MAIN.PY
#HEATMAP SETUP
# heatmap for each type of matrix
# x-axis step: 500, 1000, ..., 6000, 6200 (13)
# y-axis layers 0-11 (12)

# iterate over iterations
# compute difference of effective rank
# low difference brighter, high difference dimmer


# DEFINE results dict
iterations = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200]

results = {}
for opt in ['muon', 'adamw']:
    results[opt] = {}
    for mat in ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
        results[opt][mat] = []


def process_step(step):
    data = torch.load(f'logs/Muon/state_step{step:06d}.pt', map_location='cpu')
    model_muon = data['model']
    data = torch.load(f'logs/AdamW/state_step{step:06d}.pt', map_location='cpu')
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


# PLOT the graphs
matrix_types = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Effective Rank Difference (Muon - AdamW)')

for i, mat in enumerate(matrix_types):
    row = i // 3
    col = i % 3
    ax = axes[row][col]

    muon_arr = np.array(results['muon'][mat])   # shape (62, 12)
    adamw_arr = np.array(results['adamw'][mat]) # shape (62, 12)
    diff = np.abs(muon_arr - adamw_arr).T

    im = ax.imshow(diff, aspect='auto', cmap='inferno_r', origin='lower', vmin=0, vmax=200)
    tick_every = 10
    ax.set_xticks(range(0, len(iterations), tick_every))
    ax.set_xticklabels(iterations[::tick_every], rotation=45, fontsize=8)
    ax.set_yticks(range(12))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Layer')
    ax.set_title(mat)
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('analyse/plots/effective_rank_heatmap_prewarm_newMuon_maxAxis200V2.png', dpi=600)
plt.close()
print("Done! Plot saved to analyse/plots/effective_rank_heatmap_prewarm_newMuon_maxAxis200V2.png")