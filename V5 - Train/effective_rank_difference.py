# PACKAGES
import torch
import matplotlib.pyplot as plt
import numpy as np 
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

for step in iterations: 
    print(f"STEP {step}")

    # READ from weightspace 
    data = torch.load(f'logs/Muon/state_step{step:06d}.pt', map_location = 'cpu')
    model_muon = data['model']
    data = torch.load(f'logs/AdamW/state_step{step:06d}.pt', map_location = 'cpu')
    model_adamw = data['model']

    models = [['muon', model_muon], ['adamw', model_adamw]]

    for name, model in models:
        print(f"MODEL: {name}")
        appendices = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']

        # temporary dict 
        step_results = {mat: [] for mat in ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']}

        for appendix in appendices:
            print(f"\nAPPENDIX: {appendix}")

            if appendix == 'attn.c_attn':   
                for i in range(12): 
                    # print(f"# LAYER: {i}")
                    layer = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
                    QKV = model[layer]
                    Q, K, V = QKV.split(768, dim=0)
                    for mat_name, matrix in zip(['Q', 'K', 'V'], [Q, K, V]):
                        S = functions.svd(matrix)
                        step_results[mat_name].append(functions.effective_rank(S))
                      
            else: 
                for i in range(12): 
                    # print(f"# LAYER: {i}")
                    layer = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
                    S = functions.svd(model[layer])
                    step_results[appendix].append(functions.effective_rank(S))

        for mat in step_results: 
            results[name][mat].append(step_results[mat])

# PLOT the graphs 
matrix_types = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Effective Rank Difference (Muon - AdamW)')

for i, mat in enumerate(matrix_types):
    row = i // 3
    col = i % 3
    ax = axes[row][col]

    muon_arr = np.array(results['muon'][mat]) # shape (14, 12)
    adamw_arr = np.array(results['adamw'][mat]) # shape (14, 12)
    diff = np.abs(muon_arr - adamw_arr).T

    im = ax.imshow(diff, aspect='auto', cmap='inferno_r', origin='lower', vmin=0, vmax=300)
    ax.set_xticks(range(len(iterations)))
    ax.set_xticklabels(iterations, rotation = 45, fontsize = 7)
    ax.set_yticks(range(12))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Layer')
    ax.set_title(mat)
    fig.colorbar(im, ax = ax)

plt.tight_layout()
plt.savefig('analyse/plots/effective_rank_heatmap_prewarm_newMuon.png', dpi = 600)
plt.close()