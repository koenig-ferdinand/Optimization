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
# READ from weightspace 
data = torch.load(f'data/muon/state_step006200.pt', map_location = 'cpu')
model_muon = data['model']
data = torch.load(f'data/adamw/state_step006200.pt', map_location = 'cpu')
model_adamw = data['model']

models = {'muon': model_muon, 'adamw': model_adamw}
matrix_types = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
transpose_set = {'Q', 'K', 'V', 'mlp.c_fc'} 

# step 1, collect per layer matries for each (model, mat_type)
weights = {opt: {mat: [] for mat in matrix_types} for opt in models}
for opt, model in models.items():
    for i in range(12): 
        QKV = model[f'_orig_mod.transformer.h.{i}.attn.c_attn.weight']
        Q, K, V = QKV.split(768, dim=0)
        weights[opt]['Q'].append(Q.numpy().T)
        weights[opt]['K'].append(K.numpy().T)
        weights[opt]['V'].append(V.numpy().T)
        for app in ['attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
            W = model[f'_orig_mod.transformer.h.{i}.{app}.weight']
            if app in transpose_set: 
                W = W.T
            weights[opt][app].append(W.numpy())

# step 2 compute 12x12 DOCS matrix for each (model, mat_type)
docs_results = {opt: {} for opt in models}

for opt in models: 
    for mat in matrix_types: 
        print(f"opt: {opt}, matix: {mat}")
        M = np.zeros((12, 12))
        layers = weights[opt][mat]
        for i in range(12): 
            for j in range(i, 12): 
                M[i, j] = functions.DOCS(layers[i], layers[j])
                M[j, i] = M[i, j]
        docs_results[opt][mat] = M

# Step 3: plot, one figure per optimizer
for opt in models:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'DOCS layer similarity — {opt}')
    for i, mat in enumerate(matrix_types):
        ax = axes[i // 3][i % 3]
        im = ax.imshow(docs_results[opt][mat], cmap='inferno', origin='lower')
        ax.set_xticks(range(12)); ax.set_yticks(range(12))
        ax.set_xlabel('Layer'); ax.set_ylabel('Layer')
        ax.set_title(mat)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'analyse/plots/docs_heatmap_{opt}.png', dpi=600)
    plt.close()