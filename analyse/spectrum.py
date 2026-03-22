# PACKAGES
import torch
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------------


# FILES
import functions
# -------------------------------------------------------------------------------------------------


# SETUP: 
# functions file contains modular functions (svd, stable_rank, effective_rank,...)
# configure new modular functions in main/ separate files
# -------------------------------------------------------------------------------------------------
 

# MAIN.PY
# DEFINE results dict 
results = {}
for opt in ['muon', 'adamw']: 
    results[opt] = {}
    for mat in ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
        results[opt][mat] = {'spectrum': []}

# READ from weightspace 
data = torch.load(f'data/muon/state_step006200.pt', map_location = 'cpu')
model_muon = data['model']
data = torch.load(f'data/adamw/state_step006200.pt', map_location = 'cpu')
model_adamw = data['model']
models = [['muon', model_muon], ['adamw', model_adamw]]

for name, model in models:
    # leading singular value ratio
    print(f"MODEL: {name}")
    appendices = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
    for appendix in appendices:
        print(f"\nAPPENDIX: {appendix}")
        if appendix == 'attn.c_attn':   

            for i in range(12): 
                print(f"# LAYER: {i}")
                layer = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
                QKV = model[layer]
                Q, K, V = QKV.split(768, dim=0)
                for mat_name, matrix in zip(['Q', 'K', 'V'], [Q, K, V]):
                    S = functions.svd(matrix)
                    results[name][mat_name]['spectrum'].append(S)

        else: 
            for i in range(12): 
                print(f"# LAYER: {i}")
                layer = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
                S = functions.svd(model[layer])
                results[name][appendix]['spectrum'].append(S)

# PLOT the graphs 
matrix_types = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']

for mat in matrix_types: 
    fig, axes = plt.subplots(4, 3, figsize= (18, 22))
    fig.suptitle(f"Normalized Singular Value Distribution - {mat}")

    for i in range(12): 
        row = i//3
        col = i%3
        ax = axes[row][col]

        S_muon = results['muon'][mat]['spectrum'][i]
        S_adamw = results['adamw'][mat]['spectrum'][i]


        ax.plot(S_muon/S_muon[0], label='Muon')
        ax.plot(S_adamw/S_adamw[0], label='AdamW', linestyle='--')
        ax.set_title(f'Layer {i} (σ₁: Muon={S_muon[0]:.2f}, AdamW={S_adamw[0]:.2f})')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Normalized Singular Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'analyse/plots/spectrum_{mat}.png', dpi=300)
    plt.close()
