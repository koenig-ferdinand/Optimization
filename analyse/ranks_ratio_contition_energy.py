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
        results[opt][mat] = {'ratio': [], 'stable': [], 'effective': [], 'condition': [], 'energy': []}

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
                    results[name][mat_name]['ratio'].append(functions.ratio(S))
                    results[name][mat_name]['stable'].append(functions.stable_rank(S))
                    results[name][mat_name]['effective'].append(functions.effective_rank(S))
                    results[name][mat_name]['condition'].append(functions.condition_number(S))
                    results[name][mat_name]['energy'].append(functions.energy_k(S))

        else: 
            for i in range(12): 
                print(f"# LAYER: {i}")
                layer = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
                S = functions.svd(model[layer])
                results[name][appendix]['ratio'].append(functions.ratio(S))
                results[name][appendix]['stable'].append(functions.stable_rank(S))
                results[name][appendix]['effective'].append(functions.effective_rank(S))
                results[name][appendix]['condition'].append(functions.condition_number(S))
                results[name][appendix]['energy'].append(functions.energy_k(S))

# PLOT the graphs 
matrix_types = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
metrics = ['ratio', 'stable', 'effective', 'condition', 'energy']
metric_titles = ['Leading Singular Value Ratio', 'Stable Rank', 'Effective Rank', 'Condition Number', '# S.V. for 90 Percent of Energy']

for m, metric in enumerate(metrics): 
    fig, axes = plt.subplots(2, 3, figsize= (18, 10))
    fig.suptitle(metric_titles[m])

    for i, mat in enumerate(matrix_types): 
        row = i//3
        col = i%3
        ax = axes[row][col]

        ax.plot(range(12), results['muon'][mat][metric], label='Muon', marker='o')
        ax.plot(range(12), results['adamw'][mat][metric], label='AdamW', marker='s')
        ax.set_title(mat)
        ax.set_xlabel('Layer')
        ax.set_ylabel(metric_titles[m])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'analyse/plots/{metric}.png', dpi=600)
    plt.close()
