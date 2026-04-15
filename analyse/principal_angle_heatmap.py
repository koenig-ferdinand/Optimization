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
# DEFINE results dict 
results = {}
  
for mat in ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
    results[mat] = {}
    for i in range(12): 
        results[mat][i] = {'U_distribution': [], 'V_distribution': []}

# READ from weightspace 
data = torch.load(f'data/muon/state_step003000.pt', map_location = 'cpu')
model_muon = data['model']
data = torch.load(f'data/adamw/state_step003000.pt', map_location = 'cpu')
model_adamw = data['model']
models = [['muon', model_muon], ['adamw', model_adamw]]


# leading singular value ratio
appendices = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
for appendix in appendices:
    print(f"\nAPPENDIX: {appendix}")
    if appendix == 'attn.c_attn':   
        for i in range(12): 
            # print(f"# LAYER: {i}")
            layer = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
            QKV_muon = model_muon[layer]
            Q_muon, K_muon, V_muon = QKV_muon.split(768, dim=0)
            QKV_adamw = model_adamw[layer]
            Q_adamw, K_adamw, V_adamw = QKV_adamw.split(768, dim=0)
            for mat_name, [matrix_muon, matrix_adamw] in zip(['Q', 'K', 'V'], [[Q_muon, Q_adamw], [K_muon, K_adamw], [V_muon, V_adamw]]):
                U_angles, V_angles = functions.principal_angles(matrix_muon, matrix_adamw)
                results[mat_name][i]['U_distribution'] = U_angles
                results[mat_name][i]['V_distribution'] = V_angles

    else: 
        for i in range(12): 
            print(f"# LAYER: {i}")
            layer = f'_orig_mod.transformer.h.{i}.{appendix}.weight'
            U_angles, V_angles = functions.principal_angles(model_muon[layer], model_adamw[layer])
            results[appendix][i]['U_distribution'] = U_angles
            results[appendix][i]['V_distribution'] = V_angles

# PLOT the graphs 
matrix_types = ['Q', 'K', 'V', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']


fig, axes = plt.subplots(2, 3, figsize= (18, 10))
fig.suptitle(f"Principal Angles (U) - First 50")

for i, mat in enumerate(matrix_types): 
    row = i//3
    col = i%3
    ax = axes[row][col]

    # build 2d array (50, 12)
    angle_matrix = np.zeros((50, 12))
    for layer_idx in range(12): 
        angles_deg = np.degrees(results[mat][layer_idx]['U_distribution'][:50])
        angle_matrix[:, layer_idx] = angles_deg

    im = ax.imshow(angle_matrix, aspect='auto', cmap='inferno_r', origin='lower', vmin=45, vmax=90)
    ax.set_xticks(range(12)) 
    ax.set_yticks(range(0, 50, 5))
    ax.set_xlabel('Layer')
    ax.set_ylabel('Principal Angle Index')
    ax.set_title(mat)
    fig.colorbar(im, ax = ax)

plt.tight_layout()
plt.savefig(f'analyse/midtraining/principal_angles_heatmap_U.png', dpi=600)
plt.close()
