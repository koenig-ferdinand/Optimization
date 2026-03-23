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
data = torch.load(f'data/muon/state_step006200.pt', map_location = 'cpu')
model_muon = data['model']
data = torch.load(f'data/adamw/state_step006200.pt', map_location = 'cpu')
model_adamw = data['model']
models = [['muon', model_muon], ['adamw', model_adamw]]


# leading singular value ratio
appendices = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
for appendix in appendices:
    print(f"\nAPPENDIX: {appendix}")
    if appendix == 'attn.c_attn':   
        for i in range(12): 
            print(f"# LAYER: {i}")
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

for mat in matrix_types: 
    fig, axes = plt.subplots(4, 3, figsize= (18, 22))
    fig.suptitle(f"Principal Angle Distribution - {mat}")

    for i in range(12): 
        row = i//3
        col = i%3
        ax = axes[row][col]

        U_angles = results[mat][i]['U_distribution']
        V_angles = results[mat][i]['V_distribution']


        ax.plot(np.degrees(U_angles), label='U')
        ax.plot(np.degrees(V_angles), label='V')
        ax.set_title(f'Layer {i}')
        ax.set_xlabel('Singular Vector Index')
        ax.set_ylabel('Principal Angle (degrees)')
        ax.set_ylim(0, 95)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'analyse/plots/principal_angles_{mat}.png', dpi=500)
    plt.close()
