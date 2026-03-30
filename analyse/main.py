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
# read from weightspace 
data = torch.load(f'data/muon/state_step006200.pt', map_location = 'cpu')
model_muon = data['model']

data = torch.load(f'data/adamw/state_step006200.pt', map_location = 'cpu')
model_adamw = data['model']

# SDV
S_muon = functions.svd(model_muon['_orig_mod.transformer.h.0.attn.c_proj.weight'])
S_adamw = functions.svd(model_adamw['_orig_mod.transformer.h.0.attn.c_proj.weight'])

# NORMALIZE
# COMMENT OUT FOR NORMALIZED and Adjust Names in plot
S_muon = S_muon / S_muon[0]
S_adamw = S_adamw / S_adamw[0]

# PLOT
plt.plot(S_muon, label = 'Muon')
plt.plot(S_adamw, label = 'AdamW', linestyle='--')
plt.ylabel('NormalizedSingular Value')
plt.xlabel('Singular Value Index')
plt.locator_params(axis='y', nbins=20)
plt.grid(True,  color='gray')
plt.legend()
plt.title('Normalized Singular Value Distribution')
plt.savefig('analyse/plots/normalized_singular_value_distribution.png', dpi = 300)