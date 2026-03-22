# PACKAGES
import torch

# FILES
import functions

# SETUP: 
# functions file contains modular functions (svd, stable_rank, effective_rank,...)
# configure new modular functions in main/ separate files 

# MAIN.PY
# read from weightspace 
data = torch.load('data/state_step000100.pt', map_location = 'cpu')
model = data['model']

# ITERATE over matrices
#for name, matrix in model.items(): 
#    print(f"{name:45s} {tuple(matrix.shape)}")

# ACCESS matrices, where hidden layers range: 0-11
# model['transformer.h.0.attn.c_attn.weight'] # layer 0, attention QKV (3*768, 768)
# model['transformer.h.0.attn.c_proj.weight'] # layer 0, attention output
# model['transformer.h.0.mlp.c_fc.weight'] # layer 0, MLP expand
# model['transformer.h.0.mlp.c_proj.weight'] # layer 0, MLP compress

# SVD 
S = functions.svd(model['transformer.h.11.attn.c_proj.weight'])
print(f"\n SINGULAR VALUES: \n {S}")

# STABLE RANK
# stable_rank = functions.stable_rank(S)
# print(f"STABLE RANK: {stable_rank}")

# EFFECTIVE RANK
effective_rank = functions.effective_rank(S)
print(f"EFFECTIVE RANK: {effective_rank}")

# CONDITION NUMBER
# cond_num = functions.condition_number(S)
# print(f"CONDITION NUMBER: {cond_num}")





