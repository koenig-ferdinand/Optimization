import torch

# SVD
# POST: Vector of Singular Values (768,)
def svd(matrix): 
    U, S, V = torch.linalg.svd(matrix.float())
    return S

# STABLE RANK
# PRE: 1D vector containing singular values
def stable_rank(S): 
    return (S**2).sum() / S[0]**2

# EFFECTIVE RANK
# PRE: 1D vector containing singular values
def effective_rank(S):
    # normalize
    p = S / S.sum()
    # entopy
    p = p[p>0]
    H = -(p*p.log()).sum()
    return H.exp()

# CONDITION NUMBER
def condition_number(S): 
    return S[0]/S[-1]