import torch
from scipy.linalg import subspace_angles

# SVD
# POST: Vector of Singular Values (768,)
def svd(matrix): 
    U, S, V = torch.linalg.svd(matrix.float())
    return S

# STABLE RANK
# PRE: 1D vector of singular values
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

# leading singular value ratio
def ratio(S): 
    return S[0]/S[1]

def energy_k(S, threshold=0.9):
    total = (S**2).sum()
    running = 0
    for i in range(len(S)):
        running += S[i]**2
        if running/total >= threshold: return i + 1

# principle angles
def principal_angles(X, Y): 
    U_X, S_X, V_X = torch.linalg.svd(X.float())
    U_Y, S_Y, V_Y = torch.linalg.svd(Y.float())

    # consider min(effective_rank of X, Y)
    k = int(min(effective_rank(S_X), min(effective_rank(S_Y), 15)))

    U_angles = subspace_angles(U_X[:, :k].numpy(), U_Y[:, :k].numpy())
    V_angles = subspace_angles(V_X.T[:, :k].numpy(), V_Y.T[:, :k].numpy())

    return U_angles, V_angles



