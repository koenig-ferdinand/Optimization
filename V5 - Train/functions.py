import torch
from scipy.linalg import subspace_angles
import numpy as np
from scipy.stats import gumbel_r

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
    #k = int(min(effective_rank(S_X), min(effective_rank(S_Y), 15)))
    k = 50

    U_angles = subspace_angles(U_X[:, :k].numpy(), U_Y[:, :k].numpy())
    V_angles = subspace_angles(V_X.T[:, :k].numpy(), V_Y.T[:, :k].numpy())

    return U_angles, V_angles



def SimMat(A, B): 
    # for efficiency, first normalize, then one matrix multiplication
    
    A_norm = A / np.linalg.norm(A, axis = 0, keepdims=True)
    B_norm  = B / np.linalg.norm(B, axis = 0, keepdims=True)

    C = np.abs(A_norm.T @ B_norm)

    return C 

def MaxCosSim(A, B):
    C = SimMat(A, B)

    s_A = C.max(axis=1)

    return s_A

def DOCS(A, B):
    s_A =  MaxCosSim(A, B) 
    s_B =  MaxCosSim(B, A)

    u_A, _ = gumbel_r.fit(s_A)
    u_B, _ = gumbel_r.fit(s_B)

    return (u_A + u_B)/2

def right_singular_vectors(A, k=None): 
    _, _, Vh = np.linalg.svd(A, full_matrices=False)
    V = Vh.T    
    if k is not None: 
        V = V[:, :k]
    return V

def left_singular_vectors(A, k=None):
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    if k is not None:
        U = U[:, :k]
    return U

# SPECTRAL NORM
# PRE: 1D vector of singular values
def spectral_norm(S):
    return S[0].item()

# NUCLEAR NORM
# PRE: 1D vector of singular values
def nuclear_norm(S):
    return S.sum().item()

# RANK UTILIZATION
# PRE: 1D vector of singular values, tuple shape (rows, cols)
# POST: effective_rank / min(rows, cols) in [0, 1]
def rank_utilization(S, shape):
    return effective_rank(S).item() / min(shape)

# POWER-LAW TAIL EXPONENT (Martin & Mahoney 2021)
# PRE: 1D tensor of singular values
# POST: (alpha, r_squared) — tail exponent and fit quality; alpha in [2,4] = well-trained
def fit_power_law_tail(S, tail_fraction=0.1):
    from scipy import stats
    s = S.numpy() if hasattr(S, 'numpy') else np.array(S)
    eigenvalues = s ** 2
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) < 10:
        return float('nan'), 0.0
    n_tail = max(int(len(eigenvalues) * tail_fraction), 10)
    tail = np.sort(eigenvalues)[-n_tail:][::-1]
    log_vals = np.log10(tail)
    log_rank = np.log10(np.arange(1, len(tail) + 1))
    slope, _, r_value, _, _ = stats.linregress(log_vals, log_rank)
    return -slope, r_value ** 2

# MARCHENKO-PASTUR SIGNAL FRACTION
# PRE: 1D tensor of singular values, tuple shape (rows, cols)
# POST: fraction of singular values above the MP bulk upper edge in [0, 1]
def mp_signal_fraction(S, shape):
    m, n = max(shape), min(shape)
    gamma = n / m
    s = S.numpy() if hasattr(S, 'numpy') else np.array(S)
    eigenvalues = s ** 2
    sigma2 = eigenvalues.mean()             # mean eigenvalue ≈ m·σ²_entry, correct MP scale
    lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    return int(np.sum(eigenvalues > lambda_plus)) / len(eigenvalues)
