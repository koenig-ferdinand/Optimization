"""
Compute SVD for all extracted weight matrices.
"""

import torch
from collections import defaultdict


def compute_all_svds(weight_data, n_layers):
    """
    Compute SVD for every weight matrix.

    Args:
        weight_data: dict[layer_idx][weight_type] = tensor
        n_layers: total number of layers

    Returns:
        svd_data: dict[layer_idx][weight_type] = {"U": U, "S": S, "Vh": Vh, "shape": tuple}
    """
    svd_data = defaultdict(dict)

    for layer_idx in sorted(weight_data.keys()):
        for name, W in weight_data[layer_idx].items():
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                svd_data[layer_idx][name] = {
                    "U": U,
                    "S": S,
                    "Vh": Vh,
                    "shape": W.shape,
                }
            except Exception as e:
                print(f"  SVD failed for layer {layer_idx} {name}: {e}")

        if (layer_idx + 1) % max(1, n_layers // 4) == 0:
            print(f"  ... layer {layer_idx + 1}/{n_layers}")

    return svd_data
