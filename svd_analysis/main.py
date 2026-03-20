"""
Structural Analysis of Transformer Weight Matrices via SVD

Supports single-model analysis and two-model comparison.
SVD results are cached to disk so you only compute once per model.

Workflow:
  1. First run:  set MODEL_ID, run → computes SVD, saves cache, generates plots
  2. Second run: change MODEL_ID to another model → same thing
  3. Compare:    set COMPARE to the two cache file paths → side-by-side plots
"""

import os
import time

import torch

from extract_weights import extract_weights
from compute_svd import compute_all_svds
from methods import AVAILABLE_METHODS


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                            ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  Pick ONE mode: either SINGLE MODEL or COMPARISON.                 ║
# ║  Comment out the mode you're NOT using.                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── MODE 1: Single model analysis ────────────────────────────────────
# Analyzes one model, saves plots + caches SVD to disk for later comparison.
# After running, check the cache/ folder for the .pt filename.

MODE = "single"
MODEL_ID = "Qwen/Qwen3-0.6B"
DTYPE = "float32"                      # "float32" recommended for SVD accuracy
OUTPUT_DIR = "output"

# ── MODE 2: Compare two cached models ────────────────────────────────
# First run each model individually (Mode 1) to generate cache files.
# Then switch to this mode and set the two filenames from cache/.

# MODE = "compare"
# COMPARE_A = "Qwen_Qwen3-0.6B.pt"    # ← filename from cache/ folder
# COMPARE_B = "my_muon_model.pt"       # ← filename from cache/ folder
# OUTPUT_DIR = "output/comparison"

# ── Shared settings ──────────────────────────────────────────────────
CACHE_DIR = "cache"

# Which methods to run. None = all.
# Available: "structure", "spectra", "stable_rank", "effective_rank", "alpha",
#            "distributions", "norms", "principal_angles", "alignment",
#            "cumulative_energy", "moe_experts"
METHODS = None

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  END OF CONFIG                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


def model_id_to_filename(model_id):
    """Convert 'Qwen/Qwen3-0.6B' → 'Qwen_Qwen3-0.6B.pt'"""
    return model_id.replace("/", "_").replace("\\", "_") + ".pt"


def load_hf_model(model_id, dtype):
    """Load a HuggingFace model for weight extraction."""
    from transformers import AutoModelForCausalLM

    torch_dtype = torch.float32 if dtype == "float32" else torch.float16
    print(f"Loading model: {model_id} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded successfully.")
    return model


def save_cache(svd_data, weight_types, n_layers, label, cache_path, moe_info=None):
    """Save SVD results to disk for later comparison."""
    save_dict = {
        "svd_data": {k: dict(v) for k, v in svd_data.items()},
        "weight_types": weight_types,
        "n_layers": n_layers,
        "label": label,
        "moe_info": moe_info,
    }
    torch.save(save_dict, cache_path)
    print(f"SVD cache saved: {cache_path}")


def load_cache(cache_path):
    """Load SVD results from disk."""
    print(f"Loading cached SVD: {cache_path}")
    data = torch.load(cache_path, weights_only=False)
    return data


def analyze_model(model_id, dtype, cache_dir):
    """
    Analyze a single model: load, extract weights, compute SVD, cache.
    Returns a model_data dict.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, model_id_to_filename(model_id))

    # Check cache first
    if os.path.exists(cache_path):
        print(f"Found cached SVD for {model_id}")
        data = load_cache(cache_path)
        return {
            "label": data["label"],
            "svd_data": data["svd_data"],
            "weight_types": data["weight_types"],
            "n_layers": data["n_layers"],
            "moe_info": data.get("moe_info", None),
        }

    # No cache — compute from scratch
    model = load_hf_model(model_id, dtype)

    print("\nExtracting weight matrices...")
    weight_data, n_layers, weight_types, moe_info = extract_weights(model)
    print(f"Extracted weights from {n_layers} layers, types: {weight_types}")

    print("\nComputing SVDs for main weights...")
    t0 = time.time()
    svd_data = compute_all_svds(weight_data, n_layers)
    print(f"SVD computation complete in {time.time() - t0:.1f}s")

    # For MoE: also compute SVDs for all experts
    if moe_info is not None:
        print(f"\nComputing SVDs for {moe_info['n_experts']} experts...")
        t0 = time.time()
        expert_svd = {}  # [layer][expert_idx][wtype] = {U, S, Vh, shape}
        ed = moe_info["expert_data"]
        for layer_idx in sorted(ed.keys()):
            expert_svd[layer_idx] = {}
            for expert_idx in sorted(ed[layer_idx].keys()):
                expert_svd[layer_idx][expert_idx] = {}
                for wtype, W in ed[layer_idx][expert_idx].items():
                    try:
                        import torch as _torch
                        U, S, Vh = _torch.linalg.svd(W, full_matrices=False)
                        expert_svd[layer_idx][expert_idx][wtype] = {
                            "U": U, "S": S, "Vh": Vh, "shape": W.shape
                        }
                    except Exception as e:
                        pass
            if (layer_idx + 1) % max(1, n_layers // 4) == 0:
                print(f"  ... layer {layer_idx + 1}/{n_layers}")

        moe_info["expert_svd"] = expert_svd
        # Free raw expert weights to save memory/cache space
        moe_info.pop("expert_data", None)
        moe_info.pop("shared_expert_data", None)
        moe_info.pop("router_data", None)
        print(f"Expert SVD complete in {time.time() - t0:.1f}s")

    # Free model memory
    del model, weight_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate a short label for plot legends
    label = model_id.split("/")[-1]

    # Save cache
    save_cache(svd_data, weight_types, n_layers, label, cache_path, moe_info)

    return {
        "label": label,
        "svd_data": svd_data,
        "weight_types": weight_types,
        "n_layers": n_layers,
        "moe_info": moe_info,
    }


def resolve_methods(methods_config):
    """Turn the METHODS config into a validated list of method names."""
    if methods_config is None:
        return list(AVAILABLE_METHODS.keys())

    valid = []
    for m in methods_config:
        if m not in AVAILABLE_METHODS:
            print(f"WARNING: Unknown method '{m}'. Skipping.")
            print(f"  Available: {list(AVAILABLE_METHODS.keys())}")
        else:
            valid.append(m)
    return valid


def main():
    methods_to_run = resolve_methods(METHODS)
    if not methods_to_run:
        print("No valid methods selected.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Build the models list ─────────────────────────────────────────
    models = []

    if MODE == "compare":
        # Comparison mode: load two cached results
        print("=" * 60)
        print("COMPARISON MODE")
        print("=" * 60)
        for cache_file in [COMPARE_A, COMPARE_B]:
            cache_path = os.path.join(CACHE_DIR, cache_file)
            if not os.path.exists(cache_path):
                print(f"\nERROR: Cache file not found: {cache_path}")
                print(f"Run the model first in single mode to generate the cache.")
                print(f"Check your cache/ folder for available .pt files.")
                return
            data = load_cache(cache_path)
            models.append({
                "label": data["label"],
                "svd_data": data["svd_data"],
                "weight_types": data["weight_types"],
                "n_layers": data["n_layers"],
                "moe_info": data.get("moe_info", None),
            })
        print(f"Comparing: {models[0]['label']} vs {models[1]['label']}\n")

    elif MODE == "single":
        # Single model mode
        print("=" * 60)
        print("SINGLE MODEL ANALYSIS")
        print("=" * 60)
        model_data = analyze_model(MODEL_ID, DTYPE, CACHE_DIR)
        models.append(model_data)
        print()

    else:
        print(f"ERROR: Unknown MODE '{MODE}'. Use 'single' or 'compare'.")
        return

    # ── Run methods ───────────────────────────────────────────────────
    print(f"Running {len(methods_to_run)} analysis methods:")
    print("=" * 60)

    for method_name in methods_to_run:
        method = AVAILABLE_METHODS[method_name]
        print(f"\n[{method_name}]")
        try:
            method.run(models, OUTPUT_DIR)
        except Exception as e:
            print(f"  ERROR in {method_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"All results saved to: {OUTPUT_DIR}/")
    if MODE == "compare":
        print(f"(Comparison: {models[0]['label']} vs {models[1]['label']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
