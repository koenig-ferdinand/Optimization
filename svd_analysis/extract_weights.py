"""
Extract weight matrices from transformer models.

Supports:
  - Dense models: Llama/Qwen/Mistral/Gemma, GPT-2/Neo/J, BLOOM/Falcon, MPT, Phi
  - MoE models: DeepSeek V2/V3, Moonlight, Mixtral, Qwen-MoE, Grok

For MoE models, returns both:
  - weight_data: attention weights + ONE representative expert per layer
    (so all existing analysis methods work without modification)
  - expert_data: ALL expert weights for MoE-specific analysis
"""

from collections import defaultdict

# Maps architecture-specific parameter names → canonical short names
NAME_MAP = {
    # Llama-family (Qwen, Mistral, Gemma, Yi, DeepSeek, InternLM, etc.)
    "q_proj":           "Q",
    "k_proj":           "K",
    "v_proj":           "V",
    "o_proj":           "O",
    "gate_proj":        "Gate",
    "up_proj":          "Up",
    "down_proj":        "Down",
    # Attention variants
    "q_a_proj":         "Q_a",       # DeepSeek V2 low-rank Q (compress)
    "q_b_proj":         "Q_b",       # DeepSeek V2 low-rank Q (decompress)
    "kv_a_proj_with_mqa": "KV_a",    # DeepSeek V2 compressed KV
    "kv_b_proj":        "KV_b",      # DeepSeek V2 KV decompress
    # GPT-2 / GPT-Neo / GPT-J
    "c_attn":           "QKV_fused",
    "c_proj":           "O",
    "c_fc":             "Up",
    "c_fc2":            "Down",
    # BLOOM / Falcon
    "query_key_value":  "QKV_fused",
    "dense_h_to_4h":    "Up",
    "dense_4h_to_h":    "Down",
    # MPT
    "Wqkv":             "QKV_fused",
    "out_proj":         "O",
    # Phi
    "qkv_proj":         "QKV_fused",
    "fc1":              "Up",
    "fc2":              "Down",
}

LAYER_CONTAINERS = {"layers", "h", "block", "blocks", "decoder_layer", "layer",
                    "attention_layers"}


def find_layer_index(param_name):
    """Extract the integer layer index from a parameter name."""
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p in LAYER_CONTAINERS:
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
        if p.isdigit() and i > 0 and parts[i - 1] in LAYER_CONTAINERS:
            return int(p)
    return None


def find_expert_index(param_name):
    """
    Extract expert index if this is an expert weight.
    Returns (expert_idx, is_shared) or (None, False).

    Patterns:
      model.layers.X.mlp.experts.Y.gate_proj.weight     → expert Y
      model.layers.X.mlp.shared_experts.gate_proj.weight → shared expert
      model.layers.X.block_sparse_moe.experts.Y.w1       → Mixtral expert Y
    """
    parts = param_name.split(".")

    # Check for shared experts first
    if "shared_experts" in parts or "shared_expert" in parts:
        return None, True

    # Check for expert index
    for i, p in enumerate(parts):
        if p in ("experts", "local_experts"):
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1]), False

    return None, False


def is_router_weight(param_name):
    """Check if this is a MoE router/gating weight."""
    name_lower = param_name.lower()
    # Common router patterns
    router_patterns = [
        ".gate.weight",          # DeepSeek, Qwen-MoE
        ".router.weight",        # some implementations
        ".gate_network.",        # alternative naming
        "block_sparse_moe.gate", # Mixtral
    ]
    return any(p in name_lower for p in router_patterns)


def classify_weight(param_name):
    """Map a full parameter name to a canonical short name."""
    # Mixtral uses w1/w2/w3 naming
    parts = param_name.split(".")
    leaf = parts[-1] if parts else ""
    parent = parts[-2] if len(parts) > 1 else ""

    # Mixtral-style: experts.N.w1, w2, w3
    if leaf == "weight" and parent in ("w1", "w2", "w3"):
        return {"w1": "Gate", "w2": "Down", "w3": "Up"}[parent]

    for pattern, short in NAME_MAP.items():
        if pattern in param_name:
            return short
    return None


def extract_weights(model):
    """
    Extract 2D weight matrices from a transformer model.

    Returns:
        weight_data: dict[layer_idx][weight_type] = tensor
                     For MoE: includes attention + expert 0 as representative
        n_layers: int
        weight_types: sorted list of weight type names
        moe_info: dict with MoE metadata, or None for dense models
                  {"n_experts": int, "expert_data": dict, "moe_layers": set,
                   "has_shared_experts": bool, "router_data": dict}
    """
    weight_data = defaultdict(dict)
    expert_data = defaultdict(lambda: defaultdict(dict))  # [layer][expert_idx][wtype]
    shared_expert_data = defaultdict(dict)                 # [layer][wtype]
    router_data = defaultdict(None)                        # [layer] = tensor
    moe_layers = set()
    all_expert_indices = set()
    unmatched = []
    _seen_3d_types = set()  # track which 3D weight types we've logged

    for name, param in model.named_parameters():

        # ── Handle 3D batched expert weights FIRST ────────────────────
        # DeepSeek V2 / Moonlight store all experts in a single 3D tensor:
        #   experts.gate_up_proj  shape (64, 2816, 2048) — fused Gate+Up
        #   experts.down_proj     shape (64, 2048, 1408)
        # These often do NOT have ".weight" in the name.
        if param.dim() == 3 and param.shape[0] > 1:
            layer_idx = find_layer_index(name)
            if layer_idx is None:
                continue

            n_exp = param.shape[0]
            moe_layers.add(layer_idx)

            # Check for fused gate_up_proj: split into Gate and Up
            if "gate_up_proj" in name:
                half = param.shape[1] // 2
                if "gate_up_proj" not in _seen_3d_types:
                    _seen_3d_types.add("gate_up_proj")
                    print(f"  Found 3D fused Gate+Up: {name} "
                          f"({n_exp} experts, {param.shape[1]}×{param.shape[2]}) "
                          f"→ splitting into Gate ({half}×{param.shape[2]}) "
                          f"+ Up ({half}×{param.shape[2]})")

                for e_idx in range(n_exp):
                    all_expert_indices.add(e_idx)
                    full = param[e_idx].detach().float().cpu()
                    expert_data[layer_idx][e_idx]["Gate"] = full[:half, :]
                    expert_data[layer_idx][e_idx]["Up"] = full[half:, :]
                continue

            # Non-fused 3D expert weight
            short_name = classify_weight(name)
            if short_name is None:
                # Try matching without ".weight" suffix
                # e.g. "experts.down_proj" → "Down"
                for pattern, sn in NAME_MAP.items():
                    if pattern in name.split(".")[-1]:
                        short_name = sn
                        break

            if short_name is None:
                unmatched.append(f"{name} (3D: {tuple(param.shape)})")
                continue

            if short_name not in _seen_3d_types:
                _seen_3d_types.add(short_name)
                print(f"  Found 3D expert weight: {short_name} "
                      f"({n_exp} experts, {param.shape[1]}×{param.shape[2]})")

            for e_idx in range(n_exp):
                all_expert_indices.add(e_idx)
                expert_data[layer_idx][e_idx][short_name] = (
                    param[e_idx].detach().float().cpu()
                )
            continue

        # ── Handle 2D weight matrices ─────────────────────────────────
        if param.dim() != 2 or "weight" not in name:
            continue

        layer_idx = find_layer_index(name)
        if layer_idx is None:
            continue

        # Check if router
        if is_router_weight(name):
            router_data[layer_idx] = param.detach().float().cpu()
            moe_layers.add(layer_idx)
            continue

        # Check if expert
        expert_idx, is_shared = find_expert_index(name)

        short_name = classify_weight(name)
        if short_name is None:
            unmatched.append(name)
            continue

        tensor = param.detach().float().cpu()

        if expert_idx is not None:
            # Expert weight
            expert_data[layer_idx][expert_idx][short_name] = tensor
            moe_layers.add(layer_idx)
            all_expert_indices.add(expert_idx)
        elif is_shared:
            # Shared expert weight
            shared_expert_data[layer_idx][short_name] = tensor
            # Also put in main weight_data with "Shared_" prefix
            weight_data[layer_idx][f"Shared_{short_name}"] = tensor
        else:
            # Regular weight (attention, or dense MLP in non-MoE layers)
            weight_data[layer_idx][short_name] = tensor

    # ── Handle MoE: add representative expert to weight_data ──────────
    is_moe = len(moe_layers) > 0

    if is_moe:
        n_experts = max(all_expert_indices) + 1 if all_expert_indices else 0
        print(f"  MoE model detected: {n_experts} experts, "
              f"{len(moe_layers)} MoE layers out of {max(weight_data.keys()) + 1}")

        if shared_expert_data:
            print(f"  Shared experts found in {len(shared_expert_data)} layers")

        # For MoE layers, add expert 0 weights to weight_data as the
        # representative for standard analysis methods.
        # Prefix with "E0_" to distinguish from dense MLP weights.
        for layer_idx in moe_layers:
            if 0 in expert_data[layer_idx]:
                for wtype, tensor in expert_data[layer_idx][0].items():
                    # Only add if a dense version doesn't already exist
                    if wtype not in weight_data[layer_idx]:
                        weight_data[layer_idx][wtype] = tensor

        # Add router weights
        for layer_idx, tensor in router_data.items():
            if tensor is not None:
                weight_data[layer_idx]["Router"] = tensor

    # ── Validate ──────────────────────────────────────────────────────
    if unmatched and len(unmatched) <= 10:
        print(f"  Note: {len(unmatched)} 2D weights not matched:")
        for u in unmatched[:5]:
            print(f"    skipped: {u}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")
    elif unmatched:
        print(f"  Note: {len(unmatched)} 2D weights not matched (embeddings, norms, etc.)")

    if not weight_data:
        print("\n  ERROR: No weights matched! Debug with:")
        print("    for n, p in model.named_parameters():")
        print("        if p.dim() == 2: print(n, tuple(p.shape))")
        raise RuntimeError("No weight matrices extracted")

    n_layers = max(weight_data.keys()) + 1
    weight_types = sorted(set(k for d in weight_data.values() for k in d.keys()))

    # Build MoE info
    moe_info = None
    if is_moe:
        moe_info = {
            "n_experts": n_experts,
            "expert_data": dict(expert_data),       # [layer][expert][wtype]
            "shared_expert_data": dict(shared_expert_data),
            "router_data": dict(router_data),
            "moe_layers": moe_layers,
            "has_shared_experts": len(shared_expert_data) > 0,
        }

    return weight_data, n_layers, weight_types, moe_info
