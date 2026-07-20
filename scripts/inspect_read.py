#!/usr/bin/env python3
# Copyright 2026.
# SPDX-License-Identifier: Apache-2.0
"""Inspect attention read/OV subspaces — supports MLA (DeepSeek-V2) and GQA/MHA.

Architecture auto-detection from config.json:
  - MLA  (DeepSeek-V2): kv_lora_rank in config → W_DKV row space = read subspace
  - GQA  (Qwen2, Llama): num_key_value_heads → concat(W_K, W_V) row space
  - MHA  (standard): num_key_value_heads == num_attention_heads

For all architectures, computes:
  1. Read subspace basis (per layer)
  2. OV circuit SVD / effective rank
  3. J-space energy fraction in read subspace (if lens provided)
  4. Read subspace union coverage across all layers

Usage:
  # Quick architecture check (no weights):
  python inspect_attention_read.py --model models/Qwen2.5-3B --dry-run

  # Full SVD + unembedding overlap:
  python inspect_attention_read.py --model models/Qwen2.5-3B --unembed --layers 5 10 15 20 25 30

  # J-space projection (after fitting lens):
  python inspect_attention_read.py --model models/Qwen2.5-3B --jspace-lens out/qwen25_lens.pt --layers 8 12 16 20 24 28 32
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import torch


# ── config helpers ──────────────────────────────────────────────────────────

def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def detect_arch(cfg: dict) -> str:
    """Return 'mla', 'gqa', or 'mha'."""
    if "kv_lora_rank" in cfg:
        return "mla"
    n_q = cfg.get("num_attention_heads", 0)
    n_kv = cfg.get("num_key_value_heads", n_q)
    if n_kv < n_q:
        return "gqa"
    return "mha"


# ── safetensors loader ──────────────────────────────────────────────────────

def _tensor_loader(model_dir: str):
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
        shards = set(weight_map.values())
        if not all(os.path.exists(os.path.join(model_dir, s)) for s in shards):
            return None

        def load(name: str) -> torch.Tensor:
            shard = weight_map[name]
            with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
                return f.get_tensor(name).float()
        return load

    if os.path.exists(single):
        def load(name: str) -> torch.Tensor:
            with safe_open(single, framework="pt") as f:
                return f.get_tensor(name).float()
        return load

    return None


# ── SVD helpers ─────────────────────────────────────────────────────────────

def effective_rank(singular_values: torch.Tensor) -> dict:
    sv = singular_values
    energy = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    return {
        "n_sv": len(sv),
        "rank@99%": int((energy < 0.99).sum()) + 1,
        "rank@99.9%": int((energy < 0.999).sum()) + 1,
        "numerical_rank": int((sv > sv[0] * 1e-3).sum()),
        "sv_max": float(sv[0]),
        "sv_min": float(sv[-1]),
    }


def orthonormal_basis(rows: torch.Tensor) -> torch.Tensor:
    """Given [k, d] matrix, return orthonormal basis [d, k] of its row space."""
    _, S, Vh = torch.linalg.svd(rows, full_matrices=False)
    # Keep only directions with non-negligible singular values
    keep = (S > S[0] * 1e-10).sum().item()
    return Vh[:keep, :].T  # [d, keep]


def energy_fraction(dirs: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of Frobenius energy of `dirs` [n, d] inside `basis` [d, k]."""
    inside = (dirs @ basis).pow(2).sum()
    total = dirs.pow(2).sum()
    return float(inside / total)


# ── per-architecture subspace extraction ────────────────────────────────────

def get_weight_names_mla(layer: int) -> dict:
    pre = f"model.layers.{layer}.self_attn"
    return {
        "kv_down": f"{pre}.kv_a_proj_with_mqa.weight",   # [r+rope, d]
        "kv_up":   f"{pre}.kv_b_proj.weight",            # [H*(nope+vdim), r]
        "o_proj":  f"{pre}.o_proj.weight",               # [d, H*vdim]
        "q_proj":  f"{pre}.q_proj.weight",               # [H*q_head, d]
    }


def get_weight_names_gqa(layer: int) -> dict:
    pre = f"model.layers.{layer}.self_attn"
    return {
        "q_proj":  f"{pre}.q_proj.weight",
        "k_proj":  f"{pre}.k_proj.weight",
        "v_proj":  f"{pre}.v_proj.weight",
        "o_proj":  f"{pre}.o_proj.weight",
    }


def read_basis_mla(load, cfg: dict, layer: int) -> torch.Tensor:
    """MLA: row space of W_DKV[:r, :] — the shared content-KV read directions."""
    r = cfg["kv_lora_rank"]
    names = get_weight_names_mla(layer)
    W_DKV = load(names["kv_down"])[:r, :]  # [r, d]
    return orthonormal_basis(W_DKV)


def read_basis_gqa(load, cfg: dict, layer: int) -> torch.Tensor:
    """GQA/MHA: row space of all KV head projections (K + V combined).

    For GQA with g KV groups, this gives g * 2 * head_dim directions.
    For Qwen2.5-3B: 2 KV heads × (128 K + 128 V) = 512 dims.
    """
    d = cfg["hidden_size"]
    n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = d // cfg["num_attention_heads"]
    names = get_weight_names_gqa(layer)
    W_K = load(names["k_proj"])  # [n_kv * head_dim, d]
    W_V = load(names["v_proj"])  # [n_kv * head_dim, d]
    combined = torch.cat([W_K, W_V], dim=0)  # [2 * n_kv * head_dim, d]
    return orthonormal_basis(combined)


def ov_circuit_mla(load, cfg: dict, layer: int) -> torch.Tensor:
    """MLA OV circuit: W_O @ blkdiag(W_UV) @ W_DKV. Rank <= kv_lora_rank."""
    d, H, r = cfg["hidden_size"], cfg["num_attention_heads"], cfg["kv_lora_rank"]
    nope, vdim = cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    names = get_weight_names_mla(layer)

    W_DKV = load(names["kv_down"])[:r, :]                       # [r, d]
    kv_b = load(names["kv_up"]).view(H, nope + vdim, r)        # [H, nope+vdim, r]
    W_UV = kv_b[:, nope:, :]                                    # [H, vdim, r]
    W_O = load(names["o_proj"]).view(d, H, vdim)               # [d, H, vdim]

    ov = torch.zeros(d, d)
    for h in range(H):
        ov += W_O[:, h, :] @ W_UV[h] @ W_DKV                   # [d,vdim]@[vdim,r]@[r,d]
    return ov


def ov_circuit_gqa(load, cfg: dict, layer: int) -> torch.Tensor:
    """GQA OV circuit: sum over KV heads of W_O_slice @ W_V_h.

    Each KV head's value is broadcast to q_heads_per_kv query heads.
    """
    d = cfg["hidden_size"]
    n_q = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_q)
    head_dim = d // n_q
    q_per_kv = n_q // n_kv

    names = get_weight_names_gqa(layer)
    W_V = load(names["v_proj"]).view(n_kv, head_dim, d)         # [n_kv, hd, d]
    W_O = load(names["o_proj"]).view(d, n_q, head_dim)          # [d, n_q, hd]

    ov = torch.zeros(d, d)
    for kv_h in range(n_kv):
        # Query heads that share this KV head: kv_h*q_per_kv ... (kv_h+1)*q_per_kv
        for q_h in range(kv_h * q_per_kv, (kv_h + 1) * q_per_kv):
            ov += W_O[:, q_h, :] @ W_V[kv_h]                   # [d,hd]@[hd,d]
    return ov


def read_dim(cfg: dict) -> int:
    """Theoretical dimension of the KV read subspace."""
    arch = detect_arch(cfg)
    d = cfg["hidden_size"]
    if arch == "mla":
        return cfg["kv_lora_rank"]
    else:
        n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        head_dim = d // cfg["num_attention_heads"]
        return 2 * n_kv * head_dim  # K + V for all KV heads


# ── inspection ──────────────────────────────────────────────────────────────

def inspect_layer(load, cfg, layer, unembed=None):
    arch = detect_arch(cfg)
    d = cfg["hidden_size"]
    rdim = read_dim(cfg)
    baseline = rdim / d

    print(f"\n--- layer {layer} ({arch.upper()}) ---")

    # Read subspace basis
    if arch == "mla":
        basis = read_basis_mla(load, cfg, layer)
        ov = ov_circuit_mla(load, cfg, layer)
    else:
        basis = read_basis_gqa(load, cfg, layer)
        ov = ov_circuit_gqa(load, cfg, layer)

    print(f"  Read subspace: {basis.shape[1]} dims (theoretical max {rdim})")

    # OV circuit rank
    sv_ov = torch.linalg.svdvals(ov)
    er = effective_rank(sv_ov)
    print(f"  OV circuit [{d},{d}]: numerical_rank={er['numerical_rank']}, "
          f"rank@99%={er['rank@99%']}, sv_max={er['sv_max']:.3f}, sv_min={er['sv_min']:.2e}")

    # Unembedding overlap (logit-lens proxy)
    if unembed is not None:
        frac = energy_fraction(unembed, basis)
        verdict = "IN read subspace" if frac > 1.5 * baseline else "bypasses (residual/MLP)"
        print(f"  unembed energy in read subspace: {frac:.3f}  "
              f"(random baseline {baseline:.3f})  -> {verdict}")


def union_coverage(load, cfg, layers):
    """How much of the residual is covered by the union of all attention read subspaces."""
    arch = detect_arch(cfg)
    d = cfg["hidden_size"]
    n_layers = len(layers)

    if arch == "mla":
        r = cfg["kv_lora_rank"]
        stacked = torch.cat(
            [load(f"model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight")[:r, :]
             for l in layers], dim=0)
    else:
        n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        head_dim = d // cfg["num_attention_heads"]
        rows = []
        for l in layers:
            names = get_weight_names_gqa(l)
            W_K = load(names["k_proj"])
            W_V = load(names["v_proj"])
            rows.append(torch.cat([W_K, W_V], dim=0))
        stacked = torch.cat(rows, dim=0)

    sv = torch.linalg.svdvals(stacked)
    covered = int((sv > sv[0] * 1e-3).sum())
    print(f"\n=== union of {n_layers} layers' read subspaces ===")
    print(f"  stacked {stacked.shape[0]} read directions -> covers rank {covered}/{d} "
          f"of the residual stream")
    if covered >= d:
        print("  Attention collectively reads the entire residual stream.")
    else:
        print(f"  {d - covered} residual dims are read by NO attention layer "
              f"(pure residual/MLP channels).")


def arch_summary(cfg: dict) -> None:
    """Print architecture summary."""
    arch = detect_arch(cfg)
    d = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    hd = d // H
    n_kv = cfg.get("num_key_value_heads", H)
    rdim = read_dim(cfg)

    print("=" * 68)
    print(f"Architecture: {arch.upper()}  |  d={d}  layers={L}  heads={H}/{n_kv}"
          f"  head_dim={hd}")
    print(f"  KV read subspace dimension: {rdim}  (baseline energy = {rdim}/{d}"
          f" = {rdim/d:.3f})")
    if arch == "mla":
        print(f"  kv_lora_rank={cfg['kv_lora_rank']}  "
              f"q_lora_rank={cfg.get('q_lora_rank', 'None')}")
        print(f"  qk_nope_head_dim={cfg['qk_nope_head_dim']}  "
              f"qk_rope_head_dim={cfg['qk_rope_head_dim']}  "
              f"v_head_dim={cfg['v_head_dim']}")
        print(f"  Compression: {H * cfg['qk_nope_head_dim']} → "
              f"{cfg['kv_lora_rank']} "
              f"({H * cfg['qk_nope_head_dim'] / cfg['kv_lora_rank']:.1f}x)")
    elif arch == "gqa":
        print(f"  GQA ratio: {H}/{n_kv} = {H/n_kv:.0f}x query heads per KV head")
    print("=" * 68)


def project_jspace(lens_path, load, cfg, layers):
    """Project fitted J-lens vectors onto per-layer read subspaces."""
    from minimal_jlens import JLens

    arch = detect_arch(cfg)
    d = cfg["hidden_size"]
    rdim = read_dim(cfg)
    baseline = rdim / d

    lens = JLens.load(lens_path)
    try:
        W_U = load("lm_head.weight")  # [vocab, d]
    except KeyError:
        W_U = load("model.embed_tokens.weight")

    print(f"\n=== J-space vs read subspace ({arch.upper()}, lens: {lens_path}) ===")
    print(f"  Read subspace dim: {rdim}, random baseline: {baseline:.3f}")
    print(f"  {'Layer':<6} {'Energy frac':<14} {'vs baseline':<12} {'Verdict'}")

    for l in layers:
        if l not in lens.jacobians:
            print(f"  L{l:<5} {'—':<14} {'—':<12} (no lens data)")
            continue

        if arch == "mla":
            basis = read_basis_mla(load, cfg, l)
        else:
            basis = read_basis_gqa(load, cfg, l)

        jvecs = W_U @ lens.jacobians[l].to(W_U.dtype)  # [vocab, d]
        frac = energy_fraction(jvecs, basis)
        diff = frac - baseline
        verdict = ("ABOVE baseline" if diff > 0.03 else
                   "BELOW baseline" if diff < -0.03 else
                   "AT baseline")

        print(f"  L{l:<5} {frac:.4f}           {diff:+.4f}        {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect attention read/OV subspaces for any decoder architecture")
    ap.add_argument("--model", default="models/Qwen2.5-3B")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[0, 5, 10, 15, 20, 25, 30, 35])
    ap.add_argument("--unembed", action="store_true",
                    help="Measure unembedding (logit-lens proxy) overlap with read subspace")
    ap.add_argument("--union", action="store_true",
                    help="Measure union coverage of all layers' read subspaces")
    ap.add_argument("--jspace-lens", default=None,
                    help="Path to fitted JLens (.pt) for J-space projection")
    ap.add_argument("--dry-run", action="store_true",
                    help="Config-only, no weight loading")
    args = ap.parse_args()

    cfg = load_config(args.model)
    arch_summary(cfg)

    if args.dry_run:
        return

    load = _tensor_loader(args.model)
    if load is None:
        print("\n[weights not present — dry run. Download model first.]")
        return

    # Validate layers
    valid_layers = [l for l in args.layers if l < cfg["num_hidden_layers"]]

    # Qwen2 ties embeddings: lm_head = embed_tokens
    if args.unembed:
        try:
            W_U = load("lm_head.weight")
        except KeyError:
            W_U = load("model.embed_tokens.weight")
    else:
        W_U = None

    print("\nComputing per-layer SVD spectra (reads a few MB per tensor):")
    for layer in valid_layers:
        inspect_layer(load, cfg, layer, unembed=W_U)

    if args.union:
        union_coverage(load, cfg, valid_layers)

    if args.jspace_lens is not None:
        project_jspace(args.jspace_lens, load, cfg, valid_layers)


if __name__ == "__main__":
    main()
