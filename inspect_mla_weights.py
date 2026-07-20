# Copyright 2026.
# SPDX-License-Identifier: Apache-2.0
"""Inspect the *compressed* weights of DeepSeek-V2 MLA attention.

Motivation
----------
The paper's Jacobian lens transports the residual with the end-to-end matrix
J_l = E[d h_final / d h_l]. That J_l is NOT low-rank: the residual skip
connection and the MLP keep it full-rank (2048x2048 for V2-Lite). So the lens
is architecture-agnostic and cannot, by itself, "see" MLA's compression.

MLA's compression lives INSIDE each attention block. All 16 heads read their
content keys/values through a single shared latent of width kv_lora_rank=512,
instead of the 16*128 = 2048 dims a dense MHA would use. Concretely, if you
freeze the attention pattern A (the paper's §A.7 variant), one attention block
becomes linear in the residual and its Jacobian factors as

    d attn_out / d h  =  W_O @ A @ W_UV @ (norm') @ W_DKV        # rank <= 512

read straight off the weights, no backprop. This is the "compressed Jacobian"
of one block: a 4x-smaller object than the dense-MHA equivalent.

This script quantifies that: SVD spectra and effective ranks of
  * the content KV read map  W_DKV      (residual -> 512 latent), and
  * the frozen-attention OV circuit  W_O @ W_UV @ W_DKV   (residual -> residual).

Reading a few [576,2048]/[4096,512]/[2048,2048] tensors costs a few MB, so this
runs fine on a laptop GPU / CPU — you do NOT need to load the full model.
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def rank_story(cfg: dict) -> None:
    """Config-only accounting of where MLA compresses. Needs no weights."""
    d = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    r = cfg["kv_lora_rank"]
    nope, rope, vdim = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    q_head = nope + rope

    print("=" * 68)
    print(f"DeepSeek-V2 MLA compression account  (d_model={d}, heads={H})")
    print("=" * 68)
    print(f"  kv_lora_rank (shared latent) : {r}")
    print(f"  per head: qk_nope={nope}  qk_rope={rope}  v={vdim}  q_head_dim={q_head}")
    print()
    print("  Content read (what attention reads into keys/values):")
    print(f"    dense-MHA equivalent : {H} heads x {nope} = {H * nope} dims  (full rank)")
    print(f"    MLA (shared latent)  : rank <= {r}")
    print(f"    --> compression ratio: {H * nope / r:.1f}x")
    print()
    print("  Weight shapes to inspect:")
    print(f"    kv_a_proj_with_mqa.weight : [{r + rope}, {d}]   (W_DKV[:{r}] + W_KR[{rope}])")
    print(f"    kv_b_proj.weight          : [{H * (nope + vdim)}, {r}]   (W_UK | W_UV, per head)")
    print(f"    o_proj.weight             : [{d}, {H * vdim}]   (W_O)")
    q_rank = "None (no q compression)" if cfg.get("q_lora_rank") is None else cfg["q_lora_rank"]
    print(f"    q_proj.weight             : [{H * q_head}, {d}]   (q_lora_rank={q_rank})")
    print("=" * 68)


def _tensor_loader(model_dir: str):
    """Return f(name)->Tensor that reads a single tensor via the safetensors
    index, without loading the whole model. None if shards are absent."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single = os.path.join(model_dir, "model.safetensors")
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
        shards = set(weight_map.values())
        if not all(os.path.exists(os.path.join(model_dir, s)) for s in shards):
            return None  # shards still downloading

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


def read_basis(load, cfg: dict, layer: int) -> torch.Tensor:
    """Orthonormal basis [d, r] of the attention content-read subspace
    row(W_DKV) at `layer` — the <=512 residual directions attention reads."""
    r = cfg["kv_lora_rank"]
    W_DKV = load(f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight")[:r, :]
    # rows of W_DKV span the read subspace; right singular vectors give an
    # orthonormal basis for it.
    _, _, Vh = torch.linalg.svd(W_DKV, full_matrices=False)  # Vh: [r, d]
    return Vh.T                                              # [d, r]


def energy_fraction(dirs: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of the Frobenius energy of `dirs` [n, d] that lies inside the
    subspace spanned by orthonormal `basis` [d, k]."""
    inside = (dirs @ basis).pow(2).sum()
    total = dirs.pow(2).sum()
    return float(inside / total)


def inspect_layer(load, cfg: dict, layer: int, unembed: torch.Tensor | None = None) -> None:
    d = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    r = cfg["kv_lora_rank"]
    nope, vdim = cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    pre = f"model.layers.{layer}.self_attn"

    kv_a = load(f"{pre}.kv_a_proj_with_mqa.weight")   # [r+rope, d]
    kv_b = load(f"{pre}.kv_b_proj.weight")            # [H*(nope+vdim), r]
    o = load(f"{pre}.o_proj.weight")                  # [d, H*vdim]

    W_DKV = kv_a[:r, :]                               # content read: [r, d]
    kv_b = kv_b.view(H, nope + vdim, r)
    W_UV = kv_b[:, nope:, :]                          # [H, vdim, r]
    o = o.view(d, H, vdim)                            # [d, H, vdim]

    print(f"\n--- layer {layer} ---")
    sv_read = torch.linalg.svdvals(W_DKV)
    print(f"  W_DKV content read map [{r},{d}]:  ", effective_rank(sv_read))

    # Frozen-attention OV circuit summed over heads (attention pattern = identity,
    # i.e. the per-token write map): sum_h W_O_h @ W_UV_h @ W_DKV, rank bounded by
    # the shared 512 latent even though H*vdim = 2048.
    ov = torch.zeros(d, d)
    for h in range(H):
        ov += o[:, h, :] @ W_UV[h] @ W_DKV           # [d,vdim]@[vdim,r]@[r,d]
    sv_ov = torch.linalg.svdvals(ov)
    print(f"  OV circuit W_O·W_UV·W_DKV [{d},{d}]:", effective_rank(sv_ov))
    print(f"    (rank is capped by the shared latent r={r}, not H*v={H * vdim})")

    if unembed is not None:
        # LOGIT-LENS PROXY for "does the verbalizable content live in the
        # compressed attention read subspace?". True J-space needs a fitted
        # lens (see project_jspace); here we use W_U rows as the verbalizable
        # directions. Baseline for a random r-dim subspace of R^d is r/d.
        frac = energy_fraction(unembed, read_basis(load, cfg, layer))
        base = r / d
        verdict = "IN read subspace" if frac > 1.5 * base else "BYPASSES (residual/MLP)"
        print(f"  unembed energy in read subspace: {frac:.3f}  "
              f"(random baseline {base:.3f})  -> {verdict}")


def union_coverage(load, cfg: dict) -> None:
    """How much of the 2048-dim residual is read by *any* layer's attention
    content channel. Directions never covered are private residual/MLP channels
    — candidate carriers of workspace content that bypasses compressed attention."""
    d, r, n = cfg["hidden_size"], cfg["kv_lora_rank"], cfg["num_hidden_layers"]
    stacked = torch.cat(
        [load(f"model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight")[:r, :]
         for l in range(n)],
        dim=0,
    )  # [n*r, d]
    sv = torch.linalg.svdvals(stacked)
    covered = int((sv > sv[0] * 1e-3).sum())
    print(f"\n=== union of all {n} layers' read subspaces ===")
    print(f"  stacked {n}x{r} read directions -> covers rank {covered}/{d} "
          f"of the residual stream")
    print(f"  {d - covered} residual dims are read by NO attention layer "
          f"(pure residual/MLP channels)" if covered < d else
          "  attention collectively reads the entire residual stream")


def project_jspace(lens_path: str, load, cfg: dict, layers: list[int]) -> None:
    """REAL J-space analysis (run after fitting a lens on the A6000): fraction of
    each fitted J-lens transport's energy that lies in the attention read
    subspace. Uses the W_U @ J_l rows as the J-space directions."""
    from minimal_jlens import JLens

    lens = JLens.load(lens_path)
    W_U = load("lm_head.weight")                     # [vocab, d]
    print(f"\n=== J-space vs read subspace (lens: {lens_path}) ===")
    for l in layers:
        if l not in lens.jacobians:
            continue
        # J-lens vectors = rows of W_U @ J_l (verbalizable directions at layer l).
        jvecs = W_U @ lens.jacobians[l].to(W_U.dtype)  # [vocab, d]
        frac = energy_fraction(jvecs, read_basis(load, cfg, l))
        print(f"  L{l:>2}: J-space energy in read subspace = {frac:.3f} "
              f"(baseline {cfg['kv_lora_rank'] / cfg['hidden_size']:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DeepSeek-V2-Lite")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 13, 26])
    ap.add_argument("--unembed", action="store_true",
                    help="also measure unembedding (logit-lens proxy) overlap "
                         "with each layer's read subspace")
    ap.add_argument("--union", action="store_true",
                    help="measure the union of all layers' read subspaces")
    ap.add_argument("--jspace-lens", default=None,
                    help="path to a fitted JLens (.pt) for the REAL J-space "
                         "projection (run after fitting on the A6000)")
    args = ap.parse_args()

    cfg = load_config(args.model)
    rank_story(cfg)

    load = _tensor_loader(args.model)
    if load is None:
        print("\n[weights not present yet — dry run. Re-run once shards finish "
              "downloading to see real SVD spectra.]")
        return

    W_U = load("lm_head.weight") if args.unembed else None
    print("\nWeights found — computing real SVD spectra "
          "(reads a few MB per tensor, no full-model load):")
    for layer in args.layers:
        inspect_layer(load, cfg, layer, unembed=W_U)

    if args.union:
        union_coverage(load, cfg)
    if args.jspace_lens is not None:
        project_jspace(args.jspace_lens, load, cfg, args.layers)


if __name__ == "__main__":
    main()
