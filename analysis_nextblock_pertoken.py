#!/usr/bin/env python3
"""Per-token z-scores against the NEXT block's read subspace (GQA, local).

Companion to Table `tab:nextblock`: the aggregate E(l,l+1) elevation could in
principle be driven by the norm weighting of G_l. Here we recompute the
per-readout-token z distribution (same token set as v2lite/qwen25 stats)
against the read subspace of block l+1 instead of block l. Qwen only, since
its weights + lens are local.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).parent
OUT = ROOT / "out"

qwen_snap = sorted((ROOT / "models" / "models--Qwen--Qwen2.5-3B" / "snapshots").iterdir())[-1]
sf_files = sorted(qwen_snap.glob("model-*.safetensors"))
cfg = json.loads((qwen_snap / "config.json").read_text())
D = cfg["hidden_size"]


def load_tensor(name):
    for f in sf_files:
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name).float()
    raise KeyError(name)


lens = torch.load(OUT / "qwen25_lens.pt", map_location="cpu", weights_only=True)
J = {l: t.float() for l, t in lens["J"].items()}
W_E = load_tensor("model.embed_tokens.weight")  # tied -> also unembedding

d = np.load(OUT / "qwen25_stats.npz", allow_pickle=True)
layers_arr, tids = d["layers_arr"], d["token_ids"]
lens_layers = sorted(set(layers_arr.tolist()))

rng = np.random.RandomState(42)


def basis_and_null(W, n_null=2000):
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    keep = int((S > S[0] * 1e-10).sum())
    B = Vh[:keep, :].T.numpy()
    vecs = rng.randn(n_null, W.shape[1]).astype(np.float64)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    null = ((vecs @ B) ** 2).sum(axis=1)
    return B, null.mean(), null.std()


def read_matrix(layer):
    return torch.cat([load_tensor(f"model.layers.{layer}.self_attn.k_proj.weight"),
                      load_tensor(f"model.layers.{layer}.self_attn.v_proj.weight")], dim=0)


all_z = []
for l in lens_layers:
    B, mu0, sd0 = basis_and_null(read_matrix(l + 1))  # NEXT block's read subspace
    idx = layers_arr == l
    toks = sorted(set(tids[idx].tolist()))
    M = (W_E[toks] @ J[l]).numpy()
    frac = ((M @ B) ** 2).sum(axis=1) / (M ** 2).sum(axis=1)
    zmap = {t: (f - mu0) / sd0 for t, f in zip(toks, frac)}
    zs = np.array([zmap[int(t)] for t in tids[idx]])
    all_z.append(zs)
    print(f"L{l} vs read L{l+1}: mean frac={frac.mean():.4f} (null {mu0:.4f}) "
          f"mean z={zs.mean():+.2f} sd_z={zs.std():.2f}")

z = np.concatenate(all_z)
print(f"\nOVERALL (per-token, shifted to l+1): mean z={z.mean():+.3f} sd={z.std():.2f} "
      f"P(|z|>2)={(np.abs(z) > 2).mean():.3f} n={len(z)}")
