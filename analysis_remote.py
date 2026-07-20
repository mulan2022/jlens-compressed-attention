#!/usr/bin/env python3
"""Remote (A6000) analyses: cross-layer projection, Haar-null aggregate z,
positive control, RMSNorm-gamma-adjusted robustness.

Loads only weight matrices via safetensors slicing (no model forward).
Usage:
  python analysis_remote.py --model ~/jspace/models/DeepSeek-V2-Lite-Chat \
      --lens ~/jspace/out/v2lite_chat_lens.pt --out /tmp/mla_remote.npz
  python analysis_remote.py --model ~/jspace/models/Qwen2.5-3B \
      --lens /data29T/model_path/qwen25_lens.pt --out /tmp/gqa_remote.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_HAAR = 200


class Weights:
    def __init__(self, model_dir):
        self.dir = Path(model_dir).expanduser()
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.files = sorted(self.dir.glob("model-*.safetensors")) or [self.dir / "model.safetensors"]
        self.index = {}
        for f in self.files:
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    self.index[k] = f

    def get(self, name):
        with safe_open(self.index[name], framework="pt") as sf:
            return sf.get_tensor(name).float()

    @property
    def arch(self):
        if "kv_lora_rank" in self.cfg:
            return "mla"
        return "gqa" if self.cfg.get("num_key_value_heads", 0) < self.cfg["num_attention_heads"] else "mha"

    def unembed(self):
        if "lm_head.weight" in self.index:
            return self.get("lm_head.weight")
        return self.get("model.embed_tokens.weight")

    def read_matrix(self, layer, gamma_adjust=False):
        """Raw read map W_read s.t. attention reads W_read @ norm(h)."""
        if self.arch == "mla":
            r = self.cfg["kv_lora_rank"]
            W = self.get(f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight")[:r, :]
        else:
            W_K = self.get(f"model.layers.{layer}.self_attn.k_proj.weight")
            W_V = self.get(f"model.layers.{layer}.self_attn.v_proj.weight")
            W = torch.cat([W_K, W_V], dim=0)
        if gamma_adjust:
            g = self.get(f"model.layers.{layer}.input_layernorm.weight")
            W = W * g.unsqueeze(0)  # W @ diag(gamma)
        return W


def orthobasis(W):
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    keep = int((S > S[0] * 1e-10).sum())
    return Vh[:keep, :].T.contiguous()  # [d, r]


def agg_energy(G, B):
    """tr(B^T G B) / tr(G) — vocab-aggregate energy fraction in subspace B."""
    GB = G @ B
    return float((B * GB).sum() / torch.diagonal(G).sum())


def haar_null(G, r, n=N_HAAR, seed=0):
    d = G.shape[0]
    gen = torch.Generator(device=DEV).manual_seed(seed)
    vals = []
    for _ in range(n):
        A = torch.randn(d, r, generator=gen, device=DEV)
        Q, _ = torch.linalg.qr(A)
        vals.append(agg_energy(G, Q))
    v = np.array(vals)
    return v.mean(), v.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = Weights(args.model)
    d = w.cfg["hidden_size"]
    n_layers = w.cfg["num_hidden_layers"]
    print(f"arch={w.arch} d={d} layers={n_layers} device={DEV}")

    lens = torch.load(Path(args.lens).expanduser(), map_location="cpu", weights_only=True)
    J = {l: t.float() for l, t in lens["J"].items()}
    lens_layers = sorted(J.keys())
    print(f"lens layers: {lens_layers}, n_prompts={lens['n_prompts']}")

    W_U = w.unembed().to(DEV)
    print(f"W_U: {tuple(W_U.shape)}")

    # G matrices per lens layer
    G = {}
    for l in lens_layers:
        M = W_U @ J[l].to(DEV)
        G[l] = (M.T @ M).cpu()
        del M
        print(f"  G[{l}] done")
    W_U_cpu = W_U.cpu()
    del W_U
    torch.cuda.empty_cache()

    # read bases for all layers (raw + gamma-adjusted)
    bases, bases_g = {}, {}
    for m in range(n_layers):
        bases[m] = orthobasis(w.read_matrix(m))
        bases_g[m] = orthobasis(w.read_matrix(m, gamma_adjust=True))
    r = bases[lens_layers[0]].shape[1]
    print(f"read rank r={r}, baseline r/d={r/d:.4f}")

    # 1) pairwise cross-layer energy matrix + Haar-null z
    E = np.zeros((len(lens_layers), n_layers))
    Eg = np.zeros_like(E)
    Z = np.zeros_like(E)
    null_mu = np.zeros(len(lens_layers))
    null_sd = np.zeros(len(lens_layers))
    for i, l in enumerate(lens_layers):
        Gl = G[l].to(DEV)
        mu0, sd0 = haar_null(Gl, r, seed=1000 + l)
        null_mu[i], null_sd[i] = mu0, sd0
        for m in range(n_layers):
            E[i, m] = agg_energy(Gl, bases[m].to(DEV))
            Eg[i, m] = agg_energy(Gl, bases_g[m].to(DEV))
            Z[i, m] = (E[i, m] - mu0) / sd0
        Gl = Gl.cpu()
        torch.cuda.empty_cache()
        print(f"  L{l}: null μ={mu0:.4f} σ={sd0:.5f} | same-layer E={E[i, lens_layers.index(l) if False else m and 0 or 0]:.4f}")
        print(f"    E row: {np.array2string(E[i], precision=3, max_line_width=200)}")

    # 2) downstream-union saturation: rank and energy of union of read
    #    subspaces of layers m > l, as a function of how many layers included
    union = {}
    for i, l in enumerate(lens_layers):
        downstream = list(range(l + 1, n_layers))
        Gl = G[l].to(DEV)
        rows_acc = []
        ranks, energies, baselines = [], [], []
        for m in downstream:
            rows_acc.append(w.read_matrix(m))
            W_stack = torch.cat(rows_acc, dim=0)
            B = orthobasis(W_stack).to(DEV)
            ranks.append(B.shape[1])
            energies.append(agg_energy(Gl, B))
            baselines.append(B.shape[1] / d)
        union[l] = {"downstream": downstream, "rank": ranks,
                    "energy": energies, "baseline": baselines}
        Gl = Gl.cpu()
        torch.cuda.empty_cache()
        print(f"  union L{l}: first ranks {ranks[:5]} ... final rank {ranks[-1]} "
              f"(d={d}); E_first={energies[0]:.3f} E_final={energies[-1]:.3f}")

    # 3) positive control: alpha-mixture recovery through the same estimator
    gen = torch.Generator().manual_seed(7)
    l0 = lens_layers[len(lens_layers) // 2]
    B = bases[l0]
    P = B @ B.T
    alphas = np.linspace(0, 1, 11)
    recov_mean, recov_sd = [], []
    for a in alphas:
        outs = []
        for _ in range(500):
            v = torch.randn(d, generator=gen)
            v_in = P @ v
            v_out = v - v_in
            v_in /= v_in.norm()
            v_out /= v_out.norm()
            u = np.sqrt(a) * v_in + np.sqrt(1 - a) * v_out
            outs.append(float((B.T @ u).pow(2).sum() / u.pow(2).sum()))
        outs = np.array(outs)
        recov_mean.append(outs.mean())
        recov_sd.append(outs.std())
    print("  positive control (alpha -> measured energy):")
    for a, mzz in zip(alphas, recov_mean):
        print(f"    α={a:.1f} -> {mzz:.4f}")

    # 3b) realistic positive control: rows of the key up-projection path
    # (directions the attention actually reads) must score energy ≈ 1
    Wr = w.read_matrix(l0)
    dirs = Wr[:16] / Wr[:16].norm(dim=1, keepdim=True)
    e_real = ((dirs @ B).pow(2).sum(dim=1)).numpy()
    print(f"  read-map rows energy in own subspace: min={e_real.min():.6f} "
          f"mean={e_real.mean():.6f} (expect 1.0)")

    np.savez_compressed(
        Path(args.out).expanduser(),
        lens_layers=np.array(lens_layers),
        n_layers=n_layers, d=d, r=r,
        E=E, Eg=Eg, Z=Z, null_mu=null_mu, null_sd=null_sd,
        union=json.dumps(union),
        pc_alphas=alphas, pc_mean=np.array(recov_mean), pc_sd=np.array(recov_sd),
        pc_real=e_real,
        arch=w.arch,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
