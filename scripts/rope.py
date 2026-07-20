#!/usr/bin/env python3
"""RoPE-inclusive MLA read subspace (rank 576, baseline 0.281).

DeepSeek-V2's kv_a_proj_with_mqa emits [c^KV (512 latent); k^R (64 decoupled
RoPE key)] — both are direct linear reads of the residual stream. The primary
analysis (analysis_remote.py) uses only the first kv_lora_rank rows. This
script repeats the same-layer energy measurement with the full 576-row read
map, as a robustness variant requested in paper review.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.remote import Weights, orthobasis, agg_energy, haar_null, DEV

torch.set_grad_enabled(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = Weights(args.model)
    assert w.arch == "mla", "RoPE-inclusive variant only applies to MLA"
    d = w.cfg["hidden_size"]
    n_layers = w.cfg["num_hidden_layers"]
    print(f"arch={w.arch} d={d} layers={n_layers} device={DEV}")

    lens = torch.load(Path(args.lens).expanduser(), map_location="cpu", weights_only=True)
    J = {l: t.float() for l, t in lens["J"].items()}
    lens_layers = sorted(J.keys())

    W_U = w.unembed().to(DEV)
    G = {}
    for l in lens_layers:
        M = W_U @ J[l].to(DEV)
        G[l] = (M.T @ M).cpu()
        del M
        print(f"  G[{l}] done", flush=True)
    del W_U
    if DEV == "cuda":
        torch.cuda.empty_cache()

    def read_matrix_rope(layer):
        # full kv_a_proj_with_mqa output: [c^KV (r_latent); k^R (qk_rope_head_dim)]
        return w.get(f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight")

    bases = {m: orthobasis(read_matrix_rope(m)) for m in range(n_layers)}
    r = bases[lens_layers[0]].shape[1]
    print(f"rope-inclusive read rank r={r}, baseline r/d={r/d:.4f}")

    E = np.zeros((len(lens_layers), n_layers))
    Z = np.zeros_like(E)
    null_mu = np.zeros(len(lens_layers))
    null_sd = np.zeros(len(lens_layers))
    for i, l in enumerate(lens_layers):
        Gl = G[l].to(DEV)
        mu0, sd0 = haar_null(Gl, r, seed=3000 + l)
        null_mu[i], null_sd[i] = mu0, sd0
        for m in range(n_layers):
            E[i, m] = agg_energy(Gl, bases[m].to(DEV))
            Z[i, m] = (E[i, m] - mu0) / sd0
        z_next = (E[i, l + 1] - mu0) / sd0 if l + 1 < n_layers else float("nan")
        print(f"  L{l}: null mu={mu0:.4f} sd={sd0:.5f} | same E={E[i, l]:.4f} "
              f"z={Z[i, l]:+.1f} | next E={E[i, l + 1]:.4f} z={z_next:+.1f}", flush=True)
        Gl = Gl.cpu()
        if DEV == "cuda":
            torch.cuda.empty_cache()

    np.savez_compressed(
        Path(args.out).expanduser(),
        lens_layers=np.array(lens_layers), n_layers=n_layers, d=d, r=r,
        E=E, Z=Z, null_mu=null_mu, null_sd=null_sd,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
