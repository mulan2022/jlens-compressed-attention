#!/usr/bin/env python3
"""Addendum: raw-W_U control for the cross-layer elevation.

If E_raw(m) (energy of raw unembedding rows, J = identity) shows the same
late-layer elevation as E(l, m), the elevation reflects read-subspace/
unembedding alignment generally, not J-space structure specifically.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.remote import Weights, orthobasis, agg_energy, haar_null, DEV

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = Weights(args.model)
    d = w.cfg["hidden_size"]
    n_layers = w.cfg["num_hidden_layers"]

    W_U = w.unembed().to(DEV)
    G0 = (W_U.T @ W_U)
    del W_U

    mu0, sd0 = haar_null(G0, w.read_matrix(0).shape[0] if w.arch == "mla" else orthobasis(w.read_matrix(0)).shape[1], seed=99)
    E_raw = np.zeros(n_layers)
    for m in range(n_layers):
        B = orthobasis(w.read_matrix(m)).to(DEV)
        E_raw[m] = agg_energy(G0, B)
    print(f"raw-W_U null: μ={mu0:.4f} σ={sd0:.5f}")
    print("E_raw:", np.array2string(E_raw, precision=3, max_line_width=200))

    np.savez_compressed(Path(args.out).expanduser(),
                        E_raw=E_raw, null_mu=mu0, null_sd=sd0)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
