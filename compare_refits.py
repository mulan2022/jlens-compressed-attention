#!/usr/bin/env python3
"""Summarize the corpus-stability experiment: original lens vs two refits on
disjoint corpora. Prints (per model): same-layer E per fit, per-layer sign
agreement, mean |dE| between fits, shifted-diagonal E(l,l+1) per fit, and the
top cross-layer hotspot cells per fit. Inputs: out/{mla,gqa}_remote{,_alt1,_alt2}.npz
"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"

for tag, layers_expected in [("mla", None), ("gqa", None)]:
    fits = {}
    for name in ["", "_alt1", "_alt2"]:
        p = OUT / f"{tag}_remote{name}.npz"
        if not p.exists():
            print(f"[{tag}] missing {p.name}")
            continue
        z = np.load(p, allow_pickle=True)
        fits[name or "orig"] = z
    if len(fits) < 2:
        continue
    print(f"\n{'='*70}\n{tag.upper()}: {len(fits)} fits\n{'='*70}")

    ll = fits["orig"]["lens_layers"]
    E_same = {k: np.array([f["E"][i, l] for i, l in enumerate(f["lens_layers"])])
              for k, f in fits.items()}
    E_next = {k: np.array([f["E"][i, l + 1] for i, l in enumerate(f["lens_layers"])])
              for k, f in fits.items()}
    Z_same = {k: np.array([(f["E"][i, l] - f["null_mu"][i]) / f["null_sd"][i]
                           for i, l in enumerate(f["lens_layers"])])
              for k, f in fits.items()}

    hdr = "layer  " + "".join(f"{k:>22}" for k in fits)
    print(hdr)
    print("       " + "".join(f"{'E(l,l)':>10}{'z':>5}{'E(l,l+1)':>7}" for _ in fits))
    for j, l in enumerate(ll):
        row = f"L{l:<4}"
        for k in fits:
            row += f"{E_same[k][j]:>10.3f}{Z_same[k][j]:>+5.1f}{E_next[k][j]:>7.3f}"
        print(row)

    print("\nsame-layer E: mean per fit   ",
          "  ".join(f"{k}={E_same[k].mean():.4f}" for k in fits))
    print("shifted  E(l,l+1): mean     ",
          "  ".join(f"{k}={E_next[k].mean():.4f}" for k in fits))
    # sign agreement of per-layer deviations from baseline
    devs = {k: E_same[k] - 0.25 for k in fits}
    ref = fits and devs["orig"]
    for k in fits:
        if k == "orig":
            continue
        agree = np.mean(np.sign(devs[k]) == np.sign(ref))
        mad = np.mean(np.abs(devs[k] - ref))
        print(f"orig vs {k}: sign agreement {agree:.0%} "
              f"({int(np.sum(np.sign(devs[k])==np.sign(ref)))}/{len(ll)}), "
              f"mean |dE| diff {mad:.4f}")
    # cross-layer hotspots: top-3 cells per fit
    for k, f in fits.items():
        E, lls = f["E"], f["lens_layers"]
        flat = [(E[i, m], int(l), m) for i, l in enumerate(lls)
                for m in range(E.shape[1]) if m != l]
        top = sorted(flat, reverse=True)[:3]
        print(f"  {k} top cross-layer cells: " +
              ", ".join(f"L{l}->read{m}: {e:.3f}" for e, l, m in top))
