#!/usr/bin/env python3
"""Generate revision figures: cross-layer heatmaps, positive control, CJK share."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent.parent / "out"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

mla = np.load(OUT / "mla_remote.npz", allow_pickle=True)
gqa = np.load(OUT / "gqa_remote.npz", allow_pickle=True)
mla_raw = np.load(OUT / "mla_raw.npz")
gqa_raw_p = OUT / "gqa_raw.npz"
gqa_raw = np.load(gqa_raw_p) if gqa_raw_p.exists() else None
loc = json.loads((OUT / "local_analysis.json").read_text())

# ── Fig: cross-layer heatmaps ───────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 2, figsize=(10.5, 4.6),
    gridspec_kw={"height_ratios": [8, 1.1], "hspace": 0.34, "wspace": 0.16})

for col, (d, raw, name) in enumerate(
        [(mla, mla_raw, "A  DeepSeek-V2-Lite (MLA)"),
         (gqa, gqa_raw, "B  Qwen2.5-3B (GQA)")]):
    E, ll, nl = d["E"], d["lens_layers"], int(d["n_layers"])
    ax = axes[0, col]
    im = ax.imshow(E - 0.25, cmap="RdBu_r", vmin=-0.15, vmax=0.15,
                   aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(ll)), [f"L{l}" for l in ll])
    ax.set_xticks(range(0, nl, 4), [str(m) for m in range(0, nl, 4)])
    for i, l in enumerate(ll):
        ax.add_patch(plt.Rectangle((l - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.2))
    ax.set_title(name, loc="left")
    ax.set_ylabel("J-lens layer $l$" if col == 0 else "")

    ax2 = axes[1, col]
    if raw is not None:
        ax2.imshow(raw["E_raw"][None, :] - 0.25, cmap="RdBu_r",
                   vmin=-0.15, vmax=0.15, aspect="auto", interpolation="nearest")
    ax2.set_yticks([0], ["raw $W_U$"])
    ax2.set_xticks(range(0, nl, 4), [str(m) for m in range(0, nl, 4)])
    ax2.set_xlabel("read-subspace layer $m$")

cbar = fig.colorbar(im, ax=axes[:, :], fraction=0.025, pad=0.02)
cbar.set_label("energy fraction $-$ 0.250 (baseline)")
fig.savefig(OUT / "crosslayer.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved crosslayer.png")

# ── Fig: positive control (standalone) ──────────────────────────────────────
fig, axL = plt.subplots(1, 1, figsize=(4.4, 3.2))

axL.errorbar(mla["pc_alphas"], mla["pc_mean"], yerr=mla["pc_sd"],
             fmt="o-", ms=4, lw=1, capsize=2, label="MLA (L17 basis)")
axL.errorbar(gqa["pc_alphas"], gqa["pc_mean"], yerr=gqa["pc_sd"],
             fmt="s--", ms=4, lw=1, capsize=2, label="GQA (L24 basis)")
axL.plot([0, 1], [0, 1], "k:", lw=0.8, label="identity")
axL.set_xlabel(r"planted in-subspace energy $\alpha$")
axL.set_ylabel("measured energy fraction")
axL.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "positive_control.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved positive_control.png")

# ── Fig: CJK share (standalone) ─────────────────────────────────────────────
fig, axR = plt.subplots(1, 1, figsize=(4.4, 3.2))

for key, name, style in [("mla", "DeepSeek-V2-Lite (MLA)", "o-"),
                         ("gqa", "Qwen2.5-3B (GQA)", "s--")]:
    rep = loc[f"{key}_sec4"]
    layers = sorted(int(k) for k in rep["cjk_by_layer_en"])
    n_total = {"mla": 27, "gqa": 36}[key]
    depth = [l / (n_total - 1) for l in layers]
    w = [rep["cjk_by_layer_en"][str(l)]["prob_weighted"] for l in layers]
    line, = axR.plot(depth, w, style, ms=4, lw=1.2, label=name)
    axR.axhline(rep["vocab_cjk_share"], color=line.get_color(), ls=":", lw=0.9)
axR.set_xlabel("relative depth $l/(L-1)$")
axR.set_ylabel("CJK share of top-8 readout\n(prob-weighted, English prompts)")
axR.legend(frameon=False, fontsize=8)
axR.set_ylim(0, 0.7)

fig.tight_layout()
fig.savefig(OUT / "cjk_share.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved cjk_share.png")

# summary numbers for the tex
print("\nMLA cross-layer top cells (E):")
E, ll = mla["E"], mla["lens_layers"]
flat = [(E[i, m], int(l), m) for i, l in enumerate(ll) for m in range(E.shape[1])]
for e, l, m in sorted(flat, reverse=True)[:6]:
    print(f"  lens L{l} -> read L{m}: E={e:.3f}")
print("GQA cross-layer top cells (E):")
E, ll = gqa["E"], gqa["lens_layers"]
flat = [(E[i, m], int(l), m) for i, l in enumerate(ll) for m in range(E.shape[1])]
for e, l, m in sorted(flat, reverse=True)[:6]:
    print(f"  lens L{l} -> read L{m}: E={e:.3f}")
if gqa_raw is not None:
    print("GQA raw range:", gqa_raw["E_raw"].min().round(3), gqa_raw["E_raw"].max().round(3))
print("MLA raw range:", mla_raw["E_raw"].min().round(3), mla_raw["E_raw"].max().round(3))
