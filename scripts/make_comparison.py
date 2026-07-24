#!/usr/bin/env python3
"""Paper Figure: MLA vs GQA same-layer z-score statistics (2 models x 3 panels).

Replaces the ad-hoc out/comparison_stats.png: no duplicated in-figure title
(the LaTeX caption carries that), larger fonts for print readability.
Rows: A = DeepSeek-V2-Lite (MLA), B = Qwen2.5-3B (GQA).
Cols: per-prompt x layer mean-z heatmap | z histogram vs N(0,1) | per-token
boxplot by category. Data: out/v2lite_stats.npz, out/qwen25_stats.npz.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT = Path(__file__).parent.parent / "out"

SFX = sys.argv[1] if len(sys.argv) > 1 else ""  # e.g. "_main300"

CATEGORY_COLORS = {
    "factual_en": "#2196F3", "factual_zh": "#F44336", "code_logic": "#4CAF50",
    "sentiment": "#FF9800", "random": "#9E9E9E",
}
CATEGORY_ORDER = ["factual_en", "factual_zh", "code_logic", "sentiment", "random"]

ROWS = [
    ("A", OUT / f"v2lite_stats{SFX}.npz", "DeepSeek-V2-Lite (MLA, OV-rank 512)"),
    ("B", OUT / f"qwen25_stats{SFX}.npz", "Qwen2.5-3B (GQA, OV-rank 256)"),
]

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5})

fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.4),
                         gridspec_kw={"width_ratios": [1.35, 1.0, 1.0],
                                      "wspace": 0.30, "hspace": 0.42})

for row, (tag, npz, model_name) in enumerate(ROWS):
    d = np.load(npz, allow_pickle=True)
    pl_cats, pl_pidxs = d["pl_categories"], d["pl_prompt_idxs"]
    pl_layers, pl_mean_z = d["pl_layers"], d["pl_mean_z"]
    z_scores, tok_cats = d["z_scores"], d["categories"]
    layers = sorted(set(int(l) for l in pl_layers))

    # ── col 1: heatmap ────────────────────────────────────────────────
    ax = axes[row, 0]
    prompt_labels = []
    for cat in CATEGORY_ORDER:
        for pid in sorted(set(int(p) for p, c in zip(pl_pidxs, pl_cats) if c == cat)):
            prompt_labels.append(f"{cat}_{pid}")
    row_of = {k: i for i, k in enumerate(prompt_labels)}
    heatmap = np.full((len(prompt_labels), len(layers)), np.nan)
    for i in range(len(pl_cats)):
        heatmap[row_of[f"{pl_cats[i]}_{pl_pidxs[i]}"],
                layers.index(int(pl_layers[i]))] = pl_mean_z[i]

    vmax = max(abs(np.nanmin(heatmap)), abs(np.nanmax(heatmap)), 5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="nearest")
    bounds = {}
    for r, lab in enumerate(prompt_labels):
        bounds.setdefault(lab.rsplit("_", 1)[0], []).append(r)
    mids, names, name_colors = [], [], []
    for cat in CATEGORY_ORDER:
        if cat in bounds:
            rows = bounds[cat]
            mids.append((rows[0] + rows[-1]) / 2)
            names.append(cat)
            name_colors.append(CATEGORY_COLORS[cat])
            ax.axhline(y=rows[-1] + 0.5, color="black", lw=0.3)
    ax.set_yticks(mids, names, fontsize=7.5)
    for tick, color in zip(ax.get_yticklabels(), name_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("layer")
    ax.set_title(f"{tag}  {model_name}", loc="left", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("mean $z$", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    # ── col 2: histogram ──────────────────────────────────────────────
    ax = axes[row, 1]
    valid = pl_mean_z[~np.isnan(pl_mean_z)]
    ax.hist(valid, bins=40, density=True, alpha=0.75, color="#607D8B",
            edgecolor="white", lw=0.3, label="observed (prompt$\\times$layer)")
    x = np.linspace(-8, 8, 200)
    ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), "k--", lw=1.3,
            alpha=0.65, label="null $N(0,1)$")
    ax.axvline(valid.mean(), color="#E91E63", lw=1.6,
               label=f"mean = {valid.mean():+.2f}$\\sigma$")
    ax.set_xlabel("$z$-score")
    ax.set_ylabel("density")
    ax.set_xlim(-8, 8)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
    ax.set_title(f"{tag}1  distribution", loc="left")

    # ── col 3: boxplot ────────────────────────────────────────────────
    ax = axes[row, 2]
    box_data, box_colors, box_labels = [], [], []
    short = {"factual_en": "fact\nen", "factual_zh": "fact\nzh",
             "code_logic": "code", "sentiment": "senti-\nment", "random": "rand"}
    for cat in CATEGORY_ORDER:
        vals = z_scores[tok_cats == cat]
        if len(vals):
            box_data.append(list(vals))
            box_colors.append(CATEGORY_COLORS[cat])
            box_labels.append(f"{short[cat]}\n(n={len(vals)})")
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.55,
                    medianprops={"color": "black", "lw": 1.2},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.4})
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xticklabels(box_labels, fontsize=7.5)
    ax.set_ylabel("$z$-score")
    ax.set_title(f"{tag}2  by category", loc="left")

fig.savefig(OUT / f"comparison_stats{SFX}.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print(f"saved {OUT/f'comparison_stats{SFX}.png'} "
      f"({os.path.getsize(OUT/f'comparison_stats{SFX}.png')/1024:.0f} KB)")
