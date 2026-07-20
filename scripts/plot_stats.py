#!/usr/bin/env python3
# Copyright 2026.
# SPDX-License-Identifier: Apache-2.0
"""Plot J-space vs MLA read-subspace statistical validation results.

Input: out/jlens_stats.npz (from jlens_collect_stats.py)
Output: out/jlens_stats.png (3-panel figure)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

CATEGORY_COLORS = {
    "factual_en":  "#2196F3",  # blue
    "factual_zh":  "#F44336",  # red
    "code_logic":  "#4CAF50",  # green
    "sentiment":   "#FF9800",  # orange
    "random":      "#9E9E9E",  # grey
}
CATEGORY_ORDER = ["factual_en", "factual_zh", "code_logic", "sentiment", "random"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot J-space stats")
    ap.add_argument("--stats", default="out/jlens_stats.npz")
    ap.add_argument("--out", default="out/jlens_stats.png")
    ap.add_argument("--title", default=None, help="Override figure title")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    d = np.load(args.stats, allow_pickle=True)

    # --- Load data ---
    pl_cats = d["pl_categories"]        # [N_pl] str
    pl_pidxs = d["pl_prompt_idxs"]      # [N_pl]
    pl_layers = d["pl_layers"]          # [N_pl]
    pl_mean_z = d["pl_mean_z"]          # [N_pl]

    null_means = d["null_means"]        # [n_layers]
    null_stds = d["null_stds"]          # [n_layers]
    null_layers = d["null_layers"]      # [n_layers]

    layers = sorted(set(int(l) for l in pl_layers))
    categories = sorted(set(str(c) for c in pl_cats), key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99)

    # Per-token z-scores by category for boxplot
    z_scores = d["z_scores"]
    categories_per_token = d["categories"]

    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 6.5))

    # ── Panel A: Heatmap ───────────────────────────────────────────────
    ax_a = fig.add_axes([0.04, 0.15, 0.42, 0.78])

    # Build prompt-label -> row index
    prompt_labels: list[str] = []
    prompt_row: dict[str, int] = {}
    for cat in CATEGORY_ORDER:
        for pid in sorted(set(int(p) for p, c in zip(pl_pidxs, pl_cats) if c == cat)):
            key = f"{cat}_{pid}"
            prompt_labels.append(key)
            prompt_row[key] = len(prompt_labels) - 1

    # Fill heatmap matrix: rows=prompts, cols=layers
    n_prompts = len(prompt_labels)
    n_layers = len(layers)
    heatmap = np.full((n_prompts, n_layers), np.nan)
    for i in range(len(pl_cats)):
        key = f"{pl_cats[i]}_{pl_pidxs[i]}"
        row = prompt_row[key]
        col = layers.index(int(pl_layers[i]))
        heatmap[row, col] = pl_mean_z[i]

    # Diverging colormap centered at 0
    vmax = max(abs(np.nanmin(heatmap)), abs(np.nanmax(heatmap)), 5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_a.imshow(heatmap, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    # Category labels on y-axis
    cat_boundaries: dict[str, list[int]] = {}
    for row_idx, label in enumerate(prompt_labels):
        cat = label.split("_")[0]
        cat_boundaries.setdefault(cat, []).append(row_idx)

    ax_a.set_yticks([])
    for cat in CATEGORY_ORDER:
        if cat in cat_boundaries:
            rows = cat_boundaries[cat]
            mid = (rows[0] + rows[-1]) / 2
            ax_a.text(-1.8, mid, cat, ha="right", va="center", fontsize=7,
                      color=CATEGORY_COLORS.get(cat, "#000"), fontweight="bold")
            if len(rows) > 1:
                ax_a.axhline(y=rows[-1] + 0.5, color="black", linewidth=0.3, linestyle="-")

    ax_a.set_xticks(range(n_layers))
    ax_a.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax_a.set_xlabel("Layer", fontsize=10)
    ax_a.set_title("A  Per-prompt × layer mean z-score", fontsize=11, fontweight="bold", loc="left")

    cbar = plt.colorbar(im, ax=ax_a, shrink=0.85, pad=0.02)
    cbar.set_label("z-score", fontsize=9)

    # ── Panel B: Histogram ─────────────────────────────────────────────
    ax_b = fig.add_axes([0.53, 0.15, 0.20, 0.78])

    valid_z = pl_mean_z[~np.isnan(pl_mean_z)]
    ax_b.hist(valid_z, bins=50, density=True, alpha=0.7, color="#607D8B",
              edgecolor="white", linewidth=0.3, label="Observed\n(per prompt×layer)")

    # Overlay null: N(0, 1) for visual reference
    x = np.linspace(-8, 8, 200)
    ax_b.plot(x, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2),
              "k--", linewidth=1.5, alpha=0.6, label="Null N(0,1)")

    # Mark mean
    mean_z = float(np.mean(valid_z))
    ax_b.axvline(mean_z, color="#E91E63", linewidth=2, linestyle="-",
                 label=f"Mean = {mean_z:+.2f}σ")

    ax_b.set_xlabel("z-score", fontsize=10)
    ax_b.set_ylabel("Density", fontsize=10)
    ax_b.set_title("B  Z-score distribution", fontsize=11, fontweight="bold", loc="left")
    ax_b.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
    ax_b.set_xlim(-8, 8)

    # ── Panel C: Boxplot by category ───────────────────────────────────
    ax_c = fig.add_axes([0.78, 0.15, 0.20, 0.78])

    box_data: list[list[float]] = []
    box_colors: list[str] = []
    box_labels: list[str] = []
    for cat in CATEGORY_ORDER:
        mask = categories_per_token == cat
        vals = z_scores[mask]
        if len(vals) > 0:
            box_data.append(list(vals))
            box_colors.append(CATEGORY_COLORS.get(cat, "#999"))
            n = len(vals)
            box_labels.append(f"{cat}\n(n={n})")

    bp = ax_c.boxplot(box_data, patch_artist=True, widths=0.55,
                      medianprops={"color": "black", "linewidth": 1.2},
                      flierprops={"marker": ".", "markersize": 2, "alpha": 0.4})
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    ax_c.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_c.set_xticklabels(box_labels, fontsize=7.5)
    ax_c.set_ylabel("z-score", fontsize=10)
    ax_c.set_title("C  Per-token z-score by category", fontsize=11, fontweight="bold", loc="left")

    # ── Overall title ──────────────────────────────────────────────────
    overall = float(np.mean(valid_z))
    if args.title:
        title_str = args.title
    else:
        title_str = (f"J-space energy in MLA read subspace: overall mean z = {overall:+.2f}σ  "
                     f"(random baseline = 0σ, d=2048, r=512, n={len(valid_z)} prompt×layer pairs)")
    fig.suptitle(title_str, fontsize=10, y=0.98, fontweight="normal", style="italic", color="#555")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
