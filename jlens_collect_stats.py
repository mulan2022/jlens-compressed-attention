#!/usr/bin/env python3
# Copyright 2026.
# SPDX-License-Identifier: Apache-2.0
"""Per-prompt J-space vs MLA read-subspace statistical validation.

Design (Claude):
  1. Diverse prompt corpus (50-100 prompts, 5 categories).
  2. Null distribution: random unit vectors vs each layer's read subspace.
  3. Per-prompt, per-layer z-scores of J-space energy fraction in read subspace.
  4. Output .npz for plotting.

Runs on the A6000 (CPU-only; uses the live Gradio app's API for model forward
passes, so no extra GPU memory needed).
"""
from __future__ import annotations

import argparse, json, os, math, time
from collections import defaultdict

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Prompt corpus (~75 prompts, 5 categories)
# ---------------------------------------------------------------------------
PROMPTS: dict[str, list[str]] = {
    "factual_en": [
        "The capital of Germany is",
        "Water boils at a temperature of",
        "Einstein was born in",
        "The chemical symbol for gold is",
        "Mount Everest is located in",
        "The speed of light is approximately",
        "The largest ocean on Earth is",
        "Photosynthesis converts carbon dioxide and",
        "The French Revolution began in",
        "Shakespeare wrote the play",
        "DNA stands for",
        "The currency of Japan is the",
        "Mars is the",
        "The human heart has",
        "Gravity was discovered by",
        "The longest river in the world is",
    ],
    "factual_zh": [
        "中国的首都是",
        "地球绕太阳一圈需要",
        "人体正常体温是",
        "水的化学式是",
        "世界上最高的山峰是",
        "光的速度大约是",
        "人体有多少块骨头？",
        "太阳系中最大的行星是",
        "第一次世界大战爆发于",
        "元素周期表是由谁发明的",
    ],
    "code_logic": [
        "def quicksort(arr):",
        "SELECT * FROM users WHERE",
        "import numpy as np\n# Create a",
        "def fibonacci(n):",
        "class BinaryTree:",
        "try:\n    with open(",
        "async def fetch_data(url):",
        "git commit -m \"fix:",
        "docker run -d --name",
        "const [state, setState] = useState(",
        "SELECT COUNT(*) FROM orders GROUP BY",
    ],
    "sentiment": [
        "I really hate it when",
        "The best part of my day was",
        "I am so grateful for",
        "Nothing makes me angrier than",
        "I wish I could",
        "My heart sank when I heard",
        "I can't stop smiling because",
        "What a beautiful",
        "The worst mistake I ever made was",
        "I will never forget the moment when",
        "Honestly, I think the problem is",
    ],
    "random": [
        "zxcv qwer tyui asdf",
        "blarg flump snizzle womp",
        "the the the the the the the",
        "a b c d e f g h i j",
    ],
}


def _load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def _load_tensors(model_dir: str):
    """Return f(name)->Tensor (safetensors, single-tensor reads, CPU)."""
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(index_path):
        wm = json.load(open(index_path))["weight_map"]
        shards = set(wm.values())

        def load(name: str) -> torch.Tensor:
            shard = wm[name]
            with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
                return f.get_tensor(name).float()
        return load
    if os.path.exists(single):
        def load(name: str) -> torch.Tensor:
            with safe_open(single, framework="pt") as f:
                return f.get_tensor(name).float()
        return load
    return None


def read_basis(load, cfg: dict, layer: int) -> torch.Tensor:
    """Orthonormal basis [d, r] of the attention content-read subspace."""
    r = cfg["kv_lora_rank"]
    W_DKV = load(f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight")[:r, :]
    _, _, Vh = torch.linalg.svd(W_DKV, full_matrices=False)
    return Vh.T  # [d, r]


def energy_fraction(dirs: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of Frobenius energy of dirs [n, d] in subspace spanned by basis [d, k]."""
    inside = (dirs @ basis).pow(2).sum()
    total = dirs.pow(2).sum()
    return float(inside / total) if total > 0 else 0.0


def build_null_distribution(basis: torch.Tensor, n_samples: int = 2000) -> np.ndarray:
    """Random unit vectors from N(0,I) projected onto read subspace."""
    d = basis.shape[0]
    rng = np.random.RandomState(42)
    vecs = rng.randn(n_samples, d).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    b = basis.numpy()
    inside = np.sum((vecs @ b) ** 2, axis=1)
    return inside  # [n_samples]


def query_gradio(prompt: str, top_k: int = 8, position: int = -1,
                 base_url: str = "http://localhost:8888"):
    """Call the live Gradio app's /analyze endpoint, return (surface_md, rows).
    rows is a list of [layer_str, "token1·prob1  token2·prob2  ..."] pairs.
    """
    from gradio_client import Client
    import atexit

    # Persistent client (created once, reused across calls).
    if not hasattr(query_gradio, "_client"):
        query_gradio._client = Client(base_url)
        query_gradio._tok = None

    c = query_gradio._client
    try:
        md, rows = c.predict(prompt, int(top_k), int(position), api_name="/analyze")
    except Exception as e:
        raise RuntimeError(f"Gradio call failed: {e}")

    # rows comes back as dict with "data" key (list of lists) or plain list
    data = rows["data"] if isinstance(rows, dict) else rows
    return md, data


def count_tokens_in_readout(tok_str: str, tokenizer) -> list[int] | None:
    """Decode a readout token string back to a single token id.  Returns None if
    the string does not correspond to exactly one token."""
    ids = tokenizer.encode(tok_str, add_special_tokens=False)
    if len(ids) == 1:
        return ids
    # Try stripping leading space (common for subword tokens like " Paris").
    alt = tokenizer.encode(tok_str.lstrip(), add_special_tokens=False)
    if len(alt) == 1:
        return alt
    return None


def parse_readout_row(row: list, tokenizer) -> list[tuple[int, float]]:
    """Parse a Gradio dataframe row like ['L9', 'token1·0.12  token2·0.08  ...'].
    Returns list of (token_id, prob)."""
    results = []
    if len(row) < 2:
        return results
    parts = row[1].split("  ")
    for part in parts:
        if "·" not in part:
            continue
        tok_str, prob_str = part.rsplit("·", 1)
        try:
            prob = float(prob_str)
        except ValueError:
            continue
        ids = count_tokens_in_readout(tok_str, tokenizer)
        if ids is not None:
            results.append((ids[0], prob))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="J-space vs read-subspace stats collection")
    ap.add_argument("--model", default="models/DeepSeek-V2-Lite-Chat")
    ap.add_argument("--lens-path", default="out/v2lite_chat_lens.pt")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[9, 11, 13, 15, 17, 19, 21, 23])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=2000,
                    help="random samples for null distribution")
    ap.add_argument("--gradio-url", default="http://localhost:8888",
                    help="live Gradio app URL")
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="load only tokenizer (assume Gradio app provides model)")
    ap.add_argument("--out", default="out/jlens_stats.npz")
    args = ap.parse_args()

    from minimal_jlens import JLens

    # ------------------------------------------------------------------
    # 1. Load lens, tokenizer, weight loader.
    # ------------------------------------------------------------------
    print(f"Loading lens: {args.lens_path}")
    lens = JLens.load(args.lens_path)
    layers = [l for l in args.layers if l in lens.jacobians]
    if not layers:
        raise SystemExit(f"no fitted layers; lens has {sorted(lens.jacobians)}")
    print(f"  layers: {layers}")

    import transformers
    print(f"Loading tokenizer from {args.model} ...")
    tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    cfg = _load_config(args.model)
    load = _load_tensors(args.model)
    if load is None:
        raise SystemExit("weights not found")
    W_U = load("lm_head.weight")  # [vocab, d]

    # ------------------------------------------------------------------
    # 2. Build null distributions per layer.
    # ------------------------------------------------------------------
    print(f"Building null distributions (n={args.n_null}) ...")
    null_dists: dict[int, np.ndarray] = {}
    for l in layers:
        basis = read_basis(load, cfg, l)
        null_dists[l] = build_null_distribution(basis, args.n_null)
        print(f"  L{l:>2}: null μ={null_dists[l].mean():.5f}  σ={null_dists[l].std():.5f}")

    # ------------------------------------------------------------------
    # 3. Flatten prompts.
    # ------------------------------------------------------------------
    all_prompts: list[tuple[str, str]] = []
    for cat, prompts in PROMPTS.items():
        for p in prompts:
            all_prompts.append((cat, p))
    print(f"\nCollecting data for {len(all_prompts)} prompts "
          f"({len(PROMPTS)} categories) ...")

    # ------------------------------------------------------------------
    # 4. Per prompt: query Gradio for readout, then compute energy fractions.
    # ------------------------------------------------------------------
    records: list[dict] = []  # one dict per (prompt, layer, token)
    skipped = 0

    for idx, (cat, prompt) in enumerate(all_prompts):
        try:
            _, rows = query_gradio(prompt, top_k=args.top_k, base_url=args.gradio_url)
        except Exception as e:
            print(f"  [{idx:3d}] GRADIO_ERR {cat} {prompt[:40]!r}: {e}")
            skipped += 1
            continue

        for row in rows:
            # row[0] like "L9", row[1] like "token1·0.12  token2·0.08"
            layer_str = row[0]
            try:
                layer = int(layer_str.lstrip("L"))
            except ValueError:
                continue
            if layer not in layers:
                continue

            tok_probs = parse_readout_row(row, tok)
            for tid, prob in tok_probs:
                records.append({
                    "category": cat,
                    "prompt_idx": idx,
                    "layer": layer,
                    "token_id": tid,
                    "prob": prob,
                })

        if (idx + 1) % 15 == 0:
            print(f"  {idx + 1}/{len(all_prompts)} prompts done  "
                  f"({len(records)} records, {skipped} skipped)")

    print(f"\nCollected {len(records)} token-records from "
          f"{len(all_prompts) - skipped}/{len(all_prompts)} prompts")

    # ------------------------------------------------------------------
    # 5. Compute per-token energy fractions -> z-scores.
    # ------------------------------------------------------------------
    print("Computing J-lens vectors and energy fractions ...")
    basis_cache = {}
    jl_cache = {}  # (layer, tok_id) -> energy_frac

    for i, rec in enumerate(records):
        l = rec["layer"]
        tid = rec["token_id"]
        cache_key = (l, tid)

        if cache_key not in jl_cache:
            if l not in basis_cache:
                basis_cache[l] = read_basis(load, cfg, l)
            basis = basis_cache[l]
            j_l = lens.jacobians[l]
            w_u_row = W_U[tid]                              # [d]
            j_lens_vec = w_u_row @ j_l                      # [d]
            frac = energy_fraction(j_lens_vec.unsqueeze(0), basis)
            jl_cache[cache_key] = frac

        rec["energy_frac"] = jl_cache[cache_key]
        rec["z_score"] = (
            (jl_cache[cache_key] - null_dists[l].mean())
            / null_dists[l].std()
        )

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(records)} energy fractions computed")

    # ------------------------------------------------------------------
    # 6. Aggregate per (prompt, layer) — mean z-score over top-k.
    # ------------------------------------------------------------------
    agg: dict[tuple[int, int], list[float]] = defaultdict(list)
    for rec in records:
        agg[(rec["prompt_idx"], rec["layer"])].append(rec["z_score"])

    per_pl = []  # (category, prompt_idx, layer, mean_z)
    for (pi, l), zs in sorted(agg.items()):
        cat = records[0]["category"]  # won't work for multi-cat... let me fix
        per_pl.append((pi, l, np.mean(zs)))

    # Build a lookup: prompt_idx -> category
    idx_to_cat = {}
    for idx2, (cat2, _) in enumerate(all_prompts):
        idx_to_cat[idx2] = cat2

    per_pl_with_cat = []
    for pi, l, mz in per_pl:
        per_pl_with_cat.append((idx_to_cat.get(pi, "?"), pi, l, mz))

    # Per-category summary
    cat_summary: dict[str, list[float]] = defaultdict(list)
    for cat, pi, l, mz in per_pl_with_cat:
        cat_summary[cat].append(mz)

    print("\n=== Per-category z-score summary ===")
    for cat in sorted(cat_summary):
        zs = np.array(cat_summary[cat])
        print(f"  {cat:15s}: n={len(zs):3d}  μ={zs.mean():+.4f}  "
              f"σ={zs.std():.4f}  range=[{zs.min():+.3f}, {zs.max():+.3f}]")

    overall_z = np.array([mz for _, _, _, mz in per_pl_with_cat])
    print(f"\n  OVERALL:       n={len(overall_z):3d}  μ={overall_z.mean():+.4f}  "
          f"σ={overall_z.std():.4f}  range=[{overall_z.min():+.3f}, {overall_z.max():+.3f}]")

    # ------------------------------------------------------------------
    # 7. Save.
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        # per-token records
        categories=np.array([r["category"] for r in records], dtype=str),
        prompt_idxs=np.array([r["prompt_idx"] for r in records], dtype=np.int32),
        layers_arr=np.array([r["layer"] for r in records], dtype=np.int32),
        token_ids=np.array([r["token_id"] for r in records], dtype=np.int32),
        probs=np.array([r["prob"] for r in records], dtype=np.float32),
        energy_fracs=np.array([r["energy_frac"] for r in records], dtype=np.float32),
        z_scores=np.array([r["z_score"] for r in records], dtype=np.float32),
        # per (prompt, layer) aggregates
        pl_categories=np.array([c for c, _, _, _ in per_pl_with_cat], dtype=str),
        pl_prompt_idxs=np.array([pi for _, pi, _, _ in per_pl_with_cat], dtype=np.int32),
        pl_layers=np.array([l for _, _, l, _ in per_pl_with_cat], dtype=np.int32),
        pl_mean_z=np.array([mz for _, _, _, mz in per_pl_with_cat], dtype=np.float32),
        # null distributions
        null_layers=np.array(layers, dtype=np.int32),
        null_means=np.array([null_dists[l].mean() for l in layers], dtype=np.float32),
        null_stds=np.array([null_dists[l].std() for l in layers], dtype=np.float32),
        # metadata
        n_prompts=len(all_prompts),
        n_skipped=int(skipped),
        all_layers=np.array(layers, dtype=np.int32),
        prompt_categories=list(PROMPTS.keys()),
    )
    print(f"\nSaved {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
