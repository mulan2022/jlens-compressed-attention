#!/usr/bin/env python3
"""Direct J-space stats collection — works for MLA and GQA, no Gradio needed.

Usage:
  python jlens_stats_direct.py --model /path/to/model --lens /path/to/lens.pt \
    --layers 9 11 13 15 17 19 21 23 --out out/model_stats.npz
"""

import argparse, json, os, math
from collections import defaultdict

import torch
import numpy as np
import transformers

# ── compat: older model code expects get_usable_length on DynamicCache ────
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length()

# ── prompt corpus ───────────────────────────────────────────────────────────
PROMPTS = {
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


# ── model loading ───────────────────────────────────────────────────────────
def load_model_and_lens(model_dir, lens_path):
    """Load model + tokenizer + lens. Returns (model, tok, lens, cfg)."""
    print(f"Loading model: {model_dir}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True)
    tok = transformers.AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    from minimal_jlens import JLens
    lens = JLens.load(lens_path)
    print(f"  model={cfg.get('model_type','?')}, layers={sorted(lens.jacobians)}, "
          f"vocab={cfg['vocab_size']}")
    return model, tok, lens, cfg


# ── attention read subspace ─────────────────────────────────────────────────
def detect_arch(cfg):
    if "kv_lora_rank" in cfg:
        return "mla"
    n_q = cfg.get("num_attention_heads", 0)
    n_kv = cfg.get("num_key_value_heads", n_q)
    return "gqa" if n_kv < n_q else "mha"


def get_unembed(model, cfg):
    """Get unembedding matrix, handling tied embeddings."""
    sd = model.state_dict()
    if "lm_head.weight" in sd:
        return sd["lm_head.weight"].float().cpu()
    return sd["model.embed_tokens.weight"].float().cpu()


def read_basis_gqa(model, cfg, layer):
    """GQA: row space of all KV head projections (K + V combined)."""
    d = cfg["hidden_size"]
    n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = d // cfg["num_attention_heads"]
    sd = model.state_dict()
    pre = f"model.layers.{layer}.self_attn"
    W_K = sd[f"{pre}.k_proj.weight"].float().cpu()  # [n_kv*head_dim, d]
    W_V = sd[f"{pre}.v_proj.weight"].float().cpu()
    combined = torch.cat([W_K, W_V], dim=0)
    _, S, Vh = torch.linalg.svd(combined, full_matrices=False)
    keep = (S > S[0] * 1e-10).sum().item()
    return Vh[:keep, :].T  # [d, keep]


def read_basis_mla(model, cfg, layer):
    """MLA: row space of W_DKV[:r,:]."""
    r = cfg["kv_lora_rank"]
    sd = model.state_dict()
    W_DKV = sd[f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"].float().cpu()[:r, :]
    _, S, Vh = torch.linalg.svd(W_DKV, full_matrices=False)
    keep = (S > S[0] * 1e-10).sum().item()
    return Vh[:keep, :].T


def get_read_basis(model, cfg, layer):
    arch = detect_arch(cfg)
    if arch == "mla":
        return read_basis_mla(model, cfg, layer)
    return read_basis_gqa(model, cfg, layer)


def read_dim(cfg):
    arch = detect_arch(cfg)
    d = cfg["hidden_size"]
    if arch == "mla":
        return cfg["kv_lora_rank"]
    n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = d // cfg["num_attention_heads"]
    return 2 * n_kv * head_dim


# ── J-lens readout (direct, no Gradio) ──────────────────────────────────────
def jlens_readout(model, tok, lens, prompt, layer, top_k=8):
    """Run readout for a single layer, return list of (token_id, prob)."""
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    h = outputs.hidden_states[layer][0, -1, :]  # last token
    h_norm = model.model.norm(h)
    logits = model.lm_head(h_norm)
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)
    if top_ids.dim() == 2:
        top_ids, top_probs = top_ids[0], top_probs[0]
    return [(int(tid), float(p)) for tid, p in zip(top_ids, top_probs)]


# ── stats ───────────────────────────────────────────────────────────────────
def energy_fraction(dirs, basis):
    inside = (dirs @ basis).pow(2).sum()
    total = dirs.pow(2).sum()
    return float(inside / total) if total > 0 else 0.0


def build_null(basis, n=2000):
    d = basis.shape[0]
    rng = np.random.RandomState(42)
    vecs = rng.randn(n, d).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    b = basis.numpy()
    return np.sum((vecs @ b) ** 2, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--out", default="out/jlens_stats.npz")
    args = ap.parse_args()

    model, tok, lens, cfg = load_model_and_lens(args.model, args.lens)
    arch = detect_arch(cfg)
    d = cfg["hidden_size"]
    rdim = read_dim(cfg)
    W_U = get_unembed(model, cfg)

    layers = [l for l in args.layers if l in lens.jacobians and l < cfg["num_hidden_layers"]]
    if not layers:
        raise SystemExit(f"no valid layers; lens has {sorted(lens.jacobians)}, "
                         f"model has {cfg['num_hidden_layers']} layers")
    print(f"Arch: {arch.upper()}  d={d}  read_dim={rdim}  baseline={rdim/d:.3f}  "
          f"layers={layers}")

    # Null distributions per layer
    print(f"\nBuilding null distributions (n={args.n_null})...")
    null_dists = {}
    for l in layers:
        basis = get_read_basis(model, cfg, l)
        null_dists[l] = build_null(basis, args.n_null)
        print(f"  L{l:>2}: null mu={null_dists[l].mean():.5f}  "
              f"sigma={null_dists[l].std():.5f}")

    # Per-prompt readout
    all_prompts = []
    for cat in ["factual_en", "factual_zh", "code_logic", "sentiment", "random"]:
        for p in PROMPTS[cat]:
            all_prompts.append((cat, p))

    print(f"\nCollecting readout for {len(all_prompts)} prompts ({len(layers)} layers)...")
    records = []  # {category, prompt_idx, layer, token_id, prob}
    for idx, (cat, prompt) in enumerate(all_prompts):
        for l in layers:
            tok_probs = jlens_readout(model, tok, lens, prompt, l, args.top_k)
            for tid, prob in tok_probs:
                records.append({
                    "category": cat, "prompt_idx": idx,
                    "layer": l, "token_id": tid, "prob": prob,
                })
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(all_prompts)} prompts done ({len(records)} records)")

    # Compute energy fractions -> z-scores
    print(f"\nComputing J-space energy fractions ({len(records)} records)...")
    basis_cache = {}
    jl_cache = {}
    for i, rec in enumerate(records):
        l = rec["layer"]
        tid = rec["token_id"]
        ck = (l, tid)
        if ck not in jl_cache:
            if l not in basis_cache:
                basis_cache[l] = get_read_basis(model, cfg, l)
            basis = basis_cache[l]
            j_vec = W_U[tid] @ lens.jacobians[l].float()
            frac = energy_fraction(j_vec.unsqueeze(0), basis)
            jl_cache[ck] = frac
        rec["energy_frac"] = jl_cache[ck]
        rec["z_score"] = ((jl_cache[ck] - null_dists[l].mean())
                          / null_dists[l].std())
        if (i + 1) % 3000 == 0:
            print(f"  {i+1}/{len(records)} computed")

    # Aggregate
    idx_to_cat = {i: c for i, (c, _) in enumerate(all_prompts)}
    agg = defaultdict(list)
    for rec in records:
        agg[(rec["prompt_idx"], rec["layer"])].append(rec["z_score"])
    per_pl = []
    for (pi, l), zs in sorted(agg.items()):
        per_pl.append((idx_to_cat.get(pi, "?"), pi, l, np.mean(zs)))

    # Summary
    cat_summary = defaultdict(list)
    for cat, pi, l, mz in per_pl:
        cat_summary[cat].append(mz)

    print("\n=== Per-category z-score summary ===")
    for cat in ["factual_en", "factual_zh", "code_logic", "sentiment", "random"]:
        zs = np.array(cat_summary.get(cat, []))
        if len(zs):
            print(f"  {cat:15s}: n={len(zs):3d}  mu={zs.mean():+.4f}  "
                  f"sigma={zs.std():.4f}  range=[{zs.min():+.3f}, {zs.max():+.3f}]")
    overall = np.array([mz for _, _, _, mz in per_pl])
    print(f"\n  OVERALL: n={len(overall):3d}  mu={overall.mean():+.4f}  "
          f"sigma={overall.std():.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        categories=np.array([r["category"] for r in records], dtype=str),
        prompt_idxs=np.array([r["prompt_idx"] for r in records], dtype=np.int32),
        layers_arr=np.array([r["layer"] for r in records], dtype=np.int32),
        token_ids=np.array([r["token_id"] for r in records], dtype=np.int32),
        probs=np.array([r["prob"] for r in records], dtype=np.float32),
        energy_fracs=np.array([r["energy_frac"] for r in records], dtype=np.float32),
        z_scores=np.array([r["z_score"] for r in records], dtype=np.float32),
        pl_categories=np.array([c for c, _, _, _ in per_pl], dtype=str),
        pl_prompt_idxs=np.array([pi for _, pi, _, _ in per_pl], dtype=np.int32),
        pl_layers=np.array([l for _, _, l, _ in per_pl], dtype=np.int32),
        pl_mean_z=np.array([mz for _, _, _, mz in per_pl], dtype=np.float32),
        null_layers=np.array(layers, dtype=np.int32),
        null_means=np.array([null_dists[l].mean() for l in layers], dtype=np.float32),
        null_stds=np.array([null_dists[l].std() for l in layers], dtype=np.float32),
        n_prompts=len(all_prompts),
        n_skipped=0,
        all_layers=np.array(layers, dtype=np.int32),
    )
    print(f"\nSaved {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
