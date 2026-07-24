#!/usr/bin/env python3
"""Local analyses for paper revision (no GPU needed).

1. Distribution structure of per-token z-scores (tails, symmetry, per-layer).
2. V-only GQA read-subspace robustness (rank 256, baseline 0.125).
3. Quantification of Sec.4 phenomena: CJK share in top-8 readouts under
   English prompts (vs vocab prior), markup/formatting token share.

Outputs: out/local_analysis.json (numbers for the paper) + printed report.
"""

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "out"

SFX = sys.argv[1] if len(sys.argv) > 1 else ""  # e.g. "_main300"

MODELS = {
    "mla": {
        "npz": OUT / f"v2lite_stats{SFX}.npz",
        "lens": OUT / f"v2lite_chat_lens{SFX}.pt",
        "tok_dir": ROOT / "tokenizer",
        "label": "DeepSeek-V2-Lite (MLA)",
    },
    "gqa": {
        "npz": OUT / f"qwen25_stats{SFX}.npz",
        "lens": OUT / f"qwen25_lens{SFX}.pt",
        "tok_dir": None,  # resolved below (HF snapshot dir)
        "label": "Qwen2.5-3B (GQA)",
    },
}

qwen_snap = sorted((ROOT / "models" / "models--Qwen--Qwen2.5-3B" / "snapshots").iterdir())[-1]
MODELS["gqa"]["tok_dir"] = qwen_snap

report = {}


# ────────────────────────────────────────────────────────────────────────────
# 1. z-score distribution structure
# ────────────────────────────────────────────────────────────────────────────
def dist_stats(z):
    from scipy import stats as st
    n = len(z)
    out = {
        "n": n,
        "mean": float(z.mean()),
        "std": float(z.std()),
        "skew": float(st.skew(z)),
        "kurtosis_excess": float(st.kurtosis(z)),
        "frac_z_gt_2": float((z > 2).mean()),
        "frac_z_lt_m2": float((z < -2).mean()),
        "frac_abs_gt_2": float((np.abs(z) > 2).mean()),
        "frac_abs_gt_4": float((np.abs(z) > 4).mean()),
        "frac_abs_gt_8": float((np.abs(z) > 8).mean()),
        "q05": float(np.quantile(z, 0.05)),
        "q25": float(np.quantile(z, 0.25)),
        "median": float(np.quantile(z, 0.5)),
        "q75": float(np.quantile(z, 0.75)),
        "q95": float(np.quantile(z, 0.95)),
    }
    return out


NULL_TAILS = {"abs_gt_2": 0.0455, "abs_gt_4": 6.33e-5, "abs_gt_8": 1.2e-15}

print("=" * 78)
print("1. Z-SCORE DISTRIBUTION STRUCTURE")
print("=" * 78)
for key, m in MODELS.items():
    d = np.load(m["npz"], allow_pickle=True)
    z = d["z_scores"].astype(np.float64)
    layers = d["layers_arr"]
    cats = d["categories"]
    tids = d["token_ids"]

    rep = {"overall": dist_stats(z)}

    # unique (layer, token) dedup — repeated readout tokens are correlated
    uniq = {}
    for l, t, zz in zip(layers, tids, z):
        uniq[(int(l), int(t))] = float(zz)
    zu = np.array(list(uniq.values()))
    rep["unique_layer_token"] = dist_stats(zu)

    rep["per_layer"] = {}
    for l in sorted(set(layers.tolist())):
        rep["per_layer"][int(l)] = dist_stats(z[layers == l])

    rep["per_category"] = {}
    for c in sorted(set(cats.tolist())):
        rep["per_category"][str(c)] = dist_stats(z[cats == c])

    report[f"{key}_zdist"] = rep

    o, u = rep["overall"], rep["unique_layer_token"]
    print(f"\n--- {m['label']} ---")
    print(f"  all records      n={o['n']:5d}  mean={o['mean']:+.3f}  std={o['std']:.2f}  "
          f"skew={o['skew']:+.2f}  exkurt={o['kurtosis_excess']:+.2f}")
    print(f"  unique (l,tok)   n={u['n']:5d}  mean={u['mean']:+.3f}  std={u['std']:.2f}  "
          f"skew={u['skew']:+.2f}  exkurt={u['kurtosis_excess']:+.2f}")
    print(f"  tails (unique): P(z>+2)={u['frac_z_gt_2']:.3f}  P(z<-2)={u['frac_z_lt_m2']:.3f}  "
          f"P(|z|>4)={u['frac_abs_gt_4']:.4f}  P(|z|>8)={u['frac_abs_gt_8']:.4f}")
    print(f"  null expects:   P(|z|>2)={NULL_TAILS['abs_gt_2']:.3f}  P(|z|>4)={NULL_TAILS['abs_gt_4']:.1e}")
    print(f"  quantiles (unique): 5%={u['q05']:+.2f} 25%={u['q25']:+.2f} 50%={u['median']:+.2f} "
          f"75%={u['q75']:+.2f} 95%={u['q95']:+.2f}")
    print("  per-layer mean/std/P(|z|>2):")
    for l, s in rep["per_layer"].items():
        print(f"    L{l:>2}: {s['mean']:+.2f} / {s['std']:.2f} / {s['frac_abs_gt_2']:.3f}")


# ────────────────────────────────────────────────────────────────────────────
# 2. V-only GQA read subspace (rank 256, baseline 0.125)
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("2. V-ONLY GQA READ SUBSPACE ROBUSTNESS (Qwen2.5-3B)")
print("=" * 78)

import torch
from safetensors import safe_open

sf_files = sorted(qwen_snap.glob("model-*.safetensors"))
qcfg = json.loads((qwen_snap / "config.json").read_text())
D = qcfg["hidden_size"]
HEAD_DIM = D // qcfg["num_attention_heads"]
N_KV = qcfg["num_key_value_heads"]


def load_tensor(name):
    for f in sf_files:
        with safe_open(f, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name).float()
    raise KeyError(name)


lens = torch.load(MODELS["gqa"]["lens"], map_location="cpu", weights_only=True)
J = {l: t.float() for l, t in lens["J"].items()}
W_E = load_tensor("model.embed_tokens.weight")  # tied → also unembedding

d = np.load(MODELS["gqa"]["npz"], allow_pickle=True)
layers_arr, tids, cats = d["layers_arr"], d["token_ids"], d["categories"]
lens_layers = sorted(set(layers_arr.tolist()))

rng = np.random.RandomState(42)


def basis_and_null(W, n_null=2000):
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    keep = int((S > S[0] * 1e-10).sum())
    B = Vh[:keep, :].T.numpy()  # [d, keep]
    vecs = rng.randn(n_null, W.shape[1]).astype(np.float64)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    null = ((vecs @ B) ** 2).sum(axis=1)
    return B, null.mean(), null.std(), keep


vonly = {"per_layer": {}, "records": {"mean_z": None}}
all_z_v = []
for l in lens_layers:
    W_V = load_tensor(f"model.layers.{l}.self_attn.v_proj.weight")  # [256, 2048]
    B, mu0, sd0, rank = basis_and_null(W_V)
    idx = layers_arr == l
    toks = sorted(set(tids[idx].tolist()))
    M = (W_E[toks] @ J[l]).numpy()  # [n_tok, d]
    frac = ((M @ B) ** 2).sum(axis=1) / (M ** 2).sum(axis=1)
    zmap = {t: (f - mu0) / sd0 for t, f in zip(toks, frac)}
    zs = np.array([zmap[int(t)] for t in tids[idx]])
    all_z_v.append(zs)
    vonly["per_layer"][int(l)] = {
        "rank": rank, "null_mu": float(mu0), "null_sd": float(sd0),
        "mean_frac": float(np.mean([frac.mean()])), "mean_z": float(zs.mean()),
        "std_z": float(zs.std()),
    }
    print(f"  L{l:>2}: rank={rank}  null μ={mu0:.4f} σ={sd0:.4f}  "
          f"obs mean frac={frac.mean():.4f}  mean z={zs.mean():+.2f}  σz={zs.std():.2f}")

zv = np.concatenate(all_z_v)
vonly["overall"] = dist_stats(zv)
report["gqa_vonly"] = vonly
print(f"  OVERALL: mean z={zv.mean():+.3f}  σz={zv.std():.2f}  "
      f"P(|z|>2)={np.mean(np.abs(zv) > 2):.3f}")


# ────────────────────────────────────────────────────────────────────────────
# 3. Sec.4 quantification: CJK share & markup share in top-8 readouts
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("3. SEC.4 QUANTIFICATION")
print("=" * 78)


def has_cjk(s):
    return any("CJK" in unicodedata.name(ch, "") for ch in s)


MARKUP_RE = re.compile(r'^[\s]*[=<>{}\[\]()\\/_^#*&%$@~`|+.,:;"\'-]{2,}')


def is_markup(s):
    t = s.strip()
    if not t:
        return False
    non_alnum = sum(1 for ch in t if not ch.isalnum() and not ch.isspace())
    return (non_alnum / len(t)) > 0.5 or bool(MARKUP_RE.match(s))


from transformers import AutoTokenizer

for key, m in MODELS.items():
    tok = AutoTokenizer.from_pretrained(str(m["tok_dir"]), trust_remote_code=True)
    d = np.load(m["npz"], allow_pickle=True)
    layers_arr, tids, cats, probs = d["layers_arr"], d["token_ids"], d["categories"], d["probs"]

    # vocab prior: CJK / markup share over the full vocabulary
    vocab_n = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok)
    sample_ids = np.arange(len(tok))
    dec = tok.batch_decode([[i] for i in sample_ids])
    vocab_cjk = np.array([has_cjk(s) for s in dec])
    vocab_markup = np.array([is_markup(s) for s in dec])

    rep = {
        "vocab_size": int(len(tok)),
        "vocab_cjk_share": float(vocab_cjk.mean()),
        "vocab_markup_share": float(vocab_markup.mean()),
        "cjk_by_layer_en": {}, "markup_by_layer": {},
    }

    en_mask = cats == "factual_en"
    print(f"\n--- {m['label']} ---")
    print(f"  vocab: {len(tok)} tokens, CJK share={vocab_cjk.mean():.3f}, "
          f"markup share={vocab_markup.mean():.3f}")

    print("  CJK share in top-8 readout under factual_en prompts (unweighted / prob-weighted):")
    for l in sorted(set(layers_arr.tolist())):
        idx = en_mask & (layers_arr == l)
        toks_l = tids[idx]
        p_l = probs[idx].astype(np.float64)
        c = np.array([has_cjk(tok.decode([int(t)])) for t in toks_l])
        w = float((c * p_l).sum() / p_l.sum()) if p_l.sum() > 0 else 0.0
        rep["cjk_by_layer_en"][int(l)] = {"unweighted": float(c.mean()), "prob_weighted": w,
                                          "n": int(idx.sum())}
        print(f"    L{l:>2}: {c.mean():.3f} / {w:.3f}   (n={idx.sum()})")

    # binomial test overall vs vocab prior
    idx = en_mask
    c_all = np.array([has_cjk(tok.decode([int(t)])) for t in tids[idx]])
    from scipy import stats as st
    bt = st.binomtest(int(c_all.sum()), len(c_all), rep["vocab_cjk_share"])
    rep["cjk_en_overall"] = {"share": float(c_all.mean()), "n": int(len(c_all)),
                             "binom_p_vs_vocab": float(bt.pvalue)}
    print(f"  overall factual_en CJK share: {c_all.mean():.3f} vs vocab prior "
          f"{rep['vocab_cjk_share']:.3f}  (binom p={bt.pvalue:.2e})")

    print("  markup share by layer (all categories, unweighted / prob-weighted):")
    for l in sorted(set(layers_arr.tolist())):
        idx = layers_arr == l
        mk = np.array([is_markup(tok.decode([int(t)])) for t in tids[idx]])
        p_l = probs[idx].astype(np.float64)
        w = float((mk * p_l).sum() / p_l.sum())
        rep["markup_by_layer"][int(l)] = {"unweighted": float(mk.mean()), "prob_weighted": w}
        print(f"    L{l:>2}: {mk.mean():.3f} / {w:.3f}")
    mk_all = np.array([is_markup(tok.decode([int(t)])) for t in tids])
    rep["markup_overall"] = {"share": float(mk_all.mean()),
                             "vs_vocab": float(vocab_markup.mean())}
    print(f"  overall markup share: {mk_all.mean():.3f} vs vocab prior {vocab_markup.mean():.3f}")

    report[f"{key}_sec4"] = rep

OUT.joinpath(f"local_analysis{SFX}.json").write_text(json.dumps(report, indent=1))
print(f"\nSaved {OUT/f'local_analysis{SFX}.json'}")
