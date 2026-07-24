#!/usr/bin/env python3
"""Cluster bootstrap (resample prompts) for the Sec.4 token-share claims.

The per-record binomial tests in analysis_local.py treat all top-8 readout
tokens as independent, but tokens within a prompt (and across the 8 layers of
the same prompt) are correlated. Here we resample whole prompts with
replacement and recompute the shares, giving cluster-robust CIs and p-values.

Claims checked:
  1. CJK share of top-8 readouts under English prompts vs vocab prior
     (DeepSeek chat + base; Qwen base).
  2. Markup share across all categories vs vocab prior (chat, base, Qwen).
  3. Paired base-vs-chat markup difference (same 52 prompts).

Output: out/bootstrap_analysis.json + printed report.
"""

import json
import re
import unicodedata
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "out"
N_BOOT = 10_000
SEED = 12345

SFX = sys.argv[1] if len(sys.argv) > 1 else ""  # e.g. "_main300"

qwen_snap = sorted((ROOT / "models" / "models--Qwen--Qwen2.5-3B" / "snapshots").iterdir())[-1]

MODELS = {
    "mla_chat": {"npz": OUT / f"v2lite_stats{SFX}.npz",
                 "tok": ROOT / "tokenizer"},
    "mla_base": {"npz": OUT / f"v2lite_base_stats{SFX}.npz",
                 "tok": ROOT / "tokenizer"},
    "gqa_base": {"npz": OUT / f"qwen25_stats{SFX}.npz", "tok": qwen_snap},
}


def has_cjk(s: str) -> bool:
    return any("CJK" in unicodedata.name(ch, "") for ch in s)


MARKUP_RE = re.compile(r'^[\s]*[=<>{}\[\]()\\/_^#*&%$@~`|+.,:;"\'-]{2,}')


def is_markup(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    non_alnum = sum(1 for ch in t if not ch.isalnum() and not ch.isspace())
    return (non_alnum / len(t)) > 0.5 or bool(MARKUP_RE.match(s))


def boot_share(recs_by_prompt, n_prompts, rng, weighted=False):
    """Bootstrap the token share over resampled prompt clusters.

    recs_by_prompt: list of (flags, weights) per prompt.
    Returns (boot distribution, observed share).
    """
    def share(sel):
        num = den = 0.0
        for p in sel:
            f, w = recs_by_prompt[p]
            if weighted:
                num += float((f * w).sum())
                den += float(w.sum())
            else:
                num += float(f.sum())
                den += float(len(f))
        return num / den if den else 0.0

    obs = share(range(n_prompts))
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        sel = rng.randint(0, n_prompts, n_prompts)
        boots[b] = share(sel)
    return boots, obs


def summarize(boots, obs, prior):
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_one = float((boots <= prior).mean())  # one-sided: P(share <= prior)
    return {"observed": obs, "ci95": [float(lo), float(hi)],
            "prior": prior, "p_one_sided_vs_prior": max(p_one, 1.0 / N_BOOT),
            "p_raw_count": int((boots <= prior).sum())}


def main() -> None:
    from transformers import AutoTokenizer
    rng = np.random.RandomState(SEED)
    report = {}

    for key, m in MODELS.items():
        tok = AutoTokenizer.from_pretrained(str(m["tok"]), trust_remote_code=True)
        d = np.load(m["npz"], allow_pickle=True)
        cats, pidx, tids, probs = (d["categories"], d["prompt_idxs"],
                                   d["token_ids"], d["probs"].astype(np.float64))

        # decode once per unique token id
        uniq = np.unique(tids)
        dec = {int(t): tok.decode([int(t)]) for t in uniq}
        cjk_flag = {t: has_cjk(s) for t, s in dec.items()}
        mk_flag = {t: is_markup(s) for t, s in dec.items()}
        dec_vocab = tok.batch_decode([[i] for i in range(len(tok))])
        vocab_cjk = float(np.mean([has_cjk(s) for s in dec_vocab]))
        vocab_mk = float(np.mean([is_markup(s) for s in dec_vocab]))

        rep = {"vocab_cjk_share": vocab_cjk, "vocab_markup_share": vocab_mk}

        # ── CJK under English prompts ────────────────────────────────────
        en = cats == "factual_en"
        prompts = sorted(set(pidx[en].tolist()))
        recs = [(
            np.array([cjk_flag[int(t)] for t in tids[en & (pidx == p)]]),
            probs[en & (pidx == p)],
        ) for p in prompts]
        for weighted, tag in [(False, "unweighted"), (True, "prob_weighted")]:
            boots, obs = boot_share(recs, len(recs), rng, weighted)
            rep[f"cjk_en_{tag}"] = summarize(boots, obs, vocab_cjk)
        print(f"[{key}] CJK factual_en: obs={rep['cjk_en_unweighted']['observed']:.3f} "
              f"CI95={np.round(rep['cjk_en_unweighted']['ci95'], 3)} vs prior {vocab_cjk:.3f} "
              f"p={rep['cjk_en_unweighted']['p_one_sided_vs_prior']:.1e}")

        # ── markup over all categories ───────────────────────────────────
        prompts = sorted(set(pidx.tolist()))
        recs = [(
            np.array([mk_flag[int(t)] for t in tids[pidx == p]]),
            probs[pidx == p],
        ) for p in prompts]
        boots, obs = boot_share(recs, len(recs), rng, weighted=False)
        rep["markup_all_unweighted"] = summarize(boots, obs, vocab_mk)
        print(f"[{key}] markup all: obs={obs:.3f} "
              f"CI95={np.round(rep['markup_all_unweighted']['ci95'], 3)} vs prior {vocab_mk:.3f} "
              f"p={rep['markup_all_unweighted']['p_one_sided_vs_prior']:.1e}")

        report[key] = rep

    # ── paired base-vs-chat markup difference (same prompts) ─────────────
    d_chat = np.load(MODELS["mla_chat"]["npz"], allow_pickle=True)
    d_base = np.load(MODELS["mla_base"]["npz"], allow_pickle=True)
    tok = AutoTokenizer.from_pretrained(str(MODELS["mla_chat"]["tok"]), trust_remote_code=True)
    uniq = np.unique(np.concatenate([d_chat["token_ids"], d_base["token_ids"]]))
    mk = {int(t): is_markup(tok.decode([int(t)])) for t in uniq}
    prompts = sorted(set(d_chat["prompt_idxs"].tolist()))
    assert prompts == sorted(set(d_base["prompt_idxs"].tolist())), "prompt sets differ"

    def flags(d, p):
        return np.array([mk[int(t)] for t in d["token_ids"][d["prompt_idxs"] == p]])

    pairs = [(flags(d_base, p), flags(d_chat, p)) for p in prompts]

    def diff(sel):
        nb = np.concatenate([pairs[p][0] for p in sel]).mean()
        nc = np.concatenate([pairs[p][1] for p in sel]).mean()
        return nb - nc

    obs_d = diff(range(len(pairs)))
    boots_d = np.array([diff(rng.randint(0, len(pairs), len(pairs)))
                        for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots_d, [2.5, 97.5])
    p_neg = float((boots_d <= 0).mean())
    report["mla_base_vs_chat_markup"] = {
        "observed_diff": float(obs_d), "ci95": [float(lo), float(hi)],
        "p_one_sided_base_le_chat": max(p_neg, 1.0 / N_BOOT),
    }
    print(f"[paired] base-chat markup diff: {obs_d:+.3f} CI95=[{lo:+.3f},{hi:+.3f}] "
          f"p(base<=chat)={max(p_neg, 1/N_BOOT):.1e}")

    OUT.joinpath(f"bootstrap_analysis{SFX}.json").write_text(json.dumps(report, indent=1))
    print(f"\nSaved {OUT/f'bootstrap_analysis{SFX}.json'}")


if __name__ == "__main__":
    main()
