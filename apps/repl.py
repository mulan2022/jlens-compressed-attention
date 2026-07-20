#!/usr/bin/env python3
# Copyright 2026.
# SPDX-License-Identifier: Apache-2.0
"""Interactive J-lens 'mind reader' for DeepSeek-V2-Lite — terminal REPL.

Type a sentence; see the top-k tokens the J-space is *poised to say* at each
layer for the last token position, next to the model's actual surface
prediction. Runs entirely in your SSH terminal — no GUI, no port-forward.

    # fit a quick lens once (saves to out/v2lite_lens.pt), then chat:
    python jlens_repl.py --model models/DeepSeek-V2-Lite --lens out/v2lite_lens.pt --fit
    # later runs just load it:
    python jlens_repl.py --model models/DeepSeek-V2-Lite --lens out/v2lite_lens.pt
"""
from __future__ import annotations

import argparse
import os

import torch

from jlens import JLens, JLensModel, fit

# Fit prompts must be long enough to clear skip_first=16 (need >17 tokens), so
# these are paragraphs, not one-liners. The paper averages over ~1000; add more
# (or pass --corpus a text file) for cleaner readouts.
FIT_CORPUS = [
    "The history of the Roman Empire spans more than a thousand years of politics, "
    "war, and cultural achievement. From the founding of the Republic to the reign "
    "of the emperors, Rome expanded across three continents, built vast networks of "
    "roads and aqueducts, and left a legal and linguistic legacy that still shapes "
    "the modern world in countless everyday ways.",
    "Photosynthesis is the process by which green plants, algae, and some bacteria "
    "convert carbon dioxide and water into glucose using energy captured from "
    "sunlight. The reaction takes place in the chloroplasts, where the pigment "
    "chlorophyll absorbs light, and it releases oxygen as a byproduct that most "
    "living organisms depend on in order to breathe and survive.",
    "In computer science, a recursive function is one that solves a problem by "
    "calling itself on a smaller version of the same problem, until it reaches a "
    "base case that can be answered directly. Recursion appears throughout "
    "algorithms, from traversing trees and sorting lists to parsing nested "
    "expressions, and it often yields remarkably concise and elegant solutions.",
    "The global financial markets reacted sharply this week after the central bank "
    "announced an unexpected change to its benchmark interest rate. Investors moved "
    "quickly to reassess the value of stocks and bonds, currencies fluctuated "
    "against one another, and analysts debated whether the decision would slow "
    "rising inflation without tipping the wider economy into a painful recession.",
    "Coral reefs are among the most diverse ecosystems on the entire planet, "
    "supporting roughly a quarter of all marine species despite covering only a "
    "tiny fraction of the ocean floor. Warming seas and rising acidity now threaten "
    "these fragile structures, bleaching the corals and endangering the countless "
    "fish and invertebrates that depend on the reef for food and shelter.",
    "The French Revolution began in 1789 amid widespread anger at inequality, "
    "hunger, and the privileges of the aristocracy. Over the following decade the "
    "monarchy was abolished, a republic was declared, and waves of radical reform "
    "and violent upheaval swept the country, reshaping European politics and "
    "inspiring movements for liberty and citizenship far beyond the borders of France.",
    "A healthy diet generally includes a balance of proteins, carbohydrates, and "
    "fats, along with the vitamins and minerals found in fresh fruit and "
    "vegetables. Nutritionists often recommend limiting processed sugar and salt, "
    "drinking plenty of water, and eating a variety of whole foods to maintain "
    "energy, support the immune system, and reduce the long-term risk of disease.",
    "The orchestra fell silent as the conductor raised his baton, and then the "
    "strings began a slow, aching melody that filled the concert hall. Over the "
    "next hour the musicians moved through storms of brass and delicate passages "
    "of woodwind, building toward a final crescendo that left the audience "
    "breathless before erupting into long and grateful applause.",
    "Climate scientists warn that rising concentrations of greenhouse gases are "
    "driving global temperatures upward, melting glaciers, and raising sea levels "
    "around the world. They argue that reducing emissions from fossil fuels, "
    "protecting forests, and investing in renewable energy are essential steps if "
    "humanity hopes to avoid the most dangerous consequences of a warming planet.",
    "The detective knelt beside the shattered window and studied the muddy "
    "footprints leading across the carpet toward the empty safe. Something about "
    "the scene felt staged, she thought, too neat and too obvious, as if the "
    "intruder had wanted the family to believe a simple burglary had taken place "
    "rather than the far more troubling truth she was beginning to suspect.",
]


def load_everything(args) -> tuple[JLensModel, JLens]:
    import transformers

    print(f"loading {args.model} (bf16, cuda) ...")
    tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = JLensModel(hf, tok)
    print(f"  n_layers={model.n_layers} d_model={model.d_model}")

    if args.lens and os.path.exists(args.lens) and not args.fit:
        lens = JLens.load(args.lens)
        print(f"loaded fitted lens: {args.lens}  ({lens.n_prompts} prompts, "
              f"layers {lens.source_layers if hasattr(lens,'source_layers') else sorted(lens.jacobians)})")
    elif args.fit:
        corpus = FIT_CORPUS
        if args.corpus and os.path.exists(args.corpus):
            corpus = [l.strip() for l in open(args.corpus) if l.strip()]
        print(f"fitting lens on {len(corpus)} prompts, layers {args.layers} ...")
        lens = fit(model, corpus, source_layers=args.layers, dim_batch=args.dim_batch)
        if args.lens:
            os.makedirs(os.path.dirname(args.lens) or ".", exist_ok=True)
            lens.save(args.lens)
            print(f"saved lens -> {args.lens}")
    else:
        raise SystemExit("no lens available: pass --lens PATH to an existing file, or --fit")
    return model, lens


@torch.no_grad()
def surface_topk(model: JLensModel, prompt: str, k: int, max_seq_len: int) -> list[tuple[str, float]]:
    """What the model actually predicts next (its 'spoken' output)."""
    ids = model.encode(prompt, max_length=max_seq_len)
    logits = model.hf(input_ids=ids, use_cache=False).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    p, tok = probs.topk(k)
    return [(model.tokenizer.decode([int(t)]), float(pv)) for t, pv in zip(tok, p)]


@torch.no_grad()
def greedy_continue(model: JLensModel, prompt: str, n_new: int = 12, max_seq_len: int = 512) -> str:
    """Greedy continuation so you see a real sentence, not just top-k fragments.
    Uses use_cache=False (transformers 5.13's cached path is broken for this model)."""
    ids = model.encode(prompt, max_length=max_seq_len)
    start = ids.shape[1]
    for _ in range(n_new):
        nxt = model.hf(input_ids=ids, use_cache=False).logits[0, -1].argmax().item()
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return model.tokenizer.decode(ids[0, start:], skip_special_tokens=True)


def render(model, lens, prompt, layers, top_k, position, max_seq_len) -> str:
    surface = surface_topk(model, prompt, top_k, max_seq_len)
    cont = greedy_continue(model, prompt, n_new=12, max_seq_len=max_seq_len)
    read = lens.readout(model, prompt, layers=layers, position=position,
                        top_k=top_k, max_seq_len=max_seq_len)
    lines = [
        "",
        f"  prompt   : {prompt!r}",
        f"  CONTINUES: {cont!r}",
        f"  next-tok : " + "  ".join(f"{t!r}:{p:.2f}" for t, p in surface),
        "  " + "-" * 60,
        "  J-SPACE  (what it's poised to say, per layer):",
    ]
    for layer in sorted(read):
        toks = "  ".join(f"{t!r}:{p:.2f}" for t, p in read[layer])
        lines.append(f"    L{layer:>2}: {toks}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DeepSeek-V2-Lite")
    ap.add_argument("--lens", default="out/v2lite_lens.pt")
    ap.add_argument("--fit", action="store_true", help="fit a fresh lens before chatting")
    ap.add_argument("--corpus", default=None, help="optional text file, one prompt per line, for --fit")
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 11, 15, 19, 23])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--position", type=int, default=-1, help="token position to read (default last)")
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()

    model, lens = load_everything(args)
    read_layers = [l for l in args.layers if l in lens.jacobians] or sorted(lens.jacobians)

    print("\n" + "=" * 64)
    print("  J-lens REPL ready. Type a sentence and press Enter.")
    print("  Ctrl-D or 'quit' to exit.")
    print("=" * 64)
    while True:
        try:
            prompt = input("\n> ").strip()
        except EOFError:
            break
        if prompt in {"quit", "exit"}:
            break
        if not prompt:
            continue
        try:
            print(render(model, lens, prompt, read_layers, args.top_k,
                         args.position, args.max_seq_len))
        except Exception as e:  # keep the REPL alive on a bad prompt
            print(f"  [error: {e}]")


if __name__ == "__main__":
    main()
