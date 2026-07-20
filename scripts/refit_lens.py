#!/usr/bin/env python3
"""Batch lens fit for the corpus-stability experiment (non-interactive).

Fits a J-lens on an alternate corpus and saves it, then exits — same math as
jlens_repl.py --fit but without dropping into the interactive REPL, so it can
run under nohup / docker exec -d on the A6000 box.
"""

import argparse
import os

import torch
import transformers

from jlens import JLensModel, fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--dim_batch", type=int, default=2)
    ap.add_argument("--lens", required=True)
    args = ap.parse_args()

    print(f"loading {args.model} (bf16, cuda) ...", flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    model = JLensModel(hf, tok)
    print(f"  n_layers={model.n_layers} d_model={model.d_model}", flush=True)

    corpus = [l.strip() for l in open(args.corpus) if l.strip()]
    print(f"fitting {len(corpus)} prompts, layers {args.layers}, "
          f"dim_batch={args.dim_batch}", flush=True)
    lens = fit(model, corpus, source_layers=args.layers, dim_batch=args.dim_batch)
    os.makedirs(os.path.dirname(args.lens) or ".", exist_ok=True)
    lens.save(args.lens)
    print(f"saved lens -> {args.lens}", flush=True)


if __name__ == "__main__":
    main()
