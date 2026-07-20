#!/usr/bin/env python3
# Copyright 2026.
# SPDX-License-Identifier: Apache-2.0
"""J-lens web UI for DeepSeek-V2-Lite — the paper-style 'ask & see' demo.

Serves a small web page from the GPU server. You view it in your LOCAL browser
over an SSH tunnel — no remote desktop, no ToDesk/Sunlogin needed.

On the A6000:
    pip install gradio
    python jlens_app.py --model models/DeepSeek-V2-Lite --lens out/v2lite_lens.pt --fit

On your laptop (open a second terminal, forward the port):
    ssh -L 7860:localhost:7860 moulan@<A6000-host>
    # then browse to  http://localhost:7860

Type a sentence -> a table of the top J-space tokens at each layer, next to the
model's actual next-token prediction.
"""
from __future__ import annotations

import argparse
import os

import torch

from minimal_jlens import JLens, JLensModel, fit
from jlens_repl import FIT_CORPUS, surface_topk, greedy_continue

MODEL: JLensModel | None = None
LENS: JLens | None = None
READ_LAYERS: list[int] = []
MAX_SEQ_LEN = 512


def analyze(prompt: str, top_k: int, position: int):
    """Returns (surface_markdown, per-layer table rows) for the UI."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "_type a sentence_", []
    cont = greedy_continue(MODEL, prompt, n_new=14, max_seq_len=MAX_SEQ_LEN)
    surface = surface_topk(MODEL, prompt, int(top_k), MAX_SEQ_LEN)
    surface_md = (
        f"**Model continues (greedy):** `{cont}`\n\n"
        "**Top next-token candidates** (single next token, so these are fragments): "
        + "  ".join(f"`{t}` {p:.2f}" for t, p in surface)
    )
    read = LENS.readout(MODEL, prompt, layers=READ_LAYERS, position=int(position),
                        top_k=int(top_k), max_seq_len=MAX_SEQ_LEN)
    rows = [[f"L{layer}", "  ".join(f"{t}·{p:.2f}" for t, p in read[layer])]
            for layer in sorted(read)]
    return surface_md, rows


def build_ui():
    import gradio as gr

    with gr.Blocks(title="J-lens · DeepSeek-V2-Lite") as demo:
        gr.Markdown(
            "# J-lens: reading the model's mind\n"
            "Type a sentence. The **J-space** row shows the tokens each layer is "
            "*poised to say* at the chosen position — including thoughts that never "
            "reach the surface output."
        )
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt", lines=3,
                value="Human: Count to five and introspect deeply.\n\nAssistant:",
            )
        with gr.Row():
            top_k = gr.Slider(3, 20, value=8, step=1, label="top-k")
            position = gr.Slider(-8, -1, value=-1, step=1, label="token position (from end)")
            go = gr.Button("Read", variant="primary")
        surface = gr.Markdown()
        table = gr.Dataframe(
            headers=["layer", "top J-space tokens (token·prob)"],
            wrap=True, label="J-space readout by layer",
        )
        go.click(analyze, [prompt, top_k, position], [surface, table])
        prompt.submit(analyze, [prompt, top_k, position], [surface, table])
    return demo


def main() -> None:
    global MODEL, LENS, READ_LAYERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/DeepSeek-V2-Lite")
    ap.add_argument("--lens", default="out/v2lite_lens.pt")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 11, 15, 19, 23])
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="create a temporary public gradio.live URL (no SSH tunnel "
                         "needed, but traffic is relayed through Gradio)")
    args = ap.parse_args()

    import transformers

    print(f"loading {args.model} (bf16, cuda) ...")
    tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    MODEL = JLensModel(hf, tok)

    if args.lens and os.path.exists(args.lens) and not args.fit:
        LENS = JLens.load(args.lens)
        print(f"loaded lens {args.lens}")
    elif args.fit:
        print(f"fitting lens on {len(FIT_CORPUS)} prompts, layers {args.layers} ...")
        LENS = fit(MODEL, FIT_CORPUS, source_layers=args.layers, dim_batch=args.dim_batch)
        if args.lens:
            os.makedirs(os.path.dirname(args.lens) or ".", exist_ok=True)
            LENS.save(args.lens)
            print(f"saved lens -> {args.lens}")
    else:
        raise SystemExit("no lens: pass --lens PATH (existing) or --fit")

    READ_LAYERS = [l for l in args.layers if l in LENS.jacobians] or sorted(LENS.jacobians)
    print(f"serving on port {args.port} (layers {READ_LAYERS})")
    build_ui().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
