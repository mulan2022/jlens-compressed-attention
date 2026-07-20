# Copyright 2026.
# SPDX-License-Identifier: Apache-2.0
"""Minimal Jacobian Lens (J-lens) — a from-scratch reimplementation.

Paper: Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace
in Language Models" (https://transformer-circuits.pub/2026/workspace).

The lens reads out what an internal activation is *poised to make the model say*.
It linearly transports a residual-stream vector h_l at layer l into the final
layer's basis with the average input->output Jacobian J_l, then decodes it with
the model's own unembedding:

    J_l        = E_{prompt, t, t'>=t} [ d h_final,t' / d h_l,t ]     # [d, d]
    lens_l(h)  = softmax( W_U @ norm( J_l @ h ) )                    # over vocab

The rows of (W_U @ J_l) are the "J-lens vectors": one residual-stream direction
per vocabulary token.

------------------------------------------------------------------------------
Why this transfers to DeepSeek-V2 (MLA + MoE) with ZERO changes to the math
------------------------------------------------------------------------------
The lens only ever touches two things:
  (a) the residual stream at the OUTPUT of each decoder block, and
  (b) the final RMSNorm + unembedding (lm_head).

MLA (Multi-head Latent Attention) compresses the KV cache into a low-rank latent
INSIDE the attention block (kv_lora_rank=512 for V2-Lite) and uses decoupled
RoPE. MoE routes each token to a sparse subset of experts (top-6 of 64) INSIDE
the MLP. Neither changes:
  - the residual-stream width (still d_model = 2048), nor
  - the fact that h_final is a differentiable function of h_l.
Autograd flows through MLA and MoE transparently, so J_l stays [2048, 2048] and
the estimator below is architecture-agnostic.

One honest caveat for MoE: because each token routes to different experts, the
backward pass linearizes around whichever experts fired. Averaging over the
corpus therefore averages J_l over expert-routing configurations — a softer
linear approximation than for a dense model. Worth noting in a write-up; nothing
to change in code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

# Early positions behave as attention sinks (atypical residual statistics) and
# the final position has no next-token target; both are excluded from the average.
SKIP_FIRST_N_POSITIONS = 16


# --------------------------------------------------------------------------- #
# Model wrapper: locate the residual stack inside a HuggingFace causal LM.
# --------------------------------------------------------------------------- #
class JLensModel:
    """Thin wrapper exposing exactly what the lens needs from an HF model.

    Works for any decoder whose ``*ForCausalLM`` holds ``model.layers`` /
    ``model.norm`` / ``model.embed_tokens`` and a top-level ``lm_head`` — this
    covers DeepSeek-V2 (MLA+MoE), Qwen2/3, Llama, Mistral. The residual stream
    is captured as the *output* of each decoder block (pre final-norm), which is
    exactly the h_l / h_final the paper's Jacobian is taken between.
    """

    def __init__(self, hf_model: nn.Module, tokenizer, *, text_path: str = "model",
                 force_bos: bool = True):
        self.hf = hf_model.eval()
        self.tokenizer = tokenizer
        self.force_bos = force_bos
        # The Jacobian is w.r.t. activations, not weights: freeze all params so
        # the only autograd leaf is the residual we mark inside the hook.
        for p in self.hf.parameters():
            p.requires_grad_(False)

        text = self.hf
        for attr in text_path.split("."):
            text = getattr(text, attr)
        self.text = text                      # bare decoder (DeepseekV2Model)
        self.layers: nn.ModuleList = text.layers
        self.final_norm: nn.Module = text.norm
        self.embed: nn.Module = text.embed_tokens
        self.lm_head: nn.Module = self.hf.lm_head  # tie_word_embeddings=False on V2

        cfg = self.hf.config
        self.n_layers: int = cfg.num_hidden_layers
        self.d_model: int = cfg.hidden_size
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"config.num_hidden_layers={self.n_layers} but found "
                f"{len(self.layers)} decoder blocks"
            )

    @property
    def device(self) -> torch.device:
        return self.embed.weight.device

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        ids = enc.input_ids
        # Many tokenizers (incl. DeepSeek) don't prepend BOS by default. Without a
        # BOS the first token acts as a degraded attention sink, corrupting both
        # the surface logits and the J-lens readout — so force one on.
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if self.force_bos and bos is not None and ids[0, 0].item() != bos:
            ids = torch.cat([torch.tensor([[bos]]), ids], dim=1)[:, :max_length]
        return ids.to(self.device)

    def forward(self, input_ids: torch.Tensor):
        # Run only the residual stack (no LM head) so hooks on `layers` fire and
        # the graph is built through MLA/MoE.
        return self.text(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """[..., d_model] residual -> [..., vocab] logits (final norm + lm_head)."""
        w = self.lm_head.weight
        return self.lm_head(self.final_norm(residual.to(w.dtype).to(w.device)))


# --------------------------------------------------------------------------- #
# Residual-stream capture.
# --------------------------------------------------------------------------- #
class _Recorder:
    """Forward hooks that capture (and optionally root the autograd graph at)
    decoder-block outputs, keyed by block index."""

    def __init__(self, layers: nn.ModuleList, at: Sequence[int], graph_root: int):
        self.layers = layers
        self.indices = sorted(set(at) | {graph_root})
        self.graph_root = graph_root
        self.acts: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _hook(self, idx: int):
        def hook(_module, _inp, out):
            # HF blocks return (hidden, present_kv, ...); take the residual.
            h = out if torch.is_tensor(out) else out[0]
            if idx == self.graph_root:
                # Params are frozen, so marking this activation makes it the leaf
                # that roots a graph spanning only graph_root -> target.
                h.requires_grad_(True)
            self.acts[idx] = h
        return hook

    def __enter__(self):
        for i in self.indices:
            self._handles.append(self.layers[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def valid_positions(seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip_first + 1}")
    return idx


# --------------------------------------------------------------------------- #
# The estimator: J_l for a single prompt via dim-batched VJPs.
# --------------------------------------------------------------------------- #
@torch.enable_grad()
def jacobian_for_prompt(
    model: JLensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> dict[int, torch.Tensor]:
    """Estimate J_l = mean_p [ sum_{p'>=p} d h_target,p' / d h_l,p ] for one prompt.

    We never materialize the full 4-D Jacobian. Instead, for a block of output
    dimensions we set a one-hot cotangent at those dims across *all* valid target
    positions p' at once and backprop. Two facts make this the paper's estimator:

      * Causality: h_target,p' depends on h_l,p only for p <= p'. So a cotangent
        summed over all target positions yields, at source position p, exactly
        sum_{p'>=p} (the earlier terms are structurally zero). No masking needed.
      * We then average the resulting gradient rows over the valid source
        positions p.

    Cost: one forward + ceil(d_model / dim_batch) backward passes.
    Returns {layer: [d_model, d_model] fp32 CPU tensor}.
    """
    d = model.d_model
    target_layer = model.n_layers - 1 if target_layer is None else target_layer
    src = sorted(source_layers)
    if src[0] < 0 or src[-1] >= target_layer:
        raise ValueError(f"source_layers must be in [0, {target_layer}); got {src}")

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    pos = valid_positions(seq_len, skip_first=skip_first)

    J = {l: torch.zeros(d, d, dtype=torch.float32) for l in src}
    n_passes = math.ceil(d / dim_batch)

    with _Recorder(model.layers, at=[*src, target_layer], graph_root=min(src)) as rec:
        # Replicate the prompt dim_batch times: each batch row carries a
        # different one-hot output dimension, so one backward yields dim_batch
        # rows of J at once. (Deterministic across rows: eval mode, no dropout.)
        model.forward(input_ids.expand(dim_batch, -1))
        target_act = rec.acts[target_layer]              # [dim_batch, seq, d]
        source_acts = [rec.acts[l] for l in src]

        pos_dev = pos.to(target_act.device)
        rows = torch.arange(dim_batch, device=target_act.device)
        cot = torch.zeros_like(target_act)

        for k, start in enumerate(range(0, d, dim_batch)):
            n = min(dim_batch, d - start)
            cot.zero_()
            # batch row b -> output dim (start+b), at every valid target position.
            cot[rows[:n, None], pos_dev[None, :], start + rows[:n, None]] = 1.0
            grads = torch.autograd.grad(
                target_act, source_acts, grad_outputs=cot,
                retain_graph=(k < n_passes - 1),
            )
            for l, g in zip(src, grads):
                # g[b, p, :] = sum_{p'} d target[p',start+b] / d h_l[p]; mean over p.
                J[l][start : start + n, :] = (
                    g[:n, pos_dev, :].float().mean(dim=1).cpu()
                )
            del grads

    return J


# --------------------------------------------------------------------------- #
# Fit over a corpus, and the fitted lens object.
# --------------------------------------------------------------------------- #
@dataclass
class JLens:
    """Fitted lens: per-layer J_l plus the readout."""

    jacobians: dict[int, torch.Tensor]
    n_prompts: int
    d_model: int

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        J = self.jacobians[layer].to(residual.device, residual.dtype)
        return residual @ J.T

    @torch.no_grad()
    def readout(
        self,
        model: JLensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        position: int = -1,
        top_k: int = 10,
        max_seq_len: int = 512,
    ) -> dict[int, list[tuple[str, float]]]:
        """Return the top-k tokens the lens reads at `position` for each layer."""
        layers = sorted(self.jacobians) if layers is None else layers
        ids = model.encode(prompt, max_length=max_seq_len)
        with _Recorder(model.layers, at=layers, graph_root=min(layers)) as rec:
            model.forward(ids)
            acts = {l: rec.acts[l].detach()[0] for l in layers}  # [seq, d]

        out: dict[int, list[tuple[str, float]]] = {}
        for l in layers:
            h = acts[l][position].float()
            logits = model.unembed(self.transport(h, l))
            probs = torch.softmax(logits.float(), dim=-1)
            p, tok = probs.topk(top_k)
            out[l] = [
                (model.tokenizer.decode([int(t)]), float(pv))
                for t, pv in zip(tok, p)
            ]
        return out

    def save(self, path: str) -> None:
        torch.save(
            {"J": {l: J.half() for l, J in self.jacobians.items()},
             "n_prompts": self.n_prompts, "d_model": self.d_model},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "JLens":
        c = torch.load(path, map_location="cpu", weights_only=True)
        return cls({l: J.float() for l, J in c["J"].items()}, c["n_prompts"], c["d_model"])


def fit(
    model: JLensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    verbose: bool = True,
) -> JLens:
    """Average per-prompt Jacobians into a running mean and return a JLens."""
    tgt = model.n_layers - 1 if target_layer is None else target_layer
    src = list(range(tgt)) if source_layers is None else sorted(source_layers)

    acc = {l: torch.zeros(model.d_model, model.d_model, dtype=torch.float32) for l in src}
    done = 0
    for i, prompt in enumerate(prompts):
        try:
            per = jacobian_for_prompt(
                model, prompt, src, target_layer=tgt,
                dim_batch=dim_batch, max_seq_len=max_seq_len, skip_first=skip_first,
            )
        except ValueError as e:
            if verbose:
                print(f"  skip prompt {i}: {e}")
            continue
        for l in src:
            acc[l] += per[l]
        done += 1
        if verbose:
            nrm = max(per[l].norm().item() for l in src) / math.sqrt(model.d_model)
            print(f"  prompt {i + 1}/{len(prompts)}  max||J||/sqrt(d)={nrm:.3f}")

    if done == 0:
        raise ValueError("no prompts were long enough to fit on")
    return JLens({l: acc[l] / done for l in src}, n_prompts=done, d_model=model.d_model)


# --------------------------------------------------------------------------- #
# Demo (run on the A6000 once V2-Lite weights finish downloading).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fit + read a minimal Jacobian lens.")
    ap.add_argument("--model", default="models/DeepSeek-V2-Lite")
    ap.add_argument("--layers", type=int, nargs="+", default=[9, 13, 17])
    ap.add_argument("--prompt", default="Human: Count to five and introspect deeply.\n\nAssistant:")
    args = ap.parse_args()

    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = JLensModel(hf, tok)
    print(f"loaded {type(hf).__name__}: n_layers={model.n_layers} d_model={model.d_model}")

    # A tiny fitting corpus is enough for a qualitative demo; the paper uses ~1000.
    corpus = [
        "The history of the Roman Empire spans over a thousand years of politics and war.",
        "In organic chemistry, a functional group determines the reactivity of a molecule.",
        "She walked along the beach at sunset, listening to the waves break on the shore.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
        "The stock market reacted sharply to the central bank's interest rate decision.",
        "A recursive function calls itself with a smaller input until it reaches a base case.",
    ]
    lens = fit(model, corpus, source_layers=args.layers, dim_batch=8)

    print(f"\n=== J-lens readout @ last position ===\nprompt: {args.prompt!r}\n")
    for layer, toks in lens.readout(model, args.prompt, top_k=10).items():
        pretty = "  ".join(f"{t!r}:{p:.2f}" for t, p in toks)
        print(f"L{layer:>2}: {pretty}")
