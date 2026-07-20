# Copyright 2026.
# SPDX-License-Identifier: Apache-2.0
"""Correctness test for minimal_jlens on a tiny CPU causal decoder.

Validates the dim-batched VJP estimator in `jacobian_for_prompt` against a
brute-force Jacobian built one output element at a time. The toy model has real
causal attention, so h_target[p'] genuinely depends on h_l[p<=p'] and the
causal (sum over p'>=p) reduction is actually exercised — not just a diagonal.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from jlens._core import (
    JLensModel,
    _Recorder,
    jacobian_for_prompt,
    valid_positions,
)

torch.manual_seed(0)


class _Block(nn.Module):
    """h + causal single-head attention(h) + small MLP(h). Mixes positions
    causally so the Jacobian has real off-diagonal (p' > p) structure."""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.d = d
        for p in self.parameters():
            with torch.no_grad():
                p.mul_(0.1)  # small gain -> well-conditioned Jacobian

    def forward(self, h, **_):
        # h: [b, seq, d]
        b, s, d = h.shape
        q, k, v = self.q(h), self.k(h), self.v(h)
        scores = q @ k.transpose(-1, -2) / (d ** 0.5)
        mask = torch.triu(torch.ones(s, s), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        h = h + self.o(attn @ v)
        h = h + self.mlp(h)
        return (h,)  # HF-style tuple output


class _ByteTok:
    bos_token_id = 0

    def __call__(self, text, *, return_tensors="pt", truncation=True, max_length=128):
        ids = [self.bos_token_id] + [1 + (c % 20) for c in text.encode()][: max_length - 1]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids, **_):
        return "".join(chr(96 + int(i)) for i in ids)


class _ToyModel(nn.Module):
    """Bare decoder: .layers / .norm / .embed_tokens, callable like DeepseekV2Model."""

    def __init__(self, n_layers, d, vocab):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Block(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, input_ids=None, use_cache=False, **_):
        h = self.embed_tokens(input_ids)
        for blk in self.layers:
            h = blk(h)[0]
        return SimpleNamespace(last_hidden_state=h)


class _ToyForCausalLM(nn.Module):
    def __init__(self, n_layers=4, d=6, vocab=32):
        super().__init__()
        self.model = _ToyModel(n_layers, d, vocab)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=n_layers, hidden_size=d)

    def forward(self, input_ids=None, use_cache=False, **_):
        h = self.model(input_ids=input_ids).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(self.model.norm(h)))


def brute_force_J(model, prompt, layer, target_layer, skip_first):
    """Full Jacobian, one output element at a time, then reduced exactly like the
    estimator: J_ref[j, i] = mean_{p} sum_{p'} d target[p', j] / d h_l[p, i]."""
    ids = model.encode(prompt, max_length=128)
    seq = ids.shape[1]
    pos = valid_positions(seq, skip_first=skip_first)
    d = model.d_model

    with torch.enable_grad(), _Recorder(model.layers, at=[layer, target_layer], graph_root=layer) as rec:
        model.forward(ids)                       # batch size 1, no replication
        tgt = rec.acts[target_layer]             # [1, seq, d]
        srcs = [rec.acts[layer]]

        J_ref = torch.zeros(d, d)
        last = seq  # count backward passes remaining for retain_graph
        total = int(pos.numel()) * d
        n = 0
        for pp in pos.tolist():
            for j in range(d):
                cot = torch.zeros_like(tgt)
                cot[0, pp, j] = 1.0
                n += 1
                g = torch.autograd.grad(tgt, srcs, grad_outputs=cot, retain_graph=(n < total))[0]
                # g[0, p, i] = d target[pp, j] / d h_l[p, i]; accumulate sum over p'=pp.
                J_ref[j, :] += g[0, pos, :].sum(0)  # sum over source p (we'll divide)
        # We summed over both p' (outer loop) and p (inner .sum). Estimator means
        # over p, so divide by number of source positions.
        J_ref /= pos.numel()
    return J_ref


def main():
    hf = _ToyForCausalLM(n_layers=4, d=6, vocab=32).eval()
    model = JLensModel(hf, _ByteTok())
    prompt = "the quick brown fox jumps over the lazy dog and runs away quickly today"
    skip_first = 2  # short prompt -> small skip so we keep valid positions
    layer, target = 1, 3

    J_est = jacobian_for_prompt(
        model, prompt, [layer], target_layer=target, dim_batch=4, skip_first=skip_first
    )[layer]
    J_ref = brute_force_J(model, prompt, layer, target, skip_first)

    max_abs = (J_est - J_ref).abs().max().item()
    rel = max_abs / J_ref.abs().max().item()
    offdiag = (J_ref - torch.diag(torch.diag(J_ref))).abs().max().item()
    print(f"J shape            : {tuple(J_est.shape)}")
    print(f"max|J_ref| off-diag: {offdiag:.4e}   (proves causal position-mixing is present)")
    print(f"max abs diff       : {max_abs:.3e}")
    print(f"max rel diff       : {rel:.3e}")
    assert offdiag > 1e-4, "toy model has no cross-position structure — test is vacuous"
    assert max_abs < 1e-4, f"estimator mismatch: {max_abs}"
    print("\nPASS: dim-batched VJP estimator matches the brute-force Jacobian.")


if __name__ == "__main__":
    main()
