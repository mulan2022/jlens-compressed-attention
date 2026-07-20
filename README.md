# Jacobian Lens on Compressed-Attention Architectures

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A research toolkit for applying the **Jacobian Lens** (J-lens) interpretability method to language models with compressed attention — specifically **DeepSeek-V2-Lite (MLA)** and **Qwen2.5-3B (GQA)** — and analyzing whether J-space geometry aligns with the attention read subspace.

Based on Gurnee et al., ["Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace) (2026).

## What is the Jacobian Lens?

The Jacobian lens reads out what an internal activation is *poised to make the model say*. Given a residual-stream vector $h_l$ at layer $l$, it transports it to the final layer's basis using the average input→output Jacobian $J_l$, then decodes with the model's unembedding:

$$J_l = \mathbb{E}\left[ \frac{\partial h_{\text{final}}}{\partial h_l} \right], \quad \text{lens}_l(h) = \text{softmax}(W_U \cdot \text{norm}(J_l \cdot h))$$

The rows of $W_U J_l$ are the **J-lens vectors** — one residual-stream direction per vocabulary token — and the subspace they span is **J-space**.

## Key Result (Spoiler)

J-space is **not** preferentially aligned with the attention read subspace in either MLA or GQA. The per-token mean sits at the random baseline (MLA: −0.15σ; GQA: −0.30σ), despite a 512-dim (MLA) or 256-dim (GQA) value-pathway bottleneck. This is a *calibrated negative result*: the same measurement pipeline recovers planted in-subspace energy exactly (positive control passes), so the null is real.

## Project Structure

```
├── minimal_jlens.py                  # Core J-lens library (from-scratch, architecture-agnostic)
├── test_minimal_jlens.py             # Correctness test: VJP estimator vs brute-force Jacobian
│
├── jlens_repl.py                     # Interactive terminal REPL — "mind reader" demo
├── jlens_app.py                      # Gradio web UI for interactive exploration
│
├── jlens_stats_direct.py             # Per-token J-space vs. read-subspace statistics
├── jlens_collect_stats.py            # Older version (Gradio-based, kept for reference)
│
├── analysis_remote.py                # Cross-layer projection, Haar aggregate nulls, positive control
├── analysis_local.py                 # Z-score distributions, CJK token share, markup token analysis
├── analysis_bootstrap.py             # Cluster bootstrap for token-share significance
├── analysis_nextblock_pertoken.py    # Per-token z-scores vs. next-block read subspace (GQA)
├── analysis_nextblock_pertoken_mla.py # Same for MLA
├── analysis_rope.py                  # RoPE-inclusive MLA read subspace robustness check
├── analysis_addendum.py              # Raw-W_U control for cross-layer elevation
│
├── inspect_attention_read.py         # Universal read/OV subspace inspector (MLA/GQA/MHA)
├── inspect_mla_weights.py            # MLA compressed-weight accounting + SVD
│
├── make_figures.py                   # Generate analysis figures
├── make_comparison_figure.py         # MLA vs. GQA comparison figure
├── jlens_plot.py                     # Older plot script (jlens_collect_stats companion)
│
├── refit_lens.py                     # Batch lens fitting (corpus-stability experiment)
├── compare_refits.py                 # Compare results across lens refits
├── corpus_alt1.txt                   # Alternate fitting corpus 1
├── corpus_alt2.txt                   # Alternate fitting corpus 2
│
├── _smoke_real.py                    # End-to-end smoke test: fit lens + readout
├── _compat_test.py                   # Environment compatibility check
├── _diag_gen.py                      # Diagnostic: model forward pass (BOS behavior)
├── dl_loop.sh                        # Resilient ModelScope download script
│
├── out/                              # Analysis outputs (.npz, .json, .png)
├── models/DeepSeek-V2-Lite-tok/      # Tokenizer for DeepSeek-V2-Lite
│
├── README-CN.md                      # Chinese-language background & paper summary
└── MLA适配原理.md                     # Technical note on MLA adaptation (Chinese)
```

## Requirements

```bash
pip install torch transformers safetensors numpy scipy matplotlib
pip install gradio  # only for jlens_app.py
```

- Python ≥ 3.8, PyTorch ≥ 2.0, Transformers ≥ 4.45
- GPU with ≥ 24 GB VRAM for lens fitting (A6000 used in development)
- Most analysis scripts run on CPU (weight-slicing via safetensors, no forward pass)

## Quick Start

### 1. Download Models

```bash
# DeepSeek-V2-Lite-Chat (MLA)
modelscope download --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --local_dir models/DeepSeek-V2-Lite-Chat
# Or use the resilient retry loop:
bash dl_loop.sh

# DeepSeek-V2-Lite (base)
modelscope download --model deepseek-ai/DeepSeek-V2-Lite \
    --local_dir models/DeepSeek-V2-Lite

# Qwen2.5-3B (GQA, from HuggingFace)
huggingface-cli download Qwen/Qwen2.5-3B --local-dir models/Qwen2.5-3B
```

### 2. Verify Environment

```bash
python _compat_test.py
```

### 3. Fit a Quick Lens

```bash
python _smoke_real.py
# Fits a lens on 5 paragraphs, layers 13, 17, 21
# Saves out/v2lite_lens.pt (~65 MB)
```

### 4. Interactive Exploration

```bash
# Terminal REPL: type text, see what the model reads out at each layer
python jlens_repl.py --model models/DeepSeek-V2-Lite --lens out/v2lite_lens.pt

# Web UI (SSH port-forward to your laptop: ssh -L 7860:localhost:7860 ...)
python jlens_app.py --model models/DeepSeek-V2-Lite --lens out/v2lite_lens.pt --fit
```

## Usage

### Fit a J-lens on Your Own Corpus

```python
from minimal_jlens import JLensModel, fit
import transformers

tok = transformers.AutoTokenizer.from_pretrained("path/to/model", trust_remote_code=True)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    "path/to/model", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="cuda"
)
model = JLensModel(hf, tok)

corpus = ["paragraph one...", "paragraph two...", ...]  # ~10-1000 prompts
lens = fit(model, corpus, source_layers=[9, 11, 13, 15, 17, 19, 21, 23])

lens.readout(model, "The capital of France is", position=-1, top_k=8)
lens.save("out/my_lens.pt")
```

`JLensModel` works for any HuggingFace decoder with `model.layers` / `model.norm` / `lm_head` — covers DeepSeek-V2 (MLA+MoE), Qwen2/3, Llama, Mistral, etc.

### Run the Full Analysis Pipeline

```bash
# 1. Collect per-token statistics
python jlens_stats_direct.py --model models/DeepSeek-V2-Lite-Chat \
    --lens out/v2lite_chat_lens.pt --layers 9 11 13 15 17 19 21 23 \
    --out out/v2lite_stats.npz

python jlens_stats_direct.py --model models/Qwen2.5-3B \
    --lens out/qwen25_lens.pt --layers 12 15 18 21 24 27 30 33 \
    --out out/qwen25_stats.npz

# 2. Cross-layer projection + Haar nulls + positive control
python analysis_remote.py --model models/DeepSeek-V2-Lite-Chat \
    --lens out/v2lite_chat_lens.pt --out out/mla_remote.npz
python analysis_remote.py --model models/Qwen2.5-3B \
    --lens out/qwen25_lens.pt --out out/gqa_remote.npz

# 3. Distribution analysis + CJK/markup tokens
python analysis_local.py

# 4. Bootstrap significance
python analysis_bootstrap.py

# 5. Figures
python make_figures.py
python make_comparison_figure.py
```

### Robustness Checks

```bash
# RoPE-inclusive read subspace (MLA, rank 576 instead of 512)
python analysis_rope.py --model models/DeepSeek-V2-Lite-Chat \
    --lens out/v2lite_chat_lens.pt --out out/mla_rope.npz

# Raw unembedding control
python analysis_addendum.py --model models/DeepSeek-V2-Lite-Chat \
    --out out/mla_raw.npz

# Corpus stability: refit on disjoint corpora, compare
python refit_lens.py --model models/DeepSeek-V2-Lite-Chat \
    --corpus corpus_alt1.txt --layers 9 11 13 15 17 19 21 23 \
    --lens out/v2lite_chat_lens_alt1.pt
python analysis_remote.py --model models/DeepSeek-V2-Lite-Chat \
    --lens out/v2lite_chat_lens_alt1.pt --out out/mla_remote_alt1.npz
python compare_refits.py
```

### Inspect Attention Subspaces

```bash
# Architecture detection + subspace rank (no model weights)
python inspect_attention_read.py --model models/Qwen2.5-3B --dry-run

# Full SVD + unembedding overlap
python inspect_attention_read.py --model models/Qwen2.5-3B \
    --unembed --layers 5 10 15 20 25 30

# J-space energy fraction in read subspaces
python inspect_attention_read.py --model models/Qwen2.5-3B \
    --jspace-lens out/qwen25_lens.pt --layers 8 12 16 20 24 28 32

# MLA-specific: SVD of W_DKV and frozen-attention OV circuit
python inspect_mla_weights.py --model models/DeepSeek-V2-Lite
```

## Why J-lens Needs Zero Changes for MLA/MoE

MLA compresses the KV cache *inside* the attention block (kv_lora_rank=512 for V2-Lite). MoE routes tokens to sparse experts *inside* the MLP (top-6 of 64). Neither changes:

- The residual-stream width ($d_{\text{model}} = 2048$)
- The fact that $h_{\text{final}}$ is a differentiable function of $h_l$

Autograd flows through MLA and MoE transparently. $J_l$ stays $[2048, 2048]$ and the estimator is architecture-agnostic. The only caveat for MoE: because each token routes to different experts, the backward pass linearizes around whichever experts fired — averaging over the corpus averages $J_l$ over routing configurations.

## Read Subspace Definitions

| Architecture | Read Subspace | Rank | Baseline $r/d$ |
|:--|:--|:--|:--|
| MLA (DeepSeek-V2) | $\text{row}(W^{DKV})$ — first `kv_lora_rank` rows of `kv_a_proj_with_mqa` | 512 | 0.250 |
| GQA (Qwen2.5) | $\text{row}([W^K; W^V])$ — concat of K and V projections | 512 | 0.250 |
| GQA V-only | $\text{row}(W^V)$ | 256 | 0.125 |
| MLA + RoPE | full `kv_a_proj_with_mqa` (latent + decoupled RoPE key) | 576 | 0.281 |

Query projections are excluded because they determine attention *routing*, not content carriage — including them makes the subspace nearly full-rank and the test vacuous.

## Computational Budget

| Step | GPU | Time |
|:--|:--|:--|
| Lens fitting (10 prompts) | A6000 | 2–5 min |
| Stats collection (`jlens_stats_direct.py`) | A6000 | 5–10 min |
| Remote analyses (`analysis_remote.py`) | CPU | 1–2 min |
| Local analyses | CPU | < 30 s |
| Bootstrap | CPU | < 10 s |

## Output Files

The `out/` directory contains pre-computed analysis results:

| File | Content |
|:--|:--|
| `*_stats.npz` | Per-token energy fractions, z-scores, categories |
| `*_remote.npz` | Cross-layer energy matrices, Haar null parameters |
| `*_raw.npz` | Raw unembedding control |
| `*_rope.npz` | RoPE-inclusive variant |
| `local_analysis.json` | CJK share, markup share, distribution stats |
| `bootstrap_analysis.json` | Bootstrap CIs and p-values |
| `*.png` | Generated figures |

Lens `.pt` files (~65 MB each) are excluded from version control and must be regenerated or obtained separately.

## License

Apache License 2.0. See [LICENSE](LICENSE).

The `jacobian-lens/` subdirectory (Anthropic's reference implementation, also Apache 2.0) is gitignored — clone it separately if needed:
```bash
git clone https://github.com/anthropics/jacobian-lens.git
```

Model weights and tokenizers are subject to their respective licenses.

---

*Paper (in submission) is maintained in a separate private repository.*
