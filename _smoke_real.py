import os
import time

import torch
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()
import transformers

from minimal_jlens import JLensModel, fit

PATH = "models/DeepSeek-V2-Lite"
tok = transformers.AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    PATH, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
)
model = JLensModel(hf, tok)
print(f"model loaded: n_layers={model.n_layers} d_model={model.d_model}")
print(f"GPU after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

corpus = [
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
]
t = time.time()
lens = fit(model, corpus, source_layers=[13, 17, 21], dim_batch=2, max_seq_len=128)
print(f"fit done in {time.time()-t:.0f}s | peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

os.makedirs("out", exist_ok=True)
lens.save("out/v2lite_lens.pt")
print("saved out/v2lite_lens.pt")

for prompt in [
    "The capital of France is",
    "Human: Count to five and introspect deeply.\n\nAssistant:",
]:
    print(f"\nPROMPT: {prompt!r}")
    out = lens.readout(model, prompt, top_k=8, position=-1, max_seq_len=128)
    for l in sorted(out):
        toks = "  ".join(f"{t!r}:{p:.2f}" for t, p in out[l])
        print(f"  L{l:>2}: {toks}")
