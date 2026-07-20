import torch
from transformers.utils import logging as L

L.disable_progress_bar()
L.set_verbosity_error()
import transformers

path = "models/DeepSeek-V2-Lite"
tok = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True)
m = transformers.AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
).eval()
BOS = tok.bos_token_id
print("bos_token:", repr(tok.bos_token), "id:", BOS)


@torch.no_grad()
def greedy(ids, n=24):
    # manual greedy with use_cache=False (transformers 5.13's DynamicCache breaks
    # this model's cached path; use_cache=False is what the lens uses anyway)
    cur = ids
    for _ in range(n):
        nxt = m(input_ids=cur, use_cache=False).logits[0, -1].argmax().item()
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    return cur[0, ids.shape[1]:]


@torch.no_grad()
def top6(ids):
    t = m(input_ids=ids, use_cache=False).logits[0, -1].topk(6).indices.tolist()
    return [tok.decode([x]) for x in t]


prompts = [
    "The capital of France is",
    "Once upon a time, in a small village, there lived",
    "Water boils at a temperature of",
    "Human: What is 2+2?\n\nAssistant:",
]
for p in prompts:
    base = tok(p, return_tensors="pt").input_ids.cuda()
    with_bos = torch.cat([torch.tensor([[BOS]], device=base.device), base], dim=1)
    print("\nPROMPT:", repr(p))
    print("  [no BOS ] top6:", top6(base))
    print("  [no BOS ] gen :", repr(tok.decode(greedy(base), skip_special_tokens=True)))
    print("  [+BOS   ] top6:", top6(with_bos))
    print("  [+BOS   ] gen :", repr(tok.decode(greedy(with_bos), skip_special_tokens=True)))
