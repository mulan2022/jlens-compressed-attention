import sys
import torch
import transformers

path = "models/DeepSeek-V2-Lite"
print("transformers", transformers.__version__, "| torch", torch.__version__)
try:
    tok = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    print("LOAD+FORWARD OK | logits", tuple(out.logits.shape),
          "| top next-token:", repr(tok.decode(out.logits[0, -1].argmax().item())))
    # also confirm the residual-stack hook path (what the lens uses) works
    h = model.model(input_ids=ids, use_cache=False)
    print("residual-stack forward OK | last_hidden_state", tuple(h.last_hidden_state.shape))
    print("RESULT: COMPATIBLE")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"RESULT: INCOMPATIBLE — {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
