"""k97 capture hooks (imported by the patched vLLM modules, see sitecustomize.py).

Active only while VLLM_K97_CAPTURE_DIR is set and the file <dir>/ARM exists (the job creates it around one cold 16K
prompt and removes it after), and only for calls of >= 1024 rows (prefill chunks; boot warmups run before ARM exists,
decode steps are smaller). Saved per call, numbered in call order so chunks concatenate back into prompt order:
  moe5.NNNNN.pt / moe47.NNNNN.pt   {"x": BF16 [rows, 2560] routed-MoE input, "ids": [rows, 10], "w": [rows, 10]
                                   normalized topk weights, "n_moe": B12xExperts instances seen}; layer = construction
                                   order of B12xExperts (48 decoder layers, then the MTP drafter)
  hcdown5.NNNNN.pt / hcup5.NNNNN.pt  {"x": BF16 source of layer 5's attention HC down (K10240 -> N336) / up
                                   (K320 -> N10240) projection, "name": layer name}
"""
import os
import re

D = os.environ.get("VLLM_K97_CAPTURE_DIR")
MIN_ROWS = 1024
MOE_LAYERS = {5: "moe5", 47: "moe47"}
HC = re.compile(r"(^|\.)layers\.5\.attn_hyper_connection\.input_mix_weight_(down|up)")
_moe = {}
_seq = [0]


def armed():
    return D is not None and os.path.exists(os.path.join(D, "ARM"))


def _save(tag, obj):
    import torch
    _seq[0] += 1
    torch.save(obj, os.path.join(D, f"{tag}.{_seq[0]:05d}.pt"))


def moe_register(inst):
    if D is not None:
        _moe[id(inst)] = len(_moe)


def moe(inst, x, ids, w):
    if x.shape[0] < MIN_ROWS or not armed():
        return
    tag = MOE_LAYERS.get(_moe.get(id(inst), -1))
    if tag:
        _save(tag, {"x": x.detach().cpu(), "ids": ids.detach().cpu(), "w": w.detach().float().cpu(), "n_moe": len(_moe)})


def hc(name, x):
    if x.shape[0] < MIN_ROWS or not armed():
        return
    m = HC.search(str(name))
    if m:
        _save("hc" + m.group(2) + "5", {"x": x.detach().cpu(), "name": str(name)})
