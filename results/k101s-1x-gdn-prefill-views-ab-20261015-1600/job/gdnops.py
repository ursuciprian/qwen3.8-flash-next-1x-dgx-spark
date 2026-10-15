#!/usr/bin/env python3
"""k101: GPU time of the data-movement kernels around the GDN core in one torch-profiler trace (stdlib).

  gdnops.py diff <control trace> <arm trace>   per class: calls and ms in each trace, arm - control
  gdnops.py selftest

A trace is the hook's profile window over one cold 16K prompt (<boot>/hook/prof-k101/*.json[.gz]). The classes are the
k81 rows the views path removes (gathers, device copies, fills, the output scatter) plus the kernels it must keep
(conv, gated norm, chunked delta rule) as a control: those should show the same call count in both traces.
"""
import gzip
import json
import re
import sys

CLASSES = (  # first match wins
    ("scatter (GDN output)", lambda c, n: "_scatter_b12x_gdn_output" in n),
    ("gather (index_select)", lambda c, n: "gather" in n.lower() or "indexselect" in n.lower().replace("_", "")),
    ("memcpy DtoD", lambda c, n: c == "gpu_memcpy" and "DtoD" in n),
    ("copy kernel", lambda c, n: "direct_copy_kernel" in n),
    ("fill / memset", lambda c, n: c == "gpu_memset" or "FillFunctor" in n),
    ("conv1d (kept)", lambda c, n: "_causal_conv1d" in n),
    ("gated norm (kept)", lambda c, n: "layer_norm_fwd_kernel" in n),
    ("GDN delta rule (kept)", lambda c, n: "shareddelta" in n),
)
GPU = ("kernel", "gpu_memcpy", "gpu_memset")


def load(path):
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)


def classify(trace):
    out = {name: [0, 0.0] for name, _ in CLASSES}
    for e in trace.get("traceEvents", []):
        c, n = e.get("cat", ""), e.get("name", "")
        if c not in GPU or e.get("ph") != "X":
            continue
        for name, test in CLASSES:
            if test(c, n):
                out[name][0] += 1
                out[name][1] += float(e.get("dur", 0)) / 1000.0
                break
    return out


def diff(ctl, arm):
    a, b = classify(ctl), classify(arm)
    lines = [f"{'class':26s} {'ctl calls':>9s} {'ctl ms':>9s} {'arm calls':>9s} {'arm ms':>9s} {'arm-ctl ms':>10s}"]
    moved = 0.0
    for name, _ in CLASSES:
        d = b[name][1] - a[name][1]
        if "(kept)" not in name:
            moved += d
        lines.append(f"{name:26s} {a[name][0]:9d} {a[name][1]:9.1f} {b[name][0]:9d} {b[name][1]:9.1f} {d:+10.1f}")
    lines.append(f"data movement (non-kept rows) arm - ctl: {moved:+.1f} ms")
    return "\n".join(lines)


def selftest():
    def ev(cat, name, dur):
        return {"ph": "X", "cat": cat, "name": name, "dur": dur}
    ctl = {"traceEvents": [
        ev("kernel", "void at::native::vectorized_gather_kernel<16, long>", 1000),
        ev("gpu_memcpy", "Memcpy DtoD (Device -> Device)", 500),
        ev("kernel", "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16> >", 200),
        ev("kernel", "_scatter_b12x_gdn_output", 300),
        ev("kernel", "_causal_conv1d_fwd_kernel", 400),
        ev("kernel", "layer_norm_fwd_kernel", 100),
        ev("cpu_op", "aten::index_select", 9999),
    ]}
    arm = {"traceEvents": [
        ev("kernel", "void at::native::elementwise_kernel<128, 4, direct_copy_kernel_cuda>", 50),
        ev("kernel", "_causal_conv1d_fwd_kernel", 400),
        ev("kernel", "layer_norm_fwd_kernel", 100),
    ]}
    a = classify(ctl)
    assert a["gather (index_select)"] == [1, 1.0] and a["memcpy DtoD"] == [1, 0.5], a
    assert a["fill / memset"][0] == 1 and a["scatter (GDN output)"][0] == 1, a
    out = diff(ctl, arm)
    assert "data movement (non-kept rows) arm - ctl: -1.9 ms" in out, out
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        selftest()
    elif sys.argv[1:2] == ["diff"] and len(sys.argv) == 4:
        print(diff(load(sys.argv[2]), load(sys.argv[3])))
    else:
        sys.exit(__doc__)
