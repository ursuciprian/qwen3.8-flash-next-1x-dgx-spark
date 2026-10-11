#!/usr/bin/env python3
"""k91 moe-prefill-probe (1x #27): one routed-MoE layer of Qwen3.8-Flash-Next (H 2560, I 640, 512 experts, top-10,
checkpoint NVFP4 weights of layer 5, numerics also on layer 47) at prefill M 1264 / 3024 / 6048 / 8192, b12x lowerings
vs the canonical-layout NVFP4 W4A4 MoE of FlashInfer PR #5863 (vendored, run_moe_w4a4 called directly).

Variants (each in its own subprocess with a timeout; a failure is logged and the rest runs):
  b12x-m32               serving anchor: capacity plan max_tokens 8192, dynamic tile 32x128 (what the shipped
                         1x autotune picked, inferred from the k81 scratch dims 98272 = 8192*10/32*32 + 511*32),
                         per-expert FC1/FC2 activation scales (shipped contract)
  b12x-m64, b12x-m128    same plan, tile forced through B12X_DYNAMIC_TILE_MN, per-expert scales
  b12x-m128-l1max-split  tile 128, FC1 input scale = layer max (= VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13), so
                         shared input quantization and the NVFP4 split-materialized phase kernels become eligible
  b12x-m128-l1max-mono   as above with B12X_NVFP4_DYNAMIC_MATERIALIZED=0 (separates sharing from the split)
  fi5863                 PR #5863 kernel on the [gate; up] checkpoint order, scalar layer-max FC1 and FC2 scales
Routing (synthetic, labelled): uniform (top-10 of N(0,1) logits) and skew (Zipf 0.3 expert popularity, hottest
expert ~4.6x the mean, sampled without replacement). No router captures exist for this model.
Activations: N(0, sigma) BF16 with sigma = calibrated FC1 amax / 5 of that layer (amax = median input_scale * 2688).
Timing: eager (prefill is not graphed in serving), CUDA events, 256 MB L2 flush before every call, median of 15
after 3 warmups. Numerics: rows 0..255 at M 6048 against an FP32 dequantized-weight reference (BF16 activations,
no activation quantization), layers 5 and 47, both routings. Gate: err <= 1.10 x err(b12x-m32) + 0.005.

Usage (shipped 1x image): moe_probe.py all /out --ckpt <snapshot dir> | run <variant> /out --ckpt <dir> |
report /out (stdlib only) | selftest (stdlib only) | check --ckpt <dir> (CPU, in the image).
"""
import json, os, statistics, subprocess, sys, time, traceback

E, H, I, TOPK = 512, 2560, 640, 10
MS = (1264, 3024, 6048, 8192)
TIME_LAYER, NUM_LAYERS, NUM_M, NUM_ROWS = 5, (5, 47), 6048, 256
PATTERNS = ("uniform", "skew")
CAP = 8192
FLOP_PER_TOKEN = TOPK * 3 * 2 * H * I  # 98.3 MFLOP per token per layer
# k81 shipped 1x, routed MoE ms per chunk / 48 layers (chunks 6048, 6048, 3024, 1264): the anchor
K81_MS = {1264: 319.8 / 48, 3024: 431.0 / 48, 6048: (609.8 + 597.4) / 2 / 48}
VARIANTS = {  # run in this order (value first), so a budget stop drops the lowest-value arms
    "b12x-m32": ("b12x", {"B12X_DYNAMIC_TILE_MN": "32x128"}, "expert"),
    "b12x-m128": ("b12x", {"B12X_DYNAMIC_TILE_MN": "128x128"}, "expert"),
    "b12x-m128-l1max-split": ("b12x", {"B12X_DYNAMIC_TILE_MN": "128x128"}, "layer"),
    "fi5863": ("fi5863", {}, "layer"),
    "b12x-m128-l1max-mono": ("b12x", {"B12X_DYNAMIC_TILE_MN": "128x128", "B12X_NVFP4_DYNAMIC_MATERIALIZED": "0"}, "layer"),
    "b12x-m64": ("b12x", {"B12X_DYNAMIC_TILE_MN": "64x128"}, "expert"),
}
ANCHOR = "b12x-m32"
COLS = ("variant", "layer", "pattern", "m", "ms", "tflops", "err_ref", "active", "max_rows", "mean_rows", "config")


# ---------------------------------------------------------------- GPU side (torch imported lazily)
def load_layer(ckpt, layer):
    import torch
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts"
    files, out = {}, {}

    def get(k):
        f = idx[k]
        if f not in files:
            files[f] = safe_open(os.path.join(ckpt, f), "pt", device="cpu")
        return files[f].get_tensor(k)

    for name, p in (("g", "gate_proj"), ("u", "up_proj"), ("d", "down_proj")):
        out[name + "w"] = torch.stack([get(f"{pre}.{e}.{p}.weight") for e in range(E)]).cuda()
        out[name + "s"] = torch.stack([get(f"{pre}.{e}.{p}.weight_scale") for e in range(E)]).cuda()
        out[name + "g"] = torch.stack([get(f"{pre}.{e}.{p}.weight_scale_2").float().reshape(()) for e in range(E)]).cuda()
        out[name + "i"] = torch.stack([get(f"{pre}.{e}.{p}.input_scale").float().reshape(()) for e in range(E)]).cuda()
    return out


def make_inputs(lw, pattern, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    sigma = float(lw["gi"].median()) * 6 * 448 / 5
    x = (torch.randn(CAP, H, generator=g) * sigma).to(torch.bfloat16).cuda()
    if pattern == "uniform":
        ids = torch.topk(torch.randn(CAP, E, generator=g), TOPK, dim=-1).indices
    else:
        p = (torch.arange(E, dtype=torch.float64) + 1) ** -0.3
        p = p[torch.randperm(E, generator=g)]
        ids = torch.multinomial(p.float().expand(CAP, E), TOPK, replacement=False, generator=g)
    w = torch.softmax(torch.randn(CAP, TOPK, generator=g), dim=-1)
    return x, ids.to(torch.int32).cuda().contiguous(), w.float().cuda().contiguous()


def route_stats(ids, m):
    import torch
    c = torch.bincount(ids[:m].reshape(-1).long(), minlength=E)
    return int((c > 0).sum()), int(c.max()), m * TOPK / E


def dequant(w_u8, sf, gs):
    """[R, K/2] packed E2M1 (low nibble first) x [R, K/16] E4M3 x scalar -> fp32 [R, K]."""
    import torch
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=w_u8.device)
    b = w_u8.view(torch.uint8)
    v = lut[torch.stack(((b & 0xF).long(), (b >> 4).long()), dim=-1).reshape(b.shape[0], -1)]
    return v * sf.float().repeat_interleave(16, dim=-1) * gs


def reference(lw, x, ids, w):
    import torch
    xf = x[:NUM_ROWS].float()
    out = torch.zeros(NUM_ROWS, H, device=x.device)
    sub = ids[:NUM_ROWS].long()
    for e in torch.unique(sub).tolist():
        r, k = (sub == e).nonzero(as_tuple=True)
        gt = xf[r] @ dequant(lw["gw"][e], lw["gs"][e], lw["gg"][e]).T
        up = xf[r] @ dequant(lw["uw"][e], lw["us"][e], lw["ug"][e]).T
        y = (torch.nn.functional.silu(gt) * up).to(torch.bfloat16).float() @ dequant(lw["dw"][e], lw["ds"][e], lw["dg"][e]).T
        out.index_add_(0, r, y * w[r, k, None])
    return out


def rel_err(a, ref):
    return float((a.float() - ref).norm() / ref.norm().clamp_min(1e-30))


class B12x:
    def __init__(self, lw, fc1):
        import torch
        from b12x._lib.intrinsics import swizzle_block_scale
        from b12x.moe import fused_moe
        from b12x.preparation import PreparationSession, PreparedCall
        self.fm = fused_moe
        # vLLM order [gate; up] = b12x W31; b12x flips it in place to [up; gate] at first use, as in serving
        w13 = torch.cat([lw["gw"], lw["uw"]], dim=1).contiguous()
        s13 = swizzle_block_scale(torch.cat([lw["gs"], lw["us"]], dim=1).contiguous())
        s2 = swizzle_block_scale(lw["ds"].contiguous())
        a1 = 1.0 / (lw["gi"].max().reshape(1) if fc1 == "layer" else lw["gi"])
        plan_w = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("modelopt_nvfp4"),
                                          w13_layout=fused_moe.W13Layout.W31),
            activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A4, nonlinearity="silu",
                                                io_dtype=torch.bfloat16),
            geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I))
        self.experts = fused_moe.prepare_weights(plan=plan_w, weights=fused_moe.PackedWeights(
            w13=w13, w2=lw["dw"].contiguous(), w13_block_scales=s13, w2_block_scales=s2,
            w13_global_scales=lw["gg"].contiguous(), w2_global_scales=lw["dg"].contiguous(),
            input_scale=a1.float().contiguous(), intermediate_scale=(1.0 / lw["di"]).float().contiguous()))
        from b12x.preparation import FrozenMapping
        self.plan = fused_moe.plan_execution(
            experts=self.experts,
            capacity=fused_moe.ExecutionCapacity(max_tokens=CAP, top_k=TOPK, warmup_token_counts=(CAP,), route_num_experts=0),
            routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False),
            invocation=FrozenMapping({"tuning_route_pattern": "cyclic_disjoint_topk"}))
        hid = torch.empty(CAP, H, dtype=torch.bfloat16, device="cuda").normal_(0, 0.125)
        out = torch.empty_like(hid)
        rr = torch.arange(CAP, dtype=torch.int32, device="cuda")[:, None] * TOPK + torch.arange(TOPK, dtype=torch.int32, device="cuda")
        ids, wts = rr.remainder(E).contiguous(), torch.full((CAP, TOPK), 1.0 / TOPK, device="cuda")

        def factory(state):
            scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in state.scratch.scratch_specs())
            b = state.bind(scratch=scratch, a=hid, experts=self.experts, topk_weights=wts, topk_ids=ids, output=out,
                           input_scales_static=True)
            return PreparedCall(run=b.run, output=out, owners=(b, scratch))
        counts = getattr(self.plan, "token_counts", None)
        req = (self.plan.request(name="k91.moe", prepare_calls={c: factory for c in counts}) if counts is not None
               else self.plan.request(name="k91.moe", prepare_call=factory))
        with PreparationSession(device=torch.device("cuda"), autotune=False, compile_workers=2) as s:
            s.prepare((req,))
            s.freeze()
        nbytes = sum(spec.nbytes for spec in self.plan.scratch_specs())
        self.scratch = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        self.config = describe(self.plan)

    def fn(self, x, ids, w, out):
        b = self.fm.bind(self.plan, scratch=self.scratch, a=x, experts=self.experts, topk_weights=w, topk_ids=ids,
                         output=out, input_scales_static=True)
        return lambda: self.fm.run(binding=b)


def describe(plan):
    for obj in (plan, *getattr(plan, "variants", {}).values()):
        sel = getattr(obj, "selection", None)
        if sel is not None:
            return f"{getattr(sel, 'source', '?')} {getattr(sel, 'config', sel)}".replace("\t", " ").replace("\n", " ")
    return "?"


class FI5863:
    def __init__(self, lw, fc1):
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
        import fi5863_moe_nvfp4_w4a4 as k
        from b12x._lib.intrinsics import swizzle_block_scale
        self.k = k
        self.w13 = torch.cat([lw["gw"], lw["uw"]], dim=1).contiguous()  # [gate; up], the kernel's canonical order
        self.s13 = swizzle_block_scale(torch.cat([lw["gs"], lw["us"]], dim=1).contiguous())
        self.s2 = swizzle_block_scale(lw["ds"].contiguous())
        self.w2, self.g13, self.g2 = lw["dw"].contiguous(), lw["gg"].contiguous(), lw["dg"].contiguous()
        self.a13 = (1.0 / lw["gi"].max()).reshape(1).float()
        self.a2 = (1.0 / lw["di"].max()).reshape(1).float()
        self.ws = {}
        self.config = "PR #5863 a0997210 run_moe_w4a4, scalar layer-max a13/a2"

    def fn(self, x, ids, w, out):
        m = x.shape[0]
        if m not in self.ws:
            self.ws[m] = self.k.allocate_workspace(m, H, I, E, TOPK, True, x.device)
        ws = self.ws[m]
        return lambda: self.k.run_moe_w4a4(x, ids, w, self.w13, self.s13, self.g13, self.w2, self.s2, self.g2,
                                           self.a13, self.a2, out, ws)


def timed(fn, flush, iters=15, warm=3):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        t.append(a.elapsed_time(b))
    return statistics.median(t)


def run_variant(name, out_dir, ckpt):
    import torch
    kind, _, fc1 = VARIANTS[name]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    rows = []
    with open(os.path.join(out_dir, f"{name}.tsv"), "w") as f:
        for layer in NUM_LAYERS:
            t0 = time.time()
            lw = load_layer(ckpt, layer)
            impl = (B12x if kind == "b12x" else FI5863)(lw, fc1)
            print(f"{name} L{layer}: load+prepare {time.time() - t0:.0f} s, {impl.config}", flush=True)
            for pi, pat in enumerate(PATTERNS):
                x, ids, w = make_inputs(lw, pat, 9100 + 10 * layer + pi)
                for m in (MS if layer == TIME_LAYER else (NUM_M,)):
                    out = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
                    fn = impl.fn(x[:m], ids[:m], w[:m], out)
                    fn(); torch.cuda.synchronize()
                    err = rel_err(out[:NUM_ROWS], reference(lw, x, ids, w)) if m == NUM_M else float("nan")
                    ms = timed(fn, flush) if layer == TIME_LAYER else float("nan")
                    act, mx, mean = route_stats(ids, m)
                    row = (name, layer, pat, m, f"{ms:.4f}", f"{m * FLOP_PER_TOKEN / ms / 1e9:.1f}", f"{err:.5f}",
                           act, mx, f"{mean:.1f}", impl.config)
                    f.write("\t".join(map(str, row)) + "\n"); f.flush()
                    print("\t".join(map(str, row[:10])), flush=True)
            del impl, lw
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- driver and report (stdlib only)
def run_all(out_dir, ckpt, budget_s=1560):
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    for name, (_, env, _) in VARIANTS.items():
        left = budget_s - (time.time() - start)
        if left < 120:
            print(f"{name}: skipped, budget ({budget_s} s) spent", flush=True)
            continue
        t0 = time.time()
        with open(os.path.join(out_dir, f"{name}.log"), "w") as log:
            try:
                rc = subprocess.run([sys.executable, os.path.abspath(__file__), "run", name, out_dir, "--ckpt", ckpt],
                                    env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=min(450 if VARIANTS[name][0] == "b12x" else 300, left)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        print(f"{name}: rc={rc} {time.time() - t0:.0f} s", flush=True)


def read_rows(out_dir):
    rows = []
    for name in VARIANTS:
        p = os.path.join(out_dir, f"{name}.tsv")
        if not os.path.exists(p):
            continue
        for line in open(p):
            v = line.rstrip("\n").split("\t")
            if len(v) == len(COLS):
                r = dict(zip(COLS, v))
                for k in ("layer", "m", "active", "max_rows"):
                    r[k] = int(r[k])
                for k in ("ms", "tflops", "err_ref", "mean_rows"):
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def report(out_dir, rows=None):
    rows = read_rows(out_dir) if rows is None else rows
    by = {(r["variant"], r["layer"], r["pattern"], r["m"]): r for r in rows}
    names = [n for n in VARIANTS if any(r["variant"] == n for r in rows)]
    L = []
    # numerics gate
    gate = {}
    L.append(f"== numerics: rel L2 error vs FP32 dequant reference, rows 0..{NUM_ROWS - 1} at M {NUM_M}"
             f" (gate: <= 1.10 x {ANCHOR} + 0.005 in every cell)")
    L.append("variant".ljust(24) + "".join(f"L{l} {p}".rjust(14) for l in NUM_LAYERS for p in PATTERNS) + "  gate")
    for n in names:
        cells, ok, judged = [], True, 0
        for l in NUM_LAYERS:
            for p in PATTERNS:
                r, a = by.get((n, l, p, NUM_M)), by.get((ANCHOR, l, p, NUM_M))
                if r is None or r["err_ref"] != r["err_ref"]:
                    cells.append("n/a".rjust(14)); ok = False; continue
                cells.append(f"{r['err_ref']:14.4f}")
                if a is None or a["err_ref"] != a["err_ref"]:
                    continue  # anchor cell missing: this cell is not judged
                judged += 1
                if r["err_ref"] > 1.10 * a["err_ref"] + 0.005:
                    ok = False
        gate[n] = ("PASS" if ok else "FAIL") if judged else "NOANCHOR"
        L.append(n.ljust(24) + "".join(cells) + "  " + gate[n])
    # anchor
    L.append("")
    L.append(f"== anchor: {ANCHOR} layer {TIME_LAYER} vs k81 serving (routed MoE ms per layer per chunk)")
    dev6048 = []
    for p in PATTERNS:
        parts = []
        for m, ref in sorted(K81_MS.items()):
            r = by.get((ANCHOR, TIME_LAYER, p, m))
            if r:
                d = 100 * (r["ms"] / ref - 1)
                parts.append(f"M {m}: {r['ms']:.2f} vs {ref:.2f} ({d:+.0f}%)")
                if m == 6048:
                    dev6048.append(abs(d))
        L.append(f"  {p}: " + ("; ".join(parts) if parts else "no rows"))
    anchor_ok = bool(dev6048) and min(dev6048) <= 10
    L.append(f"ANCHOR={'OK' if anchor_ok else 'OFF'} (closest pattern within 10% at M 6048; OFF = ratios not publishable)")
    # speed
    for p in PATTERNS:
        L.append("")
        L.append(f"== speed, layer {TIME_LAYER}, {p} routing: ms per call [TFLOPS] (x = {ANCHOR} ms / variant ms)")
        L.append("M".rjust(6) + " act max/mean " + "".join(n.rjust(26) for n in names))
        for m in MS:
            a = by.get((ANCHOR, TIME_LAYER, p, m))
            r0 = next((by[(n, TIME_LAYER, p, m)] for n in names if (n, TIME_LAYER, p, m) in by), None)
            st = f" {r0['active']:3d} {r0['max_rows']:4d}/{r0['mean_rows']:5.1f}" if r0 else " " * 15
            cells = []
            for n in names:
                r = by.get((n, TIME_LAYER, p, m))
                if r is None:
                    cells.append("n/a".rjust(26)); continue
                x = f" x{a['ms'] / r['ms']:.2f}" if a else ""
                flag = "" if gate.get(n) == "PASS" else "*"
                cells.append(f"{r['ms']:8.3f} [{r['tflops']:5.1f}]{x}{flag}".rjust(26))
            L.append(f"{m:6d}{st}" + "".join(cells))
    L.append("* = numerics gate FAIL, NOANCHOR or missing: no speed claim")
    L.append("")
    L.append("== selected configs")
    for n in names:
        r = next(r for r in rows if r["variant"] == n)
        L.append(f"  {n}: {r['config']}")
    L.append(f"VERDICT anchor={'OK' if anchor_ok else 'OFF'} " + " ".join(f"{n}={gate[n]}" for n in names))
    return "\n".join(L)


def check(ckpt):
    """CPU-only dry-run check inside the image: every API name the GPU path touches, and the checkpoint keys."""
    import torch
    from b12x._lib.intrinsics import swizzle_block_scale  # noqa: F401
    from b12x.moe import fused_moe as fm
    from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall  # noqa: F401
    fm.PackedSource(format=fm.PackedSourceFormat("modelopt_nvfp4"), w13_layout=fm.W13Layout.W31)
    fm.ActivationSpec(mode=fm.ActivationMode.A4, nonlinearity="silu", io_dtype=torch.bfloat16)
    fm.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I)
    fm.ExecutionCapacity(max_tokens=CAP, top_k=TOPK, warmup_token_counts=(CAP,), route_num_experts=0)
    fm.RoutingSpec(apply_router_weight_on_input=False)
    for n in ("plan_weights", "prepare_weights", "PackedWeights", "plan_execution", "bind", "run"):
        assert hasattr(fm, n), n
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
    import fi5863_moe_nvfp4_w4a4 as k
    assert callable(k.run_moe_w4a4) and callable(k.allocate_workspace)
    from safetensors import safe_open  # noqa: F401
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    for layer in NUM_LAYERS:
        for e in (0, E - 1):
            for p in ("gate_proj", "up_proj", "down_proj"):
                for t in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                    f = idx[f"model.language_model.layers.{layer}.mlp.experts.{e}.{p}.{t}"]
                    assert os.path.exists(os.path.join(ckpt, f)), f
    print("check ok")


def selftest():
    rows = []
    for n in VARIANTS:
        for l in NUM_LAYERS:
            for p in PATTERNS:
                for m in (MS if l == TIME_LAYER else (NUM_M,)):
                    ms = K81_MS.get(m, 15.0) * (0.8 if n == "fi5863" else 1.0)
                    err = 0.05 if n != "b12x-m64" else 0.2
                    rows.append(dict(variant=n, layer=l, pattern=p, m=m, ms=ms, tflops=m * FLOP_PER_TOKEN / ms / 1e9,
                                     err_ref=err if m == NUM_M else float("nan"), active=512, max_rows=m // 20,
                                     mean_rows=m * TOPK / E, config="cfg"))
    out = report("/nonexistent", rows)
    assert "ANCHOR=OK" in out, out
    assert "fi5863=PASS" in out and "b12x-m64=FAIL" in out, out
    assert "x1.25" in out, out  # fi5863 0.8x time -> 1.25x
    out2 = report("/nonexistent", [r for r in rows if r["variant"] != ANCHOR])
    assert "ANCHOR=OFF" in out2 and "fi5863=NOANCHOR" in out2, out2
    out3 = report("/nonexistent", [r for r in rows if not (r["variant"] == ANCHOR and r["layer"] == 47)])
    assert "fi5863=PASS" in out3 and "b12x-m64=FAIL" in out3 and f"{ANCHOR}=FAIL" in out3, out3
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    ck = a[a.index("--ckpt") + 1] if "--ckpt" in a else None
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["check"]:
        check(ck)
    elif a[:1] == ["report"]:
        print(report(a[1]))
    elif a[:1] == ["all"]:
        run_all(a[1], ck)
    elif a[:1] == ["run"]:
        try:
            run_variant(a[1], a[2], ck)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(__doc__)
