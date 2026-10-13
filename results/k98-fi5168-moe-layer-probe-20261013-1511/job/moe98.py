#!/usr/bin/env python3
"""k98 fi5168-moe-layer-probe (1x #27, tp-2): one routed-MoE layer of Qwen3.8-Flash-Next (H 2560, 512 experts, top-10,
checkpoint NVFP4 weights of layer 5, numerics also on layer 47), the shipped b12x MoE against FlashInfer at 5c4058bd9b
(PR #5168 merge, b12x_fused_moe functional API, NVFP4), in two geometries:
  tp1  I 640, the single-Spark layer
  tp2  I 320, the rank-0 shard of the 2x (gate/up rows 0..319, down columns 0..319; same experts, same routing)
at prefill M 1264 / 3024 / 6048 / 9072. Reuses the k91 harness (load, reference, timing, anchor).

Containers (the job runs them one after the other, outputs exchanged through files):
  b12x  the 1x v2.3.0 image as is. b12x b45fe867 differs from the 2x image's 21e0b201 only in blockscaled dense
        lowering, so its fused MoE is the shipped MoE of both setups. Variants b12x-m32 (k91 serving anchor: dynamic
        tile 32x128, capacity plan 8192, 9072 for M 9072) and b12x-auto (no tile override); per-expert FC1/FC2
        activation scales (shipped contract).
  fi    the same image, FlashInfer source at 5c4058bd on PYTHONPATH, nvidia-cutlass-dsl 4.7.1 from staged wheels.
        Variant fi5168: w13 = [up; gate], block scales swizzled 128x4 per expert and converted to the MMA layout,
        per-expert scales in FlashInfer's SM12x contract (fp4_common quantizers divide by the global scale, so the
        global scales are the checkpoint input_scale itself): input_global_scale = FC1 input_scale, w1_alpha =
        weight_scale_2 (FI folds input_global_scale into it), fc2_input_scale = FC2 input_scale and w2_alpha =
        weight_scale_2 x FC2 input_scale (FI has no FC2 fold). k98 run 1 passed 1 / input_scale (b12x's multiplier
        convention): activation block scales underflowed E4M3, output ~0, rel L2 ~1.0.
Routing: "real" = the k97 capture (layer 5 / 47 routed-MoE inputs with their top-10 ids and weights from a cold 16K
prompt) when --cap-dir has it; "uniform" and "skew" = k91's synthetic patterns (labelled). Activations of the
synthetic patterns: N(0, sigma) with sigma from the layer's calibrated FC1 amax.
Numerics first, at every M: rows 0..255 (the same inputs at every M) against an FP32 dequantized-weight reference
(BF16 activations), layers 5 and 47, every pattern. Accuracy gate: err <= 1.10 x err(b12x-m32) + 0.005 in every cell.
Contract check (fi5168, reported separately): rel L2 difference to the b12x-m32 output <= 0.5 x err(b12x-m32). A fi5168
error above 0.5 at M 6048 skips its timing (wrong scale convention, not a speed result).
Timing: eager, CUDA events, 256 MB L2 flush before every call, median of 15 after 3 warmups (k91). Preparation: wall
time of weight preparation + plan/compile (b12x) or of the first call (fi: JIT compile + weight views + workspace), and
the GPU memory each implementation keeps beyond the raw checkpoint tensors (prepared weights, workspaces), per layer.
Usage: moe98.py b12x /out --ckpt C --num /num [--cap-dir /cap] | fi /out ... | run <variant> /out ... |
report /out (stdlib) | selftest (stdlib) | check --ckpt C [--fi] (CPU, in the image)
"""
import glob, json, math, os, statistics, subprocess, sys, time, traceback

E, H, TOPK = 512, 2560, 10
PARTS = {"tp1": 640, "tp2": 320}
MS = (1264, 3024, 6048, 9072)
TIME_LAYER, NUM_LAYERS, NUM_M, NUM_ROWS = 5, (5, 47), 6048, 256
SYNTH = ("uniform", "skew")
K81_MS = {1264: 319.8 / 48, 3024: 431.0 / 48, 6048: (609.8 + 597.4) / 2 / 48}  # k81 routed MoE ms per layer per chunk
CHUNKS = {8192: (6048, 6048, 3024, 1264), 9072: (9072, 3024, 3024, 1264)}
LAYERS = 48
VARIANTS = {"b12x-m32": ("b12x", {"B12X_DYNAMIC_TILE_MN": "32x128"}), "b12x-auto": ("b12x", {}), "fi5168": ("fi", {})}
ANCHOR = "b12x-m32"
COLS = ("variant", "part", "layer", "pattern", "m", "ms", "tflops", "err_ref", "diff_b12x", "prep_s", "extra_mib",
        "backend", "src")


def flop_per_token(i):
    return TOPK * 3 * 2 * H * i


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

    out["router"] = get(f"model.language_model.layers.{layer}.mlp.gate.weight").cuda()  # [E, H], checks the capture
    for name, p in (("g", "gate_proj"), ("u", "up_proj"), ("d", "down_proj")):
        out[name + "w"] = torch.stack([get(f"{pre}.{e}.{p}.weight") for e in range(E)]).cuda()
        out[name + "s"] = torch.stack([get(f"{pre}.{e}.{p}.weight_scale") for e in range(E)]).cuda()
        out[name + "g"] = torch.stack([get(f"{pre}.{e}.{p}.weight_scale_2").float().reshape(()) for e in range(E)]).cuda()
        out[name + "i"] = torch.stack([get(f"{pre}.{e}.{p}.input_scale").float().reshape(()) for e in range(E)]).cuda()
    return out


def shard(lw, i):
    """TP rank-0 shard of the intermediate dimension: gate/up rows [0, i), down columns [0, i)."""
    if i == lw["gw"].shape[1]:
        return lw
    s = dict(lw)
    for n in ("g", "u"):
        s[n + "w"], s[n + "s"] = lw[n + "w"][:, :i].contiguous(), lw[n + "s"][:, :i].contiguous()
    s["dw"], s["ds"] = lw["dw"][:, :, : i // 2].contiguous(), lw["ds"][:, :, : i // 16].contiguous()
    return s


def make_inputs(lw, pattern, seed, cap_dir, layer):
    """-> x [R, H] BF16, ids [R, TOPK] int32, w [R, TOPK] fp32 on GPU, R >= max(MS); source label."""
    import torch
    rows = max(MS)
    if pattern == "real":
        files = sorted(glob.glob(os.path.join(cap_dir or "/nonexistent", f"moe{layer}.*.pt")))
        parts = [torch.load(f, weights_only=True) for f in files]
        x = torch.cat([p["x"] for p in parts]).to(torch.bfloat16)
        ids = torch.cat([p["ids"] for p in parts]).to(torch.int32)
        w = torch.cat([p["w"] for p in parts]).float()
        n = min(4096, x.shape[0])
        mine = torch.topk(x[:n].cuda().float() @ lw["router"].float().T, TOPK, dim=-1).indices.sort(-1).values
        theirs = ids[:n].cuda().long().sort(-1).values
        overlap = float((mine[:, :, None] == theirs[:, None, :]).any(-1).float().mean())
        src = (f"captured {x.shape[0]} rows, router overlap {overlap:.3f}, n_moe {parts[0]['n_moe']}"
               + ("" if overlap >= 0.9 else ", capture-mismatch"))
        while x.shape[0] < rows:
            x, ids, w = (torch.cat([t, t[: rows - t.shape[0]]]) for t in (x, ids, w))
            src += ", repeated"
        return x.cuda().contiguous(), ids.cuda().contiguous(), w.cuda().contiguous(), src
    g = torch.Generator().manual_seed(seed)
    sigma = float(lw["gi"].median()) * 6 * 448 / 5
    x = (torch.randn(rows, H, generator=g) * sigma).to(torch.bfloat16)
    if pattern == "uniform":
        ids = torch.topk(torch.randn(rows, E, generator=g), TOPK, dim=-1).indices
    else:
        p = (torch.arange(E, dtype=torch.float64) + 1) ** -0.3
        p = p[torch.randperm(E, generator=g)]
        ids = torch.multinomial(p.float().expand(rows, E), TOPK, replacement=False, generator=g)
    w = torch.softmax(torch.randn(rows, TOPK, generator=g), dim=-1)
    return x.cuda(), ids.to(torch.int32).cuda().contiguous(), w.float().cuda().contiguous(), "synthetic"


def route_stats(ids, m):
    import torch
    c = torch.bincount(ids[:m].reshape(-1).long(), minlength=E)
    return int((c > 0).sum()), int(c.max())


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


def rel(a, ref):
    return float((a.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))


class B12x:
    """k91's serving-contract b12x layer; one prepared weight set, one execution plan per capacity."""

    def __init__(self, lw):
        import torch
        from b12x._lib.intrinsics import swizzle_block_scale
        from b12x.moe import fused_moe
        self.fm, self.i = fused_moe, lw["dw"].shape[2] * 2
        w13 = torch.cat([lw["gw"], lw["uw"]], dim=1).contiguous()  # vLLM [gate; up] = W31, flipped by b12x at first use
        s13 = swizzle_block_scale(torch.cat([lw["gs"], lw["us"]], dim=1).contiguous())
        s2 = swizzle_block_scale(lw["ds"].contiguous())
        plan_w = fused_moe.plan_weights(
            source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("modelopt_nvfp4"),
                                          w13_layout=fused_moe.W13Layout.W31),
            activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A4, nonlinearity="silu",
                                                io_dtype=torch.bfloat16),
            geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=self.i))
        self.experts = fused_moe.prepare_weights(plan=plan_w, weights=fused_moe.PackedWeights(
            w13=w13, w2=lw["dw"].contiguous(), w13_block_scales=s13, w2_block_scales=s2,
            w13_global_scales=lw["gg"].contiguous(), w2_global_scales=lw["dg"].contiguous(),
            input_scale=(1.0 / lw["gi"]).float().contiguous(), intermediate_scale=(1.0 / lw["di"]).float().contiguous()))
        self.plans, self.scratch = {}, {}
        for cap in (8192, max(MS)):
            self._plan(cap)
        self.config = describe(self.plans[8192])

    def _plan(self, cap):
        import torch
        from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
        fm = self.fm
        plan = fm.plan_execution(
            experts=self.experts,
            capacity=fm.ExecutionCapacity(max_tokens=cap, top_k=TOPK, warmup_token_counts=(cap,), route_num_experts=0),
            routing=fm.RoutingSpec(apply_router_weight_on_input=False),
            invocation=FrozenMapping({"tuning_route_pattern": "cyclic_disjoint_topk"}))
        hid = torch.empty(cap, H, dtype=torch.bfloat16, device="cuda").normal_(0, 0.125)
        out = torch.empty_like(hid)
        rr = torch.arange(cap, dtype=torch.int32, device="cuda")[:, None] * TOPK + torch.arange(TOPK, dtype=torch.int32, device="cuda")
        ids, wts = rr.remainder(E).contiguous(), torch.full((cap, TOPK), 1.0 / TOPK, device="cuda")

        def factory(state):
            scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in state.scratch.scratch_specs())
            b = state.bind(scratch=scratch, a=hid, experts=self.experts, topk_weights=wts, topk_ids=ids, output=out,
                           input_scales_static=True)
            return PreparedCall(run=b.run, output=out, owners=(b, scratch))
        counts = getattr(plan, "token_counts", None)
        req = (plan.request(name=f"k98.moe.c{cap}", prepare_calls={c: factory for c in counts}) if counts is not None
               else plan.request(name=f"k98.moe.c{cap}", prepare_call=factory))
        with PreparationSession(device=torch.device("cuda"), autotune=False, compile_workers=2) as s:
            s.prepare((req,))
            s.freeze()
        nbytes = sum(spec.nbytes for spec in plan.scratch_specs())
        self.plans[cap], self.scratch[cap] = plan, torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    def backend(self, m):
        return f"cap{8192 if m <= 8192 else max(MS)}"

    def fn(self, x, ids, w, out):
        cap = 8192 if x.shape[0] <= 8192 else max(MS)
        # bind inside the call, as vLLM's B12xExperts.apply does per forward
        return lambda: self.fm.run(binding=self.fm.bind(self.plans[cap], scratch=self.scratch[cap], a=x, experts=self.experts,
                                                        topk_weights=w, topk_ids=ids, output=out, input_scales_static=True))


def describe(plan):
    for obj in (plan, *getattr(plan, "variants", {}).values()):
        sel = getattr(obj, "selection", None)
        if sel is not None:
            return f"{getattr(sel, 'source', '?')} {getattr(sel, 'config', sel)}".replace("\t", " ").replace("\n", " ")
    return "?"


def swizzle_128x4(sf):
    """[E, R, C] linear block scales -> flat per-expert 128x4 swizzle (rows padded to 128, columns to 4)."""
    import torch
    e, r, c = sf.shape
    rp, cp = -(-r // 128) * 128, -(-c // 4) * 4
    p = torch.zeros(e, rp, cp, dtype=torch.uint8, device=sf.device)
    p[:, :r, :c] = sf.view(torch.uint8)
    return p.view(e, rp // 128, 4, 32, cp // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().view(-1).view(sf.dtype)


class FI5168:
    def __init__(self, lw):
        from flashinfer import b12x_fused_moe
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        import torch
        self.i = i = lw["dw"].shape[2] * 2
        self.f = b12x_fused_moe
        self.w1 = torch.cat([lw["uw"], lw["gw"]], dim=1).contiguous()  # [up; gate]: values.split(I) -> up, gate
        self.s1 = convert_sf_to_mma_layout(swizzle_128x4(torch.cat([lw["us"], lw["gs"]], dim=1)), m=2 * i, k=H, num_groups=E)
        self.w2 = lw["dw"].contiguous()
        self.s2 = convert_sf_to_mma_layout(swizzle_128x4(lw["ds"]), m=H, k=i, num_groups=E)
        # FI SM12x contract (see the module docstring): global scales = checkpoint input_scale, FC2 fold by hand
        self.a1, self.a2 = lw["gg"].float().contiguous(), (lw["dg"] * lw["di"]).float().contiguous()
        self.ig, self.fc2 = lw["gi"].float().contiguous(), lw["di"].float().contiguous()
        self.config = "FlashInfer 5c4058bd b12x_fused_moe nvfp4, per-expert scales"
        self.calib = "FI scale contract"

    def calibrate(self, lw, x, ids, w):
        """Report only (selects nothing): M 256 rel L2 error, norm ratio and cosine vs the FP32 reference."""
        import torch
        ref = reference(lw, x, ids, w)
        out = torch.empty(NUM_ROWS, H, dtype=torch.bfloat16, device="cuda")
        self.fn(x[:NUM_ROWS], ids[:NUM_ROWS], w[:NUM_ROWS], out)()
        torch.cuda.synchronize()
        o = out.float()
        cos = float((o * ref).sum() / (o.norm() * ref.norm()).clamp_min(1e-30))
        self.calib = (f"FI scale contract (M256 err {rel(o, ref):.3f} norm ratio {float(o.norm() / ref.norm()):.3f} "
                      f"cos {cos:.3f})")

    def backend(self, m):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import select_sm120_moe_backend
        return select_sm120_moe_backend(num_tokens=m, num_topk=TOPK, quant_mode="nvfp4", num_experts=E,
                                        intermediate_size=self.i, hidden_size=H, activation="silu", capacity_tokens=m)

    def fn(self, x, ids, w, out):
        return lambda: self.f(x, self.w1, self.s1, self.w2, self.s2, ids, w, E, TOPK, w1_alpha=self.a1, w2_alpha=self.a2,
                              fc2_input_scale=self.fc2, input_global_scale=self.ig, output=out, quant_mode="nvfp4")


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


def run_variant(name, out_dir, ckpt, cap_dir, num_dir):
    import torch
    kind = VARIANTS[name][0]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    patterns = (("real",) if glob.glob(os.path.join(cap_dir or "/nonexistent", "moe5.*.pt")) else ()) + SYNTH
    with open(os.path.join(out_dir, f"{name}.tsv"), "w") as f:
        for layer in NUM_LAYERS:
            full = load_layer(ckpt, layer)
            for part, i in PARTS.items():
                lw = shard(full, i)
                torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); t0 = time.time()
                impl = (B12x if kind == "b12x" else FI5168)(lw)
                t_build = time.time() - t0
                for pi, pat in enumerate(patterns):
                    x, ids, w, src = make_inputs(lw, pat, 9800 + 10 * layer + pi, cap_dir, layer)
                    if kind == "fi":  # report-only M 256 check per pattern; its time counts as preparation once
                        tc = time.time(); impl.calibrate(lw, x, ids, w); t_build += (time.time() - tc) if pi == 0 else 0
                    if kind == "fi":
                        src = f"{src}; {impl.calib}"
                    ms_list = MS if layer == TIME_LAYER else (NUM_M,)
                    # numerics first (M 6048 rows 0..255)
                    out = torch.empty(NUM_M, H, dtype=torch.bfloat16, device="cuda")
                    fn = impl.fn(x[:NUM_M], ids[:NUM_M], w[:NUM_M], out)
                    t1 = time.time(); fn(); torch.cuda.synchronize()
                    if pi == 0:
                        prep_s = t_build + time.time() - t1  # build (+ fi calibrate) + first M 6048 call (JIT, views, workspace)
                    ref = reference(lw, x, ids, w)
                    tag = f"{part}_L{layer}_{pat}"
                    broken = kind == "fi" and rel(out[:NUM_ROWS], ref) > 0.5
                    for m in ms_list:
                        o = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
                        fm = impl.fn(x[:m], ids[:m], w[:m], o)
                        fm(); torch.cuda.synchronize()  # numerics at every M: rows 0..255 see the same inputs at every M
                        err, diff = rel(o[:NUM_ROWS], ref), float("nan")
                        if kind == "b12x":
                            torch.save(o[:NUM_ROWS].cpu(), os.path.join(num_dir, f"{name}_{tag}_m{m}.pt"))
                        else:
                            pa = os.path.join(num_dir, f"{ANCHOR}_{tag}_m{m}.pt")
                            if os.path.exists(pa):
                                diff = rel(o[:NUM_ROWS].cpu(), torch.load(pa, weights_only=True))
                        ms = float("nan") if (layer != TIME_LAYER or broken) else timed(fm, flush)
                        torch.cuda.synchronize()
                        live = sum(t.numel() * t.element_size() for t in (x, ids, w, out, o, ref))
                        extra = (torch.cuda.memory_allocated() - base - live) / 2**20  # kept beyond the raw tensors + inputs
                        row = (name, part, layer, pat, m, f"{ms:.4f}", f"{m * flop_per_token(i) / ms / 1e9:.1f}",
                               f"{err:.5f}", f"{diff:.5f}",
                               f"{prep_s:.1f}", f"{extra:.0f}", impl.backend(m), src)
                        f.write("\t".join(map(str, row)) + "\n"); f.flush()
                        print("\t".join(map(str, row)), flush=True)
                    del x, ids, w
                del impl, lw
                torch.cuda.empty_cache()
            del full
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- drivers and report (stdlib only)
def run_kind(kind, out_dir, ckpt, cap_dir, num_dir, budget_s):
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    for name, (k, env) in VARIANTS.items():
        if k != kind:
            continue
        left = budget_s - (time.time() - start)
        if left < 120:
            print(f"{name}: skipped, budget ({budget_s} s) spent", flush=True)
            continue
        t0 = time.time()
        args = [sys.executable, os.path.abspath(__file__), "run", name, out_dir, "--ckpt", ckpt, "--num", num_dir]
        with open(os.path.join(out_dir, f"{name}.log"), "w") as log:
            try:
                rc = subprocess.run(args + (["--cap-dir", cap_dir] if cap_dir else []), env={**os.environ, **env},
                                    stdout=log, stderr=subprocess.STDOUT, timeout=left).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        print(f"{name}: rc={rc} {time.time() - t0:.0f} s", flush=True)


def fnum(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def read_rows(out_dir):
    rows = []
    for name in VARIANTS:
        p = os.path.join(out_dir, f"{name}.tsv")
        if os.path.exists(p):
            for line in open(p):
                v = line.rstrip("\n").split("\t")
                if len(v) == len(COLS):
                    r = dict(zip(COLS, v))
                    r["layer"], r["m"] = int(r["layer"]), int(r["m"])
                    for k in ("ms", "tflops", "err_ref", "diff_b12x", "prep_s", "extra_mib"):
                        r[k] = fnum(r[k])
                    rows.append(r)
    return rows


def report(out_dir, rows=None):
    rows = read_rows(out_dir) if rows is None else rows
    by = {(r["variant"], r["part"], r["layer"], r["pattern"], r["m"]): r for r in rows}
    names = [n for n in VARIANTS if any(r["variant"] == n for r in rows)]
    pats = list(dict.fromkeys(r["pattern"] for r in rows))
    L, verdict = [], []
    srcs = sorted({f"{r['pattern']}: {r['src']}" for r in rows})
    L.append("routing/activations: " + ("; ".join(srcs) if srcs else "none"))
    gate, contract = {}, {}
    for part, i in PARTS.items():
        L.append("")
        L.append(f"== {part} (I {i}) numerics: rel L2 error vs FP32 reference, rows 0..{NUM_ROWS - 1} (same inputs at every M), "
                 f"at M {NUM_M} [fi5168: rel L2 difference to the {ANCHOR} output]")
        L.append(f"   accuracy gate: err <= 1.10 x {ANCHOR} + 0.005 in every cell and at every M; contract check (fi5168): "
                 f"difference to {ANCHOR} <= 0.5 x err({ANCHOR}) in every cell and at every M")
        L.append("variant".ljust(12) + "".join(f"L{l} {p}".rjust(16) for l in NUM_LAYERS for p in pats) + "  accuracy contract")
        for n in names:
            cells, ok, judged, cok, cjudged = [], True, 0, True, 0
            for l in NUM_LAYERS:
                for p in pats:
                    r = by.get((n, part, l, p, NUM_M))
                    if r is None or math.isnan(r["err_ref"]):
                        cells.append("n/a".rjust(16)); ok = False; continue
                    d = "" if math.isnan(r["diff_b12x"]) else f"[{r['diff_b12x']:.3f}]"
                    cells.append(f"{r['err_ref']:.4f}{d}".rjust(16))
            for (v, pt, l, p, m), r in by.items():
                a = by.get((ANCHOR, pt, l, p, m))
                if v != n or pt != part or math.isnan(r["err_ref"]) or a is None or math.isnan(a["err_ref"]):
                    continue
                judged += 1
                ok &= r["err_ref"] <= 1.10 * a["err_ref"] + 0.005
                if not math.isnan(r["diff_b12x"]):
                    cjudged += 1
                    cok &= r["diff_b12x"] <= 0.5 * a["err_ref"]
            gate[(n, part)] = ("PASS" if ok else "FAIL") if judged else "NOANCHOR"
            contract[(n, part)] = ("PASS" if cok else "FAIL") if cjudged else "N/A"
            L.append(n.ljust(12) + "".join(cells) + f"  {gate[(n, part)]:>8} {contract[(n, part)]:>8}")
        for p in pats:
            L.append(f"  layer {TIME_LAYER} {p}, error per M " + "  ".join(
                f"{n}: " + " ".join(f"{m}={by[(n, part, TIME_LAYER, p, m)]['err_ref']:.4f}" for m in MS if (n, part, TIME_LAYER, p, m) in by)
                for n in names))
    # anchor (tp1 only: k81 is a single-Spark profile)
    L.append("")
    L.append(f"== anchor: {ANCHOR} tp1 layer {TIME_LAYER} vs k81 serving (routed MoE ms per layer per chunk); tp2: N/A (no TP=2 per-layer profile)")
    anc = {}
    for p in pats:
        parts, d6 = [], None
        for m, ref in sorted(K81_MS.items()):
            r = by.get((ANCHOR, "tp1", TIME_LAYER, p, m))
            if r and not math.isnan(r["ms"]):
                d = 100 * (r["ms"] / ref - 1)
                parts.append(f"M {m}: {r['ms']:.2f} vs {ref:.2f} ({d:+.0f}%)")
                if m == 6048:
                    d6 = d
        anc[p] = d6
        L.append(f"  {p}: " + ("; ".join(parts) if parts else "no rows"))
    best = min((abs(v), p) for p, v in anc.items() if v is not None) if any(v is not None for v in anc.values()) else None
    anchor = "N/A" if best is None else ("OK" if best[0] <= 10 else "OFF")
    L.append(f"ANCHOR={anchor}" + (f" (closest: {best[1]} at M 6048, {anc[best[1]]:+.0f}%; k91 was -13/-14% with synthetic routing)" if best else ""))
    # speed
    for part, i in PARTS.items():
        for p in pats:
            L.append("")
            L.append(f"== {part} (I {i}) speed, layer {TIME_LAYER}, {p} routing: ms per call [TFLOPS] backend; "
                     f"x = best b12x / variant")
            L.append("M".rjust(6) + " act  max " + "".join(n.rjust(30) for n in names))
            for m in MS:
                bb = [by[(n, part, TIME_LAYER, p, m)]["ms"] for n in names if n.startswith("b12x")
                      and (n, part, TIME_LAYER, p, m) in by and not math.isnan(by[(n, part, TIME_LAYER, p, m)]["ms"])]
                ctl = min(bb) if bb else None
                cells = []
                for n in names:
                    r = by.get((n, part, TIME_LAYER, p, m))
                    if r is None or math.isnan(r["ms"]):
                        cells.append("n/a".rjust(30)); continue
                    x = f" x{ctl / r['ms']:.2f}" if ctl else ""
                    flag = "" if gate.get((n, part)) == "PASS" else "*"
                    cells.append(f"{r['ms']:7.3f} [{r['tflops']:5.1f}] {r['backend'][:7]}{x}{flag}".rjust(30))
                L.append(f"{m:6d}" + " " * 10 + "".join(cells))
    L.append("* = numerics gate FAIL, NOANCHOR or missing: no speed claim")
    # preparation and memory
    L.append("")
    L.append("== preparation (s, layer 5: b12x weights + plans; fi first call incl. JIT) and GPU memory kept beyond the raw "
             "checkpoint tensors (MiB per layer, x48 layers)")
    for part in PARTS:
        for n in names:
            r = by.get((n, part, TIME_LAYER, pats[0], max(MS))) if pats else None
            if r:
                L.append(f"  {part} {n}: prep {r['prep_s']:.1f} s; kept {r['extra_mib']:.0f} MiB per layer, "
                         f"{r['extra_mib'] * LAYERS / 1024:.1f} GiB x48")
    # projection and verdict
    L.append("")
    L.append("== projection (operator only): seconds per cold 16K prompt = sum over the budget's chunks of "
             "(best b12x - fi5168) ms x 48 layers; positive = fi5168 faster")
    for part in PARTS:
        for p in pats:
            for b, chunks in CHUNKS.items():
                s, ok = 0.0, True
                for c in chunks:
                    bb = [by[(n, part, TIME_LAYER, p, c)]["ms"] for n in names if n.startswith("b12x") and (n, part, TIME_LAYER, p, c) in by]
                    f = by.get(("fi5168", part, TIME_LAYER, p, c))
                    if not bb or f is None or math.isnan(f["ms"]) or any(math.isnan(v) for v in bb):
                        ok = False; break
                    s += (min(bb) - f["ms"]) * LAYERS / 1000
                if ok:
                    L.append(f"  {part} {p} budget {b} {list(chunks)}: {s:+.3f} s")
    for part in PARTS:
        g = gate.get(("fi5168", part), "missing")
        p = "real" if "real" in pats and not any("capture-mismatch" in r["src"] for r in rows if r["pattern"] == "real") else "uniform"
        r = by.get(("fi5168", part, TIME_LAYER, p, 6048))
        bb = [by[(n, part, TIME_LAYER, p, 6048)]["ms"] for n in names if n.startswith("b12x") and (n, part, TIME_LAYER, p, 6048) in by]
        x = f"x{min(bb) / r['ms']:.2f}@6048{p}" if r and bb and not math.isnan(r["ms"]) else "no-speed"
        verdict.append(f"fi5168_{part}={g} contract={contract.get(('fi5168', part), 'missing')} {x}")
    L.append(f"VERDICT anchor={anchor} " + " ".join(verdict))
    return "\n".join(L)


def check(ckpt, fi):
    """CPU-only check inside the image: every API name the GPU path touches, and the checkpoint keys."""
    import torch  # noqa: F401
    if fi:
        import flashinfer
        from flashinfer import b12x_fused_moe  # noqa: F401
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout  # noqa: F401
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import select_sm120_moe_backend  # noqa: F401
        import importlib.metadata as md
        print("flashinfer from", flashinfer.__file__, "cutlass-dsl", md.version("nvidia-cutlass-dsl"))
    else:
        from b12x._lib.intrinsics import swizzle_block_scale  # noqa: F401
        from b12x.moe import fused_moe as fm
        from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall  # noqa: F401
        for n in ("plan_weights", "prepare_weights", "PackedWeights", "plan_execution", "bind", "run"):
            assert hasattr(fm, n), n
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
        for part, i in PARTS.items():
            for l in NUM_LAYERS:
                for p in ("real",) + SYNTH:
                    for m in (MS if l == TIME_LAYER else (NUM_M,)):
                        ms = K81_MS.get(m, 15.0) * (0.75 if n == "fi5168" else 1.0) * (1.1 if n == "b12x-auto" else 1.0)
                        err = 0.2 if not (n == "fi5168" and part == "tp2") else 0.4
                        rows.append(dict(variant=n, part=part, layer=l, pattern=p, m=m, ms=ms, tflops=1.0,
                                         err_ref=err, diff_b12x=0.05 if n == "fi5168" else float("nan"),
                                         prep_s=3.0, extra_mib=100.0, backend="dynamic", src="x"))
    out = report("/nonexistent", rows)
    assert "ANCHOR=OK" in out, out
    assert "fi5168_tp1=PASS contract=PASS x1.33@6048real" in out and "fi5168_tp2=FAIL" in out, out
    assert "4.7 GiB x48" in out, out  # 100 MiB x 48
    for r in rows:
        if r["variant"] == "fi5168" and r["part"] == "tp1" and r["pattern"] == "skew" and r["m"] == 1264:
            r["diff_b12x"] = 0.15  # > 0.5 x 0.2 at one M only: contract fails, accuracy stays
    assert "fi5168_tp1=PASS contract=FAIL" in report("/nonexistent", rows)
    for r in rows:
        if r["variant"] == "fi5168" and r["part"] == "tp1" and r["pattern"] == "uniform" and r["m"] == 9072:
            r["err_ref"] = 0.5  # one bad M fails the accuracy gate
    assert "fi5168_tp1=FAIL" in report("/nonexistent", rows)
    for r in rows:
        if r["pattern"] == "real":
            r["src"] = "captured 16384 rows, router overlap 0.412, n_moe 49, capture-mismatch"
    assert "@6048uniform" in report("/nonexistent", rows)
    out2 =report("/nonexistent", [r for r in rows if r["pattern"] != "real" and r["variant"] != ANCHOR])
    assert "ANCHOR=N/A" in out2 and "fi5168_tp1=NOANCHOR contract=N/A" in out2, out2
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k: a[a.index(k) + 1] if k in a else None
    ck, cd, nd = opt("--ckpt"), opt("--cap-dir"), opt("--num") or "/num"
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["check"]:
        check(ck, "--fi" in a)
    elif a[:1] == ["report"]:
        print(report(a[1]))
    elif a[:1] == ["b12x"]:
        run_kind("b12x", a[1], ck, cd, nd, 1500)
    elif a[:1] == ["fi"]:
        run_kind("fi", a[1], ck, cd, nd, 2100)
    elif a[:1] == ["run"]:
        try:
            run_variant(a[1], a[2], ck, cd, nd)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(__doc__)
