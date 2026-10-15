#!/usr/bin/env python3
"""k100 hc-a16-bf16 probe (1x #27): the hyperconnection (HC) projections of Qwen3.8-Flash-Next as served, against a
dequantize-to-BF16 + one BF16 GEMM path for the down projection. Built on k96's probe (backlog/k96/hc_probe.py).

Why not k90's lead: k96 timed the full served operation and found the HC down projection (K10240 -> N336) served A16
(BF16 activations, plan record a16 n128 k64 s1, 1.883 ms at M 6048), not the quantized regime k90 timed (2.70 ms).
The forced 128x128/BK64 quantized tile measured 2.228 ms there and failed numerics against A16. The A16 kernel only
has 16-row M tiles (dense_gemm weight-only check), so at prefill rows it dequantizes the 3.4 MB weight again for
every 16 rows. The candidate here dequantizes once (exact) and runs one BF16 GEMM with FP32 accumulation: b12x
exp/k100-hc-a16-bf16 (_a16_dequant.py) does the same inside blockscaled.mm.

Shapes (layer 5 attention HC of the served checkpoint, BF16 weights quantized online to MXFP8 g32 with vLLM's own
mxfp8_e4m3_quantize and packed with blockscaled.pack_weight, as HcMxfp8LinearMethod does at load):
  down  K10240 -> N336   input_mix_weight_down (320) + block_inject_weight (4) + 12 zero pad rows
  up    K320 -> N10240   input_mix_weight_up (served quantized; K padded to 384)
HC projections are replicated at TP=2 (disable_tp=True / ReplicatedLinear), so the shapes are the same on the 2x;
the plan records may differ, so the 2x row runs the same probe in the 2x image against its own seed (K100_SETUP=2x).
Inputs: the k97 capture of a cold 16K prompt (layer 5 HC sources) when --cap-dir has it, else fixed-seed synthetic
rows. Every variant reads the same tensors.
Variants, each in its own subprocess:
  ctl-seed    capacity 8192, autotune on over a copy of the image's plan seed (the served selection when no record is
              added); blockscaled.plan_regimes + blockscaled.mm with vLLM's query, provided workspace. In the arm image
              this is the patched dispatch.
  ctl         capacities 9072 / 12096 (prospective budgets): the served configs forced (down a16 n128 k64 s1, up
              quantized; the report checks them against ctl-seed), no autotune
  deq-pc      down: per call, dequantize the packed weight to BF16 and torch.mm(out_dtype=float32) -> BF16; what the
              b12x patch does (a copy of _a16_dequant.run, so it runs in the shipped image)
  deq-cached  down: the same with the BF16 weight made once outside the timing (+6.9 MB per HC instance, ~0.67 GB)
  deq-b16     down: per call, BF16 GEMM with BF16 output (cuBLAS default reduction flags), informational
  deq-up      up: dequantize + BF16 GEMM (served quantized: different activation precision), informational
  dec         decode/drafter M 1..40 (and 45..80 for 2x) at capacity 8192 with exact-M regimes as vLLM declares them;
              the job compares these output bytes between the shipped and the arm image (no decode change allowed)
  hcops       the HC norm/mix kernels as served (grouped_rmsnorm, scaled_silu, gate_mean, combine, combine_norm)
              through b12x.norm.hyperconnection with the seed: separate rows, no projection
Numerics before speed (k96 gate): per cell output sha1 vs control, and on 128 sampled rows cos / max abs rel vs
control and rel L2 error against an FP32 reference (BF16 input x dequantized MXFP8 weight). A variant passes when every
cell is bit-equal to control, or cos >= 0.9999, max abs rel <= 1e-2 and err_ref <= 1.10 x err_ref(control) + 1e-3.
Projection: seconds saved per cold 16K prompt = sum over the setup's 16K chunks of (control - variant) ms x 96 HC
down calls per step; 1x headline scaled to k81's dense-HC class through the anchor (as k96), 2x raw.
PREDICT=GO when deq-pc passes numerics, beats control at every probed M >= 1264 (the patch's MIN_ROWS) at capacity
8192, and the headline saving is >= 2% of the setup's cold 16K TTFT; else PREDICT=STOP with the reason.
Usage (in the image): hc_probe.py all /out --ckpt <snapshot> [--cap-dir <k97 capture>] | run <variant> /out ... |
compare /out | xcompare OLD NEW (torch) | report /out (stdlib) | selftest (stdlib) | check --ckpt <snapshot> (CPU)
Env: K100_SETUP=1x|2x (seed, chunks, TTFT baseline), K100_ROWS (sampled-row dir, kept out of the results),
K100_WARM (a host b12x/compile dir copied into the probe's cache once, so seeded variants load compiled kernels).
"""
import glob, hashlib, json, os, statistics, subprocess, sys, time, traceback

SETUP = os.environ.get("K100_SETUP", "1x")
SETUPS = {  # seed plan prefix in /opt/b12x-seed/preparation, 16K chunk list at budget 8192, cold 16K TTFT baseline
    "1x": dict(seed="33d9b32e", chunks=(6048, 6048, 3024, 1264), ttft=16384 / 2125,
               ttft_src="k90 arm dgx-01, 2125 tok/s", decode=40),
    "2x": dict(seed="225bf665", chunks=(5728, 5728, 2864, 2064), ttft=5.6,
               ttft_src="k76 TP=2 cold 16K, another benchmark", decode=80),
}[SETUP]
SHAPES = {"down": (10240, 336), "up": (320, 10240)}  # name: (K, N)
MIN_ROWS = 1264  # b12x exp/k100-hc-a16-bf16 _a16_dequant.MIN_ROWS
CAPS = {8192: (256, 512, 1024, 1264, 2048, 2064, 2864, 3024, 4536, 5728, 6048, 8192), 9072: (9072,), 12096: (12096,)}
DEC = tuple(m for m in (1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 25, 32, 40, 45, 60, 80) if m <= SETUPS["decode"])
EXACT = tuple(sorted({*range(1, 9), *range(5, SETUPS["decode"] + 1, 5)}))  # draft rows 1..8 and 5 x reqs (MTP 4)
HCOPS = ("grouped_rmsnorm", "scaled_silu", "gate_mean", "combine", "combine_norm")
HCOPS_MS = (1, 5, 40, 1264, 3024, 6048, 8192)
CALLS = 96  # HC down calls per step (48 layers x 2 HC mixers)
K81_US_PER_TOKEN = 48.7  # k81 dense-HC class, 1x
LAYER, NROWS = 5, 128
ROWS = os.environ.get("K100_ROWS", "/tmp/k100rows")
CACHE = "/tmp/cache-k100"  # one XDG cache for every variant subprocess (seeded once, warm kernels copied once)
SEED_DIR = "/opt/b12x-seed/preparation"
SERVED = {"down": "a16 n128 k64 s1", "up": "quantized"}  # k96 ctl-seed (1x); the report flags a different ctl-seed
VARIANTS = {
    "ctl-seed": dict(kind="mm", shapes=("down", "up"), caps=(8192,), seed=True),
    "deq-pc": dict(kind="deq", shapes=("down",), caps=tuple(CAPS), cached=False, f32=True),
    "dec": dict(kind="mm", shapes=("down", "up"), caps=(8192,), seed=True, exact=True),
    "ctl": dict(kind="mm", shapes=("down", "up"), caps=(9072, 12096), forced=True),
    "deq-cached": dict(kind="deq", shapes=("down",), caps=(8192,), cached=True, f32=True),
    "deq-b16": dict(kind="deq", shapes=("down",), caps=(8192,), cached=False, f32=False),
    "deq-up": dict(kind="deq", shapes=("up",), caps=(8192,), cached=False, f32=True),
    "hcops": dict(kind="hcops", shapes=(), caps=(), seed=True),
}
COLS = ("variant", "shape", "cap", "m", "ms", "tflops", "config", "lowering", "sha1", "err_ref", "src")
CMP = ("variant", "shape", "cap", "m", "same_bytes", "cos", "maxrel", "err_ref", "err_ctl")


def seed_plan():
    got = sorted(glob.glob(os.path.join(SEED_DIR, SETUPS["seed"] + "*.json")))
    if len(got) != 1:
        raise FileNotFoundError(f"{len(got)} seed plans {SETUPS['seed']}* in {SEED_DIR}")
    return got[0]


# ---------------------------------------------------------------- GPU side (torch imported lazily)
def load_weights(ckpt):
    import torch
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{LAYER}.attn_hyper_connection"
    get = lambda k: safe_open(os.path.join(ckpt, idx[k]), "pt", device="cpu").get_tensor(k)
    down = get(f"{pre}.input_mix_weight_down.weight")
    inject = get(f"{pre}.block_inject_weight.weight")
    pad = (-(down.shape[0] + inject.shape[0])) % 16
    w_down = torch.cat([down, inject, torch.zeros(pad, down.shape[1], dtype=down.dtype)])
    return {"down": w_down, "up": get(f"{pre}.input_mix_weight_up.weight"), "norm": get(f"{pre}.hc_norm.weight")}


def load_inputs(w, cap_dir):
    """-> {shape: BF16 [max M, K] on GPU}, source label; identical in every subprocess (k96)."""
    import torch
    mmax, out, src = max(max(v) for v in CAPS.values()), {}, {}
    for name, tag in (("down", "hcdown5"), ("up", "hcup5")):
        files = sorted(glob.glob(os.path.join(cap_dir or "/nonexistent", f"{tag}.*.pt")))
        if files:
            x = torch.cat([torch.load(f, weights_only=True)["x"] for f in files]).to(torch.bfloat16)
            n0 = x.shape[0]
            while x.shape[0] < mmax:
                x = torch.cat([x, x[: mmax - x.shape[0]]])
            src[name] = f"captured {n0} rows" + (f", repeated to {mmax}" if n0 < mmax else "")
            out[name] = x[:mmax].contiguous()
        else:
            g = torch.Generator().manual_seed(9600 + len(name))
            r = torch.randn(mmax, SHAPES[name][0], generator=g)
            x = r * (1 + w["norm"].float()) if name == "down" else torch.nn.functional.silu(r)
            out[name] = x.to(torch.bfloat16).contiguous()
            src[name] = "synthetic"
    return {k: v.cuda() for k, v in out.items()}, src


def pack(w_bf16):
    import torch
    import vllm.model_executor.layers.quantization.utils.mxfp8_utils as mu
    from b12x.gemm import blockscaled
    k = w_bf16.shape[1]
    wq, ws = mu.mxfp8_e4m3_quantize(w_bf16.cuda().contiguous())
    ws = ws[:, : k // 32].contiguous()  # as B12xMxfp8LinearKernel.process_weights_after_loading
    deq = wq.float() * torch.pow(2.0, ws.view(torch.uint8).float() - 127).repeat_interleave(32, dim=1)
    return blockscaled.pack_weight(wq, ws), deq


# copy of b12x exp/k100-hc-a16-bf16 _a16_dequant (so the candidate runs in the shipped image)
def unswizzle_scales(scale_bytes, n, k):
    import torch
    rows, cols = -(-n // 128) * 128, -(-(k // 32) // 4) * 4
    s = scale_bytes.view(torch.uint8).reshape(rows // 128, cols // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
    return s.reshape(rows, cols)[:n, : k // 32]


def dequantize(values, scale_bytes):
    import torch
    n, k = values.shape
    scale = torch.exp2(unswizzle_scales(scale_bytes, n, k).float() - 127).to(torch.bfloat16)
    return (values.view(torch.float8_e4m3fn).to(torch.bfloat16).view(n, k // 32, 32) * scale[:, :, None]).view(n, k)


def deq_call(packed, k, cached, f32):
    """-> f(a): the candidate's full op for one call, functional output as served (allocated per call)."""
    import torch
    values, scales = packed.weight.values, packed.weight.scale_mma
    flat = scales.permute(5, 2, 4, 0, 1, 3) if scales.ndim == 6 else scales  # F8_128x4 physical order (_a16.scale_storage)
    flat = flat.contiguous().view(torch.uint8).view(-1)
    n = values.shape[0]
    w0 = dequantize(values, flat)[:, :k].contiguous() if cached else None

    def run(a):
        w = w0 if cached else dequantize(values, flat)[:, :k]
        out = torch.empty(a.shape[0], n, device=a.device, dtype=torch.bfloat16)
        if f32:
            out.copy_(torch.mm(a, w.t(), out_dtype=torch.float32))
        else:
            torch.mm(a, w.t(), out=out)
        return out
    return run


def config_for(shape):
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig
    return BlockscaledConfig(mode="a16", tile_n=128, tile_k=64, split_k=1) if shape == "down" else BlockscaledConfig(mode="quantized")


def build(k, n, cap, packed, autotune, override, namespace=None, exact=()):
    """plan_regimes + PreparationSession as vLLM 5dad364d declares the linear (k96 build, with exact-M regimes)."""
    import torch
    import vllm.envs as envs
    from vllm.utils.b12x import get_b12x_dense_activation_mode
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared
    query = blockscaled.BlockscaledQuery(
        recipe="mxfp8", num_tokens=cap, in_features=k, padded_in_features=int(packed.padded_in_features),
        out_features=n, activation_mode=get_b12x_dense_activation_mode("mxfp8"), activation_scale_available=False,
        global_scale_kind="none", source_contiguous=True, source_aligned=True, workspace_form="provided",
        workspace_nbytes=envs.VLLM_B12X_BLOCKSCALED_WORKSPACE_MAX_BYTES, expected_m=None)
    plan = blockscaled.plan_regimes(query, exact_m=tuple(m for m in exact if m < cap), override=override)
    values, scales = packed.weight.values, packed.weight.scale_mma

    def factory(rows):  # as B12xBlockscaledLinear._call_factory, so autotune benchmarks the same call
        def prepare(state):
            src = torch.empty((rows, k), dtype=torch.bfloat16, device="cuda")
            ws = torch.empty(state.required_workspace, dtype=torch.uint8, device="cuda") if state.required_workspace else None
            return PreparedCall(run=lambda: state.run(src, values, scales, None, activation_scale=None, workspace=ws),
                                produce=lambda: src.fill_(0.125), owners=(values, scales), capture_safe=False)
        return prepare
    calls = {rows: factory(rows) for rows in plan.token_counts}
    t0 = time.time()
    dev = torch.device("cuda", torch.cuda.current_device())  # require_prepared compares the device index
    s = PreparationSession(device=dev, autotune=autotune, compile_workers=2, namespace=namespace)
    try:
        s.prepare((plan.request(name=f"k100.hc.{n}x{k}.c{cap}", prepare_calls=calls, benchmark_calls=calls),))
        state = require_prepared(plan, "gemm.blockscaled_precision", dev)
    except BaseException:
        s.close()
        raise
    ws = torch.empty(max(state.required_workspace, 1), dtype=torch.uint8, device="cuda")
    return plan, state, ws, time.time() - t0, s


def describe(st):
    c = st.config
    cfg = c.mode if c.mode == "quantized" else f"a16 n{c.tile_n} k{c.tile_k} s{c.split_k}"
    low = "-"
    if getattr(st, "dense", None) is not None:
        p = st.dense.lowering
        low = f"{tuple(p.mma_tiler_mn)}/BK{p.tile_k}/u{int(p.policy.large_m_unroll)}"
    if getattr(st, "a16_dequant", False):
        low = "bf16-deq>=1024"
    return cfg, low


def timed(fn, iters=10, reps=10):
    """ms per call: median over iters samples of reps back-to-back calls between two events, so host dispatch overlaps
    GPU work as in a serving step (a sync per call would expose b12x's Python dispatch but hide torch's)."""
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record(); b.synchronize()
        t.append(a.elapsed_time(b) / reps)
    return statistics.median(t)


def seed_cache():
    """The image's plan seed into this process's own cache, as the recipe command does -> (count records, namespace)."""
    import shutil
    pdir = os.path.join(os.environ.get("XDG_CACHE_HOME", "/tmp/cache"), "b12x", "compile", "preparation")
    os.makedirs(pdir, exist_ok=True)
    for fn in glob.glob(os.path.join(SEED_DIR, "*.json")):  # only when absent, as the recipe command
        if not os.path.exists(os.path.join(pdir, os.path.basename(fn))):
            shutil.copy(fn, pdir)
    nrec = lambda: sum(len(json.load(open(p)).get("records", {})) for p in glob.glob(os.path.join(pdir, "*.json")))
    return nrec, json.load(open(seed_plan()))["identity"]["namespace"]


def run_hcops(out_dir, w):
    """HC norm/mix kernels as GatedResidual calls them (hyperconnection.py at vLLM 5dad364d), per op and M."""
    import torch
    from vllm.utils.b12x import get_b12x_hyperconnection
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared
    api = get_b12x_hyperconnection()
    nrec, ns = seed_cache()
    n0 = nrec()
    streams, hidden, lowrank, eps = 4, 2560, 320, 1e-6
    width, cap = streams * hidden, 8192
    dev = torch.device("cuda", torch.cuda.current_device())
    f = dict(device=dev, dtype=torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(100)
    rnd = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
    normalized, bottleneck, block_input = torch.empty(cap, width, **f), torch.empty(cap, lowrank, **f), torch.empty(cap, hidden, **f)
    weight = w["norm"].to(**f)
    hs, gates, proj, blk, inj = rnd(cap, width), rnd(cap, width), rnd(cap, lowrank), rnd(cap, hidden), rnd(cap, streams)
    caps = api.Caps(device=dev, max_tokens=cap, hidden_size=hidden, streams=streams, lowrank=lowrank, dtype=torch.bfloat16)
    plans = {op: api.plan(caps, invocation={"operation": op, "eps": eps}) for op in HCOPS}

    def body(op, plan, m, binding):
        return {"grouped_rmsnorm": lambda: api.run_grouped_rmsnorm(hs[:m], weight, eps=eps, binding=binding),
                "scaled_silu": lambda: api.run_scaled_silu(proj[:m], binding=binding),
                "gate_mean": lambda: api.run_gate_mean(hs[:m], gates[:m], binding=binding),
                "combine": lambda: api.run_combine(hs[:m], blk[:m], inj[:m], plan=plan),
                "combine_norm": lambda: api.run_combine_norm(hs[:m], blk[:m], inj[:m], weight, eps=eps, plan=plan)}[op]

    def prep(op):
        def prepare(state):
            from b12x.norm.hyperconnection import _impl
            outs = (torch.empty(cap, width, **f), torch.empty(cap, lowrank, **f), torch.empty(cap, hidden, **f))
            run = {"grouped_rmsnorm": lambda: _impl.run_grouped_rmsnorm_impl(hs, weight, eps=eps, plan=state, out=outs[0]),
                   "scaled_silu": lambda: _impl.run_scaled_silu_impl(proj, plan=state, out=outs[1]),
                   "gate_mean": lambda: _impl.run_gate_mean_impl(hs, gates, plan=state, out=outs[2]),
                   "combine": lambda: _impl.run_combine_impl(hs, blk, inj, plan=state),
                   "combine_norm": lambda: _impl.run_combine_norm_impl(hs, blk, inj, weight, eps=eps, plan=state)}[op]
            return PreparedCall(run=run, owners=outs)
        return prepare
    rows = 0
    s = PreparationSession(device=dev, autotune=True, compile_workers=2, namespace=ns)
    try:
        s.prepare(tuple(plans[op].request(name=f"k100.hcops.{op}.m{cap}", prepare_call=prep(op), benchmark_call=prep(op)) for op in HCOPS))
        tag = " [seed hit]" if nrec() == n0 else f" [tuned +{nrec() - n0} records]"
        with open(os.path.join(out_dir, "hcops.tsv"), "w") as fh:
            for op in HCOPS:
                for m in HCOPS_MS:
                    binding = api.bind(plans[op], normalized=normalized, bottleneck=bottleneck, block_input=block_input, tokens=m)
                    try:
                        ms = timed(body(op, plans[op], m, binding))
                    except Exception:
                        fh.write(f"# hcops {op} m {m} FAILED: " + traceback.format_exc().replace("\n", " | ") + "\n"); continue
                    fh.write("\t".join(map(str, ("hcops", op, cap, m, f"{ms:.4f}", "-", "seed" + tag, "-", "-", "-", "-"))) + "\n")
                    fh.flush(); rows += 1
    finally:
        s.close()
    return rows


def run_variant(name, out_dir, ckpt, cap_dir):
    import torch
    from b12x.gemm import blockscaled
    v = VARIANTS[name]
    w = load_weights(ckpt)
    if v["kind"] == "hcops":
        return run_hcops(out_dir, w)
    nrec, ns = seed_cache() if v.get("seed") else (lambda: 0, None)
    x, src = load_inputs(w, cap_dir)
    os.makedirs(ROWS, exist_ok=True)
    nrows = 0
    with open(os.path.join(out_dir, f"{name}.tsv"), "w") as f:
        for shape in v["shapes"]:
            k, n = SHAPES[shape]
            packed, deq = pack(w[shape])
            for cap in v["caps"]:
                ms_list = DEC if v.get("exact") else CAPS[cap]
                sess = None
                try:
                    if v["kind"] == "deq":
                        call, cfg, low = deq_call(packed, k, v["cached"], v["f32"]), name, "torch.mm"
                    else:
                        n0 = nrec()
                        plan, state, ws, prep_s, sess = build(k, n, cap, packed, bool(v.get("seed")),
                                                              config_for(shape) if v.get("forced") else None, ns,
                                                              EXACT if v.get("exact") else ())
                        hit = "" if not v.get("seed") else " [seed hit]" if nrec() == n0 else f" [tuned +{nrec() - n0} records]"
                        print(f"{name} {shape} cap {cap}: prepared in {prep_s:.0f} s{hit}", flush=True)
                        call = lambda a: blockscaled.mm(a, packed, plan=plan, workspace=ws)
                except Exception:
                    f.write(f"# {name} {shape} cap {cap} FAILED: " + traceback.format_exc().replace("\n", " | ") + "\n"); f.flush()
                    continue
                for m in ms_list:
                    a = x[shape][:m]
                    if v["kind"] == "mm":
                        cfg, low = describe(state.resolve(a))
                        cfg += hit
                        if v.get("exact") and m in state.exact:
                            low = f"exact {low}"
                    r = call(a)
                    torch.cuda.synchronize()
                    sha = hashlib.sha1(r.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
                    idx = torch.linspace(0, m - 1, min(NROWS, m)).round().long().cuda()
                    rows = r[idx].float()
                    ref = a[idx].float() @ deq[:, :k].T
                    err = float((rows - ref).norm() / ref.norm().clamp_min(1e-30))
                    torch.save(rows.cpu(), os.path.join(ROWS, f"{name}_{shape}_{cap}_{m}.pt"))
                    ms = timed(lambda: call(a))
                    row = (name, shape, cap, m, f"{ms:.4f}", f"{2 * m * n * k / ms / 1e9:.1f}", cfg, low, sha, f"{err:.6f}", src[shape])
                    f.write("\t".join(map(str, row)) + "\n"); f.flush()
                    print("\t".join(map(str, row)), flush=True)
                    nrows += 1
                if sess is not None:
                    sess.close()
                    del plan, state, ws, sess
                torch.cuda.empty_cache()
    return nrows


def ctl_name(cap):
    return "ctl-seed" if cap == 8192 else "ctl"


def _cmp_rows(a, b):
    import torch
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
    return f"{cos:.7f}", f"{rel:.3e}"


def compare(out_dir):
    """Every non-control GEMM cell against its control's sampled rows -> compare.tsv."""
    import torch
    rows = read_rows(out_dir)
    ctl = {(r["shape"], r["cap"], r["m"]): r for r in rows if r["variant"] == ctl_name(r["cap"])}
    with open(os.path.join(out_dir, "compare.tsv"), "w") as f:
        for r in rows:
            c = ctl.get((r["shape"], r["cap"], r["m"]))
            if r["variant"] in ("ctl", "ctl-seed", "dec", "hcops") or c is None:
                continue
            load = lambda v: torch.load(os.path.join(ROWS, f"{v}_{r['shape']}_{r['cap']}_{r['m']}.pt"), weights_only=True)
            cos, rel = _cmp_rows(load(r["variant"]), load(c["variant"]))
            f.write("\t".join(map(str, (r["variant"], r["shape"], r["cap"], r["m"], int(r["sha1"] == c["sha1"]), cos, rel,
                                        r["err_ref"], c["err_ref"]))) + "\n")


def xcompare(old, new, old_rows, new_rows):
    """Shipped image (old) vs arm image (new), every shipped ctl-seed and dec cell (a cell missing in the arm run
    fails). Below MIN_ROWS and every dec cell must be byte-identical (the arm run preloads the shipped run's plan
    records); at MIN_ROWS and above the arm's down ctl-seed must pass the k96 gate against the shipped ctl-seed and be
    faster; the match with the shipped deq-pc bytes is reported. Prints the table and XCHECK=PASS|FAIL."""
    import torch
    o = {(r["variant"], r["shape"], r["cap"], r["m"]): r for r in read_rows(old)}
    n = {(r["variant"], r["shape"], r["cap"], r["m"]): r for r in read_rows(new)}
    ok, L = True, ["variant\tshape\tm\told_ms\tnew_ms\tx\tsame_bytes\tcos\tmaxrel\terr_old\terr_new\tsame_as_old_deq-pc\tnew_lowering"]
    keys = [k for k in o if k[0] in ("ctl-seed", "dec")]
    if not keys:
        print("no shipped cells"); print("XCHECK=FAIL"); return
    slower = []
    for k in sorted(keys, key=lambda t: (t[0], t[1], t[3])):
        a, b = o[k], n.get(k)
        if b is None:
            L.append(f"{k[0]}\t{k[1]}\t{k[3]}\tmissing in the arm image"); ok = False; continue
        if k[0] == "ctl-seed" and k[1] == "down" and k[3] >= MIN_ROWS and b["ms"] >= a["ms"]:
            slower.append(k[3])
        same = a["sha1"] == b["sha1"]
        d = o.get(("deq-pc", k[1], k[2], k[3]))
        cos, rel = _cmp_rows(torch.load(os.path.join(new_rows, f"{k[0]}_{k[1]}_{k[2]}_{k[3]}.pt"), weights_only=True),
                             torch.load(os.path.join(old_rows, f"{k[0]}_{k[1]}_{k[2]}_{k[3]}.pt"), weights_only=True))
        if k[0] == "dec" or k[3] < MIN_ROWS or k[1] == "up":
            ok &= same
        else:
            ok &= gate_cell(dict(same_bytes="1" if same else "0", cos=cos, maxrel=rel, err_ref=b["err_ref"], err_ctl=a["err_ref"]))
        L.append("\t".join(map(str, (k[0], k[1], k[3], f"{a['ms']:.4f}", f"{b['ms']:.4f}", f"{a['ms'] / b['ms']:.2f}",
                                     "yes" if same else "no", cos, rel, a["err_ref"], b["err_ref"],
                                     "-" if d is None else "yes" if d["sha1"] == b["sha1"] else "no", b["lowering"]))))
    if slower:
        L.append(f"patched dispatch not faster than shipped at M {slower} (>= MIN_ROWS {MIN_ROWS})"); ok = False
    print("\n".join(L))
    print(f"XCHECK={'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------- driver and report (stdlib only)
def run_all(out_dir, ckpt, cap_dir, only=None, budget_s=int(os.environ.get("K100_BUDGET", "3600"))):
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    import shutil
    warm, recs = os.environ.get("K100_WARM"), os.environ.get("K100_RECORDS")
    pdir = os.path.join(CACHE, "b12x", "compile", "preparation")
    if warm and os.path.isdir(warm):  # the host runtime cache's compiled b12x kernels: no cold compiles in the probe
        shutil.copytree(warm, os.path.join(CACHE, "b12x", "compile"), dirs_exist_ok=True)
        print(f"warm compile cache copied from {warm} in {time.time() - start:.0f} s", flush=True)
    if recs and glob.glob(os.path.join(recs, "*.json")):  # an earlier run's plan records: same selections (P1 vs P0)
        os.makedirs(pdir, exist_ok=True)
        for fn in glob.glob(os.path.join(recs, "*.json")):
            shutil.copy(fn, pdir)
        print(f"plan records preloaded from {recs}", flush=True)
    start = time.time()  # the budget counts variant time only
    for name in (only or VARIANTS):
        left = budget_s - (time.time() - start)
        if left < 120:
            print(f"{name}: skipped, budget ({budget_s} s) spent", flush=True)
            continue
        t0 = time.time()
        with open(os.path.join(out_dir, f"{name}.log"), "w") as log:
            env = dict(os.environ, XDG_CACHE_HOME=CACHE)
            if VARIANTS[name].get("seed"):  # the image sets B12X_AUTOTUNE=0, which makes an autotune session warmup-only
                env["B12X_AUTOTUNE"] = "1"
            try:
                rc = subprocess.run([sys.executable, os.path.abspath(__file__), "run", name, out_dir, "--ckpt", ckpt]
                                    + (["--cap-dir", cap_dir] if cap_dir else []), env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=min(int(os.environ.get("K100_VARIANT_TIMEOUT", "900")), left)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        print(f"{name}: rc={rc} {time.time() - t0:.0f} s", flush=True)
    if recs and not glob.glob(os.path.join(recs, "*.json")) and os.path.isdir(pdir):
        for fn in glob.glob(os.path.join(pdir, "*.json")):
            shutil.copy(fn, recs)
    try:
        compare(out_dir)
    except Exception:
        traceback.print_exc()
    n = len(read_rows(out_dir))
    print(f"timed cells: {n}", flush=True)
    return 0 if n else 1


def read_rows(out_dir):
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".tsv") or fn == "compare.tsv":
            continue
        for line in open(os.path.join(out_dir, fn)):
            v = line.rstrip("\n").split("\t")
            if len(v) == len(COLS) and not line.startswith("#"):
                r = dict(zip(COLS, v))
                r["cap"], r["m"], r["ms"] = int(r["cap"]), int(r["m"]), float(r["ms"])
                r["err_ref"] = float(r["err_ref"]) if r["err_ref"] != "-" else None
                rows.append(r)
    return rows


def read_cmp(out_dir):
    p, out = os.path.join(out_dir, "compare.tsv"), {}
    if os.path.exists(p):
        for line in open(p):
            v = line.rstrip("\n").split("\t")
            if len(v) == len(CMP):
                r = dict(zip(CMP, v))
                out[(r["variant"], r["shape"], int(r["cap"]), int(r["m"]))] = r
    return out


def gate_cell(c):
    if c is None:
        return False
    if c["same_bytes"] == "1":
        return True
    return float(c["cos"]) >= 0.9999 and float(c["maxrel"]) <= 1e-2 and float(c["err_ref"]) <= 1.10 * float(c["err_ctl"]) + 1e-3


def report(out_dir, rows=None, cmp=None):
    rows = read_rows(out_dir) if rows is None else rows
    cmp = read_cmp(out_dir) if cmp is None else cmp
    by = {(r["variant"], r["shape"], r["cap"], r["m"]): r for r in rows}
    names = [v for v in VARIANTS if any(k[0] == v for k in by)]
    L, gate = [], {}
    nocl = {}
    for v in names:
        cells = [k for k in by if k[0] == v and v not in ("ctl", "ctl-seed", "dec", "hcops")]
        judged = [k for k in cells if (ctl_name(k[2]), k[1], k[2], k[3]) in by]
        nocl[v] = sorted({f"c{k[2]}" for k in cells if k not in judged})
        if not cells:
            gate[v] = "CTL"
        elif any(k[2] == 8192 and k not in judged for k in cells) or not judged:
            gate[v] = "FAIL"  # the served capacity must be judged in full
        else:
            gate[v] = "PASS" if all(gate_cell(cmp.get(k)) for k in judged) else "FAIL"
    srcs = sorted({r["src"] for r in rows if r["src"] != "-"})
    L.append(f"setup {SETUP}: seed {SETUPS['seed']}, 16K chunks {list(SETUPS['chunks'])}, cold 16K TTFT {SETUPS['ttft']:.2f} s ({SETUPS['ttft_src']})")
    L.append(f"inputs: {'; '.join(srcs) if srcs else 'none'}; weights: layer {LAYER} attention HC, online MXFP8 g32")
    seeds = [r for r in rows if r["variant"] == "ctl-seed"]
    served = "N/A" if not seeds else "hit" if all(r["config"].endswith("[seed hit]") for r in seeds) else "miss"
    for sh in SHAPES:
        r = by.get(("ctl-seed", sh, 8192, 6048))
        if r and not r["config"].startswith(SERVED[sh]):
            L.append(f"note: ctl-seed {sh} config is '{r['config']}', not k96's '{SERVED[sh]}': ctl at 9072/12096 forces k96's")
    for shape in SHAPES:
        k, n = SHAPES[shape]
        vs = [v for v in names if any(kk[0] == v and kk[1] == shape for kk in by) and v not in ("dec", "hcops")]
        if not vs:
            continue
        L.append("")
        L.append(f"== {shape} K{k} N{n}: full-op ms per call, x = control / variant; numerics vs control: '=' bit-equal, "
                 f"'~' within gate, '!' fails. Control: ctl-seed at capacity 8192, ctl (forced served config) above")
        L.append("cap      M " + "".join(v.rjust(18) for v in vs) + "   control plan")
        for cap, ms in CAPS.items():
            for m in ms:
                c0 = by.get((ctl_name(cap), shape, cap, m))
                cells = []
                for v in vs:
                    r = by.get((v, shape, cap, m))
                    if r is None:
                        cells.append("n/a".rjust(18)); continue
                    ctl = v == ctl_name(cap)
                    mark = "" if ctl else ("=" if (cmp.get((v, shape, cap, m)) or {}).get("same_bytes") == "1"
                                           else "~" if gate_cell(cmp.get((v, shape, cap, m))) else "!")
                    x = f" x{c0['ms'] / r['ms']:.2f}" if c0 and not ctl else ""
                    cells.append(f"{r['ms']:7.3f}{x}{mark}".rjust(18))
                if any(c.strip() != "n/a" for c in cells):
                    L.append(f"{cap:5d} {m:6d} " + "".join(cells) + (f"   {c0['config']} {c0['lowering']}" if c0 else ""))
    dec = [r for r in rows if r["variant"] == "dec"]
    if dec:
        L.append("")
        L.append(f"== decode/drafter rows (capacity 8192, exact-M regimes {list(EXACT)}): ms per call, plan; the job "
                 "compares their bytes between the shipped and the arm image")
        for sh in SHAPES:
            L.append(f"  {sh}: " + "  ".join(f"M{r['m']} {r['ms']:.4f} ({r['config'].split(' [')[0]}{', exact' if r['lowering'].startswith('exact') else ''})"
                                           for r in dec if r["shape"] == sh))
    ops = [r for r in rows if r["variant"] == "hcops"]
    if ops:
        L.append("")
        L.append(f"== HC norm/mix kernels as served (b12x.norm.hyperconnection, capacity 8192, {ops[0]['config']}): ms per call")
        L.append("op               " + "".join(f"{m:>9}" for m in HCOPS_MS) + "   us/token at 6048")
        for op in HCOPS:
            cell = {r["m"]: r["ms"] for r in ops if r["shape"] == op}
            L.append(f"{op:<17}" + "".join(f"{cell[m]:9.4f}" if m in cell else "      n/a" for m in HCOPS_MS)
                     + (f"   {cell[6048] * 1000 / 6048:.2f}" if 6048 in cell else ""))
    # anchor (1x only): control down+up us/token over the chunk list vs k81's dense-HC class
    chunks = SETUPS["chunks"]
    have = all(("ctl-seed", sh, 8192, c) in by for c in chunks for sh in SHAPES)
    us = sum(by[("ctl-seed", sh, 8192, c)]["ms"] for c in chunks for sh in SHAPES) * CALLS * 1000 / sum(chunks) if have else None
    anchor = ("OK" if 0.85 <= us / K81_US_PER_TOKEN <= 1.15 else "OFF") if us and SETUP == "1x" else "N/A"
    scale = K81_US_PER_TOKEN / us if anchor == "OK" else None
    L.append("")
    if us:
        L.append(f"== anchor: control down+up = {us:.1f} us/token over {list(chunks)}"
                 + (f" vs k81 dense-HC class {K81_US_PER_TOKEN} (ratio {us / K81_US_PER_TOKEN:.2f}); ANCHOR={anchor} (OK = within 15%)"
                    if SETUP == "1x" else "; ANCHOR=N/A (k81 is a 1x profile)"))
    else:
        L.append("== anchor: control cells missing; ANCHOR=N/A")
    ttft = SETUPS["ttft"]
    L.append(f"== projection: seconds saved per cold 16K prompt = sum over {list(chunks)} of (control - variant) ms x {CALLS} "
             f"down calls per step" + (", k81-scaled headline (raw beside it)" if scale else ", raw headline"))
    proj = {}
    for v in [v for v in names if v.startswith("deq-") and v != "deq-up"]:
        cells = [(by.get(("ctl-seed", "down", 8192, c)), by.get((v, "down", 8192, c))) for c in chunks]
        if any(a is None or b is None for a, b in cells):
            L.append(f"  {v}: incomplete cells"); continue
        raw = sum(a["ms"] - b["ms"] for a, b in cells) * CALLS / 1000
        head = raw * scale if scale else raw
        proj[v] = head
        flag = "" if gate[v] == "PASS" else "  (numerics FAIL: no claim)"
        L.append(f"  {v}: {head:+.3f} s ({100 * head / ttft:+.1f}% of {ttft:.2f} s)" + (f", {raw:+.3f} s raw" if scale else "") + flag)
    L.append("")
    L.append("== numerics gate per variant: " + " ".join(f"{v}={gate[v]}" for v in names if gate[v] != "CTL")
             + "".join(f" ({v}: no control at {','.join(c)}, not judged there)" for v, c in nocl.items() if c))
    # decision for the serving A/B: the patched path is deq-pc
    why = []
    if gate.get("deq-pc") != "PASS":
        why.append(f"deq-pc numerics {gate.get('deq-pc', 'missing')}")
    slower = [m for m in CAPS[8192] if m >= MIN_ROWS and ("deq-pc", "down", 8192, m) in by and ("ctl-seed", "down", 8192, m) in by
              and by[("deq-pc", "down", 8192, m)]["ms"] >= by[("ctl-seed", "down", 8192, m)]["ms"]]
    missing = [m for m in CAPS[8192] if m >= MIN_ROWS and not (("deq-pc", "down", 8192, m) in by and ("ctl-seed", "down", 8192, m) in by)]
    if slower:
        why.append(f"deq-pc not faster at M {slower} (>= MIN_ROWS {MIN_ROWS})")
    if missing:
        why.append(f"cells missing at M {missing}")
    if served != "hit":
        why.append(f"control is not the served plan record (served={served})")
    head = proj.get("deq-pc")
    if head is None or head < 0.02 * ttft:
        why.append(f"projected {'n/a' if head is None else f'{head:+.3f} s'} < 2% of {ttft:.2f} s")
    pred = "GO" if not why else "STOP"
    L.append(f"PREDICT={pred}" + (f" deq-pc {head:+.3f} s ({100 * head / ttft:+.1f}%)" if head is not None else "")
             + ("" if not why else " (" + "; ".join(why) + ")"))
    L.append(f"VERDICT setup={SETUP} served={served} anchor={anchor} predict={pred} "
             + " ".join(f"{v}={gate[v]}" for v in names if gate[v] != "CTL")
             + (f" deq-pc={head:+.3f}s" if head is not None else ""))
    return "\n".join(L)


def check(ckpt):
    """CPU-only check inside the image: every API name the GPU path touches, the seed plan and the checkpoint keys."""
    import torch
    from b12x.gemm import blockscaled
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig  # noqa: F401
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared  # noqa: F401
    import vllm.envs as envs
    from vllm.utils.b12x import get_b12x_dense_activation_mode, get_b12x_hyperconnection
    import vllm.model_executor.layers.quantization.utils.mxfp8_utils as mu
    from b12x.norm.hyperconnection import _impl  # noqa: F401
    api = get_b12x_hyperconnection()
    for a in ("Caps", "plan", "bind", "run_grouped_rmsnorm", "run_scaled_silu", "run_gate_mean", "run_combine", "run_combine_norm"):
        assert hasattr(api, a), a
    for a in ("plan_regimes", "BlockscaledQuery", "pack_weight", "mm"):
        assert hasattr(blockscaled, a), a
    assert callable(mu.mxfp8_e4m3_quantize)
    assert "out_dtype" in str(torch.mm.__doc__) or hasattr(torch.ops.aten.mm, "dtype"), "torch.mm out_dtype missing"
    print("seed", seed_plan())
    try:
        from b12x.gemm.blockscaled import _a16_dequant as D
        print("arm image: _a16_dequant MIN_ROWS", D.MIN_ROWS)
        assert D.MIN_ROWS == MIN_ROWS, "probe MIN_ROWS differs from the image's"
    except ImportError:
        print("shipped image: no _a16_dequant")
    print("activation_mode", get_b12x_dense_activation_mode("mxfp8"), "workspace_nbytes", envs.VLLM_B12X_BLOCKSCALED_WORKSPACE_MAX_BYTES)
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{LAYER}.attn_hyper_connection"
    for t in ("input_mix_weight_down", "block_inject_weight", "input_mix_weight_up", "hc_norm"):
        assert os.path.exists(os.path.join(ckpt, idx[f"{pre}.{t}.weight"])), t
    print("check ok")


def selftest():
    rows, cmp = [], {}
    sh_ms = lambda v, sh, m: (m / 3000 if sh == "down" else m / 8000) * (0.5 if v == "deq-pc" and m >= 1024 else 1.0)
    for v in ("ctl-seed", "deq-pc", "ctl"):
        for sh in (("down", "up") if v != "deq-pc" else ("down",)):
            for cap in ((8192,) if v == "ctl-seed" else (9072, 12096) if v == "ctl" else tuple(CAPS)):
                for m in CAPS[cap]:
                    rows.append(dict(variant=v, shape=sh, cap=cap, m=m, ms=sh_ms(v, sh, m), tflops=0,
                                     config="a16 n128 k64 s1 [seed hit]" if sh == "down" else "quantized [seed hit]",
                                     lowering="-", sha1="a" if v != "deq-pc" else "b", err_ref=0.01, src="synthetic"))
                    if v == "deq-pc":
                        cmp[(v, sh, cap, m)] = dict(same_bytes="0", cos="0.999999", maxrel="2e-3", err_ref="0.0101", err_ctl="0.01")
    out = report("/nonexistent", rows, cmp)
    # 1x: ctl down+up over 6048,6048,3024,1264 = 16384 rows: (16384/3000 + 16384/8000) ms x 96 x 1000 / 16384 = 44.0 us/token
    assert "ANCHOR=OK" in out and "deq-pc=PASS" in out and "PREDICT=GO" in out, out
    # deq-pc halves down at every chunk (all >= 1024): raw saved = 16384/6000 ms x 96 / 1000 = 0.262 s; x 48.7/44.0 = 0.290
    assert "deq-pc: +0.290 s" in out and "+0.262 s raw" in out, out
    slow = [dict(r, ms=r["ms"] * 3) if r["variant"] == "deq-pc" and r["m"] == 1264 else r for r in rows]
    out2 = report("/nonexistent", slow, cmp)
    assert "PREDICT=STOP" in out2 and "not faster at M [1264]" in out2, out2
    bad = dict(cmp); bad[("deq-pc", "down", 8192, 6048)] = dict(same_bytes="0", cos="0.99", maxrel="5e-2", err_ref="0.05", err_ctl="0.01")
    out3 = report("/nonexistent", rows, bad)
    assert "deq-pc=FAIL" in out3 and "PREDICT=STOP" in out3 and "numerics FAIL: no claim" in out3, out3
    noctl = report("/nonexistent", [r for r in rows if r["variant"] != "ctl"], cmp)  # 2x row: --only without ctl
    assert "deq-pc=PASS" in noctl and "no control at c12096,c9072" in noctl and "PREDICT=GO" in noctl, noctl
    miss = report("/nonexistent", [r for r in rows if not (r["variant"] == "ctl-seed" and r["m"] == 8192 and r["shape"] == "down")], cmp)
    assert "deq-pc=FAIL" in miss and "PREDICT=STOP" in miss, miss
    tuned = report("/nonexistent", [dict(r, config="a16 n64 k64 s1 [tuned +1 records]") if r["variant"] == "ctl-seed" else r for r in rows], cmp)
    assert "served=miss" in tuned and "PREDICT=STOP" in tuned, tuned
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    ck = a[a.index("--ckpt") + 1] if "--ckpt" in a else None
    cd = a[a.index("--cap-dir") + 1] if "--cap-dir" in a else None
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["check"]:
        check(ck)
    elif a[:1] == ["report"]:
        print(report(a[1]))
    elif a[:1] == ["compare"]:
        compare(a[1])
    elif a[:1] == ["xcompare"]:
        xcompare(a[1], a[2], a[3], a[4])
    elif a[:1] == ["all"]:
        sys.exit(run_all(a[1], ck, cd, only))
    elif a[:1] == ["run"]:
        try:
            if not run_variant(a[1], a[2], ck, cd):
                sys.exit(f"{a[1]}: no timed cells")
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(__doc__)
