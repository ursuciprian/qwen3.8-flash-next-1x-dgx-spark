#!/usr/bin/env python3
"""k96 hc-gemm-fair-probe (1x #27): the two hyperconnection projections of Qwen3.8-Flash-Next, timed as served.

Shapes (layer 5 attention HC of the 1x checkpoint, real BF16 weights quantized online to MXFP8 g32 with vLLM's own
mxfp8_e4m3_quantize and packed with blockscaled.pack_weight, as HcMxfp8LinearMethod does at load):
  down  K10240 -> N336   input_mix_weight_down (320) + block_inject_weight (4) + 12 zero pad rows (MergedColumnParallel)
  up    K320 -> N10240   input_mix_weight_up
Inputs: the k97 capture of a cold 16K prompt (layer 5 HC sources, real activations) when --cap-dir has it, else fixed-seed
synthetic rows (down: N(0,1) x (1 + hc_norm weight); up: silu(N(0,1))). Every variant reads the same tensors.
Every call goes through blockscaled.plan_regimes with vLLM's query (B12xBlockscaledLinear.ensure_plan: activation_mode
from VLLM_B12X_DENSE_ACTIVATION_MODE, provided workspace, workspace_nbytes, expected_m None) and blockscaled.mm with a
provided workspace, so activation quantization, output allocation and workspace are part of every timing. exact_m is
left empty: those regimes only serve M <= 80, never the M probed here.
Capacity: the regime is declared at the budget a server at that chunk size would use: 8192 for M <= 8192, 9072 and
12096 for those budgets (chunk sizes of those budgets are timed at that capacity too).
Variants, each in its own subprocess (module patches stay local):
  ctl           autotune on (the serving selection: B12X_AUTOTUNE=1 measures every legal config of the capacity query)
  ctl-seed      capacity 8192, autotune on over a copy of the image's served plan seed (33d9b32e); when no new record is
                written the served selection was loaded and ctl-seed is the control at 8192 (else ctl, and the report
                compares ctl's config with the configs in the seed file)
  q-def         quantized MXFP8 activations, default lowering (the k90 "regime" column)
  q-<tile>      quantized, the large-row dense lowering forced for these shapes at every M (_large_m_shape admits them,
                LARGE_M_ROWS=2): 128x128/BK64/u0 (the tile-fix plan) and neighbours
  q-prop        quantized, 128x128/BK64/u0 at M >= 4096 only, the previous plan below (the tile-fix policy extended)
  a16-sweep     up only, capacity 8192: every legal A16 config (tile_n, tile_k, split_k); K320 has no quantized path
Numerics before speed: per cell the output sha1 vs ctl, and on 128 sampled rows cos / max abs rel vs ctl and rel L2 error
against an FP32 reference (BF16 input x dequantized MXFP8 weight). Gate per variant: every cell bit-equal to ctl, or
cos >= 0.9999, max abs rel <= 1e-2 and err_ref <= 1.10 x err_ref(ctl) + 1e-3.
Projection: seconds saved per cold 16K prompt = sum over the budget's chunks and both shapes of (ctl - variant) ms x 96
calls per step (48 layers x 2 HC mixers), chunk lists from k81/k88 (8192, 9072) and the k97 hypothesis (12096).
Anchor: ctl per-token cost at budget 8192 against k81's dense-HC class (48.7 us/token, mostly these two GEMMs).
Usage (1x v2.3.0 image): hc_probe.py all /out --ckpt <snapshot> [--cap-dir <k97 capture>] | run <variant> /out ... |
compare /out (torch) | report /out (stdlib) | selftest (stdlib) | check --ckpt <snapshot> (CPU, in the image)
"""
import glob, hashlib, json, os, statistics, subprocess, sys, time, traceback

SHAPES = {"down": (10240, 336), "up": (320, 10240)}  # name: (K, N)
MS = (1264, 3024, 4536, 6048, 8192, 9072, 12096)
CAPS = {8192: (1264, 3024, 4536, 6048, 8192), 9072: (1264, 3024, 9072), 12096: (1264, 3024, 12096)}
CHUNKS = {8192: (6048, 6048, 3024, 1264), 9072: (9072, 3024, 3024, 1264), 12096: (12096, 3024, 1264)}
CHUNK_SRC = {8192: "k81/k88 measured", 9072: "k88 measured", 12096: "hypothesis, k97 checks"}
CALLS = 96
K81_US_PER_TOKEN, K81_TTFT_16K = 48.7, 7.925
LAYER, NROWS = 5, 128
ROWS = os.environ.get("K96_ROWS", "/tmp/k96rows")  # sampled output rows for compare; container-local, not in results
MAIN = ((128, 128), 64, False)
TILES = {"q-128x128k64u0": MAIN, "q-128x128k128u0": ((128, 128), 128, False), "q-128x64k64u0": ((128, 64), 64, False),
         "q-64x128k64u0": ((64, 128), 64, False), "q-128x128k64u1": ((128, 128), 64, True)}
# name: (shapes, caps, autotune, override mode, tile patch, policy). Value order: a budget stop drops the last ones.
VARIANTS = {
    "ctl": (("down", "up"), tuple(CAPS), True, None, None, None),
    "ctl-seed": (("down", "up"), (8192,), True, None, None, None),  # autotune over the image's served plan seed
    "q-def": (("down",), tuple(CAPS), False, "quantized", None, None),
    "q-128x128k64u0": (("down",), tuple(CAPS), False, "quantized", MAIN, "all"),
    "q-prop": (("down",), tuple(CAPS), False, "quantized", MAIN, "split"),
    "a16-sweep": (("up",), (8192,), False, "a16", None, None),
    "q-128x128k128u0": (("down",), tuple(CAPS), False, "quantized", TILES["q-128x128k128u0"], "all"),
    "q-128x64k64u0": (("down",), tuple(CAPS), False, "quantized", TILES["q-128x64k64u0"], "all"),
    "q-64x128k64u0": (("down",), tuple(CAPS), False, "quantized", TILES["q-64x128k64u0"], "all"),
    "q-128x128k64u1": (("down",), tuple(CAPS), False, "quantized", TILES["q-128x128k64u1"], "all"),
}
SEED_DIR = "/opt/b12x-seed/preparation"  # the served b12x plan records in the 1x v2.3.0 image (recipe `command:`)
SEED_PLAN = "33d9b32e995b59bc4b0d3cdfb5fc2247f3c7b2151c09569ffd3dbaf85c4562e3.json"
A16 = tuple((n, k, s) for n in (64, 128) for k in (64, 128) for s in (1, 2, 4, 8))
COLS = ("variant", "shape", "cap", "m", "ms", "tflops", "config", "lowering", "sha1", "err_ref", "src")
CMP = ("variant", "shape", "cap", "m", "same_bytes", "cos", "maxrel", "err_ref", "err_ctl")


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
    """-> {shape: BF16 [max(MS), K] on GPU}, source label; identical in every subprocess."""
    import glob
    import torch
    mmax, out, src = max(MS), {}, {}
    for name, tag in (("down", "hcdown5"), ("up", "hcup5")):
        files = sorted(glob.glob(os.path.join(cap_dir or "/nonexistent", f"{tag}.*.pt")))
        if files:
            x = torch.cat([torch.load(f, weights_only=True)["x"] for f in files]).to(torch.bfloat16)
            src[name] = f"captured {x.shape[0]} rows"
            while x.shape[0] < mmax:
                x = torch.cat([x, x[: mmax - x.shape[0]]])
                src[name] = f"captured {len(files)} chunks, {src[name].split()[1]} rows, repeated to {mmax}"
            out[name] = x[:mmax].contiguous()
        else:
            g = torch.Generator().manual_seed(9600 + len(name))
            k = SHAPES[name][0]
            r = torch.randn(mmax, k, generator=g)
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


def patch(tile, policy):
    from b12x.gemm.blockscaled import _preparation as P
    if tile is None:
        return
    mn, bk, unroll = tile
    P._LARGE_M_DENSE = {"mma_tiler_mn": mn, "_tile_k_override": bk, "_large_m_unroll_override": unroll}
    P._large_m_shape = lambda q: q.recipe == "mxfp8" and q.expected_m is None
    if policy == "all":
        P.LARGE_M_ROWS = 2  # small regime of 1 row: every probed M takes the capacity regime with the forced tile


def build(k, n, cap, packed, autotune, override):
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
    plan = blockscaled.plan_regimes(query, exact_m=(), override=override)
    values, scales = packed.weight.values, packed.weight.scale_mma

    def factory(rows):  # as B12xBlockscaledLinear._call_factory (vLLM 5dad364d), so autotune benchmarks the same call
        def prepare(state):
            src = torch.empty((rows, k), dtype=torch.bfloat16, device="cuda")
            ws = torch.empty(state.required_workspace, dtype=torch.uint8, device="cuda") if state.required_workspace else None

            def produce():
                src.fill_(0.125)

            def run():
                state.run(src, values, scales, None, activation_scale=None, workspace=ws)
            return PreparedCall(run=run, produce=produce, owners=(values, scales), capture_safe=False)
        return prepare
    calls = {rows: factory(rows) for rows in plan.token_counts}
    t0 = time.time()
    with PreparationSession(device=torch.device("cuda"), autotune=autotune, compile_workers=2) as s:
        s.prepare((plan.request(name=f"k96.hc.{n}x{k}.c{cap}", prepare_calls=calls, benchmark_calls=calls),))
        s.freeze()
    state = require_prepared(plan, "gemm.blockscaled_precision", torch.device("cuda"))
    ws = torch.empty(max(state.required_workspace, 1), dtype=torch.uint8, device="cuda")
    return plan, state, ws, time.time() - t0


def describe(st):
    c = st.config
    cfg = c.mode if c.mode == "quantized" else f"a16 n{c.tile_n} k{c.tile_k} s{c.split_k}"
    low = "-"
    if getattr(st, "dense", None) is not None:
        p = st.dense.lowering
        low = f"{tuple(p.mma_tiler_mn)}/BK{p.tile_k}/u{int(p.policy.large_m_unroll)}"
    return cfg, low


def timed(fn, iters=20):
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        t.append(a.elapsed_time(b))
    return statistics.median(t)


def sample_rows(m):
    import torch
    return torch.linspace(0, m - 1, NROWS).round().long().cuda()


def run_variant(name, out_dir, ckpt, cap_dir):
    import torch
    from b12x.gemm import blockscaled
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig
    shapes, caps, autotune, mode, tile, policy = VARIANTS[name]
    patch(tile, policy)
    pdir = os.path.join(os.environ.get("XDG_CACHE_HOME", "/tmp/cache"), "b12x", "compile", "preparation")
    if name == "ctl-seed":  # as the recipe command: the image's plan seed into this process's own cache
        import shutil
        os.makedirs(pdir, exist_ok=True)
        for fn in glob.glob(os.path.join(SEED_DIR, "*.json")):
            shutil.copy(fn, pdir)
        shutil.copy(os.path.join(SEED_DIR, SEED_PLAN), os.path.join(out_dir, "seed-plan.json"))
    nrec = lambda: sum(len(json.load(open(p)).get("records", {})) for p in glob.glob(os.path.join(pdir, "*.json")))
    w = load_weights(ckpt)
    x, src = load_inputs(w, cap_dir)
    os.makedirs(ROWS, exist_ok=True)
    configs = [(name, None)] if mode != "a16" else [(f"a16-n{n}k{k}s{s}", BlockscaledConfig(mode="a16", tile_n=n, tile_k=k, split_k=s)) for n, k, s in A16]
    if mode == "quantized":
        configs = [(name, BlockscaledConfig(mode="quantized"))]
    with open(os.path.join(out_dir, f"{name}.tsv"), "w") as f:
        for shape in shapes:
            k, n = SHAPES[shape]
            packed, deq = pack(w[shape])
            for vname, override in configs:
                for cap in caps:
                    try:
                        n0 = nrec() if name == "ctl-seed" else 0
                        plan, state, ws, prep_s = build(k, n, cap, packed, autotune, override)
                        n1 = nrec() if name == "ctl-seed" else 0
                    except Exception:
                        f.write(f"# {vname} {shape} cap {cap} FAILED: " + traceback.format_exc().replace("\n", " | ") + "\n"); f.flush()
                        continue
                    print(f"{vname} {shape} cap {cap}: prepared in {prep_s:.0f} s", flush=True)
                    for m in CAPS[cap]:
                        a = x[shape][:m]
                        cfg, low = describe(state.resolve(a))
                        if name == "ctl-seed":
                            cfg += " [seed hit]" if n1 == n0 else f" [tuned +{n1 - n0} records]"
                        r = blockscaled.mm(a, packed, plan=plan, workspace=ws)
                        torch.cuda.synchronize()
                        sha = hashlib.sha1(r.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
                        idx = sample_rows(m)
                        rows = r[idx].float()
                        ref = a[idx].float() @ deq.T
                        err = float((rows - ref).norm() / ref.norm().clamp_min(1e-30))
                        torch.save(rows.cpu(), os.path.join(ROWS, f"{vname}_{shape}_{cap}_{m}.pt"))
                        ms = timed(lambda: blockscaled.mm(a, packed, plan=plan, workspace=ws))
                        row = (vname, shape, cap, m, f"{ms:.4f}", f"{2 * m * n * k / ms / 1e9:.1f}", cfg, low, sha,
                               f"{err:.6f}", src[shape])
                        f.write("\t".join(map(str, row)) + "\n"); f.flush()
                        print("\t".join(map(str, row)), flush=True)
                    del plan, state, ws
                    torch.cuda.empty_cache()


FALLBACK = {"down": "q-def", "up": "a16-n64k64s1"}  # default-config controls when the autotuned ctl is missing


def controls(rows):
    """-> f(shape, cap): 'ctl-seed' at capacity 8192 when it loaded the served record (no new records), else 'ctl'
    when it has rows, else the default-config fallback."""
    seed = {r["shape"] for r in rows if r["variant"] == "ctl-seed" and r["config"].endswith("[seed hit]")}
    ctl = {r["shape"] for r in rows if r["variant"] == "ctl"}
    return lambda sh, cap: "ctl-seed" if cap == 8192 and sh in seed else "ctl" if sh in ctl else FALLBACK[sh]


def compare(out_dir):
    import torch
    rows = read_rows(out_dir)
    ctln = controls(rows)
    ctl = {(r["shape"], r["cap"], r["m"]): r for r in rows if r["variant"] == ctln(r["shape"], r["cap"])}
    with open(os.path.join(out_dir, "compare.tsv"), "w") as f:
        for r in rows:
            c = ctl.get((r["shape"], r["cap"], r["m"]))
            if r["variant"] == ctln(r["shape"], r["cap"]) or c is None:
                continue
            a = torch.load(os.path.join(ROWS, f"{r['variant']}_{r['shape']}_{r['cap']}_{r['m']}.pt"), weights_only=True)
            b = torch.load(os.path.join(ROWS, f"{c['variant']}_{r['shape']}_{r['cap']}_{r['m']}.pt"), weights_only=True)
            cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            rel = ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
            f.write("\t".join(map(str, (r["variant"], r["shape"], r["cap"], r["m"], int(r["sha1"] == c["sha1"]),
                                        f"{cos:.7f}", f"{rel:.3e}", r["err_ref"], c["err_ref"]))) + "\n")


# ---------------------------------------------------------------- driver and report (stdlib only)
def run_all(out_dir, ckpt, cap_dir, budget_s=3000):
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    for name in VARIANTS:
        left = budget_s - (time.time() - start)
        if left < 180:
            print(f"{name}: skipped, budget ({budget_s} s) spent", flush=True)
            continue
        t0 = time.time()
        with open(os.path.join(out_dir, f"{name}.log"), "w") as log:
            env = {**os.environ, **({"XDG_CACHE_HOME": "/tmp/cache-seed"} if name == "ctl-seed" else {})}
            try:
                rc = subprocess.run([sys.executable, os.path.abspath(__file__), "run", name, out_dir, "--ckpt", ckpt]
                                    + (["--cap-dir", cap_dir] if cap_dir else []), env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=min(1200 if name == "ctl" else 600, left)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        print(f"{name}: rc={rc} {time.time() - t0:.0f} s", flush=True)
    try:
        compare(out_dir)
    except Exception:
        traceback.print_exc()


def read_rows(out_dir):
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".tsv") or fn == "compare.tsv":
            continue
        for line in open(os.path.join(out_dir, fn)):
            v = line.rstrip("\n").split("\t")
            if len(v) == len(COLS) and not line.startswith("#"):
                r = dict(zip(COLS, v))
                r["cap"], r["m"] = int(r["cap"]), int(r["m"])
                r["ms"], r["err_ref"] = float(r["ms"]), float(r["err_ref"])
                rows.append(r)
    return rows


def read_cmp(out_dir):
    p = os.path.join(out_dir, "compare.tsv")
    out = {}
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


def seed_assignments(out_dir):
    """Configs of the served plan seed (seed-plan.json, copied by ctl-seed) -> {describe-style config: records}."""
    p = os.path.join(out_dir, "seed-plan.json")
    out = {}
    if os.path.exists(p):
        for rec in json.load(open(p)).get("records", {}).values():
            c = rec.get("config") or {}
            if set(c) >= {"mode", "tile_n", "tile_k", "split_k"}:
                k = c["mode"] if c["mode"] == "quantized" else f"a16 n{c['tile_n']} k{c['tile_k']} s{c['split_k']}"
                out[k] = out.get(k, 0) + 1
    return out


def report(out_dir, rows=None, cmp=None, seed=None):
    rows = read_rows(out_dir) if rows is None else rows
    cmp = read_cmp(out_dir) if cmp is None else cmp
    seed = seed_assignments(out_dir) if seed is None else seed
    by = {(r["variant"], r["shape"], r["cap"], r["m"]): r for r in rows}
    names = list(dict.fromkeys(r["variant"] for r in rows))
    ctln = controls(rows)
    covers = lambda v, sh: any(k[0] == v and k[1] == sh for k in by)
    L, gate = [], {}
    for v in names:
        cells = [k for k in by if k[0] == v and v != ctln(k[1], k[2])]
        gate[v] = "CTL" if not cells else ("PASS" if all(gate_cell(cmp.get(k)) for k in cells) else "FAIL")
    srcs = sorted({r["src"] for r in rows})
    L.append(f"inputs: {'; '.join(srcs) if srcs else 'none'}; weights: layer {LAYER} attention HC, online MXFP8 g32")
    L.append("control per shape and capacity: " + "; ".join(
        f"{sh} " + " ".join(f"c{cap}={ctln(sh, cap)}" for cap in CAPS) for sh in SHAPES)
        + f" (ctl-seed = the served plan record, ctl = autotuned here, {FALLBACK['down']}/{FALLBACK['up']} = default-config fallback)")
    # served plan record
    L.append("")
    L.append("== served plan record (seed 33d9b32e) at capacity 8192: ctl-seed autotunes over a copy of the image seed; "
             "no new record = the served selection was loaded")
    seeds = [r for r in rows if r["variant"] == "ctl-seed"]
    served = "N/A" if not seeds else "hit" if all(r["config"].endswith("[seed hit]") for r in seeds) else "miss"
    for sh in SHAPES:
        rs = by.get(("ctl-seed", sh, 8192, max(CAPS[8192])))
        rc = by.get(("ctl", sh, 8192, max(CAPS[8192])))
        cc = rc["config"] if rc else "n/a"
        L.append(f"  {sh}: ctl-seed {rs['config'] if rs else 'n/a'}; ctl {cc}"
                 + (f"; ctl's config appears in {seed.get(cc, 0)} of {sum(seed.values())} seed records" if seed and rc else ""))
    if seed:
        L.append("  seed record configs: " + ", ".join(f"{k}: {n}" for k, n in sorted(seed.items(), key=lambda kv: -kv[1])))
        L.append("  (seed records are keyed by hash, so a miss cannot name the HC record; this only shows whether ctl's config is one the server uses)")
    for shape in SHAPES:
        k, n = SHAPES[shape]
        L.append("")
        L.append(f"== {shape} K{k} N{n}: full-op ms per call (blockscaled.mm with activation quantization and workspace), "
                 f"x = control / variant; numerics vs control: '=' bit-equal, '~' within gate, '!' fails")
        ctls = {ctln(shape, cap) for cap in CAPS}
        vs = [v for v in names if (not v.startswith("a16-") or v in ctls) and covers(v, shape)]
        L.append("cap      M " + "".join(v.rjust(22) for v in vs))
        for cap, ms in CAPS.items():
            c = ctln(shape, cap)
            for m in ms:
                c0 = by.get((c, shape, cap, m))
                cells = []
                for v in vs:
                    r = by.get((v, shape, cap, m))
                    if r is None:
                        cells.append("n/a".rjust(22)); continue
                    mark = "" if v == c else ("=" if (cmp.get((v, shape, cap, m)) or {}).get("same_bytes") == "1"
                                              else "~" if gate_cell(cmp.get((v, shape, cap, m))) else "!")
                    x = f" x{c0['ms'] / r['ms']:.2f}" if c0 and v != c else ""
                    cells.append(f"{r['ms']:7.3f}{x}{mark}".rjust(22))
                L.append(f"{cap:5d} {m:6d} " + "".join(cells))
        L.append("plans (config / dense lowering) at the largest M of each capacity:")
        for v in vs:
            got = [f"c{cap}: {by[(v, shape, cap, max(ms))]['config']} {by[(v, shape, cap, max(ms))]['lowering']}"
                   for cap, ms in CAPS.items() if (v, shape, cap, max(ms)) in by]
            if got:
                L.append(f"  {v}: " + "; ".join(got))
    # anchor first: it scales the projection to k81's measured per-token cost
    have = all((ctln(sh, 8192), sh, 8192, c) in by for c in CHUNKS[8192] for sh in SHAPES)
    us = sum(by[(ctln(sh, 8192), sh, 8192, c)]["ms"] for c in CHUNKS[8192] for sh in SHAPES) * CALLS * 1000 / sum(CHUNKS[8192]) if have else None
    anchor = "OK" if us and 0.85 <= us / K81_US_PER_TOKEN <= 1.15 else "OFF" if us else "N/A"
    scale = K81_US_PER_TOKEN / us if us else None
    L.append("")
    L.append(f"== anchor: control at budget 8192 = {us:.1f} us/token vs k81 dense-HC class {K81_US_PER_TOKEN} us/token "
             f"(ratio {us / K81_US_PER_TOKEN:.2f}); ANCHOR={anchor} (OK = within 15%)" if us else "== anchor: control cells missing; ANCHOR=N/A")
    L.append("")
    L.append(f"== projection: seconds saved per cold 16K prompt = sum over chunks and covered shapes of (control - variant) ms "
             f"x {CALLS} calls per step; k81-scaled = that x {K81_US_PER_TOKEN} / control us/token. Headline: "
             + ("k81-scaled (ANCHOR=OK)" if anchor == "OK" else f"raw (ANCHOR={anchor}, the k81 scaling is shown but not used)"))
    best = {}
    for b, chunks in CHUNKS.items():
        L.append(f"budget {b} chunks {list(chunks)} ({CHUNK_SRC[b]}):")
        for v in [v for v in names if gate[v] != "CTL" and v not in ("ctl", "ctl-seed") and not (v.startswith("a16-") and b != 8192)]:
            saved, missing = 0.0, False
            for c in chunks:
                for sh in SHAPES:
                    cn = ctln(sh, b)
                    if not covers(v, sh) or v == cn:
                        continue
                    c0, r = by.get((cn, sh, b, c)), by.get((v, sh, b, c))
                    if c0 is None or r is None:
                        missing = True; continue
                    saved += (c0["ms"] - r["ms"]) * CALLS / 1000
            if missing:
                L.append(f"  {v}: incomplete cells"); continue
            k81s = saved * scale if scale else None
            hv = k81s if anchor == "OK" else saved
            other = f", {k81s:+.3f} s k81-scaled" if k81s is not None and anchor != "OK" else f", {saved:+.3f} s raw" if anchor == "OK" else ""
            flag = "" if gate[v] == "PASS" else "  (numerics FAIL: no claim)"
            L.append(f"  {v}: {hv:+.3f} s ({100 * hv / K81_TTFT_16K:+.1f}% of k81's {K81_TTFT_16K} s TTFT){other}{flag}")
            if gate[v] == "PASS" and hv > best.get(b, ("", 0.0))[1]:
                best[b] = (v, hv)
    sweep = sorted({r["variant"] for r in rows if r["variant"].startswith("a16-")})
    if sweep:
        L.append("")
        L.append(f"== up K320 N10240, A16 configs at capacity 8192 (ms per M; control {ctln('up', 8192)})")
        L.append("config            " + "".join(f"{m:>9}" for m in CAPS[8192]))
        for v in list(dict.fromkeys([ctln("up", 8192)] + sweep)):
            L.append(f"{v:<18}" + "".join(f"{by[(v, 'up', 8192, m)]['ms']:9.3f}" if (v, "up", 8192, m) in by else "      n/a"
                                         for m in CAPS[8192]))
    L.append("")
    L.append("== numerics gate per variant: " + " ".join(f"{v}={gate[v]}" for v in names if gate[v] != "CTL"))
    bb = " ".join(f"b{b}:{best[b][0]}{best[b][1]:+.2f}s" for b in CHUNKS if b in best) or "none"
    L.append(f"VERDICT anchor={anchor} served={served} best={bb} control8192={ctln('down', 8192)}/{ctln('up', 8192)} "
             + " ".join(f"{v}={gate[v]}" for v in names if gate[v] != "CTL" and not v.startswith("a16-")))
    return "\n".join(L)


def check(ckpt):
    """CPU-only check inside the image: every API name the GPU path touches, and the checkpoint keys."""
    import torch  # noqa: F401
    from b12x.gemm import blockscaled
    from b12x.gemm.blockscaled import _preparation as P
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared  # noqa: F401
    import vllm.envs as envs
    from vllm.utils.b12x import get_b12x_dense_activation_mode
    import vllm.model_executor.layers.quantization.utils.mxfp8_utils as mu
    for a in ("_large_m_shape", "_LARGE_M_DENSE", "LARGE_M_ROWS", "_dense_lowering"):
        assert hasattr(P, a), a
    assert os.path.exists(os.path.join(SEED_DIR, SEED_PLAN)), f"served plan seed {SEED_PLAN} not in {SEED_DIR}"
    for a in ("plan_regimes", "BlockscaledQuery", "pack_weight", "mm"):
        assert hasattr(blockscaled, a), a
    assert callable(mu.mxfp8_e4m3_quantize)
    BlockscaledConfig(mode="a16", tile_n=64, tile_k=64, split_k=1)
    print("activation_mode", get_b12x_dense_activation_mode("mxfp8"), "workspace_nbytes", envs.VLLM_B12X_BLOCKSCALED_WORKSPACE_MAX_BYTES)
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{LAYER}.attn_hyper_connection"
    for t in ("input_mix_weight_down", "block_inject_weight", "input_mix_weight_up", "hc_norm"):
        f = idx[f"{pre}.{t}.weight"]
        assert os.path.exists(os.path.join(ckpt, f)), f
    print("check ok")


def selftest():
    rows, cmp = [], {}
    for v, (shapes, caps, _, mode, tile, _) in VARIANTS.items():
        if mode == "a16" or v == "ctl-seed":
            continue
        for shape in shapes:
            for cap in caps:
                for m in CAPS[cap]:
                    ms = m / 2000 * (0.5 if v == "q-128x128k64u0" else 1.0) * (1.2 if v == "q-64x128k64u0" else 1.0)
                    rows.append(dict(variant=v, shape=shape, cap=cap, m=m, ms=ms, tflops=0, config="quantized",
                                     lowering="x", sha1="a" if v in ("ctl", "q-def") else "b", err_ref=0.01, src="synthetic"))
                    if v != "ctl":
                        cmp[(v, shape, cap, m)] = dict(same_bytes="1" if v == "q-def" else "0", cos="0.99999",
                                                       maxrel="1e-3", err_ref="0.0105" if v != "q-128x64k64u0" else "0.05",
                                                       err_ctl="0.01")
    out = report("/nonexistent", rows, cmp, {"quantized": 10, "a16 n64 k64 s1": 5})
    assert "q-def=PASS" in out and "q-128x128k64u0=PASS" in out and "q-128x64k64u0=FAIL" in out, out
    # b8192 down saving of the half-time variant: chunks 6048,6048,3024,1264 = 16384 rows -> ctl 8.192 ms, saved 4.096 ms x 96
    # b8192 raw down saving of the half-time variant: chunks 16384 rows -> ctl 8.192 ms, saved 4.096 ms x 96 = 0.393 s;
    # anchor: ctl down+up 16.384 ms x 96 / 16384 = 96 us/token vs 48.7 -> k81-scaled 0.393 x 48.7 / 96 = 0.199 s
    # ANCHOR=OFF -> raw headline 0.393 s, k81-scaled 0.393 x 48.7 / 96 = 0.199 s shown beside it
    assert "q-128x128k64u0: +0.393 s (+5.0% of k81's 7.925 s TTFT), +0.199 s k81-scaled" in out, out
    assert "best=b8192:q-128x128k64u0+0.39s" in out and "ANCHOR=OFF" in out and "control8192=ctl/ctl" in out, out
    assert "served=N/A" in out and "ctl's config appears in 10 of 15 seed records" in out, out
    out3 = report("/nonexistent", [r for r in rows if r["variant"] != "ctl"], cmp, {})
    assert "control8192=q-def/a16-n64k64s1" in out3 and "down c8192=q-def" in out3, out3
    assert "q-128x128k64u0: +0.393 s" in out3 and "ANCHOR=N/A" in out3, out3  # up fallback absent: no k81 scaling
    seedrows = [dict(r, variant="ctl-seed", config="quantized [seed hit]") for r in rows if r["variant"] == "ctl" and r["cap"] == 8192]
    out4 = report("/nonexistent", rows + seedrows, cmp, {})
    assert "served=hit" in out4 and "control8192=ctl-seed/ctl-seed" in out4 and "down c8192=ctl-seed c9072=ctl" in out4, out4
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    ck = a[a.index("--ckpt") + 1] if "--ckpt" in a else None
    cd = a[a.index("--cap-dir") + 1] if "--cap-dir" in a else None
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["check"]:
        check(ck)
    elif a[:1] == ["report"]:
        print(report(a[1]))
    elif a[:1] == ["compare"]:
        compare(a[1])
    elif a[:1] == ["all"]:
        run_all(a[1], ck, cd)
    elif a[:1] == ["run"]:
        try:
            run_variant(a[1], a[2], ck, cd)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(__doc__)
