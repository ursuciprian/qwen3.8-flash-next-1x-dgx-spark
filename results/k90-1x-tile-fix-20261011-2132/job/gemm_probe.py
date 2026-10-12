#!/usr/bin/env python3
"""k90 phase 0 (1x #27): the served MXFP8 dense projections through blockscaled.plan_regimes, as vLLM declares them
(b12x_blockscaled.py ensure_plan: BlockscaledQuery for max_num_batched_tokens 8192, expected_m None, provided
workspace), timed per live M. Run the same file in the shipped image and in the tile-fix image for a before/after.
  regime rows: ms per call (median of 20, CUDA events), the tile/BK/unroll of the regime that served that M, and a
               sha1 of the output (fixed seeds: identical inputs in both images, so equal hashes = byte-identical)
  cand rows:   (new image only, informational) the dense GEMM alone with a forced 128x128/BK64/unroll-off plan for
               max_rows 8192, to see whether a shape without the fix has the same large-M cliff
Saves the K6144 N2560 outputs at M >= 4536 for `compare`.
Usage: gemm_probe.py run OUT_DIR [cand] | gemm_probe.py compare OLD_DIR NEW_DIR
"""
import hashlib, statistics, sys, traceback

import torch

CAP = 8192
MS = (1264, 3024, 4536, 6048, 8192)
# (N, K, label): from the k81 trace (1x TP=1) plus the TP=2 row-parallel width of out_proj/o_proj
SHAPES = ((2560, 6144, "out_proj/o_proj"), (2560, 3072, "out_proj/o_proj TP2"), (336, 10240, "hc proj"),
          (1280, 2560, "shared gate_up"), (16384, 2560, "in_proj_qkvz"))
SAVE = {(2560, 6144)}


def weight(n, k):
    from b12x.gemm import blockscaled
    w = (torch.randn(n, k, device="cuda") / 4).to(torch.float8_e4m3fn)
    s = (torch.randint(-2, 2, (n, k // 32), device="cuda") + 127).to(torch.uint8)
    return blockscaled.pack_weight(w, s)


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def regimes(n, k, packed, source, out, out_dir):
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared
    query = blockscaled.BlockscaledQuery(recipe="mxfp8", num_tokens=CAP, in_features=k, padded_in_features=k,
                                         out_features=n, workspace_form="provided", expected_m=None)
    plan = blockscaled.plan_regimes(query, exact_m=())
    values, scales = packed.weight.values, packed.weight.scale_mma

    def factory(rows):
        def prepare(state):
            ws = torch.empty(state.required_workspace, dtype=torch.uint8, device="cuda") if state.required_workspace else None
            return PreparedCall(run=lambda: state.run(source[:rows], values, scales, None, activation_scale=None, workspace=ws))
        return prepare
    calls = {rows: factory(rows) for rows in plan.token_counts}
    with PreparationSession(device=source.device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="probe", prepare_calls=calls),))
        session.freeze()
        state = require_prepared(plan, "gemm.blockscaled_precision", source.device)
        ws = torch.empty(max(state.required_workspace, 1), dtype=torch.uint8, device="cuda")
        for m in MS:
            a = source[:m]
            p = state.resolve(a).dense.lowering
            tile = f"{tuple(p.mma_tiler_mn)}/BK{p.tile_k}/u{int(p.policy.large_m_unroll)}"
            r = blockscaled.mm(a, packed, plan=plan, workspace=ws)
            torch.cuda.synchronize()
            digest = hashlib.sha1(r.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]
            if (n, k) in SAVE and m >= 4536:
                torch.save(r.cpu(), f"{out_dir}/out_{n}x{k}_{m}.pt")
            ms = timed(lambda: blockscaled.mm(a, packed, plan=plan, workspace=ws))
            out.write(f"regime\t{n}\t{k}\t{m}\t{ms:.4f}\t{2 * m * n * k / ms / 1e9:.1f}\t{tile}\t{digest}\t{plan.token_counts}\n")
            out.flush()


def candidate(n, k, out):
    """Dense GEMM alone (k88 method) for max_rows 8192 with the forced 128x128/BK64/unroll-off plan."""
    from b12x.gemm import blockscaled
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch
    from b12x.preparation import PreparationSession, PreparedCall
    q = lambda rows: (lambda t: (t.values, t.scale_mma))(quantize_mxfp8_rows_torch(torch.randn(rows, k, 1, device="cuda", dtype=torch.bfloat16) / 4))
    rhs, lhs = q(n), q(CAP)
    opts = dict(ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu", c_dtype="bfloat16", sf_vec_size=32,
                mma_tiler_mn=(128, 128), _tile_k_override=64, _large_m_unroll_override=False)
    query = blockscaled.query_from_call(lhs, rhs, **opts)
    plan = blockscaled.plan(query)
    with PreparationSession(device=lhs[0].device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="cand", prepare_call=lambda st: PreparedCall(
            run=lambda: st.run(*lhs, *rhs, None, None, None, torch.bfloat16, None))),))
        session.freeze()
        for m in MS:
            a = q(m)
            ms = timed(lambda: blockscaled.mm(a, rhs, plan=plan))
            out.write(f"cand\t{n}\t{k}\t{m}\t{ms:.4f}\t{2 * m * n * k / ms / 1e9:.1f}\t(128, 128)/BK64/u0\t-\t-\n")
            out.flush()


def run(out_dir, cand):
    out = open(f"{out_dir}/gemm_probe.tsv", "w")
    out.write("kind\tn\tk\tm\tms\tTFLOPS\tplan\tsha1\ttoken_counts\n")
    for n, k, label in SHAPES:
        torch.manual_seed(n * 7 + k)
        try:
            packed = weight(n, k)
            source = torch.randn(CAP, k, device="cuda", dtype=torch.bfloat16) / 4
            regimes(n, k, packed, source, out, out_dir)
        except Exception:
            out.write(f"# regime {label} FAILED\n# " + traceback.format_exc().replace("\n", "\n# ") + "\n"); out.flush()
        if cand:
            try:
                candidate(n, k, out)
            except Exception:
                out.write(f"# cand {label} FAILED\n# " + traceback.format_exc().replace("\n", "\n# ") + "\n"); out.flush()
        torch.cuda.empty_cache()
    out.close()


def compare(old, new):
    import glob, os
    rows = {}
    for d, tag in ((old, "old"), (new, "new")):
        for line in open(f"{d}/gemm_probe.tsv"):
            f = line.rstrip("\n").split("\t")
            if f[0] in ("regime", "cand"):
                rows[(tag if f[0] == "regime" else "cand", int(f[1]), int(f[2]), int(f[3]))] = f
    ok = True
    print("n\tk\tm\told_ms\tnew_ms\tspeedup\told_plan\tnew_plan\tcand_ms\tsame_bytes")
    for n, k, label in SHAPES:
        for m in MS:
            o, w, c = rows.get(("old", n, k, m)), rows.get(("new", n, k, m)), rows.get(("cand", n, k, m))
            if not o or not w:
                print(f"{n}\t{k}\t{m}\tmissing"); ok = False; continue
            same = o[7] == w[7]
            if m < 4096 and not same:
                ok = False
            print(f"{n}\t{k}\t{m}\t{o[4]}\t{w[4]}\t{float(o[4]) / float(w[4]):.2f}\t{o[6]}\t{w[6]}\t{c[4] if c else '-'}\t{'yes' if same else 'no'}")
    for f in sorted(glob.glob(f"{new}/out_*.pt")):
        g = f"{old}/{os.path.basename(f)}"
        a, b = torch.load(f, weights_only=True).float(), torch.load(g, weights_only=True).float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rel = ((a - b).abs().max() / b.abs().max()).item()
        good = cos >= 0.9999 and rel < 1e-2 and bool(torch.isfinite(a).all())
        ok &= good
        print(f"# {os.path.basename(f)}: cos {cos:.6f} max_abs_rel {rel:.2e} {'ok' if good else 'MISMATCH'}")
    w = rows.get(("new", 2560, 6144, 6048)); o = rows.get(("old", 2560, 6144, 6048))
    if not (w and o and float(w[4]) < 0.8 * float(o[4]) and w[6].startswith("(128, 128)/BK64")):
        print("# no large-M plan or no speedup for K6144 N2560 at M 6048"); ok = False
    print(f"CHECK={'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "cand")
    else:
        compare(sys.argv[2], sys.argv[3])
