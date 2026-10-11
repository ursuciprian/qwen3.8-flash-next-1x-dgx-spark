#!/usr/bin/env python3
"""k88 phase 0 (1x #27): why is the MXFP8 dense GEMM K6144 N2560 (GDN out_proj, attention o_proj) 4.7x / 3.4x slower at
M 6048 than at M 3024 in serving (k81)? Serving runs prefill rows through the capacity regime of a plan declared for
max_rows = max_num_batched_tokens with expected_m None (vllm b12x_blockscaled.py ensure_plan), whose dense lowering is
b12x's default (not autotuned). This probe times the same native MXFP8 dense GEMM per M through:
  capN        plan for max_rows N, expected_m None (the serving capacity regime), N = 8192 / 9072 / 15120
  cap8192-u0  as cap8192 with the large-M unroll forced off (_large_m_unroll_override=False); this is what the
              policy-fix image (b12x 6fd6b023) does for the capacity regime
  cap8192-t128k64-u0  128x128 tile, BK64, unroll off (the plan b12x's source calls the large-M win for K >= 4096)
  exact       plan for exactly M rows (expected_m = M)
Operands: random BF16 quantized to MXFP8 rows (per-32 E8M0 scales), like the served MXFP8 copies. Output: one TSV row per
(shape, variant, M) with the median of 20 timed calls (CUDA events) and the selected default plan, plus a summary.
Run in the serving image: python3 gemm_probe.py [out.tsv]. Each variant is isolated: a failure is logged, the rest runs.
"""
import statistics, sys, traceback
from contextlib import contextmanager

import torch

# (n, k, label, variants): in_proj is only a linear-scaling reference, so it gets two capacity variants
SHAPES = ((2560, 6144, "out_proj/o_proj", ("cap8192", "cap8192-u0", "cap8192-t128k64-u0", "cap9072", "cap15120", "exact")),
          (16384, 2560, "in_proj_qkvz", ("cap8192", "cap9072")))
EXACT_MS = (3024, 6048, 9072)
MS = (1264, 3024, 4536, 6048, 7560, 8192, 9072, 15120)
VARIANTS = (("cap8192", 8192, {}), ("cap8192-u0", 8192, {"_large_m_unroll_override": False}),
            ("cap8192-t128k64-u0", 8192, {"mma_tiler_mn": (128, 128), "_tile_k_override": 64, "_large_m_unroll_override": False}),
            ("cap9072", 9072, {}), ("cap15120", 15120, {}), ("exact", None, {}))
OPTS = dict(ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu", c_dtype="bfloat16", sf_vec_size=32)


def operand(rows, k):
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch
    q = quantize_mxfp8_rows_torch((torch.randn(rows, k, 1, device="cuda", dtype=torch.bfloat16) / 4))
    return q.values, q.scale_mma


@contextmanager
def prepared(lhs, rhs, **options):
    """tests/gemm/_blockscaled.py prepared(), dense branch: default lowering, no autotune."""
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparationSession, PreparedCall
    query = blockscaled.query_from_call(lhs, rhs, **options)
    plan = blockscaled.plan(query)

    def call(state):
        return PreparedCall(run=lambda: state.run(*lhs, *rhs, None, None, None, getattr(torch, query.output_dtype), None))
    with PreparationSession(device=lhs[0].device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="probe", prepare_call=call),))
        session.freeze()
        yield plan


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


def default_plan(m, n, k, expected_m):
    try:
        from b12x._lib.dense_gemm import _select_default_dense_gemm_plan
        from b12x._lib.utils import get_num_sm
        p = _select_default_dense_gemm_plan(m, n, k, get_num_sm(torch.device("cuda")), is_mxfp8=True, expected_m=expected_m)
        return str(getattr(p, "mma_tiler_mn", p))
    except Exception as e:  # informational only
        return f"? ({type(e).__name__})"


def main():
    from b12x.gemm import blockscaled
    out = open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w")
    out.write("shape\tn\tk\tvariant\tm\tms\tTFLOPS\tdefault_tile\n")
    res = {}
    for n, k, label, wanted in SHAPES:
        torch.manual_seed(n + k)
        rhs = operand(n, k)
        ops = {m: operand(m, k) for m in MS}
        for name, cap, extra in VARIANTS:
            if name not in wanted:
                continue
            caps = [(m, m) for m in EXACT_MS] if cap is None else [(cap, None)]
            for rows, exact_m in caps:
                try:
                    lhs = ops.get(rows) or operand(rows, k)
                    opts = dict(OPTS, **extra, **({"expected_m": exact_m} if exact_m else {}))
                    with prepared(lhs, rhs, **opts) as plan:
                        for m in MS:
                            if m > rows or (exact_m and m != exact_m):
                                continue
                            a = ops[m]
                            ms = timed(lambda: blockscaled.mm(a, rhs, plan=plan))
                            tf = 2 * m * n * k / ms / 1e9
                            res[(label, name, m)] = ms
                            out.write(f"{label}\t{n}\t{k}\t{name}\t{m}\t{ms:.4f}\t{tf:.1f}\t{default_plan(rows, n, k, exact_m)}\n")
                            out.flush()
                except Exception:
                    out.write(f"# {label} {name} rows={rows}: FAILED\n# " + traceback.format_exc().replace("\n", "\n# ") + "\n")
                    out.flush()
    out.write("\n# summary: ms per call (us per row in brackets)\n")
    for _, _, label, _ in SHAPES:
        out.write(f"# {label}\n#   {'M':>6s} " + " ".join(f"{v[0][:18]:>18s}" for v in VARIANTS) + "\n")
        for m in MS:
            out.write(f"#   {m:6d} " + " ".join(
                f"{res[(label, v[0], m)]:8.3f} [{1000 * res[(label, v[0], m)] / m:6.3f}]" if (label, v[0], m) in res else f"{'':>18s}"
                for v in VARIANTS) + "\n")
    out.close()


if __name__ == "__main__":
    main()
