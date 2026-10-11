k91-moe-prefill-probe result: one routed-MoE layer (checkpoint NVFP4 weights, layer 5 timed, layers 5 and 47 for numerics) at prefill M 1,264 / 3,024 / 6,048 / 8,192 in the shipped 1x image on one Spark. b12x serving anchor (capacity 8192, M32 tile) against M128, M64, layer-max FC1 scale with and without the NVFP4 split-materialized kernels, and the vendored FlashInfer #5863 kernel. Synthetic routing (uniform and skewed); no router captures exist. Measure only; design note docs/notes/moe-prefill-dispatch.md.

Verdict: anchor=OFF b12x-m32=PASS b12x-m128=PASS b12x-m128-l1max-split=PASS fi5863=PASS b12x-m128-l1max-mono=PASS b12x-m64=PASS (probe exit 0)

```
== numerics: rel L2 error vs FP32 dequant reference, rows 0..255 at M 6048 (gate: <= 1.10 x b12x-m32 + 0.005 in every cell)
variant                     L5 uniform       L5 skew   L47 uniform      L47 skew  gate
b12x-m32                        0.1593        0.1675        0.2682        0.2256  PASS
b12x-m128                       0.1593        0.1675        0.2682        0.2256  PASS
b12x-m128-l1max-split           0.1600        0.1663        0.2685        0.2257  PASS
fi5863                          0.1575        0.1604        0.1575        0.1576  PASS
b12x-m128-l1max-mono            0.1602        0.1664        0.2686        0.2258  PASS
b12x-m64                        0.1593        0.1675        0.2682        0.2256  PASS

== anchor: b12x-m32 layer 5 vs k81 serving (routed MoE ms per layer per chunk)
  uniform: M 1264: 7.30 vs 6.66 (+10%); M 3024: 8.42 vs 8.98 (-6%); M 6048: 10.81 vs 12.57 (-14%)
  skew: M 1264: 7.30 vs 6.66 (+10%); M 3024: 8.41 vs 8.98 (-6%); M 6048: 10.97 vs 12.57 (-13%)
ANCHOR=OFF (closest pattern within 10% at M 6048; OFF = ratios not publishable)

== speed, layer 5, uniform routing: ms per call [TFLOPS] (x = b12x-m32 ms / variant ms)
     M act max/mean                   b12x-m32                 b12x-m128     b12x-m128-l1max-split                    fi5863      b12x-m128-l1max-mono                  b12x-m64
  1264 512   42/ 24.7       7.304 [ 17.0] x1.00       8.981 [ 13.8] x0.81      12.013 [ 10.3] x0.61       6.445 [ 19.3] x1.13       8.886 [ 14.0] x0.82       7.314 [ 17.0] x1.00
  3024 512   83/ 59.1       8.421 [ 35.3] x1.00      10.275 [ 28.9] x0.82      15.895 [ 18.7] x0.53       7.604 [ 39.1] x1.11      10.135 [ 29.3] x0.83       8.407 [ 35.4] x1.00
  6048 512  149/118.1      10.813 [ 55.0] x1.00      12.797 [ 46.5] x0.85      21.438 [ 27.7] x0.50       9.707 [ 61.2] x1.11      12.588 [ 47.2] x0.86      10.645 [ 55.9] x1.02
  8192 512  197/160.0      12.995 [ 62.0] x1.00      15.574 [ 51.7] x0.83      27.812 [ 29.0] x0.47      12.073 [ 66.7] x1.08      15.170 [ 53.1] x0.86      12.726 [ 63.3] x1.02

== speed, layer 5, skew routing: ms per call [TFLOPS] (x = b12x-m32 ms / variant ms)
     M act max/mean                   b12x-m32                 b12x-m128     b12x-m128-l1max-split                    fi5863      b12x-m128-l1max-mono                  b12x-m64
  1264 512  100/ 24.7       7.298 [ 17.0] x1.00       8.982 [ 13.8] x0.81      12.218 [ 10.2] x0.60       6.509 [ 19.1] x1.12       8.860 [ 14.0] x0.82       7.322 [ 17.0] x1.00
  3024 512  239/ 59.1       8.415 [ 35.3] x1.00      10.269 [ 28.9] x0.82      16.440 [ 18.1] x0.51       7.604 [ 39.1] x1.11      10.109 [ 29.4] x0.83       8.436 [ 35.2] x1.00
  6048 512  511/118.1      10.967 [ 54.2] x1.00      12.906 [ 46.1] x0.85      22.013 [ 27.0] x0.50       9.852 [ 60.3] x1.11      12.657 [ 47.0] x0.87      10.826 [ 54.9] x1.01
  8192 512  708/160.0      13.328 [ 60.4] x1.00      15.315 [ 52.6] x0.87      30.510 [ 26.4] x0.44      11.842 [ 68.0] x1.13      14.930 [ 53.9] x0.89      12.954 [ 62.2] x1.03
* = numerics gate FAIL, NOANCHOR or missing: no speed claim

== selected configs
  b12x-m32: ?
  b12x-m128: ?
  b12x-m128-l1max-split: ?
  fi5863: PR #5863 a0997210 run_moe_w4a4, scalar layer-max a13/a2
  b12x-m128-l1max-mono: ?
  b12x-m64: ?
VERDICT anchor=OFF b12x-m32=PASS b12x-m128=PASS b12x-m128-l1max-split=PASS fi5863=PASS b12x-m128-l1max-mono=PASS b12x-m64=PASS
```

Raw TSV and logs stay on dgx-01 in `/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k91-moe-prefill-probe-20261011-2125`. Results: {RESULTS_URL}
