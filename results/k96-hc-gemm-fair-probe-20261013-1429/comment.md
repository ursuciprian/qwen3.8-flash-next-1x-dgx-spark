k96-hc-gemm-fair-probe result: the two hyperconnection GEMMs of the 1x v2.3.0 stack (down K10240 to N336, up K320 to N10240, layer 5 weights of the served checkpoint, quantized to MXFP8 the way the server does at load) on identical inputs and weights for every variant. Each variant goes through `blockscaled.plan_regimes` and `blockscaled.mm` with the server's query: activation quantization, output and workspace contract included in every timing. This replaces the k90 candidate column, which timed a prequantized GEMM on other random inputs.

Control is the served plan record when the probe can load it from the image's plan seed (`ctl-seed`, capacity 8192), otherwise the autotuned selection, as a server with `B12X_AUTOTUNE=1` picks it; the report says which one it got. The candidates force the large-row dense lowering (128x128/BK64 without unroll and neighbours) through the same dispatch. `q-prop` keeps the previous plan below 4096 rows, the way the v2.3.0 tile fix does for K6144 N2560. The up projection has K320, which has no quantized path, so its candidates are the A16 launch configs.

Numerics come before speed: per cell I compare the output bytes with the control, the cosine and max relative difference on 128 sampled rows, and the error against an FP32 reference. A variant without a pass in every cell gets no speed claim. Projected savings use the chunk lists of k81/k88 (budgets 8192 and 9072) and the 12096 hypothesis that k97 checks, times 96 HC calls per step. The headline figure is scaled to k81's measured dense-HC cost (48.7 us/token) through the anchor ratio, with the raw probe figure beside it. They are probe projections, not server measurements.

Verdict: anchor=N/A served=N/A best=none control8192=q-def/a16-n64k64s1  (probe exit 0)

```
inputs: none; weights: layer 5 attention HC, online MXFP8 g32
control per shape and capacity: down c8192=q-def c9072=q-def c12096=q-def; up c8192=a16-n64k64s1 c9072=a16-n64k64s1 c12096=a16-n64k64s1 (ctl-seed = the served plan record, ctl = autotuned here, q-def/a16-n64k64s1 = default-config fallback)

== served plan record (seed 33d9b32e) at capacity 8192: ctl-seed autotunes over a copy of the image seed; no new record = the served selection was loaded
  down: ctl-seed n/a; ctl n/a
  up: ctl-seed n/a; ctl n/a
  seed record configs: a16 n64 k64 s1: 56, quantized: 52, a16 n64 k128 s1: 46, a16 n64 k64 s4: 29, a16 n128 k64 s1: 24, a16 n64 k64 s8: 20, a16 n64 k128 s4: 16, a16 n64 k128 s8: 7, a16 n128 k128 s1: 7, a16 n128 k64 s4: 5, a16 n64 k128 s2: 3, a16 n128 k128 s8: 2
  (seed records are keyed by hash, so a miss cannot name the HC record; this only shows whether ctl's config is one the server uses)

== down K10240 N336: full-op ms per call (blockscaled.mm with activation quantization and workspace), x = control / variant; numerics vs control: '=' bit-equal, '~' within gate, '!' fails
cap      M 
 8192   1264 
 8192   3024 
 8192   4536 
 8192   6048 
 8192   8192 
 9072   1264 
 9072   3024 
 9072   9072 
12096   1264 
12096   3024 
12096  12096 
plans (config / dense lowering) at the largest M of each capacity:

== up K320 N10240: full-op ms per call (blockscaled.mm with activation quantization and workspace), x = control / variant; numerics vs control: '=' bit-equal, '~' within gate, '!' fails
cap      M 
 8192   1264 
 8192   3024 
 8192   4536 
 8192   6048 
 8192   8192 
 9072   1264 
 9072   3024 
 9072   9072 
12096   1264 
12096   3024 
12096  12096 
plans (config / dense lowering) at the largest M of each capacity:

== anchor: control cells missing; ANCHOR=N/A

== projection: seconds saved per cold 16K prompt = sum over chunks and covered shapes of (control - variant) ms x 96 calls per step; k81-scaled = that x 48.7 / control us/token. Headline: raw (ANCHOR=N/A, the k81 scaling is shown but not used)
budget 8192 chunks [6048, 6048, 3024, 1264] (k81/k88 measured):
budget 9072 chunks [9072, 3024, 3024, 1264] (k88 measured):
budget 12096 chunks [12096, 3024, 1264] (hypothesis, k97 checks):

== numerics gate per variant: 
VERDICT anchor=N/A served=N/A best=none control8192=q-def/a16-n64k64s1 
```
