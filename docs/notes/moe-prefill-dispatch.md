# Routed-MoE prefill dispatch: options and probe plan

Status: design note, 2026-10-11. Tracks #27. Nothing here ships yet. The micro-benchmark k91-moe-prefill-probe is prepared but not armed.

## Where the time goes

The k81 profile ([results/k81-1x-prefill-profile-20261011-0852](../../results/k81-1x-prefill-profile-20261011-0852)) measured one cold 16K prompt on 1x v2.2.0:

- TTFT is 7.93 s, which is 484 us/token.
- The routed MoE (b12x NVFP4 W4A4) costs 119.5 us/token, 26% of the GPU time.
- Each prefill step costs 248 ms + 58.9 us/token, from a least-squares fit over the 6048 / 6048 / 3024 / 1264 chunks.

Per layer per chunk, that is 12.6 ms at 6048 tokens, 9.0 ms at 3024 and 6.7 ms at 1264.

The 248 ms is the expert weight stream: 48 layers x 1.42 GB = 68 GB per step at about 270 GB/s. Every grouped-GEMM backend pays it once per step, because at prefill sizes every expert has work. With top-10 of 512, the mean is 24.7 routed rows per expert at 1264 tokens and 118 at 6048. Under uniform routing the chance that a given expert gets no token at 1264 is about e^-24.7. Skipping idle experts saves nothing here. b12x already builds expert-major work only for routed rows (append-only expert rows, then a work list per expert and M tile).

A 16K prompt takes 4 steps, because the 3024-token mamba alignment cuts a budget of 8192 to 6048-token steps. That puts about 1.0 s of the 7.9 s TTFT in weight streaming, whatever the MoE backend.

Two limits on what MoE dispatch can buy:

- Removing the routed MoE completely would give 484 -> 365 us/token, 1.33x. That is the ceiling for MoE work. Dense GEMMs, attention, GDN and norms make up the other 74%.
- Fewer, larger steps cut the fixed weight stream for every backend. 16K in 2 steps instead of 4 saves about 0.5 s. k88 tried 9072 (3 steps) and lost 4.5% / 5.5% cold 16K. The cause is the K6144 N2560 MXFP8 tile cliff above about 4.5K rows, which k90 targets. That A/B should be repeated after the k90 fix.

## What the shipped b12x path does at prefill

I inferred this from the code at b12x 21e0b201 / vLLM 5dad364d, the k81 trace and the dgx-01 preparation cache.

The serving plan for prefill is the capacity variant at max_tokens 8192 (`B12xExperts.get_b12x_preparation_units`). One plan serves every chunk size from 1264 to 8192. Decode sizes have their own exact variants.

That capacity variant runs the monolithic `MoEDynamicKernel` with a 32-row M tile:

- The k81 packed-A scratch is [98272, 1280]. That equals (8192 x 10 / 32 + 511) x 32, so the tile is 32 rows (M128 would give 147328 rows).
- The preparation cache contains exactly one dynamic M32 record measured over 4 candidates.
- The autotuner measured on the cyclic route pattern, which puts exactly 160 rows on every expert.
- The default heuristic would pick M128 at that capacity (96 or more rows per expert).

The kernel works per (expert, M tile, N128 slice of the 640-wide intermediate):

- FC1 runs for that slice.
- The FC2 partial, a full 2560-wide BF16 row, is scatter-added into the output with atomics.

With 5 slices and 10 routes, every token row takes 50 atomic partial adds per layer. That is 256 KB of atomic traffic per token per layer, 12 MB per token over 48 layers. M32 tiles also restream each expert's weight slices once per 32-row tile, about 4 times per expert at 6048 tokens; L2 absorbs part of that.

My working hypothesis is that atomics and restreaming make up most of the 58.9 us/token marginal. That works out to about 80 TFLOPS on the per-token part, below the 121-127 TFLOPS the MXFP8 dense GEMMs reach on the same GPU in k81 and k88. The hypothesis is unverified. k91 tests it directly with M64 and M128.

b12x already has a large-M lowering that serving never selects, the NVFP4 split-materialized M128 path:

- The dynamic kernel runs only routing and input quantization.
- `Nvfp4MaterializedPhase1Kernel` and `Nvfp4MaterializedPhase2Kernel` then run FC1 (with SiLU and requant) and full-K FC2 against the same swizzled weight tensors. That means one output partial per route instead of five.

It is eligible only when every expert shares one FC1 input scale. This checkpoint has per-expert FC1 input scales, with a 2.0x to 4.2x spread within a layer (layers 0, 5, 23, 47), so the path never engages. Every dynamic record in the preparation cache has `nvfp4_share_input: false`.

`VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13` makes it eligible with no code change. Two caveats:

- That env applies at every M, so it changes decode numerics too.
- The b12x source records a comparable W4A8 split losing 16% to 36% on GB10.

## Options

All rows assume the weights stay where b12x prepared them:

- Packed E2M1 payload [E, 2I, H/2] and [E, H, I/2], unchanged from the checkpoint.
- E4M3 block scales per 16, in the 128x4 swizzle.
- FC1 halves flipped in place by b12x from vLLM's [gate; up] to [up; gate] at first use.

A second copy of the 68 GB of experts does not fit, so the "Extra memory" column only counts scratch.

The ms/layer column is for M 6048. "#5863 table" means FlashInfer PR #5863's published DGX Spark numbers for this exact layer (H 2560, I 640, 512 experts, top-10, uniform routing, L2 flush, CUDA graphs). I interpolated linearly between its 4096 and 8192 rows. Its b12x column at 8192 (15.3 ms) matches the k81 fit at 8192 (5.2 + 10.1 = 15.2 ms), so its absolute numbers transfer.

| Backend | SM121 | Layout vs b12x storage | Extra memory | ms/layer at 6048 (TFLOPS) | Effort |
|---|---|---|---|---|---|
| b12x dynamic M32, shipped | yes | same | none | 12.6 (47), k81 | none |
| b12x dynamic M64 / M128, per-expert scales | yes | same | none | unknown; #5863 table "b12x forced" 12.2 (49) | small: capacity-band tile in b12x, like the dense M-bucket fix |
| b12x split-materialized M128, layer-max FC1 scale | yes | same | scratch only | unknown | zero code via env (numerics change at all M) or small in b12x (shared scale only in the capacity plan) |
| FlashInfer #5863 canonical NVFP4 W4A4 (open, CuTe DSL) | yes, measured on GB10 | payload and scales byte-identical; FC1 order [gate; up] vs our [up; gate] | workspace per token bucket only | 10.7 (56), #5863 table; 1.18x b12x at 8192, 1.09x at 4096 | medium (below) |
| FlashInfer CUTLASS fused MoE, NVFP4 SM120 path (FlashInfer 0.7.0 in the image, imports) | yes | matches b12x after the flip ([up; gate], swizzled); one FC1 input scale | workspace | 12.8 (46), #5863 table | small, but no gain expected |
| vLLM CUTLASS `cutlass_moe_fp4` (SM120 grouped GEMM 128x128x128) | code path (`ENABLE_NVFP4_SM120`); not checked in the image | payload and scales same; expects [gate; up], conflicts with the b12x flip | permuted BF16 copies of the routed rows, about 1 GB of traffic per layer at 6048 | no GB10 data; unfused row shuffles in and out | medium, low expectation |
| Marlin W4A16 MoE (BF16 math) | yes | no: Marlin tile repack plus E4M3 to S0E5M3 scales | a second 68 GB copy, does not fit | BF16 mm on GB10 measures 79-84 TFLOPS, so not above a good W4A4 at large M | out for a split dispatch; r12 replaced b12x with it and decode lost 6-11% per step |
| b12x W4A16 (AUTO A16 consumer, same storage) | yes | same | none | BF16 math | out at large M |
| FlashInfer b12x backend (`flashinfer_b12x`) | yes | its own preparation from BF16 | a second copy | #5863 table 12.7 at 6048 | out |

What the #5863 row needs before it could serve:

- FC1 half order. GEMM1 addresses gate rows at `n0` and up rows at `I + n0`. With our in-place [up; gate] storage that is a swap of the two row offsets in the B-operand and scale-factor loads (two lines in a vendored copy). The alternative is keeping b12x off the flip, which b12x does not support for gated NVFP4.
- FC2 activation scale. The kernel takes one scalar per GEMM. The FC1 scalar is the same layer-max change as the b12x split path. FC2 is worse: the down-proj input scale spreads 8x to 1000x across experts (1000x at layer 47). The GEMM1 epilogue already knows the expert, so reading `a2[e]` instead of `a2[0]` is a small patch. Without it, layer-max FC2 is a numerics risk.
- Token buckets. Kernels are specialized per exact token count. Serving needs bucket padding with expert id -1 (the kernel drops those slots) and warmup at boot. Prefill is not graphed, so capture is not a constraint.
- Dispatch. In `B12xExperts.apply`, route M >= threshold to the vendored kernel and keep b12x below it. The threshold env has to be declared in `vllm/envs.py` compile factors (the b12x plan-population rule).
- DSL. The PR measured with CuTe DSL 4.7.1; the image has 4.7.0. The file imports cleanly in the shipped image (CPU check, 2026-10-11). GPU compile is untested until k91.

## Recommendation

1. Run k91 first. It tells me which of three paths to build, in order of cost:
   - If b12x M128 or M64 with per-expert scales beats the M32 anchor at 3024 and 6048, the fix is a capacity-band tile policy in b12x. That is the smallest change: decode and numerics are untouched, and no new backend is involved.
   - If the split-materialized M128 variant wins and passes numerics, enable the shared FC1 scale only for the capacity plan in b12x, not through the env. Decode keeps its per-expert scales.
   - If #5863 beats every b12x variant by at least 10% at 6048 and passes numerics, vendor it as a prefill-only path with the half-order and per-expert `a2` patches and an M threshold (`VLLM_B12X_MOE_PREFILL_MIN_TOKENS`), the same pattern as `VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS`.
2. Expected size: even the #5863 path saves about 4.3 ms per layer over the four chunks of a 16K prompt, about 0.2 s of 7.93 s (2.6%), using the #5863 table ratios. The step-count lever (2 steps instead of 4, after k90) is about twice that, and it adds to whichever MoE backend wins.
3. Any winner then goes through the usual 1x A/B: cold TTFT 16K/64K/128K plus tg512 c1/c8, where low concurrency must not regress. After that comes the quality gate (IFEval, fidelity), because every layer-max scale variant changes numerics.

## Micro-benchmark plan: k91-moe-prefill-probe

Job: dgx-01 `~/GEN-AI/backlog/k91` (run.sh, moe_probe.py, vendor/fi5863_moe_nvfp4_w4a4.py, verbatim from #5863 head a0997210). Dry-run exits 0; the dry-run includes a CPU-only API and checkpoint-key check inside the shipped image.

The job:

- takes the gpu-lock;
- stops the 2x (it spans both Sparks);
- runs one container of the shipped 1x image (`tp1-d3-hf-20261010-21e0b201-5dad364d-warm`) on dgx-01, as container `k91-probe` with `--init`, a 28 min timeout, 7.5 min per b12x variant and 5 min for fi5863, and a 26 min budget, then removes the container;
- restores the 2x with a pong.

ETA is about 38 min.

Inputs:

- Checkpoint NVFP4 experts of the 1x served revision. Layer 5 is timed. Layers 5 and 47 are used for numerics; 47 has the widest down-proj scale spread.
- BF16 activations at the layer's calibrated FC1 range: sigma = median input_scale x 2688 / 5.
- Two synthetic routings, because no router captures exist for this model. Uniform is top-10 of N(0,1) logits. Skew uses Zipf 0.3 expert popularity, so the hottest expert sees about 4.6x the mean. The report prints active experts and max/mean rows per expert for each M.

Variants (each in its own process):

| Variant | What it isolates |
|---|---|
| b12x-m32 | serving anchor: capacity plan 8192, tile 32x128, per-expert scales |
| b12x-m128, b12x-m64 | tile choice at prefill M, numerics unchanged |
| b12x-m128-l1max-split | layer-max FC1 scale, so shared input and the split-materialized phase kernels |
| b12x-m128-l1max-mono | the same with the split disabled (`B12X_NVFP4_DYNAMIC_MATERIALIZED=0`), separates sharing from the split |
| fi5863 | #5863 `run_moe_w4a4` on the [gate; up] order with scalar layer-max FC1/FC2 scales |

Measurement:

- M 1264 / 3024 / 6048 (the 16K chunks) plus 8192 for the #5863 table.
- Eager, matching serving. CUDA events, a 256 MB L2 flush before every call, median of 15 after 3 warmups.
- No clock locking; the comparison is ratios within one run.

Gates:

- Anchor: b12x-m32 has to land within 10% of the k81 per-layer time at 6048 (12.6 ms). Otherwise the report marks the anchor OFF and I do not publish ratios.
- Numerics: rows 0..255 at M 6048 against an FP32 dequantized-weight reference. A variant passes when its error is at most 1.10 x the anchor's + 0.005 in every cell (2 layers x 2 routings) where the anchor has a value. A failing variant, or one with no judged cell, gets no speed claim.

Output: `k91.txt`, with a numerics table, the anchor check, speed tables per routing (ms, TFLOPS, ratio vs the anchor) and the selected b12x configs. The raw TSV and per-variant logs go in the results dir.

To arm (not done): `JOBS=k91 GRACE0=60 setsid nohup bash ~/GEN-AI/backlog/runner.sh > ~/GEN-AI/backlog/runner-k91.nohup 2>&1 < /dev/null &` on dgx-01, after the k90 queue.
