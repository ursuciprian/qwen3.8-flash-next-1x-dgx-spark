# Routed-MoE prefill dispatch: options and the k91 result

Status: design note, 2026-10-11; updated 2026-10-15 with k96/k97/k99/k100/k101/k102 outcomes. Tracks #27. Keep the shipped MoE backend. Experiments: [#64](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/issues/64) (k96, k100), [#65](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/issues/65) (k101), [#66](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/issues/66) (k97, k102) and [TP=2 #174](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/issues/174) (k99).

## Where the time goes

The k81 profile ([results/k81-1x-prefill-profile-20261011-0852](../../results/k81-1x-prefill-profile-20261011-0852)) measured one cold 16K prompt on 1x v2.2.0:

- TTFT is 7.93 s, which is 484 us/token.
- The routed MoE (b12x NVFP4 W4A4) costs 119.5 us/token, 26% of the GPU time.
- Each prefill step costs 248 ms + 58.9 us/token, from a least-squares fit over the 6048 / 6048 / 3024 / 1264 chunks.

Per layer per chunk, that is 12.6 ms at 6048 tokens, 9.0 ms at 3024 and 6.7 ms at 1264.

The 248 ms is the expert weight stream: 48 layers x 1.42 GB = 68 GB per step at about 270 GB/s. Every grouped-GEMM backend pays it once per step, because at prefill sizes every expert has work. With top-10 of 512, the mean is 24.7 routed rows per expert at 1264 tokens and 118 at 6048. Skipping idle experts saves nothing here.

A 16K prompt takes four steps: 6048, 6048, 3024, 1264 tokens. The k88
9072-budget experiment also took four: 9072, 3024, 3024, 1264. It did not remove
an expert-weight pass; the earlier three-step description was incorrect. k97
and k102 logged the recurrent-state reuse boundary that forces the extra split.
12096 does remove one step, but k102s on the tile-fixed v2.3.0 still lost cold
16K prefill by 12.9 to 18.3%, and lost at 64K/128K too. Keep budget 8192.

k90's K6144/N2560 tile fix shipped in v2.3.0 after the k90b decode recheck;
it improved cold prefill by 1.6 to 2.3%. Fewer chunks alone do not establish a
speedup when their larger kernels cost more per token.

## What the shipped b12x path does at prefill

I inferred this from the code at b12x 21e0b201 / vLLM 5dad364d, the k81 profile and the dgx-01 preparation cache.

The serving plan for prefill is the capacity variant at max_tokens 8192 (`B12xExperts.get_b12x_preparation_units`). It runs the monolithic `MoEDynamicKernel` with a 32-row M tile, chosen by the autotuner on a cyclic route pattern. The kernel works per (expert, M tile, N128 slice of the 640-wide intermediate): FC1 for that slice, then the full 2560-wide BF16 FC2 partial is scatter-added into the output with atomics. With 5 slices and 10 routes, every token row takes 50 atomic partial adds per layer.

b12x also has an NVFP4 split-materialized M128 path that writes one FC2 partial per route instead of five. It engages only when every expert shares one FC1 input scale. This checkpoint has per-expert FC1 input scales (2.0x to 4.2x spread within a layer), so serving never selects it. `VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13` forces a shared scale at every M.

My hypothesis before k91 was that atomics and weight restreaming per 32-row tile made up most of the 58.9 us/token marginal. k91 refutes it: M64 only ties the shipped M32 tile, and M128 (fewer restreams) and the split path (a fifth of the atomics) are both slower.

## Options

All rows assume the weights stay where b12x prepared them (packed E2M1 payload, E4M3 block scales per 16 in the 128x4 swizzle, FC1 halves flipped to [up; gate]). A second copy of the 68 GB of experts does not fit.

"k91" is the measured speed ratio vs the shipped M32 tile (x = M32 time / variant time), layer 5, M 1264 to 8192, uniform and skew routing. Every k91 variant passed the numerics gate. The k91 anchor was OFF, so the ratios are indicative, not absolute numbers. "Expected" is what this note assumed before k91, from k81 or from FlashInfer PR #5863's published GB10 table.

| Backend | SM121 | Layout vs b12x storage | Extra memory | Expected ms/layer at 6048 | k91 | Effort |
|---|---|---|---|---|---|---|
| b12x dynamic M32, shipped | yes | same | none | 12.6, k81 | 1.00 (anchor) | none |
| b12x dynamic M64, per-expert scales | yes | same | none | 12.2, #5863 table "b12x forced" | 1.00-1.03 | small |
| b12x dynamic M128, per-expert scales | yes | same | none | same 12.2 as the M64 row | 0.81-0.87 | small |
| b12x split-materialized M128, layer-max FC1 scale | yes | same | scratch only | not estimated | 0.44-0.61 | env or small |
| b12x M128, layer-max FC1 scale, no split | yes | same | none | not estimated | 0.82-0.89 | env |
| FlashInfer #5863 canonical NVFP4 W4A4 | yes | same payload and scales; FC1 order [gate; up] | workspace per token bucket | 10.7, #5863 table | 1.08-1.13 | medium |
| FlashInfer CUTLASS fused MoE, NVFP4 SM120 | yes | same after the flip; one FC1 input scale | workspace | 12.8, #5863 table | not run | small, no gain expected |
| vLLM CUTLASS `cutlass_moe_fp4` | code path only | expects [gate; up] | permuted BF16 row copies | no GB10 data | not run | medium |
| Marlin W4A16 MoE | yes | repack plus scale conversion | second 68 GB copy, does not fit | BF16 math | not run | out |
| b12x W4A16 (same storage) | yes | same | none | BF16 math | not run | out at large M |
| FlashInfer b12x backend | yes | own preparation from BF16 | second copy | 12.7, #5863 table | not run | out |

Serving #5863 would still need a vendored copy with the FC1 half-order swap, per-expert FC2 activation scales (`a2[e]`), token-bucket padding with warmup, and an M threshold declared in `vllm/envs.py`.

## Recommendation (updated 2026-10-15)

1. Keep the shipped b12x prefill path. k91's alternatives did not justify
   integration. The newer FlashInfer #5168 path was subsequently tested on TP=2
   in k99: cold TTFT changed only -0.3 to -0.5%, within noise, while KV capacity
   fell 15% (about 4.4 GiB per rank). The in-place expert layout still needed
   substantial scale/workspace caches. Do not ship it on the operator gain alone.
2. Keep `max_num_batched_tokens=8192`. k102s tested 12096 with the intended
   chunk reduction and failed. Reopen only with a measured large-M kernel fix.
3. Stop treating the k90 HC candidate column as a speedup. The fair k96 probe
   used the served seed, identical inputs, and full-operation timing; no tested
   alternative cleared its numerical gate with a useful gain. The earlier
   apparent 2x comparison also omitted activation quantization on one side.
4. GDN views are the best measured candidate: k101s gained 3.2 to 4.8% cold-prefill
   throughput on 1x and reduced TP=2 TTFT by 2.2 to 2.5%, with essentially unchanged
   KV capacity. Before shipping 1x, resolve its -5.9% c1 decode reading on dgx-01
   (95% interval entirely negative), then run release quality and boot checks.
   The recorded noise-band `WIN` alone is insufficient to dismiss that signal.
5. The k100ab serving A/B is recorded in [#64](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/issues/64). The HC down-projection BF16
   path passes its operator gate and projects 3.8% less 1x TTFT / 4.7% less
   TP=2 TTFT. This is a different approach from k96.
   Tested b12x `a1ad1b69` starts at 1264 rows, despite the issue text saying 1024.
6. Profile QSA at 128K and HC combine/norm after qualifying the candidates.
   These remain separate opportunities; neither is a demonstrated new speedup.

Results and the next checks: k96 and k100 in [#64](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/issues/64), k101 in
[#65](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/issues/65), k97 and k102 in [#66](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/issues/66), k99 in
[TP=2 #174](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/issues/174). The table above keeps the original k91
measurements as historical operator evidence.

## How k91 measured

Results: [results/k91-moe-prefill-probe-20261011-2125](../../results/k91-moe-prefill-probe-20261011-2125).

- One container of the shipped 1x image on one Spark. Checkpoint NVFP4 experts; layer 5 timed, layers 5 and 47 for numerics.
- BF16 activations at the layer's calibrated FC1 range. Two synthetic routings, because no router captures exist for this model: uniform (top-10 of N(0,1) logits) and skew (Zipf 0.3 expert popularity).
- M 1264 / 3024 / 6048 / 8192. Eager, CUDA events, a 256 MB L2 flush before every call, median of 15 after 3 warmups. No clock locking; the comparison is ratios within one run.
- Numerics gate: rows 0..255 at M 6048 against an FP32 dequantized-weight reference. A variant passes when its error is at most 1.10 x the anchor's + 0.005 in every cell (2 layers x 2 routings). All six passed.
- Anchor gate: b12x M32 has to land within 10% of the k81 serving time per layer at 6048. It landed 14% below k81 serving at M 6048 (synthetic routing), so the anchor is OFF: I report ratios as indicative and take no absolute ms from k91.
