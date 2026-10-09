# Benchmarks: full tables

Release names follow [VERSIONS.md](../VERSIONS.md). Section headings keep the old build names so existing links keep working; each build section opens with its release name. The current default is 1× v2.1.0 (old name v3e); every other section is history, kept for comparison. The README's capability table and charts read [`docs/data/capability.csv`](data/capability.csv).


Every single-Spark (TP=1) build, newest first, then the runs that compared the 1× and 2× setups side by side. The 2×
rows in those runs are kept for reference; the 2× builds have their own tables in
[qwen3.8-flash-next-dgx-spark-tp-2](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/blob/main/docs/BENCHMARKS.md).

## Single Spark v3e: retrained MTP drafter (2026-10-08)

Release 1× v2.1.0 (old name v3e), the current default.

v3e = v3d with the 24 dense BF16 `mtp.*` tensors retrained (#97 refit run 1). Checkpoint
`ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` @ `16c9bd54` (v3d: `244cb6fe`); only `model-00034-of-00036.safetensors`
differs. Training: 72 optimizer steps of 8192 anchors (0.66 epoch) on about 3.5M tokens of v3d's own outputs (1,466
training documents, 896,564 anchors; 70 held-out documents, 64,425 anchors), KL to the served target's top-20 at 6
chained draft depths, learning rate 2e-5 after a 20-step warm-up, fp8 drafter KV emulated as served. Drafter experts
and the 131,072-id draft head stay at the served NVFP4 values.

Offline acceptance per draft position on the held-out set (vLLM's cumulative rate), v3d → v3e:

| Category | T | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | tokens/step at 4 drafts |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| all | 0 | 0.878 → 0.905 | 0.750 → 0.797 | 0.635 → 0.699 | 0.534 → 0.614 | 0.449 → 0.542 | 0.377 → 0.480 | 3.797 → 4.015 |
| all | 1 | 0.853 → 0.876 | 0.680 → 0.713 | 0.514 → 0.551 | 0.372 → 0.409 | 0.260 → 0.294 | 0.177 → 0.206 | 3.419 → 3.548 |
| agentic | 0 | 0.824 → 0.860 | 0.660 → 0.716 | 0.527 → 0.590 | 0.420 → 0.488 | 0.340 → 0.410 | 0.278 → 0.351 | 3.431 → 3.655 |
| chat | 0 | 0.793 → 0.827 | 0.587 → 0.643 | 0.434 → 0.502 | 0.323 → 0.393 | 0.245 → 0.315 | 0.186 → 0.257 | 3.136 → 3.365 |
| code | 0 | 0.924 → 0.944 | 0.830 → 0.870 | 0.729 → 0.792 | 0.631 → 0.717 | 0.540 → 0.646 | 0.457 → 0.580 | 4.114 → 4.324 |
| math | 0 | 0.954 → 0.968 | 0.883 → 0.917 | 0.794 → 0.853 | 0.705 → 0.785 | 0.617 → 0.714 | 0.536 → 0.645 | 4.335 → 4.522 |
| tools | 0 | 0.897 → 0.924 | 0.775 → 0.831 | 0.669 → 0.735 | 0.574 → 0.655 | 0.492 → 0.586 | 0.429 → 0.531 | 3.914 → 4.145 |

Live screen (k56): Thunderdome, one arm per Spark (dgx-01, dgx-02), v3d control and v3e arm booted alternately, 2
passes, T=0 probe cells plus llama-benchy (T=1, 2 runs). Noise band = the cell's control boot-to-boot spread, at least
1%. Acceptance per draft position, T=0 probe cells pooled:

| Spark | pos 1 | pos 2 | pos 3 | pos 4 |
|---|:---:|:---:|:---:|:---:|
| dgx-01 | 0.819 → 0.856 | 0.661 → 0.713 | 0.535 → 0.593 | 0.431 → 0.489 |
| dgx-02 | 0.818 → 0.854 | 0.666 → 0.708 | 0.539 → 0.589 | 0.438 → 0.488 |

| Cell | dgx-01 | dgx-02 |
|---|:---:|:---:|
| probe fresh c1 | +5.66% (4.73%) | +2.84% (8.54%) |
| probe fresh c4 | +4.88% (3.05%) | +7.90% (2.82%) |
| probe fresh c8 | +6.79% (3.56%) | +8.08% (2.61%) |
| probe 16K c4 | +5.47% (3.65%) | +5.74% (3.23%) |
| probe counting c8 | +1.19% (1.00%) | −0.11% (2.77%) |
| 16K c8 wall time | +2.86% (1.00%) | +2.04% (1.00%) |
| llama-benchy pp2048 c1 | +0.01% (1.00%), 1819.0 → 1819.2 t/s | −0.89% (2.02%), 1833.2 → 1816.9 t/s |
| llama-benchy tg512 c1 | +4.31% (7.55%), 53.2 → 55.5 t/s | +1.60% (10.26%), 58.1 → 59.0 t/s |
| llama-benchy tg512 c8 | +3.68% (4.85%), 124.9 → 129.5 t/s | +1.20% (3.58%), 130.7 → 132.3 t/s |

Verdict PROMOTE on both Sparks. Gate (both Sparks): hardmode 91, TC-45 100, fidelity 20/20 at 8k/32k/64k/128k
(~245k seeds: dgx-01 one seed 19/20, 119/120 overall; dgx-02 120/120), stragglers c8/c12/c16 with 0 preemptions, min
MemAvailable 13.92 GiB. Jev (TypeSafe System One) on the same numbers: ship, confidence 0.97
([`jev-ship.json`](../results/thunderdome-k56-20261007/jev-ship.json)). Raw files:
[`results/thunderdome-k56-20261007/`](../results/thunderdome-k56-20261007/).

Shipped image check (hfship, one boot on dgx-01 from the HF cache with the seed entries removed): seed HIT, drafter
REFIT, fresh c4 acceptance 0.846 / 0.691 / 0.566 / 0.462. Its llama-benchy coding grid gives tg512 56.9 t/s at c1 and
132.9 t/s at c8 (depth 0). Raw files: [`results/tp1-v3e-hf-20261008/`](../results/tp1-v3e-hf-20261008/).

## Single Spark v3d (2026-10-05)

Release 1× v2.0.0 (old name v3d), the default from 2026-10-06 to 2026-10-08.

Recipe `qwen3.8-flash-next-1x-dgx-spark` (one GB10, TP=1), image
`ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3d-hf-20261005-21e0b201-5dad364d-warm`
(digest `sha256:81ac7975869814102843b4ca58e5ea219c3e938518b8f87d1a6fbb6edd89ede2`), checkpoint
[`ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE`](https://huggingface.co/ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE)
@ `244cb6fe` (weights identical to the first upload `f35e321b`; only the model card changed).

The checkpoint is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1` with one change: the GDN `in_proj_qkv`,
`in_proj_z` and `out_proj` weights of all 36 GDN layers are requantized from the BF16 base to weight-only NVFP4, with a
per-block MSE scale search. Every other tensor is byte-identical to the base.

v3d = v3c + `VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS=41`:

- Each of those layers also holds the base checkpoint's MXFP8 weights, read from shard 35 of `7c4f1bc1`. The recipe
  fetches that shard before the server starts if the HF cache does not have it.
- Calls of 41 or more rows (prefill) use the MXFP8 copy; decode steps (at most 8 × 5 = 40 rows) use NVFP4.
- The copy costs about 1.1 GiB.

Versions: vLLM `exp/v3d` (= `exp/v3c` 50330171 + the dispatch), b12x 21e0b201.

The bench below booted the same checkpoint files from a local snapshot path, sha256 identical to the HF repo for all
49 checkpoint files. It used image `tp1-v3d-20261005-21e0b201-5dad364d-warm` (digest `sha256:32012ffd…`).

The b12x plan and the torch compile key include the model path, so the published image is seeded from boots of the
recipe as committed, at HF snapshot 244cb6fe:

- The cold boot loaded the re-keyed plan, compiled without autotuning and took 337 s.
- A boot of the published image, with those cache entries removed from the runtime cache, loaded them from the image
  and took 171 s. All 72 MXFP8 copies loaded.
- Quick llama-benchy cells on that boot (one run of 3, T=1): pp2048 c1 1,757, tg512 c1 55.6 ± 3.4, c8 125.5 ± 6.1.
  Accept/draft was 0.55–0.56 (0.58–0.63 in the grid below). Single cells of this grid vary by up to ~10% between
  runs. A CPU requant job ran on the other Spark at the same time.

Files: [`hf-seed-check/`](../results/tp1-v3d-20261005/hf-seed-check/). The shard-35 fetch was checked separately on
an empty HF cache inside the image, and it downloaded the two files byte-identical to `7c4f1bc1`.

Raw files: [`results/tp1-v3d-20261005/`](../results/tp1-v3d-20261005/) (gate, screen) and
[`bench/`](../results/tp1-v3d-20261005/bench/) (every table below; one boot on dgx-01).

**Coding**: llama-benchy `--prompt-mode task`, 2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 /
top-p 0.95 / top-k 20, prefix caching, 3 runs, one boot. Total tok/s, mean ± sd over runs.

| depth | test | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| 0 | pp2048 | 1788 ± 5 | 1976 ± 100 | 2095 ± 29 | 2202 ± 11 |
| 0 | tg512 | 59.8 ± 4.6 | 80.2 ± 2.7 | 111.4 ± 0.8 | 139.5 ± 2.9 |
| 16k | ctx_pp (fill) | 2088 ± 50 | 2075 ± 55 | 2161 ± 22 | 2174 ± 17 |
| 16k | pp2048 | 1056 ± 10 | 1142 ± 3 | 1178 ± 46 | 1262 ± 0 |
| 16k | tg512 | 62.6 ± 7.4 | 80.5 ± 5.6 | 101.4 ± 5.9 | 111.0 ± 1.2 |

Tokens per step 3.1–3.5 (accept/draft 0.53–0.63). Against the v3c grid: tg512 c1 51.8 → 59.8, c4 95.3 → 111.4,
c8 125.1 → 139.5. Prefill cells are 0–5% lower than in the v3c grid; single cells of this grid vary by up to ~10%
between runs, and the paired screen measured pp2048 c1 at −0.9% (noise 1.0%).

**High-acceptance**: counting (T=0, thinking off, 5 rounds per level, every round saved) and copy-heavy (3 rounds per
task count, low effort). The v3c copy-heavy row was rerun in the same window on the other Spark, with the shipped v3c
recipe.

| workload | c1 | c2 | c4 | c8 | tokens/step |
|---|---|---|---|---|---|
| counting, max of 5 rounds | 84.8 | 154.9 | 239.8 | 377.8 | 4.82–5.00 per round |
| counting, median of 5 rounds | 83.0 | 145.1 | 235.3 | 368.3 | |
| copy-heavy, max of 3 rounds | 80.8 | 132.0 | 188.5 | 281.5 | 4.90–4.96 |
| copy-heavy, v3c same window | 74.8 | 122.3 | 189.6 | 268.4 | 4.91–4.96 |

Copy-heavy at 3/5/6/7 tasks: v3d 156.3 / 211.0 / 224.9 / 258.7, v3c 143.8 / 203.2 / 218.5 / 246.0.

**llm-inference-bench** (decode 30 s per cell, server default sampling), aggregate tok/s (tokens per step):

| ctx | c1 | c4 | c8 |
|---|---|---|---|
| 0 | 47.4 (2.63) | 116.6 (2.99) | 173.4 (3.14) |
| 16K | 51.2 (2.85) | 112.6 (2.95) | 191.5 (3.45) |
| 64K | 56.9 (3.21) | 111.8 (2.89) | 179.5 (3.27) |

Standalone prefill 8K / 16K / 32K / 64K / 128K: 2,137 / 2,182 / 2,147 / 2,066 / 1,880 tok/s, 0.9–1.2% below v3c
(2,157 / 2,209 / 2,174 / 2,085 / 1,899). hotel-lights x8: 6/8. The KV pool is 14 GiB as in v3c, 993,754 tokens. Lowest
MemAvailable 13.30 GiB over the run, with 0 preemptions.

**Screen vs v3c** (one Spark, boots in the order v3c, v3d, v3d, v3c, T=0 probes; noise is the cell's own band).
Report: [`screen-v3d-c41-vs-v3c.txt`](../results/tp1-v3d-20261005/screen-v3d-c41-vs-v3c.txt).

| cell | change | noise |
|---|---|---|
| fresh c1 | +7.9% | 8.1% |
| fresh c4 | +2.2% | 3.0% |
| fresh c8 | +4.4% (CI +2.5, +6.3) | 1.9% |
| 16K c4 | +5.9% (CI +2.9, +8.8) | 3.0% |
| counting c8 | +7.8% (CI +6.2, +9.3) | 2.1% |
| 16K c8, wall time per run | 127 / 127 s -> 126 / 126 s | |
| pp2048 c1 | -0.9% | 1.0% |
| tg512 c1 / c8 | 50.0 -> 59.7 (+19.3%) / +3.9% | 4.3% / 5.1% |

Acceptance per draft position: v3c 0.820 / 0.667 / 0.544 / 0.442, v3d 0.816 / 0.667 / 0.540 / 0.440. The weights
change, so there is no output identity check. Logprobs vs v3c: mean |Δ| 0.034–0.048, self-noise 0.033–0.042.

A second arm on the other Spark served calls of up to 127 rows from NVFP4 (cutoff 128). It lost 2.3% on pp2048
(noise 1.0%) and was dropped ([report](../results/tp1-v3d-20261005/screen-v3d-c128-vs-v3c.txt)).

**Quality gate**:

- hardmode 91/100 (v3c 93; run-to-run band 86-93), TC-45 100/100.
- Fidelity 20/20 at 8k/32k/64k/128k, plus 128k seeds 11 and 13 at 20/20.
- Batch stragglers c8/c12/c16: 0 preemptions, 3.98-3.99 accepted per 4 drafts.
- Min MemAvailable 14.51 GiB (dgx-01) / 14.04 GiB (dgx-02).

Files: `gate-*` in the results directory.

## Single Spark v3c (2026-10-05)

Release 1× v1.3.0 (old name v3c), history.

Recipe `qwen3.8-flash-next-1x-dgx-spark` (one GB10, TP=1), image
`ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3c-20261005-21e0b201-50330171-warm`
(digest `sha256:ba140406cabf0fbbbd13d0f605c42881c2442079619aa2fa7297eed2e0e31bff`), checkpoint revision `7c4f1bc1`.
v3c = v3b + `VLLM_GDN_SHARED_PREFILL_STAGING=1` (one GDN prefill staging buffer for all 36 GDN layers, frees ~8.8 GiB)
+ `VLLM_GDN_COMPACT_RECORDS=1` (MTP draft-step GDN records in a 4.3 MiB per-layer side buffer; a request holds 37 GDN
state blocks instead of 185) + KV pool 14 GiB (993,754 tokens; v3b 6 GiB, 379,362 tokens). vLLM `exp/v3c` 50330171,
b12x 21e0b201. The image carries the torch compile cache for the compact-records compile key; a boot from the published
image with those cache entries removed loaded them from the image (186 s to healthy).
Raw files: [`results/tp1-v3c-20261005/`](../results/tp1-v3c-20261005/) (gate, screen) and
[`bench/`](../results/tp1-v3c-20261005/bench/) (every table below; one boot on dgx-01).

**Coding**: llama-benchy `--prompt-mode task`, 2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 /
top-p 0.95 / top-k 20, prefix caching, 3 runs, one boot. Total tok/s, mean ± sd over runs.

| depth | test | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| 0 | pp2048 | 1799 ± 7 | 2034 ± 87 | 2097 ± 11 | 2197 ± 23 |
| 0 | tg512 | 51.8 ± 1.0 | 80.3 ± 2.8 | 95.3 ± 1.9 | 125.1 ± 13.5 |
| 16k | ctx_pp (fill) | 2142 ± 12 | 2175 ± 11 | 2229 ± 5 | 2215 ± 0 |
| 16k | pp2048 | 1065 ± 9 | 1145 ± 3 | 1238 ± 2 | 1276 ± 1 |
| 16k | tg512 | 52.4 ± 7.7 | 78.7 ± 6.7 | 106.4 ± 1.3 | 109.3 ± 4.3 |

16k c8: 109.3 tok/s (v3b 22.1, v3a 19.7). The 14 GiB pool holds all 8 requests, so none wait for KV space.
Tokens per step 3.2–3.4 (accept/draft 0.54–0.60).

**High-acceptance**: counting (T=0, thinking off, 5 rounds per level, every round saved) and copy-heavy (3 rounds per
task count, low effort).

| workload | c1 | c2 | c4 | c8 | tokens/step |
|---|---|---|---|---|---|
| counting, max of 5 rounds | 76.9 | 142.4 | 243.8 | 360.1 | 4.86–5.00 per round |
| counting, median of 5 rounds | 76.4 | 135.1 | 232.5 | 357.8 | |
| copy-heavy, max of 3 rounds | 74.7 | 123.4 | 183.8 | 265.3 | 4.93–4.95 |

Copy-heavy at 3/5/6/7 tasks: 147.9 / 220.1 / 223.0 / 244.3.

**KV capacity**: per 3,024 tokens of context a request takes one KV page in each of 13 attention groups, plus 37 GDN
state pages with compact records (185 without). Method and serve-log token counts:
[`kv-capacity.txt`](../results/tp1-v3c-20261005/kv-capacity.txt).

| context + 512 out | v3b (6 GiB, 1,958 pages) | v3c (14 GiB, 4,568 pages) |
|---|---|---|
| 16K | 7.4 requests | 39.7 |
| 64K | 4.2 | 14.1 |
| 128K | 2.6 | 7.5 |

`max_num_seqs` is 8. A 16 GiB pool (5,221 pages, 8.6 requests at 128K) was screened on the other Spark and ended
INCONCLUSIVE: 16K c4 -2.0% (noise 1.5%), lowest MemAvailable 12.6 GiB
([report](../results/tp1-v3c-20261005/screen-v3c-kv16-vs-v3b.txt)).

**Screen vs v3b** (one Spark, boots in the order v3b, v3c, v3c, v3b, T=0 probes; noise is the cell's own band).
Report: [`screen-v3c-kv14-vs-v3b.txt`](../results/tp1-v3c-20261005/screen-v3c-kv14-vs-v3b.txt).

| cell | change | noise |
|---|---|---|
| fresh c1 | +0.8% | 6.5% |
| fresh c4 | +2.5% (CI +0.9, +4.1) | 1.6% |
| fresh c8 | -0.5% | 1.9% |
| 16K c4 | +2.4% | 3.6% |
| counting c8 | +3.1% (CI +2.3, +4.1) | 1.0% |
| 16K c8, wall time per run | 347 / 370 s -> 125 / 126 s | |
| pp2048 c1 | +0.7% | 3.3% |
| tg512 c1 / c8 | -0.3% / +2.2% | 9.8% / 7.8% |

Acceptance per draft position: v3b 0.818 / 0.667 / 0.545 / 0.447, v3c 0.818 / 0.667 / 0.542 / 0.446. T=0 output
identity check passed. Logprobs vs v3b: mean |Δ| 0.033–0.036, self-noise 0.036–0.041.

**Quality gate**: hardmode 93/100 (v3b 91; run-to-run band 86-93), TC-45 100/100, fidelity 20/20 at
8k/32k/64k/128k plus 128k seeds 11 and 13 at 20/20, batch stragglers c8/c12/c16 with 0 preemptions (3.98-3.99 accepted
per 4 drafts), min MemAvailable 15.62 GiB (dgx-01) / 15.01 GiB (dgx-02). Files: `gate-*` in the results directory.

## Single Spark v3b (2026-10-04)

Release 1× v1.2.0 (old name v3b), history.

Recipe `qwen3.8-flash-next-1x-dgx-spark` (one GB10, TP=1), image
`ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3b-20261004-5bf24021-0632e506-warm`
(digest `sha256:7308411dca81d1454cfc84725f87c9db80a6963ad7d4568a0255a046d51b90fa`), checkpoint revision `7c4f1bc1`.
v3b = v3a + `VLLM_PLE_MMAP_PREFILL_WILLNEED=1`: prefill-sized PLE table gathers get read-ahead before the copy, so
rows that left the page cache are no longer read back one page fault at a time. Decode is unchanged.
Raw files: [`results/tp1-v3b-20261004/`](../results/tp1-v3b-20261004/).

**Coding**: llama-benchy `--prompt-mode task`, 2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 /
top-p 0.95 / top-k 20, prefix caching, 3 runs, one boot. Total tok/s, mean ± sd over runs.

| depth | test | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| 0 | pp2048 | 1767 ± 29 | 2037 ± 14 | 2063 ± 21 | 2156 ± 45 |
| 0 | tg512 | 55.0 ± 3.1 | 76.5 ± 9.1 | 99.8 ± 2.9 | 130.2 ± 9.1 |
| 16k | ctx_pp (fill) | 2103 ± 40 | 2153 ± 14 | 2198 ± 0 | 1984 ± 42 |
| 16k | pp2048 | 1045 ± 15 | 1130 ± 9 | 1219 ± 1 | 119 ± 4 |
| 16k | tg512 | 54.6 ± 4.2 | 76.3 ± 8.7 | 99.0 ± 1.9 | 22.1 ± 2.0 |

16k c8: the 6 GiB KV pool fills and requests are deferred (same in v3a).

**A/B vs v3a settings** (same image, knob off vs on; 6 fresh boots per arm over 2 Sparks in ABBA order, llama-benchy
T=0, 3 runs per cell). Total tok/s, mean over boots; noise is the pooled run-to-run spread.
Report: [`ab-report-v3b-vs-v3a.txt`](../results/tp1-v3b-20261004/ab-report-v3b-vs-v3a.txt).

| cell | v3a | v3b | change | noise |
|---|---|---|---|---|
| pp2048 c1 | 1163.7 | 1747.7 | +50.2% | 2.3% |
| pp2048 c4 | 1724.8 | 2058.4 | +19.3% | 1.7% |
| pp2048 c8 | 1963.1 | 2144.8 | +9.3% | 2.4% |
| 16k fill c1 | 2000.8 | 2093.0 | +4.6% | 1.5% |
| 16k fill c2-c8 | | | -0.0 to +1.8% | 1.5-1.9% |
| tg512 c1-c8, depth 0 and 16k | | | -3.1 to +1.9% | 2.5-14% |

pp2048 c1 within-node sd over boots fell from 32.0 to 14.9 tok/s. The prefill-sized gathers in that cell took 33-34 s in
total per run before and 17-18 s after.

**Quality gate**: hardmode 91/100 (v3a 92; run-to-run band 86-93), TC-45 100/100, fidelity 20/20 at
8k/32k/64k/128k plus 128k seeds 11 and 13 at 20/20, batch stragglers c8-c16 with 0 preemptions (3.98-3.99 accepted per
draft), min MemAvailable 14.01 GiB.

## Single Spark v3a (2026-10-04)

Release 1× v1.1.0 (old name v3a), history.

Recipe `qwen3.8-flash-next-1x-dgx-spark` (one GB10, TP=1), image
`ghcr.io/ursuciprian/spark-vllm-b12x:tp1-v3a-20261004-5bf24021-7fa812b3-warm`, checkpoint revision `7c4f1bc1`.
v3a = v2 + `VLLM_PLE_MMAP_KEEPALIVE_MS=50` (keeps the NVMe drive out of its power-saving state between decode steps).
Raw files: [`results/tp1-v3a-20261004/`](../results/tp1-v3a-20261004/).

**Coding**: llama-benchy `--prompt-mode task`, 2048 new prompt tokens, up to 512 out, thinking on, temperature 1.0 /
top-p 0.95 / top-k 20, prefix caching, 3 runs, one boot. Total tok/s, mean ± sd over runs.

| depth | test | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| 0 | pp2048 | 1201 ± 20 | 1583 ± 155 | 1666 ± 96 | 1947 ± 59 |
| 0 | tg512 | 50.0 ± 1.3 | 73.3 ± 7.8 | 103.7 ± 2.4 | 117.6 ± 4.8 |
| 16k | pp2048 | 1010 ± 40 | 1116 ± 30 | 1231 ± 3 | 99 ± 20 |
| 16k | tg512 | 51.2 ± 4.8 | 77.6 ± 10.9 | 102.5 ± 3.4 | 19.7 ± 2.6 |

16k c8: the 6 GiB KV pool fills and requests are deferred (same in v2); a fix is in progress.

**Paired A/B vs v2** (ABBA, 4 passes over 2 Sparks, temperature 0, same prompts per pair).
Change in % with 95% CI; negative step time is faster.

| cell | n | step time | tok/s |
|---|---|---|---|
| fresh c1 | 32 | -0.7 [-1.0, -0.5] | +2.3 [+0.0, +4.1] |
| fresh c2 | 32 | -0.6 [-1.0, -0.1] | +1.5 [-0.6, +3.5] |
| fresh c4 | 48 | -8.4 [-8.8, -8.0] | +7.6 [+5.8, +9.4] |
| fresh c8 | 32 | -5.1 [-5.6, -4.7] | +4.9 [+3.1, +6.8] |
| 16k c1 | 32 | -0.9 [-1.3, -0.5] | +2.3 [+0.2, +4.4] |
| 16k c2 | 32 | -0.6 [-1.1, -0.1] | +2.8 [+1.2, +4.4] |
| 16k c4 | 48 | -8.0 [-8.5, -7.4] | +7.7 [+5.9, +9.4] |
| counting c1-c4 | 32-48 | -0.6 to +0.1, within CI | +0.0 to +0.6, within CI |
| counting c8 | 32 | -3.1 [-3.5, -2.8] | +3.6 [+3.0, +4.2] |

16k c8 (n=12) is not measurable in either arm (KV pool churn). MTP acceptance per draft position is unchanged in every
cell (e.g. fresh c4 0.80/0.63/0.49/0.39 in both).

**Quality gate**: hardmode 92/100, TC-45 100/100, fidelity 20/20 at 8k/32k/64k/128k plus 128k seeds 11 and 13 at 20/20,
batch stragglers c8-c16 with 0 preemptions (3.98-3.99 accepted per draft), min MemAvailable 13.97 GiB.

## llm-inference-bench, 2× b1.4 and 1× v3b / v3c (2026-10-05)

llm-inference-bench 0.7.6: decode 30 s per cell at c1/c4/c8 with 0 / 16K / 64K tokens of context in each prompt,
standalone cold prefill 8K-128K, hotel-lights (8 runs at c8), server default sampling. One boot per setup. Aggregate
decode tok/s, with tokens per step from the server's spec-decode counters.

| setup | ctx | c1 | c4 | c8 |
|---|---|---|---|---|
| 2× b1.4 | 0 | 64.8 (2.69) | 171.2 (2.79) | 248.6 (2.82) |
| 2× b1.4 | 16K | 73.3 (2.86) | 175.0 (2.92) | 254.9 (2.97) |
| 2× b1.4 | 64K | 77.8 (3.08) | 165.4 (2.85) | 257.4 (3.01) |
| 1× v3c | 0 | 43.7 (2.67) | 106.2 (2.87) | 165.7 (3.12) |
| 1× v3c | 16K | 43.0 (2.66) | 116.9 (3.16) | 165.6 (3.28) |
| 1× v3c | 64K | 43.3 (2.68) | 113.9 (3.12) | 160.4 (3.18) |
| 1× v3b | 0 | 46.9 (2.86) | 116.5 (3.08) | 168.4 (3.18) |
| 1× v3b | 16K | 45.3 (2.78) | 112.0 (3.07) | 174.8 (3.30) |
| 1× v3b | 64K | 44.3 (2.76) | 106.3 (2.94) | not run: 8 × 64K does not fit 379,362 tokens |

Tokens per step follow the sampled text and differ between runs; decode tok/s divided by tokens per step is within
3% between v3b and v3c in every c1 and c4 cell.

| standalone prefill, tok/s | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| 2× b1.4 | 2,857 | 2,931 | 2,829 | 2,664 | 2,384 |
| 1× v3b | 2,162 | 2,207 | 2,169 | 2,087 | 1,901 |
| 1× v3c | 2,157 | 2,209 | 2,174 | 2,085 | 1,899 |

hotel-lights x8: 2× b1.4 8/8, 1× v3b 7/8, 1× v3c 5/8. Of the three v3c misses, two gave no final number the scorer
could read and one gave 49 (expected 48); with 8 runs the difference from v3b is not significant (Fisher exact
p = 0.57). A 32-run rerun per recipe ([#87](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/issues/87))
gave v3c 21/32 and v3b 23/32 (Fisher exact p = 0.79), with every run ending on its own stop token
([`results/hotel-ab-20261005/runs.jsonl`](../results/hotel-ab-20261005/runs.jsonl)).

Counting sweep in the same run (5 rounds per level, every round saved), max / median of 5 rounds:

| setup | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|
| 2× b1.4 | 122.4 / 119.5 | 221.7 / 211.1 | 389.8 / 350.0 | 558.0 / 544.1 | 795.3 / 787.3 |
| 1× v3b | 77.1 / 76.4 | 138.9 / 133.7 | 243.6 / 220.5 | 348.7 / 344.9 | |
| 1× v3c | 76.9 / 76.4 | 142.4 / 135.1 | 243.8 / 232.5 | 360.1 / 357.8 | |

Raw files: [`results/lib-bench-20261005/`](../results/lib-bench-20261005/) and
[`results/tp1-v3c-20261005/bench/`](../results/tp1-v3c-20261005/bench/).

## Copy-heavy and counting refresh (2026-10-04)

2× Spark b1.4 (`b1.4-20261001-b7fbaf96-a7e649d8-warm`) and 1× Spark v3a (`tp1-v3a-20261004-5bf24021-7fa812b3-warm`) on
both Sparks, one boot each. Raw files: [`results/showcase-20261004/`](../results/showcase-20261004/)
(`A/` 2×, `B/dgx01/` and `B/dgx02/` 1×).

**Copy-heavy benchmark**: 1–8 concurrent copy tasks from a shared cached prefix, low reasoning effort, 1,500 tokens out,
3 rounds per task count; tok/s over the window where all tasks decode. Max of 3 rounds / median of 3 rounds.

| build | 1 | 2 | 4 | 8 | tokens/step |
|---|---|---|---|---|---|
| 2× b1.4 | 117.2 / 116.4 | 187.1 / 187.0 | 290.1 / 285.9 | 439.4 / 436.8 | 4.91–4.95 |
| 1× v3a, dgx-01 | 73.9 / 73.4 | 114.8 / 114.5 | 181.2 / 174.8 | 270.8 / 262.9 | 4.91–4.97 |
| 1× v3a, dgx-02 | 74.9 / 74.4 | 117.0 / 115.0 | 192.8 / 184.6 | 263.1 / 260.8 | 4.91–4.97 |

**Counting** ("list the numbers from 1 to 300", temperature 0, thinking off, 5 rounds). Aggregate tok/s; each file
stores the median of its 5 rounds only.

| build | c1 | c2 | c4 | c8 | c16 | tokens/step |
|---|---|---|---|---|---|---|
| 2× b1.4 | 119.5 | 216.6 | 362.9 | 541.2 | 770.1 | 4.95–4.97 |
| 1× v3a, dgx-01 | 75.0 | 133.3 | 220.4 | 348.9 | | 4.94–4.97 |
| 1× v3a, dgx-02 | 76.5 | 138.7 | 219.6 | 353.1 | | 4.93–4.98 |

The 2× counting run is below the max 2026-10-01 b1.4 run in every cell (120.3 / 366.5 / 541.3 / 779.7 at c1/c4/c8/c16),
so the README keeps the 2026-10-01 values.

**Default-prompt llama-benchy, 1× v3a on dgx-02**: default book prompt, pp2048 / tg128, prefix caching, 3 runs, total
tok/s mean ± sd. Accepted per draft 0.50–0.61 (3.0–3.4 tokens/step).

| depth | pp2048 c1 | pp2048 c2 | pp2048 c5 | tg128 c1 | tg128 c2 | tg128 c5 |
|---|---|---|---|---|---|---|
| 0 | 1160 ± 59 | 1490 ± 65 | 2264 ± 4 | 53.6 ± 2.4 | 73.4 ± 5.6 | 107.8 ± 2.1 |
| 4k | 1072 ± 58 | 1191 ± 71 | 1409 ± 3 | 49.6 ± 6.0 | 81.9 ± 4.8 | 99.4 ± 1.2 |
| 8k | 950 ± 3 | 1007 ± 38 | 1054 ± 1 | 51.1 ± 1.8 | 77.5 ± 3.4 | 74.7 ± 3.7 |
| 16k | 1095 ± 9 | 1146 ± 31 | 1267 ± 1 | 47.6 ± 2.5 | 76.9 ± 6.9 | 104.9 ± 5.0 |
| 32k | 787 ± 4 | 841 ± 1 | 882 ± 1 | 50.3 ± 2.9 | 73.7 ± 3.7 | 70.3 ± 1.2 |
| 64k | 774 ± 6 | 826 ± 4 | 50 ± 4 | 55.5 ± 2.7 | 73.3 ± 1.4 | 3.2 ± 0.3 |
| 100k | 1209 ± 7 | 1297 ± 5 | not run | 53.1 ± 1.5 | 67.4 ± 9.0 | not run |

64k c5: decode drops to ~3 tok/s; five ~68k-token contexts take most of the ~379k-token KV pool, the same kind of limit
as 16k c8 in the coding grid. 100k c5 and every depth at c10 beyond 8k were skipped (KV size) or cut by the 90-minute cap; the c10
rows that ran (max_num_seqs 8, so 2 requests queue) are in `B/benchy/g4.md.live.md`. This run predates v3b, whose
prefill read-ahead raises pp2048 c1.

## Coding probe, 36 prompts in four languages (2026-10-06)

Single-request decode speed on short coding requests: 12 Python and 8 each C++, Rust and Go (write a function, fix a
bug, refactor, write tests). One request at a time, up to 768 tokens out, each prompt sent once per setting:

- T=0, thinking off: temperature 0, `enable_thinking=false`.
- Server defaults: only model, messages and `max_tokens`, so the recipe's sampling and its `medium` thinking default
  apply. 34 of 36 requests (2×) and 33 of 36 (1×) stopped at 768 tokens, and about two thirds of the streamed chunks
  were reasoning, so these rows measure thinking plus the start of the answer.

Decode tok/s = (completion tokens − 1) / (time of last token − time of first token). Tokens/step comes from the
`vllm:spec_decode_*` counters around each request. Setups: 2× Spark b1.4 (the shipped recipe, dgx-01 + dgx-02) and
1× Spark v3d (the shipped recipe and published image `tp1-v3d-hf-20261005-21e0b201-5dad364d-warm`, alone on dgx-02).
Raw files and the probe script: [`results/coding-probe-k55-20261006/`](../results/coding-probe-k55-20261006/).

Decode tok/s, median of the prompts (max in brackets):

| Prompts | 2× b1.4, T=0, thinking off | 1× v3d, T=0, thinking off | 2× b1.4, server defaults | 1× v3d, server defaults |
|---|:---:|:---:|:---:|:---:|
| 12 Python | 103.1 (109.9) | 73.2 (77.0) | 87.7 (97.2) | 60.0 (65.8) |
| 8 C++ | 113.5 (120.6) | 75.6 (81.8) | 88.3 (93.8) | 62.1 (67.2) |
| 8 Rust | 108.5 (115.7) | 76.3 (79.7) | 86.3 (90.7) | 58.1 (66.3) |
| 8 Go | 104.8 (115.5) | 71.6 (77.4) | 85.3 (93.2) | 59.1 (64.2) |
| All 36 | 106.2 (120.6) | 72.9 (81.8) | 87.5 (97.2) | 59.7 (67.2) |
| Tokens/step, all 36 | 4.01 | 3.96 | 3.44 | 3.38 |
| TTFT median, all 36 | 129 ms | 206 ms | 131 ms | 199 ms |

Each cell is one pass over its prompts on one boot. An earlier 1× v3d boot that ran only the 12 Python prompts gave
71.0 (T=0, thinking off) and 62.3 (server defaults), against 73.2 and 60.0 above.

## High concurrency, max_num_seqs 32 (2026-10-05)

Measured with `max_num_seqs` 32 and CUDA graphs up to 160 rows. The shipped recipes use 16 (2×) and 8 (1×), and the
quality gate was not run at this cap. One fresh boot per setup: 2× Spark b1.4 (image `b1.4-20261001-b7fbaf96-a7e649d8-warm`,
KV 3,615,479 tokens) and 1× Spark v3d (image `tp1-v3d-20261005-21e0b201-5dad364d-warm`, checkpoint files at a local
path, KV 993,754 tokens). Raw files: [`results/high-conc-k46b-20261005/`](../results/high-conc-k46b-20261005/)
([`summary.md`](../results/high-conc-k46b-20261005/summary.md) has every cell, including medians, TTFT and per-request speed).

**Counting** (T=0, thinking off, 320 tokens out, 5 rounds per level), aggregate tok/s, max of 5 rounds:

| setup | c1 | c8 | c16 | c32 |
|---|---|---|---|---|
| 2× b1.4 | 121.9 | 552.4 | 781.9 | 994.4 |
| 1× v3d | 84.6 | 366.2 | 522.2 | 675.9 |

**Copy-heavy** (low effort, 1,500 tokens out, 3 rounds per stream count), window tok/s, max of 3 rounds:

| setup | 1 | 8 | 16 | 32 |
|---|---|---|---|---|
| 2× b1.4 | 116.7 | 439.4 | 644.9 | 910.7 |
| 1× v3d | 80.3 | 282.6 | 405.2 | 565.2 |

Tokens per step 4.91–5.00 in both workloads.

**Coding** (llama-benchy task mode, pp2048 tg512, depth 0, T=1.0, 3 runs), tg tok/s total, max of 3 runs:

| setup | c16 | c32 | tokens/step |
|---|---|---|---|
| 2× b1.4 | 232.0 | 290.7 | 3.28 / 3.32 |
| 1× v3d | 152.1 | 198.7 | 3.28 / 3.37 |

**Straggler probe** at c8/c16/c32: 0 preemptions on both setups (3.98–3.99 accepted per 4 drafts); c32 round wall
10.0 s (2×) and 15.1 s (1×) for 320 tokens per request.

**Memory**: lowest MemAvailable 5.19 GiB (dgx-01) / 6.58 GiB (dgx-02) on 2×, 4.46 GiB on 1×. A 1× boot at
`max_num_seqs` 16 on the other Spark was stopped by the 4 GiB MemAvailable guard during startup.

