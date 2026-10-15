k100ab-1x-hc-a16-bf16-ab result: a BF16 path for the hyperconnection down projection (K10240 to N336) in prefill, against the 1x v2.3.0 stack.

Background: k96 showed that the down projection is served A16 (BF16 activations, plan record a16 n128 k64 s1, 1.883 ms at M 6048), so k90's quantized 128x128/BK64 candidate does not apply to the served path. The A16 kernel only has 16-row M tiles and dequantizes the 3.4 MB weight again for every 16 rows. My b12x branch exp/k100-hc-a16-bf16 (5b77e18e) sends prefill calls of 1024 rows or more to an exact BF16 dequantization of the weight plus one BF16 GEMM with FP32 accumulation. Decode (exact-M regimes), smaller calls and every other shape keep the A16 kernel. Plan records and plan population do not change.

The job runs a probe first: identical captured inputs and real layer 5 weights, the served plan seed as control, numerics before speed (k96 gate), decode rows M 1 to 40, the HC norm/mix kernels as separate rows, and the TP=2 row on the 2x image. The serving A/B (4 boots per side per Spark, ctl arm arm ctl ctl arm arm ctl, cold TTFT 16K/64K/128K, llama-benchy c1/c4/c8, MTP acceptance per position) only runs when the probe projects at least 2% on cold 16K TTFT.

RULE=WIN (per Spark: arm-01=PASS arm-02=PASS; cold PASS; probe GO deq-pc +0.296 s (+3.8%); 2x: k100b (TP=2 A/B, not run by this job))

```
k100ab 2026-10-15 23:15:15 EEST: b12x A16 BF16 large-row path for the HC down projection (arm image ghcr.io/ursuciprian/spark-vllm-b12x:k100-20261015-a1ad1b69-5dad364d, b12x a1ad1b69) vs 1x v2.3.0
RULE=WIN (per Spark: arm-01=PASS arm-02=PASS; cold PASS; probe GO deq-pc +0.296 s (+3.8%); 2x: k100b (TP=2 A/B, not run by this job))

== probe run (k100): /home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k100-1x-hc-a16-bf16-20261015-2035
== projection: seconds saved per cold 16K prompt = sum over [6048, 6048, 3024, 1264] of (control - variant) ms x 96 down calls per step, k81-scaled headline (raw beside it)
  deq-pc: +0.296 s (+3.8% of 7.71 s), +0.254 s raw
  deq-cached: +0.343 s (+4.5% of 7.71 s), +0.295 s raw
  deq-b16: +0.311 s (+4.0% of 7.71 s), +0.267 s raw

== numerics gate per variant: deq-pc=PASS deq-cached=PASS deq-b16=PASS deq-up=FAIL
PREDICT=GO deq-pc +0.296 s (+3.8%)
VERDICT setup=1x served=hit anchor=OK predict=GO deq-pc=PASS deq-cached=PASS deq-b16=PASS deq-up=FAIL deq-pc=+0.296s

screen: thunderdome exit 0 (DONE: arm=INCONCLUSIVE arm=INCONCLUSIVE); per Spark: arm-01=PASS arm-02=PASS; cold PASS
per Spark: arm-01=PASS arm-02=PASS; cold PASS; probe GO deq-pc +0.296 s (+3.8%); 2x: k100b (TP=2 A/B, not run by this job)

-- arm-01 (dgx-01)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.851 arm 0.854 shift +0.004
  pos 1: control 0.704 arm 0.704 shift +0.001
  pos 2: control 0.581 arm 0.576 shift -0.005
  pos 3: control 0.475 arm 0.470 shift -0.005
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      +2.27%  noise  2.90%            n=8 CI [-0.8,+5.0] control boot-to-boot -0.18%, step +0.26%
  probe  fresh-c4      -0.90%  noise  1.77%            n=32 CI [-2.5,+0.7] control boot-to-boot +1.77%, step +0.32%
  probe  fresh-c8      -0.28%  noise  2.64%            n=64 CI [-1.6,+1.0] control boot-to-boot -2.64%, step -0.32%
  benchy pp2048 (c1)   +3.20%  noise  2.36%  better    control 1782.6 arm 1839.6 t/s (boots 4/4)
  benchy tg512 (c1)    +4.97%  noise  3.60%  better    control 56.2 arm 59.0 t/s (boots 4/4)
  benchy tg512 (c4)    +4.65%  noise  5.61%            control 111.3 arm 116.5 t/s (boots 4/4)
  benchy tg512 (c8)    -2.71%  noise  4.71%            control 140.4 arm 136.6 t/s (boots 4/4)
== logits_equiv (reported, not judged)
better: benchy pp2048 (c1) +3.20% (noise 2.36%); benchy tg512 (c1) +4.97% (noise 3.60%)
VERDICT=PROMOTE
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   +3.20%  CI [+0.60,+5.79]  control 1809.4 1767.3 1782.5 1771.3 | arm 1820.0 1848.5 1878.8 1811.2
  tg512 (c1)    +4.97%  CI [-6.02,+15.97]  control 55.8 57.1 55.5 56.4 | arm 64.7 57.1 57.1 57.0
  tg512 (c4)    +4.65%  CI [+1.35,+7.96]  control 111.3 110.9 109.0 114.1 | arm 115.0 116.0 115.6 119.4
  tg512 (c8)    -2.71%  CI [-6.42,+1.00]  control 144.6 139.4 137.9 139.6 | arm 134.3 139.1 139.1 133.9
== post checks per boot
  ctl-p3 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-01/ctl-p1: kv 993,754 ok
  ctl-p4 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-01/ctl-p2: kv 993,754 ok
  ctl-p5 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-01/ctl-p3: kv 993,754 ok
  ctl-p6 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-01/ctl-p4: kv 993,754 ok
  arm-p3 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-01/arm-p1: kv 993,754 ok
  arm-p4 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-01/arm-p2: kv 993,754 ok
  arm-p5 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-01/arm-p3: kv 993,754 ok
  arm-p6 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-01/arm-p4: kv 993,754 ok
worse: none
better: benchy pp2048 (c1)   +3.20%  noise  2.36%  better    control 1782.6 arm 1839.6 t/s (boots 4/4); benchy tg512 (c1)    +4.97%  noise  3.60%  better    control 56.2 arm 59.0 t/s (boots 4/4)
RULE=PASS
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  arm-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  arm-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 

-- arm-02 (dgx-02)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.852 arm 0.852 shift +0.001
  pos 1: control 0.704 arm 0.705 shift +0.001
  pos 2: control 0.578 arm 0.580 shift +0.003
  pos 3: control 0.472 arm 0.476 shift +0.004
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      -1.66%  noise  3.29%            n=8 CI [-4.9,+1.7] control boot-to-boot -1.10%, step -0.02%
  probe  fresh-c4      +1.36%  noise  4.57%            n=32 CI [-0.2,+2.9] control boot-to-boot -4.57%, step -0.49%
  probe  fresh-c8      +0.58%  noise  3.17%            n=64 CI [-0.7,+1.7] control boot-to-boot +3.17%, step +0.05%
  benchy pp2048 (c1)   +2.21%  noise  2.01%  better    control 1794.3 arm 1834.0 t/s (boots 4/4)
  benchy tg512 (c1)    -2.96%  noise 14.42%            control 59.0 arm 57.2 t/s (boots 4/4)
  benchy tg512 (c4)    +0.15%  noise  6.90%            control 110.2 arm 110.3 t/s (boots 4/4)
  benchy tg512 (c8)    +6.05%  noise  5.42%  better    control 133.8 arm 141.8 t/s (boots 4/4)
== logits_equiv (reported, not judged)
better: benchy pp2048 (c1) +2.21% (noise 2.01%); benchy tg512 (c8) +6.05% (noise 5.42%)
VERDICT=PROMOTE
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   +2.21%  CI [+0.69,+3.74]  control 1792.4 1774.2 1810.3 1800.1 | arm 1853.1 1836.6 1828.5 1817.7
  tg512 (c1)    -2.96%  CI [-14.49,+8.57]  control 54.4 62.9 56.3 62.3 | arm 55.3 56.4 60.7 56.5
  tg512 (c4)    +0.15%  CI [-6.22,+6.53]  control 112.2 109.0 105.9 113.5 | arm 110.2 104.4 114.2 112.5
  tg512 (c8)    +6.05%  CI [+0.61,+11.49]  control 134.5 131.2 138.3 131.0 | arm 136.3 143.2 140.8 147.1
== post checks per boot
  ctl-p3 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-02/ctl-p1: kv 993,754 ok
  ctl-p4 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-02/ctl-p2: kv 993,754 ok
  ctl-p5 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-02/ctl-p3: kv 993,754 ok
  ctl-p6 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-02/ctl-p4: kv 993,754 ok
  arm-p3 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-02/arm-p1: kv 993,754 ok
  arm-p4 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-02/arm-p2: kv 993,754 ok
  arm-p5 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-02/arm-p3: kv 993,754 ok
  arm-p6 -> k100ab-1x-hc-a16-bf16-ab-20261015-2047/arm-02/arm-p4: kv 993,754 ok
worse: none
better: benchy pp2048 (c1)   +2.21%  noise  2.01%  better    control 1794.3 arm 1834.0 t/s (boots 4/4); benchy tg512 (c8)    +6.05%  noise  5.42%  better    control 133.8 arm 141.8 t/s (boots 4/4)
RULE=PASS
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  arm-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  arm-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 

== cold TTFT
arm-01   16K: control   2126 arm   2196 tok/s (+3.3%, noise 1.0%) better
arm-01   64K: control   1980 arm   2047 tok/s (+3.4%, noise 1.3%) better
arm-01  128K: control   1755 arm   1804 tok/s (+2.8%, noise 1.1%) better
arm-01 COLD=PASS
arm-02   16K: control   2123 arm   2186 tok/s (+3.0%, noise 1.0%) better
arm-02   64K: control   1992 arm   2065 tok/s (+3.7%, noise 1.7%) better
arm-02  128K: control   1757 arm   1804 tok/s (+2.7%, noise 1.0%) better
arm-02 COLD=PASS
COLD_ALL=PASS
== arm-01: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2126 ctl-p2 2123 ctl-p3 2138 ctl-p4 2119 arm-p1 2196 arm-p2 2196 arm-p3 2197 arm-p4 2194, arm vs ctl +3.3%
  64K: ctl-p1 1970 ctl-p2 1995 ctl-p3 1972 ctl-p4 1980 arm-p1 2057 arm-p2 2052 arm-p3 2047 arm-p4 2032, arm vs ctl +3.4%
  128K: ctl-p1 1742 ctl-p2 1760 ctl-p3 1758 ctl-p4 1760 arm-p1 1795 arm-p2 1804 arm-p3 1809 arm-p4 1810, arm vs ctl +2.8%
== arm-02: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2125 ctl-p2 2126 ctl-p3 2117 ctl-p4 2123 arm-p1 2181 arm-p2 2200 arm-p3 2172 arm-p4 2193, arm vs ctl +3.0%
  64K: ctl-p1 1974 ctl-p2 2007 ctl-p3 1986 ctl-p4 2000 arm-p1 2055 arm-p2 2070 arm-p3 2061 arm-p4 2074, arm vs ctl +3.7%
  128K: ctl-p1 1753 ctl-p2 1765 ctl-p3 1760 ctl-p4 1749 arm-p1 1804 arm-p2 1805 arm-p3 1803 arm-p4 1805, arm vs ctl +2.7%

== profile arm-01 ctl-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1792088169392823481.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       602.2       664.7       416.5       318.6     122.2   25.5%    38.6     226.3      67.0
GDN core (conv, chunk delta, norm)                 355.7       482.5       179.9        78.7      66.9   14.0%             -23.5      72.7
attention (QSA indexer + sparse)                   316.9       415.5       198.8        87.1      62.2   13.0%              17.9      57.8
norms / hyperconnection                            360.3       380.5       181.3        72.8      60.7   12.7%              -6.3      62.3
dense: b12x other                                  297.6       408.0       159.5        65.8      56.8   11.8%             -16.1      60.7
dense: GDN in_proj (qkvz+ba)                       150.6       256.6        76.5        32.4      31.5    6.6%    96.4     -22.5      37.0
MTP drafter (prefill)                              190.2       193.5        93.9        35.6      31.3    6.5%              -5.3      32.6
dense: GDN out_proj                                 90.5       120.6        38.6        14.2      16.1    3.4%    70.3     -14.7      19.7
shared expert + router                              79.6        92.0        37.0        14.9      13.6    2.8%    43.8      -5.9      15.1
dense: attention o_proj                             39.0        45.1        23.5         7.6       7.0    1.5%    53.7       0.2       7.0
fused elementwise (inductor)                        35.5        38.7        18.0         7.5       6.1    1.3%              -0.5       6.2
sampler / bookkeeping                                8.2         8.7         7.6         7.5       1.9    0.4%               7.1       0.2
PLE embedding                                        7.9         9.2         3.9         1.9       1.4    0.3%              -0.0       1.4
dense: BF16 cuBLAS in compiled graph                 5.1         4.6         3.0         1.0       0.8    0.2%               0.3       0.8
step setup (inputs, KV, metadata)                    2.4         4.0         3.3         3.2       0.8    0.2%               3.2      -0.0
GPU busy (sum of kernels)                         2541.8      3124.2      1441.2       748.9     479.5                     160.2     440.4
GPU idle inside the step                           202.0       182.0        94.5        46.3      32.0
step span (first kernel to last)                  2742.6      3304.3      1535.2       792.2     511.1                     162.8     471.4
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile arm-01 arm-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1792089193252695382.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       603.9       639.7       415.7       317.4     120.6   26.2%    39.1     230.0      64.5
GDN core (conv, chunk delta, norm)                 348.9       432.9       182.4        76.6      63.5   13.8%             -11.5      66.3
attention (QSA indexer + sparse)                   316.0       418.3       198.8        86.7      62.2   13.5%              17.1      58.1
norms / hyperconnection                            357.7       376.0       178.6        74.4      60.2   13.1%              -4.6      61.3
dense: b12x other                                  209.3       340.9       117.0        61.3      44.5    9.6%              -6.9      46.2
MTP drafter (prefill)                              189.2       195.3        92.9        35.2      31.3    6.8%              -6.3      32.8
dense: GDN in_proj (qkvz+ba)                       151.9       228.9        77.9        33.5      30.0    6.5%   101.1     -14.7      33.6
dense: GDN out_proj                                 97.4       115.0        39.5        14.2      16.2    3.5%    69.7     -14.5      19.8
shared expert + router                              83.0        89.5        37.4        15.9      13.8    3.0%    43.4      -4.9      15.0
dense: attention o_proj                             38.2        45.4        20.2         6.7       6.7    1.5%    56.0      -2.3       7.3
fused elementwise (inductor)                        35.7        38.4        17.7         7.3       6.0    1.3%              -0.8       6.2
PLE embedding                                        7.9        24.4         3.9         1.9       2.3    0.5%              -3.6       3.2
sampler / bookkeeping                                8.4         8.2         8.4         7.3       2.0    0.4%               7.4       0.2
dense: BF16 cuBLAS in compiled graph                 5.4         8.0         2.4         1.2       1.0    0.2%              -0.7       1.2
step setup (inputs, KV, metadata)                    2.4         3.6         3.6         2.9       0.8    0.2%               3.2      -0.0
GPU busy (sum of kernels)                         2455.1      2964.5      1396.2       742.3     461.3                     186.7     415.7
GPU idle inside the step                           174.3       161.4        93.5        46.9      29.1
step span (first kernel to last)                  2628.4      3124.1      1489.1       786.2     490.0                     200.2     441.1
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile arm-02 ctl-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1792088210436839020.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       602.4       602.7       452.7       319.3     120.7   25.7%    39.1     259.5      57.3
attention (QSA indexer + sparse)                   318.5       374.4       252.6        88.3      63.1   13.4%              55.5      49.5
GDN core (conv, chunk delta, norm)                 355.8       354.1       240.9        76.8      62.7   13.4%              34.9      54.2
norms / hyperconnection                            357.6       358.3       197.7        73.1      60.2   12.8%               8.0      58.3
dense: b12x other                                  299.5       298.9       222.9        65.7      54.1   11.5%              40.5      44.3
MTP drafter (prefill)                              191.7       195.2        95.6        35.2      31.6    6.7%              -5.5      32.9
dense: GDN in_proj (qkvz+ba)                       150.2       150.5       156.2        31.7      29.8    6.4%   101.9      41.7      19.6
dense: GDN out_proj                                 91.4        89.8        56.7        14.1      15.4    3.3%    73.6       1.3      15.1
shared expert + router                              79.9        80.9        45.7        14.6      13.5    2.9%    44.3       0.7      13.3
dense: attention o_proj                             37.6        40.1        23.6         7.3       6.6    1.4%    57.0       1.3       6.3
fused elementwise (inductor)                        36.1        35.4        21.1         7.3       6.1    1.3%               1.5       5.7
sampler / bookkeeping                                8.2         8.2         7.7         7.5       1.9    0.4%               7.3       0.2
PLE embedding                                        8.2         7.8        10.7         1.7       1.7    0.4%               3.4       0.9
dense: BF16 cuBLAS in compiled graph                 5.2         5.3         4.1         1.5       1.0    0.2%               1.1       0.7
step setup (inputs, KV, metadata)                    2.4         4.0         3.2         2.8       0.8    0.2%               2.8       0.1
GPU busy (sum of kernels)                         2544.6      2605.4      1791.6       746.8     469.3                     454.2     358.4
GPU idle inside the step                           174.7       158.4        98.4        43.4      29.0
step span (first kernel to last)                  2718.3      2762.5      1889.3       787.2     497.9                     467.3     383.8
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile arm-02 arm-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1792089245915047355.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       604.9       636.8       413.0       316.1     120.3   26.1%    39.2     227.9      64.7
GDN core (conv, chunk delta, norm)                 346.9       430.8       184.2        75.9      63.3   13.8%             -10.3      65.9
attention (QSA indexer + sparse)                   317.3       399.2       198.8        85.9      61.1   13.3%              20.5      56.1
norms / hyperconnection                            358.6       375.1       176.9        74.5      60.1   13.1%              -5.4      61.4
dense: b12x other                                  213.1       360.9       115.8        59.1      45.7    9.9%             -15.0      49.4
MTP drafter (prefill)                              189.8       196.0        94.8        35.2      31.5    6.8%              -5.7      32.9
dense: GDN in_proj (qkvz+ba)                       151.3       217.6        78.3        32.1      29.3    6.4%   103.8     -13.1      32.5
dense: GDN out_proj                                 98.4       114.4        38.5        13.7      16.2    3.5%    70.0     -15.5      20.0
shared expert + router                              84.4        88.7        36.6        15.5      13.7    3.0%    43.5      -5.8      15.2
dense: attention o_proj                             38.6        43.8        20.6         6.6       6.7    1.5%    56.4      -1.8       7.1
fused elementwise (inductor)                        35.9        38.7        17.5         7.4       6.1    1.3%              -1.0       6.3
PLE embedding                                        7.9        24.9         3.9         1.9       2.4    0.5%              -3.7       3.3
sampler / bookkeeping                                8.9         8.6         7.9         7.5       2.0    0.4%               7.1       0.3
dense: BF16 cuBLAS in compiled graph                 5.4         7.2         2.4         1.0       1.0    0.2%              -0.6       1.1
step setup (inputs, KV, metadata)                    2.4         3.6         3.2         3.2       0.8    0.2%               3.3      -0.0
GPU busy (sum of kernels)                         2463.8      2946.4      1392.4       735.7     460.1                     180.7     416.0
GPU idle inside the step                           203.9       160.1        98.3        45.9      31.0
step span (first kernel to last)                  2666.6      3104.7      1490.0       778.9     490.7                     189.5     444.5
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== prebakes
-- prebake-dgx01/try1  boot 287 s, KV pool 6442450944 bytes, min MemAvailable 22.49 GiB       2 Dynamo bytecode transform time plans measured: 0 pong: pong 
-- prebake-dgx02/try1  boot 284 s, KV pool 6442450944 bytes, min MemAvailable 22.85 GiB       2 Dynamo bytecode transform time plans measured: 0 pong: pong 
```

Results: {RESULTS_URL}
