k90 result: the b12x large-row MXFP8 capacity regime (128x128/BK64 without the large-M unroll for live M >= 4096 on wide-N, K >= 4096 projections; a 4095-row regime keeps the shipped plan below 4096; decode unchanged) against the shipped 1x v2.2.0, ctl arm arm ctl on both Sparks. Measure only; nothing ships from this job.

Screens: `tf: thunderdome exit 0: tf-01=FAIL tf-02=PASS; cold PASS;2x: skipped, no TP=2 projection changes plan (K3072 N2560 lowers as shipped at every probed M; see phase 0);`. RULE PASS/FLAT = no cell from the T=0 probes, pp2048 or tg512 c1/c8 worse than its noise band.

```
k90 2026-10-11 22:50:24 EEST: 1x v2.2.0 on the b12x large-row tile-fix image (ghcr.io/ursuciprian/spark-vllm-b12x:tf-20261011-b45fe867-5dad364d, b12x b45fe867d6b386c95d75e9a46fcf646e0b9e7005) vs shipped

== phase 0 GEMM probe through plan_regimes (capacity 8192): shipped image (old) vs arm image (new); cand = forced 128x128/BK64/u0 dense GEMM alone
n	k	m	old_ms	new_ms	speedup	old_plan	new_plan	cand_ms	same_bytes
2560	6144	1264	0.4403	0.4536	0.97	(64, 128)/BK128/u1	(64, 128)/BK128/u1	0.4396	yes
2560	6144	3024	1.0107	1.1036	0.92	(64, 128)/BK128/u1	(64, 128)/BK128/u1	0.9230	yes
2560	6144	4536	1.9827	1.7535	1.13	(64, 128)/BK128/u1	(128, 128)/BK64/u0	1.3579	yes
2560	6144	6048	4.1998	2.2861	1.84	(64, 128)/BK128/u1	(128, 128)/BK64/u0	1.7728	yes
2560	6144	8192	5.7538	3.1419	1.83	(64, 128)/BK128/u1	(128, 128)/BK64/u0	2.3897	yes
2560	3072	1264	0.2233	0.2305	0.97	(64, 128)/BK128/u1	(64, 128)/BK128/u1	0.2214	yes
2560	3072	3024	0.5095	0.5230	0.97	(64, 128)/BK128/u1	(64, 128)/BK128/u1	0.4657	yes
2560	3072	4536	0.7382	0.7527	0.98	(64, 128)/BK128/u1	(64, 128)/BK128/u1	0.6945	yes
2560	3072	6048	1.0246	1.0348	0.99	(64, 128)/BK128/u1	(64, 128)/BK128/u1	0.9001	yes
2560	3072	8192	1.6306	1.6743	0.97	(64, 128)/BK128/u1	(64, 128)/BK128/u1	1.2013	yes
336	10240	1264	0.3247	0.3279	0.99	(64, 64)/BK128/u1	(64, 64)/BK128/u1	0.3638	yes
336	10240	3024	1.3110	1.2872	1.02	(64, 64)/BK128/u1	(64, 64)/BK128/u1	0.7008	yes
336	10240	4536	2.0504	1.9838	1.03	(64, 64)/BK128/u1	(64, 64)/BK128/u1	1.0299	yes
336	10240	6048	2.7165	2.7011	1.01	(64, 64)/BK128/u1	(64, 64)/BK128/u1	1.3629	yes
336	10240	8192	3.6608	3.6573	1.00	(64, 64)/BK128/u1	(64, 64)/BK128/u1	1.8720	yes
1280	2560	1264	0.1528	0.1548	0.99	(64, 64)/BK128/u1	(64, 64)/BK128/u1	0.1231	yes
1280	2560	3024	0.3219	0.3293	0.98	(64, 64)/BK128/u1	(64, 64)/BK128/u1	0.2093	yes
1280	2560	4536	0.4905	0.4944	0.99	(64, 64)/BK128/u1	(64, 64)/BK128/u1	0.2954	yes
1280	2560	6048	0.6488	0.6484	1.00	(64, 64)/BK128/u1	(64, 64)/BK128/u1	0.3976	yes
1280	2560	8192	0.9185	0.9235	0.99	(64, 64)/BK128/u1	(64, 64)/BK128/u1	0.5183	yes
16384	2560	1264	0.8292	0.8559	0.97	(64, 128)/BK128/u1	(64, 128)/BK128/u1	0.7835	yes
16384	2560	3024	1.9027	1.9643	0.97	(64, 128)/BK128/u1	(64, 128)/BK128/u1	1.7450	yes
16384	2560	4536	2.7991	2.8867	0.97	(64, 128)/BK128/u1	(64, 128)/BK128/u1	2.5323	yes
16384	2560	6048	3.7299	3.8607	0.97	(64, 128)/BK128/u1	(64, 128)/BK128/u1	3.2990	yes
16384	2560	8192	6.0389	6.0629	1.00	(64, 128)/BK128/u1	(64, 128)/BK128/u1	4.3752	yes
# out_2560x6144_4536.pt: cos 1.002265 max_abs_rel 0.00e+00 ok
# out_2560x6144_6048.pt: cos 1.003142 max_abs_rel 0.00e+00 ok
# out_2560x6144_8192.pt: cos 1.006182 max_abs_rel 0.00e+00 ok
CHECK=PASS

tf: thunderdome exit 0 (DONE: tf=INCONCLUSIVE tf=INCONCLUSIVE); per Spark: tf-01=FAIL tf-02=PASS; cold PASS
2x: skipped, no TP=2 projection changes plan (K3072 N2560 lowers as shipped at every probed M; see phase 0)

-- tf-01 (dgx-01)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.856 arm 0.853 shift -0.004
  pos 1: control 0.706 arm 0.702 shift -0.003
  pos 2: control 0.583 arm 0.571 shift -0.012
  pos 3: control 0.478 arm 0.462 shift -0.015
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      -4.98%  noise  6.56%            n=4 CI [-8.7,-1.0] control boot-to-boot +6.56%, step -0.20%
  probe  fresh-c4      -0.63%  noise  3.15%            n=16 CI [-3.4,+2.3] control boot-to-boot +3.15%, step +0.00%
  probe  fresh-c8      -0.99%  noise  1.81%            n=32 CI [-2.8,+0.8] control boot-to-boot +0.83%, step +1.90%
  probe  d16k-c4      missing
  probe  count-c8     missing
  wall   d16k-c8      missing
  benchy pp2048 (c1)   -1.51%  noise  2.46%            control 1806.9 arm 1779.6 t/s (boots 2/2)
  benchy tg512 (c1)    +6.00%  noise 12.80%            control 57.0 arm 60.5 t/s (boots 2/2)
  benchy tg512 (c8)    -3.13%  noise  2.68%  worse     control 141.3 arm 136.9 t/s (boots 2/2)
== logits_equiv (reported, not judged)
reason: benchy tg512 (c8) -3.13% (noise 2.68%)
reason: probe d16k-c4: missing
reason: probe count-c8: missing
reason: wall d16k-c8: missing
better: none
VERDICT=INCONCLUSIVE
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   -1.51%  CI [-20.14,+17.12]  control 1829.1 1784.7 | arm 1794.0 1765.2
  tg512 (c1)    +6.00%  CI [-84.02,+96.02]  control 53.4 60.7 | arm 58.7 62.2
  tg512 (c8)    -3.13%  CI [-15.92,+9.66]  control 142.5 140.1 | arm 137.6 136.1
== post checks per boot
  ctl-p3 -> k90-1x-tile-fix-20261011-2132/tf-01/ctl-p1: kv 993,754 ok
  ctl-p4 -> k90-1x-tile-fix-20261011-2132/tf-01/ctl-p2: kv 993,754 ok
  arm-p3 -> k90-1x-tile-fix-20261011-2132/tf-01/arm-p1: kv 993,754 ok
  arm-p4 -> k90-1x-tile-fix-20261011-2132/tf-01/arm-p2: kv 993,754 ok
worse: benchy tg512 (c8)    -3.13%  noise  2.68%  worse     control 141.3 arm 136.9 t/s (boots 2/2)
better: none
RULE=FAIL
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 

-- tf-02 (dgx-02)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.855 arm 0.849 shift -0.006
  pos 1: control 0.712 arm 0.703 shift -0.009
  pos 2: control 0.585 arm 0.576 shift -0.010
  pos 3: control 0.480 arm 0.472 shift -0.008
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      -1.50%  noise  8.17%            n=4 CI [-6.8,+4.3] control boot-to-boot -8.17%, step +1.43%
  probe  fresh-c4      -0.99%  noise  1.74%            n=16 CI [-2.8,+0.7] control boot-to-boot -1.27%, step -0.31%
  probe  fresh-c8      -0.83%  noise  2.02%            n=32 CI [-2.9,+1.1] control boot-to-boot -1.34%, step +1.17%
  probe  d16k-c4      missing
  probe  count-c8     missing
  wall   d16k-c8      missing
  benchy pp2048 (c1)   -1.21%  noise  1.98%            control 1815.1 arm 1793.2 t/s (boots 2/2)
  benchy tg512 (c1)    -3.56%  noise 21.12%            control 61.2 arm 59.0 t/s (boots 2/2)
  benchy tg512 (c8)    +3.01%  noise  2.64%  better    control 139.2 arm 143.4 t/s (boots 2/2)
== logits_equiv (reported, not judged)
reason: probe d16k-c4: missing
reason: probe count-c8: missing
reason: wall d16k-c8: missing
better: benchy tg512 (c8) +3.01% (noise 2.64%)
VERDICT=INCONCLUSIVE
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   -1.21%  CI [-12.63,+10.22]  control 1831.1 1799.2 | arm 1796.8 1789.7
  tg512 (c1)    -3.56%  CI [-138.17,+131.04]  control 54.7 67.6 | arm 58.5 59.5
  tg512 (c8)    +3.01%  CI [-3.23,+9.24]  control 139.0 139.5 | arm 142.8 144.1
== post checks per boot
  ctl-p3 -> k90-1x-tile-fix-20261011-2132/tf-02/ctl-p1: kv 993,754 ok
  ctl-p4 -> k90-1x-tile-fix-20261011-2132/tf-02/ctl-p2: kv 993,754 ok
  arm-p3 -> k90-1x-tile-fix-20261011-2132/tf-02/arm-p1: kv 993,754 ok
  arm-p4 -> k90-1x-tile-fix-20261011-2132/tf-02/arm-p2: kv 993,754 ok
worse: none
better: benchy tg512 (c8)    +3.01%  noise  2.64%  better    control 139.2 arm 143.4 t/s (boots 2/2)
RULE=PASS
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 

== cold TTFT tf
tf-01   16K: control   2090 arm   2125 tok/s (+1.7%, noise 1.0%) better
tf-01   64K: control   1978 arm   2008 tok/s (+1.6%, noise 1.0%) better
tf-01  128K: control   1721 arm   1761 tok/s (+2.3%, noise 1.0%) better
tf-01 COLD=PASS
tf-02   16K: control   2086 arm   2129 tok/s (+2.0%, noise 1.0%) better
tf-02   64K: control   1988 arm   2028 tok/s (+2.0%, noise 1.0%) better
tf-02  128K: control   1734 arm   1769 tok/s (+2.0%, noise 1.0%) better
tf-02 COLD=PASS
COLD_ALL=PASS
== tf-01: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2089 ctl-p2 2091 arm-p1 2136 arm-p2 2114, arm vs ctl +1.7%
  64K: ctl-p1 1983 ctl-p2 1972 arm-p1 2019 arm-p2 1998, arm vs ctl +1.6%
  128K: ctl-p1 1725 ctl-p2 1717 arm-p1 1765 arm-p2 1757, arm vs ctl +2.3%
== tf-02: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2089 ctl-p2 2083 arm-p1 2132 arm-p2 2125, arm vs ctl +2.0%
  64K: ctl-p1 1993 ctl-p2 1983 arm-p1 2043 arm-p2 2014, arm vs ctl +2.0%
  128K: ctl-p1 1738 ctl-p2 1730 arm-p1 1774 arm-p2 1764, arm vs ctl +2.0%

== profile tf-01 ctl-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1791745270043667819.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       597.5       639.1       412.5       317.8     120.1   25.2%    39.3     230.3      63.8
GDN core (conv, chunk delta, norm)                 354.2       443.1       180.1        77.5      64.4   13.5%             -15.3      68.1
attention (QSA indexer + sparse)                   316.1       404.9       200.6        87.5      61.6   12.9%              21.8      56.3
norms / hyperconnection                            354.2       372.6       182.4        73.1      60.0   12.6%              -2.3      60.5
dense: b12x other                                  299.5       368.9       156.3        65.3      54.3   11.4%              -9.7      56.7
MTP drafter (prefill)                              185.9       193.1        94.1        35.6      31.0    6.5%              -4.1      32.0
dense: GDN in_proj (qkvz+ba)                       145.5       216.7        73.7        31.0      28.5    6.0%   106.6     -15.1      32.2
dense: GDN out_proj                                152.9       165.0        37.9        14.1      22.6    4.7%    50.2     -39.2      32.2
shared expert + router                              80.1        87.1        36.8        15.1      13.4    2.8%    44.7      -4.8      14.5
dense: attention o_proj                             57.6        61.6        22.3         7.2       9.1    1.9%    41.6      -8.7      11.2
fused elementwise (inductor)                        35.6        38.9        17.4         7.2       6.1    1.3%              -1.1       6.3
PLE embedding                                        7.8        26.4         4.3         1.7       2.5    0.5%              -4.0       3.4
sampler / bookkeeping                                8.8         8.6         7.5         7.3       2.0    0.4%               6.8       0.3
dense: BF16 cuBLAS in compiled graph                 4.6         7.3         2.5         1.3       1.0    0.2%              -0.1       1.0
step setup (inputs, KV, metadata)                    2.4         3.7         3.5         3.2       0.8    0.2%               3.5      -0.1
GPU busy (sum of kernels)                         2602.6      3037.1      1432.4       744.9     477.1                     157.9     438.6
GPU idle inside the step                           192.1       160.4       123.3        56.8      32.5
step span (first kernel to last)                  2793.6      3195.9      1555.0       798.9     509.2                     192.2     462.3
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile tf-01 arm-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1791746229815043057.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       598.3       639.7       411.9       320.5     120.3   25.4%    39.2     232.3      63.6
GDN core (conv, chunk delta, norm)                 354.0       457.7       179.7        78.2      65.3   13.8%             -18.1      69.7
attention (QSA indexer + sparse)                   316.4       423.5       198.5        87.2      62.6   13.2%              16.2      58.6
norms / hyperconnection                            357.3       378.9       180.7        73.3      60.4   12.8%              -5.1      61.7
dense: b12x other                                  297.2       383.0       158.8        66.9      55.3   11.7%              -9.7      57.7
MTP drafter (prefill)                              189.2       192.0        93.1        36.2      31.2    6.6%              -4.5      32.3
dense: GDN in_proj (qkvz+ba)                       146.7       234.0        78.0        31.7      29.9    6.3%   101.5     -16.4      33.9
dense: GDN out_proj                                 90.2       117.6        39.3        14.2      15.9    3.4%    71.0     -13.6      19.3
shared expert + router                              78.5        87.9        37.0        15.8      13.4    2.8%    44.7      -3.9      14.3
dense: attention o_proj                             37.6        46.0        21.5         7.2       6.9    1.4%    55.0      -1.2       7.1
fused elementwise (inductor)                        35.4        38.3        17.5         7.5       6.0    1.3%              -0.6       6.2
PLE embedding                                        8.2        26.0         4.8         2.0       2.5    0.5%              -3.4       3.3
sampler / bookkeeping                                8.5         8.9         7.7         7.2       2.0    0.4%               6.8       0.3
dense: BF16 cuBLAS in compiled graph                 4.5         7.1         2.7         1.0       0.9    0.2%              -0.3       1.0
step setup (inputs, KV, metadata)                    2.7         3.6         3.3         3.4       0.8    0.2%               3.5      -0.1
GPU busy (sum of kernels)                         2524.5      3044.2      1434.3       752.4     473.4                     182.0     428.9
GPU idle inside the step                           172.3       154.4        97.7        48.5      28.9
step span (first kernel to last)                  2695.7      3196.9      1531.4       798.0     501.8                     201.3     452.7
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile tf-02 ctl-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1791745312651496558.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       600.5       643.6       414.5       317.2     120.6   25.3%    39.1     229.1      64.7
GDN core (conv, chunk delta, norm)                 351.4       435.0       179.8        76.2      63.6   13.4%             -14.2      67.1
attention (QSA indexer + sparse)                   317.1       403.9       199.3        87.4      61.5   12.9%              21.0      56.4
norms / hyperconnection                            348.2       365.3       177.4        72.5      58.8   12.3%              -2.4      59.4
dense: b12x other                                  301.3       368.3       156.0        67.1      54.5   11.4%              -8.5      56.6
MTP drafter (prefill)                              192.2       195.7        96.3        35.8      31.7    6.7%              -4.8      32.9
dense: GDN in_proj (qkvz+ba)                       143.1       215.2        75.7        30.5      28.3    5.9%   107.2     -13.7      31.7
dense: GDN out_proj                                153.0       167.2        38.0        13.6      22.7    4.8%    49.9     -40.2      32.5
shared expert + router                              80.6        86.3        36.3        15.3      13.3    2.8%    44.8      -4.8      14.5
dense: attention o_proj                             57.2        63.1        21.0         7.9       9.1    1.9%    41.5      -9.0      11.3
fused elementwise (inductor)                        35.8        38.6        17.6         7.4       6.1    1.3%              -0.8       6.3
PLE embedding                                        9.2        25.6         4.0         2.0       2.5    0.5%              -3.9       3.4
sampler / bookkeeping                                8.7         8.5         8.1         7.7       2.0    0.4%               7.4       0.2
dense: BF16 cuBLAS in compiled graph                 4.6         7.3         2.4         1.0       0.9    0.2%              -0.5       1.1
step setup (inputs, KV, metadata)                    2.4         3.7         3.2         3.1       0.8    0.2%               3.1      -0.0
GPU busy (sum of kernels)                         2605.0      3027.2      1429.4       744.5     476.5                     157.7     438.0
GPU idle inside the step                           161.3       161.9       122.1        45.4      29.9
step span (first kernel to last)                  2765.2      3187.4      1550.9       787.0     506.0                     187.4     460.3
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile tf-02 arm-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1791746287063614590.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       598.1       646.3       414.6       318.5     120.7   25.5%    39.1     230.2      64.5
GDN core (conv, chunk delta, norm)                 354.4       450.0       179.0        77.1      64.7   13.7%             -17.9      69.1
attention (QSA indexer + sparse)                   317.4       426.3       198.3        87.0      62.8   13.3%              15.1      59.1
norms / hyperconnection                            353.2       372.9       179.9        73.1      59.8   12.6%              -3.4      60.6
dense: b12x other                                  298.3       376.4       156.4        67.8      54.9   11.6%              -8.8      57.0
MTP drafter (prefill)                              192.1       195.8        95.1        35.1      31.6    6.7%              -6.1      33.1
dense: GDN in_proj (qkvz+ba)                       149.5       229.9        78.0        32.4      29.9    6.3%   101.6     -15.5      33.7
dense: GDN out_proj                                 91.4       114.7        39.6        14.7      15.9    3.4%    71.3     -12.5      18.9
shared expert + router                              79.0        87.3        37.0        14.8      13.3    2.8%    44.9      -4.7      14.5
dense: attention o_proj                             37.6        47.0        23.5         7.5       7.1    1.5%    53.5      -0.1       7.1
fused elementwise (inductor)                        35.8        38.4        17.6         7.5       6.1    1.3%              -0.7       6.2
PLE embedding                                        8.2        26.3         5.3         1.7       2.5    0.5%              -3.4       3.4
sampler / bookkeeping                                8.4         8.2         7.9         7.8       2.0    0.4%               7.6       0.1
dense: BF16 cuBLAS in compiled graph                 4.8         7.3         2.3         1.1       0.9    0.2%              -0.6       1.1
step setup (inputs, KV, metadata)                    2.4         3.5         3.5         2.8       0.7    0.2%               3.1      -0.0
GPU busy (sum of kernels)                         2530.6      3030.3      1438.1       748.8     472.9                     182.3     428.4
GPU idle inside the step                           178.0       154.5        99.1        46.5      29.2
step span (first kernel to last)                  2707.4      3183.0      1536.5       792.3     501.7                     199.0     453.1
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== prebakes
-- prebake-tf-prebake-dgx01/try1  boot 301 s, KV pool 6442450944 bytes, min MemAvailable 22.54 GiB       2 Dynamo bytecode transform time plans measured: 187 pong: pong 
-- prebake-tf-prebake-dgx02/try1  boot 284 s, KV pool 6442450944 bytes, min MemAvailable 23.05 GiB       2 Dynamo bytecode transform time plans measured: 170 pong: pong 
```

Traces stay on dgx-01 in `/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k90-1x-tile-fix-20261011-2132`. Results: {RESULTS_URL}
