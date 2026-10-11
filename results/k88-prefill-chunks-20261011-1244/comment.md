k88-prefill-chunks result: `max_num_batched_tokens` 9072 (16K in 3 steps instead of 4 under the 3,024-token mamba alignment) against the shipped 8192 on 1x v2.2.0, ctl arm arm ctl on both Sparks, plus a probe of the K6144 N2560 MXFP8 GEMM that k81 found 4.7x slower at M 6,048 than at M 3,024. Measure only; nothing ships from this job.

1x per Spark: mnbt-01=PASS mnbt-02=PASS (RULE: no cell worse than its noise band and at least one better = PASS, nothing either way = FLAT). Cold TTFT: FAIL. 2x: not run (dropped from this job; a 2x A/B follows a real fix).

```
k88 2026-10-11 14:15:26 EEST: 1x v2.2.0 max_num_batched_tokens 9072 vs 8192 (both with the profiler config); P1 FAIL; P2 not run (dropped from this job; a 2x A/B follows a real fix)

== P0 GEMM probe (dgx-01, shipped 1x image): ms per call by M; capN = serving capacity regime for max_rows N
# summary: ms per call (us per row in brackets)
# out_proj/o_proj
#        M            cap8192         cap8192-u0 cap8192-t128k64-u0            cap9072           cap15120              exact
#     1264    0.332 [ 0.263]    0.332 [ 0.263]    0.429 [ 0.340]    0.332 [ 0.263]    0.332 [ 0.263]                   
#     3024    0.783 [ 0.259]    0.786 [ 0.260]    0.927 [ 0.307]    0.786 [ 0.260]    0.787 [ 0.260]    0.739 [ 0.245]
#     4536    1.525 [ 0.336]    1.517 [ 0.334]    1.338 [ 0.295]    1.519 [ 0.335]    1.515 [ 0.334]                   
#     6048    3.635 [ 0.601]    3.643 [ 0.602]    1.767 [ 0.292]    3.649 [ 0.603]    3.658 [ 0.605]    3.641 [ 0.602]
#     7560    4.409 [ 0.583]    4.411 [ 0.583]    2.183 [ 0.289]    4.414 [ 0.584]    4.410 [ 0.583]                   
#     8192    4.879 [ 0.596]    4.888 [ 0.597]    2.360 [ 0.288]    4.887 [ 0.597]    4.879 [ 0.596]                   
#     9072                                                             5.502 [ 0.607]    5.503 [ 0.607]    5.505 [ 0.607]
#    15120                                                                                9.098 [ 0.602]                   
# in_proj_qkvz
#        M            cap8192         cap8192-u0 cap8192-t128k64-u0            cap9072           cap15120              exact
#     1264    0.772 [ 0.611]                                          0.774 [ 0.612]                                      
#     3024    1.783 [ 0.590]                                          1.782 [ 0.589]                                      
#     4536    2.609 [ 0.575]                                          2.612 [ 0.576]                                      
#     6048    3.517 [ 0.581]                                          3.576 [ 0.591]                                      
#     7560    4.492 [ 0.594]                                          4.572 [ 0.605]                                      
#     8192    5.672 [ 0.692]                                          5.680 [ 0.693]                                      
#     9072                                                             5.792 [ 0.638]                                      
#    15120                                                                                                                  
failed probe variants: 0

== prebake (arm, 6 GiB KV pool)
-- dgx01 try1 
boot 573 s, KV pool 6442450944 bytes, min MemAvailable 10.79 GiB
      2 Dynamo bytecode transform time
plans measured: 6993
pong: pong
-- dgx02 try1 
boot 584 s, KV pool 6442450944 bytes, min MemAvailable 11.50 GiB
      2 Dynamo bytecode transform time
plans measured: 6818
pong: pong

== P1 thunderdome exit 0 (DONE: arm=INCONCLUSIVE arm=INCONCLUSIVE); per Spark: mnbt-01=PASS mnbt-02=PASS; cold: FAIL

-- mnbt-01 (dgx-01)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.850 arm 0.854 shift +0.003
  pos 1: control 0.702 arm 0.707 shift +0.005
  pos 2: control 0.576 arm 0.584 shift +0.008
  pos 3: control 0.470 arm 0.477 shift +0.006
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      +3.93%  noise  8.91%            n=4 CI [-0.2,+8.0] control boot-to-boot +8.91%, step +0.00%
  probe  fresh-c4      +2.27%  noise  3.06%            n=16 CI [-0.8,+5.3] control boot-to-boot -0.10%, step -0.64%
  probe  fresh-c8      -1.14%  noise  2.02%            n=32 CI [-2.6,+0.4] control boot-to-boot +2.02%, step -0.06%
  probe  d16k-c4       -1.57%  noise  3.01%            n=16 CI [-4.5,+1.6] control boot-to-boot -0.55%, step +0.43%
  probe  count-c8     missing
  wall   d16k-c8      missing
  benchy pp2048 (c1)   -0.30%  noise  1.69%            control 1810.2 arm 1804.7 t/s (boots 2/2)
  benchy tg512 (c1)    +1.21%  noise  3.23%            control 58.9 arm 59.6 t/s (boots 2/2)
  benchy tg512 (c8)    +5.87%  noise  2.62%  better    control 135.8 arm 143.8 t/s (boots 2/2)
== logits_equiv (reported, not judged)
reason: probe count-c8: missing
reason: wall d16k-c8: missing
better: benchy tg512 (c8) +5.87% (noise 2.62%)
VERDICT=INCONCLUSIVE
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   -0.30%  CI [-12.45,+11.84]  control 1794.9 1825.5 | arm 1812.8 1796.7
  tg512 (c1)    +1.21%  CI [-19.09,+21.51]  control 59.6 58.1 | arm 60.2 59.0
  tg512 (c8)    +5.87%  CI [+0.62,+11.11]  control 136.3 135.2 | arm 143.7 143.9
== post checks per boot
  ctl-p3 -> k88-prefill-chunks-20261011-1244/mnbt-01/ctl-p1: kv 993,754 ok
  ctl-p4 -> k88-prefill-chunks-20261011-1244/mnbt-01/ctl-p2: kv 993,754 ok
  arm-p3 -> k88-prefill-chunks-20261011-1244/mnbt-01/arm-p1: kv 993,754 ok
  arm-p4 -> k88-prefill-chunks-20261011-1244/mnbt-01/arm-p2: kv 993,754 ok
worse: none
better: benchy tg512 (c8)    +5.87%  noise  2.62%  better    control 135.8 arm 143.8 t/s (boots 2/2)
RULE=PASS
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 9072 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 9072 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 

-- mnbt-02 (dgx-02)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.849 arm 0.852 shift +0.002
  pos 1: control 0.698 arm 0.702 shift +0.004
  pos 2: control 0.572 arm 0.576 shift +0.004
  pos 3: control 0.464 arm 0.468 shift +0.004
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      +0.68%  noise  3.70%            n=4 CI [-2.8,+4.6] control boot-to-boot -1.29%, step -0.22%
  probe  fresh-c4      +3.05%  noise  2.65%  better    n=16 CI [+0.3,+5.6] control boot-to-boot -2.50%, step -0.36%
  probe  fresh-c8      -0.69%  noise  1.73%            n=32 CI [-2.5,+1.0] control boot-to-boot -0.05%, step -0.56%
  probe  d16k-c4       -1.98%  noise  2.84%            n=16 CI [-4.6,+0.5] control boot-to-boot -2.84%, step +0.01%
  probe  count-c8     missing
  wall   d16k-c8      missing
  benchy pp2048 (c1)   -0.14%  noise  1.36%            control 1812.0 arm 1809.5 t/s (boots 2/2)
  benchy tg512 (c1)    -3.05%  noise  5.31%            control 58.8 arm 57.0 t/s (boots 2/2)
  benchy tg512 (c8)    +9.29%  noise  3.89%  better    control 133.1 arm 145.4 t/s (boots 2/2)
== logits_equiv (reported, not judged)
reason: probe count-c8: missing
reason: wall d16k-c8: missing
better: probe fresh-c4 +3.05% (noise 2.65%); benchy tg512 (c8) +9.29% (noise 3.89%)
VERDICT=INCONCLUSIVE
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   -0.14%  CI [-9.67,+9.40]  control 1804.2 1819.8 | arm 1798.4 1820.6
  tg512 (c1)    -3.05%  CI [-39.23,+33.12]  control 57.3 60.3 | arm 56.2 57.8
  tg512 (c8)    +9.29%  CI [-32.10,+50.69]  control 135.7 130.5 | arm 148.9 142.0
== post checks per boot
  ctl-p3 -> k88-prefill-chunks-20261011-1244/mnbt-02/ctl-p1: kv 993,754 ok
  ctl-p4 -> k88-prefill-chunks-20261011-1244/mnbt-02/ctl-p2: kv 993,754 ok
  arm-p3 -> k88-prefill-chunks-20261011-1244/mnbt-02/arm-p1: kv 993,754 ok
  arm-p4 -> k88-prefill-chunks-20261011-1244/mnbt-02/arm-p2: kv 993,754 ok
worse: none
better: probe  fresh-c4      +3.05%  noise  2.65%  better    n=16 CI [+0.3,+5.6] control boot-to-boot -2.50%, step -0.36%; benchy tg512 (c8)    +9.29%  noise  3.89%  better    control 133.1 arm 145.4 t/s (boots 2/2)
RULE=PASS
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 9072 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 9072 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 

== cold TTFT (hook: 3 cold prompts per length per boot, same prompts in every boot)
mnbt-01   16K: control   2079 arm   1987 tok/s (-4.5%, noise 4.3%) WORSE
mnbt-01   64K: control   1901 arm   1875 tok/s (-1.4%, noise 1.0%) WORSE
mnbt-01  128K: control   1729 arm   1729 tok/s (-0.0%, noise 1.0%) flat
mnbt-01 COLD=FAIL
mnbt-02   16K: control   2111 arm   1995 tok/s (-5.5%, noise 1.0%) WORSE
mnbt-02   64K: control   1918 arm   1903 tok/s (-0.8%, noise 1.0%) flat
mnbt-02  128K: control   1722 arm   1740 tok/s (+1.1%, noise 1.1%) flat
mnbt-02 COLD=FAIL
COLD_ALL=FAIL
== mnbt-01: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2035 ctl-p2 2124 arm-p1 1978 arm-p2 1995, arm vs ctl -4.5%
  64K: ctl-p1 1895 ctl-p2 1908 arm-p1 1872 arm-p2 1879, arm vs ctl -1.4%
  128K: ctl-p1 1730 ctl-p2 1729 arm-p1 1723 arm-p2 1735, arm vs ctl -0.0%
== mnbt-02: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2113 ctl-p2 2109 arm-p1 2009 arm-p2 1982, arm vs ctl -5.5%
  64K: ctl-p1 1909 ctl-p2 1927 arm-p1 1896 arm-p2 1910, arm vs ctl -0.8%
  128K: ctl-p1 1712 ctl-p2 1731 arm-p1 1733 arm-p2 1746, arm vs ctl +1.1%

== profile mnbt-01 ctl-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1791713930648665082.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       597.5       634.2       414.2       319.2     119.9   25.3%    39.3     233.5      62.9
GDN core (conv, chunk delta, norm)                 352.5       437.3       178.2        77.3      63.8   13.4%             -14.8      67.4
attention (QSA indexer + sparse)                   315.1       393.4       199.1        86.8      60.7   12.8%              23.2      55.0
norms / hyperconnection                            352.1       368.2       181.2        73.1      59.5   12.5%              -1.4      59.8
dense: b12x other                                  299.1       352.7       155.8        65.6      53.3   11.2%              -6.0      54.8
MTP drafter (prefill)                              193.1       196.5        95.8        36.1      31.8    6.7%              -5.2      33.1
dense: GDN in_proj (qkvz+ba)                       143.1       212.9        75.8        31.0      28.3    6.0%   107.5     -12.6      31.3
dense: GDN out_proj                                153.4       165.8        37.2        14.9      22.7    4.8%    50.0     -39.2      32.2
shared expert + router                              80.5        85.2        37.1        14.7      13.3    2.8%    45.0      -4.6      14.4
dense: attention o_proj                             57.9        63.1        22.2         7.5       9.2    1.9%    41.0      -8.9      11.4
fused elementwise (inductor)                        35.6        37.3        17.4         7.4       6.0    1.3%              -0.6       6.1
PLE embedding                                        8.4        26.8         3.9         1.8       2.5    0.5%              -4.3       3.5
sampler / bookkeeping                                8.3         8.6         7.7         7.6       2.0    0.4%               7.3       0.2
dense: BF16 cuBLAS in compiled graph                 4.4         7.6         2.4         1.0       0.9    0.2%              -0.5       1.1
step setup (inputs, KV, metadata)                    2.4         3.6         3.3         3.3       0.8    0.2%               3.5      -0.1
GPU busy (sum of kernels)                         2603.5      2993.3      1431.3       747.3     474.6                     169.3     433.2
GPU idle inside the step                           164.0       181.2       102.1        46.2      30.1
step span (first kernel to last)                  2766.4      3172.8      1532.8       790.6     504.3                     184.4     459.3
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile mnbt-01 arm-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1791715054021806697.pt.trace.json.gz: 4 prefill chunks (9072, 3024, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       899.1       415.4       408.8       324.7     125.0   25.3%    37.7     200.2      76.1
GDN core (conv, chunk delta, norm)                 661.7       174.8       174.4        78.9      66.5   13.5%             -42.2      76.8
attention (QSA indexer + sparse)                   565.0       199.8       209.9        91.5      65.1   13.2%              19.8      60.2
norms / hyperconnection                            568.5       184.3       183.1        73.0      61.6   12.5%              -8.0      63.5
dense: b12x other                                  553.6       153.7       155.3        65.8      56.7   11.5%             -29.1      63.8
dense: GDN in_proj (qkvz+ba)                       346.4        74.8        74.5        32.5      32.2    6.5%    94.2     -39.8      42.0
MTP drafter (prefill)                              290.7        96.7        98.7        33.2      31.7    6.4%              -3.6      32.6
dense: GDN out_proj                                248.0        37.6        38.2        14.0      20.6    4.2%    54.9     -45.5      31.7
shared expert + router                             134.9        39.7        39.1        15.5      14.0    2.8%    42.7      -6.1      15.5
dense: attention o_proj                             86.7        21.5        20.4         6.9       8.3    1.7%    45.6      -9.0      10.5
fused elementwise (inductor)                        58.0        17.3        17.5         7.5       6.1    1.2%              -1.8       6.6
PLE embedding                                       38.0         4.6         3.9         1.9       3.0    0.6%              -8.3       5.0
sampler / bookkeeping                                9.4         7.5         7.5         7.3       1.9    0.4%               6.7       0.3
dense: BF16 cuBLAS in compiled graph                 9.7         2.7         2.6         1.0       1.0    0.2%              -0.6       1.1
step setup (inputs, KV, metadata)                    2.8         3.2         3.3         2.9       0.7    0.2%               3.2      -0.0
GPU busy (sum of kernels)                         4472.7      1433.9      1436.9       756.6     494.4                      35.8     485.6
GPU idle inside the step                           248.3        95.7        88.2        46.3      29.2
step span (first kernel to last)                  4719.0      1528.9      1524.5       800.1     523.2                      48.1     511.5
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile mnbt-02 ctl-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1791713994873352256.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       654.8       622.8       454.8       336.4     126.3   24.6%    37.4     260.2      62.7
GDN core (conv, chunk delta, norm)                 462.6       357.6       216.6        90.6      68.8   13.4%              10.5      66.2
attention (QSA indexer + sparse)                   352.0       382.0       256.4       110.5      67.2   13.1%              69.1      50.3
norms / hyperconnection                            381.3       361.2       190.2        79.2      61.8   12.0%               3.9      60.8
dense: b12x other                                  388.9       306.4       189.9        82.1      59.0   11.5%              17.1      54.9
dense: GDN in_proj (qkvz+ba)                       248.7       150.0       122.5        44.0      34.5    6.7%    88.1      13.9      31.1
MTP drafter (prefill)                              192.8       194.0       136.7        35.5      34.1    6.7%              15.8      30.3
dense: GDN out_proj                                170.5       155.7        47.6        19.2      24.0    4.7%    47.2     -31.3      31.6
shared expert + router                              89.5        82.8        44.0        18.3      14.3    2.8%    41.7       0.8      14.1
dense: attention o_proj                             60.1        61.7        25.3         8.6       9.5    1.9%    39.7      -6.5      11.1
fused elementwise (inductor)                        38.3        35.9        20.3         8.6       6.3    1.2%               1.7       5.9
PLE embedding                                       25.9        10.2         3.9         4.4       2.7    0.5%              -2.0       3.2
sampler / bookkeeping                                8.4         8.2        10.1         7.6       2.1    0.4%               8.6      -0.0
dense: BF16 cuBLAS in compiled graph                 7.0         7.5         2.4         2.2       1.2    0.2%               0.0       1.2
step setup (inputs, KV, metadata)                    2.4         4.1         3.2         5.2       0.9    0.2%               5.0      -0.3
GPU busy (sum of kernels)                         3083.1      2740.0      1724.0       852.5     512.7                     366.8     423.1
GPU idle inside the step                           196.0       175.8        96.4        49.1      31.6
step span (first kernel to last)                  3277.8      2914.1      1819.6       898.7     543.8                     375.7     452.1
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile mnbt-02 arm-p1 (one cold 16K prompt)
== dp0_pp0_tp0_dcp0_ep0_rank0.1791715112853011765.pt.trace.json.gz: 4 prefill chunks (9072, 3024, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       852.8       412.4       528.1       319.9     129.0   25.3%    36.6     256.3      66.4
attention (QSA indexer + sparse)                   567.1       199.2       295.8        91.8      70.4   13.8%              51.3      57.9
GDN core (conv, chunk delta, norm)                 602.4       175.7       240.9        79.5      67.0   13.2%               3.1      66.3
norms / hyperconnection                            563.9       184.8       203.4        74.0      62.6   12.3%               1.9      62.2
dense: b12x other                                  521.0       156.1       219.5        64.3      58.7   11.5%               5.8      57.2
dense: GDN in_proj (qkvz+ba)                       285.1        74.8       155.5        31.2      33.4    6.5%    91.0      10.2      30.9
MTP drafter (prefill)                              287.9        96.1        96.6        34.8      31.5    6.2%              -2.9      32.1
dense: GDN out_proj                                238.5        37.9        55.6        13.2      21.1    4.1%    53.8     -36.2      29.9
shared expert + router                             139.1        39.8        48.0        15.4      14.8    2.9%    40.4      -4.2      15.8
dense: attention o_proj                             85.0        19.7        24.6         6.6       8.3    1.6%    45.5      -7.7      10.2
fused elementwise (inductor)                        57.5        17.6        20.8         7.3       6.3    1.2%              -0.4       6.4
PLE embedding                                       39.6         4.5         3.8         2.0       3.0    0.6%              -8.8       5.2
sampler / bookkeeping                                8.7         7.9         7.9         7.1       1.9    0.4%               7.2       0.2
dense: BF16 cuBLAS in compiled graph                10.2         2.8         2.5         1.0       1.0    0.2%              -0.8       1.2
step setup (inputs, KV, metadata)                    2.9         3.7         3.2         3.0       0.8    0.2%               3.4      -0.0
GPU busy (sum of kernels)                         4261.7      1433.0      1906.3       751.2     509.8                     278.3     441.8
GPU idle inside the step                           285.7        98.2        95.0        57.5      32.7
step span (first kernel to last)                  4545.6      1530.5      2000.6       805.8     542.1                     288.0     471.8
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms
```

Traces stay on dgx-01 in `/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k88-prefill-chunks-20261011-1244`. Results: {RESULTS_URL}
