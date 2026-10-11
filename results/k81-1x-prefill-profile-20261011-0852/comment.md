k81 result: torch-profiler breakdown of one cold 16K prefill on 1x v2.2.0 (c1, `max_num_batched_tokens` 8192 as shipped), plus MIN_TOKENS 0 and 1 arms on dgx-01 (6 GiB KV pool, own cache). The MTP-off arm and the b12x rebase arm on dgx-02 were stopped before measuring (reasons below and in the NOTE.txt / STOPPED.txt files of the results).

What the shipped 1x v2.2.0 does with one cold 16K prompt (c1, d01-ship):

- TTFT 7.93 s (2,067 tok/s, 484 us per prompt token). The trace covers all of it: the prefill steps take 7.94 s of GPU span and the GPU is never idle between steps, so the CPU, scheduler and HTTP side add close to nothing. Inside the steps the GPU sits idle 7% of the time (33 us/token).
- The 16K prompt runs in four steps of 6,048, 6,048, 3,024 and 1,264 tokens, not two of 8,192. With the mamba cache in align mode every chunk ends on a 3,024-token block boundary, so `max_num_batched_tokens` 8192 gives 6,048-token chunks, and the prompt stops at the last boundary (15,120) before the 1,264-token tail. A step costs about 266 ms fixed plus 420 us per token (least-squares line over the four chunks), so the two extra steps cost about 0.5 s (6 to 7% of TTFT).
- Routed MoE is the largest class: 119.5 us/token, 26% of GPU time. Per step it is 248 ms fixed plus 59 us per token. The fixed part matches one pass over all routed expert weights (about 68 GB of NVFP4 over 48 layers) at about 270 GB/s, the GB10 memory bandwidth: every expert gets rows in a 6,048-token chunk, so each step reads every expert once. The per-token part runs at about 80 TFLOPS of FP4 work.
- Hyperconnection is the second largest cost once its parts are added: the combine/norm and gate kernels on the 4 x 2,560 = 10,240-wide residual (56.7 us/token) plus the low-rank HC GEMMs (M x 10240 -> 336 and M x 320 -> 10240, 48.7 us/token, about 30 TFLOPS). Together about 105 us/token, 22%.
- Then QSA attention 59.7 us/token (13%), GDN core 58.0 (13%, of which about 40% is gather, copy, fill and scatter around the chunked delta-rule kernels), MTP drafter prefill 31.1 (7%), GDN in_proj 24.2 (MXFP8, about 125 TFLOPS), GDN out_proj 21.7, shared expert + router 12.6, attention o_proj 8.9, PLE 1.3.
- The dense GEMM with K 6144 and N 2560 (GDN out_proj on the MXFP8 copy, attention o_proj on NVFP4) takes 3.65 ms per call at M 6,048 but 0.78 ms at M 3,024: 4.7x the time for 2x the rows. GDN in_proj scales linearly (1.90 vs 1.00 ms). That looks like a poor plan choice at M 6,048; linear scaling would save about 190 ms per 16K prompt (about 2.4%).
- MIN_TOKENS 0 (GDN projections on W4A16 at prefill as well): TTFT 9.59 s (+21%). GDN in_proj goes from 24 to 105 us/token and out_proj from 22 to 39. So the MXFP8 prefill copy saves 1.66 s per 16K prompt for its 1.1 GiB.
- MIN_TOKENS 1 (MXFP8 at every M): TTFT 7.85 s, within 1% of the shipped 41. Prefill chunks have M of 1,264 or more, so both settings take the same MXFP8 path at prefill, as expected.
- MTP off was not measured. vLLM refuses VLLM_GDN_DEFERRED_CHECKPOINTS and VLLM_GDN_COMPACT_RECORDS without speculative tokens; with both removed the GDN prefill takes another kernel path that needed 91 new plans at about 1 min each, so it would not have isolated the drafter anyway. The drafter prefill cost comes from the shipped trace: 31.1 us/token (6.9%), mostly the MTP feedback prefill GEMM (277 of 510 ms over the 4 chunks) plus one QSA attention and one W4A16 MoE layer.
- The b12x rebase image was not measured. On the single Spark it needs a cold autotune of 271 GDN prefill plans (then 271 GDN attention plans) after the KV pool is reserved, at 1 to 2.5 min per plan with one compile worker, so 4 h or more. I stopped that arm after 22 min to keep the 2x serving downtime short; the 297 blockscaled GEMM plans had finished in about 9 min. It needs its own overnight prebake job.

Fix hypotheses, largest expected gain first:

1. MoE prefill: the dynamic MoE kernel is weight-bandwidth bound for 248 ms per step and then runs the per-token work at about 80 TFLOPS. Fewer, larger steps cut the weight passes (16K in 2 steps instead of 4: about -0.5 s, -6%), and a prefill-only grouped GEMM path near 200 TFLOPS FP4 on the per-token part would save about 35 us/token (-7%).
2. Large-M plans: before retrying bigger chunks (k78 found 16384 slower, not faster), profile a 16384-token step and look for the same superlinear jump as the out_proj plan at 6,048. Fixing the plan choice is the precondition for item 1's fewer steps.
3. Hyperconnection: the combine/norm kernel moves its 6,048 x 10,240 BF16 input at about 105 GB/s, under 40% of bandwidth, and the skinny HC GEMMs reach about 30 TFLOPS. Halving the HC share would save about 50 us/token (-10%).
4. out_proj / o_proj plan at M 6,048: about -2.4%.

Together that is roughly 25%, about 360 us/token. Another public stack on the same hardware reads 16K at about 250 us/token, so the rest has to come from attention and the GDN core data movement as well; no single class closes the gap.

Rooflines I use: GB10 FP4 headline 1 PFLOP is with 2:4 sparsity, so about 500 TFLOPS dense FP4 at most; BF16 torch.mm measured 79 to 84 TFLOPS on these Sparks (2026-09-30 probe); memory bandwidth 273 GB/s. TFLOPS in the tables are 2 x active parameters per token divided by the class time, so they include each class's own quantize kernels.

```
== TTFT, one cold 16K prompt at c1 (3 unprofiled requests per boot, prompt tokens / mean TTFT)
unit               TTFT s   tok/s  us/token  each TTFT s | boot facts
d01-min0            9.586    1709     585.1  9.49, 9.58, 9.69 | tp1-d3-hf-20261010-21e0b201-5dad364d-warm MIN_TOKENS 0, MTP 4, KV 6 GiB, boot 243 s, torch compiled, 0 plans measured
d01-min1            7.847    2088     479.0  7.85, 7.87, 7.82 | tp1-d3-hf-20261010-21e0b201-5dad364d-warm MIN_TOKENS 1, MTP 4, KV 6 GiB, boot 202 s, torch compiled, 0 plans measured
d01-nomtp             n/a | 
d01-ship            7.925    2067     483.7  7.97, 7.93, 7.88 | tp1-d3-hf-20261010-21e0b201-5dad364d-warm MIN_TOKENS 41, MTP 4, KV 14 GiB, boot 186 s, AOT loaded, 0 plans measured

== d01-min0: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       607.4       606.6       431.4       320.9     120.0   21.8%    39.3     248.0      59.5
dense: GDN in_proj (qkvz+ba)                       642.7       642.1       298.6       128.1     104.5   18.9%    29.1     -17.3     108.7
attention (QSA indexer + sparse)                   315.6       371.1       199.9        87.9      59.5   10.8%              29.6      52.2
GDN core (conv, chunk delta, norm)                 344.6       343.5       172.1        75.9      57.1   10.4%               3.8      56.2
norms / hyperconnection                            346.1       348.2       171.5        69.6      57.1   10.4%              -3.9      58.0
dense: b12x other                                  296.6       297.6       152.7        67.2      49.7    9.0%               6.9      48.0
dense: GDN out_proj                                236.0       237.0       124.2        45.4      39.2    7.1%    28.9      -0.7      39.4
MTP drafter (prefill)                              189.2       192.6        95.2        35.8      31.3    5.7%              -4.0      32.3
shared expert + router                              82.5        81.6        38.2        14.5      13.2    2.4%    45.2      -4.0      14.2
dense: attention o_proj                             55.6        58.9        21.0         6.9       8.7    1.6%    43.4      -8.7      10.8
fused elementwise (inductor)                        36.3        36.5        17.7         7.8       6.0    1.1%              -0.1       6.0
sampler / bookkeeping                                8.9         8.1         7.7         7.2       1.9    0.4%               6.9       0.3
PLE embedding                                        8.5         9.1         4.3         1.8       1.4    0.3%              -0.1       1.5
dense: BF16 cuBLAS in compiled graph                 4.7         4.7         2.7         1.0       0.8    0.1%               0.2       0.8
step setup (inputs, KV, metadata)                    2.4         3.5         3.6         2.9       0.8    0.1%               3.2      -0.0
GPU busy (sum of kernels)                         3177.1      3241.2      1740.8       872.9     551.3                     260.0     487.8
GPU idle inside the step                           224.4       191.0       114.2        55.7      35.7
step span (first kernel to last)                  3400.3      3430.5      1854.4       925.7     586.6                     274.4     519.6
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== d01-min1: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       607.0       609.8       430.6       327.5     120.5   26.5%    39.1     253.1      58.7
attention (QSA indexer + sparse)                   319.0       370.4       198.5        87.4      59.5   13.1%              27.8      52.7
GDN core (conv, chunk delta, norm)                 349.1       350.4       176.3        75.6      58.1   12.8%               3.1      57.3
norms / hyperconnection                            346.2       347.5       176.2        71.1      57.4   12.6%               0.1      57.4
dense: b12x other                                  297.7       293.5       157.1        66.3      49.7   10.9%               9.0      47.5
MTP drafter (prefill)                              190.0       193.5        93.6        34.7      31.2    6.9%              -6.2      32.8
dense: GDN in_proj (qkvz+ba)                       144.7       142.3        73.6        30.5      23.9    5.3%   127.3       1.4      23.5
dense: GDN out_proj                                152.5       152.5        38.8        14.3      21.9    4.8%    51.8     -35.7      30.6
shared expert + router                              77.3        78.1        36.2        14.5      12.6    2.8%    47.5      -3.0      13.3
dense: attention o_proj                             57.3        60.8        21.2         7.2       8.9    2.0%    42.2      -9.1      11.2
fused elementwise (inductor)                        36.0        35.4        17.3         7.5       5.9    1.3%              -0.2       5.9
sampler / bookkeeping                                8.3         8.1         7.5         8.0       2.0    0.4%               7.7       0.1
PLE embedding                                        8.0         7.8         3.9         1.9       1.3    0.3%               0.2       1.3
dense: BF16 cuBLAS in compiled graph                 4.6         4.8         3.1         1.0       0.8    0.2%               0.4       0.7
step setup (inputs, KV, metadata)                    2.4         3.6         3.5         3.0       0.8    0.2%               3.3      -0.0
GPU busy (sum of kernels)                         2600.2      2658.7      1437.4       750.5     454.5                     251.9     393.0
GPU idle inside the step                           223.3       249.3       162.4        62.8      42.6
step span (first kernel to last)                  2822.5      2906.4      1599.2       810.6     496.7                     284.8     427.2
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== d01-ship: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       609.8       597.4       431.0       319.8     119.5   26.5%    39.5     248.3      58.9
attention (QSA indexer + sparse)                   315.2       373.9       199.6        89.0      59.7   13.2%              30.0      52.4
GDN core (conv, chunk delta, norm)                 347.2       351.2       176.4        75.7      58.0   12.8%               3.5      57.2
norms / hyperconnection                            342.5       340.9       174.7        71.2      56.7   12.6%               1.7      56.3
dense: b12x other                                  294.0       286.3       153.8        64.3      48.7   10.8%               7.9      46.8
MTP drafter (prefill)                              188.6       194.1        94.8        32.6      31.1    6.9%              -7.3      32.9
dense: GDN in_proj (qkvz+ba)                       143.5       146.0        74.6        31.7      24.2    5.3%   125.7       2.4      23.6
dense: GDN out_proj                                152.3       151.3        38.1        13.4      21.7    4.8%    52.3     -36.6      30.6
shared expert + router                              77.9        77.3        35.8        14.7      12.6    2.8%    47.6      -2.9      13.3
dense: attention o_proj                             56.6        60.1        21.3         7.1       8.9    2.0%    42.6      -8.8      11.0
fused elementwise (inductor)                        35.2        35.5        17.4         7.4       5.8    1.3%              -0.1       5.9
sampler / bookkeeping                                8.0         8.5         7.7         7.2       1.9    0.4%               7.0       0.2
PLE embedding                                        7.6         7.7         3.8         1.9       1.3    0.3%               0.3       1.2
dense: BF16 cuBLAS in compiled graph                 4.7         4.6         2.5         1.0       0.8    0.2%               0.1       0.8
step setup (inputs, KV, metadata)                    2.4         4.0         3.3         3.2       0.8    0.2%               3.3      -0.0
GPU busy (sum of kernels)                         2585.5      2638.9      1434.8       740.3     451.6                     248.7     390.9
GPU idle inside the step                           197.8       187.1       103.1        57.7      33.3
step span (first kernel to last)                  2782.3      2824.3      1537.2       795.3     484.6                     266.4     419.5
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== dgx-01: dense GEMM path and MTP (shipped 14 GiB pool; arms 6 GiB pool)
us/token                                             ship41           min0           min1           d min0         d min1
routed MoE (b12x NVFP4 W4A4)                          119.5          120.0          120.5             +0.5           +1.0
dense: GDN in_proj (qkvz+ba)                           24.2          104.5           23.9            +80.3           -0.3
attention (QSA indexer + sparse)                       59.7           59.5           59.5             -0.2           -0.1
GDN core (conv, chunk delta, norm)                     58.0           57.1           58.1             -0.9           +0.1
norms / hyperconnection                                56.7           57.1           57.4             +0.4           +0.7
dense: b12x other                                      48.7           49.7           49.7             +1.0           +1.0
dense: GDN out_proj                                    21.7           39.2           21.9            +17.6           +0.2
MTP drafter (prefill)                                  31.1           31.3           31.2             +0.2           +0.1
shared expert + router                                 12.6           13.2           12.6             +0.7           +0.0
dense: attention o_proj                                 8.9            8.7            8.9             -0.2           +0.1
fused elementwise (inductor)                            5.8            6.0            5.9             +0.2           +0.0
sampler / bookkeeping                                   1.9            1.9            2.0             +0.0           +0.0
PLE embedding                                           1.3            1.4            1.3             +0.2           +0.0
dense: BF16 cuBLAS in compiled graph                    0.8            0.8            0.8             +0.0           +0.0
step setup (inputs, KV, metadata)                       0.8            0.8            0.8             -0.0           -0.0
GPU busy                                              451.6          551.3          454.5            +99.6           +2.9
GPU idle in steps                                      33.3           35.7           42.6             +2.4           +9.3

== top kernels, d01-ship prefill chunks
      ms  calls  class | kernel | input dims of the innermost op
  1956.8    192  routed MoE (b12x NVFP4 W4A4) | kernel_cutlass_kernel_b12xmoe_sharedkernelssiluMoEDynamicKernelSilu_ob | [[98272, 1280, 1], [125788160], [15728640], [1, 2560], [81920], [81920], [1], [1], [1], [1], [1], [15355], [15355], [153
   781.5    873  dense: b12x other | kernel_cutlass_kernel_b12x_libdense_gemmDenseGemmKernel_object_at__Cop | [[6048, 10240], [336, 10240], [32, 4, 3, 4, 80, 1], [], [], [860160], [], [], [], []]
   771.5     48  attention (QSA indexer + sparse) | kernel_cutlass_kernel_b12xattentionpaged_selected_forward_SelectedPosi | [[], [6048, 24, 256], [6048, 4, 128], [6048, 128], [6048], [6048], [6048, 3], [8], [9], [8], [8], [201883904], [4568, 30
   563.2    380  norms / hyperconnection | kernel_cutlass_kernel_b12xnormhyperconnection_cute_PackedCombineNorm_o | [[6048, 10240], [6048, 2560], [6048, 4], [10240], [], []]
   379.2    288  dense: GDN in_proj (qkvz+ba) | kernel_cutlass_kernel_b12x_libdense_gemmDenseGemmKernel_object_at__Cop | [[6048, 2560], [16384, 2560], [32, 4, 128, 4, 20, 1], [], [], [22675712], [], [], [], []]
   304.0    388  norms / hyperconnection | _gate_mean_kernel | [[6048, 10240], [6048, 10240], [8192, 2560], []]
   303.2    144  dense: GDN out_proj | kernel_cutlass_kernel_b12x_libdense_gemmDenseGemmKernel_object_at__Cop | [[6048, 6144], [2560, 6144], [32, 4, 20, 4, 48, 1], [], [], [52068608], [], [], [], []]
   276.5      8  MTP drafter (prefill) | kernel_cutlass_kernel_b12xsequencemtp_feedback_cute_prefillMTPPrefillB | [[6048, 2560], [6048, 4, 2560], [2560], [10240], [2560, 2560], [2560, 2560], [251789312], [6048, 4, 2560], [], []]
   181.1   1152  GDN core (conv, chunk delta, norm) | void at::native::vectorized_gather_kernel<16, long>(char*, char*, long | [[6048, 10240], [], [6048, 10240], [], [6048, 10240]]
   178.7    288  GDN core (conv, chunk delta, norm) | kernel_cutlass_kernel_b12xsequence_shareddelta_prefill_cute_kernels_Re | [[6048, 16384], [6048, 96], [6048, 48, 128], []]
   172.0    288  GDN core (conv, chunk delta, norm) | kernel_cutlass_kernel_b12xsequence_shareddelta_prefill_cute_kernels_Pr | [[6048, 16384], [6048, 96], [6048, 48, 128], []]
   121.9     48  attention (QSA indexer + sparse) | kernel_cutlass_kernel_b12xattentionqsa_score_cute_RepresentativeScoreK | [[], [6048, 24, 256], [6048, 4, 128], [6048, 128], [6048], [6048], [6048, 3], [8], [9], [8], [8], [201883904], [4568, 30
   113.5    144  GDN core (conv, chunk delta, norm) | _causal_conv1d_fwd_kernel | [[6048, 16384], [6048, 96], [6048, 48, 128], []]
   104.8    432  GDN core (conv, chunk delta, norm) | Memcpy DtoD (Device -> Device) | [[6048, 10240], [6048, 10240], []]
    99.3     47  dense: attention o_proj | kernel_cutlass_kernel_b12x_libdense_gemmDenseGemmKernel_object_at__Cop | [[6048, 6144], [2560, 6144], [32, 4, 20, 4, 48, 1], [], [], [52068608], [], [], [], []]
    90.4    144  GDN core (conv, chunk delta, norm) | layer_norm_fwd_kernel | [[6048, 16384], [6048, 96], [6048, 48, 128], []]
    77.5    384  shared expert + router | kernel_cutlass_kernel_b12x_libdense_gemmDenseGemmKernel_object_at__Cop | [[6048, 2560], [1280, 2560], [32, 4, 10, 4, 20, 1], [], [], [21709056], [], [], [], []]
    68.9    288  GDN core (conv, chunk delta, norm) | _scatter_b12x_gdn_output | [[6048, 16384], [6048, 96], [6048, 48, 128], []]
    64.0      4  MTP drafter (prefill) | kernel_cutlass_kernel_b12xattentionpaged_selected_forward_SelectedPosi | [[], [6048, 24, 256], [6048, 4, 128], [6048, 128], [6048], [6048], [6048, 3], [8], [9], [8], [8], [201883904], [4568, 30
    51.8    144  dense: GDN out_proj | _quantize | [[6048, 6144], [2560, 6144], [32, 4, 20, 4, 48, 1], [], [], [52068608], [], [], [], []]
    45.5      4  MTP drafter (prefill) | kernel_cutlass_kernel_b12xmoe_sharedkernelsw4a16kernelW4A16FusedMoeKer | [[6048, 2560], [6048, 2560], [6048, 2560], [], [], [], []]
    36.4    144  GDN core (conv, chunk delta, norm) | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunc | [[6048, 48, 128], []]
    35.5     32  MTP drafter (prefill) | kernel_cutlass_kernel_b12x_libdense_gemmDenseGemmKernel_object_at__Cop | [[1, 2560], [248320, 2560], [32, 4, 1940, 4, 20, 1], [], [], [19865600], [], [], [], []]
    33.0    192  shared expert + router | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_imp | [[6048, 1], [6048, 2560]]
    32.5    140  norms / hyperconnection | triton_poi_fused_add_hyperconnection_combine_norm_slice_0 | [[6048, 2560], [6048, 2560], []]

boot facts per unit:
  d01-min0: tp1-d3-hf-20261010-21e0b201-5dad364d-warm MIN_TOKENS 0, MTP 4, KV 6 GiB, boot 243 s, torch compiled, 0 plans measured 
  d01-min1: tp1-d3-hf-20261010-21e0b201-5dad364d-warm MIN_TOKENS 1, MTP 4, KV 6 GiB, boot 202 s, torch compiled, 0 plans measured 
  d01-nomtp:  
  d01-ship: tp1-d3-hf-20261010-21e0b201-5dad364d-warm MIN_TOKENS 41, MTP 4, KV 14 GiB, boot 186 s, AOT loaded, 0 plans measured 
  d02-rb-prebake-try1:  
guard dgx01: min MemAvailable 11.88 GiB
guard dgx02: min MemAvailable 15.28 GiB
```

Traces stay on dgx-01 in `/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k81-1x-prefill-profile-20261011-0852` (prof-k81-*). Results: {RESULTS_URL}
