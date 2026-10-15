k101 result: I tested GDN prefill without token gathers (`VLLM_GDN_PREFILL_VIEWS=1`, vLLM `10d0acc85f` on the v2.3.0 image) against the shipped 1x v2.3.0, 4 boots per side on each Spark in the order ctl arm arm ctl ctl arm arm ctl. When a step has no speculative rows, the GDN layer reads the projections as views and writes its output in place, instead of gathering them by token index and scattering the output back. Same kernels and numerics; the GPU test checks bit identity against the gather path. Measure only; nothing ships from this job.

`RULE=WIN`. Per Spark: `gv-01=PASS gv-02=PASS`. Cold TTFT: `COLD_ALL=PASS`. Quality identity:` gv-01=UNSTABLE gv-02=UNSTABLE`. GDN data-movement kernels per cold 16K prompt (arm - control):` gv-01:-422.1 ms gv-02:-421.7 ms`. 2x: WIN.

Cold prefill is faster beyond boot noise at 16K on both Sparks and no decode, probe or acceptance cell is worse. Greedy completions: gv-01=UNSTABLE gv-02=UNSTABLE (UNSTABLE means the control boots do not reproduce their own T=0 completions, so identity rests on the GPU bit-identity test).

```
k101 2026-10-15 18:32:28 EEST: 1x v2.3.0 + VLLM_GDN_PREFILL_VIEWS=1 (spark-vllm-b12x:k101-gv-10d0acc8-b45fe867, vLLM 10d0acc85fb450472230558eda063c9059c20234) vs shipped, 4 boots per side per Spark
RULE=WIN

thunderdome exit 0 (DONE: gv=INCONCLUSIVE gv=INCONCLUSIVE); per Spark: gv-01=PASS gv-02=PASS; cold PASS; ident: gv-01=UNSTABLE gv-02=UNSTABLE; views: gv-01:ok gv-02:ok; GDN data movement per 16K: gv-01:-422.1 ms gv-02:-421.7 ms

== GPU tests (stage 1, k101 on dgx-02)
arm image:
  test_gdn_prefill_b12x.py::test_group_worklist_copies_update_captured_buffers_without_aliasing PASSED
  test_gdn_prefill_b12x.py::test_mixed_gdn_graph_replays_prefill_decode_verification_and_empty_worklists PASSED
  test_gdn_prefill_b12x.py::test_prefill_capacity_family_covers_exact_scheduler_limit[128-expected3] PASSED
  test_gdn_prefill_b12x.py::test_prefill_capacity_family_covers_exact_scheduler_limit[16-expected1] PASSED
  test_gdn_prefill_b12x.py::test_prefill_capacity_family_covers_exact_scheduler_limit[1-expected0] PASSED
  test_gdn_prefill_b12x.py::test_prefill_capacity_family_covers_exact_scheduler_limit[33-expected2] PASSED
  test_gdn_prefill_b12x.py::test_prefill_pooled_state_graph_replays_changed_lengths_and_slots PASSED
  test_gdn_prefill_b12x.py::test_prepared_gdn_decode_plan_is_scoped_to_the_bound_recurrent_pool PASSED
  test_gdn_prefill_views.py::test_identity_tokens_follow_worklist_copies PASSED
  test_gdn_prefill_views.py::test_identity_tokens_match_the_device_worklist PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides0-40] PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides1-64] PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides2-None] PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides3-None] PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides4-None] PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides5-None] PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides6-None] PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides7-None] PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides8-None] PASSED
  test_gdn_prefill_views.py::test_views_condition_is_host_only[overrides9-None] PASSED
  test_gdn_prefill_views.py::test_views_path_is_bit_identical_to_the_general_path PASSED
shipped image, pinned tests:
  test_gdn_prefill_b12x.py::test_group_worklist_copies_update_captured_buffers_without_aliasing PASSED
  test_gdn_prefill_b12x.py::test_mixed_gdn_graph_replays_prefill_decode_verification_and_empty_worklists PASSED
  test_gdn_prefill_b12x.py::test_prefill_capacity_family_covers_exact_scheduler_limit[128-expected3] PASSED
  test_gdn_prefill_b12x.py::test_prefill_capacity_family_covers_exact_scheduler_limit[16-expected1] PASSED
  test_gdn_prefill_b12x.py::test_prefill_capacity_family_covers_exact_scheduler_limit[1-expected0] PASSED
  test_gdn_prefill_b12x.py::test_prefill_capacity_family_covers_exact_scheduler_limit[33-expected2] PASSED
  test_gdn_prefill_b12x.py::test_prefill_pooled_state_graph_replays_changed_lengths_and_slots PASSED
  test_gdn_prefill_b12x.py::test_prepared_gdn_decode_plan_is_scoped_to_the_bound_recurrent_pool PASSED
pinned tests failing on the arm but passing on the control: none
TESTS=PASS

-- gv-01 (dgx-01)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.851 arm 0.851 shift -0.000
  pos 1: control 0.703 arm 0.702 shift -0.001
  pos 2: control 0.577 arm 0.573 shift -0.004
  pos 3: control 0.466 arm 0.467 shift +0.001
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      -1.73%  noise  3.99%            n=8 CI [-4.7,+1.0] control boot-to-boot +3.99%, step +0.21%
  probe  fresh-c4      +1.99%  noise  3.85%            n=32 CI [-0.2,+4.1] control boot-to-boot +3.85%, step -0.22%
  probe  fresh-c8      +0.15%  noise  1.16%            n=64 CI [-1.0,+1.3] control boot-to-boot +0.78%, step -0.27%
  benchy pp2048 (c1)   +4.22%  noise  2.42%  better    control 1792.3 arm 1867.9 t/s (boots 4/4)
  benchy tg512 (c1)    -5.91%  noise  6.27%            control 59.3 arm 55.8 t/s (boots 4/4)
  benchy tg512 (c4)    +1.20%  noise  6.76%            control 112.1 arm 113.5 t/s (boots 4/4)
  benchy tg512 (c8)    +1.02%  noise  5.96%            control 136.3 arm 137.7 t/s (boots 4/4)
== logits_equiv (reported, not judged)
better: benchy pp2048 (c1) +4.22% (noise 2.42%)
VERDICT=PROMOTE
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   +4.22%  CI [+1.69,+6.75]  control 1808.8 1765.3 1806.0 1789.1 | arm 1862.8 1864.0 1907.4 1837.4
  tg512 (c1)    -5.91%  CI [-10.31,-1.50]  control 59.5 57.5 59.3 61.2 | arm 54.6 57.0 57.1 54.8
  tg512 (c8)    +1.02%  CI [-4.04,+6.08]  control 140.4 136.6 135.8 132.3 | arm 140.5 141.2 136.9 132.0
== post checks per boot
  ctl-p3 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-01/ctl-p1: kv 993,754 ok
  ctl-p4 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-01/ctl-p2: kv 993,754 ok
  ctl-p5 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-01/ctl-p3: kv 993,754 ok
  ctl-p6 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-01/ctl-p4: kv 993,754 ok
  arm-p3 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-01/arm-p1: kv 993,754 ok
  arm-p4 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-01/arm-p2: kv 993,754 ok
  arm-p5 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-01/arm-p3: kv 993,754 ok
  arm-p6 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-01/arm-p4: kv 993,754 ok
worse: none
better: benchy pp2048 (c1)   +4.22%  noise  2.42%  better    control 1792.3 arm 1867.9 t/s (boots 4/4)
RULE=PASS
ident: 0 prompts stable across 4 control boots, arm identical on 0; 8 control-unstable (not judged)
  prompt 0: control boots disagree (4 variants), arm matches one: False
  prompt 1: control boots disagree (4 variants), arm matches one: False
  prompt 2: control boots disagree (4 variants), arm matches one: False
  prompt 3: control boots disagree (4 variants), arm matches one: False
  prompt 4: control boots disagree (4 variants), arm matches one: False
  prompt 5: control boots disagree (4 variants), arm matches one: False
  prompt 6: control boots disagree (4 variants), arm matches one: False
  prompt 7: control boots disagree (4 variants), arm matches one: False
IDENT=UNSTABLE
== GDN data-movement kernels, one cold 16K prompt (ctl-p1 vs arm-p1)
class                      ctl calls    ctl ms arm calls    arm ms arm-ctl ms
scatter (GDN output)             288      89.2         0       0.0      -89.2
gather (index_select)           1560     194.6       408       1.1     -193.5
memcpy DtoD                      976     121.4       544      15.6     -105.9
copy kernel                      152       0.3       584       3.3       +3.0
fill / memset                   1751      39.5      1607       2.9      -36.6
conv1d (kept)                    288     157.5       144     204.2      +46.7
gated norm (kept)                144      91.3       144     146.3      +55.0
GDN delta rule (kept)            720     385.0       720     379.8       -5.2
data movement (non-kept rows) arm - ctl: -422.1 ms
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 1 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 1 
  arm-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 1 
  arm-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 1 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 0 
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 0 
  ctl-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 0 
  ctl-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 0 

-- gv-02 (dgx-02)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.851 arm 0.857 shift +0.006
  pos 1: control 0.702 arm 0.709 shift +0.007
  pos 2: control 0.576 arm 0.583 shift +0.006
  pos 3: control 0.469 arm 0.476 shift +0.007
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      +1.62%  noise  3.80%            n=8 CI [-2.5,+5.1] control boot-to-boot +0.41%, step -0.29%
  probe  fresh-c4      +2.26%  noise  5.26%            n=32 CI [+0.3,+4.3] control boot-to-boot +5.26%, step +0.92%
  probe  fresh-c8      -0.51%  noise  1.24%            n=64 CI [-1.7,+0.6] control boot-to-boot -1.24%, step +0.23%
  benchy pp2048 (c1)   +3.39%  noise  1.25%  better    control 1786.7 arm 1847.4 t/s (boots 4/4)
  benchy tg512 (c1)    -1.33%  noise  5.06%            control 57.2 arm 56.4 t/s (boots 4/4)
  benchy tg512 (c4)    +0.83%  noise  9.07%            control 110.9 arm 111.9 t/s (boots 4/4)
  benchy tg512 (c8)    +1.75%  noise  7.57%            control 136.7 arm 139.1 t/s (boots 4/4)
== logits_equiv (reported, not judged)
better: benchy pp2048 (c1) +3.39% (noise 1.25%)
VERDICT=PROMOTE
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   +3.39%  CI [+2.57,+4.22]  control 1799.2 1776.9 1788.6 1782.1 | arm 1843.9 1851.4 1840.4 1853.7
  tg512 (c1)    -1.33%  CI [-8.45,+5.80]  control 57.4 55.8 58.7 56.7 | arm 58.6 53.3 58.7 55.0
  tg512 (c8)    +1.75%  CI [-3.92,+7.43]  control 132.7 132.2 139.4 142.6 | arm 136.2 141.8 139.7 138.8
== post checks per boot
  ctl-p3 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-02/ctl-p1: kv 993,754 ok
  ctl-p4 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-02/ctl-p2: kv 993,754 ok
  ctl-p5 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-02/ctl-p3: kv 993,754 ok
  ctl-p6 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-02/ctl-p4: kv 993,754 ok
  arm-p3 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-02/arm-p1: kv 993,754 ok
  arm-p4 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-02/arm-p2: kv 993,754 ok
  arm-p5 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-02/arm-p3: kv 993,754 ok
  arm-p6 -> k101s-1x-gdn-prefill-views-ab-20261015-1600/gv-02/arm-p4: kv 993,754 ok
worse: none
better: benchy pp2048 (c1)   +3.39%  noise  1.25%  better    control 1786.7 arm 1847.4 t/s (boots 4/4)
RULE=PASS
ident: 0 prompts stable across 4 control boots, arm identical on 0; 8 control-unstable (not judged)
  prompt 0: control boots disagree (4 variants), arm matches one: False
  prompt 1: control boots disagree (4 variants), arm matches one: False
  prompt 2: control boots disagree (4 variants), arm matches one: False
  prompt 3: control boots disagree (4 variants), arm matches one: False
  prompt 4: control boots disagree (4 variants), arm matches one: False
  prompt 5: control boots disagree (4 variants), arm matches one: False
  prompt 6: control boots disagree (4 variants), arm matches one: False
  prompt 7: control boots disagree (4 variants), arm matches one: False
IDENT=UNSTABLE
== GDN data-movement kernels, one cold 16K prompt (ctl-p1 vs arm-p1)
class                      ctl calls    ctl ms arm calls    arm ms arm-ctl ms
scatter (GDN output)             288      87.1         0       0.0      -87.1
gather (index_select)           1560     194.3       408       1.0     -193.2
memcpy DtoD                      976     122.5       544      15.5     -107.0
copy kernel                      152       0.3       584       3.2       +2.9
fill / memset                   1751      40.2      1607       2.9      -37.3
conv1d (kept)                    288     123.1       144     204.9      +81.8
gated norm (kept)                144      91.3       144     149.0      +57.7
GDN delta rule (kept)            720     389.5       720     378.0      -11.5
data movement (non-kept rows) arm - ctl: -421.7 ms
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 1 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 1 
  arm-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 1 
  arm-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 1 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 0 
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 0 
  ctl-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 0 
  ctl-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 views 0 

== cold TTFT
gv-01   16K: control   2128 arm   2227 tok/s (+4.7%, noise 1.0%) better
gv-01   64K: control   1987 arm   2068 tok/s (+4.1%, noise 1.4%) better
gv-01  128K: control   1756 arm   1820 tok/s (+3.6%, noise 1.0%) better
gv-01 COLD=PASS
gv-02   16K: control   2121 arm   2221 tok/s (+4.8%, noise 1.0%) better
gv-02   64K: control   2002 arm   2084 tok/s (+4.1%, noise 1.0%) better
gv-02  128K: control   1763 arm   1818 tok/s (+3.2%, noise 1.0%) better
gv-02 COLD=PASS
COLD_ALL=PASS
== gv-01: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2124 ctl-p2 2133 arm-p1 2216 arm-p2 2229, arm vs ctl +4.4%
  64K: ctl-p1 1973 ctl-p2 1981 arm-p1 2076 arm-p2 2076, arm vs ctl +5.0%
  128K: ctl-p1 1752 ctl-p2 1753 arm-p1 1819 arm-p2 1820, arm vs ctl +3.8%
== gv-02: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2121 ctl-p2 2129 arm-p1 2234 arm-p2 2208, arm vs ctl +4.5%
  64K: ctl-p1 2011 ctl-p2 2001 arm-p1 2087 arm-p2 2075, arm vs ctl +3.7%
  128K: ctl-p1 1767 ctl-p2 1762 arm-p1 1832 arm-p2 1813, arm vs ctl +3.3%

== profile gv-01 ctl-p1 (one cold 16K prompt, k81 classes)
== dp0_pp0_tp0_dcp0_ep0_rank0.1792071025804501288.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       596.1       661.0       417.6       318.6     121.7   25.5%    38.8     229.0      65.8
GDN core (conv, chunk delta, norm)                 353.8       455.8       177.9        76.2      64.9   13.6%             -20.4      69.9
attention (QSA indexer + sparse)                   314.9       443.5       198.9        87.7      63.8   13.4%              12.7      60.7
norms / hyperconnection                            358.5       376.4       181.7        72.9      60.4   12.7%              -4.7      61.5
dense: b12x other                                  297.9       400.4       157.2        66.1      56.2   11.8%             -15.3      60.0
MTP drafter (prefill)                              190.3       195.6        96.3        35.1      31.6    6.6%              -5.0      32.8
dense: GDN in_proj (qkvz+ba)                       147.8       252.0        78.0        31.6      31.1    6.5%    97.7     -20.8      36.2
dense: GDN out_proj                                 90.0       113.7        39.4        14.6      15.7    3.3%    72.0     -12.2      18.7
shared expert + router                              79.0        91.3        37.3        16.2      13.7    2.9%    43.8      -4.2      14.7
dense: attention o_proj                             37.0        47.1        21.5         7.8       6.9    1.5%    54.5      -0.7       7.1
fused elementwise (inductor)                        35.4        39.8        17.9         7.3       6.1    1.3%              -1.0       6.4
sampler / bookkeeping                                8.2         8.5         8.0         7.4       2.0    0.4%               7.3       0.2
PLE embedding                                        8.3         7.9         3.8         1.9       1.3    0.3%               0.1       1.3
dense: BF16 cuBLAS in compiled graph                 4.5         4.5         2.4         1.0       0.8    0.2%               0.2       0.7
step setup (inputs, KV, metadata)                    2.4         3.6         3.4         2.8       0.7    0.2%               3.1      -0.0
GPU busy (sum of kernels)                         2524.0      3101.1      1441.6       747.1     476.9                     168.0     435.9
GPU idle inside the step                           170.8       157.2        99.2        46.1      28.9
step span (first kernel to last)                  2693.7      3256.6      1540.1       790.3     505.4                     185.4     460.1
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile gv-01 arm-p1 (one cold 16K prompt, k81 classes)
== dp0_pp0_tp0_dcp0_ep0_rank0.1792072074867332069.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       647.5       616.1       413.3       317.3     121.7   26.6%    38.8     224.2      67.0
attention (QSA indexer + sparse)                   351.4       371.4       200.5        87.9      61.7   13.5%              21.8      56.4
norms / hyperconnection                            382.3       361.0       180.1        73.0      60.8   13.3%              -7.4      62.6
dense: b12x other                                  399.6       306.4       158.8        66.4      56.8   12.4%             -16.0      60.8
GDN core (conv, chunk delta, norm)                 365.7       212.4       106.9        48.7      44.8    9.8%             -30.2      52.2
MTP drafter (prefill)                              191.1       195.4        94.8        36.2      31.6    6.9%              -4.8      32.8
dense: GDN in_proj (qkvz+ba)                       245.3       150.5        83.3        34.3      31.3    6.9%    96.9     -14.6      34.9
dense: GDN out_proj                                112.4        90.6        39.0        14.6      15.7    3.4%    72.3     -12.2      18.6
shared expert + router                              92.6        80.7        37.2        14.9      13.8    3.0%    43.4      -6.2      15.3
dense: attention o_proj                             42.6        40.1        23.0         7.0       6.9    1.5%    54.8      -0.3       7.0
fused elementwise (inductor)                        38.7        36.2        17.9         7.9       6.1    1.3%              -0.4       6.2
PLE embedding                                       25.5         8.9         4.3         1.6       2.5    0.5%              -4.0       3.4
sampler / bookkeeping                                8.8         9.0         7.6         7.4       2.0    0.4%               6.8       0.3
dense: BF16 cuBLAS in compiled graph                 7.1         5.0         2.5         1.1       1.0    0.2%              -0.4       1.1
step setup (inputs, KV, metadata)                    2.4         3.8         3.2         3.3       0.8    0.2%               3.3      -0.0
GPU busy (sum of kernels)                         2913.0      2487.6      1372.5       721.7     457.4                     159.5     418.5
GPU idle inside the step                           167.4       156.8       117.8        44.0      29.7
step span (first kernel to last)                  3078.9      2642.8      1489.7       763.0     486.7                     185.6     441.4
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile gv-02 ctl-p1 (one cold 16K prompt, k81 classes)
== dp0_pp0_tp0_dcp0_ep0_rank0.1792071077860673662.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       598.8       602.5       459.3       317.4     120.7   25.7%    39.1     261.9      56.8
GDN core (conv, chunk delta, norm)                 355.4       356.9       244.0        76.0      63.0   13.4%              35.3      54.4
attention (QSA indexer + sparse)                   315.3       370.8       258.6        87.2      63.0   13.4%              59.1      48.5
norms / hyperconnection                            358.5       358.3       200.1        72.1      60.4   12.9%               8.1      58.4
dense: b12x other                                  297.1       292.7       225.0        65.9      53.8   11.4%              43.7      43.1
MTP drafter (prefill)                              190.7       196.6        95.4        36.2      31.7    6.7%              -4.8      32.8
dense: GDN in_proj (qkvz+ba)                       149.0       149.1       163.1        32.1      30.1    6.4%   100.9      46.2      18.8
dense: GDN out_proj                                 92.8        88.6        57.9        14.6      15.5    3.3%    73.1       2.3      14.9
shared expert + router                              79.6        80.3        47.1        14.5      13.5    2.9%    44.2       1.6      13.1
dense: attention o_proj                             36.8        40.4        23.9         7.2       6.6    1.4%    57.2       1.5       6.2
fused elementwise (inductor)                        35.9        35.6        20.8         7.3       6.1    1.3%               1.3       5.8
sampler / bookkeeping                                8.1         8.4         7.6         7.9       2.0    0.4%               7.5       0.1
PLE embedding                                        9.1         9.0        10.5         1.8       1.8    0.4%               3.0       1.1
dense: BF16 cuBLAS in compiled graph                 4.6         4.6         3.8         1.0       0.9    0.2%               0.8       0.7
step setup (inputs, KV, metadata)                    2.4         3.9         3.4         3.4       0.8    0.2%               3.6      -0.1
GPU busy (sum of kernels)                         2534.2      2597.5      1820.5       744.6     469.8                     471.1     354.8
GPU idle inside the step                           158.2       149.6        96.4        44.7      27.4
step span (first kernel to last)                  2691.3      2745.6      1916.3       786.6     496.8                     490.6     377.1
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== profile gv-02 arm-p1 (one cold 16K prompt, k81 classes)
== dp0_pp0_tp0_dcp0_ep0_rank0.1792072133403382621.pt.trace.json.gz: 4 prefill chunks (6048, 6048, 3024, 1264 tokens), GPU time per op class
op class                                       chunk1 ms   chunk2 ms   chunk3 ms   chunk4 ms  us/token  % busy  TFLOPS  fixed ms marg us/t
routed MoE (b12x NVFP4 W4A4)                       658.4       607.0       414.3       319.2     122.0   26.6%    38.7     226.1      66.8
attention (QSA indexer + sparse)                   350.4       374.8       200.0        88.7      61.9   13.5%              21.7      56.6
norms / hyperconnection                            385.2       363.1       178.4        72.6      61.0   13.3%              -9.7      63.4
dense: b12x other                                  409.1       304.1       156.6        66.2      57.1   12.5%             -18.9      61.8
GDN core (conv, chunk delta, norm)                 366.6       211.5       107.7        49.3      44.9    9.8%             -29.3      52.0
dense: GDN in_proj (qkvz+ba)                       253.8       152.8        83.2        32.8      31.9    7.0%    95.2     -18.5      36.4
MTP drafter (prefill)                              190.9       195.3        94.6        37.0      31.6    6.9%              -4.1      32.6
dense: GDN out_proj                                113.1        88.9        38.3        14.3      15.5    3.4%    72.8     -12.6      18.6
shared expert + router                              92.1        79.7        38.4        15.5      13.8    3.0%    43.4      -4.7      14.9
dense: attention o_proj                             40.1        40.3        20.1         6.7       6.5    1.4%    57.7      -1.7       6.9
fused elementwise (inductor)                        39.8        35.7        17.9         7.3       6.1    1.3%              -1.0       6.4
PLE embedding                                       26.3         8.1         4.9         1.7       2.5    0.5%              -3.7       3.4
sampler / bookkeeping                                8.7         8.3         8.4         7.3       2.0    0.4%               7.3       0.2
dense: BF16 cuBLAS in compiled graph                 6.9         4.9         2.3         1.1       0.9    0.2%              -0.5       1.0
step setup (inputs, KV, metadata)                    2.4         3.8         3.6         2.8       0.8    0.2%               3.1       0.0
GPU busy (sum of kernels)                         2943.8      2478.5      1368.6       722.5     458.6                     153.4     421.1
GPU idle inside the step                           170.9       180.5       119.2        50.2      31.8
step span (first kernel to last)                  3113.2      2657.4      1487.2       769.9     490.0                     179.8     446.1
fixed ms / marg us/t: least-squares line over the chunks, ms = fixed + marginal x chunk tokens (attention also grows with context)
GPU idle between prefill chunks (CPU scheduling under the profiler): 0.0, 0.0, 0.0 ms

== prebakes
-- prebake-dgx01/try1  boot 166 s, KV pool 6442450944 bytes, min MemAvailable 21.92 GiB       1 Directly load AOT compilation       1 Dynamo bytecode transform time plans measured: 0 views lines: 1 pong: pong 
-- prebake-dgx02/try1  boot 142 s, KV pool 6442450944 bytes, min MemAvailable 22.36 GiB       2 Directly load AOT compilation plans measured: 0 views lines: 1 pong: pong 

== 2x (TP=2): RULE2=WIN
k101 2x 2026-10-15 18:32:35 EEST: VLLM_GDN_PREFILL_VIEWS=1 on the 2x v2.1.0 (spark-vllm-b12x:k101-gv-10d0acc8-d21d7ade) vs shipped
bake: prebake boot 172s Dynamo bytecode transform time; full boot 172s Directly load AOT compilation; rank 0 measured 0
screen (ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4):
== cold TTFT (s, mean of 3 per boot); noise = larger boot-to-boot spread of either side (floor 0.5%)
  16K: ctl 5.64 5.60 5.65 5.63 | arm 5.44 5.54 5.47 5.50 -> -2.5% (-0.14 s; noise 1.9%) better
  64K: ctl 24.64 24.62 24.60 24.63 | arm 24.11 24.05 23.97 24.04 -> -2.4% (-0.58 s; noise 0.6%) better
  128K: ctl 55.29 55.28 55.22 55.29 | arm 53.82 54.22 53.82 54.30 -> -2.2% (-1.23 s; noise 0.9%) better
== llama-benchy pp2048 + tg512 (tok/s, side means; band = max(k92 band, side spread))
  pp2048 (c1): ctl 2758.0 arm 2892.4 +4.9% (band 4.2%)
  tg512 (c1): ctl 78.4 arm 84.9 +8.3% (band 15.5%)
  tg512 (c4): ctl 160.4 arm 166.2 +3.6% (band 10.6%)
  tg512 (c8): ctl 193.2 arm 195.2 +1.0% (band 11.7%)
== MTP acceptance per draft position (fresh c8, T=0), pooled per side
  ctl: 0.844 0.696 0.571 0.466
  arm: 0.850 0.704 0.576 0.470
== path: views lines (rank 0 / rank 1) and KV pool per boot
  ctl-p1: views lines 0 / 0; KV 3525263 tokens, 29.3 GiB available for KV
  ctl-p2: views lines 0 / 0; KV 3528961 tokens, 29.33 GiB available for KV
  ctl-p3: views lines 0 / 0; KV 3521751 tokens, 29.27 GiB available for KV
  ctl-p4: views lines 0 / 0; KV 3529885 tokens, 29.34 GiB available for KV
  arm-p1: views lines 1 / 1; KV 3521381 tokens, 29.27 GiB available for KV
  arm-p2: views lines 1 / 1; KV 3522120 tokens, 29.27 GiB available for KV
  arm-p3: views lines 1 / 1; KV 3524709 tokens, 29.29 GiB available for KV
  arm-p4: views lines 1 / 1; KV 3528406 tokens, 29.32 GiB available for KV
  KV pool arm vs ctl: -0.1% tokens, -0.02 GiB per rank
better: 16K, 64K, 128K
RULE2=WIN
```

Traces stay on dgx-01 in `/home/nvidia/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k101s-1x-gdn-prefill-views-ab-20261015-1600`. Results: {RESULTS_URL}
