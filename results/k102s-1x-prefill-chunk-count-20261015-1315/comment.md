k102s result (dgx01 dgx02): I logged every prefill split on the 1x v2.3.0 (`VLLM_DEBUG_PREFILL_SPLITS=1`, vLLM `820b9c57e0`) at max_num_batched_tokens 8192 12096, for cold 16K, 64K and 128K prompts, with the memory each budget needs, and ran a server A/B only for a budget that removes a step at 16K. Alignment, the mandatory recurrent-state stops and prefix caching are unchanged; only the budget moves.

`RULE=LOSS`. A/B budget: `12096`. Per Spark: `ab-01=FAIL(acceptance) ab-02=PASS`. Cold TTFT: `COLD_ALL=FAIL`.

At budget 12096 a cell or a cold length is worse beyond noise, so the larger budget does not meet my rule.

```
k102s 2026-10-15 15:55:29 EEST: prefill chunk count on 1x v2.3.0 (spark-vllm-b12x:k102-cl-820b9c57-b45fe867, vLLM 820b9c57e0ac84649a56191e0713801bce0f9edf), budgets 8192 12096
RULE=LOSS

== chunks per cold prompt (split log) and min MemAvailable (GiB: boot incl. compile/autotune, prompts; guard aborts < 2)
-- dgx01 budget 8192: mem 14.39 13.55; plans measured: 0 GPU KV cache size: 993,754 tokens Directly load AOT: 2; Dynamo bytecode transform: 0 reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) 
  prompt 16384: 4 steps [6048, 6048, 3024, 1264]
          0 +   6048  block-align
       6048 +   6048  block-align
      12096 +   3024  last-cache-position
      15120 +   1264  prompt-end
  prompt 65536: 12 steps [6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 3024, 2032]
          0 +   6048  block-align
       6048 +   6048  block-align
      12096 +   6048  block-align
      18144 +   6048  block-align
      24192 +   6048  block-align
      30240 +   6048  block-align
      36288 +   6048  block-align
      42336 +   6048  block-align
      48384 +   6048  block-align
      54432 +   6048  block-align
      60480 +   3024  last-cache-position
      63504 +   2032  prompt-end
  prompt 131072: 23 steps [6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 3024, 1040]
          0 +   6048  block-align
       6048 +   6048  block-align
      12096 +   6048  block-align
      18144 +   6048  block-align
      24192 +   6048  block-align
      30240 +   6048  block-align
      36288 +   6048  block-align
      42336 +   6048  block-align
      48384 +   6048  block-align
      54432 +   6048  block-align
      60480 +   6048  block-align
      66528 +   6048  block-align
      72576 +   6048  block-align
      78624 +   6048  block-align
      84672 +   6048  block-align
      90720 +   6048  block-align
      96768 +   6048  block-align
     102816 +   6048  block-align
     108864 +   6048  block-align
     114912 +   6048  block-align
     120960 +   6048  block-align
     127008 +   3024  last-cache-position
     130032 +   1040  prompt-end
-- dgx01 budget 12096: mem 4.36 9.35; plans measured: 7887 GPU KV cache size: 993,754 tokens Directly load AOT: 0; Dynamo bytecode transform: 2 reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) 
  prompt 16384: 3 steps [12096, 3024, 1264]
          0 +  12096  budget
      12096 +   3024  last-cache-position
      15120 +   1264  prompt-end
  prompt 65536: 7 steps [12096, 12096, 12096, 12096, 12096, 3024, 2032]
          0 +  12096  budget
      12096 +  12096  budget
      24192 +  12096  budget
      36288 +  12096  budget
      48384 +  12096  budget
      60480 +   3024  last-cache-position
      63504 +   2032  prompt-end
  prompt 131072: 13 steps [12096, 12096, 12096, 12096, 12096, 12096, 12096, 12096, 12096, 12096, 6048, 3024, 1040]
          0 +  12096  budget
      12096 +  12096  budget
      24192 +  12096  budget
      36288 +  12096  budget
      48384 +  12096  budget
      60480 +  12096  budget
      72576 +  12096  budget
      84672 +  12096  budget
      96768 +  12096  budget
     108864 +  12096  budget
     120960 +   6048  reachable
     127008 +   3024  last-cache-position
     130032 +   1040  prompt-end
-- dgx02 budget 8192: mem 14.88 13.94; plans measured: 0 GPU KV cache size: 993,754 tokens Directly load AOT: 2; Dynamo bytecode transform: 0 reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) 
  prompt 16384: 4 steps [6048, 6048, 3024, 1264]
          0 +   6048  block-align
       6048 +   6048  block-align
      12096 +   3024  last-cache-position
      15120 +   1264  prompt-end
  prompt 65536: 12 steps [6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 3024, 2032]
          0 +   6048  block-align
       6048 +   6048  block-align
      12096 +   6048  block-align
      18144 +   6048  block-align
      24192 +   6048  block-align
      30240 +   6048  block-align
      36288 +   6048  block-align
      42336 +   6048  block-align
      48384 +   6048  block-align
      54432 +   6048  block-align
      60480 +   3024  last-cache-position
      63504 +   2032  prompt-end
  prompt 131072: 23 steps [6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 3024, 1040]
          0 +   6048  block-align
       6048 +   6048  block-align
      12096 +   6048  block-align
      18144 +   6048  block-align
      24192 +   6048  block-align
      30240 +   6048  block-align
      36288 +   6048  block-align
      42336 +   6048  block-align
      48384 +   6048  block-align
      54432 +   6048  block-align
      60480 +   6048  block-align
      66528 +   6048  block-align
      72576 +   6048  block-align
      78624 +   6048  block-align
      84672 +   6048  block-align
      90720 +   6048  block-align
      96768 +   6048  block-align
     102816 +   6048  block-align
     108864 +   6048  block-align
     114912 +   6048  block-align
     120960 +   6048  block-align
     127008 +   3024  last-cache-position
     130032 +   1040  prompt-end
-- dgx02 budget 12096: mem 11.51 9.70; plans measured: 0 GPU KV cache size: 993,754 tokens Directly load AOT: 1; Dynamo bytecode transform: 1 reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) reachable boundaries [16383] -> [15120, 12096] (lookup alignment 3024, state alignment 3024, eagle True, lookup drops eagle block True) 
  prompt 16384: 3 steps [12096, 3024, 1264]
          0 +  12096  budget
      12096 +   3024  last-cache-position
      15120 +   1264  prompt-end
  prompt 65536: 7 steps [12096, 12096, 12096, 12096, 12096, 3024, 2032]
          0 +  12096  budget
      12096 +  12096  budget
      24192 +  12096  budget
      36288 +  12096  budget
      48384 +  12096  budget
      60480 +   3024  last-cache-position
      63504 +   2032  prompt-end
  prompt 131072: 13 steps [12096, 12096, 12096, 12096, 12096, 12096, 12096, 12096, 12096, 12096, 6048, 3024, 1040]
          0 +  12096  budget
      12096 +  12096  budget
      24192 +  12096  budget
      36288 +  12096  budget
      48384 +  12096  budget
      60480 +  12096  budget
      72576 +  12096  budget
      84672 +  12096  budget
      96768 +  12096  budget
     108864 +  12096  budget
     120960 +   6048  reachable
     127008 +   3024  last-cache-position
     130032 +   1040  prompt-end
== steps per cold prompt (dgx01 dgx02)
    8192:  16K 4 4  64K 12 12 128K 23 23
   12096:  16K 3 3  64K 7 7 128K 13 13
A/B budget: 12096 (16K steps 3 vs 4 at 8192)

budget 12096: thunderdome exit 0 (DONE: b12096=KILL b12096=INCONCLUSIVE); per Spark: ab-01=FAIL(acceptance) ab-02=PASS; cold FAIL; chunk check: ok

== CPU tests (k102 image)
test_mamba_prefill_split_log.py::test_debug_log_is_off_by_default PASSED
test_mamba_prefill_split_log.py::test_debug_log_names_each_stop PASSED
test_mamba_prefill_split_log.py::test_predicted_schedules[131072-12096-steps5] PASSED
test_mamba_prefill_split_log.py::test_predicted_schedules[131072-8192-steps4] PASSED
test_mamba_prefill_split_log.py::test_predicted_schedules[16384-12096-steps0] PASSED
test_mamba_prefill_split_log.py::test_predicted_schedules[65536-12096-steps3] PASSED
test_mamba_prefill_split_log.py::test_predicted_schedules[65536-8192-steps1] PASSED
test_mamba_prefill_split_log.py::test_predicted_schedules[65536-9072-steps2] PASSED
test_mamba_prefill_split_log.py::test_reproduces_the_profiled_16k_schedules PASSED
test_mamba_sparse_cleanup.py::test_async_sparse_cleanup[256-0] PASSED
test_mamba_sparse_cleanup.py::test_async_sparse_cleanup[256-3] PASSED
test_mamba_sparse_cleanup.py::test_async_sparse_cleanup[256-7] PASSED
test_mamba_sparse_cleanup.py::test_async_sparse_cleanup[64-0] PASSED
test_mamba_sparse_cleanup.py::test_async_sparse_cleanup[64-3] PASSED
test_mamba_sparse_cleanup.py::test_async_sparse_cleanup[64-7] PASSED
test_mamba_sparse_cleanup.py::test_cleanup_does_not_rescan_history_or_advance_past_table PASSED

-- ab-01 (dgx-01)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.850 arm 0.851 shift +0.001
  pos 1: control 0.703 arm 0.702 shift -0.001
  pos 2: control 0.577 arm 0.574 shift -0.003
  pos 3: control 0.468 arm 0.467 shift -0.001
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      +1.12%  noise  6.61%            n=8 CI [-3.4,+4.8] control boot-to-boot +6.61%, step -0.52%
  probe  fresh-c4      +0.17%  noise  1.90%            n=32 CI [-1.6,+2.2] control boot-to-boot +0.54%, step -0.01%
  probe  fresh-c8      +1.43%  noise  2.70%            n=64 CI [+0.5,+2.4] control boot-to-boot +2.70%, step -0.59%
  benchy pp2048 (c1)   -6.11%  noise  2.05%  WORSE>2x  control 1787.0 arm 1677.9 t/s (boots 4/4)
  benchy tg512 (c1)    -0.06%  noise  8.07%            control 57.5 arm 57.5 t/s (boots 4/4)
  benchy tg512 (c4)    +3.06%  noise  6.13%            control 109.7 arm 113.1 t/s (boots 4/4)
  benchy tg512 (c8)    +4.91%  noise 11.83%            control 134.4 arm 141.0 t/s (boots 4/4)
== logits_equiv (reported, not judged)
reason: benchy pp2048 (c1) -6.11% (noise 2.05%)
better: none
VERDICT=KILL
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   -6.11%  CI [-7.48,-4.73]  control 1788.6 1782.6 1770.1 1806.8 | arm 1674.8 1687.2 1662.9 1686.7
  tg512 (c1)    -0.06%  CI [-5.54,+5.42]  control 56.5 55.2 59.9 58.3 | arm 58.3 57.6 58.0 56.0
  tg512 (c8)    +4.91%  CI [-3.51,+13.33]  control 134.9 140.1 124.2 138.2 | arm 141.9 139.0 136.9 146.1
== post checks per boot
  ctl-p3 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-01/ctl-p1: kv 993,754 ok
  ctl-p4 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-01/ctl-p2: kv 993,754 ok
  ctl-p5 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-01/ctl-p3: kv 993,754 ok
  ctl-p6 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-01/ctl-p4: kv 993,754 ok
  arm-p3 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-01/arm-p1: kv 993,754 ok
  arm-p4 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-01/arm-p2: kv 993,754 ok
  arm-p5 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-01/arm-p3: kv 993,754 ok
  arm-p6 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-01/arm-p4: kv 993,754 ok
worse: benchy pp2048 (c1)   -6.11%  noise  2.05%  WORSE>2x  control 1787.0 arm 1677.9 t/s (boots 4/4)
better: none
RULE=FAIL
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 12096 chunks16384 12096,3024,1264 chunks65536 12096,12096,12096,12096,12096,3024,2032 chunks131072 12096,12096,12096,12096,12096,12096,12096,12096,12096,12096,6048,3024,1040 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 12096 chunks16384 12096,3024,1264 chunks65536 12096,12096,12096,12096,12096,3024,2032 chunks131072 12096,12096,12096,12096,12096,12096,12096,12096,12096,12096,6048,3024,1040 
  arm-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 12096 chunks16384 12096,3024,1264 chunks65536 12096,12096,12096,12096,12096,3024,2032 chunks131072 12096,12096,12096,12096,12096,12096,12096,12096,12096,12096,6048,3024,1040 
  arm-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 12096 chunks16384 12096,3024,1264 chunks65536 12096,12096,12096,12096,12096,3024,2032 chunks131072 12096,12096,12096,12096,12096,12096,12096,12096,12096,12096,6048,3024,1040 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 chunks16384  chunks65536  chunks131072  
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 chunks16384  chunks65536  chunks131072  
  ctl-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 chunks16384  chunks65536  chunks131072  
  ctl-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 chunks16384  chunks65536  chunks131072  

-- ab-02 (dgx-02)
== acceptance per draft position (T=0 probe cells, pooled)
  pos 0: control 0.854 arm 0.853 shift -0.001
  pos 1: control 0.702 arm 0.706 shift +0.004
  pos 2: control 0.575 arm 0.582 shift +0.007
  pos 3: control 0.469 arm 0.476 shift +0.007
== cells: arm vs control % (noise = the cell's noise band, %)
  probe  fresh-c1      -1.33%  noise  3.81%            n=8 CI [-5.0,+2.7] control boot-to-boot +0.37%, step -0.21%
  probe  fresh-c4      +1.47%  noise  2.19%            n=32 CI [-0.7,+3.7] control boot-to-boot -1.03%, step +0.50%
  probe  fresh-c8      +0.80%  noise  1.25%            n=64 CI [-0.5,+2.0] control boot-to-boot -1.25%, step +0.69%
  benchy pp2048 (c1)   -0.33%  noise  2.66%            control 1798.6 arm 1792.7 t/s (boots 4/4)
  benchy tg512 (c1)    -7.43%  noise 10.15%            control 59.5 arm 55.1 t/s (boots 4/4)
  benchy tg512 (c4)    +0.25%  noise  5.19%            control 111.6 arm 111.9 t/s (boots 4/4)
  benchy tg512 (c8)    +5.55%  noise  4.26%  better    control 140.0 arm 147.8 t/s (boots 4/4)
== logits_equiv (reported, not judged)
better: benchy tg512 (c8) +5.55% (noise 4.26%)
VERDICT=PROMOTE
== boot-level Welch 95% CI, benchy cells (reported; the rule uses the noise bands above)
  pp2048 (c1)   -0.33%  CI [-2.26,+1.61]  control 1803.2 1824.8 1789.3 1776.9 | arm 1781.2 1776.0 1815.1 1798.4
  tg512 (c1)    -7.43%  CI [-14.80,-0.05]  control 60.5 61.6 60.3 55.6 | arm 54.7 53.3 58.1 54.1
  tg512 (c8)    +5.55%  CI [+2.14,+8.96]  control 137.4 143.4 138.5 140.6 | arm 145.9 151.5 145.9 147.8
== post checks per boot
  ctl-p3 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-02/ctl-p1: kv 993,754 ok
  ctl-p4 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-02/ctl-p2: kv 993,754 ok
  ctl-p5 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-02/ctl-p3: kv 993,754 ok
  ctl-p6 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-02/ctl-p4: kv 993,754 ok
  arm-p3 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-02/arm-p1: kv 993,754 ok
  arm-p4 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-02/arm-p2: kv 993,754 ok
  arm-p5 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-02/arm-p3: kv 993,754 ok
  arm-p6 -> k102s-1x-prefill-chunk-count-20261015-1315/ab-02/arm-p4: kv 993,754 ok
worse: none
better: benchy tg512 (c8)    +5.55%  noise  4.26%  better    control 140.0 arm 147.8 t/s (boots 4/4)
RULE=PASS
  arm-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 12096 chunks16384 12096,3024,1264 chunks65536 12096,12096,12096,12096,12096,3024,2032 chunks131072 12096,12096,12096,12096,12096,12096,12096,12096,12096,12096,6048,3024,1040 
  arm-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 12096 chunks16384 12096,3024,1264 chunks65536 12096,12096,12096,12096,12096,3024,2032 chunks131072 12096,12096,12096,12096,12096,12096,12096,12096,12096,12096,6048,3024,1040 
  arm-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 12096 chunks16384 12096,3024,1264 chunks65536 12096,12096,12096,12096,12096,3024,2032 chunks131072 12096,12096,12096,12096,12096,12096,12096,12096,12096,12096,6048,3024,1040 
  arm-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 12096 chunks16384 12096,3024,1264 chunks65536 12096,12096,12096,12096,12096,3024,2032 chunks131072 12096,12096,12096,12096,12096,12096,12096,12096,12096,12096,6048,3024,1040 
  ctl-p1 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 chunks16384  chunks65536  chunks131072  
  ctl-p2 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 chunks16384  chunks65536  chunks131072  
  ctl-p3 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 chunks16384  chunks65536  chunks131072  
  ctl-p4 hook: plans_measured 0 kv GPU KV cache size: 993,754 tokens aot Directly load AOT compilation mnbt max_num_batched_tokens': 8192 chunks16384  chunks65536  chunks131072  

== cold TTFT
ab-01   16K: control   2133 arm   1743 tok/s (-18.3%, noise 1.6%) WORSE
ab-01   64K: control   2003 arm   1829 tok/s (-8.7%, noise 1.3%) WORSE
ab-01  128K: control   1767 arm   1658 tok/s (-6.2%, noise 1.2%) WORSE
ab-01 COLD=FAIL
ab-02   16K: control   2121 arm   1848 tok/s (-12.9%, noise 1.0%) WORSE
ab-02   64K: control   2016 arm   1870 tok/s (-7.3%, noise 1.0%) WORSE
ab-02  128K: control   1762 arm   1693 tok/s (-3.9%, noise 1.0%) WORSE
ab-02 COLD=FAIL
COLD_ALL=FAIL
== ab-01: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2152 ctl-p2 2133 arm-p1 1764 arm-p2 1757, arm vs ctl -17.8%
  64K: ctl-p1 2013 ctl-p2 2010 arm-p1 1830 arm-p2 1829, arm vs ctl -9.0%
  128K: ctl-p1 1775 ctl-p2 1769 arm-p1 1652 arm-p2 1667, arm vs ctl -6.3%
== ab-02: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)
  16K: ctl-p1 2114 ctl-p2 2126 arm-p1 1819 arm-p2 1854, arm vs ctl -13.4%
  64K: ctl-p1 2007 ctl-p2 2022 arm-p1 1875 arm-p2 1868, arm vs ctl -7.1%
  128K: ctl-p1 1754 ctl-p2 1768 arm-p1 1683 arm-p2 1695, arm vs ctl -4.1%
```

Results: {RESULTS_URL}
