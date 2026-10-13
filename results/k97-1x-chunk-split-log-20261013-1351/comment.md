k97-chunk-split-log result: the prefill chunks the 1x v2.3.0 recipe really runs at `max_num_batched_tokens` 8192, 9072 and 12096, why each chunk ends where it does, and the peak memory of each budget. Cold 16K and 64K prompts (the same prompts at every budget), one Spark, the full 14 GiB KV pool. I logged the `_mamba_block_aligned_split` decisions that reach the stop selection (its early returns are not logged) and every scheduled prefill step through an env-gated mod loaded from the boot's runtime cache; the image is unchanged. No A/B: this only decides whether a budget A/B is worth running.

Summary: b8192:0K=1x[19];16K=4x[6048,6048,3024,1264];64K=12x | b9072:0K=1x[19];16K=4x[9072,3024,3024,1264];64K=9x | b12096:0K=1x[19];16K=3x[12096,3024,1264];64K=7x

```
== memory: min MemAvailable (GiB) and peak used (MemTotal - min available) per phase; guard aborts below 2 GiB
budget  phase      rc   seconds  min_avail_GiB  peak_used_GiB
  8192  boot        0      300           14.3          107.4
  8192  prompts     0       41           13.6          108.1
  8192  capture     0       16           13.9          107.8
  9072  boot        0      691            5.0          116.7
  9072  prompts     0       41           11.6          110.1
 12096  boot        0      708            4.6          117.1
 12096  prompts     0       42            8.8          112.9

== budget 8192: prefill chunks per cold prompt (start, tokens, why the chunk ends there)
  prompt 19 tokens: 1 steps [19]  block 3024
          0 +    19  prompt end
  prompt 16384 tokens: 4 steps [6048, 6048, 3024, 1264]  block 3024
          0 +  6048  block align @6048
       6048 +  6048  block align @12096
      12096 +  3024  stop last_cache @15120
      15120 +  1264  prompt end
  prompt 65536 tokens: 12 steps [6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 6048, 3024, 2032]  block 3024
          0 +  6048  block align @6048
       6048 +  6048  block align @12096
      12096 +  6048  block align @18144
      18144 +  6048  block align @24192
      24192 +  6048  block align @30240
      30240 +  6048  block align @36288
      36288 +  6048  block align @42336
      42336 +  6048  block align @48384
      48384 +  6048  block align @54432
      54432 +  6048  block align @60480
      60480 +  3024  stop last_cache @63504
      63504 +  2032  prompt end

== budget 9072: prefill chunks per cold prompt (start, tokens, why the chunk ends there)
  prompt 19 tokens: 1 steps [19]  block 3024
          0 +    19  prompt end
  prompt 16384 tokens: 4 steps [9072, 3024, 3024, 1264]  block 3024
          0 +  9072  budget
       9072 +  3024  stop reuse @12096
      12096 +  3024  stop last_cache @15120
      15120 +  1264  prompt end
  prompt 65536 tokens: 9 steps [9072, 9072, 9072, 9072, 9072, 9072, 6048, 3024, 2032]  block 3024
          0 +  9072  budget
       9072 +  9072  budget
      18144 +  9072  budget
      27216 +  9072  budget
      36288 +  9072  budget
      45360 +  9072  budget
      54432 +  6048  stop reuse @60480
      60480 +  3024  stop last_cache @63504
      63504 +  2032  prompt end

== budget 12096: prefill chunks per cold prompt (start, tokens, why the chunk ends there)
  prompt 19 tokens: 1 steps [19]  block 3024
          0 +    19  prompt end
  prompt 16384 tokens: 3 steps [12096, 3024, 1264]  block 3024
          0 + 12096  budget
      12096 +  3024  stop last_cache @15120
      15120 +  1264  prompt end
  prompt 65536 tokens: 7 steps [12096, 12096, 12096, 12096, 12096, 3024, 2032]  block 3024
          0 + 12096  budget
      12096 + 12096  budget
      24192 + 12096  budget
      36288 + 12096  budget
      48384 + 12096  budget
      60480 +  3024  stop last_cache @63504
      63504 +  2032  prompt end

VERDICT b8192:0K=1x[19];16K=4x[6048,6048,3024,1264];64K=12x | b9072:0K=1x[19];16K=4x[9072,3024,3024,1264];64K=9x | b12096:0K=1x[19];16K=3x[12096,3024,1264];64K=7x
```

How to read the reasons: `budget` = the chunk was capped by the token budget and ended on a block boundary; `stop last_cache` = the last cacheable block boundary of the prompt; `stop reuse` = a reachable prefix-reuse boundary of the recurrent state manager; `block align` = the end was floored to the mamba block size; `prompt end` = the tail of the prompt.

Boot facts per budget (a boot that failed at 14 GiB was retried after a 6 GiB prebake; both are listed):

```
b8192 boot:
  recipe: max_num_batched_tokens 8192, kv_cache_memory_bytes 15032385536
  boot: boot_s 300 pong pong 
  Directly load AOT: 2; Dynamo bytecode transform: 0
  b12x plans measured: 0
  Model loading took 75.39 GiB
  Initial free memory 114.03 GiB
  GPU KV cache size: 993,754 tokens
  K97MOD active lines: 5; k97 mod errors: 0; Traceback: 19
b9072 boot:
  recipe: max_num_batched_tokens 9072, kv_cache_memory_bytes 15032385536
  boot: boot_s 691 pong pong 
  Directly load AOT: 0; Dynamo bytecode transform: 2
  b12x plans measured: 7843
  Model loading took 75.72 GiB
  Initial free memory 113.98 GiB
  GPU KV cache size: 993,754 tokens
  K97MOD active lines: 5; k97 mod errors: 0; Traceback: 0
b12096 boot:
  recipe: max_num_batched_tokens 12096, kv_cache_memory_bytes 15032385536
  boot: boot_s 707 pong pong 
  Directly load AOT: 0; Dynamo bytecode transform: 2
  b12x plans measured: 8190
  Model loading took 76.89 GiB
  Initial free memory 113.69 GiB
  GPU KV cache size: 993,754 tokens
  K97MOD active lines: 5; k97 mod errors: 0; Traceback: 0
```

Routing and activation capture for the k96 and k98 probes (layer 5 and 47 routed-MoE inputs with their real top-10 ids, layer 5 HC GEMM inputs, one cold 16K prompt on the 8192 boot; files stay on the Spark):

```
capture prompt rc=0; files in /home/nvidia/GEN-AI/backlog/k97/capture:
      4 hcdown5
      4 hcup5
      4 moe47
      4 moe5
493M
```
