k95b result: the k95 upgrade-path boot of the 1x v2.3.0 candidate failed on a runtime cache from v2.2.0: both builds compute the same torch AOT keys (the b12x build is not part of the key), and the v2.2.0 graphs hold b12x plan handles that point at other plans in the new b12x (`ValueError: plan belongs to moe.decode, not norm.hyperconnection`). The recipe now gives v2.3.0 its own compile cache root (`VLLM_CACHE_ROOT=/cache/runtime/vllm-b12x-b45fe867`) and seeds the image's AOT cache there. Same image, same gate (the model, kernels and flags did not change).

- Fresh user (empty runtime cache, plain `sparkrun run` of the recipe): up in 307 s, torch AOT loaded from `/cache/runtime/vllm-b12x-b45fe867` (2 graphs, no Dynamo compile), 0 b12x plans measured, plan 33d9b32e at 303 records; usage on a repeated prompt: prompt_tokens=14239 cached_tokens=12096.
- Upgrade from v2.2.0 (my real runtime cache, v2.2.0 artifacts at the old root, plan at 303 records after k95): up in 197 s, AOT loaded from the new root, 0 plans measured, no Traceback, no ValueError.
- Rollback to the v2.2.0 recipe right after: up in 162 s, AOT loaded (its own artifacts at the old root), 0 plans measured, no ValueError.
- Logits (20 prompts, top-5 logprobs): fresh against upgrade boot, identical greedy tokens, 472 entries beyond 1e-3; two captures on the same fresh k95 boot already differ in 623 entries, so the difference is inside the self-noise of this capture.

Verdict: PASS.

Results: {RESULTS_URL}
