k95-1x-ship-check result: I checked the 1x v2.3.0 candidate (v2.2.0 on the b12x large-row tile-fix image, plus `--enable-prompt-tokens-details`) on one Spark. Image `ghcr.io/ursuciprian/spark-vllm-b12x:k95-20261013-b45fe867-5dad364d-warm`.

- Fresh-user boot (a new sparkrun HOME with an empty runtime cache, plain `sparkrun run` of the recipe): up in 304 s, torch AOT loaded from the image (2 graphs, no Dynamo compile), 0 b12x plans measured in every group (`scripts/check_seed.py` PASS), plan 33d9b32e at 303 records, 72 MXFP8 copies.
- Usage field: on the second of two identical prompts, prompt_tokens=14239 cached_tokens=12096.
- Gate part on this Spark: PASS: hardmode 91, TC-45 100.0, straggler preemptions 0/0/0/0/0/0. Long-context retrieval runs on the other Spark (k95f).
- llama-benchy on the same boot: pp2048 c1 1791, tg512 c1 56.3, tg512 c8 137.2; the tile-fix arm of k90b had pp2048 c1 1773 to 1789, tg512 c1 57.5 to 57.9 and c8 135.4 to 136.2 tok/s.
- Upgrade path (the runtime cache of a v2.2.0 user, which holds v2.2.0's torch AOT artifacts under the same keys): boot failed, CHECK; no logits (the server never came up).

Verdict: the fresh-user boot and my gate half pass, but the upgrade path FAILS: on a v2.2.0 runtime cache the engine dies at load with `ValueError: plan belongs to moe.decode, not norm.hyperconnection`. Both builds compute the same torch AOT keys (the b12x build is not part of the key), so the v2.2.0 graphs, which hold b12x plan handles, are loaded with the new b12x. v2.3.0 does not ship like this; k95b rechecks it with a compile cache root of its own.

Results: {RESULTS_URL}
