k106s-1x-ship-check result: I built the 1x v2.4.0 release image and booted it the way a new user would, with a plain `sparkrun run` of the release recipe on one Spark (1x v2.3.0 plus the HC down projection BF16 path, `VLLM_GDN_PREFILL_VIEWS=1` and a compile cache root of its own).

- Image `ghcr.io/ursuciprian/spark-vllm-b12x:k106s-20261017-a1ad1b69-5dad364d-10d0acc8-warm`, digest `sha256:30fe678ed7ae5d4deb277be137497bcf597e46e829471b7bed24edd4294a84f1`. It is the image the A/B, the c1 recheck and the gate ran on, plus one layer with the torch AOT cache of the A/B arm boots. Every layer of the tested image is in it unchanged, and the vLLM and b12x code is byte-identical to the tested image. The second Spark pulled it from ghcr and got the same image.
- Before the boot I moved the plan file and the new compile cache root out of the runtime cache, so the image had to supply them. Up in 256 s, WARM (compile cache loaded from the image, no compile), seed check PASS, plan records 303. The GDN views path is on and the KV pool is 993,754 tokens.
- Pong and `/v1/models` answer; the server runs `--revision 03f4a057`.
- T=0 fresh-c8 acceptance per draft position: 0.856 0.707 0.580 0.467 (A/B arm: 0.849 0.700 0.577 0.464).
- llama-benchy (T=1, 3 runs, boots 1) against the A/B arm means: pp2048 (c1) 1901.5 -> 1901.5 (-0.7% vs k103 arm 1915.6, band 1.0%); tg512 (c1) 60.7 -> 60.7 (+3.9% vs k103 arm 58.4, band 9.04%); tg512 (c8) 144.3 -> 144.3 (+6.0% vs k103 arm 136.2, band 7.13%). For reference, not judged: the c1 recheck arm on the same image had pp2048 c1 1887.0 and tg512 c1 59.0.

Verdict: PASS. The quality gate is k103g (same image code, weights, plans and flag).

Results: {RESULTS_URL}
