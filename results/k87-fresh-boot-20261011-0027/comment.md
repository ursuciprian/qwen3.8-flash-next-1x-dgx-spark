Fresh-user boot of 1x v2.2.0 and the v2.2.0 TTFT cell at ISL 16K (k87-fresh-boot).

(a) I ran the README quick start on dgx-02 as a new user would: a fresh sparkrun HOME with no registries and an empty runtime cache dir (a job-local copy; my real cache was not touched), then `sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark` and `sparkrun run qwen3.8-flash-next-1x-dgx-spark --hosts <spark> --solo` (with `--no-follow` so the job does not attach to the log). Checkpoint and image were already on disk, which is the README's "model and image on disk" case. Time is from the start of `sparkrun run` to the first `/health` 200.

```
recipe: Container:    ghcr.io/ursuciprian/spark-vllm-b12x:tp1-d3-hf-20261010-21e0b201-5dad364d-warm
sparkrun run returned after 29 s; /health 200 after 298 s from the start of sparkrun run; pong: pong
runtime cache mount: /home/nvidia/GEN-AI/backlog/k87/fresh-20261011-0028/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923 -> /cache/runtime
TritonBundler warning lines: 19
Traceback lines: 19
ERROR lines: 0
AOT backbone: 2 'Directly load AOT compilation' line(s)
runtime cache after boot: 726M
first TritonBundler lines:
(EngineCore pid=543) [rank0]:W1010 21:32:17.109000 543 torch/_inductor/triton_bundler.py:242] RuntimeError: ('Cubin file saved by TritonBundler not found at %s', '/cache/runtime/triton/SYN7JF4PJN4SJSYPXUX4RACLS5CF67FDL5JJVPQDGMSLNJ3UM7NA/tr
(EngineCore pid=543) [rank0]:W1010 21:32:17.178000 543 torch/_inductor/triton_bundler.py:242] RuntimeError: ('Cubin file saved by TritonBundler not found at %s', '/cache/runtime/triton/5STXNF6ZSHPUVNG5SNUJQJ4WHC37XLHVLYG2AHMJXOHUUQHPAQQA/tr
(EngineCore pid=543) [rank0]:W1010 21:32:17.186000 543 torch/_inductor/triton_bundler.py:242] RuntimeError: ('Cubin file saved by TritonBundler not found at %s', '/cache/runtime/triton/S3FXRMX26UOXUBKBTF7KJ4TOIDL6ZPV6ORH6JE7UMO326S6K4HXQ/tr
(EngineCore pid=543) [rank0]:W1010 21:32:17.193000 543 torch/_inductor/triton_bundler.py:242] RuntimeError: ('Cubin file saved by TritonBundler not found at %s', '/cache/runtime/triton/5RMQ2QTUDS42HGABENHR7KTUO2VWAE5YKV4FQVRDPHSAMHHR6TEQ/tr
(EngineCore pid=543) [rank0]:W1010 21:32:17.197000 543 torch/_inductor/triton_bundler.py:242] RuntimeError: ('Cubin file saved by TritonBundler not found at %s', '/cache/runtime/triton/4RKARYAVD3NZBNG2JTNP6FAS7GKOZAI77YJNTR746YCOQ4OZPHCQ/tr
```

(b) TTFT at ISL 16K on 1x v2.2.0 with the normal runtime cache, both Sparks at once: llama-benchy pp 16384, tg 32, depth 0, c1, 3 runs, no warm-up run, a new prompt each run (cold prefix), the same arguments as the pp sweep of k76. The 2x v2.0.0 cell is the k76 value (5.644 s, sd 0.012), measured on the shipped v2.0.0 image, so I did not rerun it.

```
dgx01: TTFT 8.234 s (sd 0.238; runs 8.46, 8.26, 7.99), prefill 1998 tok/s
dgx02: TTFT 8.163 s (sd 0.484; runs 8.59, 8.27, 7.64), prefill 2019 tok/s
pooled: TTFT 8.199 s (sd 0.343, 6 runs on 2 Sparks), prefill 2008 tok/s
```

Results: {RESULTS_URL}
