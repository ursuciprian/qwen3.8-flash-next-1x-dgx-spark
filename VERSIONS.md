# Versions

Release names for the single-Spark (TP=1) recipe `qwen3.8-flash-next-1x-dgx-spark`. The 2× Spark (TP=2) recipe has
its own list in the [tp-2 repo](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/blob/main/VERSIONS.md);
versions are per repo, so write the setup with the number when both appear together ("1× v2.1.0", "2× v1.5.0").

## Scheme

- **Release** `vMAJOR.MINOR.PATCH`, only for builds that became the default recipe. The date is the day the recipe PR
  that made it the default was merged.
  - MAJOR: only when users must act: new main-model weights to download (a different checkpoint, for example the
    GDN-MSE requant), a recipe rename, new required flags, a new minimum sparkrun.
  - MINOR: quality-gated speed or capability changes that need no action: drafter (even as a new HF revision),
    kernels, precision of activations, KV or the HC mixers, image, KV pool.
  - PATCH: fixes without speed claims: plan seed, recipe bug, boot fix.
  - Qwen4 will be v3.0.0 in both recipe repos.
- **Old names** (v2 to v3e) stay valid as aliases. They look like versions, so they are always written with "old
  name": "v2.1.0 (old name v3e)", or "v3d (old name)" on their own.
- **Drafters**: D0 = the checkpoint's original MTP drafter, D1 = refit run 1 (shipped), D2 = refit run 2 (never
  shipped), D3 = refit run 3a, D3b = refit run 3b.
- **Experiments**: `k<NN>-<slug>`, for example `k76-capability-matrix`; results folders
  start with the same name.
- **Images**: existing image tags keep their old form (`tp1-v3e-hf-20261008-21e0b201-5dad364d-warm`). Images built
  from k76 on also get a tag with the setup and the release, for example `1x-v2.2.0`.
- Each recipe header carries a line `# Release: v2.1.0 (old name v3e)`.

## Releases

Newest first. Every 1× release up to v2.1.0 was promoted in the tp-2 repo, before this repo was split out of it on
2026-10-08; PR numbers are in that repo. "Measured" is the release's own A/B against the one before it; details in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Release | Old name | Default since | Recipe PR (tp-2 repo) | What changed | Bump | Measured |
|---|---|---|---|---|---|---|
| **v2.2.0** | none | 2026-10-10 | [#24](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/pull/24) (this repo) | Retrained MTP drafter D3 (refit run 3a: more data, one epoch from D1), checkpoint revision `03f4a057` (only the drafter tensors in `model-00034` differ from `16c9bd54`) | MINOR: drafter only; users coming from v2.1.0 download one 4.5 GB shard | acceptance +0.021 to +0.027 per position at pos 2-4; k56c/k56d dgx-01 fresh c4 +3.4%, fresh c8 +3.3%, 16K c4 +4.6%; k80 on the published stack fresh c8 +2.1% (dgx-01), fresh c4 +2.6% (dgx-02); no cell worse beyond noise |
| v2.1.0 | v3e | 2026-10-08 | [#116](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/116) | Retrained MTP drafter D1, checkpoint revision `16c9bd54` (only the 24 drafter tensors differ from `244cb6fe`) | MINOR: drafter only; the new HF revision differs from `244cb6fe` only in the shard with the 24 drafter tensors | acceptance +0.036 to +0.058 per position; probe fresh c4 +4.9 / +7.9%, fresh c8 +6.8 / +8.1%, 16K c4 +5.5 / +5.7% (dgx-01 / dgx-02); no cell worse beyond noise |
| v2.0.0 | v3d | 2026-10-06 | [#91](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/91) | New checkpoint `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE`: GDN weights in NVFP4 for decode, MXFP8 copy for prefill | MAJOR: new checkpoint, numerics change | counting c8 +7.8%, 16K c4 +5.9%, fresh c8 +4.4%, tg512 c1 50.0 to 59.7 tok/s; pp2048 −0.9% (noise 1.0%) |
| v1.3.0 | v3c | 2026-10-05 | [#86](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/86) | Shared GDN prefill staging, compact GDN records, KV pool 6 to 14 GiB | MINOR: memory and KV only | coding 16k c8 22.1 to 109.3 tok/s; 16K c8 probe 347–370 s to 125 s; counting c8 +3.1%, fresh c4 +2.5% |
| v1.2.0 | v3b | 2026-10-04 | [#72](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/72) | PLE prefill read-ahead (`VLLM_PLE_MMAP_PREFILL_WILLNEED=1`) | MINOR: faster, same outputs | pp2048 +50% at 1 request, +19% at 4, +9% at 8; decode within noise |
| v1.1.0 | v3a | 2026-10-04 | [#60](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/60) | NVMe keepalive (`VLLM_PLE_MMAP_KEEPALIVE_MS=50`), compile cache in the image | MINOR: faster, same outputs | step −8.4% at 4 requests, −5.1% at 8, −8.0% at 16k c4; 1–2 requests −0.5 to −0.9% |
| v1.0.0 | v2 | 2026-10-02 | [#59](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2/pull/59) | First public `qwen3.8-flash-next-1x-dgx-spark`: PLE table through the page cache with WILLNEED before each decode gather, 131k-id draft vocab | First release | WILLNEED: step −16.9% fresh c1, −8 to −9% at 2–8 requests; draft vocab: step −2.7 to −4.3% at 1–4 |

Every release passed the quality gate. v1.1.0 was the default for about 8 hours and v1.3.0 for about 19; neither was
rolled back, each was replaced by the next release.

## Tags

The commits before 2026-10-08 are the rewritten copies of the tp-2 promote commits made when this repo was split out
(same recipe file, byte for byte). They have no `.sparkrun/registry.yaml`, so those tags mark history; run those
recipes by path or from the tp-2 registry.

| Release | Tag on commit |
|---|---|
| v2.2.0 | `12b6d520` (#24) |
| v2.1.0 | `73e3c1e1` (first commit with this repo's registry, the same day as #116) |
| v2.0.0 | `d08bda54` (copy of #91) |
| v1.3.0 | `302a013c` (copy of #86) |
| v1.2.0 | `eac8eb90` (copy of #72) |
| v1.1.0 | `771618fb` (copy of #60) |
| v1.0.0 | `5363d2e2` (copy of #59) |

## Manifests

Image tags are under `ghcr.io/ursuciprian/spark-vllm-b12x`. vLLM commits are in `ursuciprian/vllm`, b12x commits in
`ursuciprian/b12x`. Plan seed = the b12x plan file baked into the image (hash prefix, record count). The base image
commit is unknown for every release.

| Release | Image tag | Digest | vLLM | b12x | Checkpoint | Drafter | Plan seed |
|---|---|---|---|---|---|---|---|
| v2.2.0 | `tp1-d3-hf-20261010-21e0b201-5dad364d-warm` (also `1x-v2.2.0`) | `sha256:1c191c0a5f816750145e19e104ed812e9b3e63148b00fac901892f310a96a438` | `5dad364d05ac` | `21e0b201dd48` | `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` @ `03f4a0570496` | D3 | `33d9b32e` (289) |
| v2.1.0 | `tp1-v3e-hf-20261008-21e0b201-5dad364d-warm` | `sha256:5a9aa728ed6d2b984a40b16dbb51b572de0a1b988899147aca6ef43819e6eac9` | `5dad364d05ac` | `21e0b201dd48` | `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` @ `16c9bd54788d` | D1 | `3205a613` (289) |
| v2.0.0 | `tp1-v3d-hf-20261005-21e0b201-5dad364d-warm` | `sha256:81ac7975869814102843b4ca58e5ea219c3e938518b8f87d1a6fbb6edd89ede2` | `5dad364d05ac` | `21e0b201dd48` | `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` @ `244cb6fe99ff` | D0 | `f7894ea6` (289) |
| v1.3.0 | `tp1-v3c-20261005-21e0b201-50330171-warm` | `sha256:ba140406cabf0fbbbd13d0f605c42881c2442079619aa2fa7297eed2e0e31bff` | `503301710397` | `21e0b201dd48` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1a2d6` | D0 | `54b3e768` (302), per the Dockerfile; image not checked |
| v1.2.0 | `tp1-v3b-20261004-5bf24021-0632e506-warm` | `sha256:7308411dca81d1454cfc84725f87c9db80a6963ad7d4568a0255a046d51b90fa` | `0632e5069a88` | `5bf240217fc3` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1a2d6` | D0 | `54b3e768` (302) |
| v1.1.0 | `tp1-v3a-20261004-5bf24021-7fa812b3-warm` | `sha256:22d30f43f1a2bcaf13dea9a20c4df144ec8462c372e10322f4f112f718fe439a` | `7fa812b3310b` | `5bf240217fc3` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1a2d6` | D0 | `54b3e768` (302) |
| v1.0.0 | `tp1-20261002-5bf24021-884b4ff6-warm` | `sha256:d172e0eacd9765d31a5c57fdf0bdc9784b701d4b56f95b3f1e03814cb6163454` | `884b4ff68a90` | `5bf240217fc3` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1a2d6` | D0 | `54b3e768` (302) |

Recipe commits (tp-2 repo): v2.1.0 `a0a3c858`, v2.0.0 `030cbc20`, v1.3.0 `05099e20`, v1.2.0 `120e79a3`, v1.1.0
`5538c381`, v1.0.0 `8022b8b7`. Digests were read back from ghcr on 2026-10-09 and match the values recorded at
release time. Seed hashes and record counts come from the repo at each recipe commit; the image layers themselves
were not inspected.
