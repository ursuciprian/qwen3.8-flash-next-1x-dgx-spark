# Renames

The single-Spark recipe names over time. The recipes moved here from
[qwen3.8-flash-next-dgx-spark-tp-2](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2) on 2026-10-08
with the same names; that repo keeps a copy of both for existing `sparkrun run` users.

## 2026-10-08: single-Spark v3e promoted

| name | content now |
|---|---|
| `qwen3.8-flash-next-1x-dgx-spark` | v3e: v3d + the retrained MTP drafter, checkpoint `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` @ `16c9bd54`, image `tp1-v3e-hf-20261008-21e0b201-5dad364d-warm` (experimental) |
| `qwen3.8-flash-next-1x-dgx-spark-previous` | the v3d recipe that was `qwen3.8-flash-next-1x-dgx-spark` from 2026-10-05 to 2026-10-08 |
| (was `-previous`, v3c) | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v3c-20261005.yaml` (not listed; run by path) |

## 2026-10-05: single-Spark v3d promoted

| name | content now |
|---|---|
| `qwen3.8-flash-next-1x-dgx-spark` | v3d: v3c + NVFP4 GDN weights for decode and an MXFP8 copy for prefill (`VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS=41`), checkpoint `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` @ `244cb6fe`, image `tp1-v3d-hf-20261005-21e0b201-5dad364d-warm` (experimental) |
| `qwen3.8-flash-next-1x-dgx-spark-previous` | the v3c recipe that was `qwen3.8-flash-next-1x-dgx-spark` on 2026-10-05 |
| (was `-previous`, v3b) | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v3b-20261005.yaml` (not listed; run by path) |

## 2026-10-05: single-Spark v3c promoted

| name | content now |
|---|---|
| `qwen3.8-flash-next-1x-dgx-spark` | v3c: v3b + shared GDN prefill staging + compact GDN records, KV pool 14 GiB, image `tp1-v3c-20261005-21e0b201-50330171-warm` (experimental) |
| `qwen3.8-flash-next-1x-dgx-spark-previous` | the v3b recipe that was `qwen3.8-flash-next-1x-dgx-spark` from 2026-10-04 to 2026-10-05 |
| (was `-previous`, v3a) | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v3a-20261004.yaml` (not listed; run by path) |

## 2026-10-04: single-Spark v3b promoted

| name | content now |
|---|---|
| `qwen3.8-flash-next-1x-dgx-spark` | v3b: v3a + `VLLM_PLE_MMAP_PREFILL_WILLNEED=1`, image `tp1-v3b-20261004-5bf24021-0632e506-warm` (experimental) |
| `qwen3.8-flash-next-1x-dgx-spark-previous` | the v3a recipe that was `qwen3.8-flash-next-1x-dgx-spark` on 2026-10-04 |
| (was `-previous`, v2) | `archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v2-20261002.yaml` (not listed; run by path) |

## 2026-10-04: single-Spark v3a promoted

| name | content now |
|---|---|
| `qwen3.8-flash-next-1x-dgx-spark` | v3a: v2 + `VLLM_PLE_MMAP_KEEPALIVE_MS=50`, image `tp1-v3a-20261004-5bf24021-7fa812b3-warm` (experimental) |
| `qwen3.8-flash-next-1x-dgx-spark-previous` (new) | the v2 recipe that was `qwen3.8-flash-next-1x-dgx-spark` from 2026-10-02 to 2026-10-04 |

