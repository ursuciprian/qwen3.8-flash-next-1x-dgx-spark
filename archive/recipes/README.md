# Archived recipes

Superseded single-Spark builds. sparkrun does not list them; run one by path, for example
`sparkrun run archive/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v3c-20261005.yaml --hosts <spark> --solo`.
Each file's header says what changed and links its results.

| recipe | build | image | results |
|---|---|---|---|
| [`qwen3.8-flash-next-1x-dgx-spark-v2.0.0-20261008.yaml`](qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v2.0.0-20261008.yaml) | v2.0.0 (old name v3d), 2026-10-05 | `tp1-v3d-hf-20261005-21e0b201-5dad364d-warm` | [`results/tp1-v3d-20261005/`](../../results/tp1-v3d-20261005/) |
| [`qwen3.8-flash-next-1x-dgx-spark-v3c-20261005.yaml`](qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v3c-20261005.yaml) | v3c, 2026-10-05 | `tp1-v3c-20261005-21e0b201-50330171-warm` | [`results/tp1-v3c-20261005/`](../../results/tp1-v3c-20261005/) |
| [`qwen3.8-flash-next-1x-dgx-spark-v3b-20261005.yaml`](qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v3b-20261005.yaml) | v3b, 2026-10-04 | `tp1-v3b-20261004-5bf24021-0632e506-warm` | [`results/tp1-v3b-20261004/`](../../results/tp1-v3b-20261004/) |
| [`qwen3.8-flash-next-1x-dgx-spark-v3a-20261004.yaml`](qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v3a-20261004.yaml) | v3a, 2026-10-04 | `tp1-v3a-20261004-5bf24021-7fa812b3-warm` | [`results/tp1-v3a-20261004/`](../../results/tp1-v3a-20261004/) |
| [`qwen3.8-flash-next-1x-dgx-spark-v2-20261002.yaml`](qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-v2-20261002.yaml) | v2, 2026-10-02 | `tp1-20261002-5bf24021-884b4ff6-warm` | [`results/tp1-v2-20261002/`](../../results/tp1-v2-20261002/) |

The v3d → v3e drafter arms that were screened but not promoted (k54, k58, k62, k67) ran from recipes inside their
results directories; see [results/README.md](../../results/README.md).
