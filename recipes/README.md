# Recipes

What `sparkrun recipe list` shows for this registry. Both recipes run Qwen3.8-Flash-Next on one DGX Spark (TP=1),
pin checkpoint `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` by revision, use a public ghcr image, and need no mods,
no host mounts and no `--trust`.

| recipe | checkpoint revision | image | use |
|---|---|---|---|
| [`qwen3.8-flash-next-1x-dgx-spark`](qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml) | `16c9bd54` | `tp1-v3e-hf-20261008-21e0b201-5dad364d-warm` | Current build, v3e (retrained MTP drafter). |
| [`qwen3.8-flash-next-1x-dgx-spark-previous`](qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-previous.yaml) | `244cb6fe` | `tp1-v3d-hf-20261005-21e0b201-5dad364d-warm` | Rollback: v3d, original drafter. |

```sh
sparkrun run qwen3.8-flash-next-1x-dgx-spark --hosts <spark> --solo
```

Earlier builds (v2 to v3c) are in [`archive/recipes/`](../archive/recipes/README.md); sparkrun does not scan it. Name
history: [RENAMES.md](RENAMES.md). `scripts/validate_recipes.py` fails any recipe here that lacks `--revision`, uses an
image without a registry host, or needs mods or volumes.
