# Recipes

What `sparkrun recipe list` shows for this registry. Both recipes run Qwen3.8-Flash-Next on one DGX Spark (TP=1),
pin checkpoint `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE` by revision, use a public ghcr image, and need no mods,
no host mounts and no `--trust`.

| recipe | checkpoint revision | image | use |
|---|---|---|---|
| [`qwen3.8-flash-next-1x-dgx-spark`](qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml) | `03f4a057` | `k106s-20261017-a1ad1b69-5dad364d-10d0acc8-warm` | Current build, v2.4.0 (HC down projection BF16 path, GDN prefill views). |
| [`qwen3.8-flash-next-1x-dgx-spark-previous`](qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark-previous.yaml) | `03f4a057` | `k95-20261013-b45fe867-5dad364d-warm` | Rollback: v2.3.0 (b12x large-row MXFP8 tile fix). |

```sh
sparkrun run qwen3.8-flash-next-1x-dgx-spark --hosts <spark> --solo
```

Earlier builds (v2 to v2.2.0) are in [`archive/recipes/`](../archive/recipes/README.md); sparkrun does not scan it. Name
history: [RENAMES.md](RENAMES.md). `scripts/validate_recipes.py` fails any recipe here that lacks `--revision`, uses an
image without a registry host, or needs mods or volumes.
