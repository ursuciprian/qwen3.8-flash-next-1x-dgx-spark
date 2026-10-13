"""k97 mod (2026-10-13): env-gated logging and capture hooks for the 1x serving container, without editing the image.

Loaded through PYTHONPATH=/cache/runtime/k97mod (the job puts this directory into the boot's own runtime cache). When
VLLM_K97_SPLIT_LOG=1 or VLLM_K97_CAPTURE_DIR is set, an import hook patches the source text of three vLLM modules as
they are imported (the files on disk stay untouched):
  vllm.v1.core.sched.scheduler                   K97SPLIT: every _mamba_block_aligned_split call (start, asked tokens,
                                                 block-aligned end, the stops inside the chunk and which one won)
                                                 K97STEP: the prefill tokens each step actually schedules per request
  vllm.model_executor.kernels.linear.b12x_blockscaled   capture hook in B12xBlockscaledLinear.run (HC GEMM inputs)
  vllm.model_executor.layers.fused_moe.b12x      capture hook in B12xExperts.apply (routed-MoE input, topk ids/weights)
Every anchor must occur exactly once (vLLM 5dad364d); otherwise the import raises and the boot fails loudly.
Nothing happens when neither variable is set.
"""
import importlib.abc
import importlib.util
import os
import sys

_SPLIT_NAMES = '("next_block", "retention", "last_cache", "tail_hash", "shared_prefix")'

PATCHES = {
    "vllm.v1.core.sched.scheduler": (
        ("logger = init_logger(__name__)\n",
         'logger = init_logger(__name__)\n_K97_SPLIT = __import__("os").environ.get("VLLM_K97_SPLIT_LOG") == "1"\n'),
        ("        end = min((s for s in stops if start < s < end), default=end)\n",
         "        if _K97_SPLIT:\n"
         f"            _k97n = {_SPLIT_NAMES}\n"
         '            _k97h = sorted((s, _k97n[i] if i < 5 else "reuse") for i, s in enumerate(stops) if start < s < end)\n'
         '            logger.info("K97SPLIT req=%s prompt=%d start=%d asked=%d aligned_end=%d block=%d internal_ckpt=%s '
         'eagle=%s stops=%s final=%d", request.request_id, request.num_prompt_tokens, start, num_new_tokens, end, '
         "block_size, use_internal_checkpoint, self.use_eagle, _k97h, (_k97h[0][0] if _k97h else end) - start)\n"
         "        end = min((s for s in stops if start < s < end), default=end)\n"),
        ("        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n",
         "        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n"
         "        if _K97_SPLIT:\n"
         "            for _k97r, _k97c in num_scheduled_tokens.items():\n"
         "                _k97q = self.requests.get(_k97r)\n"
         "                if _k97q is not None and _k97q.num_computed_tokens < _k97q.num_prompt_tokens:\n"
         '                    logger.info("K97STEP step=%d req=%s prompt=%d start=%d n=%d budget=%d", self.current_step, '
         "_k97r, _k97q.num_prompt_tokens, _k97q.num_computed_tokens, _k97c, self.max_num_scheduled_tokens)\n"),
    ),
    "vllm.model_executor.kernels.linear.b12x_blockscaled": (
        ('        """Execute the prepared regime for ``source``; op-body only."""\n',
         '        """Execute the prepared regime for ``source``; op-body only."""\n'
         "        __import__(\"k97cap\").hc(self.layer_name, source)\n"),
    ),
    "vllm.model_executor.layers.fused_moe.b12x": (
        ("        super().__init__(moe_config, quant_config)\n",
         "        super().__init__(moe_config, quant_config)\n"
         "        __import__(\"k97cap\").moe_register(self)\n"),
        ("        _require_b12x_fused_moe().run(binding=binding)\n",
         "        __import__(\"k97cap\").moe(self, hidden_states, topk_ids, topk_weights)\n"
         "        _require_b12x_fused_moe().run(binding=binding)\n"),
    ),
}


def apply(src, patches, name="?"):
    for old, new in patches:
        n = src.count(old)
        if n != 1:
            raise ImportError(f"k97 mod: anchor found {n} times in {name}: {old.strip()[:80]!r}")
        src = src.replace(old, new)
    return src


class _Loader(importlib.abc.Loader):
    def __init__(self, origin, name):
        self.origin, self.name = origin, name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(self.origin) as f:
            src = apply(f.read(), PATCHES[self.name], self.name)
        exec(compile(src, self.origin, "exec"), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            raise ImportError(f"k97 mod: no source file for {name}")
        spec.loader = _Loader(spec.origin, name)
        return spec


if os.environ.get("VLLM_K97_SPLIT_LOG") == "1" or os.environ.get("VLLM_K97_CAPTURE_DIR"):
    sys.meta_path.insert(0, _Finder())
    print(f"K97MOD active in pid {os.getpid()}", file=sys.stderr, flush=True)
