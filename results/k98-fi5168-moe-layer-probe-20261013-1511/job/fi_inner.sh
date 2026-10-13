#!/usr/bin/env bash
# k98 FlashInfer container entry (throwaway container of the 1x v2.3.0 image): install the staged nvidia-cutlass-dsl
# 4.7.1 wheels (the image has 4.7.0; FlashInfer's b12x extra pins 4.7.1), put the FlashInfer source at 5c4058bd on
# PYTHONPATH (CuTe DSL kernels compile at first call), then run moe98.py with the given arguments.
set -uo pipefail
pip install -q --no-index --no-deps /wheels/*.whl 2>&1 | grep -v "Running pip as the 'root' user" | tail -3
export PYTHONPATH=/src PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
import importlib.metadata as md, flashinfer, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
for p in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-core", "nvidia-cutlass-dsl-libs-cu13", "apache-tvm-ffi"):
    try: print(p, md.version(p))
    except Exception as e: print(p, "MISSING", e)
print("flashinfer from", flashinfer.__file__)
PY
exec python3 /k98/moe98.py "$@"
