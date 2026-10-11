#!/usr/bin/env bash
# k91-moe-prefill-probe (2026-10-11, 1x #27): micro-benchmark of one routed-MoE layer at prefill M 1264/3024/6048/8192,
# b12x lowerings (serving anchor M32, M128, M64, layer-max FC1 scale with and without the NVFP4 split-materialized
# phase kernels) vs the canonical-layout NVFP4 W4A4 MoE of FlashInfer PR #5863 (vendored kernel file, called directly).
# Checkpoint NVFP4 expert weights (layer 5 timed, layers 5 and 47 for numerics), synthetic uniform and skewed routing.
# Design note: docs/notes/moe-prefill-dispatch.md in the 1x repo. Measure only; nothing ships from this job.
#   P0  stop the 2x (it spans both Sparks), run moe_probe.py all in the shipped 1x image on dgx-01 (one Spark,
#       container k91-probe with --init, timeout 28 min, per variant 7.5 min b12x / 5 min fi5863, budget 26 min),
#       remove the container, restore the 2x (EXIT trap, lib.sh).
# ETA ~38 min: stop ~1 + probe <= 28 + report 1 + restore ~8.
# Output: $RES/p0/*.tsv|*.log, k91.txt; comment.md + RESULT here.
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap restores the shipped 2x.
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k91; K=$HOME/GEN-AI/backlog/k91
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k91-moe-prefill-probe-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
SHIP_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-d3-hf-20261010-21e0b201-5dad364d-warm   # shipped 1x v2.2.0
HFM=$HOME/.cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE
CKPT=$HFM/snapshots/03f4a0570496bbe741189f4a2a7d611679e85406   # 1x served revision
VENDOR_SHA=b91b043293176c7cb1e8833c1a86a0197f2d81416071fcbbb34b6b019812127f   # PR #5863 head a0997210, verbatim
PRIV=$(cat "$K/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

drun() { # $1 = gpu|cpu, rest = probe args; the checkpoint dir is mounted at its host path so blob symlinks resolve
  local g=(); [ "$1" = gpu ] && g=(--gpus all); shift
  docker run --rm --init --name k91-probe "${g[@]}" --ipc=host --network none --entrypoint python3 \
    -e CUTE_DSL_ARCH=sm_121a -e XDG_CACHE_HOME=/tmp/cache -e B12X_COMPILE_WORKERS=2 -e PYTHONUNBUFFERED=1 \
    -v "$K:/k91:ro" -v "$HFM:$HFM:ro" ${OUT:+-v "$OUT:/out"} "$SHIP_IMG" /k91/moe_probe.py "$@"; }
preflight() {
  local ok=0 f
  for f in "$K/moe_probe.py" "$K/vendor/fi5863_moe_nvfp4_w4a4.py" "$CKPT/model.safetensors.index.json"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ "$(sha256sum "$K/vendor/fi5863_moe_nvfp4_w4a4.py" | cut -d' ' -f1)" = "$VENDOR_SHA" ] || { echo "vendored #5863 file changed"; ok=1; }
  python3 -m py_compile "$K/moe_probe.py" || { echo "moe_probe.py does not compile"; ok=1; }
  python3 "$K/moe_probe.py" selftest > /dev/null || { echo "moe_probe.py selftest failed"; ok=1; }
  docker rm -f k91-probe > /dev/null 2>&1   # a stale probe container from an earlier killed run
  docker image inspect "$SHIP_IMG" > /dev/null 2>&1 || { echo "$SHIP_IMG missing on dgx-01"; ok=1; }
  timeout 300 env OUT= bash -c "$(declare -f drun); K=$K HFM=$HFM SHIP_IMG=$SHIP_IMG; drun cpu check --ckpt $CKPT" \
    > "$K/check.txt" 2>&1 && grep -qx 'check ok' "$K/check.txt" || { echo "in-image CPU check failed ($K/check.txt)"; ok=1; }
  for f in "$K"/*.sh "$K"/*.py; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  preflight; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: stop 2x; dgx-01 moe_probe (6 variants, layer 5 timed at M 1264/3024/6048/8192, numerics L5+L47); restore 2x"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job" "$RES/p0"; cp "$K"/run.sh "$K"/moe_probe.py "$RES/job/"
st "running: P0 stop 2x, MoE probe on dgx-01"
stop_all
OUT=$RES/p0
child timeout -k 30 1680 bash -c "$(declare -f drun); K=$K HFM=$HFM SHIP_IMG=$SHIP_IMG OUT=$OUT; drun gpu all /out --ckpt $CKPT" \
  > "$RES/p0/probe.log" 2>&1; rc=$?
docker rm -f k91-probe > /dev/null 2>&1   # a hard timeout kills the docker CLI, not the container
log "P0 probe exit=$rc"
st "running: report"
python3 "$K/moe_probe.py" report "$RES/p0" > "$RES/k91.txt" 2>&1
V=$(sed -n 's/^VERDICT //p' "$RES/k91.txt")
{ echo "k91-moe-prefill-probe result: one routed-MoE layer (checkpoint NVFP4 weights, layer 5 timed, layers 5 and 47 for numerics) at prefill M 1,264 / 3,024 / 6,048 / 8,192 in the shipped 1x image on one Spark. b12x serving anchor (capacity 8192, M32 tile) against M128, M64, layer-max FC1 scale with and without the NVFP4 split-materialized kernels, and the vendored FlashInfer #5863 kernel. Synthetic routing (uniform and skewed); no router captures exist. Measure only; design note docs/notes/moe-prefill-dispatch.md."
  echo; echo "Verdict: $V (probe exit $rc)"
  echo; echo '```'; head -c 50000 "$RES/k91.txt"; echo '```'
  echo; echo "Raw TSV and logs stay on dgx-01 in \`$RES\`. Results: {RESULTS_URL}"; } > "$K/comment.md"
grep -qiE "$PRIV" "$K/comment.md" && { mkdir -p "$K/private"; mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k91 comment.md matched the private pattern (not posted; $RES)"; exit 1; }
cp "$K/comment.md" "$RES/comment.md"
[ $rc = 0 ] && grep -q '^VERDICT ' "$RES/k91.txt" && FINAL="DONE: k91 $V ($RES/k91.txt)" || FINAL="FAILED: k91 probe exit $rc ($RES/p0/probe.log)"
exit 0
