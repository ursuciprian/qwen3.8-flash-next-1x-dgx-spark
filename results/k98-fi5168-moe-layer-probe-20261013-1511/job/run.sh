#!/usr/bin/env bash
# k98-fi5168-moe-layer-probe (2026-10-13, 1x #27 for I640, tp-2 for I320): one routed-MoE layer, the shipped b12x MoE
# against FlashInfer at 5c4058bd9b (PR #5168 merge: SM12x fused MoE, true-extent I320 weight views, routing and
# transfer changes), for the single-Spark layer (I640) and the rank-0 shard of the 2x (I320), M 1264 / 3024 / 6048 / 9072.
# Reuses the k91 harness (checkpoint NVFP4 weights of layer 5 timed, layers 5 and 47 for numerics, FP32 reference,
# k81 anchor). Routing: the k97 capture (real top-10 ids, weights and MoE inputs of a cold 16K prompt) when present,
# plus k91's synthetic uniform and skew patterns. Measure only; nothing ships, nothing goes upstream.
#   P0  b12x container (1x v2.3.0 image as is, --network none, 25 min): b12x-m32 (k91 serving anchor) and b12x-auto;
#       saves the M 6048 outputs (rows 0..255) to ~/GEN-AI/backlog/k98/num for the FlashInfer comparison.
#   P1  FlashInfer container (same image, throwaway, 35 min): staged nvidia-cutlass-dsl 4.7.1 wheels installed, the
#       5c4058bd source (pre-cloned in k98/fi-src, HEAD checked) on PYTHONPATH; numerics first (vs FP32 and vs the b12x
#       output), then timing, first-call preparation time and kept GPU memory.
#   Report: numerics gate, anchor (tp1 vs k81; tp2 N/A), speed per M and routing, preparation and memory x48 layers,
#   operator-only projection per 16K prompt.
# ETA ~60 min: stop ~1 + b12x <= 25 + FlashInfer <= 35 (JIT compile of the CuTe DSL kernels included) + report 1.
# Output: $RES/p0 (b12x), $RES/p1 (FlashInfer), k98.txt; comment.md (1x #27) + comment-tp2.md (tp-2, by hand) + RESULT here.
# Contract (backlog/runner.sh): dgx01 lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE first
# line DONE/FAILED (lib.sh EXIT trap). Usage: bash run.sh --dry-run | via the runner (JOBS="k97 k96 k98"). Stop: kill -TERM <pid>.
NEEDS="dgx01"
set -u
J=k98; K=$HOME/GEN-AI/backlog/k98
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k98-fi5168-moe-layer-probe-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
IMG=ghcr.io/ursuciprian/spark-vllm-b12x:k95-20261013-b45fe867-5dad364d-warm   # 1x v2.3.0; its fused MoE = the 2x's (21e0b201)
HFM=$HOME/.cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE
CKPT=$HFM/snapshots/03f4a0570496bbe741189f4a2a7d611679e85406   # expert weights are the same in every GDN-MSE revision
FISHA=5c4058bd9b75967de947971d7f28d36bd600d888
CAPD=$BL/k97/capture; NUMD=$K/num
PRIV=$(cat "$K/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

drun() { # b12x|fi gpu|cpu, moe98 args
  local kind=$1 g=(); [ "$2" = gpu ] && g=(--gpus all); shift 2
  local c=(); [ -d "$CAPD" ] && c=(-v "$CAPD:/cap:ro")
  local common=(--rm --init --name k98-$kind "${g[@]}" --ipc=host -e CUTE_DSL_ARCH=sm_121a -e XDG_CACHE_HOME=/tmp/cache
    -e B12X_COMPILE_WORKERS=2 -e PYTHONUNBUFFERED=1 -v "$K:/k98:ro" -v "$HFM:$HFM:ro" -v "$NUMD:/num" "${c[@]}" ${OUT:+-v "$OUT:/out"})
  if [ $kind = b12x ]; then docker run "${common[@]}" --network none --entrypoint python3 "$IMG" /k98/moe98.py "$@"
  else docker run "${common[@]}" --network none -v "$K/fi-src:/src:ro" -v "$K/wheels:/wheels:ro" --entrypoint bash "$IMG" /k98/fi_inner.sh "$@"; fi; }
fill() { # template out KEY=file...: {KEY} -> file contents (stdlib)
  python3 - "$@" <<'PY'
import sys
t = open(sys.argv[1]).read()
for kv in sys.argv[3:]:
    k, f = kv.split("=", 1)
    t = t.replace("{" + k + "}", open(f).read().rstrip("\n"))
open(sys.argv[2], "w").write(t)
PY
}
host_checks() { # no containers: safe while other jobs run
  local ok=0 f n
  for f in "$K/moe98.py" "$K/fi_inner.sh" "$K/comment.tmpl" "$K/comment-tp2.tmpl" "$CKPT/model.safetensors.index.json" "$K/fi-src/flashinfer/fused_moe/cute_dsl/b12x_moe.py"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ "$(git -C "$K/fi-src" rev-parse HEAD 2>/dev/null)" = $FISHA ] || { echo "fi-src HEAD is not $FISHA"; ok=1; }
  [ -z "$(git -C "$K/fi-src" status --porcelain --untracked-files=no 2>/dev/null)" ] || { echo "fi-src has local changes"; ok=1; }
  n=$(ls "$K"/wheels/nvidia_cutlass_dsl*-4.7.1-*.whl 2>/dev/null | wc -l); [ "$n" = 4 ] || { echo "expected 4 staged cutlass-dsl 4.7.1 wheels, found $n"; ok=1; }
  python3 -m py_compile "$K/moe98.py" || { echo "moe98.py does not compile"; ok=1; }
  bash -n "$K/fi_inner.sh" || { echo "fi_inner.sh syntax"; ok=1; }
  python3 "$K/moe98.py" selftest > /dev/null || { echo "moe98.py selftest failed"; ok=1; }
  docker image inspect "$IMG" > /dev/null 2>&1 || { echo "$IMG missing on dgx-01"; ok=1; }
  for f in "$K"/run.sh "$K"/*.py "$K"/fi_inner.sh "$K"/comment*.tmpl; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }
preflight() { # under the lock: host checks plus the in-image CPU checks of both containers
  local ok=0 k; host_checks || ok=1; mkdir -p "$NUMD"
  for k in b12x fi; do
    docker rm -f k98-$k > /dev/null 2>&1   # a stale container from an earlier killed run
    timeout 600 env OUT= bash -c "$(declare -f drun); K=$K HFM=$HFM IMG=$IMG CAPD=$CAPD NUMD=$NUMD; drun $k cpu check --ckpt $CKPT $([ $k = fi ] && echo --fi)" \
      > "$K/check-$k.txt" 2>&1 && grep -qx 'check ok' "$K/check-$k.txt" || { echo "in-image CPU check ($k) failed ($K/check-$k.txt)"; ok=1; }
  done
  grep -q 'nvidia-cutlass-dsl 4.7.1' "$K/check-fi.txt" && grep -q 'flashinfer from /src/' "$K/check-fi.txt" \
    || { echo "FlashInfer container did not get cutlass-dsl 4.7.1 + the 5c4058bd source ($K/check-fi.txt)"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  host_checks; rc=$?
  [ -d "$CAPD" ] && echo "routing: k97 capture + synthetic ($(ls "$CAPD" | grep -c '^moe') MoE chunk files)" || echo "routing: no k97 capture yet, synthetic only (run k97 first)"
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: stop dgx-01 serving; b12x container (b12x-m32, b12x-auto); FlashInfer $FISHA container (fi5168); tp1 I640 + tp2 I320; report"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
eval "$(trap -p EXIT | sed "s/^trap -- '/trap -- 'docker rm -f k98-b12x k98-fi > \/dev\/null 2>\&1; /")"   # probe containers go on any exit
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md" "$K/comment-tp2.md"
st "running: preflight"
rm -rf "$NUMD"; preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job" "$RES/p0" "$RES/p1"; cp "$K"/run.sh "$K"/moe98.py "$K"/fi_inner.sh "$K"/comment*.tmpl "$K"/check-*.txt "$RES/job/"
st "running: P0 stop dgx-01, b12x MoE layer probe"
stop_all
OUT=$RES/p0
child timeout -k 30 1560 bash -c "$(declare -f drun); K=$K HFM=$HFM IMG=$IMG CAPD=$CAPD NUMD=$NUMD OUT=$OUT; drun b12x gpu b12x /out --ckpt $CKPT --num /num $([ -d "$CAPD" ] && echo --cap-dir /cap)" \
  > "$RES/p0/probe.log" 2>&1; rc0=$?
docker rm -f k98-b12x > /dev/null 2>&1; log "P0 b12x exit=$rc0"
st "running: P1 FlashInfer 5c4058bd MoE layer probe"
OUT=$RES/p1
child timeout -k 30 2160 bash -c "$(declare -f drun); K=$K HFM=$HFM IMG=$IMG CAPD=$CAPD NUMD=$NUMD OUT=$OUT; drun fi gpu fi /out --ckpt $CKPT --num /num $([ -d "$CAPD" ] && echo --cap-dir /cap)" \
  > "$RES/p1/probe.log" 2>&1; rc1=$?
docker rm -f k98-fi > /dev/null 2>&1; log "P1 FlashInfer exit=$rc1"
st "running: report"
mkdir -p "$RES/all"; cp "$RES"/p0/*.tsv "$RES"/p1/*.tsv "$RES/all/" 2>/dev/null
python3 "$K/moe98.py" report "$RES/all" > "$RES/k98.txt" 2>&1
V=$(sed -n 's/^VERDICT //p' "$RES/k98.txt")
echo "$V (b12x exit $rc0, FlashInfer exit $rc1)" > "$RES/verdict.txt"
for c in comment comment-tp2; do   # comment.md -> 1x #27 (poster); comment-tp2.md -> the tp-2 issue (by hand)
  fill "$K/$c.tmpl" "$K/$c.md" VERDICT="$RES/verdict.txt" REPORT="$RES/k98.txt"
  sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/$c.md"
  grep -qiE "$PRIV" "$K/$c.md" && { mkdir -p "$K/private"; mv "$K/$c.md" "$K/private/"; FINAL="FAILED: k98 $c.md matched the private pattern (not posted; $RES)"; exit 1; }
  cp "$K/$c.md" "$RES/$c.md"; done
[ $rc0 = 0 ] && [ $rc1 = 0 ] && grep -q '^VERDICT ' "$RES/k98.txt" && FINAL="DONE: k98 $V ($RES/k98.txt)" \
  || FINAL="FAILED: k98 b12x exit $rc0, FlashInfer exit $rc1 ($RES/k98.txt)"
exit 0
